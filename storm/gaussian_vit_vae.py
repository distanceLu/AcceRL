import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from einops import rearrange, reduce

from storm.vit_decoder import ViTDecoder
from replay_buffer import ObsReplayBuffer
from storm.actor_critic_model import get_vla, get_processor

from experiments.robot.openvla_utils import prepare_images_for_vla

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        # loss = reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()
    

class DistHead(nn.Module):
    def __init__(self, image_feat_dim, stoch_dim):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim * stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # logits = torch.clamp(logits, min=-20, max=20)
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1 - mixing_ratio) * probs
        # mixed_probs = mixed_probs.clamp(min=1e-6)
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B L N (K C) -> B L N K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits
    

class ViTVAE(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.stoch_dim = 64
        self.stoch_flattened_dim = self.stoch_dim * self.stoch_dim
        # Encoder
        self.encoder = encoder
        # Freeze encoder parameters
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
            
        self.projector = projector

        # Distribution Head
        self.dist_head = DistHead(self.stoch_flattened_dim, self.stoch_dim)

        # Decoder
        self.image_decoder = ViTDecoder(embed_dim=self.stoch_flattened_dim, depth=12)

        self.mse_loss_func = MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)


    def straight_through_gradient(self, logits):
        dist = OneHotCategorical(logits=logits)
        sample = dist.sample() + dist.probs - dist.probs.detach()
        return sample.to(torch.bfloat16)

    def calculate_kl_loss(self, post_logits, free_bits=1):
        # post_logits.shape: (1, 8, 512, 64, 64)
        post_dist = OneHotCategorical(logits=post_logits)
        prior_dist = OneHotCategorical(probs=torch.ones_like(post_logits) / self.stoch_dim)
        kl_div = torch.distributions.kl.kl_divergence(post_dist, prior_dist)
        # kl_div = reduce(kl_div, "B L N D -> B L N", "sum")
        kl_div = kl_div.mean()
        # kl_div = torch.max(torch.ones_like(kl_div)*free_bits, kl_div)

        return kl_div


    def flatten_sample(self, sample):
        return rearrange(sample, "B L N K C -> B L N (K C)")

    def update(self, obs, current_steps=0, logger=None):
        B, L, C, H, W = obs.shape # (B, L, 12, 224, 224)
        obs_reshape = obs.reshape(B*L, C, H, W)
        # Encode image
        patch_features = self.encoder(obs_reshape) # (B*L, 512, 2176)
        _, N, _ = patch_features.shape
        patch_features = patch_features.reshape(B, L, N, -1)
        projected_patch_embeddings = self.projector(patch_features) # (B, L, 256*2, 4096)

        # Get posterior logits
        post_logits = self.dist_head.forward_post(projected_patch_embeddings) # (B, L, 512, 64, 64)

        # Sample using straight-through gradient
        sample = self.straight_through_gradient(post_logits.float()).to(torch.bfloat16) # (B, L, 512, 64, 64)
        flattened_sample = self.flatten_sample(sample) # (B, L, 512, 4096)

        # Decode image
        obs_hat = self.image_decoder(flattened_sample) # (B, L, 12, 224, 224)

        reconstruction_loss = self.mse_loss_func(obs_hat, obs)
        kl_loss = self.calculate_kl_loss(post_logits.float())
        total_loss = reconstruction_loss + kl_loss

        # gradient descent
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            logger.add_scalar("reconstruction_loss", reconstruction_loss.item(), current_steps)
            logger.add_scalar("kl_loss", kl_loss.item(), current_steps)
            logger.add_scalar("total_loss", total_loss.item(), current_steps)        

        return reconstruction_loss.item(), kl_loss.item(), total_loss.item()
    
def obs_process(processor, obs, task_label):
    all_images = [obs["full_image"], obs["wrist_image"]]
    all_images = prepare_images_for_vla(all_images, cfg)
    primary_image = all_images.pop(0)

    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    inputs = processor(prompt, primary_image)

    if all_images:
        all_wrist_inputs = [
            processor(prompt, image_wrist) for image_wrist in all_images
        ]
        # Concatenate all images
        primary_pixel_values = inputs["pixel_values"]
        all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
        inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

    return inputs["pixel_values"]

if __name__ ==  "__main__":

    from experiments.robot.libero.run_libero_eval import GenerateConfig
    from rl.libero_env import LiberoEnvWrapper
    import random
    import numpy as np
    from tqdm import tqdm
    import time
    import copy

    from tensorboardX import SummaryWriter

    DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

    TRAIN_ITERS = 100000
    BATCH_SIZE = 1
    BATCH_LENGTH = 8
    SAVE_MODEL = False
    SAVE_FREQ = 100

    BENCHMARK = "libero_spatial"
    TENSORBOARD_LOG = True
    LOG_DIR = f"./vit_vae_log/{BENCHMARK}/{DATE_TIME}"

    if TENSORBOARD_LOG:
        logger = SummaryWriter(logdir=LOG_DIR)

    print(DEVICE)
    # Instantiate config
    cfg = GenerateConfig(
        pretrained_checkpoint="/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/",
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=8,
        unnorm_key="libero_spatial_no_noops",
    )

    vla = get_vla(cfg, device=DEVICE)

    processor = get_processor(cfg)

    vision_backbone = copy.deepcopy(vla.vision_backbone).cpu()
    vision_backbone = vision_backbone.to(DEVICE, dtype=torch.bfloat16)
    projector = copy.deepcopy(vla.projector).cpu()
    projector = projector.to(DEVICE, dtype=torch.bfloat16)
    del vla

    vit_vae = ViTVAE(encoder=vision_backbone, projector=projector)
    vit_vae.to(DEVICE, dtype=torch.bfloat16)

    replay_buffer = ObsReplayBuffer(obs_shape=(12, 224, 224), num_envs=1, warmup_length=BATCH_SIZE*BATCH_LENGTH, 
                                    max_length=TRAIN_ITERS, store_on_gpu=False, device=DEVICE)

    task_id = random.randint(0, 9)
    env = LiberoEnvWrapper(
        benchmark_name=BENCHMARK,
        task_id=task_id,  # 随机任务 ID
        image_size=224,
        render_mode="rgb_array",
    )

    current_obs, info = env.reset(seed=0)

    for total_steps in tqdm(range(TRAIN_ITERS)):

        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        current_image = obs_process(processor, current_obs, env.task_description)
        replay_buffer.append(current_image)

        done_flag = np.logical_or(done, truncated)

        if done_flag:
            # task_id = random.randint(0, 9)
            # env = LiberoEnvWrapper(
            #     benchmark_name=BENCHMARK,
            #     task_id=task_id,  # 随机任务 ID
            #     image_size=224,
            #     render_mode="rgb_array",
            # )
            obs, info = env.reset(seed=total_steps)

        current_obs = obs

        if replay_buffer.ready():
            obs_sample = replay_buffer.sample(batch_size=BATCH_SIZE, external_batch_size=None, batch_length=BATCH_LENGTH) # (16, 64, 12, 224, 224)
            if TENSORBOARD_LOG:
                reconstruction_loss, kl_loss, total_loss = vit_vae.update(obs=obs_sample, current_steps=total_steps, logger=logger)
            else:
                reconstruction_loss, kl_loss, total_loss = vit_vae.update(obs=obs_sample)


            print(f"step: {total_steps}, total_loss:, {total_loss:.4f}, reconstruction_loss: {reconstruction_loss:.4f}, kl_loss: {kl_loss:.4f}")

            if SAVE_MODEL and (total_steps+1)%SAVE_FREQ==0:
                save_dir = f"{LOG_DIR}/checkpoint"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(vit_vae.state_dict(), f"{save_dir}/vit_vae.pth")

    
    # obs = torch.randn((2, 4, 12, 224, 224)).to(DEVICE, dtype=torch.bfloat16)


    # vit_vae.update(obs)