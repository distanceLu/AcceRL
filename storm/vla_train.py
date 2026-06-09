import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import gymnasium
import argparse
import numpy as np
from einops import rearrange
import torch
from collections import deque
from tqdm import tqdm
import colorama
import shutil
import os
from utils import seed_np_torch, Logger, load_config
from vla_replay_buffer import ReplayBuffer
from storm import env_wrapper
import agents
from storm.vla_world_model2 import WorldModel
import time
from rl.utils import prepare_one_obs
from experiments.robot.openvla_utils import get_processor
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from rl.libero_env import LiberoEnvChunk
from rl.actor_critic_model import ActorCritic

from experiments.robot.libero.libero_utils import GenerateConfig

# OpenVLA 加载配置
USE_BF16: bool = True
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
PRETRAINED_CHECKPOINT = "/cpfs01/lcx_workspace/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/"


def build_single_env(env_name, image_size, seed, skip):
    # Convert image_size to a tuple if it's a list or integer
    if isinstance(image_size, int):
        image_size = (image_size, image_size)  # Assumes square dimensions (e.g., 84 → (84, 84))
    elif isinstance(image_size, list):
        image_size = tuple(image_size)  # Convert list to tuple

    if "v3" in env_name:
        import metaworld
        env = gymnasium.make('Meta-World/MT1', env_name=env_name, seed=seed, render_mode="rgb_array", camera_name="corner", width=image_size[0], height=image_size[1]) # MT1 with the reach environment
        env = env_wrapper.MetaWorldWrapper(env, env_name=env_name, shape=image_size)
        env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=skip)
    elif "libero" in env_name:
        task_id = 5
        env = LiberoEnvChunk(
            benchmark_name=env_name,
            task_id=task_id,
            image_size=224,
            render_mode="rgb_array",
            chunk_num=NUM_ACTIONS_CHUNK,
        )
    else:
        raise NotImplementedError(env_name)
    return env


def build_vec_env(env_names, image_size, num_envs, seed, skip):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size, seed, skip)
    env_fns = []
    env_fns = [lambda_generator(env_names[i], image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


class FakeVecEnv:
    def __init__(self, env_name, processor, cfg, dtype):
        task_id = 5
        self.env = LiberoEnvChunk(
            benchmark_name=env_name[0],
            task_id=task_id,
            image_size=224,
            render_mode="rgb_array",
            chunk_num=NUM_ACTIONS_CHUNK,
        )
        self.processor = processor
        self.cfg = cfg
        self.dtype = dtype

    def reset(self):
        obs, info = self.env.reset()
        return [self.process_obs(obs)], [info]
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action[0])
        if terminated or truncated:
            obs, _ = self.env.reset()
        return [self.process_obs(obs)], np.array([reward]), np.array([terminated]), np.array([truncated]), [info]
    
    def sample_action(self):
        return np.array([self.env.action_space.sample()])
    
    def process_obs(self, raw_obs):
        return prepare_one_obs(self.cfg, self.processor, raw_obs, self.env.task_description, self.dtype)


def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, demonstration_batch_size, batch_length, logger, train_steps):
    obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length) # (batch, batch_length, dim)
    world_model.update(obs, action, reward, termination, train_steps, logger=logger)


@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, agent: agents.ActorCriticAgent,
                             imagine_batch_size, imagine_demonstration_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, train_steps, logger):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()

    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        train_steps=train_steps,
        logger=logger,
    )
    return latent, action, None, None, reward_hat, termination_hat


@torch.no_grad()
def world_model_imitation_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, teacher,
                             batch_size, demonstration_batch_size, batch_length,
                             ):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        batch_size, demonstration_batch_size, batch_length)
    latent, action, reward, termination, teacher_action = world_model.imitation_data(
        sample_obs, sample_action, sample_reward, sample_termination, teacher,
    )
    return latent, teacher_action, reward, termination


def joint_train_world_model_agent(env_name, max_steps, num_envs, image_size,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel, agent: agents.ActorCriticAgent,
                                  train_dynamics_every_steps, train_agent_every_steps,
                                  batch_size, demonstration_batch_size, batch_length,
                                  imagine_batch_size, imagine_demonstration_batch_size,
                                  imagine_context_length, imagine_batch_length,
                                  save_every_steps, seed, logger, skip, multi_task, cfg, teacher_model):
    # create ckpt dir
    if args.save_model:
        os.makedirs(f"ckpt/{args.n}", exist_ok=True)

    # build vec env, not useful in the Atari100k setting
    # but when the max_steps is large, you can use parallel envs to speed up
    # vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, seed=seed, skip=skip)
    processor = get_processor(cfg)
    vec_env = FakeVecEnv(env_name, processor, cfg, TORCH_DTYPE)

    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)

    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    # context_instruction = deque(maxlen=16)
    # instruction = replay_buffer.sample_context_instruction()

    task_name = "_".join(env_name) if multi_task else env_name[0]
    # sample and train
    for total_steps in tqdm(range(max_steps)):
    # for total_steps in tqdm(range(max_steps//num_envs//skip)):
        start_time = time.time()
        # sample part >>>
        if replay_buffer.ready():
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.sample_action()
                else:
                    flat_obs = []
                    for obs in context_obs:
                        flat_obs.extend(obs)
                    inputs_batch = world_model.prepare_inputs_batch(flat_obs)
                    context_latent = world_model.encode_obs(inputs_batch['pixel_values'], inputs_batch['proprio']) # (num_envs, len(context_obs), 1024)
                    model_context_action = np.stack(list(context_action), axis=1) # (num_envs, len(context_action), 4)
                    model_context_action = torch.Tensor(model_context_action).cuda()
                    # model_context_instruction = torch.cat(list(context_instruction), dim=1)
                    last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action, inputs_batch)
                    # print('prior_flattened_sample', prior_flattened_sample.shape) # (num_envs, 1, 1024)
                    action = agent.sample_as_env_action(
                        last_dist_feat,
                        greedy=False
                    )

            # context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255) # (1, 64, 64, 3) -> (1, 1, 3, 64, 64)
            context_obs.append(current_obs)
            context_action.append(action) # (num_envs, 4)
            # context_instruction.append(instruction) # (num_envs, 1, dim)
        else:
            action = vec_env.sample_action()

        obs, reward, done, truncated, info = vec_env.step(action) # (num_envs, dim)
        replay_buffer.append(current_obs, action, reward, np.logical_or(done, truncated))
        
        print(f"sample time: {(time.time() - start_time)*1000:.2f}ms")

        env_steps = total_steps*num_envs*skip
        train_steps = total_steps
        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            avg_reward = 0
            num_dones = 0
            for i in range(num_envs):
                if done_flag[i]:
                    if multi_task:
                        logger.log(f"sample/{env_name[i]}_reward", sum_reward[i], train_steps)
                    # logger.log(f"sample/{env_name}_episode_steps", current_info["episode_frame_number"][i]//skip, train_steps)  # framskip=4
                    # logger.log("replay_buffer/length", llen(replay_buffer), train_steps)
                    avg_reward += sum_reward[i]
                    num_dones += 1
                    sum_reward[i] = 0

            # obs, info = vec_env.reset(indices=indices)
            avg_reward = avg_reward / num_dones
            logger.log(f"sample/{task_name}_reward", avg_reward, train_steps)
            logger.log("replay_buffer/length", len(replay_buffer), train_steps)
            logger.log("sample/env_steps", env_steps, train_steps)

        # update current_obs, current_info and sum_rewarsd
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part

        # train world model part >>>
        start_time = time.time()
        if replay_buffer.ready() and total_steps % (train_dynamics_every_steps//num_envs) == 0:
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=batch_size,
                demonstration_batch_size=demonstration_batch_size,
                batch_length=batch_length,
                train_steps=train_steps,
                logger=logger
            )
        print(f"train world model time: {(time.time() - start_time)*1000:.2f}ms")
        # <<< train world model part

        # train agent part >>>
        start_time = time.time()
        if replay_buffer.ready() and total_steps % (train_agent_every_steps//num_envs) == 0 and total_steps*num_envs >= 0:
            if total_steps % (save_every_steps//num_envs) == 0:
                log_video = True
            else:
                log_video = False
            latent, teacher_action, imitate_reward, imitate_termination = world_model_imitation_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                teacher=teacher_model,
                batch_size=batch_size,
                demonstration_batch_size=demonstration_batch_size,
                batch_length=batch_length,
            )
            agent.imitate(latent, teacher_action, imitate_reward, imitate_termination, train_steps, logger)

            # imagine_latent, agent_action, agent_logprob, agent_value, imagine_reward, imagine_termination = world_model_imagine_data(
            #     replay_buffer=replay_buffer,
            #     world_model=world_model,
            #     agent=agent,
            #     imagine_batch_size=imagine_batch_size,
            #     imagine_demonstration_batch_size=imagine_demonstration_batch_size,
            #     imagine_context_length=imagine_context_length,
            #     imagine_batch_length=imagine_batch_length,
            #     log_video=log_video,
            #     train_steps=train_steps,
            #     logger=logger
            # )

            # agent.update(
            #     latent=imagine_latent,
            #     action=agent_action,
            #     old_logprob=agent_logprob,
            #     old_value=agent_value,
            #     reward=imagine_reward,
            #     termination=imagine_termination,
            #     train_steps=train_steps,
            #     logger=logger
            # )
        print(f"train agent time: {(time.time() - start_time)*1000:.2f}ms")
        # <<< train agent part

        # save model per episode
        if total_steps % (save_every_steps) == 0 and args.save_model:
            print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), f"ckpt/{args.n}/world_model_{total_steps}.pth")
            torch.save(agent.state_dict(), f"ckpt/{args.n}/agent_{total_steps}.pth")


def build_world_model(conf, args, action_dim, cfg, dtype):
    model = WorldModel(
        in_channels=conf.Models.WorldModel.InChannels,
        action_dim=action_dim,
        instruction_dim=384 if args.use_instruction else 0,
        transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
        transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
        transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
        transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads,
        dist=args.dist,
        cfg=cfg,
        dtype=dtype,
    )
    model.to(model.device)
    return model


def build_agent(conf, args, action_dim):
    return agents.ActorCriticAgent(
        feat_dim=conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
        dist=args.dist,
        is_pool=True,
    ).cuda()


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, default="openvla_oft_libero5_wm_extract_valid_act")
    parser.add_argument("-seed", type=int, default=1)
    parser.add_argument("-config_path", type=str, default="storm/config_files/vla_STORM.yaml")
    parser.add_argument("-env_name", type=str, default="libero_spatial")
    parser.add_argument("-trajectory_path", type=str, default="D_TRAJ/reach-v3.pkl")
    parser.add_argument("-dist", type=str, default="normal")
    parser.add_argument("-skip", type=int, default=4)
    parser.add_argument("-use_instruction", type=int, default=0)
    parser.add_argument("-save_model", type=int, default=0)
    
    cfg = GenerateConfig(
            pretrained_checkpoint=PRETRAINED_CHECKPOINT,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_images_in_input=2,
            use_proprio=True,
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=True,
            num_open_loop_steps=NUM_ACTIONS_CHUNK,
            unnorm_key="libero_spatial_no_noops",
            device=torch.device("cuda")
        )
    args = parser.parse_args()
    conf = load_config(args.config_path)
    args.use_instruction = bool(args.use_instruction)
    args.save_model = bool(args.save_model)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # set seed
    seed_np_torch(seed=args.seed)
    # tensorboard writer
    path = f"storm_runs/{args.n}_{int(time.time())}"
    logger = Logger(path=path)
    # copy config file
    shutil.copy(args.config_path, f"{path}/config.yaml")

    envs_list = args.env_name.split()
    if len(envs_list) > 1:
        args.env_name = envs_list
        conf.JointTrainAgent.NumEnvs = len(envs_list)
        args.multi_task = True
    else:
        args.env_name = envs_list * conf.JointTrainAgent.NumEnvs
        args.multi_task = False

    conf.JointTrainAgent.TrainDynamicsEverySteps = conf.JointTrainAgent.NumEnvs
    conf.JointTrainAgent.TrainAgentEverySteps = conf.JointTrainAgent.NumEnvs

    if conf.BasicSettings.ReplayBufferOnGPU:
        instructions_buffer = torch.empty((conf.JointTrainAgent.NumEnvs, 384), dtype=torch.float32, device="cuda", requires_grad=False)
    else:
        instructions_buffer = np.empty((conf.JointTrainAgent.NumEnvs, 384), dtype=np.float32)

    # distinguish between tasks, other debugging options are removed for simplicity
    if conf.Task == "JointTrainAgent":
        # getting action_dim with dummy env
        dummy_env = build_single_env(args.env_name[0], conf.BasicSettings.ImageSize, seed=0, skip=args.skip)
        if args.dist == "onehot":
            action_dim = dummy_env.action_space.n
        else:
            action_dim = dummy_env.action_space.shape[0]

        # build world model and agent
        world_model = build_world_model(conf, args, action_dim, cfg, TORCH_DTYPE)
        agent = build_agent(conf, args, action_dim)
        teacher_model = ActorCritic(cfg, torch.bfloat16)
        teacher_model.eval()

        world_model_params = sum(p.numel() for p in world_model.parameters())
        agent_params = sum(p.numel() for p in agent.parameters())
        print(f"world_model_params: {world_model_params}, agent_params: {agent_params}, total_params: {world_model_params+agent_params}") 

        # build replay buffer
        replay_buffer = ReplayBuffer(
            obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
            num_envs=conf.JointTrainAgent.NumEnvs,
            action_dim=action_dim,
            dist=args.dist,
            # instruction_buffer=instructions_buffer,
            max_length=conf.JointTrainAgent.BufferMaxLength,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU
        )

        # judge whether to load demonstration trajectory
        if conf.JointTrainAgent.UseDemonstration:
            print(colorama.Fore.MAGENTA + f"loading demonstration trajectory from {args.trajectory_path}" + colorama.Style.RESET_ALL)
            replay_buffer.load_trajectory(path=args.trajectory_path)

        # train
        joint_train_world_model_agent(
            env_name=args.env_name,
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            train_dynamics_every_steps=conf.JointTrainAgent.TrainDynamicsEverySteps,
            train_agent_every_steps=conf.JointTrainAgent.TrainAgentEverySteps,
            batch_size=conf.JointTrainAgent.BatchSize, # 16
            demonstration_batch_size=conf.JointTrainAgent.DemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            batch_length=conf.JointTrainAgent.BatchLength, # 64
            imagine_batch_size=conf.JointTrainAgent.ImagineBatchSize,
            imagine_demonstration_batch_size=conf.JointTrainAgent.ImagineDemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            imagine_context_length=conf.JointTrainAgent.ImagineContextLength,
            imagine_batch_length=conf.JointTrainAgent.ImagineBatchLength,
            save_every_steps=conf.JointTrainAgent.SaveEverySteps,
            seed=args.seed,
            logger=logger,
            skip=args.skip,
            multi_task=args.multi_task,
            cfg=cfg,
            teacher_model=teacher_model,
        )
    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")
