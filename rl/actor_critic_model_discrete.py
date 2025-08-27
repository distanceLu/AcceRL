import os
os.makedirs("obs", exist_ok=True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import pickle
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from contextlib import nullcontext
import numpy as np
from PIL import Image
import datetime
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
# from experiments.robot.openvla_utils import get_vla


# Core OpenVLA components
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
)

from experiments.robot.robot_utils import (
    invert_gripper_action,
)

# Masks used to extract action-related hidden states
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)

# Constants
from prismatic.vla.constants import (
    NUM_ACTIONS_CHUNK,
    ACTION_DIM,
    PROPRIO_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from typing import Any

# 显式类：避免依赖 auto_map
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_vla(cfg: Any) -> torch.nn.Module:
    """
    只读加载 OpenVLA：不修改 checkpoint 内的 config.json。
    """
    print("Instantiating pretrained VLA policy (read-only, no config.json mutation)...")

    # 1) 显式加载 Config（不会触发 auto_map 也不会写文件）
    vla_cfg = OpenVLAConfig.from_pretrained(
        cfg.pretrained_checkpoint,
        trust_remote_code=True,   # 允许自定义类
    )

    # 2) 显式加载模型（不走 Auto*，不需要 auto_map）
    vla = OpenVLAForActionPrediction.from_pretrained(
        cfg.pretrained_checkpoint,
        config=vla_cfg,
        torch_dtype=torch.float32, #bfloat16
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Load the model
    # vla = OpenVLAForActionPrediction.from_pretrained(
    #     cfg.pretrained_checkpoint,
    #     # attn_implementation="flash_attention_2",
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )

    # 3) FiLM（若启用）
    if getattr(cfg, "use_film", False):
        from experiments.robot.openvla_utils import _apply_film_to_vla
        vla = _apply_film_to_vla(vla, cfg)

    # 4) 设定输入图像数量
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()

    # 5) 未量化时放到目标设备
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # 6) 加载数据集统计（归一化/反归一化用）
    from experiments.robot.openvla_utils import _load_dataset_stats
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)

    return vla


class ActorCritic(nn.Module):
    """
    Actor-Critic for OpenVLA-based continuous control.

    forward(inputs_batch) returns:
      - actions_all: sampled actions in (-1, 1), shape (B, NUM_ACTIONS_CHUNK, ACTION_DIM)  [squashed Gaussian]
      - mu_all: mean actions from action_head.predict_action(...), shape (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
      - log_std_all: condition-independent log-std broadcast to all chunks, shape (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
      - value: state value estimate, shape (B,)
    """

    def __init__(self, cfg, torch_dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg

        # Device / dtype
        self.vla = get_vla(cfg)
        self.device = self.vla.device
        self.model_dtype = torch_dtype
        self.vla = self.vla.to(dtype=self.model_dtype)

        self.vocab_size = self.vla.config.text_config.vocab_size - self.vla.config.pad_to_multiple_of
        self.bins = np.linspace(-1, 1, self.vla.config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0


        # Keep processor for external preparation (forward 接收已组装好的 batch，但依旧保留 processor)
        self.processor = get_processor(cfg)

        # Heads
        # self.action_head = get_action_head(cfg, llm_dim=self.vla.llm_dim)
        # self.action_head = self.action_head.to(self.device).to(dtype=self.model_dtype)

        # self.proprio_projector = get_proprio_projector(
        #     cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM
        # )
        # self.proprio_projector = self.proprio_projector.to(self.device).to(dtype=self.model_dtype)
        self.proprio_projector = None
        # Condition-independent log_std parameter (float32 for stability)
        self.log_std_param = nn.Parameter(torch.full((NUM_ACTIONS_CHUNK, ACTION_DIM), -2, dtype=self.model_dtype, device=self.device))

        # Value head: mean-pool over text tokens from the last hidden layer -> scalar
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.vla.llm_dim),
            nn.Linear(self.vla.llm_dim, self.vla.llm_dim),
            nn.Tanh(),
            nn.Linear(self.vla.llm_dim, 1),
        ).to(self.device).to(dtype=self.model_dtype)

    def normalize_proprio(self, proprio: Any) -> np.ndarray:
        """
        Normalize proprioception data using self.vla.norm_stats[self.cfg.unnorm_key]["proprio"].
        Accepts numpy array or torch tensor; returns numpy array in [-1, 1].
        """
        # Convert to numpy
        if isinstance(proprio, torch.Tensor):
            proprio = proprio.detach().cpu().numpy()
        else:
            proprio = np.asarray(proprio)

        norm_stats = self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
            proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
            proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        normalized_proprio = np.clip(
            np.where(
                mask,
                2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
                proprio,
            ),
            a_min=-1.0,
            a_max=1.0,
        )
        return normalized_proprio

    def batch_process_obs(self, inputs_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Right-pad variable-length sequences across a list of samples and stack into a batch on self.vla.device.
        Expects each item to contain: input_ids, attention_mask, labels, pixel_values, proprio, etc.
        """
        # 目标序列最大长度（对齐到同一个 max_len，确保各 key 同长）
        max_len = max(it["input_ids"].size(1) for it in inputs_list)
        pad_id = int(self.vla.pad_token_id)

        # 对每条样本进行右侧 padding
        for it in inputs_list:
            cur_len = it["input_ids"].size(1)
            if cur_len < max_len:
                pad_amt = max_len - cur_len
                bsz = it["input_ids"].size(0)  # 通常为 1

                # input_ids: pad_id
                pad_ids = it["input_ids"].new_full((bsz, pad_amt), pad_id)
                it["input_ids"] = torch.cat([it["input_ids"], pad_ids], dim=1)

                # attention_mask: 0
                pad_mask = it["attention_mask"].new_zeros((bsz, pad_amt))
                it["attention_mask"] = torch.cat([it["attention_mask"], pad_mask], dim=1)

                # labels: -100
                pad_labels = it["labels"].new_full((bsz, pad_amt), -100)
                it["labels"] = torch.cat([it["labels"], pad_labels], dim=1)

        # 聚合成 batch，并移动到目标设备
        inputs: Dict[str, torch.Tensor] = {}
        keys = inputs_list[0].keys()
        for k in keys:
            tensors = [it[k] for it in inputs_list if it.get(k) is not None and isinstance(it[k], torch.Tensor)]
            # 如果 tensors 不为空，则拼接；否则跳过
            if tensors:
                inputs[k] = torch.cat(tensors, dim=0).to(self.vla.device)
            else:
                # 可选：跳过该键
                print(f"Warning: Key '{k}' has no valid tensors, skipping...")
        # inputs["proprio"] = inputs["proprio"].to(torch.float32)
        return inputs

    def prepare_inputs_batch(self, inputs_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        对多条样本执行：
          - 归一化 proprio 到 [-1, 1]
          - 基本一致性检查
          - 序列右侧 padding 并拼 batch
        """
        # Normalize proprio for each sample and run per-sample checks
        for it in inputs_list:
            # Normalize proprio using internal norm stats
            # proprio_norm = self.normalize_proprio(it["proprio"])
            # it["proprio"] = torch.tensor(it["proprio"], dtype=torch.float32)

            # Consistency check
            assert it["input_ids"].size(1) == it["attention_mask"].size(1) == it["labels"].size(1), \
                "Per-sample sequence lengths of input_ids/attention_mask/labels must match."

        # Batchify
        return self.batch_process_obs(inputs_list)

    def _compute_num_patches(self) -> int:
        num_patches = (
            self.vla.vision_backbone.get_num_patches()
            * self.vla.vision_backbone.get_num_images_in_input()
        )
        if self.cfg.use_proprio:
            num_patches += 1
        return num_patches

    def _extract_actions_hidden(self, logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        From last_hidden_states, extract the text-token hiddens corresponding
        to current + next actions, as (B, NUM_ACTIONS_CHUNK*ACTION_DIM, D).
        """
        ground_truth_token_ids = batch["labels"][:, 1:].to(self.device)  # (B, text_len-1)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)  # (B, text_len-1)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)      # (B, text_len-1)
        action_mask = current_action_mask | next_actions_mask

        num_patches = self._compute_num_patches()
        text_hidden_states = logits[:, num_patches:-1]  # (B, text_len, D)

        B, _, D = text_hidden_states.shape
        actions_hidden_states = (
            text_hidden_states[action_mask]
            .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)
            .to(self.model_dtype)
        )
        return actions_hidden_states

    def _forward_vla(self, batch: Dict[str, torch.Tensor]):
        """
        Single VLA forward that returns output with hidden states.
        """
        # ctx = torch.autocast("cuda", dtype=self.model_dtype) if self.device.type == "cuda" else nullcontext()
        # with ctx:
        self.vla: OpenVLAForActionPrediction
        output = self.vla.forward(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            pixel_values=batch["pixel_values"].to(self.model_dtype).to(self.device),
            labels=batch["labels"].to(self.device),  # for mask derivation and potential loss
            output_hidden_states=True,
            proprio=batch["proprio"] if self.cfg.use_proprio else None,
            proprio_projector=self.proprio_projector if self.cfg.use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=self.cfg.use_film,
        )
        return output

    def _compute_value_from_hidden(self, last_hidden_states: torch.Tensor) -> torch.Tensor:
        num_patches = self._compute_num_patches()
        text_hidden = last_hidden_states[:, num_patches:-1]  # (B, text_len, D)
        pooled = text_hidden.mean(dim=1)                     # (B, D)
        value = self.value_head(pooled.to(self.model_dtype)).squeeze(-1)  # (B,)
        return value.to(torch.float32)

    def forward(self, inputs_batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          actions_all: (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
          mu_all:      (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
          log_std_all: (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
          value:       (B,)
        """
        # Sanity checks
        for k in ("input_ids", "attention_mask", "pixel_values", "labels"):
            if k not in inputs_batch:
                raise KeyError(f"inputs_batch missing key: {k}")

        # 1. 提取动作部分的原始logits（未argmax）   
        output = self._forward_vla(inputs_batch)
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)

        logits = output.logits
        action_logits = self._extract_actions_hidden(logits, inputs_batch)


        # 2. 构建分类分布（每个动作维度独立）
        dist_per_dim = [
            torch.distributions.Categorical(logits=action_logits[:, i, :]) 
            for i in range(NUM_ACTIONS_CHUNK * ACTION_DIM)
        ]

        # 3. 采样动作（联合采样所有维度）
        # actions_all = torch.stack([dist.sample() for dist in dist_per_dim], dim=1)  # (B, NUM_ACTIONS_CHUNK * ACTION_DIM)
        actions_all = torch.argmax(action_logits, dim=2)
        # 4. 计算对数概率和熵（PPO需要）
        log_probs = torch.stack([
            dist.log_prob(actions_all[:, i]) 
            for i, dist in enumerate(dist_per_dim)
        ], dim=1).sum(dim=1)  # (B,)

        entropy = torch.stack([dist.entropy() for dist in dist_per_dim], dim=1).sum(dim=1)  # (B,)

        # 5. 反规范化离散动作（映射到实际值）
        discretized_actions = self.vocab_size - actions_all.cpu().numpy()
        # discretized_actions = self.vocab_size - predicted_action
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]  # (B, NUM_ACTIONS_CHUNK * ACTION_DIM)
        normalized_actions = torch.from_numpy(normalized_actions.reshape(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)).to(
            device=actions_all.device, dtype=torch.float32
        )

        # 6. 计算价值函数
        value = self._compute_value_from_hidden(last_hidden_states)  # (B,)

        return (
            normalized_actions,          # 替换原actions_all：实际动作值 (B, T, A)
            log_probs.to(torch.float32), # 替换原mu_all：对数概率 (B,)
            entropy.to(torch.float32),   # 替换原log_std_all：熵 (B,)
            value.to(torch.float32)      # 状态价值 (B,)
        )

if __name__ == "__main__":
    import sys
    import numpy as np

    # Libero env wrapper and helpers
    from rl.libero_env import LiberoEnvWrapper
    from rl.utils import prepare_one_obs
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    # Precision policy to match the example
    USE_BF16: bool = False
    TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

    # Instantiate config
    cfg = GenerateConfig(
        pretrained_checkpoint="/cpfs01/jinshiji_workspace/openvla_oft_rl/runs/openvla-7b-oft-finetuned-2_gpus_batch_size_16",
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key="libero_spatial_no_noops",
    )

    # Create ActorCritic policy
    actor = ActorCritic(cfg, TORCH_DTYPE)
    actor.eval()
    for key, value in actor.named_parameters():
        if value.dtype != TORCH_DTYPE:
            print(f"Warning: Parameter {key} has dtype {value.dtype}, expected {TORCH_DTYPE}.")

    print("正在初始化 LiberoEnvWrapper...")

    BENCHMARK = "libero_spatial"
    TASK_ID = 3  # e.g., pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate

    try:
        env = LiberoEnvWrapper(
            benchmark_name=BENCHMARK,
            task_id=TASK_ID,
            image_size=224,
            render_mode="rgb_array",
        )
    except Exception as e:
        print("\n--- 初始化失败 ---")
        print(f"错误: {e}")
        print("\n请确保：")
        print("1. 您已按照 LIBERO 的说明安装了所有依赖项。")
        print("2. 您已下载了 'libero_spatial' 数据集并放置在正确的位置。")
        print("3. 当前工作目录正确，以便脚本能够找到必要的工具函数。")
        sys.exit(1)

    print("\n--- 环境信息 ---")
    print(f"任务 ID: {env.task_id}")
    print(f"任务名称: {env.task.name}")
    print(f"任务描述: {env.task_description}")
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    print(f"最大步数: {env.max_episode_steps}")
    print("------------------\n")

    # Reset environment
    print("正在重置环境...")
    obs, info = env.reset()
    print("环境重置成功。")

    # Run one episode with the ActorCritic policy
    terminated, truncated = False, False
    total_reward = 0.0
    step = 0
    images_for_gif = []

    # 初始化动作队列
    from collections import deque
    action_queue = deque()
    
    while not (terminated or truncated):
        # Prepare single-sample inputs
        # with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
        #     observation = pickle.load(file)
        inputs_t = prepare_one_obs(cfg, actor.processor, obs, env.task_description, TORCH_DTYPE)  #obs observation, observation["task_description"]

        # 使用类方法封装的预处理：对列表执行 归一化 proprio + 一致性检查 + batchify
        inputs_batch = actor.prepare_inputs_batch([inputs_t])

        # Get actions from policy (all chunks); 用第一个 chunk 与环境交互
        with torch.inference_mode():
        # with torch.no_grad():
            actions_all, mu_all, log_std_all, value = actor.forward(inputs_batch)

        # 取第一个 chunk 的动作，并进行反归一化
        action_norm = actions_all[0, 0].cpu().numpy().astype(np.float32)  # in (-1, 1)
        action_env = actor.vla._unnormalize_actions(action_norm, cfg.unnorm_key)

        # action_env = invert_gripper_action(action_env) 

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action_env)
        full_image = Image.fromarray(obs['full_image'])
        # save_path = os.path.join("obs", f"full_image_{step}.png")  # 路径拼接
        # full_image.save(save_path)
        images_for_gif.append(full_image)

        total_reward += float(reward)
        step += 1
        print(
            f"Step {step}: "
            f"Reward={reward:.4f}, Terminated={terminated}, Truncated={truncated}, "
            f"Success={info.get('is_success', False)}"
        )
    if images_for_gif:
        gif_path = os.path.join("obs", "animation.gif")
        # 设置每帧显示时间（毫秒），这里设为100ms（0.1秒）
        images_for_gif[0].save(
            gif_path,
            save_all=True,
            append_images=images_for_gif[1:],
            duration=100,  # 每帧显示时间
            loop=0  # 0表示无限循环
        )
        print(f"GIF动画已保存到: {gif_path}")

    print("\nEpisode 结束。")
    print(f"总步数: {step}")
    print(f"总奖励: {total_reward:.4f}")

    env.close()
    print("环境已关闭。")