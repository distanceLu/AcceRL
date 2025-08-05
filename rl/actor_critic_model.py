import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Dict, Any, Tuple, Optional
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


class ActorCriticVLA(nn.Module):
    """
    一个封装了 OpenVLAForActionPrediction 和 L1RegressionActionHead 的 Actor-Critic 类，
    专为 PPO 等强化学习算法设计。

    它利用 VLA 模型生成动作的中间隐藏状态，然后：
    1. 使用预设的 L1RegressionActionHead 作为 Actor 来预测动作均值。
    2. 使用一个新增的线性层作为 Critic 来预测状态价值。
    3. 使用一个可学习的参数来定义策略的探索标准差。
    """
    def __init__(
        self, 
        vla_model: OpenVLAForActionPrediction, 
        action_head: L1RegressionActionHead
    ):
        """
        初始化函数。

        Args:
            vla_model (OpenVLAForActionPrediction): 预训练好的 OpenVLA 模型实例。
            action_head (L1RegressionActionHead): 预训练好的 L1 回归动作头实例。
        """
        super().__init__()
        self.vla = vla_model
        self.action_mean_head = action_head

        # --- 智能的设备和数据类型对齐 (根据您的建议) ---
        # 1. 从传入的、已经配置好的vla_model中获取其设备和数据类型
        #    我们通过检查它的一个参数来安全地做到这一点。
        try:
            ref_param = next(self.vla.parameters())
            device = ref_param.device
            dtype = ref_param.dtype
        except StopIteration:
            # 如果模型没有参数，我们退回到默认值（这在实践中不太可能发生）
            device = "cpu"
            dtype = torch.float32
            print("[ActorCriticVLA WARNING] Could not determine device and dtype from vla_model. Falling back to defaults.")

        # 2. 使用获取到的device和dtype来初始化新的层和参数
        self.value_head = nn.Linear(self.vla.llm_dim, 1)
        self.value_head.to(device=device, dtype=dtype)

        # 创建初始张量时就指定好device和dtype
        initial_log_std = torch.zeros(
            1, NUM_ACTIONS_CHUNK, self.action_mean_head.action_dim, 
            device=device, 
            dtype=dtype
        )
        self.action_log_std = nn.Parameter(initial_log_std)
        # --- 对齐结束 ---

    def forward(
        self,
        **predict_action_kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        模型的前向传播。它接收 `vla.predict_action` 所需的所有参数。

        Args:
            **predict_action_kwargs: 一个字典，包含所有需要传递给 `self.vla.predict_action` 的参数，
                                     例如 `input_ids`, `pixel_values`, `attention_mask` 等。

        Returns:
            - action (torch.Tensor): 采样得到的动作块 (B, chunk_len, action_dim)。
            - log_prob (torch.Tensor): 采样动作块的对数概率 (B, 1)。
            - entropy (torch.Tensor): 动作分布的熵 (B, 1)。
            - value (torch.Tensor): 状态价值 (B,)。
        """
        # 步骤 1: 从 VLA 模型获取 actions_hidden_states
        # 我们通过将 action_head 设置为 None 来截获 VLA 的中间隐藏状态。
        kwargs_for_vla = predict_action_kwargs.copy()
        kwargs_for_vla['action_head'] = None
        
        # 调用 VLA 模型
        # 注意：在RL训练中，如果VLA部分被冻结，可以将其放入 torch.no_grad() 上下文中以提高效率。
        _, actions_hidden_states = self.vla.predict_action(**kwargs_for_vla)

        # 步骤 2: 演员网络 (Actor) - 预测动作均值
        # 使用预设的动作头来获取动作的均值
        action_mean = self.action_mean_head.predict_action(actions_hidden_states)

        # 步骤 3: 评论家网络 (Critic) - 预测状态价值
        # 为了计算单一的状态价值，我们将序列化的隐藏状态聚合成一个向量（例如，通过取平均值）。
        # actions_hidden_states shape: (B, chunk_len * action_dim, hidden_dim)
        aggregated_state = torch.mean(actions_hidden_states, dim=1)  # Shape: (B, hidden_dim)
        value = self.value_head(aggregated_state)

        # 步骤 4: 构建随机策略并从中采样
        # 将 log_std 扩展到与均值相同的形状，以便进行逐元素的分布创建
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        action_distribution = Normal(action_mean, action_std)

        # 从分布中采样一个动作块
        action = action_distribution.sample()

        # 计算整个动作块的对数概率和熵
        # PPO损失函数通常需要一个标量的log_prob和entropy，所以我们在动作维度上求和
        log_prob = action_distribution.log_prob(action).sum(dim=(-2, -1), keepdim=True)
        entropy = action_distribution.entropy().sum(dim=(-2, -1), keepdim=True)
        
        # 返回所有PPO需要的张量，确保价值张量的形状正确 (B,)
        return action, log_prob, entropy, value.squeeze(-1)
    

if __name__ == "__main__":
    import os
    import pickle
    import torch
    import numpy as np
    from experiments.robot.libero.run_libero_eval import GenerateConfig
    from experiments.robot.openvla_utils import (
        get_action_head, 
        get_processor, 
        get_proprio_projector, 
        get_vla, 
        prepare_images_for_vla,
        normalize_proprio
    )
    from prismatic.vla.constants import PROPRIO_DIM

    # --------------------------------------------------------------------------
    # 您的原始加载代码 (保持不变)
    # --------------------------------------------------------------------------
    os.environ["MUJOCO_GL"] = "osmesa"
    print("--> Loading configuration and models...")

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
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key="libero_spatial_no_noops",
    )

    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

    with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
        observation = pickle.load(file)
    print("--> Models and sample observation loaded successfully.")

    # --------------------------------------------------------------------------
    # 新增部分：集成 ActorCriticVLA 并执行前向传播
    # --------------------------------------------------------------------------

    # 1. 使用已加载的组件创建 Actor-Critic 模型
    print("\n--> Initializing ActorCriticVLA with pre-loaded components...")
    actor_critic_model = ActorCriticVLA(
        vla_model=vla,
        action_head=action_head
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic_model.to(DEVICE)
    proprio_projector.to(DEVICE)
    print(f"--> All model components moved to {DEVICE}.")

    # --------------------------------------------------------------------------
    # 准备输入 (严格遵循 get_vla_action 的逻辑)
    # --------------------------------------------------------------------------
    print("\n--> Preparing inputs for the forward pass (strictly following reference)...")
    with torch.inference_mode():
        all_images = [observation["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs_img for k, obs_img in observation.items() if "wrist" in k])
        
        all_prepared_images = prepare_images_for_vla(all_images, cfg)

        primary_image = all_prepared_images.pop(0)
        additional_images = all_prepared_images

        prompt = f"In: What action should the robot take to {observation['task_description'].lower()}?\nOut:"

        # **严格照抄**: 直接在 processor 输出上调用 .to()
        # 这会智能地转换数据类型，避免之前的问题
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
        
        if additional_images:
            # **严格照抄**: 对每个额外图像也执行相同的操作
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in additional_images
            ]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        proprio = None
        if cfg.use_proprio:
            # **严格照抄**: 保持 proprio 为 numpy 数组，让模型内部处理转换
            proprio_state = observation["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(proprio_state, proprio_norm_stats)
            # 为其增加一个 batch 维度以匹配模型输入
            proprio = np.expand_dims(proprio, axis=0)

        predict_action_kwargs = {
            **inputs,
            "unnorm_key": cfg.unnorm_key,
            "proprio": proprio,
            "proprio_projector": proprio_projector,
        }

        # --------------------------------------------------------------------------
        # 执行前向传播
        # --------------------------------------------------------------------------
        print("--> Executing a forward pass through ActorCriticVLA...")
        action_chunk, log_prob, entropy, value = actor_critic_model(**predict_action_kwargs)

        # --------------------------------------------------------------------------
        # 打印结果
        # --------------------------------------------------------------------------
        print("\n--- FORWARD PASS RESULTS ---")
        print(f"Sampled Action Chunk Shape: {action_chunk.shape}")
        print(f"First action in chunk: {action_chunk[0, 0, :]}")
        print(f"Log Probability Shape: {log_prob.shape}")
        print(f"Log Probability: {log_prob.item():.4f}")
        print(f"Entropy Shape: {entropy.shape}")
        print(f"Entropy: {entropy.item():.4f}")
        print(f"State Value Shape: {value.shape}")
        print(f"State Value: {value.item():.4f}")