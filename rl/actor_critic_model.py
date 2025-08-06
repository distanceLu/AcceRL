import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from typing import Any, Tuple, Dict
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# 假设这些辅助函数位于可访问的路径
# 如果不在，您需要确保它们可以被导入
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head, 
    get_processor, 
    get_proprio_projector, 
    get_vla, 
    prepare_images_for_vla,
    normalize_proprio
)


class ActorCriticVLA(nn.Module):
    """
    一个封装了 OpenVLA 及其相关组件的 Actor-Critic 类，专为强化学习设计。

    该类在内部处理模型的加载和初始化，并提供一个高级接口 `get_action` 
    来从原始观测数据中获取动作及其相关 PPO 张量。

    核心功能:
    1. __init__ 方法通过一个配置对象 (cfg) 自动加载 VLA、处理器和动作头。
    2. 使用预设的 L1RegressionActionHead 作为 Actor 来预测动作均值。
    3. 使用一个新增的线性层作为 Critic 来预测状态价值。
    4. 使用一个可学习的参数来定义策略的探索标准差。
    5. 提供 get_action 方法，封装了从 observation 到 action 的所有预处理和前向传播步骤。
    """
    def __init__(self, cfg: GenerateConfig):
        """
        初始化函数。

        Args:
            cfg (GenerateConfig): 包含所有模型路径和配置参数的对象。
        """
        super().__init__()
        print("--> [ActorCriticVLA] Initializing components from configuration...")
        self.cfg = cfg

        # 步骤 1: 根据配置加载所有必要的组件
        self.vla: OpenVLAForActionPrediction = get_vla(self.cfg)
        self.processor = get_processor(self.cfg)
        self.action_mean_head: L1RegressionActionHead = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        
        # 只有在使用 proprio 时才加载投影仪
        if self.cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                self.cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM
            )
        else:
            self.proprio_projector = None

        # 步骤 2: 智能地获取设备和数据类型以初始化新层
        try:
            ref_param = next(self.vla.parameters())
            device = ref_param.device
            dtype = ref_param.dtype
        except StopIteration:
            device = "cpu"
            dtype = torch.float32
            print("[ActorCriticVLA WARNING] Could not determine device/dtype from VLA. Falling back to defaults.")

        # 步骤 3: 初始化 Critic (价值头) 和可学习的标准差
        self.value_head = nn.Linear(self.vla.llm_dim, 1).to(device=device, dtype=dtype)

        initial_log_std = torch.zeros(
            1, NUM_ACTIONS_CHUNK, self.action_mean_head.action_dim, 
            device=device, 
            dtype=dtype
        )
        self.action_log_std = nn.Parameter(initial_log_std)
        print("--> [ActorCriticVLA] Initialization complete.")

    def forward(
        self,
        **predict_action_kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        模型的核心前向传播。它接收 `vla.predict_action` 所需的所有参数。
        这个方法主要被 get_action 调用。

        Args:
            **predict_action_kwargs: 包含 `input_ids`, `pixel_values` 等的字典。

        Returns:
            - action (torch.Tensor): 采样得到的动作块 (B, chunk_len, action_dim)。
            - log_prob (torch.Tensor): 采样动作块的对数概率 (B, 1)。
            - entropy (torch.Tensor): 动作分布的熵 (B, 1)。
            - value (torch.Tensor): 状态价值 (B,)。
        """
        # 步骤 1: 从 VLA 获取 actions_hidden_states
        kwargs_for_vla = predict_action_kwargs.copy()
        kwargs_for_vla['action_head'] = None
        
        # 如果 VLA 被冻结，在训练循环中可以将其放入 torch.no_grad() 上下文
        _, actions_hidden_states = self.vla.predict_action(**kwargs_for_vla)

        # 步骤 2: Actor - 预测动作均值
        action_mean = self.action_mean_head.predict_action(actions_hidden_states)

        # 步骤 3: Critic - 预测状态价值
        aggregated_state = torch.mean(actions_hidden_states, dim=1)
        value = self.value_head(aggregated_state)

        # 步骤 4: 构建随机策略并采样
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        action_distribution = Normal(action_mean, action_std)

        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action).sum(dim=(-2, -1), keepdim=True)
        entropy = action_distribution.entropy().sum(dim=(-2, -1), keepdim=True)
        
        return action, log_prob, entropy, value.squeeze(-1)

    def get_action(
        self, 
        observation: Dict[str, Any], 
        inference_mode: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从原始观测数据中获取动作。
        此方法封装了所有预处理步骤和模型的前向传播。

        Args:
            observation (Dict[str, Any]): 包含图像和状态信息的观测字典。
            inference_mode (bool): 如果为 True，则在 `torch.inference_mode()` 下运行，
                                   不计算梯度，适用于评估。默认为 False。

        Returns:
            - action (torch.Tensor): 采样得到的动作块。
            - log_prob (torch.Tensor): 动作的对数概率。
            - entropy (torch.Tensor): 动作分布的熵。
            - value (torch.Tensor): 预测的状态价值。
        """
        # 根据 inference_mode 选择是否使用梯度上下文
        context = torch.inference_mode() if inference_mode else torch.enable_grad()
        
        with context:
            # 确定输入应发送到的设备和数据类型
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype

            # 准备图像输入
            all_images = [observation["full_image"]]
            if self.cfg.num_images_in_input > 1:
                all_images.extend([obs_img for k, obs_img in observation.items() if "wrist" in k])
            
            all_prepared_images = prepare_images_for_vla(all_images, self.cfg)
            primary_image = all_prepared_images.pop(0)
            additional_images = all_prepared_images

            # 准备文本输入
            prompt = f"In: What action should the robot take to {observation['task_description'].lower()}?\nOut:"
            inputs = self.processor(prompt, primary_image).to(device=device, dtype=dtype)
            
            if additional_images:
                all_wrist_inputs = [
                    self.processor(prompt, image_wrist).to(device=device, dtype=dtype) for image_wrist in additional_images
                ]
                primary_pixel_values = inputs["pixel_values"]
                all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
                inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

            # 准备本体感受输入
            proprio = None
            if self.cfg.use_proprio:
                proprio_state = observation["state"]
                proprio_norm_stats = self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]
                proprio = normalize_proprio(proprio_state, proprio_norm_stats)
                proprio = np.expand_dims(proprio, axis=0) # 添加 batch 维度

            # 组装 forward 方法所需的所有参数
            predict_action_kwargs = {
                **inputs,
                "unnorm_key": self.cfg.unnorm_key,
                "proprio": proprio,
                "proprio_projector": self.proprio_projector,
            }

            # 调用核心 forward 方法
            return self.forward(**predict_action_kwargs)

    def to(self, device: torch.device) -> "ActorCriticVLA":
        """
        将所有相关模型组件移动到指定的设备。

        Args:
            device (torch.device): 目标设备 (例如, torch.device("cuda"))。

        Returns:
            self (ActorCriticVLA): 移动到设备后的模型实例。
        """
        super().to(device)
        self.vla.to(device)
        self.action_mean_head.to(device)
        if self.proprio_projector:
            self.proprio_projector.to(device)
        print(f"--> [ActorCriticVLA] All components moved to {device}.")
        return self


if __name__ == "__main__":
    import os
    os.environ["MUJOCO_GL"] = "osmesa"
    
    # 步骤 1: 创建配置
    print("--> Loading configuration...")
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

    # 步骤 2: 仅用 cfg 初始化整个 ActorCriticVLA 模型
    # 模型加载和组件初始化现在都在类的内部完成
    actor_critic_model = ActorCriticVLA(cfg)

    # 步骤 3: 将模型移动到设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用我们重写的 .to() 方法来确保所有子模块都被移动
    actor_critic_model.to(DEVICE)

    # 步骤 4: 加载样本观测数据
    print("\n--> Loading sample observation...")
    with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
        observation = pickle.load(file)
    print("--> Sample observation loaded successfully.")

    # 步骤 5: 使用新的 get_action 方法执行前向传播
    # 注意：我们将 inference_mode 设置为 True 进行评估
    print("\n--> Executing a forward pass using the get_action method...")
    action_chunk, log_prob, entropy, value = actor_critic_model.get_action(
        observation, 
        inference_mode=True
    )

    # 步骤 6: 打印结果
    print("\n--- FORWARD PASS RESULTS ---")
    print(f"Sampled Action Chunk Shape: {action_chunk.shape}")
    print(f"First action in chunk: {action_chunk[0, 0, :]}")
    print(f"Log Probability Shape: {log_prob.shape}")
    print(f"Log Probability: {log_prob.item():.4f}")
    print(f"Entropy Shape: {entropy.shape}")
    print(f"Entropy: {entropy.item():.4f}")
    print(f"State Value Shape: {value.shape}")
    print(f"State Value: {value.item():.4f}")