#!/usr/bin/env python3
"""
Ctrl-World 适配层：对外暴露与 Diamond / WorldModelEnvBatch 一致的接口。
当前版本为 smoke-test 级别，用于验证数据格式与通信，后续可替换生产逻辑。
"""
from typing import Any, Dict, List, Optional, Tuple
import time
import torch
import torch.nn.functional as F
from torch import Tensor

from rl.utils import prepare_one_obs_batch, prepare_inputs_batch
from envs.utils import tensor_to_image_batch
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline


class CtrlWorldEnvBatch:
    """
    把 Ctrl-World 包装成 AcceRL 期望的世界模型环境。

    接口对齐 WorldModelEnvBatch：
        reset(obs, act, instructions) -> (next_obs, info)
        step(act) -> (next_obs, rew, end, trunc, info)
        predict_next_obs() -> next_obs
        predict_rew_end(next_obs) -> (rew, end, info)
        imagine(obs, act, instructions, actions) -> dict

    主要转换：
        1. AcceRL 单张图 [B,3,H,W] -> Ctrl-World 多视角高度堆叠图
        2. 像素 -> SVD-VAE latent
        3. 历史 latent buffer 维护
        4. Ctrl-World pipeline 预测未来 latent -> VAE decode -> 像素
        5. 复用 OpenVLA reward model 产生 rew/end
    """

    def __init__(
        self,
        ctrl_world_model: Any,
        cfg: Any,
        reward_model: Optional[Any] = None,
        reward_cfg: Optional[Any] = None,
        processor: Optional[Any] = None,
        torch_dtype: Optional[torch.dtype] = None,
        instructions: Optional[List[str]] = None,
        num_cams: int = 2,
        target_height: int = 192,
        target_width: int = 320,
        num_frames_pred: int = 1,
        num_inference_steps: int = 4,
    ) -> None:
        self.ctrl_world = ctrl_world_model
        self.pipeline = ctrl_world_model.pipeline
        self.vae = self.pipeline.vae
        self.action_encoder = ctrl_world_model.action_encoder
        self.tokenizer = ctrl_world_model.tokenizer
        self.text_encoder = ctrl_world_model.text_encoder

        self.reward_model = reward_model
        self.reward_cfg = reward_cfg
        self.processor = processor
        self.torch_dtype = torch_dtype if torch_dtype is not None else torch.bfloat16

        self.instructions = instructions if instructions is not None else []
        self.num_cams = num_cams
        self.target_height = target_height
        self.target_width = target_width
        self.num_frames_pred = num_frames_pred
        self.num_inference_steps = num_inference_steps

        self.num_history = int(ctrl_world_model.args.num_history)
        self.action_dim = int(ctrl_world_model.args.action_dim)
        self.latent_h = target_height // 8
        self.latent_w = target_width // 8
        self.horizon = getattr(cfg, "horizon", 220)

        self.batch_size: int = 0
        self.obs_buffer: Optional[Tensor] = None          # [B, T, C, H, W] 像素 [-1,1]
        self.latent_history: Optional[Tensor] = None      # [B, num_history, 4, latent_h*num_cams, latent_w]
        self.act_buffer: Optional[Tensor] = None          # [B, T, action_dim]
        self.ep_len: Optional[Tensor] = None
        self.alive_mask: Optional[Tensor] = None
        self.last_success_prob: Optional[Tensor] = None

    @property
    def device(self) -> torch.device:
        return self.vae.device

    # ------------------------------------------------------------------
    # 内部工具：单视角 -> 多视角 -> latent
    # ------------------------------------------------------------------
    def _repeat_views(self, x: Tensor) -> Tensor:
        """把单张图复制成 num_cams 个视角并在高度方向堆叠。"""
        # x: [B, C, H, W]
        x = F.interpolate(
            x, size=(self.target_height, self.target_width),
            mode="bilinear", align_corners=False,
        )
        # 沿高度复制
        x = x.repeat(1, 1, self.num_cams, 1)
        return x  # [B, C, target_height*num_cams, target_width]

    def _encode_obs_to_latent(self, obs: Tensor) -> Tensor:
        """obs: [B, C, H, W] in [-1,1] -> latent: [B, 4, latent_h*num_cams, latent_w]"""
        x = self._repeat_views(obs)
        x = x.to(dtype=self.vae.dtype, device=self.vae.device)
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent

    def _build_latent_history(self, obs_seq: Tensor) -> Tensor:
        """把 reset 收到的观测序列编码成 num_history 长度的 latent history。"""
        B, T, C, H, W = obs_seq.shape
        latents = [self._encode_obs_to_latent(obs_seq[:, t]) for t in range(T)]
        latents = torch.stack(latents, dim=1)  # [B, T, 4, latent_h*num_cams, latent_w]
        if T < self.num_history:
            pad = latents[:, :1].repeat(1, self.num_history - T, 1, 1, 1)
            latents = torch.cat([pad, latents], dim=1)
        else:
            latents = latents[:, -self.num_history:]
        return latents

    def _decode_latent_to_obs(self, latent: Tensor) -> Tensor:
        """
        latent: [B, F, 4, latent_h*num_cams, latent_w]
        return: [B, F, C, H_orig, W_orig] in [-1,1]，取第一个视角并 resize 回原始尺寸
        """
        B, num_frames, C, H, W = latent.shape
        latent_flat = latent.flatten(0, 1) / self.vae.config.scaling_factor

        # VAE decode（AutoencoderKLTemporalDecoder 需要 num_frames 参数）
        decoded = []
        chunk_size = max(1, num_frames)
        for i in range(0, latent_flat.shape[0], chunk_size):
            chunk = latent_flat[i:i + chunk_size]
            decode_kwargs = {"num_frames": chunk.shape[0]}
            decoded.append(self.vae.decode(chunk, **decode_kwargs).sample)
        decoded = torch.cat(decoded, dim=0)  # [B*num_frames, 3, target_height*num_cams, target_width]
        decoded = decoded.reshape(B, num_frames, 3, self.target_height * self.num_cams, self.target_width)

        # 只取第一个视角
        decoded = decoded[:, :, :, :self.target_height, :]

        # resize 回 AcceRL 原始尺寸（由 reset 时 obs_buffer 决定）
        target_size = self.obs_buffer.shape[-2:]
        if target_size != (self.target_height, self.target_width):
            decoded = F.interpolate(
                decoded.flatten(0, 1), size=target_size,
                mode="bilinear", align_corners=False,
            ).reshape(B, num_frames, 3, *target_size)

        return decoded.clamp(-1, 1)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset(
        self,
        obs: Tensor,
        act: Tensor,
        instructions: Optional[List[str]] = None,
        initial_step_counts: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Args:
            obs: [B, T, C, H, W] 观测序列，范围 [-1,1]
            act: [B, T-1, action_dim]
            instructions: list of str
        Returns:
            (next_obs, info), next_obs: [B, C, H, W]
        """
        B, T, C, H, W = obs.shape
        self.batch_size = B
        self.obs_buffer = obs
        self.act_buffer = act
        self.ep_len = (
            initial_step_counts.to(obs.device).long()
            if initial_step_counts is not None
            else torch.zeros(B, dtype=torch.long, device=obs.device)
        )
        self.alive_mask = torch.ones(B, dtype=torch.bool, device=obs.device)

        if instructions is not None:
            assert len(instructions) == B
            self.instructions = instructions
        elif len(self.instructions) == 0:
            self.instructions = [""] * B
        elif len(self.instructions) == 1:
            self.instructions = self.instructions * B

        self.latent_history = self._build_latent_history(obs)

        # 初始化 last_success_prob
        next_obs_batch = obs[:, -1]
        self.last_success_prob, _, _ = self.predict_rew_end(next_obs_batch)

        return self.obs_buffer[:, -1], {}

    @torch.no_grad()
    def predict_rew_end(self, next_obs: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """
        next_obs: [B, C, H, W] in [-1,1]
        Returns: (rew: [B], end: [B], info)
        """
        if self.reward_model is None:
            # smoke test 模式：没有 reward model 时返回 0
            B = next_obs.shape[0]
            device = next_obs.device
            return (
                torch.zeros(B, device=device),
                torch.zeros(B, dtype=torch.long, device=device),
                {},
            )

        B = next_obs.shape[0]
        images_batch = tensor_to_image_batch(next_obs)  # list of uint8 HWC
        obs_batch = [{"full_image": img} for img in images_batch]
        instructions = [
            self.instructions[i] if i < len(self.instructions) else ""
            for i in range(B)
        ]

        inputs_list = prepare_one_obs_batch(
            self.reward_cfg, self.processor, obs_batch, instructions, self.torch_dtype
        )
        if self.reward_cfg is not None:
            for inputs in inputs_list:
                if (not getattr(self.reward_cfg, "use_proprio", False)) \
                        and ("proprio" in inputs) and (inputs["proprio"] is None):
                    inputs.pop("proprio", None)

        batch_inputs = prepare_inputs_batch(self.reward_model, inputs_list)
        logits = self.reward_model.forward(batch_inputs)
        probs = torch.softmax(logits, dim=-1)
        rew = probs[:, 1]
        end = logits.argmax(dim=-1)
        return rew, end, {}

    # ------------------------------------------------------------------
    # 无状态接口（供 Worker 端维护 latent_history 时使用）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_next_stateless(
        self,
        current_obs: Tensor,
        latent_history: Tensor,
        act_history: Tensor,
        instructions: List[str],
    ) -> Tuple[Tensor, Tensor]:
        """
        无状态预测下一帧。Worker 自行维护 latent_history，每次把完整状态传入。

        Args:
            current_obs:     [B, C, H, W] in [-1, 1]
            latent_history:  [B, num_history, 4, latent_h*num_cams, latent_w]
            act_history:     [B, num_history + num_frames_pred - 1, action_dim]
            instructions:    List[str], len == B
        Returns:
            next_obs:    [B, C, H, W] in [-1, 1]
            next_latent: [B, 4, latent_h*num_cams, latent_w]  (pipeline 直接输出，避免重复 VAE encode)
        """
        B = current_obs.shape[0]
        current_latent = self._encode_obs_to_latent(current_obs)  # [B, 4, H*num_cams, W]

        total_len = self.num_history + self.num_frames_pred
        if act_history.shape[1] < total_len:
            pad = torch.zeros(
                B, total_len - act_history.shape[1], self.action_dim,
                dtype=act_history.dtype, device=act_history.device,
            )
            act_cond = torch.cat([pad, act_history], dim=1)
        else:
            act_cond = act_history[:, -total_len:]

        act_dtype = self.action_encoder.action_encode[0].weight.dtype
        act_cond = act_cond.to(act_dtype)

        text_token = self.action_encoder(
            act_cond, instructions, self.tokenizer, self.text_encoder
        )

        _, latents = CtrlWorldDiffusionPipeline.__call__(
            self.pipeline,
            image=current_latent,
            text=text_token,
            width=self.target_width,
            height=self.target_height * self.num_cams,
            num_frames=self.num_frames_pred,
            history=latent_history,
            num_inference_steps=self.num_inference_steps,
            decode_chunk_size=self.num_frames_pred,
            max_guidance_scale=1.0,
            fps=7,
            motion_bucket_id=127,
            output_type="latent",
            return_dict=False,
            frame_level_cond=True,
        )
        # latents: [B, num_frames_pred, 4, latent_h*num_cams, latent_w]
        next_latent = latents[:, 0]  # [B, 4, latent_h*num_cams, latent_w]
        next_obs = self._decode_latent_to_obs(latents)[:, 0]  # [B, C, H, W]
        return next_obs, next_latent

    def init_latent_history(self, obs_seq: Tensor) -> Tensor:
        """
        Worker 端调用：从观测序列初始化 latent_history。
        obs_seq: [B, T, C, H, W] in [-1, 1]
        Returns: [B, num_history, 4, latent_h*num_cams, latent_w]
        """
        return self._build_latent_history(obs_seq)

    @staticmethod
    def update_latent_history(latent_history: Tensor, new_latent: Tensor) -> Tensor:
        """
        Worker 端调用：滑动更新 latent_history（弹出最旧帧，加入新帧）。
        latent_history: [B, num_history, 4, ...]
        new_latent:     [B, 4, ...]
        Returns:        [B, num_history, 4, ...]
        """
        new_latent = new_latent.unsqueeze(1)  # [B, 1, 4, ...]
        return torch.cat([latent_history[:, 1:], new_latent], dim=1)

    # ------------------------------------------------------------------
    # 有状态接口（向后兼容，内部委托给无状态方法）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_next_obs(self) -> Tuple[Tensor, List[Any]]:
        """
        有状态版本：使用 self.obs_buffer / self.latent_history / self.act_buffer 预测下一帧。
        Returns: (next_obs: [B, C, H, W], trajectory: list)
        """
        current_obs = self.obs_buffer[:, -1]  # [B, C, H, W]
        next_obs, next_latent = self.predict_next_stateless(
            current_obs=current_obs,
            latent_history=self.latent_history,
            act_history=self.act_buffer,
            instructions=self.instructions,
        )
        # 缓存 next_latent 供 step() 使用，避免重复 VAE encode
        self._cached_next_latent = next_latent
        return next_obs, []

    @torch.no_grad()
    def step(self, act: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        """
        act: [B, action_dim]
        Returns: (next_obs, rew, end, trunc, info)
        """
        B = self.batch_size
        act = act * self.alive_mask.unsqueeze(-1)
        self.act_buffer = torch.cat([self.act_buffer, act.unsqueeze(1)], dim=1)

        predict_obs_start = time.time()
        next_obs, _ = self.predict_next_obs()
        predict_obs_time = time.time() - predict_obs_start

        # mask dead envs
        next_obs = torch.where(
            self.alive_mask.view(B, 1, 1, 1).expand_as(next_obs),
            next_obs,
            self.obs_buffer[:, -1],
        )

        predict_rew_start = time.time()
        success_prob, end, _ = self.predict_rew_end(next_obs)
        predict_rew_time = time.time() - predict_rew_start

        success_prob = torch.where(self.alive_mask, success_prob, self.last_success_prob)
        end = end * self.alive_mask.long()

        rew = success_prob - self.last_success_prob
        rew = rew * self.alive_mask.float()
        self.last_success_prob = success_prob

        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long() * self.alive_mask.long()

        # update obs buffer
        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs

        # update latent history（使用 pipeline 输出的 latent，避免重复 VAE encode）
        if hasattr(self, '_cached_next_latent') and self._cached_next_latent is not None:
            self.latent_history = self.update_latent_history(self.latent_history, self._cached_next_latent)
            self._cached_next_latent = None
        else:
            new_latent = self._encode_obs_to_latent(next_obs).unsqueeze(1)
            self.latent_history = torch.cat([self.latent_history[:, 1:], new_latent], dim=1)

        # pop oldest action
        self.act_buffer = self.act_buffer[:, 1:]

        dead = torch.logical_or(end.bool(), trunc.bool()) if self.reward_model is not None else trunc.bool()
        dead = dead & self.alive_mask
        self.alive_mask = self.alive_mask & (~dead)

        info = {
            "ep_len": self.ep_len.cpu().numpy(),
            "truncated": trunc.cpu().numpy(),
            "dead": dead.cpu().numpy(),
            "alive": self.alive_mask.cpu().numpy(),
            "success_prob": success_prob.cpu().numpy(),
            "predict_obs_time": predict_obs_time,
            "predict_rew_time": predict_rew_time,
        }
        return self.obs_buffer[:, -1], rew, end, trunc, info

    @torch.no_grad()
    def imagine(
        self,
        obs: Tensor,
        act: Tensor,
        instructions: Optional[List[str]] = None,
        actions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        自回归想象轨迹。
        obs: [B, T, C, H, W]
        act: [B, T-1, action_dim]
        actions: [B, H, action_dim]
        """
        B, T, C, H, W = obs.shape
        H_plan = self.horizon
        initial_obs, _ = self.reset(obs, act, instructions)

        obs_list = [initial_obs]
        act_list, rew_list, end_list, trunc_list, alive_list = [], [], [], [], []

        if actions is None:
            actions = torch.zeros(B, H_plan, self.action_dim, device=self.device, dtype=act.dtype)
        else:
            assert actions.shape == (B, H_plan, self.action_dim)

        for step in range(H_plan):
            act_step = actions[:, step]
            next_obs, rew, end, trunc, info = self.step(act_step)
            obs_list.append(next_obs)
            act_list.append(act_step)
            rew_list.append(rew)
            end_list.append(end)
            trunc_list.append(trunc)
            alive_list.append(self.alive_mask)
            if not self.alive_mask.any():
                remaining = H_plan - step - 1
                if remaining > 0:
                    obs_list.extend([next_obs] * remaining)
                    act_list.extend([torch.zeros_like(act_step)] * remaining)
                    rew_list.extend([torch.zeros_like(rew)] * remaining)
                    end_list.extend([torch.zeros_like(end)] * remaining)
                    trunc_list.extend([torch.zeros_like(trunc)] * remaining)
                    alive_list.extend([torch.zeros_like(self.alive_mask)] * remaining)
                break

        return {
            "obs": torch.stack([initial_obs] + obs_list[1:], dim=1),
            "act": torch.stack(act_list, dim=1),
            "rew": torch.stack(rew_list, dim=1),
            "end": torch.stack(end_list, dim=1),
            "trunc": torch.stack(trunc_list, dim=1),
            "alive": torch.stack(alive_list, dim=1),
        }
