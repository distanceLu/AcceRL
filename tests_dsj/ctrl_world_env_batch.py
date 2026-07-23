#!/usr/bin/env python3
"""
Ctrl-World 适配层：对外暴露与 Diamond / WorldModelEnvBatch 一致的接口。
实现与 AcceRL Ctrl-World checkpoint 一致的双视角、frame-aligned chunk 推理。
"""
from typing import Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor

from ctrl_world.models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline


class CtrlWorldEnvBatch:
    """
    把 Ctrl-World 包装成 AcceRL 期望的世界模型环境。

    AcceRL integration uses the stateless chunk interface:
        init_latent_state(obs) -> (history, current)
        predict_chunk_stateless(current, history, actions, instructions)

    主要转换：
        1. agentview / wrist 分别通过 SVD-VAE 编码
        2. 各视角 latent 沿高度拼接
        3. 历史 latent buffer 维护
        4. Ctrl-World 预测 [current + 4 future]，仅消费 future
        5. Reward/end remain in AcceRL's independent RewardInferenceActor
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
        num_frames_pred: int = 5,
        num_inference_steps: int = 50,
        condition_low: Optional[Tensor] = None,
        condition_high: Optional[Tensor] = None,
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
        if self.num_frames_pred < 2:
            raise ValueError("Ctrl-World must predict at least [current, next], so num_frames_pred >= 2")
        if condition_low is None or condition_high is None:
            raise ValueError("Ctrl-World action condition bounds are required")
        self.condition_low = torch.as_tensor(condition_low, dtype=torch.float32)
        self.condition_high = torch.as_tensor(condition_high, dtype=torch.float32)

        self.num_history = int(ctrl_world_model.args.num_history)
        self.action_dim = int(ctrl_world_model.args.action_dim)
        self.latent_h = target_height // 8
        self.latent_w = target_width // 8
        self.horizon = getattr(cfg, "horizon", 220)

    @property
    def device(self) -> torch.device:
        return self.vae.device

    # ------------------------------------------------------------------
    # 内部工具：每个视角独立 VAE 编解码，再沿 latent 高度拼接
    # ------------------------------------------------------------------
    def _encode_obs_to_latent(self, obs: Tensor) -> Tensor:
        """obs: [B,M,C,H,W] -> [B,4,M*latent_h,latent_w]."""
        if obs.ndim != 5:
            raise ValueError(f"Expected multi-view obs [B,M,C,H,W], got {tuple(obs.shape)}")
        B, M, C, H, W = obs.shape
        if M != self.num_cams:
            raise ValueError(f"Expected {self.num_cams} camera views, got {M}")
        x = obs.reshape(B * M, C, H, W)
        x = F.interpolate(
            x, size=(self.target_height, self.target_width),
            mode="bilinear", align_corners=False,
        ).to(dtype=self.vae.dtype, device=self.vae.device)
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        latent = latent.reshape(B, M, *latent.shape[1:])
        return latent.permute(0, 2, 1, 3, 4).reshape(
            B, latent.shape[2], M * latent.shape[3], latent.shape[4]
        )

    def _encode_obs_sequence(self, obs_seq: Tensor) -> Tensor:
        """Encode [B,T,M,C,H,W] into combined-view latents."""
        if obs_seq.ndim != 6:
            raise ValueError(f"Expected obs sequence [B,T,M,C,H,W], got {tuple(obs_seq.shape)}")
        B, T, M, C, H, W = obs_seq.shape
        latents = [self._encode_obs_to_latent(obs_seq[:, t]) for t in range(T)]
        return torch.stack(latents, dim=1)

    def init_latent_state(self, obs_seq: Tensor) -> Tuple[Tensor, Tensor]:
        """Return history strictly before current, plus current latent."""
        latents = self._encode_obs_sequence(obs_seq)
        previous, current = latents[:, :-1], latents[:, -1]
        if previous.shape[1] == 0:
            previous = current.unsqueeze(1)
        if previous.shape[1] < self.num_history:
            pad = previous[:, :1].repeat(
                1, self.num_history - previous.shape[1], 1, 1, 1
            )
            previous = torch.cat([pad, previous], dim=1)
        else:
            previous = previous[:, -self.num_history:]
        return previous, current

    def _decode_latent_to_obs(
        self, latent: Tensor, output_size: Optional[Tuple[int, int]] = None
    ) -> Tensor:
        """
        latent: [B, F, 4, latent_h*num_cams, latent_w]
        return: [B,F,M,C,H,W] in [-1,1].
        """
        B, num_frames, C, H, W = latent.shape
        if H != self.latent_h * self.num_cams or W != self.latent_w:
            raise ValueError(
                f"Unexpected combined latent size {(H, W)}; expected "
                f"{(self.latent_h * self.num_cams, self.latent_w)}"
            )
        per_view = latent.reshape(
            B, num_frames, C, self.num_cams, self.latent_h, self.latent_w
        ).permute(0, 3, 1, 2, 4, 5)
        latent_flat = per_view.reshape(
            B * num_frames * self.num_cams, C, self.latent_h, self.latent_w
        ) / self.vae.config.scaling_factor

        # VAE decode（AutoencoderKLTemporalDecoder 需要 num_frames 参数）
        decoded = []
        chunk_size = max(1, num_frames)
        for i in range(0, latent_flat.shape[0], chunk_size):
            chunk = latent_flat[i:i + chunk_size]
            decode_kwargs = {"num_frames": chunk.shape[0]}
            decoded.append(self.vae.decode(chunk, **decode_kwargs).sample)
        decoded = torch.cat(decoded, dim=0).reshape(
            B, self.num_cams, num_frames, 3, self.target_height, self.target_width
        ).permute(0, 2, 1, 3, 4, 5)
        target_size = output_size or (self.target_height, self.target_width)
        if tuple(target_size) != (self.target_height, self.target_width):
            decoded = F.interpolate(
                decoded.flatten(0, 2), size=target_size,
                mode="bilinear", align_corners=False,
            ).reshape(B, num_frames, self.num_cams, 3, *target_size)

        return decoded.clamp(-1, 1)

    # ------------------------------------------------------------------
    # 无状态接口（供 Worker 端维护 latent_history 时使用）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_chunk_stateless(
        self,
        current_latent: Tensor,
        latent_history: Tensor,
        action_condition: Tensor,
        instructions: List[str],
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        无状态预测下一帧。Worker 自行维护 latent_history，每次把完整状态传入。

        Args:
            current_latent:  [B,4,latent_h*num_cams,latent_w]
            latent_history:  [B, num_history, 4, latent_h*num_cams, latent_w]
            action_condition:[B, num_history + num_frames_pred, action_dim], raw actions
            instructions:    List[str], len == B
        Returns:
            future_obs:    [B,num_frames_pred-1,M,C,H,W]
            future_latents:[B,num_frames_pred-1,4,...]
        """
        total_len = self.num_history + self.num_frames_pred
        if action_condition.shape[1:] != (total_len, self.action_dim):
            raise ValueError(
                f"Expected action condition [B,{total_len},{self.action_dim}], "
                f"got {tuple(action_condition.shape)}"
            )
        low = self.condition_low.to(action_condition.device)
        high = self.condition_high.to(action_condition.device)
        act_cond = 2.0 * (action_condition - low) / (high - low + 1e-8) - 1.0
        act_cond = act_cond.clamp(-1.0, 1.0)
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
        # Decode the full temporal chunk (the temporal VAE expects the training
        # sequence), then discard frame 0 because it reconstructs current.
        decoded = self._decode_latent_to_obs(latents, output_size=output_size)
        future_latents = latents[:, 1:]
        future_obs = decoded[:, 1:]
        return future_obs, future_latents

    def init_latent_history(self, obs_seq: Tensor) -> Tensor:
        """
        Worker 端调用：从观测序列初始化 latent_history。
        obs_seq: [B,T,M,C,H,W] in [-1, 1]
        Returns: [B, num_history, 4, latent_h*num_cams, latent_w]
        """
        history, _ = self.init_latent_state(obs_seq)
        return history

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
