import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from collections import defaultdict
from typing import Dict, List

from .tools import move_tensors_to_device, get_gpu_memory, merge_tensors

@torch.no_grad()
def evaluate_with_metrics_ddp(
    model: nn.Module,
    raw_model: nn.Module,
    val_dataloader,
    img_keys: list,
    num_samples: int = 256,
    device: str = 'cuda',
    master_process: bool = False,
    world_size: int = 1,
    horizon: int = None,
    bootstrap: int = None,
):
    """
    Evaluate model with FID/FVD on master process only.

    Only rank 0 generates videos and computes metrics.
    Other ranks wait at a barrier.
    """
    import gc

    eval_metrics = defaultdict(float)

    # Non-master ranks: just wait at the barrier
    if not master_process:
        if world_size > 1:
            dist.barrier()
        return eval_metrics

    # --- Master process only from here ---
    get_gpu_memory()
    print(f"Master process generating {num_samples} samples for evaluation...")

    real_videos = defaultdict(list)
    fake_videos = defaultdict(list)
    samples_collected = 0

    for valid_batch in val_dataloader:
        valid_batch = move_tensors_to_device(valid_batch, device='cuda')
        val_memory = {k: v for k, v in valid_batch['memory'].items()} if 'memory' in valid_batch else None
        val_task = valid_batch.get('task', {})

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, decoded_obs_pred = raw_model.generate_videos_full(
                obs=valid_batch['obs'],
                actions=valid_batch['actions'],
                instructions=val_task,
                horizon=horizon,
                memory=val_memory,
                bootstrap=bootstrap,
            )

        T_pred = decoded_obs_pred[img_keys[0]].shape[1]
        for key in img_keys:
            real_videos[key].append(valid_batch['obs'][key][:, :T_pred].cpu())
            fake_videos[key].append(decoded_obs_pred[key].float().cpu())

        samples_collected += len(valid_batch['obs'][img_keys[0]])

        del valid_batch, val_memory, val_task, decoded_obs_pred
        torch.cuda.empty_cache()

        if samples_collected >= num_samples:
            break

    # Concatenate videos one key at a time and free memory immediately
    for k in img_keys:
        real_list = real_videos[k]
        fake_list = fake_videos[k]
        real_videos[k] = torch.cat(real_list, dim=0)[:num_samples]
        fake_videos[k] = torch.cat(fake_list, dim=0)[:num_samples]
        del real_list, fake_list
        gc.collect()

    print(f"Generation complete. Offloading model to CPU...")

    # Offload model to free GPU for metric computation
    # if isinstance(model, DDP):
    #     model.module.cpu()
    # else:
    #     model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # Compute metrics
    from .model_metrics import compute_fvd, compute_fid, compute_lpips
    print("Computing metrics...")

    for key in img_keys:
        print(f"Computing FVD for key: {key}...")
        eval_metrics[f"fvd_{key}"] = compute_fvd(
            real_videos[key], fake_videos[key], batch_size=8, device=device
        )
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Computing FID for key: {key}...")
        eval_metrics[f"fid_{key}"] = compute_fid(
            real_videos[key], fake_videos[key], batch_size=8, device=device
        )
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Computing LPIPS for key: {key}...")
        eval_metrics[f"lpips_{key}"] = compute_lpips(
            real_videos[key], fake_videos[key], batch_size=8, device=device
        )
        torch.cuda.empty_cache()
        gc.collect()

        eval_metrics[f"psnr_{key}"] = 0
        eval_metrics[f"ssim_{key}"] = 0

        print(f"Key: {key} | FVD: {eval_metrics[f'fvd_{key}']:.2f} | FID: {eval_metrics[f'fid_{key}']:.2f} | LPIPS: {eval_metrics[f'lpips_{key}']:.4f}")

    del real_videos, fake_videos
    torch.cuda.empty_cache()
    gc.collect()

    # Move model back to GPU
    print("Moving model back to GPU...")
    torch.cuda.empty_cache()

    # if isinstance(model, DDP):
    #     model.module.to(device)
    # else:
    #     model.to(device)

    get_gpu_memory()

    # Signal other ranks that evaluation is done
    if world_size > 1:
        dist.barrier()

    return eval_metrics





@torch.no_grad()
def evaluate_video_with_metrics_ddp(
    real_videos: Dict[str, torch.Tensor],
    pred_videos_sets: List[Dict[str, torch.Tensor]],
    img_keys: List[str],
    batch_size: int = 8,
    device: str = 'cuda',
) -> List[Dict[str, float]]:
    """
    Compute FID, FVD, LPIPS for each pred set vs real videos.

    Args:
        real_videos: Dict mapping img_key -> (N, T, 3, H, W) tensor or List[Tensor] of (T_i, 3, H, W).
                     List format supports variable-length videos; padded to max T internally.
        pred_videos_sets: List of dicts. Each dict maps img_key -> tensor or List[Tensor].
        img_keys: List of image keys (e.g. ['exterior_1_left', 'wrist_left']).
        batch_size: Batch size for metric computation.
        device: Device to run on.

    Returns:
        List of metric dicts, one per pred set. Each dict has keys like fvd_{key}, fid_{key}, lpips_{key}.
    """
    import gc
    from .model_metrics import compute_fvd, compute_fid, compute_lpips

    def _to_tensor(v):
        """Convert to tensor. Handles List[Tensor] (variable-length) by padding to max T."""
        if isinstance(v, list):
            # List of (T_i, 3, H, W) -> pad to max T, stack to (N, T_max, 3, H, W)
            max_t = max(t.shape[0] for t in v)
            padded = []
            for t in v:
                if t.shape[0] < max_t:
                    last = t[-1:].expand(max_t - t.shape[0], -1, -1, -1)
                    t = torch.cat([t, last], dim=0)
                padded.append(t)
            return torch.stack(padded, dim=0)
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v).float()
        return v.float() if v.dtype != torch.float32 else v

    def _align_lengths(real: torch.Tensor, pred: torch.Tensor):
        """Truncate to min length for LPIPS (requires same shape)."""
        # Assume (N, T, 3, H, W) or (N, 3, T, H, W)
        if real.ndim != 5 or pred.ndim != 5:
            return real, pred
        n_real, n_pred = real.shape[0], pred.shape[0]
        t_dim = 2 if real.shape[2] == 3 else 1  # (N,3,T,H,W) vs (N,T,3,H,W)
        t_real = real.shape[t_dim]
        t_pred = pred.shape[t_dim]
        n_min = min(n_real, n_pred)
        t_min = min(t_real, t_pred)
        if t_dim == 2:  # (N, 3, T, H, W)
            real = real[:n_min, :, :t_min]
            pred = pred[:n_min, :, :t_min]
        else:  # (N, T, 3, H, W)
            real = real[:n_min, :t_min]
            pred = pred[:n_min, :t_min]
        return real, pred

    results = []
    for set_idx, pred_videos in enumerate(pred_videos_sets):
        eval_metrics = defaultdict(float)
        print(f"\n--- Evaluating pred set {set_idx + 1}/{len(pred_videos_sets)} ---")

        for key in img_keys:
            if key not in real_videos or key not in pred_videos:
                print(f"Skipping key {key} (missing in real or pred)")
                continue

            real = _to_tensor(real_videos[key]).to(device)
            pred = _to_tensor(pred_videos[key]).to(device)

            # Clamp to [0, 1] if needed
            real = real.clamp(0, 1)
            pred = pred.clamp(0, 1)

            # FVD (handles different N, T via distribution stats)
            print(f"Computing FVD for key: {key}...")
            eval_metrics[f"fvd_{key}"] = compute_fvd(
                real, pred, batch_size=batch_size, device=device
            )
            torch.cuda.empty_cache()
            gc.collect()

            # FID (treats video frames as images)
            print(f"Computing FID for key: {key}...")
            eval_metrics[f"fid_{key}"] = compute_fid(
                real, pred, batch_size=batch_size, device=device
            )
            torch.cuda.empty_cache()
            gc.collect()

            # LPIPS (requires same shape; align lengths)
            real_aligned, pred_aligned = _align_lengths(real, pred)
            print(f"Computing LPIPS for key: {key}...")
            eval_metrics[f"lpips_{key}"] = compute_lpips(
                real_aligned, pred_aligned, batch_size=batch_size, device=device
            )
            torch.cuda.empty_cache()
            gc.collect()

            eval_metrics[f"psnr_{key}"] = 0.0
            eval_metrics[f"ssim_{key}"] = 0.0

            print(
                f"Key: {key} | FVD: {eval_metrics[f'fvd_{key}']:.2f} | "
                f"FID: {eval_metrics[f'fid_{key}']:.2f} | LPIPS: {eval_metrics[f'lpips_{key}']:.4f}"
            )

        results.append(dict(eval_metrics))

    torch.cuda.empty_cache()
    gc.collect()
    return results


def sample_and_merge_batch(
    exp_dataloader,
    pi_dataloader
):
    exp_batch = next(exp_dataloader)
    pi_batch = next(pi_dataloader)

    exp_batch = move_tensors_to_device(exp_batch, device='cuda')
    pi_batch = move_tensors_to_device(pi_batch, device='cuda')
    data = merge_tensors(exp_batch, pi_batch)
    
    return data    
