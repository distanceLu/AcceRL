import os

import math
import torch
import torch.nn as nn
import numpy as np

from typing import Dict
from einops import rearrange
import yaml

from typing import List, Optional, Tuple
import imageio

class EMA:
    def __init__(
        self,
        model: torch.nn.Module,
        beta: float = 0.999,
        update_after_step: int = 100,
        update_every: int = 1,
    ):
        """
        EMA wrapper using Diffusers' EMAModel.
        Tracks ONLY parameters that require gradients.
        Includes warmup where decay ramps from 0 to beta.
        """
        from diffusers.training_utils import EMAModel

        self.model = model
        self.beta = beta

        self.params = [p for p in model.parameters() if p.requires_grad]

        self.ema = EMAModel(
            parameters=self.params,
            decay=beta,
            use_ema_warmup=True,
            update_after_step=update_after_step,
            update_every=update_every,
        )

    def to(self, device):
        self.ema.to(device)
        return self

    @torch.no_grad()
    def update(self):
        """
        Call AFTER optimizer.step().
        """
        self.ema.step(self.params)

    def apply_to(self, model: torch.nn.Module):
        """
        Copy EMA weights into the given model.
        """
        self.ema.copy_to(
            p for p in model.parameters() if p.requires_grad
        )

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state):
        self.ema.load_state_dict(state)

    # ---------------- Context manager ----------------
    class _EMAContext:
        def __init__(self, ema: "EMA"):
            self.ema = ema
            self.backup = None

        def __enter__(self):
            # Backup current trainable params
            self.backup = [
                p.detach().clone()
                for p in self.ema.model.parameters()
                if p.requires_grad
            ]

            # Apply EMA params
            self.ema.ema.copy_to(
                p for p in self.ema.model.parameters() if p.requires_grad
            )

            return self.ema.model

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original params
            idx = 0
            for p in self.ema.model.parameters():
                if p.requires_grad:
                    p.data.copy_(self.backup[idx])
                    idx += 1
            self.backup = None

    def use_ema_weights(self):
        return self._EMAContext(self)


from types import SimpleNamespace

def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [namespace_to_dict(v) for v in obj]
    else:
        return obj

def dict_to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    else:
        return obj


def save_checkpoint(
    chkpt_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Dict,
    step: int,
    suffix: str = '',
    save_config: bool = True,
    atomic: bool = True,
):
    """Save checkpoint to a single checkpoint.pt file (atomic overwrite)."""
    print (f'Saving model at step: {step} at {chkpt_dir}')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "ema": model.ema.state_dict(),
        "step": step,
    }
    ckpt_path = os.path.join(chkpt_dir, f'checkpoint{suffix}.pt')
    if atomic:
        tmp_path = ckpt_path + '.tmp'
        torch.save(state, tmp_path)
        os.replace(tmp_path, ckpt_path)
    else:
        torch.save(state, ckpt_path)

    if save_config:
        with open(os.path.join(chkpt_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(namespace_to_dict(cfg), f, sort_keys=False)


def load_checkpoint(
    chkpt_dir: str,
    device='cpu',
    weights_only: bool = True,
    checkpoint_name: str = 'checkpoint.pt',
):
    """Load checkpoint."""
    
    state = torch.load(
        os.path.join(chkpt_dir, checkpoint_name),
        map_location=device,
        # weights_only=False
    )
    
    if weights_only:
        return state
    
    with open(os.path.join(chkpt_dir, 'config.yaml'), "r") as f:
        cfg = dict_to_namespace(yaml.safe_load(f))
        
    print (f'Loaded model from step: {state["step"]} from {os.path.join(chkpt_dir, checkpoint_name)}')
    return state, cfg


def move_tensors_to_device(
    data: Dict[str, np.ndarray],
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:

    """Convert numpy arrays to torch tensors and move to device."""
    tensor_data = {}
    
    for key, value in data.items():
        if key == 'text':
            tensor_data[key] = value  # FIXME: keep as is for text processing later
            continue
        if isinstance(value, dict):
            tensor_data[key] = move_tensors_to_device(value, device=device)
        else:
            tensor_data[key] = value.to(device)

    return tensor_data

# @torch.compile
def merge_tensors(
    data1: Dict[str, torch.Tensor],
    data2: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:

    """Convert numpy arrays to torch tensors and move to device."""
    tensor_data = {}
    
    for key, value in data1.items():
        if key == 'text':
            tensor_data[key] = value  # FIXME: keep as is for text processing later
            continue
        if isinstance(value, dict):
            tensor_data[key] = merge_tensors(data1[key], data2[key])
        else:
            tensor_data[key] = torch.cat([value, data2[key]], dim=0)

    return tensor_data


def get_lr(
    it: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float
) -> float:
    # 1) linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * it / warmup_steps

    # 2) cosine decay to min_lr for the rest of the steps
    if it > max_steps:
        return min_lr

    # 3) In between use cosine decay to min_lr
    decay_steps = max_steps - warmup_steps
    if decay_steps <= 0:
        return min_lr
    decay_ratio = (it - warmup_steps) / decay_steps
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 1..0
    return min_lr + coeff * (max_lr - min_lr)


def cycle(loader):
    while True:
        for data in loader:
            yield data


def get_gpu_memory():
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    free = total - reserved
    print(f"Total: {total:.1f} MB | Reserved: {reserved:.1f} MB | Allocated: {allocated:.1f} MB | Free: {free:.1f} MB")


def _load_and_split(path: str, num_views=3, img_keys=None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load video, split top/bottom and by view. Returns (real_views, pred_views) per key."""
        print(f"Loading video from {path}...")
        reader = imageio.get_reader(path, format="ffmpeg")
        frames = [np.array(f) for f in reader]
        reader.close()
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        arr = np.clip(arr, 0, 1)
        # arr: (T, H_total, W_total, C)
        T, H_total, W_total, C = arr.shape
        H = H_total // 2
        W_per_view = W_total // num_views

        top = arr[:, :H]    # real: (T, H, W_total, C)
        bottom = arr[:, H:]  # pred: (T, H, W_total, C)

        real_views = {}
        pred_views = {}
        for i, key in enumerate(img_keys):
            start_w = i * W_per_view
            end_w = (i + 1) * W_per_view
            real_views[key] = top[:, :, start_w:end_w, :]   # (T, H, W, C)
            pred_views[key] = bottom[:, :, start_w:end_w, :]
        return real_views, pred_views

def load_videos_split_top_bottom(
    path_dir: str,
    img_keys: List[str],
    match_filenames: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]], List[str]]:
    """
    Load videos saved by wm_eval.py: all views in one file.
    Layout: top half = real (views side-by-side), bottom half = pred (same).
    Width = views concatenated horizontally in img_keys order.
    Returns lists of tensors (no stacking) since videos may have different lengths.

    Args:
        path_dir: Path to directory containing video files.
        img_keys: List of view keys in horizontal order (e.g. ['exterior_1_left', 'exterior_2_left', 'wrist_left']).
        match_filenames: If provided, only load files in this list (for cross-directory pairing).
                         Use filenames from a prior load to ensure same trajectory order.

    Returns:
        Tuple of (real_videos, pred_videos, loaded_filenames):
        - real_videos: Dict[str, List[Tensor]] each tensor (T_i, 3, H, W) per key
        - pred_videos: Dict[str, List[Tensor]] each tensor (T_i, 3, H, W) per key
        - loaded_filenames: Order of files loaded (for pairing with another directory)
    """
    import re
    num_views = len(img_keys)

    # Get sorted list of video files (deterministic order; natural sort for eval_sample0,1,2,10)
    all_files = [f for f in os.listdir(path_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if match_filenames is not None:
        match_set = set(match_filenames)
        all_files = [f for f in all_files if f in match_set]
        if all_files:
            # Preserve order of match_filenames so indices align across directories
            all_files.sort(key=lambda f: match_filenames.index(f) if f in match_filenames else 999999)
        else:
            # No filename overlap; fall back to sorted order (indices may not align)
            all_files = [f for f in os.listdir(path_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            all_files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])
            import warnings
            warnings.warn(f"No matching filenames in {path_dir}; using sorted order. Cross-directory pairing may be wrong.")
    else:
        all_files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])

    # Collect per-key lists across all paths (keep as lists; videos may have different T)
    real_by_key = {k: [] for k in img_keys}
    pred_by_key = {k: [] for k in img_keys}

    for path in all_files:
        full_path = os.path.join(path_dir, path)
        if not os.path.isfile(full_path):
            continue
        real_views, pred_views = _load_and_split(full_path, num_views=num_views, img_keys=img_keys)
        for key in img_keys:
            # (T, H, W, C) -> (T, 3, H, W) tensor
            arr = real_views[key]
            t = torch.from_numpy(np.transpose(arr, (0, 3, 1, 2))).float()
            real_by_key[key].append(t)
            arr = pred_views[key]
            t = torch.from_numpy(np.transpose(arr, (0, 3, 1, 2))).float()
            pred_by_key[key].append(t)

    real_videos = {k: real_by_key[k] for k in img_keys}
    pred_videos = {k: pred_by_key[k] for k in img_keys}

    return real_videos, pred_videos, all_files
