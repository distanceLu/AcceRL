#!/usr/bin/env python3
"""
Compute FID, FVD, LPIPS, PSNR, SSIM from saved per-camera .npy files.

Works with either:
- a legacy `views_dir` containing both `gt_<cam>.npy` and `pred_<cam>.npy`
- separate `gt_views_dir` and `pred_views_dir`

Each .npy file is (T, H, W, 3) uint8.

Camera names are auto-discovered from the first trajectory directory,
or can be specified with --cameras.

Usage:
    # Auto-discover cameras, compute all metrics:
    python -m weaver.utils.metrics \
        --views_dir ./eval/views --num_frames 50

    # Specific cameras and frame range:
    python -m weaver.utils.metrics \
        --views_dir ./eval/views --cameras wrist_image image \
        --start_frame 0 --num_frames 32

    # Skip slow metrics for quick check:
    python -m weaver.utils.metrics \
        --views_dir ./eval/views --num_frames 50 --skip_fvd --skip_fid
"""

import argparse
import gc
import json
import os
import tempfile
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from tqdm import tqdm


# =============================================================================
# Feature extractors
# =============================================================================

class InceptionV3Features(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.inception.eval()
        self.inception.fc = nn.Identity()
        for param in self.inception.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        if x.shape[-2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = x * 2 - 1
        return self.inception(x)


class I3DFeatures(nn.Module):
    _WEIGHT_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"

    def __init__(self):
        super().__init__()
        self.model = self._load_i3d_model()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def _load_i3d_model():
        cache_dir = os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "i3d"
        )
        os.makedirs(cache_dir, exist_ok=True)
        filepath = os.path.join(cache_dir, "i3d_torchscript.pt")
        if not os.path.isfile(filepath):
            print(f"Downloading I3D weights to {filepath} ...")
            torch.hub.download_url_to_file(I3DFeatures._WEIGHT_URL, filepath, progress=True)
        return torch.jit.load(filepath, map_location="cpu")

    @torch.no_grad()
    def forward(self, x):
        x = x * 2 - 1
        if x.shape[-2:] != (224, 224):
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x = x.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4).contiguous()
        return self.model(x, return_features=True)


# =============================================================================
# Statistics
# =============================================================================

def compute_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m} too large")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


# =============================================================================
# Metrics
# =============================================================================

def compute_fid_streaming(gt_views_dir, pred_views_dir, traj_dirs, cam_name, start_frame=0, num_frames=None,
                          batch_size=32, device='cuda'):
    """FID using pytorch-fid, streaming frames as PNGs to temp dirs."""
    from pytorch_fid import fid_score

    sf = start_frame
    ef = sf + num_frames if num_frames else None
    total_frames = 0

    with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as fake_dir:
        for traj_dir in traj_dirs:
            gt = np.load(os.path.join(gt_views_dir, traj_dir, f"gt_{cam_name}.npy"))[sf:ef]
            pred = np.load(os.path.join(pred_views_dir, traj_dir, f"pred_{cam_name}.npy"))[sf:ef]
            for i in range(len(gt)):
                idx = total_frames + i
                Image.fromarray(gt[i]).save(os.path.join(real_dir, f"{idx:06d}.png"))
                Image.fromarray(pred[i]).save(os.path.join(fake_dir, f"{idx:06d}.png"))
            total_frames += len(gt)

        print(f"  FID: {total_frames} frames saved, computing...")
        fid = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir], batch_size=batch_size, device=device, dims=2048,
        )
    torch.cuda.empty_cache()
    return fid


@torch.no_grad()
def compute_fvd(real_videos, fake_videos, batch_size=16, device='cuda'):
    """FVD using I3D features. Expects (N, C, T, H, W) float [0,1]."""
    if real_videos.ndim == 5 and real_videos.shape[1] != 3:
        real_videos = real_videos.permute(0, 2, 1, 3, 4)
    if fake_videos.ndim == 5 and fake_videos.shape[1] != 3:
        fake_videos = fake_videos.permute(0, 2, 1, 3, 4)

    feature_extractor = I3DFeatures().to(device).eval()
    print(f"  FVD: {len(real_videos)} real clips, {len(fake_videos)} fake clips")

    def extract_features(clips):
        features = []
        for i in range(0, len(clips), batch_size):
            batch = clips[i:i + batch_size].to(device)
            features.append(feature_extractor(batch).cpu().numpy())
        return np.concatenate(features, axis=0)

    real_features = extract_features(real_videos)
    fake_features = extract_features(fake_videos)
    mu_r, sigma_r = compute_statistics(real_features)
    mu_f, sigma_f = compute_statistics(fake_features)
    fvd = calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

    feature_extractor.cpu()
    torch.cuda.empty_cache()
    return fvd


@torch.no_grad()
def compute_lpips_streaming(gt_views_dir, pred_views_dir, traj_dirs, cam_name, start_frame=0, num_frames=None,
                            batch_size=64, device='cuda', impl='lpips'):
    """LPIPS (VGG) computed streaming, one video at a time."""
    if impl == 'lpips':
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    elif impl == 'torchmetrics':
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type='vgg',
            reduction='none',
            normalize=False,
        ).to(device).eval()
    else:
        raise ValueError(f"Unknown LPIPS implementation: {impl}")

    sf = start_frame
    ef = sf + num_frames if num_frames else None
    total_score = 0.0
    total_frames = 0

    for traj_dir in traj_dirs:
        gt = np.load(os.path.join(gt_views_dir, traj_dir, f"gt_{cam_name}.npy"))[sf:ef]
        pred = np.load(os.path.join(pred_views_dir, traj_dir, f"pred_{cam_name}.npy"))[sf:ef]
        real_t = torch.from_numpy(gt).float().permute(0, 3, 1, 2) / 255.0
        pred_t = torch.from_numpy(pred).float().permute(0, 3, 1, 2) / 255.0

        for i in range(0, len(real_t), batch_size):
            r = real_t[i:i + batch_size].to(device) * 2 - 1
            f = pred_t[i:i + batch_size].to(device) * 2 - 1
            scores = lpips_fn(r, f)
            total_score += scores.sum().item()
            total_frames += scores.numel()

        del real_t, pred_t

    lpips_fn.cpu()
    torch.cuda.empty_cache()
    return total_score / total_frames


@torch.no_grad()
def compute_psnr_streaming(gt_views_dir, pred_views_dir, traj_dirs, cam_name, start_frame=0, num_frames=None,
                           batch_size=64, device='cuda'):
    """PSNR from TorchMetrics for images normalized to [0, 1]."""
    from torchmetrics.functional.image import peak_signal_noise_ratio

    sf = start_frame
    ef = sf + num_frames if num_frames else None
    total_psnr = 0.0
    total_frames = 0

    for traj_dir in traj_dirs:
        gt = np.load(os.path.join(gt_views_dir, traj_dir, f"gt_{cam_name}.npy"))[sf:ef]
        pred = np.load(os.path.join(pred_views_dir, traj_dir, f"pred_{cam_name}.npy"))[sf:ef]
        real_t = torch.from_numpy(gt).float().permute(0, 3, 1, 2) / 255.0
        pred_t = torch.from_numpy(pred).float().permute(0, 3, 1, 2) / 255.0

        for i in range(0, len(real_t), batch_size):
            r = real_t[i:i + batch_size].to(device)
            f = pred_t[i:i + batch_size].to(device)
            psnr = peak_signal_noise_ratio(
                f, r,
                data_range=1.0,
                reduction='none',
                dim=(1, 2, 3),
            )
            total_psnr += psnr.sum().item()
            total_frames += psnr.numel()

        del real_t, pred_t

    torch.cuda.empty_cache()
    return total_psnr / total_frames


@torch.no_grad()
def compute_ssim_streaming(gt_views_dir, pred_views_dir, traj_dirs, cam_name, start_frame=0, num_frames=None,
                           batch_size=16, device='cuda'):
    """SSIM from TorchMetrics for images normalized to [0, 1]."""
    from torchmetrics.functional.image import structural_similarity_index_measure

    sf = start_frame
    ef = sf + num_frames if num_frames else None
    total_ssim = 0.0
    total_frames = 0

    for traj_dir in traj_dirs:
        gt = np.load(os.path.join(gt_views_dir, traj_dir, f"gt_{cam_name}.npy"))[sf:ef]
        pred = np.load(os.path.join(pred_views_dir, traj_dir, f"pred_{cam_name}.npy"))[sf:ef]
        real_t = torch.from_numpy(gt).float().permute(0, 3, 1, 2) / 255.0
        pred_t = torch.from_numpy(pred).float().permute(0, 3, 1, 2) / 255.0

        for i in range(0, len(real_t), batch_size):
            r = real_t[i:i + batch_size].to(device)
            f = pred_t[i:i + batch_size].to(device)
            ssim_per_frame = structural_similarity_index_measure(
                f, r,
                gaussian_kernel=True,
                sigma=1.5,
                kernel_size=11,
                reduction='none',
                data_range=1.0,
                k1=0.01,
                k2=0.03,
            )
            total_ssim += ssim_per_frame.sum().item()
            total_frames += ssim_per_frame.numel()

        del real_t, pred_t

    torch.cuda.empty_cache()
    return total_ssim / total_frames


# =============================================================================
# Data loading helpers
# =============================================================================

def discover_cameras(gt_views_dir, pred_views_dir):
    """Auto-discover camera names from the first available trajectory directory."""
    source_dir = gt_views_dir if gt_views_dir is not None else pred_views_dir
    prefix = "gt_" if gt_views_dir is not None else "pred_"
    traj_dirs = sorted(d for d in os.listdir(source_dir)
                       if os.path.isdir(os.path.join(source_dir, d)))
    if not traj_dirs:
        return []
    first_dir = os.path.join(source_dir, traj_dirs[0])
    cameras = []
    for f in sorted(os.listdir(first_dir)):
        if f.startswith(prefix) and f.endswith(".npy"):
            cam = f[len(prefix):-4]
            cameras.append(cam)
    return cameras


def parse_excluded_traj_ids(items):
    """Parse excluded trajectory ids from tokens like '60', '60-79'."""
    excluded = set()
    if not items:
        return excluded
    for item in items:
        token = str(item).strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start_i = int(start_s)
            end_i = int(end_s)
            if end_i < start_i:
                start_i, end_i = end_i, start_i
            excluded.update(str(i) for i in range(start_i, end_i + 1))
        else:
            excluded.add(str(int(token)))
    return excluded


def get_traj_dirs(gt_views_dir, pred_views_dir, cameras, max_trajs=None, exclude_traj_ids=None):
    """Get sorted list of trajectory directories present in both dirs with all cameras."""
    traj_dirs = sorted(d for d in os.listdir(pred_views_dir)
                       if os.path.isdir(os.path.join(pred_views_dir, d)))
    valid = []
    excluded = exclude_traj_ids or set()
    for d in traj_dirs:
        if d in excluded:
            continue
        gt_traj_path = os.path.join(gt_views_dir, d)
        pred_traj_path = os.path.join(pred_views_dir, d)
        if not os.path.isdir(gt_traj_path):
            continue
        if all(os.path.exists(os.path.join(gt_traj_path, f"gt_{cam}.npy"))
               and os.path.exists(os.path.join(pred_traj_path, f"pred_{cam}.npy"))
               for cam in cameras):
            valid.append(d)
    if max_trajs is not None:
        valid = valid[:max_trajs]
    return valid


def load_single_camera(gt_views_dir, pred_views_dir, traj_dirs, cam_name, start_frame=0, num_frames=None):
    """Load a single camera across all trajectories.

    Returns lists of per-video tensors, each (T_i, C, H, W) float [0,1].
    """
    sf = start_frame
    ef = sf + num_frames if num_frames else None
    real_list, pred_list = [], []
    for traj_dir in traj_dirs:
        gt = np.load(os.path.join(gt_views_dir, traj_dir, f"gt_{cam_name}.npy"))[sf:ef]
        pred = np.load(os.path.join(pred_views_dir, traj_dir, f"pred_{cam_name}.npy"))[sf:ef]
        real_list.append(torch.from_numpy(gt).float().permute(0, 3, 1, 2) / 255.0)
        pred_list.append(torch.from_numpy(pred).float().permute(0, 3, 1, 2) / 255.0)
    return real_list, pred_list


def pad_and_stack(vid_list, max_t, pad_short_clips=False):
    """Stack videos to a fixed length.

    When pad_short_clips is False, clips shorter than max_t are dropped.
    When pad_short_clips is True, short clips are padded by repeating the
    final frame until they reach max_t.
    """
    stacked = []
    for v in vid_list:
        t = v.shape[0]
        if t >= max_t:
            stacked.append(v[:max_t])
            continue
        if not pad_short_clips or t == 0:
            continue
        pad = v[-1:].repeat(max_t - t, 1, 1, 1)
        stacked.append(torch.cat([v, pad], dim=0))
    if not stacked:
        raise ValueError(f"No clips available to stack for max_t={max_t}")
    return torch.stack(stacked, dim=0)


def sliding_window_fvd_clips(vid_list, window_size=16, stride=8, pad_short_clips=True):
    """Extract training-style sliding FVD clips from per-video tensors.

    Mirrors the original WEAVER sliding-window metric implementation: clips are
    generated every `stride` frames and the final possible window is included.
    """
    clips = []
    for vid in vid_list:
        t = vid.shape[0]
        if t == 0:
            continue
        if t < window_size:
            if not pad_short_clips:
                continue
            pad = vid[-1:].repeat(window_size - t, 1, 1, 1)
            clips.append(torch.cat([vid, pad], dim=0))
            continue

        starts = list(range(0, t - window_size + 1, stride))
        if starts[-1] + window_size < t:
            starts.append(t - window_size)
        for s in starts:
            clips.append(vid[s:s + window_size])

    if not clips:
        raise ValueError(
            f"No clips available for sliding FVD: window_size={window_size}, stride={stride}"
        )
    return torch.stack(clips, dim=0)


def _sample_clip_start_indices(video_len, clip_len, num_clips, sampling, rng):
    """Return clip start indices for a single video."""
    max_start = max(video_len - clip_len, 0)
    if num_clips <= 0:
        return []
    if sampling == "first":
        return [0] * num_clips
    if sampling == "uniform":
        if num_clips == 1:
            return [max_start // 2]
        return [int(round(x)) for x in np.linspace(0, max_start, num_clips)]
    if sampling == "random":
        return [rng.randint(0, max_start) for _ in range(num_clips)]
    raise ValueError(f"Unknown sampling mode: {sampling}")


def sample_fvd_clips(vid_list, clip_len, num_clips=1, sampling="first", temporal_stride=1,
                     pad_short_clips=False, seed=0):
    """Sample fixed-length clips for FVD from per-video tensors.

    Args:
        vid_list: list of (T, C, H, W) float videos.
        clip_len: number of frames per sampled clip after temporal stride.
        num_clips: clips to sample per video.
        sampling: first | uniform | random
        temporal_stride: frame step within each clip.
        pad_short_clips: repeat last frame if video is too short for the requested clip.
        seed: RNG seed for random sampling.
    """
    rng = random.Random(seed)
    clips = []
    raw_span = 1 + (clip_len - 1) * temporal_stride

    for vid in vid_list:
        t = vid.shape[0]
        if t == 0:
            continue

        if t < raw_span:
            if not pad_short_clips:
                continue
            pad = vid[-1:].repeat(raw_span - t, 1, 1, 1)
            vid = torch.cat([vid, pad], dim=0)
            t = vid.shape[0]

        starts = _sample_clip_start_indices(t, raw_span, num_clips, sampling, rng)
        for s in starts:
            idx = torch.arange(s, s + raw_span, temporal_stride)
            clips.append(vid[idx])

    if not clips:
        raise ValueError(
            f"No clips available for FVD clip sampling: clip_len={clip_len}, "
            f"num_clips={num_clips}, temporal_stride={temporal_stride}"
        )

    return torch.stack(clips, dim=0)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute FID/FVD/LPIPS/PSNR/SSIM from saved per-camera .npy files")
    parser.add_argument('--views_dir', type=str, default=None,
                        help="Legacy path to views/ directory containing both gt_<cam>.npy and pred_<cam>.npy")
    parser.add_argument('--gt_views_dir', type=str, default=None,
                        help="Directory containing gt_<cam>.npy in per-trajectory subdirs")
    parser.add_argument('--pred_views_dir', type=str, default=None,
                        help="Directory containing pred_<cam>.npy in per-trajectory subdirs")
    parser.add_argument('--cameras', nargs='*', default=None,
                        help="Camera names (default: auto-discover from first trajectory)")
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=50,
                        help="Number of frames to use")
    parser.add_argument('--max_trajs', type=int, default=None)
    parser.add_argument('--exclude_traj_ids', nargs='*', default=None,
                        help="Trajectory ids or ranges to exclude, e.g. 60 61 70-79")
    parser.add_argument('--fvd_window', type=int, default=16)
    parser.add_argument('--fvd_stride', type=int, default=8)
    parser.add_argument('--fvd_clip_len', type=int, default=None,
                        help="If set, compute FVD on sampled clips of this many frames instead of the whole sliced video")
    parser.add_argument('--fvd_num_clips_per_video', type=int, default=1,
                        help="Number of FVD clips to sample from each video when --fvd_clip_len is set")
    parser.add_argument('--fvd_sampling', type=str, default='first',
                        choices=['first', 'uniform', 'random'],
                        help="Clip start sampling mode for FVD when --fvd_clip_len is set")
    parser.add_argument('--fvd_temporal_stride', type=int, default=1,
                        help="Temporal stride within each sampled FVD clip")
    parser.add_argument('--fvd_seed', type=int, default=0,
                        help="Random seed for FVD clip sampling")
    parser.set_defaults(fvd_sliding_window=True, pad_short_clips=True)
    parser.add_argument('--fvd_sliding_window', dest='fvd_sliding_window', action='store_true',
                        help="Use training-style sliding FVD windows with --fvd_window/--fvd_stride")
    parser.add_argument('--no_fvd_sliding_window', dest='fvd_sliding_window', action='store_false',
                        help="Disable training-style sliding FVD windows")
    parser.add_argument('--pad_short_clips', dest='pad_short_clips', action='store_true',
                        help="For FVD, pad short videos/clips by repeating the last frame instead of dropping them")
    parser.add_argument('--no_pad_short_clips', dest='pad_short_clips', action='store_false',
                        help="Drop FVD clips that are shorter than the requested length")
    parser.add_argument('--skip_fvd', action='store_true')
    parser.add_argument('--skip_fid', action='store_true')
    parser.add_argument('--skip_lpips', action='store_true')
    parser.add_argument('--lpips_impl', choices=['lpips', 'torchmetrics'], default='lpips')
    parser.add_argument('--skip_psnr', action='store_true')
    parser.add_argument('--skip_ssim', action='store_true')
    parser.add_argument('--output', type=str, default=None,
                        help="Path to save metrics JSON (default: <views_dir>/../metrics.json)")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.views_dir is not None:
        gt_views_dir = args.views_dir
        pred_views_dir = args.views_dir
    else:
        if args.gt_views_dir is None or args.pred_views_dir is None:
            raise ValueError("Provide either --views_dir or both --gt_views_dir and --pred_views_dir")
        gt_views_dir = args.gt_views_dir
        pred_views_dir = args.pred_views_dir

    # Discover or use specified cameras
    if args.cameras:
        cameras = args.cameras
    else:
        cameras = discover_cameras(gt_views_dir, pred_views_dir)
        if not cameras:
            print("No cameras found! Check your views_dir.")
            return
    print(f"Cameras: {cameras}")

    excluded = parse_excluded_traj_ids(args.exclude_traj_ids)
    traj_dirs = get_traj_dirs(
        gt_views_dir,
        pred_views_dir,
        cameras,
        max_trajs=args.max_trajs,
        exclude_traj_ids=excluded,
    )
    if not traj_dirs:
        print("No valid trajectories found!")
        return

    sf = args.start_frame
    ef = sf + args.num_frames if args.num_frames else 'end'
    print(f"Found {len(traj_dirs)} trajectories, frames [{sf}:{ef}]")
    if excluded:
        print(f"Excluded trajectories: {sorted(excluded, key=int)}")

    metrics = {}

    for cam in cameras:
        print(f"\n{'='*60}")
        print(f"Camera '{cam}': {len(traj_dirs)} videos")
        print(f"{'='*60}")

        if not args.skip_fvd:
            assert args.num_frames is not None, "FVD requires --num_frames to fix video length"
            real_list, pred_list = load_single_camera(
                gt_views_dir, pred_views_dir, traj_dirs, cam,
                start_frame=args.start_frame, num_frames=args.num_frames,
            )
            if args.fvd_sliding_window:
                real_stacked = sliding_window_fvd_clips(
                    real_list,
                    window_size=args.fvd_window,
                    stride=args.fvd_stride,
                    pad_short_clips=args.pad_short_clips,
                )
                pred_stacked = sliding_window_fvd_clips(
                    pred_list,
                    window_size=args.fvd_window,
                    stride=args.fvd_stride,
                    pad_short_clips=args.pad_short_clips,
                )
                print(
                    f"  Computing FVD ({len(real_stacked)} sliding clips, "
                    f"window={args.fvd_window}, stride={args.fvd_stride})..."
                )
            elif args.fvd_clip_len is not None:
                real_stacked = sample_fvd_clips(
                    real_list,
                    clip_len=args.fvd_clip_len,
                    num_clips=args.fvd_num_clips_per_video,
                    sampling=args.fvd_sampling,
                    temporal_stride=args.fvd_temporal_stride,
                    pad_short_clips=args.pad_short_clips,
                    seed=args.fvd_seed,
                )
                pred_stacked = sample_fvd_clips(
                    pred_list,
                    clip_len=args.fvd_clip_len,
                    num_clips=args.fvd_num_clips_per_video,
                    sampling=args.fvd_sampling,
                    temporal_stride=args.fvd_temporal_stride,
                    pad_short_clips=args.pad_short_clips,
                    seed=args.fvd_seed,
                )
                print(
                    f"  Computing FVD ({len(real_stacked)} clips, clip_len={args.fvd_clip_len}, "
                    f"stride={args.fvd_temporal_stride}, sampling={args.fvd_sampling})..."
                )
            else:
                real_stacked = pad_and_stack(real_list, args.num_frames, pad_short_clips=args.pad_short_clips)
                pred_stacked = pad_and_stack(pred_list, args.num_frames, pad_short_clips=args.pad_short_clips)
                print(f"  Computing FVD ({len(real_stacked)} videos, {args.num_frames} frames)...")
            metrics[f"fvd_{cam}"] = compute_fvd(
                real_stacked, pred_stacked, batch_size=8, device=device,
            )
            print(f"  FVD: {metrics[f'fvd_{cam}']:.2f}")
            del real_list, pred_list, real_stacked, pred_stacked
            torch.cuda.empty_cache()
            gc.collect()

        if not args.skip_fid:
            print(f"  Computing FID (streaming)...")
            metrics[f"fid_{cam}"] = compute_fid_streaming(
                gt_views_dir, pred_views_dir, traj_dirs, cam,
                start_frame=args.start_frame, num_frames=args.num_frames,
                batch_size=32, device=device,
            )
            print(f"  FID: {metrics[f'fid_{cam}']:.2f}")
            torch.cuda.empty_cache()
            gc.collect()

        if not args.skip_lpips:
            print(f"  Computing LPIPS (streaming)...")
            metrics[f"lpips_{cam}"] = compute_lpips_streaming(
                gt_views_dir, pred_views_dir, traj_dirs, cam,
                start_frame=args.start_frame, num_frames=args.num_frames,
                batch_size=64, device=device, impl=args.lpips_impl,
            )
            print(f"  LPIPS: {metrics[f'lpips_{cam}']:.4f}")
            torch.cuda.empty_cache()
            gc.collect()

        if not args.skip_psnr:
            print(f"  Computing PSNR (streaming)...")
            metrics[f"psnr_{cam}"] = compute_psnr_streaming(
                gt_views_dir, pred_views_dir, traj_dirs, cam,
                start_frame=args.start_frame, num_frames=args.num_frames,
                batch_size=64, device=device,
            )
            print(f"  PSNR: {metrics[f'psnr_{cam}']:.2f}")
            torch.cuda.empty_cache()
            gc.collect()

        if not args.skip_ssim:
            print(f"  Computing SSIM (streaming)...")
            metrics[f"ssim_{cam}"] = compute_ssim_streaming(
                gt_views_dir, pred_views_dir, traj_dirs, cam,
                start_frame=args.start_frame, num_frames=args.num_frames,
                batch_size=16, device=device,
            )
            print(f"  SSIM: {metrics[f'ssim_{cam}']:.4f}")
            torch.cuda.empty_cache()
            gc.collect()

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Summary (frames [{args.start_frame}:{args.start_frame + args.num_frames if args.num_frames else 'end'}])")
    print(f"{'='*60}")
    for cam in cameras:
        print(f"\n  {cam}:")
        for metric_name in ['fid', 'fvd', 'lpips', 'psnr', 'ssim']:
            mk = f"{metric_name}_{cam}"
            if mk in metrics:
                fmt = '.4f' if metric_name in ('lpips', 'ssim') else '.2f'
                print(f"    {metric_name:8s} {metrics[mk]:{fmt}}")
    print(f"{'='*60}")

    # Save
    output_root = args.views_dir if args.views_dir is not None else pred_views_dir
    output_path = args.output or os.path.join(os.path.dirname(output_root), "metrics.json")
    with open(output_path, 'w') as f:
        json.dump({
            "config": {
                "views_dir": args.views_dir,
                "gt_views_dir": gt_views_dir,
                "pred_views_dir": pred_views_dir,
                "cameras": cameras,
                "start_frame": args.start_frame,
                "num_frames": args.num_frames,
                "exclude_traj_ids": sorted(excluded, key=int),
                "fvd_window": args.fvd_window,
                "fvd_stride": args.fvd_stride,
                "fvd_sliding_window": args.fvd_sliding_window,
                "fvd_clip_len": args.fvd_clip_len,
                "fvd_num_clips_per_video": args.fvd_num_clips_per_video,
                "fvd_sampling": args.fvd_sampling,
                "fvd_temporal_stride": args.fvd_temporal_stride,
                "fvd_seed": args.fvd_seed,
                "num_trajectories": len(traj_dirs),
            },
            "metrics": {k: float(v) for k, v in metrics.items()},
        }, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
