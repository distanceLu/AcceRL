"""
FID (Fréchet Inception Distance) and FVD (Fréchet Video Distance) Implementation
Computes perceptual similarity metrics for images and videos
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from typing import Tuple, Optional
import warnings
from einops import rearrange
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

import lpips
# LPIPS network – choose 'alex', 'vgg', or 'squeeze'
lpips_loss_fn = lpips.LPIPS(net='vgg') #.to(device)  
lpips_loss_fn.eval()  # disable training mode


# Try to import required models
try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    INCEPTION_AVAILABLE = True
except ImportError:
    INCEPTION_AVAILABLE = False
    warnings.warn("torchvision not available. Install with: pip install torchvision")

try:
    from torcheval.metrics.functional import frechet_inception_distance as fid_score
except ImportError:
    pass



def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of features.
    
    Args:
        features: (N, D) array of features
    
    Returns:
        mu: (D,) mean vector
        sigma: (D, D) covariance matrix
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Calculate Fréchet distance between two Gaussian distributions.
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Args:
        mu1, mu2: Mean vectors
        sigma1, sigma2: Covariance matrices
        eps: Small value for numerical stability
    
    Returns:
        fid: Fréchet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        print(f"Warning: FID calculation produced inf/nan. Adding {eps} to diagonal.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m} too large")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return float(fid)


# =============================================================================
# FID (Fréchet Inception Distance) - For Images
# =============================================================================

class InceptionV3Features(nn.Module):
    """
    Inception V3 feature extractor for FID computation.
    Returns 2048-dimensional features from the last pooling layer.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        
        if not INCEPTION_AVAILABLE:
            raise ImportError("torchvision not available")
        
        # Load pretrained Inception V3
        self.inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False
        )
        
        self.inception.eval()
        
        # Remove the final classification layer
        self.inception.fc = nn.Identity()
        
        # Freeze all parameters
        for param in self.inception.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: Images of shape (B, 3, H, W) in range [0, 1]
        Returns:
            features: (B, 2048) feature vectors
        """
        # Inception expects input in [-1, 1]
        x = x * 2 - 1
        
        # Resize to 299x299 if needed
        if x.shape[-2:] != (299, 299):
            x = torch.nn.functional.interpolate(
                x, size=(299, 299), mode='bilinear', align_corners=False
            )
        
        # Extract features
        features = self.inception(x)
        
        return features


@torch.no_grad()
def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    batch_size: int = 50,
    device: str = 'cuda'
) -> float:
    """
    Compute FID between real and generated images.
    
    Args:
        real_images: (N, 3, H, W) tensor in [0, 1]
        fake_images: (M, 3, H, W) tensor in [0, 1]
        batch_size: Batch size for feature extraction
        device: Device to run on
    
    Returns:
        fid: FID score (lower is better)
    """
    
    # Handle different input formats
    if real_images.ndim == 5 and real_images.shape[2] == 3:
        # (B, T, 3, H, W) -> (B, 3, T, H, W)
        real_images = rearrange(real_images, 'b t c h w -> (b t) c h w')
    if fake_images.ndim == 5 and fake_images.shape[2] == 3:
        fake_images = rearrange(fake_images, 'b t c h w -> (b t) c h w')
        
    # Initialize feature extractor
    feature_extractor = InceptionV3Features()
    feature_extractor.to(device)
    feature_extractor.eval()
    
    def extract_features(images):
        """Extract features from images in batches"""
        features = []
        num_batches = (len(images) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            batch = images[i * batch_size:(i + 1) * batch_size].to(device)
            feat = feature_extractor(batch)
            features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    # Extract features
    print("Extracting features from real images...")
    real_features = extract_features(real_images)
    
    print("Extracting features from fake images...")
    fake_features = extract_features(fake_images)
    
    # Compute statistics
    mu_real, sigma_real = compute_statistics(real_features)
    mu_fake, sigma_fake = compute_statistics(fake_features)
    
    # Compute FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    feature_extractor.cpu()  # free up memory
    torch.cuda.empty_cache()  # free up GPU memory
    return fid


@torch.no_grad()
def compute_psnr_ssim(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    batch_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute SSIM & PSNR in smaller chunks to avoid OOM.
    
    Args:
        real, fake: (N, C, H, W) or (B, T, C, H, W)
        batch_size: how many images to process at once
    """
    assert real_images.shape == fake_images.shape
    
    # flatten video case (B, T, C, H, W) → (B*T, C, H, W)
    if real_images.ndim == 5:
        B, T, C, H, W = real_images.shape
        real_images = rearrange(real_images, 'b t c h w -> (b t) c h w')
        fake_images = rearrange(fake_images, 'b t c h w -> (b t) c h w')
    ssim_fn = SSIM(data_range=1.0) 
     
    N = real_images.size(0)
    psnr_list, ssim_list = [], []

    for i in range(0, N, batch_size):
        r = real_images[i:i+batch_size]
        f = fake_images[i:i+batch_size]

        # --- PSNR ---
        mse = torch.mean((r - f) ** 2, dim=[1,2,3])  # per-image
        psnr = 20 * torch.log10(1.0 / (mse + 1e-8)).mean()  # average batch
        psnr_list.append(psnr)

        # --- SSIM ---
        ssim = ssim_fn(f, r)   # already averaged over the batch
        ssim_list.append(ssim)

    # final average over all chunks
    psnr_final = torch.stack(psnr_list).mean().item()
    ssim_final = torch.stack(ssim_list).mean().item()
    return psnr_final, ssim_final

# =============================================================================
# FVD (Fréchet Video Distance) - For Videos
# =============================================================================

class I3DFeatures(nn.Module):
    """
    I3D (Inflated Inception V3) feature extractor for FVD computation.
    Uses the I3D model pretrained on Kinetics-400, which is the standard
    feature extractor for FVD as defined in "Towards Accurate Generative
    Models of Video: A New Metric & Challenges" (Unterthiner et al., 2019).

    Weights are automatically downloaded from the canonical PyTorch-I3D
    conversion (originally from https://github.com/piergiaj/pytorch-i3d).
    """

    _WEIGHT_URL = (
        "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
    )

    def __init__(self):
        super().__init__()
        self.model = self._load_i3d_model()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def _load_i3d_model():
        """Download and cache the I3D TorchScript model."""
        import os
        cache_dir = os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "i3d",
        )
        os.makedirs(cache_dir, exist_ok=True)
        filepath = os.path.join(cache_dir, "i3d_torchscript.pt")

        if not os.path.isfile(filepath):
            print(f"Downloading I3D weights to {filepath} ...")
            torch.hub.download_url_to_file(
                I3DFeatures._WEIGHT_URL, filepath, progress=True
            )

        model = torch.jit.load(filepath, map_location="cpu")
        return model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Videos of shape (B, 3, T, H, W) in range [0, 1]
        Returns:
            features: (B, 400) pre-softmax logits
        """
        x = x * 2 - 1
        # Resize ourselves to avoid .view() crash on non-contiguous tensors
        if x.shape[-2:] != (224, 224):
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
            x = x.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4).contiguous()
        return self.model(x, return_features=True)


_i3d_features: Optional[I3DFeatures] = None


def _sliding_window_clips(videos: torch.Tensor, window_size: int = 16, stride: int = 8):
    """
    Extract sliding-window clips from a batch of videos (StyleGAN-V style).

    Args:
        videos: (N, 3, T, H, W) tensor
        window_size: number of frames per clip (I3D expects 16)
        stride: step between consecutive windows

    Returns:
        clips: (num_total_clips, 3, window_size, H, W) tensor on CPU
    """
    N, C, T, H, W = videos.shape
    clips = []
    for i in range(N):
        vid = videos[i]  # (3, T, H, W)
        if T < window_size:
            # Pad short videos by repeating the last frame
            pad = vid[:, -1:].expand(-1, window_size - T, -1, -1)
            clips.append(torch.cat([vid, pad], dim=1).unsqueeze(0))
        else:
            starts = list(range(0, T - window_size + 1, stride))
            # Ensure we always include the last possible window
            if starts[-1] + window_size < T:
                starts.append(T - window_size)
            for s in starts:
                clips.append(vid[:, s:s + window_size].unsqueeze(0))
    return torch.cat(clips, dim=0)


@torch.no_grad()
def compute_fvd(
    real_videos: torch.Tensor,
    fake_videos: torch.Tensor,
    batch_size: int = 16,
    device: str = 'cuda',
    window_size: int = 16,
    stride: int = 8,
) -> float:
    """
    Compute FVD between real and generated videos using the StyleGAN-V approach.

    Instead of extracting one feature vector per video, this uses a sliding
    window of `window_size` frames (stride `stride`) to extract multiple I3D
    feature vectors per video. All clip-level features across all videos are
    pooled together before computing the Fréchet distance. This handles
    variable-length videos naturally and gives a more robust estimate.

    Reference: Skorokhodov et al., "StyleGAN-V: A Continuous Video Generator
    with Arbitrary Video Length", CVPR 2022.

    Args:
        real_videos: (N, 3, T, H, W) or (N, T, 3, H, W) tensor in [0, 1]
        fake_videos: (M, 3, T, H, W) or (M, T, 3, H, W) tensor in [0, 1]
        batch_size: Batch size for I3D feature extraction
        device: Device to run on
        window_size: Number of frames per sliding-window clip (default 16)
        stride: Step between consecutive windows (default 8)

    Returns:
        fvd: FVD score (lower is better)
    """
    # Handle different input formats
    if real_videos.ndim == 5 and real_videos.shape[2] == 3:
        # (B, T, 3, H, W) -> (B, 3, T, H, W)
        real_videos = real_videos.permute(0, 2, 1, 3, 4)
    if fake_videos.ndim == 5 and fake_videos.shape[2] == 3:
        fake_videos = fake_videos.permute(0, 2, 1, 3, 4)

    # Lazily initialize the shared feature extractor
    global _i3d_features
    if _i3d_features is None:
        _i3d_features = I3DFeatures()
    _i3d_features.to(device)
    _i3d_features.eval()
    feature_extractor = _i3d_features

    # Extract sliding-window clips from all videos
    real_clips = _sliding_window_clips(real_videos, window_size, stride)
    fake_clips = _sliding_window_clips(fake_videos, window_size, stride)
    print(f"FVD: {len(real_clips)} real clips, {len(fake_clips)} fake clips "
          f"(window={window_size}, stride={stride})")

    def extract_features(clips):
        """Extract I3D features from clips in batches."""
        features = []
        num_batches = (len(clips) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch = clips[i * batch_size:(i + 1) * batch_size].to(device)
            feat = feature_extractor(batch)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)

    # Extract features from all clips
    print("Extracting features from real video clips...")
    real_features = extract_features(real_clips)

    print("Extracting features from fake video clips...")
    fake_features = extract_features(fake_clips)

    # Compute statistics over the pooled clip-level features
    mu_real, sigma_real = compute_statistics(real_features)
    mu_fake, sigma_fake = compute_statistics(fake_features)

    # Compute FVD (same Fréchet distance formula as FID)
    fvd = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    feature_extractor.cpu()  # free up memory
    torch.cuda.empty_cache()

    return fvd


@torch.no_grad()
def compute_lpips(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    batch_size=256,
    device='cuda'
) -> float:
    """
    Computes LPIPS safely in chunks to avoid OOM.
    Works for both images and frames of videos.
    
    Args:
        real, fake: (N, C, H, W) or (B, T, C, H, W)
    """
    assert real_images.shape == fake_images.shape, f"Real images shape: {real_images.shape}, Fake images shape: {fake_images.shape}"

    lpips_loss_fn.to(device)
    
    # Flatten video format → (B*T, C, H, W)
    if real_images.ndim == 5:
        B, T, C, H, W = real_images.shape
        real_images = real_images.reshape(B*T, C, H, W)
        fake_images = fake_images.reshape(B*T, C, H, W)

    scores = []

    for i in range(0, len(real_images), batch_size):
        r = real_images[i:i+batch_size].to(device)
        f = fake_images[i:i+batch_size].to(device)

        r = r * 2 - 1  # scale to [-1, 1]
        f = f * 2 - 1

        loss = lpips_loss_fn(r, f)  # (N, 1, 1, 1)
        scores.append(loss.mean().cpu())

    lpips_loss_fn.cpu()  # free up memory
    torch.cuda.empty_cache()  # free up GPU memory

    return torch.stack(scores).mean().item()


# =============================================================================
# Utility Functions
# =============================================================================

def compute_fid_from_datasets(
    real_dataloader,
    fake_dataloader,
    max_samples: Optional[int] = None,
    device: str = 'cuda'
) -> float:
    """
    Compute FID from two dataloaders.
    
    Args:
        real_dataloader: DataLoader yielding real images
        fake_dataloader: DataLoader yielding fake images
        max_samples: Maximum number of samples to use (None = all)
        device: Device to run on
    
    Returns:
        fid: FID score
    """
    def collect_images(dataloader, max_samples):
        images = []
        total = 0
        
        for batch in dataloader:
            if isinstance(batch, dict):
                batch = batch['image']  # Handle dict outputs
            
            images.append(batch)
            total += len(batch)
            
            if max_samples and total >= max_samples:
                break
        
        images = torch.cat(images, dim=0)
        if max_samples:
            images = images[:max_samples]
        
        return images
    
    print("Collecting real images...")
    real_images = collect_images(real_dataloader, max_samples)
    
    print("Collecting fake images...")
    fake_images = collect_images(fake_dataloader, max_samples)
    
    return compute_fid(real_images, fake_images, device=device)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("FID and FVD Metrics - Example Usage")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")
    
    # =========================================================================
    # Example 1: Compute FID for images
    # =========================================================================
    print("Example 1: Computing FID for images")
    print("-" * 60)
    
    # Generate dummy data
    num_real = 100
    num_fake = 100
    
    real_images = torch.rand(num_real, 3, 256, 256)  # Random real images
    fake_images = torch.rand(num_fake, 3, 256, 256)  # Random fake images
    
    fid_score = compute_fid(real_images, fake_images, batch_size=10, device=device)
    print(f"FID Score: {fid_score:.2f}")
    print("(Lower is better. Random images should give FID ~200-300)\n")
    
    # =========================================================================
    # Example 2: Compute FVD for videos
    # =========================================================================
    print("Example 2: Computing FVD for videos")
    print("-" * 60)
    
    # Generate dummy video data
    num_videos = 50
    num_frames = 16
    
    real_videos = torch.rand(num_videos, 3, num_frames, 64, 64)
    fake_videos = torch.rand(num_videos, 3, num_frames, 64, 64)
    
    fvd_score = compute_fvd(real_videos, fake_videos, batch_size=8, device=device)
    print(f"FVD Score: {fvd_score:.2f}")
    print("(Lower is better. Random videos should give high FVD)\n")
    
    # =========================================================================
    # Example 3: More realistic - similar distributions
    # =========================================================================
    print("Example 3: FID with similar distributions")
    print("-" * 60)
    
    # Create fake images that are close to real images (add noise)
    real_images = torch.rand(100, 3, 256, 256)
    fake_images = real_images + torch.randn_like(real_images) * 0.1
    fake_images = fake_images.clamp(0, 1)
    
    fid_score = compute_fid(real_images, fake_images, batch_size=10, device=device)
    print(f"FID Score: {fid_score:.2f}")
    print("(Should be much lower since distributions are similar)\n")
    
    print("="*60)
    print("✓ Examples complete!")
    print("="*60)
