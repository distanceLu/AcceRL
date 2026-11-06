import torch
import torch.nn as nn
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder, AutoencoderKL
from typing import Tuple
from transformers import CLIPTextModel, CLIPTokenizer


class ClipEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", device='cuda'):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
        # Freeze CLIP weights.
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    @property
    def feature_dim(self):
        return 768
    
    def forward(self, text_list):
        # text_list is a list of strings: ["pick up the bowl", "open the door"]
        device = next(self.text_encoder.parameters()).device
        
        inputs = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        ).to(device)

        # We use .pooler_output for the sentence-level embedding
        outputs = self.text_encoder(**inputs)
        sentence_embedding = outputs.pooler_output # Shape: [Batch, 768]
        
        return sentence_embedding


def get_task_encoder(
    config: dict,
    device: str = "cpu",
) -> nn.Module:

    print(config)
    encoder = ClipEncoder(device=device)
    return encoder


class SVDEncoder(nn.Module):
    """
    Wrapper around Stable Diffusion's VAE encoder.
    Produces latents for images (B, C, H, W) or videos (B, T, C, H, W).
    """
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2-1",
        image_size: Tuple[int, int] = (256, 256),
        spatial_size: int = 4,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device
        self.latent_h = (image_size[0] // 8, image_size[1] // 8)

        self.spatial_size = spatial_size
        # Load pretrained SD VAE encoder
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            model_name, subfolder="vae"
        ).to(device)

        # Extract scaling factor from model config
        self.scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)

        # Freeze parameters
        for p in self.vae.parameters():
            p.requires_grad = False

        # Normalization buffers (SD expects [-1,1])
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1))

    @property
    def feature_dim(self):
        return 4 * self.spatial_size**2

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes:
            (B, C, H, W)  →  (B, 4, H/8, W/8)
            (B, T, C, H, W) → (B, T, 4, H/8, W/8)
        """
        is_video = (x.ndim == 5)

        if is_video:
            B, T = x.shape[:2]
            x = rearrange(x, "b t c h w -> (b t) c h w")

        # Normalize to [-1,1]
        x = self.normalize(x)

        # Encode
        # TODO: With compile can lead to failues, check this later... 
        # latents = self.vae.encode(x).latent_dist.sample()
        latents = self.vae.encode(x).latent_dist.mode()
        latents = latents * self.scaling_factor
        
        latents = rearrange(latents,
                            "b d (h h1) (w w1) -> b (d h1 w1) h w",
                            h1=self.spatial_size, w1=self.spatial_size)

        latents = rearrange(latents, "b d h w -> b (h w) d")        

        if is_video:
            latents = rearrange(latents, "(b t) n d -> b t n d", b=B)

        return latents

    # @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decodes:
            (B, C, H/8, W/8) → (B, 3, H, W)
            (B, T, C, H/8, W/8) → (B, T, 3, H, W)
        """
        is_video = (latents.ndim == 4)

        if is_video:
            B, T = latents.shape[:2]
            latents = rearrange(latents, "b t n d -> (b t) n d")
        
        l_dim = (self.latent_h[0] // self.spatial_size,
                 self.latent_h[1] // self.spatial_size)

        latents = rearrange(latents, "b (h w) d -> b d h w", h=l_dim[0], w=l_dim[1])
        latents = rearrange(latents,
                            "b (d h1 w1) h w -> b d (h h1) (w w1)",
                            h1=self.spatial_size, w1=self.spatial_size)

        # Stable Diffusion decoder expects unscaled latents
        latents = latents / self.scaling_factor

        imgs = self.vae.decode(latents, num_frames=1).sample

        # Convert from [-1,1] → [0,1]
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)

        if is_video:
            imgs = rearrange(imgs, "(b t) c h w -> b t c h w", b=B)

        return imgs


class SD3Encoder(nn.Module):
    """
    Wrapper around Stable Diffusion 3's VAE encoder (AutoencoderKL).
    Supports both scaling factor and shift factor for latent normalization.
    Produces latents for images (B, C, H, W) or videos (B, T, C, H, W).
    """
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        image_size: Tuple[int, int] = (256, 256),
        spatial_size: int = 4,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device
        self.latent_h = (image_size[0] // 8, image_size[1] // 8)
        self.spatial_size = spatial_size

        # Load pretrained SD3 VAE encoder
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae"
        ).to(device)

        # Extract scaling factor and shift factor from model config
        self.scaling_factor = getattr(self.vae.config, 'scaling_factor', 1.5305)
        self.shift_factor = getattr(self.vae.config, 'shift_factor', 0.0609)

        # Freeze parameters
        for p in self.vae.parameters():
            p.requires_grad = False

        # Normalization buffers (SD expects [-1,1])
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1))

    @property
    def feature_dim(self):
        return self.vae.config.latent_channels * self.spatial_size**2

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def scale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply SD3 latent scaling: (latents - shift) * scaling"""
        return (latents - self.shift_factor) * self.scaling_factor

    def unscale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Reverse SD3 latent scaling: latents / scaling + shift"""
        return latents / self.scaling_factor + self.shift_factor

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes:
            (B, C, H, W)  →  (B, N, D) where N = (H/8/spatial_size) * (W/8/spatial_size), D = latent_channels * spatial_size^2
            (B, T, C, H, W) → (B, T, N, D)
        """
        is_video = (x.ndim == 5)

        if is_video:
            B, T = x.shape[:2]
            x = rearrange(x, "b t c h w -> (b t) c h w")

        # Normalize to [-1,1]
        x = self.normalize(x)

        # Encode
        latents = self.vae.encode(x).latent_dist.mode()

        # Apply SD3 scaling with shift
        latents = self.scale_latents(latents)

        # Rearrange to patch tokens
        latents = rearrange(latents,
                            "b d (h h1) (w w1) -> b (d h1 w1) h w",
                            h1=self.spatial_size, w1=self.spatial_size)

        latents = rearrange(latents, "b d h w -> b (h w) d")

        if is_video:
            latents = rearrange(latents, "(b t) n d -> b t n d", b=B)

        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decodes:
            (B, N, D) → (B, 3, H, W)
            (B, T, N, D) → (B, T, 3, H, W)
        """
        is_video = (latents.ndim == 4)

        if is_video:
            B, T = latents.shape[:2]
            latents = rearrange(latents, "b t n d -> (b t) n d")

        l_dim = (self.latent_h[0] // self.spatial_size,
                 self.latent_h[1] // self.spatial_size)

        latents = rearrange(latents, "b (h w) d -> b d h w", h=l_dim[0], w=l_dim[1])
        latents = rearrange(latents,
                            "b (d h1 w1) h w -> b d (h h1) (w w1)",
                            h1=self.spatial_size, w1=self.spatial_size)

        # Reverse SD3 scaling
        latents = self.unscale_latents(latents)

        imgs = self.vae.decode(latents).sample

        # Convert from [-1,1] → [0,1]
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)

        if is_video:
            imgs = rearrange(imgs, "(b t) c h w -> b t c h w", b=B)

        return imgs


def get_encoder(
    config: dict,
    image_size: int = 256,
    device: str = "cpu",
) -> nn.Module:

    print(config)

    # Handle both dict and object configs
    if isinstance(config, dict):
        config_name = config.get('name', config.get('encoder_name', ''))
        config_freeze = config.get('freeze', True)
        config_spatial_size = config.get('spatial_size', 4)
    else:
        config_name = config.name
        config_freeze = config.freeze
        config_spatial_size = config.spatial_size

    if config_name.startswith("stabilityai"):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        # Use SD3Encoder for SD3 models, SVDEncoder for others
        if "stable-diffusion-3" in config_name:
            encoder = SD3Encoder(
                model_name=config_name,
                image_size=image_size,
                spatial_size=config_spatial_size,
                device=device
            )
        else:
            encoder = SVDEncoder(
                model_name=config_name,
                image_size=image_size,
                spatial_size=config_spatial_size,
                device=device
            )
        train_decoder = False
    else:
        raise ValueError(f"Unknown encoder type: {config_name}")

    if config_freeze:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder, train_decoder
