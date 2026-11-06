import torch
import torch.nn as nn

from einops import rearrange


class ImgDecoder(nn.Module):
    def __init__(
        self,
        d_embed: int = 384,
        out_ch: int = 3,
        base_ch: int = 512,
        use_sigmoid: bool = True
    ):
        super().__init__()

        # Upsampling sequence: 16x16 → 32x32 → 64x64 → 128x128 → 256x256
        norm = lambda c: nn.GroupNorm(num_groups=8, num_channels=c)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_embed, base_ch, 4, 2, 1),   # up 16→32
            norm(base_ch),
            nn.SiLU(),

            nn.ConvTranspose2d(base_ch, base_ch//2, 4, 2, 1),  # 32→64
            norm(base_ch//2),
            nn.SiLU(),

            nn.ConvTranspose2d(base_ch//2, base_ch//4, 4, 2, 1),  # 64→128
            norm(base_ch//4),
            nn.SiLU(),

            nn.ConvTranspose2d(base_ch//4, base_ch//8, 4, 2, 1),  # 128→256
            norm(base_ch//8),
            nn.SiLU(),

            nn.Conv2d(base_ch//8, out_ch, 3, 1, 1),
        )

        self.output_act = nn.Sigmoid() if use_sigmoid else nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, N, D] where N = 16*16 = 256
        returns: [B, 3, 256, 256]
        """
        B, N, D = z.size()
        h = w = int(N ** 0.5)
        # z ÷= z.permute(0, 2, 1).reshape(B, D, h, w)÷
        z = rearrange(z, 'b n d -> b d h w', h=h, w=w)
        out = self.decoder(z)
        return self.output_act(out)
