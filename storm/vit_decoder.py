import torch
import torch.nn as nn
from torch.nn import Identity
import torch.nn.functional as F
from prismatic.extern.hf.modeling_prismatic import PrismaticProjector

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(approximate='none')
        self.drop1 = nn.Dropout(drop)
        self.norm = Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = Identity()
        self.k_norm = Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                            attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = Identity()
        self.drop_path1 = Identity()
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.ls2 = Identity()
        self.drop_path2 = Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class PatchUnembed(nn.Module):
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=1152):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.ConvTranspose2d(
            embed_dim, in_chans, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        self.norm = Identity()

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.num_patches, "Input sequence length doesn't match expected patches"
        
        # (B, N, C) -> (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, 
                                     self.img_size // self.patch_size, 
                                     self.img_size // self.patch_size)
        x = self.proj(x)
        x = self.norm(x)
        return x

class ViTDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=14, in_chans=6, 
                 embed_dim=1152, depth=27, num_heads=16, 
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.decode_projector = PrismaticProjector(True, 4096, embed_dim) ####
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            DecoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.patch_unembed = PatchUnembed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )

    def forward(self, x):
        # Input shape: (B*L, 256, 4096)
        B, L, N, D = x.shape
        x = x.reshape(B*L, N, D)
        patch_features = torch.split(x, [256] * 2, dim=1)
        all_images = []
        for x in patch_features:
            x = self.decode_projector(x)
            x = self.blocks(x)
            x = self.norm(x)
            x = self.patch_unembed(x)  # (B, 6, 224, 224)
            all_images.append(x)
        all_images = torch.cat(all_images, dim=1) # (B*L, 12, 224, 224)
        _, C, H, W = all_images.shape
        all_images = all_images.reshape(B, L, C, H, W)
        return all_images