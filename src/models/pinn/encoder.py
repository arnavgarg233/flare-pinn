# src/models/pinn/encoder.py
"""
Tiny CNN encoder for conditioning the PINN coordinate MLP.
Extracts spatial features from observed magnetogram frames.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGN(nn.Module):
    """Conv + GroupNorm + activation block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class TinyEncoder(nn.Module):
    """
    Lightweight CNN encoder for spatial feature extraction.
    
    Takes observed Bz frames (t <= t0) and produces:
    - L: Latent feature map [N, C, H, W] for per-point conditioning
    - g: Global code [N, D] for FiLM conditioning
    
    Architecture: 2-3 conv stages with minimal capacity to avoid overfitting.
    """
    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 32,
        global_dim: int = 64,
        dropout: float = 0.05
    ):
        super().__init__()
        self.stem = ConvGN(in_channels, 32)
        self.block1 = ConvGN(32, 32)
        self.dropout1 = nn.Dropout2d(dropout)
        
        self.down = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.block2 = ConvGN(64, 64)
        self.dropout2 = nn.Dropout2d(dropout)
        
        # Latent map projection
        self.proj_L = nn.Conv2d(64, latent_channels, 1)
        
        # Global code via spatial pooling + MLP
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.g_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_channels, global_dim),
            nn.SiLU(),
            nn.Linear(global_dim, global_dim)
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, C_in, H, W] - Stacked observed Bz frames (t <= t0 only!)
        
        Returns:
            L: [N, C_latent, H, W] - Spatial feature map
            g: [N, D] - Global context code
        """
        # Encoder path
        h = self.stem(x)
        h = self.dropout1(self.block1(h))
        
        # Downsample + process
        h_down = self.down(h)
        h_down = self.dropout2(self.block2(h_down))
        
        # Upsample back to original resolution for L
        L = self.proj_L(h_down)
        L = F.interpolate(L, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Global code from L
        g = self.g_mlp(self.global_pool(L))
        
        return L, g


# OLD FUNCTION REMOVED - replaced by latent_sampling.py
# Use sample_latent_soft_bilinear() or sample_latent_nearest() instead
# These support 2nd-order gradients for physics loss

