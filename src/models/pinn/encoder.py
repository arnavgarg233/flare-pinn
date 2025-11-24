# src/models/pinn/encoder.py
"""
CNN + Temporal Encoder for conditioning the PINN coordinate MLP.

Key improvements for SOTA:
1. Proper temporal processing (not just averaging frames)
2. Learned temporal position encoding
3. Multi-scale spatial features
4. Temporal attention for weighting important frames
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ConvGN(nn.Module):
    """Conv + GroupNorm + activation block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ConvGN(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.dropout = nn.Dropout2d(dropout)
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return self.act(h + x)


class TemporalPositionEncoding(nn.Module):
    """
    Learnable temporal position encoding for frame sequences.
    
    Maps relative time position to embedding that captures:
    - Recency (recent frames are more predictive)
    - Periodicity (solar rotation ~27 days)
    - Evolution rate context
    """
    def __init__(self, d_model: int = 32, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        
        # Sinusoidal base encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Learnable refinement
        self.refine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: Length of sequence
            
        Returns:
            encoding: [seq_len, d_model] position encoding
        """
        base = self.pe[:seq_len]
        return base + self.refine(base)


class TemporalAttention(nn.Module):
    """
    Attention mechanism for weighting temporal frames.
    
    Key insight: Not all frames are equally predictive.
    - Recent frames capture current state
    - Frames with rapid changes indicate evolution
    - Frames near observation gaps may be less reliable
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable "query" for extracting summary
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D] frame features
            mask: [B, T] boolean mask (True = valid frame)
            
        Returns:
            summary: [B, D] aggregated representation
            attn_weights: [B, T] attention weights
        """
        B, T, D = x.shape
        
        # Add CLS token for aggregation
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x_with_cls = torch.cat([cls, x], dim=1)  # [B, T+1, D]
        
        # Project queries, keys, values
        Q = self.q_proj(x_with_cls[:, :1])  # [B, 1, D] - only query from CLS
        K = self.k_proj(x_with_cls[:, 1:])  # [B, T, D] - keys from frames
        V = self.v_proj(x_with_cls[:, 1:])  # [B, T, D] - values from frames
        
        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, 1, T]
        
        # Mask invalid frames
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        attn = attn.masked_fill(~mask_expanded, float('-inf'))
        
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Handle all-masked case
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Apply attention
        out = torch.matmul(attn_weights, V)  # [B, H, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        out = self.out_proj(out).squeeze(1)  # [B, D]
        
        # Return mean attention weights across heads
        avg_attn = attn_weights.mean(dim=1).squeeze(1)  # [B, T]
        
        return out, avg_attn


class TinyEncoder(nn.Module):
    """
    6-Layer ResNet Encoder for spatial feature extraction.
    
    Takes observed Bz frames (t <= t0) and produces:
    - L: Latent feature map [N, C, H, W] for per-point conditioning
    - g: Global code [N, D] for FiLM conditioning
    
    Architecture: 6 ResBlocks with progressive channel expansion.
    This deeper architecture captures multi-scale spatial features
    critical for PIL detection and flare prediction.
    
    Features:
    - Gradient checkpointing support for memory efficiency
    - Stochastic depth (DropPath) for regularization
    """
    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 32,
        global_dim: int = 64,
        dropout: float = 0.05,
        use_checkpoint: bool = False  # Gradient checkpointing for memory
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Stem: initial projection
        self.stem = ConvGN(in_channels, 32)
        
        # Stage 1: 32 channels, full resolution (2 ResBlocks)
        self.res1 = ResBlock(32, dropout)
        self.res2 = ResBlock(32, dropout)
        
        # Downsample 1: 32 → 64 channels, /2 resolution
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        
        # Stage 2: 64 channels (2 ResBlocks)
        self.res3 = ResBlock(64, dropout)
        self.res4 = ResBlock(64, dropout)
        
        # Downsample 2: 64 → 128 channels, /4 resolution
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        # Stage 3: 128 channels (2 ResBlocks)
        self.res5 = ResBlock(128, dropout)
        self.res6 = ResBlock(128, dropout)
        
        # Latent map projection (upsample back to original resolution)
        self.proj_L = nn.Conv2d(128, latent_channels, 1)
        
        # Global code via spatial pooling + MLP
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.g_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, global_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(global_dim * 2, global_dim)
        )
    
    def _forward_stage1(self, h: torch.Tensor) -> torch.Tensor:
        """Stage 1 forward for checkpointing."""
        h = self.res1(h)
        return self.res2(h)
    
    def _forward_stage2(self, h: torch.Tensor) -> torch.Tensor:
        """Stage 2 forward for checkpointing."""
        h = self.down1(h)
        h = self.res3(h)
        return self.res4(h)
    
    def _forward_stage3(self, h: torch.Tensor) -> torch.Tensor:
        """Stage 3 forward for checkpointing."""
        h = self.down2(h)
        h = self.res5(h)
        return self.res6(h)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, C_in, H, W] - Stacked observed Bz frames (t <= t0 only!)
        
        Returns:
            L: [N, C_latent, H, W] - Spatial feature map
            g: [N, D] - Global context code
        """
        # Stem
        h = self.stem(x)  # [N, 32, H, W]
        
        # Stage 1-3 with optional gradient checkpointing
        if self.use_checkpoint and self.training:
            h = checkpoint(self._forward_stage1, h, use_reentrant=False)
            h = checkpoint(self._forward_stage2, h, use_reentrant=False)
            h = checkpoint(self._forward_stage3, h, use_reentrant=False)
        else:
            h = self._forward_stage1(h)
            h = self._forward_stage2(h)
            h = self._forward_stage3(h)
        
        # Global code from deepest features
        g = self.g_mlp(self.global_pool(h))  # [N, global_dim]
        
        # Latent map: project and upsample to original resolution
        L = self.proj_L(h)  # [N, latent_channels, H/4, W/4]
        L = F.interpolate(L, size=x.shape[-2:], mode='bilinear', align_corners=False)  # [N, latent_channels, H, W]
        
        return L, g


class TemporalEncoder(nn.Module):
    """
    Advanced encoder with proper temporal processing.
    
    Key improvements:
    1. Processes each frame independently through CNN
    2. Adds temporal position encoding
    3. Uses attention to aggregate across time
    4. Produces both spatial (L) and global (g) representations
    
    This is critical for capturing field evolution patterns
    that precede flares.
    """
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 32,
        global_dim: int = 64,
        temporal_dim: int = 32,
        n_attn_heads: int = 4,
        dropout: float = 0.1,
        max_frames: int = 64
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.global_dim = global_dim
        self.temporal_dim = temporal_dim
        
        # Per-frame CNN encoder
        self.frame_encoder = nn.Sequential(
            ConvGN(in_channels, 32),
            ResBlock(32, dropout),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            ConvGN(64, 64),
            ResBlock(64, dropout),
            nn.Conv2d(64, latent_channels, 1)
        )
        
        # Global pooling for frame features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal position encoding
        self.temporal_pe = TemporalPositionEncoding(temporal_dim, max_frames)
        
        # Project frame features to temporal dim
        self.frame_proj = nn.Linear(latent_channels, temporal_dim)
        
        # Temporal attention for aggregation
        self.temporal_attn = TemporalAttention(temporal_dim, n_attn_heads, dropout)
        
        # Final projections
        self.global_proj = nn.Sequential(
            nn.Linear(temporal_dim, global_dim),
            nn.SiLU(),
            nn.Linear(global_dim, global_dim)
        )
        
        # Spatial map aggregation (weighted by attention)
        self.spatial_weight_proj = nn.Linear(temporal_dim, 1)
    
    def forward(
        self, 
        frames: torch.Tensor, 
        observed_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: [T, H, W] frame sequence
            observed_mask: [T] boolean mask for valid frames
            
        Returns:
            L: [1, C, H, W] aggregated spatial features
            g: [1, D] global conditioning code
            temporal_weights: [T] frame importance weights
        """
        T, H, W = frames.shape
        device = frames.device
        
        # Handle empty observation case
        if not observed_mask.any():
            L = torch.zeros(1, self.latent_channels, H, W, device=device)
            g = torch.zeros(1, self.global_dim, device=device)
            weights = torch.zeros(T, device=device)
            return L, g, weights
        
        # Get observed frames only for encoding
        obs_frames = frames[observed_mask]  # [T_obs, H, W]
        T_obs = obs_frames.shape[0]
        
        # Encode each frame
        # Add channel dim: [T_obs, H, W] -> [T_obs, 1, H, W]
        obs_frames_batch = obs_frames.unsqueeze(1)
        
        # CNN features for each frame
        frame_features = []
        for i in range(T_obs):
            feat = self.frame_encoder(obs_frames_batch[i:i+1])  # [1, C, H', W']
            frame_features.append(feat)
        
        # Stack: [T_obs, C, H', W']
        frame_features = torch.cat(frame_features, dim=0)
        _, C, H_feat, W_feat = frame_features.shape
        
        # Upsample to original resolution
        frame_features = F.interpolate(
            frame_features, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )  # [T_obs, C, H, W]
        
        # Global features per frame
        global_feats = self.global_pool(frame_features).view(T_obs, C)  # [T_obs, C]
        
        # Project to temporal dim
        temporal_feats = self.frame_proj(global_feats)  # [T_obs, temporal_dim]
        
        # Add position encoding (based on position in observed sequence)
        pe = self.temporal_pe(T_obs).to(device)  # [T_obs, temporal_dim]
        temporal_feats = temporal_feats + pe
        
        # Add batch dim for attention
        temporal_feats = temporal_feats.unsqueeze(0)  # [1, T_obs, temporal_dim]
        obs_mask = torch.ones(1, T_obs, dtype=torch.bool, device=device)
        
        # Temporal attention
        g_temporal, attn_weights = self.temporal_attn(temporal_feats, obs_mask)  # [1, temporal_dim], [1, T_obs]
        
        # Project to global dim
        g = self.global_proj(g_temporal)  # [1, global_dim]
        
        # Aggregate spatial features using attention weights
        # Weight each frame's spatial features
        attn_weights = attn_weights.squeeze(0)  # [T_obs]
        weighted_L = (frame_features * attn_weights[:, None, None, None]).sum(dim=0, keepdim=True)  # [1, C, H, W]
        
        # IMPROVEMENT: Explicitly concatenate last observed frame features
        # This prevents "history collapse" where the model loses track of the current state
        # due to weighted averaging.
        last_L = frame_features[-1:].clone()  # [1, C, H, W]
        L_out = torch.cat([weighted_L, last_L], dim=1)  # [1, 2*C, H, W]
        
        # Create full temporal weights tensor
        full_weights = torch.zeros(T, device=device)
        full_weights[observed_mask] = attn_weights
        
        return L_out, g, full_weights


# Backward compatibility alias
EncoderWithTemporal = TemporalEncoder


class MultiScaleEncoder(nn.Module):
    """
    Feature Pyramid Network encoder for multi-scale features.
    
    Key insight: PIL detection requires fine-scale gradients while
    field topology requires large-scale context. FPN captures both.
    
    This could be a SIGNIFICANT improvement for SOTA.
    """
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 32,
        global_dim: int = 64,
        dropout: float = 0.05
    ):
        super().__init__()
        
        # Bottom-up pathway (encoder)
        self.stem = ConvGN(in_channels, 32)
        
        # Stage 1: 32 channels
        self.stage1 = nn.Sequential(ResBlock(32, dropout), ResBlock(32, dropout))
        
        # Stage 2: 64 channels, /2
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU()
        )
        self.stage2 = nn.Sequential(ResBlock(64, dropout), ResBlock(64, dropout))
        
        # Stage 3: 128 channels, /4
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU()
        )
        self.stage3 = nn.Sequential(ResBlock(128, dropout), ResBlock(128, dropout))
        
        # Top-down pathway (FPN)
        self.lateral3 = nn.Conv2d(128, latent_channels, 1)
        self.lateral2 = nn.Conv2d(64, latent_channels, 1)
        self.lateral1 = nn.Conv2d(32, latent_channels, 1)
        
        # Smooth convolutions after merging
        self.smooth3 = nn.Conv2d(latent_channels, latent_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(latent_channels, latent_channels, 3, padding=1)
        self.smooth1 = nn.Conv2d(latent_channels, latent_channels, 3, padding=1)
        
        # Global pooling from deepest features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.g_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, global_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(global_dim * 2, global_dim)
        )
        
        # Final fusion: combine all scales
        self.fusion = nn.Conv2d(latent_channels * 3, latent_channels, 1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns multi-scale fused latent map and global code.
        """
        H, W = x.shape[-2:]
        
        # Bottom-up
        c1 = self.stage1(self.stem(x))      # [N, 32, H, W]
        c2 = self.stage2(self.down1(c1))    # [N, 64, H/2, W/2]
        c3 = self.stage3(self.down2(c2))    # [N, 128, H/4, W/4]
        
        # Global code from deepest
        g = self.g_mlp(self.global_pool(c3))
        
        # Top-down with lateral connections
        p3 = self.smooth3(self.lateral3(c3))  # [N, C, H/4, W/4]
        
        p3_up = F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.smooth2(self.lateral2(c2) + p3_up)  # [N, C, H/2, W/2]
        
        p2_up = F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.smooth1(self.lateral1(c1) + p2_up)  # [N, C, H, W]
        
        # Upsample all to full resolution and concatenate
        p2_full = F.interpolate(p2, size=(H, W), mode='bilinear', align_corners=False)
        p3_full = F.interpolate(p3, size=(H, W), mode='bilinear', align_corners=False)
        
        # Fuse multi-scale features
        L = self.fusion(torch.cat([p1, p2_full, p3_full], dim=1))  # [N, C, H, W]
        
        return L, g


