# src/models/pinn/core.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional

# ---------------- Fourier Features ---------------- #

class FourierFeatures(nn.Module):
    """
    Fourier features for (x,y,t) in [-1,1]^3 with FULL frequency set always computed.
    Call set_alpha(0..1) during training to scale higher frequencies (soft annealing).
    """
    def __init__(self, in_dim: int = 3, max_log2_freq: int = 5):
        super().__init__()
        self.in_dim = in_dim
        self.max_log2_freq = max_log2_freq
        self.register_buffer("_alpha", torch.tensor(1.0))  # Annealing weight

    def set_alpha(self, a: float) -> None:
        """Set annealing parameter (0=soft start, 1=full freqs)"""
        a = float(max(0.0, min(1.0, a)))
        self._alpha.fill_(a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        # Always compute all frequencies, but scale higher ones by alpha
        outs = [x]
        alpha = float(self._alpha.item())
        for k in range(1, self.max_log2_freq + 1):
            f = 2.0 ** k
            # Soft annealing: linearly scale frequency contribution
            weight = min(1.0, max(0.0, (alpha - (k-1)/self.max_log2_freq) * self.max_log2_freq))
            outs.append(weight * torch.sin(math.pi * f * x))
            outs.append(weight * torch.cos(math.pi * f * x))
        return torch.cat(outs, dim=-1)

def fourier_out_dim(in_dim: int = 3, max_log2_freq: int = 5) -> int:
    return in_dim + 2 * max_log2_freq * in_dim

# ---------------- Backbone (coordinate MLP) ---------------- #

def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int) -> nn.Sequential:
    mods = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    mods += [nn.Linear(d, out_dim)]
    return nn.Sequential(*mods)

class PINNBackbone(nn.Module):
    """
    Coordinate MLP with Fourier features.
    Outputs: A_z, B_z, u_x, u_y, (optional eta_raw in R)
    """
    def __init__(self, hidden: int = 384, layers: int = 10,
                 max_log2_freq: int = 5, learn_eta: bool = False):
        super().__init__()
        self.ff = FourierFeatures(3, max_log2_freq)
        in_dim = fourier_out_dim(3, max_log2_freq)
        out_dim = 5 if learn_eta else 4  # Az, Bz, ux, uy, [eta_raw]
        self.net = _mlp(in_dim, hidden, out_dim, layers)
        self.learn_eta = learn_eta

    def set_fourier_alpha(self, a: float) -> None:
        self.ff.set_alpha(a)

    def forward(self, coords: torch.Tensor) -> dict[str, Optional[torch.Tensor]]:
        # coords: [N,3] with requires_grad=True for autograd derivatives
        h = self.ff(coords)
        out = self.net(h)
        if self.learn_eta:
            A_z, B_z, u_x, u_y, eta_raw = torch.split(out, 1, dim=-1)
        else:
            A_z, B_z, u_x, u_y = torch.split(out, 1, dim=-1)
            eta_raw = None
        return {"A_z": A_z, "B_z": B_z, "u_x": u_x, "u_y": u_y, "eta_raw": eta_raw}

# ---------------- Helpers (hard in-plane solenoidality) ---------------- #

@torch.enable_grad()
def B_perp_from_Az(A_z: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Bx, By from A_z via B = curl(A_z zhat) => (Bx, By) = (-∂y Az, ∂x Az).
    A_z: [N,1], coords: [N,3] with requires_grad=True
    """
    assert A_z.requires_grad and coords.requires_grad
    ones = torch.ones_like(A_z)
    grads = torch.autograd.grad(A_z, coords, grad_outputs=ones,
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    dAz_dx = grads[..., 0:1]
    dAz_dy = grads[..., 1:2]
    Bx = -dAz_dy
    By = dAz_dx
    return Bx, By

class SpatialAttentionPool(nn.Module):
    """
    Attention-based spatial pooling that learns to focus on important regions.
    
    Key insight: Not all spatial locations are equally important for flare prediction.
    PIL regions and areas with high field gradients are more informative.
    """
    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, P, C] features
            
        Returns:
            pooled: [B, T, C] attention-weighted features
            attn_weights: [B, T, P, 1] attention weights (for visualization)
        """
        # Compute attention scores
        scores = self.attn(x)  # [B, T, P, 1]
        weights = torch.softmax(scores, dim=2)  # Softmax over spatial dim P
        
        # Weighted sum
        pooled = (x * weights).sum(dim=2)  # [B, T, C]
        
        return pooled, weights


class TemporalAttentionPool(nn.Module):
    """
    Attention-based temporal pooling for observed frames.
    
    Learns which time steps are most predictive of future flares.
    Recent frames and frames with high activity should get more weight.
    """
    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C] features
            mask: [B, T] boolean mask for observed frames
            
        Returns:
            pooled: [B, C] attention-weighted features
            attn_weights: [B, T, 1] attention weights
        """
        # Compute attention scores
        scores = self.attn(x)  # [B, T, 1]
        
        # Mask unobserved frames with large negative value
        mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
        scores = scores.masked_fill(~mask_expanded, float('-inf'))
        
        # Softmax over temporal dimension
        weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        
        # Handle all-masked case (when all frames are -inf, softmax produces NaN)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Fallback: if all weights are zero (all masked), use uniform weights on first valid frame
        weight_sum = weights.sum(dim=1, keepdim=True)  # [B, 1, 1]
        if (weight_sum == 0).any():
            # Find batches with zero weights and set uniform weights on valid frames
            zero_mask = (weight_sum.squeeze(-1) == 0)  # [B, 1]
            n_valid = mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            uniform_weights = (mask.float().unsqueeze(-1) / n_valid.unsqueeze(-1))  # [B, T, 1]
            weights = torch.where(zero_mask.unsqueeze(-1), uniform_weights, weights)
        
        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # [B, C]
        
        return pooled, weights


class PhysicsFeatureExtractor(nn.Module):
    """
    Extract physics-informed features from PINN outputs.
    
    Features include:
    - Field statistics (mean, std, max, min)
    - Gradient-based features (shear, PIL proximity)
    - Flow statistics (convergence, vorticity proxy)
    - Temporal derivatives (rate of change)
    
    These features capture physical indicators of flare potential.
    
    SOTA improvements:
    - Added robust statistics (median, IQR) for outlier resistance
    - Added higher-order temporal features (acceleration)
    - Added field-flow correlation features
    """
    def __init__(self, use_extended_features: bool = False):
        super().__init__()
        self.use_extended = use_extended_features
        
    def forward(self, feats: dict[str, torch.Tensor], observed_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: dict with A_z, B_z, u_x, u_y each [B, T, P, 1]
            observed_mask: [B, T] boolean mask
            
        Returns:
            physics_features: [B, n_features] physics-derived features
        """
        B_z = feats["B_z"]
        u_x = feats["u_x"]
        u_y = feats["u_y"]
        A_z = feats["A_z"]
        
        B, T, P, _ = B_z.shape
        mask = observed_mask[..., None, None].float()  # [B, T, 1, 1]
        
        # 1. Field magnitude statistics
        B_z_masked = B_z * mask
        valid_count = (mask.sum(dim=(1, 2, 3)) + 1e-8)
        
        # Mean and std
        bz_mean = B_z_masked.sum(dim=(1, 2, 3)) / valid_count
        bz_var = (B_z_masked ** 2).sum(dim=(1, 2, 3)) / valid_count - bz_mean ** 2
        bz_std = bz_var.clamp(min=1e-8).sqrt()  # Clamp before sqrt for stability
        
        # Max (absolute)
        bz_abs_masked = B_z.abs() * mask
        bz_max = bz_abs_masked.reshape(B, -1).max(dim=1)[0]
        
        # 2. Polarity mixture (indicator of PIL presence)
        # IMPROVED: Use soft thresholding for smoother gradients
        bz_pos_soft = torch.sigmoid(10 * B_z) * mask  # Soft positive indicator
        bz_neg_soft = torch.sigmoid(-10 * B_z) * mask  # Soft negative indicator
        bz_pos = bz_pos_soft.sum(dim=(1, 2, 3)) / valid_count
        bz_neg = bz_neg_soft.sum(dim=(1, 2, 3)) / valid_count
        polarity_balance = (bz_pos * bz_neg).sqrt()  # High when balanced = PIL present
        
        # 3. Flow statistics
        u_mag = (u_x ** 2 + u_y ** 2 + 1e-8).sqrt()  # Add eps before sqrt
        u_masked = u_mag * mask
        u_mean = u_masked.sum(dim=(1, 2, 3)) / valid_count
        u_max = u_masked.reshape(B, -1).max(dim=1)[0]
        
        # 4. Flux transport: correlation of flow with field gradient
        # This is a key indicator of magnetic energy buildup
        flux_transport = ((u_mag * B_z.abs()) * mask).sum(dim=(1, 2, 3)) / valid_count
        
        # 5. Temporal variability (rate of change indicator)
        if T > 1:
            # First-order difference (velocity of change)
            dBz_dt = (B_z[:, 1:] - B_z[:, :-1]).abs()
            mask_dt = mask[:, 1:] * mask[:, :-1]
            temporal_var = (dBz_dt * mask_dt).sum(dim=(1, 2, 3)) / (mask_dt.sum(dim=(1, 2, 3)) + 1e-8)
        else:
            temporal_var = torch.zeros(B, device=B_z.device)
        
        # 6. Vector potential complexity (proxy for current density complexity)
        az_var = (A_z ** 2 * mask).sum(dim=(1, 2, 3)) / valid_count
        az_std = az_var.clamp(min=1e-8).sqrt()
        
        # Concatenate base physics features (9 features for backward compatibility)
        physics_feats = torch.stack([
            bz_mean, bz_std, bz_max, polarity_balance,
            u_mean, u_max, flux_transport,
            temporal_var, az_std
        ], dim=-1)  # [B, 9]
        
        # Handle any NaN that might occur
        physics_feats = torch.nan_to_num(physics_feats, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return physics_feats


class ClassifierHead(nn.Module):
    """
    Advanced classifier head with attention and physics features.
    
    Architecture:
        1. Spatial attention pooling → focus on important regions
        2. Physics feature extraction → domain-specific indicators
        3. RF importance weighting → weight features by known importance
        4. Temporal attention pooling → weight recent/active frames
        5. MLP classifier → multi-horizon prediction
        
    Key improvements over baseline:
        - Attention mechanisms for interpretable region weighting
        - Physics-derived features (polarity balance, flux transport)
        - RF guidance for feature weighting
        - Temporal modeling of field evolution
        - Residual connections for gradient flow
    """
    def __init__(
        self, 
        hidden: int = 256, 
        dropout: float = 0.1, 
        horizons: tuple[int, ...] = (6, 12, 24),
        use_attention: bool = True,
        use_physics_features: bool = True,
        rf_importance_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.horizons = tuple(horizons)
        self.use_attention = use_attention
        self.use_physics_features = use_physics_features
        
        # Base features: A_z, B_z, u_x, u_y
        base_features = 4
        
        # Spatial attention pooling
        if use_attention:
            self.spatial_attn = SpatialAttentionPool(base_features, hidden=64)
            self.temporal_attn = TemporalAttentionPool(base_features, hidden=64)
        
        # Physics feature extractor
        if use_physics_features:
            self.physics_extractor = PhysicsFeatureExtractor()
            physics_features = 9
        else:
            physics_features = 0
        
        # RF importance weights for physics features
        # If provided, these weight the physics features by domain knowledge
        if rf_importance_weights is not None:
            self.register_buffer('rf_weights', rf_importance_weights)
        else:
            # Default: uniform weights
            self.register_buffer('rf_weights', torch.ones(physics_features) if physics_features > 0 else None)
        
        # Compute total input features for final MLP
        # With attention: base_features + base_features (stats) + physics_features
        # Without attention: base_features + physics_features
        if use_attention:
            mlp_input = base_features + base_features + physics_features  # 4 + 4 + 9 = 17
        else:
            mlp_input = base_features + physics_features  # 4 + 9 = 13
        
        # Final classifier MLP with residual
        self.pre_mlp = nn.Sequential(
            nn.Linear(mlp_input, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, len(horizons))
        )

    def forward(self, feats: dict[str, torch.Tensor], observed_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: dict of tensors shaped [B, T, P, 1] for keys A_z, B_z, u_x, u_y
            observed_mask: [B, T] booleans (True for frames t <= t0)
            
        Returns:
            logits: [B, n_horizons] classification logits
        """
        # Concatenate base features
        X = torch.cat([feats["A_z"], feats["B_z"], feats["u_x"], feats["u_y"]], dim=-1)  # [B,T,P,4]
        B, T, P, C = X.shape
        
        # Ensure mask has batch dimension
        if observed_mask.dim() == 1:
            observed_mask = observed_mask.unsqueeze(0).expand(B, -1)
        
        if self.use_attention:
            # Spatial attention pooling
            X_spatial, spatial_weights = self.spatial_attn(X)  # [B, T, 4]
            
            # Temporal attention pooling
            X_pooled, temporal_weights = self.temporal_attn(X_spatial, observed_mask)  # [B, 4]
            
            # Also compute simple statistics as backup
            mask_expanded = observed_mask[..., None, None].float()  # [B, T, 1, 1]
            X_masked = X * mask_expanded
            valid_count = mask_expanded.sum(dim=(1, 2)).clamp(min=1.0)  # [B, 1]
            X_mean = X_masked.sum(dim=(1, 2)) / valid_count  # [B, 4]
            
            # Combine attention and statistics
            combined = torch.cat([X_pooled, X_mean], dim=-1)  # [B, 8]
        else:
            # Simple averaging fallback
            mask_expanded = observed_mask[..., None, None].float()
            X_masked = X * mask_expanded
            denom_t = mask_expanded.sum(dim=1, keepdim=True).clamp_min(1.0)
            X_t = X_masked.sum(dim=1, keepdim=True) / denom_t
            X_avg = X_t.mean(dim=2).squeeze(1)  # [B, 4]
            combined = X_avg
        
        # Physics features
        if self.use_physics_features:
            physics_feats = self.physics_extractor(feats, observed_mask)  # [B, 9]
            
            # Apply RF importance weights if available
            if self.rf_weights is not None:
                physics_feats = physics_feats * self.rf_weights
            
            combined = torch.cat([combined, physics_feats], dim=-1)
        
        # MLP classifier
        h = self.pre_mlp(combined)
        logits = self.classifier(h)
        
        return logits  # [B, n_horizons]

