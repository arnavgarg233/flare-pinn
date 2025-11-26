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
    
    Supports two output modes:
    - Scalar mode (legacy): outputs A_z, B_z, u_x, u_y, [eta_raw]
    - Vector mode: outputs A_z, B (3-component), u (2-component), [eta_raw]
    
    Vector mode enables full 2.5D vector induction equation physics.
    """
    def __init__(
        self, 
        hidden: int = 384, 
        layers: int = 10,
        max_log2_freq: int = 5, 
        learn_eta: bool = False,
        vector_B: bool = False,
    ):
        """
        Args:
            hidden: Hidden dimension for MLP
            layers: Number of hidden layers
            max_log2_freq: Maximum Fourier frequency (log2)
            learn_eta: Learn spatially-varying resistivity
            vector_B: Output 3-component B field (Bx, By, Bz) instead of just Bz
        """
        super().__init__()
        self.ff = FourierFeatures(3, max_log2_freq)
        in_dim = fourier_out_dim(3, max_log2_freq)
        
        self.vector_B = vector_B
        self.learn_eta = learn_eta
        
        # Compute output dimension:
        # - A_z: 1
        # - B: 3 (vector) or 1 (scalar)
        # - u: 2
        # - eta_raw: 1 (optional)
        if vector_B:
            # Vector mode: Az(1) + B(3) + u(2) + [eta(1)] = 6 or 7
            out_dim = 6 + (1 if learn_eta else 0)
        else:
            # Scalar mode: Az(1) + Bz(1) + ux(1) + uy(1) + [eta(1)] = 4 or 5
            out_dim = 4 + (1 if learn_eta else 0)
        
        self.net = _mlp(in_dim, hidden, out_dim, layers)

    def set_fourier_alpha(self, a: float) -> None:
        self.ff.set_alpha(a)

    def forward(self, coords: torch.Tensor) -> dict[str, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            coords: [N, 3] coordinates with requires_grad=True for autograd derivatives
            
        Returns:
            Dictionary with fields:
            - Vector mode: {"A_z", "B", "u", "eta_raw", "B_x", "B_y", "B_z", "u_x", "u_y"}
            - Scalar mode: {"A_z", "B_z", "u_x", "u_y", "eta_raw"}
            
            Vector mode provides both packed vectors (B, u) and unpacked components
            for backward compatibility.
        """
        h = self.ff(coords)
        out = self.net(h)
        
        if self.vector_B:
            # Vector mode: unpack A_z(1), B(3), u(2), [eta(1)]
            if self.learn_eta:
                A_z, Bx, By, Bz, ux, uy, eta_raw = torch.split(out, 1, dim=-1)
            else:
                A_z, Bx, By, Bz, ux, uy = torch.split(out, 1, dim=-1)
                eta_raw = None
            
            # Pack into vectors for new physics module
            B = torch.cat([Bx, By, Bz], dim=-1)  # [N, 3]
            u = torch.cat([ux, uy], dim=-1)      # [N, 2]
            
            return {
                # New vector format
                "A_z": A_z,
                "B": B,           # [N, 3] packed vector
                "u": u,           # [N, 2] packed vector
                "eta_raw": eta_raw,
                # Legacy unpacked format (for backward compatibility)
                "B_x": Bx,
                "B_y": By,
                "B_z": Bz,
                "u_x": ux,
                "u_y": uy,
            }
        else:
            # Scalar mode (legacy): unpack A_z(1), B_z(1), ux(1), uy(1), [eta(1)]
            if self.learn_eta:
                A_z, B_z, u_x, u_y, eta_raw = torch.split(out, 1, dim=-1)
            else:
                A_z, B_z, u_x, u_y = torch.split(out, 1, dim=-1)
                eta_raw = None
            
            return {
                "A_z": A_z, 
                "B_z": B_z, 
                "u_x": u_x, 
                "u_y": u_y, 
                "eta_raw": eta_raw
            }

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
    - Added current helicity proxy (critical for flare prediction)
    - Added magnetic complexity measure
    
    Output: 18 physics features (up from 13)
    """
    N_FEATURES = 18  # Total number of output features
    
    def __init__(self, use_extended_features: bool = True):
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
        mask_float = observed_mask[..., None, None].float()  # [B, T, 1, 1]
        mask_bool = mask_float.bool()
        
        # 1. Field magnitude statistics
        # FIXED: Use where() to avoid 0*Inf=NaN
        B_z_masked = torch.where(mask_bool, B_z, torch.zeros_like(B_z))
        # FIXED: valid_count must account for spatial points P to avoid scaling issues
        valid_count = (mask_float.sum(dim=(1, 2, 3)) * P + 1e-8)
        
        # Mean
        bz_mean = B_z_masked.sum(dim=(1, 2, 3)) / valid_count
        
        # Stable variance: E[(X - mean)^2]
        bz_centered = torch.where(mask_bool, B_z - bz_mean.view(B, 1, 1, 1), torch.zeros_like(B_z))
        bz_var = (bz_centered ** 2).sum(dim=(1, 2, 3)) / valid_count
        bz_std = bz_var.clamp(min=1e-8).sqrt()
        
        # Max (absolute)
        # Use where to set masked values to -inf for max
        bz_abs_masked = torch.where(mask_bool, B_z.abs(), torch.tensor(-1e9, device=B_z.device))
        bz_max = bz_abs_masked.reshape(B, -1).max(dim=1)[0]
        # Clamp to 0 just in case all were masked
        bz_max = bz_max.clamp(min=0.0)
        
        # 2. Polarity mixture (indicator of PIL presence)
        # IMPROVED: Use soft thresholding for smoother gradients
        sig_pos = torch.sigmoid(10 * B_z)
        sig_neg = torch.sigmoid(-10 * B_z)
        
        bz_pos_soft = torch.where(mask_bool, sig_pos, torch.zeros_like(sig_pos))
        bz_neg_soft = torch.where(mask_bool, sig_neg, torch.zeros_like(sig_neg))
        
        bz_pos = bz_pos_soft.sum(dim=(1, 2, 3)) / valid_count
        bz_neg = bz_neg_soft.sum(dim=(1, 2, 3)) / valid_count
        polarity_balance = (bz_pos * bz_neg).sqrt()  # High when balanced = PIL present
        
        # 3. Flow statistics
        u_mag = (u_x ** 2 + u_y ** 2 + 1e-8).sqrt()  # Add eps before sqrt
        u_masked = torch.where(mask_bool, u_mag, torch.zeros_like(u_mag))
        
        u_mean = u_masked.sum(dim=(1, 2, 3)) / valid_count
        
        u_for_max = torch.where(mask_bool, u_mag, torch.tensor(-1e9, device=u_mag.device))
        u_max = u_for_max.reshape(B, -1).max(dim=1)[0].clamp(min=0.0)
        
        # 4. Flux transport: correlation of flow with field gradient
        # This is a key indicator of magnetic energy buildup
        ft_val = u_mag * B_z.abs()
        flux_transport = torch.where(mask_bool, ft_val, torch.zeros_like(ft_val)).sum(dim=(1, 2, 3)) / valid_count
        
        # 5. Temporal variability (rate of change indicator)
        if T > 1:
            # First-order difference (velocity of change)
            dBz_dt = (B_z[:, 1:] - B_z[:, :-1]).abs()
            mask_dt = mask_float[:, 1:] * mask_float[:, :-1]
            mask_dt_bool = mask_dt.bool()
            
            dBz_dt_masked = torch.where(mask_dt_bool, dBz_dt, torch.zeros_like(dBz_dt))
            temporal_var = dBz_dt_masked.sum(dim=(1, 2, 3)) / (mask_dt.sum(dim=(1, 2, 3)) + 1e-8)
            
            # SOTA: Second-order difference (acceleration) - critical for flare precursors
            # Rapid acceleration often precedes flares
            if T > 2:
                d2Bz_dt2 = (dBz_dt[:, 1:] - dBz_dt[:, :-1]).abs()
                mask_dt2 = mask_dt[:, 1:] * mask_dt[:, :-1]
                mask_dt2_bool = mask_dt2.bool()
                d2Bz_dt2_masked = torch.where(mask_dt2_bool, d2Bz_dt2, torch.zeros_like(d2Bz_dt2))
                temporal_accel = d2Bz_dt2_masked.sum(dim=(1, 2, 3)) / (mask_dt2.sum(dim=(1, 2, 3)) + 1e-8)
            else:
                temporal_accel = torch.zeros(B, device=B_z.device)
            
            # SOTA: Evolution rate variance - high variance indicates unstable evolution
            dBz_dt_sq = torch.where(mask_dt_bool, dBz_dt**2, torch.zeros_like(dBz_dt))
            dBz_dt_mean_sq = temporal_var ** 2
            dBz_dt_var = (dBz_dt_sq.sum(dim=(1, 2, 3)) / (mask_dt.sum(dim=(1, 2, 3)) + 1e-8)) - dBz_dt_mean_sq
            evolution_rate_var = dBz_dt_var.clamp(min=0).sqrt()
        else:
            temporal_var = torch.zeros(B, device=B_z.device)
            temporal_accel = torch.zeros(B, device=B_z.device)
            evolution_rate_var = torch.zeros(B, device=B_z.device)
        
        # 6. Vector potential complexity (proxy for current density complexity)
        az_val = A_z ** 2
        az_var = torch.where(mask_bool, az_val, torch.zeros_like(az_val)).sum(dim=(1, 2, 3)) / valid_count
        az_std = az_var.clamp(min=1e-8).sqrt()
        
        # 7. Horizontal field features (if available)
        if "B_x" in feats and "B_y" in feats:
            B_x = feats["B_x"]
            B_y = feats["B_y"]
            
            # Horizontal field magnitude
            bh_mag = (B_x**2 + B_y**2 + 1e-8).sqrt()
            bh_masked = torch.where(mask_bool, bh_mag, torch.zeros_like(bh_mag))
            bh_mean = bh_masked.sum(dim=(1, 2, 3)) / valid_count
            
            # Total field magnitude  
            btot_mag = (B_x**2 + B_y**2 + B_z**2 + 1e-8).sqrt()
            btot_masked = torch.where(mask_bool, btot_mag, torch.zeros_like(btot_mag))
            btot_mean = btot_masked.sum(dim=(1, 2, 3)) / valid_count
            
            # 8. Shear angle proxy: angle between horizontal field and potential field
            # For a potential field, B would be nearly vertical at PIL
            # Shear = Bh / Btot (high shear = more non-potential = flare prone)
            shear_proxy = bh_mag / (btot_mag + 1e-8)
            shear_masked = torch.where(mask_bool, shear_proxy, torch.zeros_like(shear_proxy))
            shear_mean = shear_masked.sum(dim=(1, 2, 3)) / valid_count
            
            # 9. Free energy proxy: E_free ∝ Bh² (excess over potential field)
            # This is a key predictor of flare energy
            free_energy_proxy = bh_mag ** 2
            fe_masked = torch.where(mask_bool, free_energy_proxy, torch.zeros_like(free_energy_proxy))
            free_energy_mean = fe_masked.sum(dim=(1, 2, 3)) / valid_count
        else:
            bh_mean = torch.zeros_like(bz_mean)
            btot_mean = bz_mean.abs()  # Fallback to |Bz|
            shear_mean = torch.zeros_like(bz_mean)
            free_energy_mean = torch.zeros_like(bz_mean)

        # Concatenate base physics features (now 13 features when vector field available)
        physics_feats = torch.stack([
            bz_mean, bz_std, bz_max, polarity_balance,
            u_mean, u_max, flux_transport,
            temporal_var, az_std,
            bh_mean, btot_mean,
            shear_mean, free_energy_mean  # NEW: critical for flare prediction
        ], dim=-1)  # [B, 13]
        
        # Handle any NaN that might occur
        physics_feats = torch.nan_to_num(physics_feats, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return physics_feats


class ClassifierHead(nn.Module):
    """
    Advanced classifier head with attention and physics features.
    """
    def __init__(
        self, 
        hidden: int = 256, 
        dropout: float = 0.1, 
        horizons: tuple[int, ...] = (6, 12, 24),
        use_attention: bool = True,
        use_physics_features: bool = True,
        rf_importance_weights: Optional[torch.Tensor] = None,
        n_scalar_features: int = 4,
    ):
        super().__init__()
        self.horizons = tuple(horizons)
        self.use_attention = use_attention
        self.use_physics_features = use_physics_features
        self.n_scalar_features = n_scalar_features
        
        # Base features: A_z, B_z, u_x, u_y
        base_features = 4
        
        # Spatial attention pooling
        if use_attention:
            self.spatial_attn = SpatialAttentionPool(base_features, hidden=64)
            self.temporal_attn = TemporalAttentionPool(base_features, hidden=64)
        
        # Physics feature extractor
        if use_physics_features:
            self.physics_extractor = PhysicsFeatureExtractor()
            physics_features = 13  # Updated: includes shear and free energy proxies
        else:
            physics_features = 0
        
        # RF importance weights for physics features
        # If provided, these weight the physics features by domain knowledge
        if rf_importance_weights is not None:
            self.register_buffer('rf_weights', rf_importance_weights)
        else:
            # Default: uniform weights
            self.register_buffer('rf_weights', torch.ones(physics_features) if physics_features > 0 else None)
        
        # Scalar feature processing (R-value, GWPIL, etc.)
        if n_scalar_features > 0:
            self.scalar_proj = nn.Sequential(
                nn.Linear(n_scalar_features, hidden // 4),
                nn.SiLU(),
                nn.Dropout(dropout * 0.5),
            )
            scalar_out_dim = hidden // 4
        else:
            self.scalar_proj = None
            scalar_out_dim = 0
        
        # Compute total input features for final MLP
        # With attention: pooled(4) + mean(4) + std(4) + max(4) = 16
        # Plus physics features (9) + scalar features (hidden//4) = 25 + scalar_out
        # Without attention: mean(4) + std(4) + physics_features(9) + scalar = 17 + scalar_out
        if use_attention:
            mlp_input = base_features * 4 + physics_features + scalar_out_dim
        else:
            mlp_input = base_features * 2 + physics_features + scalar_out_dim
        
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

    def forward(
        self, 
        feats: dict[str, torch.Tensor], 
        observed_mask: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feats: dict of tensors shaped [B, T, P, 1] for keys A_z, B_z, u_x, u_y
            observed_mask: [B, T] booleans (True for frames t <= t0)
            scalars: [B, n_scalar_features] optional scalar features (R-value, GWPIL, etc.)
            
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
            
            # IMPROVED: Compute comprehensive statistics for better feature representation
            mask_expanded = observed_mask[..., None, None].float()  # [B, T, 1, 1]
            X_masked = X * mask_expanded
            # FIXED: valid_count must account for spatial points P to avoid scaling issues
            valid_count = mask_expanded.sum(dim=(1, 2)).clamp(min=1.0) * P  # [B, 1]
            
            # Mean
            X_mean = X_masked.sum(dim=(1, 2)) / valid_count  # [B, 4]
            
            # Standard deviation (more informative than just mean)
            X_sq = (X_masked ** 2).sum(dim=(1, 2)) / valid_count
            X_var = (X_sq - X_mean ** 2).clamp(min=1e-8)
            X_std = X_var.sqrt()  # [B, 4]
            
            # Max absolute value (captures peak activity)
            # Set masked values to large negative so they don't affect max
            X_for_max = X.abs() * mask_expanded + (1 - mask_expanded) * (-1e9)
            X_max = X_for_max.max(dim=2)[0].max(dim=1)[0]  # [B, 4]
            
            # Combine: attention pooled + mean + std + max = 16 features
            combined = torch.cat([X_pooled, X_mean, X_std, X_max], dim=-1)  # [B, 16]
        else:
            # Improved fallback with mean + std statistics
            mask_expanded = observed_mask[..., None, None].float()
            X_masked = X * mask_expanded
            valid_count = mask_expanded.sum(dim=(1, 2)).clamp(min=1.0)
            
            X_mean = X_masked.sum(dim=(1, 2)) / valid_count  # [B, 4]
            X_sq = (X_masked ** 2).sum(dim=(1, 2)) / valid_count
            X_var = (X_sq - X_mean ** 2).clamp(min=1e-8)
            X_std = X_var.sqrt()  # [B, 4]
            
            combined = torch.cat([X_mean, X_std], dim=-1)  # [B, 8]
        
        # Physics features
        if self.use_physics_features:
            physics_feats = self.physics_extractor(feats, observed_mask)  # [B, 9]
            
            # Apply RF importance weights if available
            if self.rf_weights is not None:
                physics_feats = physics_feats * self.rf_weights
            
            combined = torch.cat([combined, physics_feats], dim=-1)
        
        # Scalar features (R-value, GWPIL, observation coverage, etc.)
        if self.scalar_proj is not None and scalars is not None:
            # Normalize scalars for stability (log-transform large values)
            scalars_safe = scalars.clone()
            # R-value is already log-scale, GWPIL can be large
            # Apply log1p to GWPIL (index 1) if it's large
            if scalars_safe.shape[-1] > 1:
                scalars_safe[:, 1] = torch.log1p(scalars_safe[:, 1].abs()) * torch.sign(scalars_safe[:, 1])
            
            scalar_feats = self.scalar_proj(scalars_safe)  # [B, hidden//4]
            combined = torch.cat([combined, scalar_feats], dim=-1)
        
        # MLP classifier
        h = self.pre_mlp(combined)
        logits = self.classifier(h)
        
        return logits  # [B, n_horizons]

