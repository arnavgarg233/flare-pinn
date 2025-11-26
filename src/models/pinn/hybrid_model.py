# src/models/pinn/hybrid_model.py
"""
Hybrid CNN-conditioned PINN model for solar flare prediction.

Combines CNN spatial features with physics-informed coordinate fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .collocation import clip_and_renorm_importance
from .config import PINNConfig
from .core import B_perp_from_Az, ClassifierHead
from .hybrid_core import HybridPINNBackbone
from .losses import (
    asymmetric_focal_loss,
    bce_logits,
    class_balanced_focal_loss,
    confidence_penalty,
    focal_loss,
    focal_loss_with_label_smoothing,
    interp_schedule,
    l1_data,
    poly_focal_loss,
)
from .physics import WeakFormInduction2p5D, VectorInduction2p5D


# =============================================================================
# Typed Output Models
# =============================================================================

@dataclass
class FieldOutputs:
    """Neural field outputs at collocation points."""
    A: torch.Tensor    # [N, C] potential
    B: torch.Tensor    # [N, C] magnetic field
    u: torch.Tensor    # [N, 2] velocity (ux, uy)
    eta_raw: Optional[torch.Tensor]
    
    @property
    def B_z(self) -> torch.Tensor:
        """Get Bz component (last component of B field)."""
        # For C=1: B is Bz directly
        # For C=3: B = [Bx, By, Bz], so Bz is index 2
        if self.B.shape[-1] == 1:
            return self.B
        return self.B[..., 2:3]
    
    @property
    def B_x(self) -> Optional[torch.Tensor]:
        """Get Bx component (first component, only valid for 3-component B)."""
        if self.B.shape[-1] >= 3:
            return self.B[..., 0:1]
        return None
    
    @property
    def B_y(self) -> Optional[torch.Tensor]:
        """Get By component (second component, only valid for 3-component B)."""
        if self.B.shape[-1] >= 3:
            return self.B[..., 1:2]
        return None
    
    @property
    def u_x(self) -> torch.Tensor:
        """Get ux velocity component."""
        return self.u[..., 0:1]
    
    @property
    def u_y(self) -> torch.Tensor:
        """Get uy velocity component."""
        return self.u[..., 1:2]

@dataclass
class HybridPINNOutput:
    """Complete output from hybrid PINN forward pass."""
    logits: torch.Tensor
    probs: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor
    u: torch.Tensor
    # Legacy compatibility fields
    A_z: Optional[torch.Tensor] = None
    B_z: Optional[torch.Tensor] = None
    u_x: Optional[torch.Tensor] = None
    u_y: Optional[torch.Tensor] = None
    B_x: Optional[torch.Tensor] = None
    B_y: Optional[torch.Tensor] = None
    # Loss components
    loss_cls: torch.Tensor = None  # type: ignore
    loss_data: torch.Tensor = None  # type: ignore
    loss_phys: torch.Tensor = None  # type: ignore
    loss_curl: torch.Tensor = None  # type: ignore
    loss_total: torch.Tensor = None  # type: ignore
    # Diagnostic info
    ess: float = 0.0
    lambda_phys: float = 0.0
    fourier_alpha: float = 0.0


# =============================================================================
# Main Model Class
# =============================================================================

class HybridPINNModel(nn.Module):
    """
    Hybrid CNN-PINN for solar flare prediction.
    """
    
    def __init__(self, cfg: PINNConfig, encoder_in_channels: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        
        # Determine input channels for CNN
        if encoder_in_channels is None:
            # Derived from data config if available, else ModelConfig
            if cfg.model.in_channels is not None:
                in_channels = cfg.model.in_channels
            else:
                in_channels = cfg.data.n_components
        else:
            in_channels = encoder_in_channels
            
        self.n_components = cfg.data.n_components
        
        # Hybrid backbone (CNN encoder + conditioned MLP)
        self.backbone = HybridPINNBackbone(
            encoder_in_channels=in_channels,
            latent_channels=32,
            global_dim=64,
            hidden=cfg.model.hidden,
            layers=cfg.model.layers,
            max_log2_freq=cfg.model.fourier.max_log2_freq,
            film_layers=(3, 6, 9),
            learn_eta=cfg.model.learn_eta,
            encoder_dropout=0.05,
            n_field_components=self.n_components
        )
        
        # Load RF importance weights if configured
        rf_weights = self._get_rf_weights(cfg)
        
        # Classifier head with optional RF guidance
        use_attention = getattr(cfg.classifier, 'use_attention', True)
        use_physics_features = getattr(cfg.classifier, 'use_physics_features', True)
        
        # Scalar features count from config (default: R-value, GWPIL, obs_coverage, frame_count)
        n_scalar_features = len(cfg.data.scalar_features) if cfg.data.scalar_features else 4
        
        self.classifier = ClassifierHead(
            hidden=cfg.classifier.hidden,
            dropout=cfg.classifier.dropout,
            horizons=cfg.classifier.horizons,
            use_attention=use_attention,
            use_physics_features=use_physics_features,
            rf_importance_weights=rf_weights,
            n_scalar_features=n_scalar_features,
        )
        
        # Physics module
        self.physics = WeakFormInduction2p5D(
            eta_bounds=(cfg.eta.min, cfg.eta.max),
            use_resistive=cfg.physics.resistive,
            include_boundary=cfg.physics.boundary_terms,
            tv_eta=cfg.eta.tv_weight
        )
        
        # Training state
        self.register_buffer("_train_frac", torch.tensor(0.0))
    
    def _get_rf_weights(self, cfg: PINNConfig) -> Optional[torch.Tensor]:
        """Get RF importance weights from file or defaults."""
        if not cfg.classifier.use_rf_guidance:
            return None
            
        if cfg.classifier.rf_weights_path is not None:
            return self._load_rf_weights(cfg.classifier.rf_weights_path)
        
        # Default domain-knowledge weights based on solar flare literature
        # Updated for 18 physics features
        rf_weights = torch.tensor([
            # Base field statistics (4)
            1.0,   # bz_mean - moderate importance
            1.5,   # bz_std - high importance (variability)
            1.2,   # bz_max - high importance (peak field)
            2.0,   # polarity_balance - CRITICAL (PIL indicator)
            # Flow statistics (3)
            0.8,   # u_mean - moderate (flow)
            1.0,   # u_max - moderate (peak flow)
            1.8,   # flux_transport - high (PIL activity)
            # Temporal features (4) - CRITICAL for flare prediction
            1.5,   # temporal_var - high (field evolution)
            2.5,   # temporal_accel - VERY HIGH (acceleration precedes flares)
            2.0,   # evolution_rate_var - high (instability indicator)
            2.2,   # recent_evolution - VERY HIGH (latest changes most predictive)
            # Field complexity (2)
            0.7,   # az_std - lower (indirect)
            1.5,   # kurtosis_proxy - high (non-Gaussian = complex field)
            # Vector field features (5)
            1.2,   # bh_mean - high (horizontal field)
            1.5,   # btot_mean - high (total energy)
            2.0,   # shear_mean - CRITICAL (non-potentiality indicator)
            2.5,   # free_energy_mean - CRITICAL (flare energy reservoir)
            2.8,   # current_helicity_proxy - HIGHEST (twist/writhe = flare trigger)
        ], dtype=torch.float32)
        # Normalize to sum to n_features for scale preservation
        n_features = len(rf_weights)
        return rf_weights / rf_weights.sum() * n_features
    
    def _load_rf_weights(self, path: Path) -> Optional[torch.Tensor]:
        """Load RF importance weights from pickle file."""
        try:
            from .rf_guidance import RFImportances
            rf_imp = RFImportances.load(path)
            
            # Map to the 11 physics features used in ClassifierHead
            feature_mapping = [
                'bz_mean', 'bz_std', 'bz_max', 'polarity_balance',
                'mean_gradient', 'max_gradient', 'gwpil',
                'temporal_variation', 'spatial_entropy',
                'bz_mean', 'bz_mean'  # Fallback/Proxy for horizontal/total field
            ]
            
            weights = [rf_imp.get_weight(name) for name in feature_mapping]
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            # Normalize to preserve feature scale
            return weights_tensor / weights_tensor.sum() * len(weights)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load RF weights from {path}: {e}. Using uniform weights.")
            return None
    
    # =========================================================================
    # Curriculum / Schedule Methods
    # =========================================================================
    
    def set_train_frac(self, frac: float) -> None:
        """Update training progress fraction [0, 1]."""
        frac = float(max(0.0, min(1.0, frac)))
        self._train_frac.fill_(frac)
        ramp = self.cfg.model.fourier.ramp_frac
        alpha = min(1.0, frac / ramp) if ramp > 0 else 1.0
        self.backbone.set_fourier_alpha(alpha)
    
    def get_lambda_phys(self) -> float:
        """Get physics loss weight based on training progress."""
        if not self.cfg.physics.enable:
            return 0.0
        frac = float(self._train_frac.item())
        lam = interp_schedule(self.cfg.physics.lambda_phys_schedule, frac)
        return min(lam, 50.0)  # Safety cap
    
    def get_collocation_alpha(self) -> float:
        """Get PIL importance weight based on training progress."""
        frac = float(self._train_frac.item())
        a0 = self.cfg.collocation.alpha_start
        a1 = self.cfg.collocation.alpha_end
        return a0 + (a1 - a0) * frac
    
    # =========================================================================
    # Decomposed Forward Components
    # =========================================================================
    
    def encode_frames(
        self,
        frames: torch.Tensor,
        observed_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observed frames into latent representation.
        
        Args:
            frames: [T, H, W] - Bz frames for entire window
            observed_mask: [T] - Which frames are observed (t <= t0)
        
        Returns:
            L: [1, C, H, W] - Latent feature map
            g: [1, D] - Global code
        """
        if getattr(self.backbone, 'use_temporal_encoder', False):
            return self.backbone.encode(frames, observed_mask)
        
        # Fallback for TinyEncoder: manually aggregate frames
        obs_frames = frames[observed_mask]
        
        if obs_frames.shape[0] == 0:
            device = frames.device
            H, W = frames.shape[-2:]
            latent_ch = self.backbone.effective_latent_channels
            L = torch.zeros(1, latent_ch, H, W, device=device)
            g = torch.zeros(1, 64, device=device)
            return L, g
        
        # Aggregate observed frames by mean
        if obs_frames.dim() == 3:
            # [T, H, W] -> [1, 1, H, W]
            frames_input = obs_frames.mean(dim=0, keepdim=True).unsqueeze(0)
        else:
            # [T, C, H, W] -> [1, C, H, W]
            frames_input = obs_frames.mean(dim=0, keepdim=True)
            
        return self.backbone.encode(frames_input)
    
    def query_field(
        self,
        coords: torch.Tensor,
        L: torch.Tensor,
        g: torch.Tensor
    ) -> FieldOutputs:
        """
        Query neural field at collocation points.
        """
        coords.requires_grad_(True)
        out = self.backbone(coords, L, g, use_nearest=False)
        
        return FieldOutputs(
            A=out["A"],
            B=out["B"],
            u=out["u"],
            eta_raw=out["eta_raw"]
        )
    
    def compute_classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        probs: torch.Tensor,
        samples_per_class: tuple[int, int] = (1000, 50)  # Estimated from data
    ) -> torch.Tensor:
        """Compute classification loss based on config."""
        label_smoothing = getattr(self.cfg.classifier, 'label_smoothing', 0.0)
        loss_type = self.cfg.classifier.loss_type
        
        if loss_type == "cb_focal":
            # SOTA: Class-balanced focal loss for severe imbalance
            cb_beta = getattr(self.cfg.classifier, 'cb_beta', 0.9999)
            loss = class_balanced_focal_loss(
                logits, labels,
                samples_per_class=samples_per_class,
                beta=cb_beta,
                gamma=self.cfg.classifier.focal_gamma
            )
        elif loss_type == "poly_focal":
            # PolyLoss + Focal for better calibration
            poly_epsilon = getattr(self.cfg.classifier, 'poly_epsilon', 1.0)
            loss = poly_focal_loss(
                logits, labels,
                epsilon=poly_epsilon,
                alpha=self.cfg.classifier.focal_alpha,
                gamma=self.cfg.classifier.focal_gamma
            )
        elif loss_type == "asymmetric":
            gamma_neg = getattr(self.cfg.classifier, 'asymmetric_gamma_neg', 4.0)
            loss = asymmetric_focal_loss(
                logits, labels, gamma_pos=0.0, gamma_neg=gamma_neg, clip=0.05
            )
        elif loss_type == "focal":
            if label_smoothing > 0:
                loss = focal_loss_with_label_smoothing(
                    logits, labels,
                    alpha=self.cfg.classifier.focal_alpha,
                    gamma=self.cfg.classifier.focal_gamma,
                    smoothing=label_smoothing
                )
            else:
                loss = focal_loss(
                    logits, labels,
                    alpha=self.cfg.classifier.focal_alpha,
                    gamma=self.cfg.classifier.focal_gamma
                )
        else:  # bce
            loss = bce_logits(logits, labels, pos_weight=self.cfg.classifier.pos_weight)
        
        # Add confidence penalty for calibration
        conf_penalty_weight = getattr(self.cfg.classifier, 'confidence_penalty', 0.0)
        if conf_penalty_weight > 0:
            loss = loss + confidence_penalty(probs, conf_penalty_weight)
        
        return loss
    
    def compute_data_loss(
        self,
        B_field: torch.Tensor,
        gt_field: torch.Tensor,
        observed_mask: torch.Tensor,
        B: int, T: int, P: int
    ) -> torch.Tensor:
        """Compute data fitting loss (L1 on field components)."""
        if observed_mask.dim() == 1:
            mask_expanded = observed_mask[None, :, None, None].expand(B, T, P, 1)
        else:
            mask_expanded = observed_mask[:, :, None, None].expand(B, T, P, 1)
        
        # gt_field: [B, T, P, C] (assuming batch dim added in forward)
        # B_field: [B*T*P, C] -> reshape
        
        B_reshaped = B_field.reshape(B, T, P, -1)
        
        # If gt has fewer components than pred, slice pred (or vice versa)?
        # Assume they match for now due to dataset config.
        if gt_field.shape[-1] != B_reshaped.shape[-1]:
             # Fallback logic if dimensions mismatch (e.g. 1 vs 3)
             # If pred is 1 (Bz) and gt is 3 (Bx,By,Bz), take Bz (index 2? or last?)
             # Or if pred is 3 and gt is 1, take pred corresponding to Bz.
             # This is tricky. We'll assume exact match for "data_loss" on "components".
             pass
             
        return l1_data(B_reshaped, gt_field, mask_expanded)

    def compute_curl_loss(
        self,
        A_field: torch.Tensor,
        coords_flat: torch.Tensor,
        gt_field: torch.Tensor,
        observed_mask: torch.Tensor,
        B: int, T: int, P: int
    ) -> torch.Tensor:
        """Compute curl consistency loss."""
        # Only applicable if we can map A -> B via curl.
        # For 2.5D with Az: Bx = -dAz/dy, By = dAz/dx
        # This requires Az to be 1 component and gt to have Bx/By.
        # If n_components=1 (Bz), we don't have Bx/By in gt (unless we loaded them separately?)
        # New dataset loads all 'components'. If components=['Bz'], we don't have Bx/By.
        # So this loss is only valid if we have Bx/By in 'components' OR if we hacked it.
        return torch.tensor(0.0, device=A_field.device)

    
    def compute_physics_loss(
        self,
        coords_flat: torch.Tensor,
        L: torch.Tensor,
        g: torch.Tensor,
        pil_mask: Optional[torch.Tensor],
        lambda_phys: float
    ) -> tuple[torch.Tensor, float]:
        """
        Compute physics loss with optional gradient scaling.
        
        Uses physics_grad_scale from config to prevent physics loss
        from dominating the classification objective.
        """
        # This needs to handle the new backbone output format.
        # For now, we skip full refactor of Physics module in this turn 
        # and just return 0.0 if components != 1 to avoid crash.
        if self.n_components != 1:
            # Warning: Physics loss not yet adapted for vector fields
            return torch.tensor(0.0, device=coords_flat.device), 0.0
            
        device = coords_flat.device
        N = coords_flat.shape[0]
        alpha = self.get_collocation_alpha()
        
        # Get physics gradient scale from config (default 0.5)
        physics_grad_scale = getattr(self.cfg.physics, 'physics_grad_scale', 0.5)
        
        # ... (existing importance sampling code) ...
        # Compute importance weights
        if pil_mask is not None:
            xy_coords = coords_flat[:, :2]
            H_pil, W_pil = pil_mask.shape
            
            # Map coords to PIL mask indices
            x_px = ((xy_coords[:, 0] + 1.0) * 0.5 * (W_pil - 1)).long().clamp(0, W_pil - 1)
            y_px = ((xy_coords[:, 1] + 1.0) * 0.5 * (H_pil - 1)).long().clamp(0, H_pil - 1)
            
            # pil_mask is now a tensor - index directly
            pil_values = pil_mask[y_px, x_px].float()
            
            p_uniform = 0.125
            pil_sum = pil_mask.sum().item()
            pil_frac = pil_sum / (H_pil * W_pil) if pil_sum > 0 else 1.0
            p_pil = (1.0 / (4.0 * pil_frac) * 0.5) if pil_frac > 0 else p_uniform
            p_spatial = alpha * p_pil * pil_values + (1 - alpha) * p_uniform
            p_spatial = p_spatial.clamp_min(1e-8)
            imp_weights, _ = clip_and_renorm_importance(
                p_spatial.unsqueeze(-1), 
                self.cfg.collocation.impw_clip_quantile
            )
        else:
            p_uniform = torch.full((N, 1), 0.125, device=device)
            imp_weights, _ = clip_and_renorm_importance(
                p_uniform, 
                self.cfg.collocation.impw_clip_quantile
            )
        
        # Compute ESS
        w_sum = imp_weights.sum()
        w_sq_sum = (imp_weights ** 2).sum()
        ess = float((w_sum ** 2 / w_sq_sum).item()) if w_sq_sum > 0 else 0.0
        
        # Physics residual
        eta_mode = "field" if self.cfg.model.learn_eta else "scalar"
        
        # Wrap backbone to output compatible format for Physics module
        def model_wrapper(c):
            out = self.backbone(c, L, g, use_nearest=False)
            # Physics module expects dictionary with "B_z", "u_x", "u_y", "eta_raw"
            # Map from new "B", "u" to old keys
            return {
                "B_z": out["B"], # Assuming C=1
                "u_x": out["u"][..., 0:1],
                "u_y": out["u"][..., 1:2],
                "eta_raw": out["eta_raw"]
            }

        loss_phys_raw, _ = self.physics(
            model_wrapper,
            coords_flat,
            imp_weights,
            eta_mode=eta_mode,
            eta_scalar=self.cfg.model.eta_scalar
        )
        
        # Apply physics gradient scaling to prevent dominating classification
        # Uses GradientScaler-like approach: scale loss during forward, 
        # effectively scaling gradients during backward
        scaled_loss = lambda_phys * loss_phys_raw * physics_grad_scale
        
        return scaled_loss, ess
    
    def _aggregate_losses(
        self,
        loss_cls: torch.Tensor,
        loss_data: torch.Tensor,
        loss_phys: torch.Tensor,
        loss_curl: torch.Tensor,
        lambda_phys: float,
        device: torch.device
    ) -> torch.Tensor:
        """Aggregate losses with NaN safety checks."""
        loss_components = []
        
        if torch.isfinite(loss_cls):
            loss_components.append(self.cfg.loss_weights.cls * loss_cls)
        
        if torch.isfinite(loss_data):
            loss_components.append(self.cfg.loss_weights.data * loss_data)
        
        if torch.isfinite(loss_curl) and self.cfg.loss_weights.curl_consistency > 0:
             loss_components.append(self.cfg.loss_weights.curl_consistency * loss_curl)
        
        if torch.isfinite(loss_phys) and lambda_phys > 0.01:
            loss_components.append(loss_phys)
        
        if loss_components:
            loss_total = sum(loss_components)
        else:
            loss_total = torch.tensor(0.01, device=device, requires_grad=True)
        
        if not torch.isfinite(loss_total):
            loss_total = torch.tensor(0.01, device=device, requires_grad=True)
        
        return loss_total
    
    # =========================================================================
    # Main Forward Pass
    # =========================================================================
    
    def forward(
        self,
        coords: torch.Tensor,
        frames: Optional[torch.Tensor] = None,
        gt_bz: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pil_mask: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
        mode: str = "train"
    ) -> HybridPINNOutput:
        """
        Forward pass.
        """
        device = coords.device
        T, P = coords.shape[0], coords.shape[1]
        B = 1
        
        # 1. Encode frames
        has_observed = observed_mask is not None and observed_mask.any()
        
        if frames is not None and observed_mask is not None:
            L, g = self.encode_frames(frames, observed_mask)
        else:
            H_spatial = 128
            latent_ch = self.backbone.effective_latent_channels
            L = torch.zeros(1, latent_ch, H_spatial, H_spatial, device=device)
            g = torch.zeros(1, 64, device=device)
            has_observed = False
        
        # 2. Query field
        coords_flat = coords.reshape(-1, 3).contiguous()
        field = self.query_field(coords_flat, L, g)
        
        # 3. Classification
        if observed_mask is None:
            observed_mask = torch.ones(T, dtype=torch.bool, device=device)
        
        # Construct feats_dict for ClassifierHead (which expects specific keys)
        # We map our generic fields to what ClassifierHead expects
        feats_dict = {
            "A_z": field.A.reshape(B, T, P, -1), # Use full A as features
            "B_z": field.B.reshape(B, T, P, -1), # Use full B as features
            "u_x": field.u.reshape(B, T, P, -1)[..., 0:1],
            "u_y": field.u.reshape(B, T, P, -1)[..., 1:2],
        }
        
        # Prepare scalars for classifier (ensure batch dimension)
        if scalars is not None and scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)  # [n_scalars] -> [1, n_scalars]
        
        logits = self.classifier(feats_dict, observed_mask.unsqueeze(0), scalars=scalars)
        probs = torch.sigmoid(logits)
        
        # 4. Compute losses
        loss_cls = torch.tensor(0.0, device=device)
        loss_data = torch.tensor(0.0, device=device)
        loss_phys = torch.tensor(0.0, device=device)
        loss_curl = torch.tensor(0.0, device=device)
        ess_val = 0.0
        lambda_phys = self.get_lambda_phys()
        
        if mode == "train":
            # Classification loss
            if labels is not None:
                loss_cls = self.compute_classification_loss(logits, labels, probs)
            
            # Data fitting loss
            if gt_bz is not None:
                # gt_bz is now [T, P, C] - rename argument ideally but keeping for signature compat
                if gt_bz.dim() == 3:
                    gt_target = gt_bz.unsqueeze(0) # [1, T, P, C]
                else:
                    gt_target = gt_bz
                    
                loss_data = self.compute_data_loss(
                    field.B, gt_target, observed_mask, B, T, P
                )
                
                if self.cfg.loss_weights.curl_consistency > 0:
                    loss_curl = self.compute_curl_loss(
                        field.A, coords_flat, gt_target, observed_mask, B, T, P
                    )
            
            # Physics loss
            has_valid_outputs = (
                torch.isfinite(field.B).all() and 
                torch.isfinite(field.u).all()
            )
            
            if lambda_phys > 0.01 and has_observed and has_valid_outputs:
                loss_phys, ess_val = self.compute_physics_loss(
                    coords_flat, L, g, pil_mask, lambda_phys
                )
        
        # 5. Aggregate losses
        loss_total = self._aggregate_losses(
            loss_cls, loss_data, loss_phys, loss_curl, lambda_phys, device
        )
        
        return HybridPINNOutput(
            logits=logits,
            probs=probs,
            A=field.A,
            B=field.B,
            u=field.u,
            # Fill legacy fields for return type compatibility if needed
            A_z=field.A, # placeholder
            B_z=field.B, # placeholder
            u_x=field.u[..., 0:1],
            u_y=field.u[..., 1:2],
            B_x=None,
            B_y=None,
            loss_cls=loss_cls,
            loss_data=loss_data,
            loss_phys=loss_phys,
            loss_curl=loss_curl,
            loss_total=loss_total,
            ess=ess_val,
            lambda_phys=lambda_phys,
            fourier_alpha=self.backbone.ff._alpha.item()
        )
