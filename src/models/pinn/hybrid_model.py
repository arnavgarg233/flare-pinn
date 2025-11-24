# src/models/pinn/hybrid_model.py
"""
Hybrid CNN-conditioned PINN model for solar flare prediction.

Combines CNN spatial features with physics-informed coordinate fields.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from .config import PINNConfig
from .hybrid_core import HybridPINNBackbone
from .core import ClassifierHead, B_perp_from_Az
from .physics import WeakFormInduction2p5D
from .losses import (
    focal_loss, focal_loss_with_label_smoothing, bce_logits, l1_data, 
    interp_schedule, asymmetric_focal_loss, confidence_penalty
)
from .collocation import clip_and_renorm_importance


@dataclass
class HybridPINNOutput:
    """Output from hybrid PINN forward pass."""
    # Same as PINNOutput
    logits: torch.Tensor
    probs: torch.Tensor
    A_z: torch.Tensor
    B_z: torch.Tensor
    u_x: torch.Tensor
    u_y: torch.Tensor
    B_x: Optional[torch.Tensor]
    B_y: Optional[torch.Tensor]
    loss_cls: torch.Tensor
    loss_data: torch.Tensor
    loss_phys: torch.Tensor
    loss_total: torch.Tensor
    ess: float
    lambda_phys: float
    fourier_alpha: float


class HybridPINNModel(nn.Module):
    """
    Hybrid CNN-PINN for solar flare prediction.
    
    Workflow:
        1. Encode observed frames (t <= t0) → latent (L, g)
        2. Sample collocation points
        3. Query conditioned neural field at collocation points
        4. Compute physics residuals + data loss + classification loss
        5. Backprop through entire pipeline
    
    Advantages over pure coordinate PINN:
        - CNN provides spatial inductive bias
        - Learns translation-equivariant features
        - Better at capturing local PIL structures
    """
    def __init__(self, cfg: PINNConfig, encoder_in_channels: int = 1):
        super().__init__()
        self.cfg = cfg
        
        # Hybrid backbone (CNN encoder + conditioned MLP)
        self.backbone = HybridPINNBackbone(
            encoder_in_channels=encoder_in_channels,
            latent_channels=32,
            global_dim=64,
            hidden=cfg.model.hidden,
            layers=cfg.model.layers,
            max_log2_freq=cfg.model.fourier.max_log2_freq,
            film_layers=(3, 6, 9),
            learn_eta=cfg.model.learn_eta,
            encoder_dropout=0.05
        )
        
        # Load RF importance weights if configured
        rf_weights = None
        if cfg.classifier.use_rf_guidance and cfg.classifier.rf_weights_path is not None:
            rf_weights = self._load_rf_weights(cfg.classifier.rf_weights_path)
        elif cfg.classifier.use_rf_guidance:
            # Use default domain-knowledge weights if RF not provided
            # Based on solar flare prediction literature
            rf_weights = torch.tensor([
                1.0,   # bz_mean - moderate importance
                1.5,   # bz_std - high importance (variability)
                1.2,   # bz_max - high importance (peak field)
                2.0,   # polarity_balance - CRITICAL (PIL indicator)
                0.8,   # u_mean - moderate (flow)
                1.0,   # u_max - moderate (peak flow)
                1.8,   # flux_transport - high (PIL activity)
                1.5,   # temporal_var - high (field evolution)
                0.7,   # az_std - lower (indirect)
            ], dtype=torch.float32)
            # Normalize to sum to n_features for scale preservation
            rf_weights = rf_weights / rf_weights.sum() * 9.0
        
        # Classifier head with optional RF guidance
        use_attention = getattr(cfg.classifier, 'use_attention', True)
        use_physics_features = getattr(cfg.classifier, 'use_physics_features', True)
        
        self.classifier = ClassifierHead(
            hidden=cfg.classifier.hidden,
            dropout=cfg.classifier.dropout,
            horizons=cfg.classifier.horizons,
            use_attention=use_attention,
            use_physics_features=use_physics_features,
            rf_importance_weights=rf_weights
        )
        
        # Physics module (same as pure PINN)
        self.physics = WeakFormInduction2p5D(
            eta_bounds=(cfg.eta.min, cfg.eta.max),
            use_resistive=cfg.physics.resistive,
            include_boundary=cfg.physics.boundary_terms,
            tv_eta=cfg.eta.tv_weight
        )
        
        # Training state
        self.register_buffer("_train_frac", torch.tensor(0.0))
    
    def _load_rf_weights(self, path: Path) -> Optional[torch.Tensor]:
        """Load RF importance weights from pickle file."""
        try:
            from .rf_guidance import RFImportances
            rf_imp = RFImportances.load(path)
            
            # Map to the 9 physics features used in ClassifierHead
            feature_mapping = [
                'bz_mean', 'bz_std', 'bz_max', 'polarity_balance',
                'mean_gradient', 'max_gradient', 'gwpil',  # Proxies for u_mean, u_max, flux_transport
                'temporal_variation', 'spatial_entropy'  # Proxies for temporal_var, az_std
            ]
            
            weights = []
            for name in feature_mapping:
                w = rf_imp.get_weight(name)
                weights.append(w)
            
            weights = torch.tensor(weights, dtype=torch.float32)
            # Normalize to preserve feature scale
            weights = weights / weights.sum() * len(weights)
            
            return weights
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load RF weights from {path}: {e}. Using uniform weights.")
            return None
    
    def set_train_frac(self, frac: float) -> None:
        frac = float(max(0.0, min(1.0, frac)))
        self._train_frac.fill_(frac)
        ramp = self.cfg.model.fourier.ramp_frac
        if ramp > 0:
            alpha = min(1.0, frac / ramp)
        else:
            alpha = 1.0
        self.backbone.set_fourier_alpha(alpha)
    
    def get_lambda_phys(self) -> float:
        if not self.cfg.physics.enable:
            return 0.0
        frac = float(self._train_frac.item())
        return interp_schedule(self.cfg.physics.lambda_phys_schedule, frac)
    
    def get_collocation_alpha(self) -> float:
        frac = float(self._train_frac.item())
        a0 = self.cfg.collocation.alpha_start
        a1 = self.cfg.collocation.alpha_end
        return a0 + (a1 - a0) * frac
    
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
        # Only encode observed frames (no future leakage!)
        # frames shape is [T, H, W]
        
        if getattr(self.backbone, 'use_temporal_encoder', False):
            # TemporalEncoder handles raw frames and masking internally
            # It expects [T, H, W] and [T] mask
            return self.backbone.encode(frames, observed_mask)
        
        # Fallback for TinyEncoder: manually aggregate frames
        obs_frames = frames[observed_mask]  # [T_obs, H, W]
        
        if obs_frames.shape[0] == 0:
            # No observed frames - return zeros
            device = frames.device
            H, W = frames.shape[-2:]
            L = torch.zeros(1, 32, H, W, device=device)
            g = torch.zeros(1, 64, device=device)
            return L, g
        
        # Aggregate observed frames into single channel by taking mean
        # This provides temporal context while maintaining single-channel input
        # Shape: [T_obs, H, W] -> [1, 1, H, W]
        frames_input = obs_frames.mean(dim=0, keepdim=True).unsqueeze(0)
        
        # Encode
        L, g = self.backbone.encode(frames_input)
        return L, g
    
    def forward(
        self,
        coords: torch.Tensor,
        frames: Optional[torch.Tensor] = None,  # NEW: full frames for encoding
        gt_bz: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pil_mask: Optional[np.ndarray] = None,
        mode: str = "train"
    ) -> HybridPINNOutput:
        """
        Forward pass with CNN conditioning.
        
        Args:
            coords: [T, P, 3] - Collocation coordinates
            frames: [T, H, W] - Full Bz frames (for encoder input)
            gt_bz: [T, P, 1] - Ground truth at collocation points
            observed_mask: [T] - Which frames are observed
            labels: [1, n_horizons] - Classification targets
            pil_mask: [H, W] - PIL mask for importance sampling
            mode: "train" or "eval"
        """
        device = coords.device
        original_shape = coords.shape
        T, P = original_shape[0], original_shape[1]
        B = 1
        
        # Encode frames if provided
        if frames is not None and observed_mask is not None:
            L, g = self.encode_frames(frames, observed_mask)
        else:
            # Fallback: zero latents (shouldn't happen in practice)
            H_spatial = 128  # Default spatial size
            L = torch.zeros(1, 32, H_spatial, H_spatial, device=device)
            g = torch.zeros(1, 64, device=device)
        
        # Flatten coords and enable gradients for physics loss
        coords_flat = coords.reshape(-1, 3).contiguous()
        coords_flat.requires_grad_(True)
        
        # Query conditioned field
        # NEW: No more detach! Soft-bilinear sampler supports 2nd-order gradients
        out = self.backbone(coords_flat, L, g, use_nearest=False)
        A_z = out["A_z"]
        B_z = out["B_z"]
        u_x = out["u_x"]
        u_y = out["u_y"]
        eta_raw = out["eta_raw"]
        
        # Optional: compute in-plane field (requires gradients)
        B_x, B_y = None, None
        if self.cfg.loss_weights.curl_consistency > 0:
            # Only compute if we need it for loss AND have gradients
            if A_z.requires_grad and coords_flat.requires_grad:
                B_x, B_y = B_perp_from_Az(A_z, coords_flat)
        
        # Reshape for classifier
        feats_dict = {
            "A_z": A_z.reshape(B, T, P, 1),
            "B_z": B_z.reshape(B, T, P, 1),
            "u_x": u_x.reshape(B, T, P, 1),
            "u_y": u_y.reshape(B, T, P, 1),
        }
        
        # Classifier
        if observed_mask is None:
            observed_mask = torch.ones(T, dtype=torch.bool, device=device)
        
        logits = self.classifier(feats_dict, observed_mask.unsqueeze(0))
        probs = torch.sigmoid(logits)
        
        # ===== Losses (same as pure PINN) =====
        
        # 1. Classification
        loss_cls = torch.tensor(0.0, device=device)
        if mode == "train" and labels is not None:
            label_smoothing = getattr(self.cfg.classifier, 'label_smoothing', 0.0)
            loss_type = self.cfg.classifier.loss_type
            
            if loss_type == "asymmetric":
                # Asymmetric focal loss - particularly good for imbalanced data
                gamma_neg = getattr(self.cfg.classifier, 'asymmetric_gamma_neg', 4.0)
                loss_cls = asymmetric_focal_loss(
                    logits, labels,
                    gamma_pos=0.0,  # No focusing on positives
                    gamma_neg=gamma_neg,
                    clip=0.05
                )
            elif loss_type == "focal":
                if label_smoothing > 0:
                    loss_cls = focal_loss_with_label_smoothing(
                        logits, labels,
                        alpha=self.cfg.classifier.focal_alpha,
                        gamma=self.cfg.classifier.focal_gamma,
                        smoothing=label_smoothing
                    )
                else:
                    loss_cls = focal_loss(
                        logits, labels,  # Pass logits, not probs!
                        alpha=self.cfg.classifier.focal_alpha,
                        gamma=self.cfg.classifier.focal_gamma
                    )
            else:  # bce
                loss_cls = bce_logits(logits, labels, pos_weight=self.cfg.classifier.pos_weight)
            
            # Add confidence penalty for calibration (if enabled)
            conf_penalty_weight = getattr(self.cfg.classifier, 'confidence_penalty', 0.0)
            if conf_penalty_weight > 0:
                loss_cls = loss_cls + confidence_penalty(probs, conf_penalty_weight)
        
        # 2. Data fitting
        loss_data = torch.tensor(0.0, device=device)
        if mode == "train" and gt_bz is not None:
            if observed_mask.dim() == 1:
                mask_expanded = observed_mask[None, :, None, None].expand(B, T, P, 1)
            else:
                mask_expanded = observed_mask[:, :, None, None].expand(B, T, P, 1)
            
            if gt_bz.dim() == 3:
                gt_bz = gt_bz.unsqueeze(0)
            loss_data = l1_data(feats_dict["B_z"], gt_bz, mask_expanded)
        
        # 3. Physics loss
        loss_phys = torch.tensor(0.0, device=device)
        ess_val = 0.0
        lambda_phys = self.get_lambda_phys()
        
        if mode == "train" and lambda_phys > 1e-8:
            alpha = self.get_collocation_alpha()
            N = coords_flat.shape[0]
            
            # Importance weights (same as pure PINN)
            if pil_mask is not None:
                xy_coords = coords_flat[:, :2]
                H_pil, W_pil = pil_mask.shape
                x_px = ((xy_coords[:, 0] + 1.0) * 0.5 * (W_pil - 1)).long().clamp(0, W_pil-1)
                y_px = ((xy_coords[:, 1] + 1.0) * 0.5 * (H_pil - 1)).long().clamp(0, H_pil-1)
                pil_values = torch.from_numpy(pil_mask[y_px.cpu().numpy(), x_px.cpu().numpy()]).to(device).float()
                
                p_uniform = 0.125
                pil_frac = pil_mask.sum() / (H_pil * W_pil) if pil_mask.sum() > 0 else 1.0
                p_pil = (1.0 / (4.0 * pil_frac) * 0.5) if pil_frac > 0 else p_uniform
                p_spatial = alpha * p_pil * pil_values + (1 - alpha) * p_uniform
                p_spatial = p_spatial.clamp_min(1e-8)
                imp_weights, _ = clip_and_renorm_importance(p_spatial.unsqueeze(-1), self.cfg.collocation.impw_clip_quantile)
            else:
                p_uniform = torch.full((N, 1), 0.125, device=device)
                imp_weights, _ = clip_and_renorm_importance(p_uniform, self.cfg.collocation.impw_clip_quantile)
            
            # ESS = (sum(w))^2 / sum(w^2) - measure of effective sample size
            w_sum = imp_weights.sum()
            w_sq_sum = (imp_weights ** 2).sum()
            ess_val = float((w_sum ** 2 / w_sq_sum).item()) if w_sq_sum > 0 else 0.0
            
            # Compute physics residual
            # Soft-bilinear sampler supports 2nd-order gradients
            # Physics loss trains BOTH encoder and decoder end-to-end
            eta_mode = "field" if self.cfg.model.learn_eta else "scalar"
            loss_phys_raw, phys_info = self.physics(
                lambda coords: self.backbone(coords, L, g, use_nearest=False),
                coords_flat,
                imp_weights,
                eta_mode=eta_mode,
                eta_scalar=self.cfg.model.eta_scalar
            )
            loss_phys = lambda_phys * loss_phys_raw
        
        # Total loss
        # NEW: Can now include physics loss directly - no grid_sample issues!
        loss_total = (
            self.cfg.loss_weights.cls * loss_cls +
            self.cfg.loss_weights.data * loss_data +
            loss_phys
        )
        
        return HybridPINNOutput(
            logits=logits,
            probs=probs,
            A_z=A_z,
            B_z=B_z,
            u_x=u_x,
            u_y=u_y,
            B_x=B_x,
            B_y=B_y,
            loss_cls=loss_cls,
            loss_data=loss_data,
            loss_phys=loss_phys,
            loss_total=loss_total,
            ess=ess_val,
            lambda_phys=lambda_phys,
            fourier_alpha=self.backbone.ff._alpha.item()
        )

