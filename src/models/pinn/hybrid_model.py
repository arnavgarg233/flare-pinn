# src/models/pinn/hybrid_model.py
"""
Hybrid CNN-conditioned PINN model for solar flare prediction.

Combines CNN spatial features with physics-informed coordinate fields.
"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from .config import PINNConfig
from .hybrid_core import HybridPINNBackbone
from .core import ClassifierHead, B_perp_from_Az
from .physics import WeakFormInduction2p5D
from .losses import focal_loss, bce_logits, l1_data, interp_schedule
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
        
        # Classifier head (same as pure PINN)
        self.classifier = ClassifierHead(
            hidden=cfg.classifier.hidden,
            dropout=cfg.classifier.dropout,
            horizons=cfg.classifier.horizons
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
        
        # Cache for (L, g) per window
        self._latent_cache = {}
    
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
        
        # Optional: compute in-plane field
        B_x, B_y = None, None
        if self.cfg.loss_weights.curl_consistency > 0 or mode == "eval":
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
            if self.cfg.classifier.loss_type == "focal":
                loss_cls = focal_loss(
                    probs, labels,
                    alpha=self.cfg.classifier.focal_alpha,
                    gamma=self.cfg.classifier.focal_gamma
                )
            else:
                loss_cls = bce_logits(logits, labels, pos_weight=self.cfg.classifier.pos_weight)
        
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
            
            ess_val = float((imp_weights.sum() ** 2) / (imp_weights ** 2).sum().item())
            
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

