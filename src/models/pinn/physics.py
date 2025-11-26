# src/models/pinn/physics.py
"""
Physics-Informed Loss Module for 2.5D MHD Induction Equation.

Implements weak-form residuals for the FULL VECTOR induction equation
with multi-scale test functions for robust physics enforcement.

Mathematical Foundation (2.5D Vector Induction):
=================================================
The resistive MHD induction equation:
    ∂B/∂t = ∇×(u×B) - ∇×(η∇×B)

Under 2.5D assumptions:
- All quantities depend on (x, y, t) only: ∂_z(·) = 0
- B = (Bx, By, Bz) - all three components exist
- u = (ux, uy, 0) - in-plane velocity only

The component equations become:
    ∂Bx/∂t = ∂y(ux·By - uy·Bx) - ∂y[η(∂x·By - ∂y·Bx)]
    ∂By/∂t = -∂x(ux·By - uy·Bx) + ∂x[η(∂x·By - ∂y·Bx)]
    ∂Bz/∂t = -∇⊥·(Bz·u) + ∇⊥·(η∇⊥Bz)

Solenoidal constraint (div B = 0):
    ∂x·Bx + ∂y·By = 0 (since ∂z·Bz = 0)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from pydantic import BaseModel


class VectorPhysicsResidualInfo(BaseModel):
    """Diagnostic information from vector physics residual computation.
    
    Tracks component-wise residuals for detailed physics analysis.
    """
    # Component-wise induction residuals
    loss_induction_Bx: float
    loss_induction_By: float
    loss_induction_Bz: float
    loss_induction_total: float
    
    # Solenoidal constraint (div B = 0)
    loss_divergence_B: float
    
    # Optional velocity divergence (incompressibility)
    loss_divergence_u: float
    
    # Regularization
    loss_tv_eta: float
    
    # Summary statistics
    residual_Bx_mean: float
    residual_By_mean: float
    residual_Bz_mean: float
    residual_Bx_max: float
    residual_By_max: float
    residual_Bz_max: float
    
    class Config:
        """Pydantic config for frozen instances."""
        frozen = True


# Legacy dataclass for backward compatibility
@dataclass
class PhysicsResidualInfo:
    """Legacy diagnostic information (for backward compatibility)."""
    loss_induction: float
    loss_divergence: float
    loss_tv_eta: float
    residual_mean: float
    residual_max: float


class MultiScaleTestFunction(nn.Module):
    """
    Multi-scale test functions for weak-form PDE residuals.
    
    Uses a combination of:
    1. Linear test functions (captures large-scale physics)
    2. Fourier modes (captures multi-scale structure)
    3. Localized bump functions (focuses on active regions)
    
    This significantly improves convergence compared to single random linear.
    """
    def __init__(self, n_fourier_modes: int = 4, n_random: int = 4):
        super().__init__()
        self.n_fourier = n_fourier_modes
        self.n_random = n_random
        
    def forward(
        self, coords: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Generate test functions and their spatial gradients at collocation points.
        
        Args:
            coords: [N, 3] coordinates (x, y, t) in [-1, 1]^3
            
        Returns:
            phis: List of test function values [N, 1]
            dphi_dx: List of x-derivatives [N, 1]
            dphi_dy: List of y-derivatives [N, 1]
        """
        x, y, t = coords[..., 0:1], coords[..., 1:2], coords[..., 2:3]
        device = coords.device
        
        phis = []
        dphi_dx_list = []
        dphi_dy_list = []
        
        # 1. Constant test function (integral form)
        phis.append(torch.ones_like(x))
        dphi_dx_list.append(torch.zeros_like(x))
        dphi_dy_list.append(torch.zeros_like(y))
        
        # 2. Linear test functions
        # φ₁ = x → ∂φ₁/∂x = 1
        phis.append(x)
        dphi_dx_list.append(torch.ones_like(x))
        dphi_dy_list.append(torch.zeros_like(y))
        
        # φ₂ = y → ∂φ₂/∂y = 1
        phis.append(y)
        dphi_dx_list.append(torch.zeros_like(x))
        dphi_dy_list.append(torch.ones_like(y))
        
        # 3. Fourier test functions (multi-scale)
        for k in range(1, self.n_fourier + 1):
            freq = math.pi * k
            
            # sin(k*π*x) modes
            phi_sin_x = torch.sin(freq * x)
            dphi_sin_x = freq * torch.cos(freq * x)
            phis.append(phi_sin_x)
            dphi_dx_list.append(dphi_sin_x)
            dphi_dy_list.append(torch.zeros_like(y))
            
            # sin(k*π*y) modes
            phi_sin_y = torch.sin(freq * y)
            dphi_sin_y = freq * torch.cos(freq * y)
            phis.append(phi_sin_y)
            dphi_dx_list.append(torch.zeros_like(x))
            dphi_dy_list.append(dphi_sin_y)
            
            # Cross terms: sin(k*π*x)*sin(k*π*y)
            phi_cross = torch.sin(freq * x) * torch.sin(freq * y)
            dphi_cross_x = freq * torch.cos(freq * x) * torch.sin(freq * y)
            dphi_cross_y = freq * torch.sin(freq * x) * torch.cos(freq * y)
            phis.append(phi_cross)
            dphi_dx_list.append(dphi_cross_x)
            dphi_dy_list.append(dphi_cross_y)
        
        # 4. Random linear combinations (for variational stability)
        for _ in range(self.n_random):
            with torch.no_grad():
                a = torch.randn(4, device=device)
            # phi = a0 + a1*x + a2*y + a3*t
            # dphi/dx = a1, dphi/dy = a2
            phi_rand = a[0] + a[1] * x + a[2] * y + a[3] * t
            phis.append(phi_rand)
            dphi_dx_list.append(a[1] * torch.ones_like(x))
            dphi_dy_list.append(a[2] * torch.ones_like(y))
        
        return phis, dphi_dx_list, dphi_dy_list


from .config import PhysicsConfig, EtaConfig

class VectorInduction2p5D(nn.Module):
    """
    Weak-form of 2.5D MHD Vector Induction Equation.
    
    Physics equations (component form under ∂_z = 0):
    ================================================
    
    For Bx:
        ∂Bx/∂t = ∂y(ux·By - uy·Bx) - ∂y[η·Jz]
        where Jz = ∂x·By - ∂y·Bx (z-component of current density)
        
    For By:
        ∂By/∂t = -∂x(ux·By - uy·Bx) + ∂x[η·Jz]
        
    For Bz:
        ∂Bz/∂t = -∇⊥·(Bz·u) + ∇⊥·(η·∇⊥Bz)
        
    Weak form (multiply by test φ and integrate by parts):
    ======================================================
    
    Bx equation:
        ∫ φ·∂Bx/∂t dx = -∫ (∂yφ)·(ux·By - uy·Bx) dx + ∫ (∂yφ)·η·Jz dx
        
    By equation:
        ∫ φ·∂By/∂t dx = ∫ (∂xφ)·(ux·By - uy·Bx) dx - ∫ (∂xφ)·η·Jz dx
        
    Bz equation (same as scalar):
        ∫ φ·∂Bz/∂t dx = ∫ ∇⊥φ·(Bz·u) dx - ∫ ∇⊥φ·(η·∇⊥Bz) dx
        
    Solenoidal constraint:
        ∇·B = ∂x·Bx + ∂y·By = 0 (soft penalty)
    
    This formulation:
        1. Requires only first derivatives (not second)
        2. Naturally handles discontinuities
        3. Works well with importance-weighted Monte Carlo integration
        4. Enforces physics for ALL three components of B
    """
    
    def __init__(
        self, 
        physics_cfg: PhysicsConfig,
        eta_cfg: EtaConfig,
        n_fourier_modes: int = 3,
        n_random_tests: int = 2,
        residual_normalization: str = "adaptive",
    ):
        """
        Args:
            physics_cfg: Physics configuration object
            eta_cfg: Resistivity configuration object
            n_fourier_modes: Number of Fourier modes in test functions
            n_random_tests: Number of random linear test functions
            residual_normalization: "fixed", "adaptive", or "per_scale"
        """
        super().__init__()
        self.eta_min = eta_cfg.min
        self.eta_max = eta_cfg.max
        self.tv_eta = eta_cfg.tv_weight
        
        self.use_resistive = physics_cfg.resistive
        self.include_boundary = physics_cfg.boundary_terms
        
        # Vector physics options
        self.enforce_div_free_u = False # Not in PhysicsConfig yet?
        self.enforce_div_free_B = True
        self.div_B_weight = physics_cfg.div_B_weight
        self.component_weights = physics_cfg.component_weights
        
        self.normalization = residual_normalization
        
        # Advanced options
        self.use_uncertainty_weighting = physics_cfg.use_uncertainty_weighting
        self.enforce_force_free = physics_cfg.enforce_force_free
        self.force_free_weight = physics_cfg.force_free_weight
        
        self.use_causal_weighting = physics_cfg.use_causal_weighting
        self.causal_tol = physics_cfg.causal_tol
        
        self.enable_gradient_clamping = physics_cfg.enable_gradient_clamping
        self.gradient_clamp_value = physics_cfg.gradient_clamp_value
        
        self.test_fn = MultiScaleTestFunction(n_fourier_modes, n_random_tests)
        
        # Learnable scaling factors for each loss component
        self.register_buffer("_loss_scale", torch.tensor(1.0))
        
        # Running estimates for normalization (more stable than per-batch percentiles)
        self.register_buffer("_B_scale_ema", torch.tensor(1.0))
        self.register_buffer("_u_scale_ema", torch.tensor(0.1))
        self._ema_momentum = 0.99
        
        # SOTA: Learned uncertainty weights for automatic loss balancing (Kendall et al. 2018)
        # log(σ²) for each component - initialized to balance roughly equal losses
        if self.use_uncertainty_weighting:
            self.log_var_Bx = nn.Parameter(torch.tensor(0.0))
            self.log_var_By = nn.Parameter(torch.tensor(0.0))
            self.log_var_Bz = nn.Parameter(torch.tensor(0.0))
            self.log_var_div = nn.Parameter(torch.tensor(-1.0))  # Higher precision for div constraint
        
        # Per-component EMA for adaptive scaling (more granular than combined)
        self.register_buffer("_Bx_scale_ema", torch.tensor(1.0))
        self.register_buffer("_By_scale_ema", torch.tensor(1.0))
        self.register_buffer("_Bz_scale_ema", torch.tensor(1.0))
    
    def _compute_spatial_gradients(
        self,
        field: torch.Tensor,
        coords: torch.Tensor,
        field_name: str = "field"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute spatial and temporal gradients of a field.
        
        Args:
            field: [N, 1] field values
            coords: [N, 3] coordinates with requires_grad=True
            field_name: Name for error messages
            
        Returns:
            df_dx, df_dy, df_dt: [N, 1] gradient components
            
        Note:
            Uses retain_graph=True since we need multiple gradient computations.
            The graph will be freed when the physics loss backward pass completes.
        """
        # Check if gradient computation is possible
        if not coords.requires_grad:
            zeros = torch.zeros_like(field)
            return zeros, zeros, zeros
        
        ones = torch.ones_like(field)
        try:
            grads = torch.autograd.grad(
                field, coords,
                grad_outputs=ones,
                create_graph=self.training,  # Only create graph during training
                retain_graph=True,
                only_inputs=True
            )[0]
        except RuntimeError as e:
            import warnings
            warnings.warn(f"Gradient computation failed for {field_name}: {e}")
            zeros = torch.zeros_like(field)
            return zeros, zeros, zeros
        
        # Soft clamp gradients to prevent explosion while preserving gradient flow
        # Using tanh-based soft clamp instead of hard clamp for better backprop
        if self.enable_gradient_clamping:
            scale = self.gradient_clamp_value
            grads = scale * torch.tanh(grads / scale)
        
        # Extract components (view operation, no memory overhead)
        df_dx = grads[..., 0:1]
        df_dy = grads[..., 1:2]
        df_dt = grads[..., 2:3]
        
        return df_dx, df_dy, df_dt
    
    def _compute_residual_Bz(
        self,
        Bz: torch.Tensor,
        ux: torch.Tensor,
        uy: torch.Tensor,
        eta: torch.Tensor,
        dBz_dx: torch.Tensor,
        dBz_dy: torch.Tensor,
        dBz_dt: torch.Tensor,
        phis: list[torch.Tensor],
        dphi_dx_list: list[torch.Tensor],
        dphi_dy_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute weak-form residual for Bz component.
        
        Equation: ∂Bz/∂t = -∇⊥·(Bz·u) + ∇⊥·(η·∇⊥Bz)
        
        Weak form:
            ∫ φ·∂Bz/∂t dx = ∫ ∇⊥φ·(Bz·u) dx - ∫ ∇⊥φ·(η·∇⊥Bz) dx
        """
        residuals = []
        
        for phi, dphi_dx, dphi_dy in zip(phis, dphi_dx_list, dphi_dy_list):
            # Time derivative term: ∫ φ·∂Bz/∂t
            term_time = phi * dBz_dt
            
            # Transport term (after integration by parts): ∫ ∇⊥φ·(Bz·u)
            # = ∫ (∂φ/∂x·Bz·ux + ∂φ/∂y·Bz·uy)
            term_transport = dphi_dx * (Bz * ux) + dphi_dy * (Bz * uy)
            
            # Resistive diffusion term: -∫ ∇⊥φ·(η·∇⊥Bz)
            # = -∫ (∂φ/∂x·η·∂Bz/∂x + ∂φ/∂y·η·∂Bz/∂y)
            if self.use_resistive:
                term_resistive = -(dphi_dx * (eta * dBz_dx) + dphi_dy * (eta * dBz_dy))
            else:
                term_resistive = torch.zeros_like(term_time)
            
            # Weak form residual: ∫φ·∂Bz/∂t - ∫∇φ·(Bz·u) + ∫∇φ·(η∇Bz) = 0
            # term_resistive already has negative sign from the IBP
            # So: term_time - term_transport - term_resistive = 0
            residual = term_time - term_transport - term_resistive
            residuals.append(residual)
        
        return torch.cat(residuals, dim=-1)  # [N, n_tests]
    
    def _compute_residual_Bx(
        self,
        dBx_dt: torch.Tensor,
        F_conv: torch.Tensor,
        Jz: torch.Tensor,
        eta: torch.Tensor,
        phis: list[torch.Tensor],
        dphi_dx_list: list[torch.Tensor],
        dphi_dy_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute weak-form residual for Bx component.
        
        Equation: ∂Bx/∂t = ∂y(ux·By - uy·Bx) - ∂y[η·Jz]
        where Jz = ∂x·By - ∂y·Bx, F_conv = ux·By - uy·Bx
        
        Weak form (after integration by parts):
            ∫φ·∂Bx/∂t + ∫(∂yφ)·F_conv - ∫(∂yφ)·η·Jz = 0
        
        Args:
            dBx_dt: Time derivative of Bx
            F_conv: Pre-computed convective flux (ux·By - uy·Bx)
            Jz: Pre-computed current density z-component
            eta: Resistivity field
        """
        residuals = []
        
        for phi, dphi_dx, dphi_dy in zip(phis, dphi_dx_list, dphi_dy_list):
            # Time derivative term: ∫ φ·∂Bx/∂t
            term_time = phi * dBx_dt
            
            # Advection term (after IBP): ∫(∂yφ)·F_conv
            term_advection = dphi_dy * F_conv
            
            # Resistive diffusion term (after IBP): ∫(∂yφ)·η·Jz
            if self.use_resistive:
                term_resistive = dphi_dy * (eta * Jz)
            else:
                term_resistive = torch.zeros_like(term_time)
            
            # Weak form residual: ∫φ·∂Bx/∂t + ∫(∂yφ)·F_conv - ∫(∂yφ)·η·Jz = 0
            residual = term_time + term_advection - term_resistive
            residuals.append(residual)
        
        return torch.cat(residuals, dim=-1)  # [N, n_tests]
    
    def _compute_residual_By(
        self,
        dBy_dt: torch.Tensor,
        F_conv: torch.Tensor,
        Jz: torch.Tensor,
        eta: torch.Tensor,
        phis: list[torch.Tensor],
        dphi_dx_list: list[torch.Tensor],
        dphi_dy_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute weak-form residual for By component.
        
        Equation: ∂By/∂t = -∂x(ux·By - uy·Bx) + ∂x[η·Jz]
        where Jz = ∂x·By - ∂y·Bx, F_conv = ux·By - uy·Bx
        
        Weak form (after integration by parts):
            ∫φ·∂By/∂t - ∫(∂xφ)·F_conv + ∫(∂xφ)·η·Jz = 0
        
        Args:
            dBy_dt: Time derivative of By
            F_conv: Pre-computed convective flux (ux·By - uy·Bx)
            Jz: Pre-computed current density z-component
            eta: Resistivity field
        """
        residuals = []
        
        for phi, dphi_dx, dphi_dy in zip(phis, dphi_dx_list, dphi_dy_list):
            # Time derivative term: ∫ φ·∂By/∂t
            term_time = phi * dBy_dt
            
            # Advection term (after IBP): ∫(∂xφ)·F_conv
            term_advection = dphi_dx * F_conv
            
            # Resistive diffusion term (after IBP): ∫(∂xφ)·η·Jz
            if self.use_resistive:
                term_resistive = dphi_dx * (eta * Jz)
            else:
                term_resistive = torch.zeros_like(term_time)
            
            # Weak form residual: ∫φ·∂By/∂t - ∫(∂xφ)·F_conv + ∫(∂xφ)·η·Jz = 0
            residual = term_time - term_advection + term_resistive
            residuals.append(residual)
        
        return torch.cat(residuals, dim=-1)  # [N, n_tests]
    
    def _apply_causal_weighting(self, residuals: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply causal weighting to residuals (Wang et al. 2022).
        
        Weights residuals at time t based on cumulative loss at time < t.
        w(t) = exp(-epsilon * sum_{tau < t} L(tau))
        
        Args:
            residuals: [N, 1] or [N, C] residuals
            coords: [N, 3] coordinates (t is index 2)
            
        Returns:
            Weighted residuals
        """
        if not self.use_causal_weighting:
            return residuals
            
        # Extract time coordinate
        t = coords[..., 2]
        
        # Sort by time
        t_sorted, indices = torch.sort(t)
        
        # Compute cumulative loss (L1 norm of residuals)
        # We use the mean of absolute residuals at each time step?
        # Since coordinates are continuous, we can just use cumulative sum on sorted residuals
        
        # Gather residuals in time order
        res_sorted = residuals[indices]
        res_abs = res_sorted.abs().mean(dim=-1) # [N]
        
        # Cumulative sum of losses
        # We want to penalize current time if previous times are not solved
        # weight_i = exp(-tol * sum_{j<i} loss_j)
        # This forces the model to solve early times first
        
        # To make this stable, we bucket by time or use a running sum
        cumulative_loss = torch.cumsum(res_abs, dim=0)
        
        # Offset: w[i] depends on sum_{j<i}, so shift
        cumulative_loss = torch.roll(cumulative_loss, 1, 0)
        cumulative_loss[0] = 0.0
        
        # Compute weights
        # Normalize cumulative loss to avoid vanishing gradients too fast?
        # Wang et al. use a specific epsilon. Here we use causal_tol
        weights = torch.exp(-self.causal_tol * cumulative_loss)
        
        # Apply weights (shape [N, 1])
        weights = weights.unsqueeze(-1)
        res_weighted = res_sorted * weights
        
        # Scatter back to original order? 
        # Actually, since we are summing the loss anyway, we can just return the weighted sorted residuals
        # But for consistency with other returns (like info), we might want original order.
        # The loss computation takes mean(). sum(res_weighted^2) is the same sorted or unsorted.
        # BUT, we return 'info' with max/mean stats. Those should be on the weighted or unweighted?
        # Usually we want to minimize Weighted Loss.
        
        # Let's unsort to preserve correspondence with inputs
        res_unsorted = torch.empty_like(residuals)
        res_unsorted.scatter_(0, indices.unsqueeze(-1).expand_as(residuals), res_weighted)
        
        return res_unsorted

    def _normalize_residuals(
        self,
        residuals: torch.Tensor,
        B_abs: torch.Tensor,
        u_abs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive normalization to residuals.
        
        Uses exponential moving average of field scales for stability,
        avoiding noisy per-batch percentile estimates.
        
        FIX: Properly handle first iteration by using batch stats directly
        when EMA hasn't been initialized yet.
        """
        if self.normalization == "adaptive":
            # Compute batch statistics
            with torch.no_grad():
                B_q75 = torch.quantile(B_abs.flatten(), 0.75).clamp(min=0.05)
                u_q75 = torch.quantile(u_abs.flatten(), 0.75).clamp(min=0.02)
                
                # Update EMA if training (more stable than raw percentiles)
                if self.training:
                    # FIX: Check if this is first iteration (EMA still at initial value)
                    # Use higher momentum on first few updates for faster warmup
                    is_initial = (self._B_scale_ema == 1.0) and (B_q75 != 1.0)
                    if is_initial:
                        # First update: initialize directly to batch stats
                        self._B_scale_ema.copy_(B_q75)
                        self._u_scale_ema.copy_(u_q75)
                    else:
                        self._B_scale_ema.lerp_(B_q75, 1 - self._ema_momentum)
                        self._u_scale_ema.lerp_(u_q75, 1 - self._ema_momentum)
                
                # Use EMA for normalization
                B_scale = self._B_scale_ema.clamp(min=0.05)
                u_scale = self._u_scale_ema.clamp(min=0.02)
            
            norm_factor = (B_scale * u_scale).sqrt().clamp(min=0.1, max=10.0)
            return residuals / norm_factor
        elif self.normalization == "per_scale":
            with torch.no_grad():
                residual_scales = residuals.abs().mean(dim=0, keepdim=True).clamp(min=1e-8)
            return residuals / residual_scales
        return residuals  # "fixed": no normalization
    
    def forward(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        imp_weights: torch.Tensor,
        eta_mode: str = "scalar",
        eta_scalar: float = 0.01
    ) -> tuple[torch.Tensor, VectorPhysicsResidualInfo]:
        """
        Compute physics loss from weak-form residuals for full vector B field.
        
        Args:
            model: Neural field model that maps coords -> {"B": [N,3], "u": [N,2], "eta_raw": [N,1]}
                   B = (Bx, By, Bz), u = (ux, uy)
            coords: [N, 3] collocation points with requires_grad=True
            imp_weights: [N, 1] importance weights (normalized, mean=1)
            eta_mode: "scalar" (fixed) or "field" (learned spatially-varying)
            eta_scalar: Value of eta when eta_mode="scalar"
            
        Returns:
            loss: Scalar physics loss
            info: VectorPhysicsResidualInfo with component-wise diagnostics
        """
        # Forward through model to get fields
        out = model(coords)
        
        # Handle both old (scalar) and new (vector) output formats
        # PINNBackboneOutput supports dict-style access or attribute access
        if hasattr(out, "B") and out.B is not None:
            # New vector format: B is [N, 3]
            B = out.B
            Bx, By, Bz = B[..., 0:1], B[..., 1:2], B[..., 2:3]
        elif "B" in out and out["B"] is not None:
             # Legacy dict format
            B = out["B"]
            Bx, By, Bz = B[..., 0:1], B[..., 1:2], B[..., 2:3]
        else:
            # Legacy scalar format: only B_z available
            # Handle object or dict access
            Bz = getattr(out, "B_z", None)
            if Bz is None: Bz = out.get("B_z")
            
            Bx = getattr(out, "B_x", None)
            if Bx is None: Bx = out.get("B_x")
            
            By = getattr(out, "B_y", None)
            if By is None: By = out.get("B_y")
            
            # If Bx, By not available, fall back to scalar-only physics
            if Bx is None or By is None:
                return self._forward_scalar_fallback(
                    model, coords, imp_weights, out, eta_mode, eta_scalar
                )
        
        # Get velocity components
        if hasattr(out, "u") and out.u is not None:
            # New vector format: u is [N, 2]
            u = out.u
            ux, uy = u[..., 0:1], u[..., 1:2]
        elif "u" in out and out["u"] is not None:
            u = out["u"]
            ux, uy = u[..., 0:1], u[..., 1:2]
        else:
            # Legacy format
            ux = getattr(out, "u_x", None)
            if ux is None: ux = out["u_x"]
            
            uy = getattr(out, "u_y", None)
            if uy is None: uy = out["u_y"]
        
        # Get resistivity
        eta_raw = getattr(out, "eta_raw", None)
        if eta_raw is None and isinstance(out, dict):
             eta_raw = out.get("eta_raw")
             
        if eta_raw is not None and eta_mode == "field":
            eta = torch.sigmoid(eta_raw)
            eta = self.eta_min + (self.eta_max - self.eta_min) * eta
            eta_l2_reg = self.tv_eta * (eta ** 2).mean() if self.tv_eta > 0 else Bz.new_tensor(0.0)
            tv_reg = eta_l2_reg
        else:
            eta = Bz.new_full(Bz.shape, float(eta_scalar))
            tv_reg = Bz.new_tensor(0.0)
        
        # Compute all required gradients
        dBx_dx, dBx_dy, dBx_dt = self._compute_spatial_gradients(Bx, coords, "Bx")
        dBy_dx, dBy_dy, dBy_dt = self._compute_spatial_gradients(By, coords, "By")
        dBz_dx, dBz_dy, dBz_dt = self._compute_spatial_gradients(Bz, coords, "Bz")
        
        # Check for gradient computation failures
        if (dBx_dt.abs().max() == 0 and dBy_dt.abs().max() == 0 and dBz_dt.abs().max() == 0):
            # All gradients are zero - likely a computation failure
            zero_loss = Bz.new_tensor(0.0, requires_grad=True)
            info = VectorPhysicsResidualInfo(
                loss_induction_Bx=0.0, loss_induction_By=0.0, loss_induction_Bz=0.0,
                loss_induction_total=0.0, loss_divergence_B=0.0, loss_divergence_u=0.0,
                loss_tv_eta=0.0, residual_Bx_mean=0.0, residual_By_mean=0.0,
                residual_Bz_mean=0.0, residual_Bx_max=0.0, residual_By_max=0.0,
                residual_Bz_max=0.0
            )
            return zero_loss, info
        
        # Get test functions
        phis, dphi_dx_list, dphi_dy_list = self.test_fn(coords)
        
        # Pre-compute shared terms (optimization: avoid redundant computation)
        # Current density z-component: Jz = ∂x·By - ∂y·Bx
        Jz = dBy_dx - dBx_dy
        
        # Convective flux: F_conv = ux·By - uy·Bx (shared by Bx and By equations)
        F_conv = ux * By - uy * Bx
        
        # Compute residuals for each component
        residuals_Bx = self._compute_residual_Bx(
            dBx_dt, F_conv, Jz, eta,
            phis, dphi_dx_list, dphi_dy_list
        )
        residuals_By = self._compute_residual_By(
            dBy_dt, F_conv, Jz, eta,
            phis, dphi_dx_list, dphi_dy_list
        )
        residuals_Bz = self._compute_residual_Bz(
            Bz, ux, uy, eta, dBz_dx, dBz_dy, dBz_dt,
            phis, dphi_dx_list, dphi_dy_list
        )
        
        # Normalize residuals
        B_abs = (Bx.abs() + By.abs() + Bz.abs()) / 3.0
        u_abs = (ux.abs() + uy.abs()) / 2.0
        
        residuals_Bx = self._normalize_residuals(residuals_Bx, B_abs, u_abs)
        residuals_By = self._normalize_residuals(residuals_By, B_abs, u_abs)
        residuals_Bz = self._normalize_residuals(residuals_Bz, B_abs, u_abs)
        
        # Apply causal weighting if enabled
        if self.use_causal_weighting:
            residuals_Bx = self._apply_causal_weighting(residuals_Bx, coords)
            residuals_By = self._apply_causal_weighting(residuals_By, coords)
            residuals_Bz = self._apply_causal_weighting(residuals_Bz, coords)
        
        # Clamp and handle NaN
        def safe_residual(r: torch.Tensor) -> torch.Tensor:
            r = r.clamp(-10.0, 10.0)
            return torch.nan_to_num(r, nan=0.0, posinf=1.0, neginf=-1.0)
        
        residuals_Bx = safe_residual(residuals_Bx)
        residuals_By = safe_residual(residuals_By)
        residuals_Bz = safe_residual(residuals_Bz)
        
        # Compute component losses (importance-weighted MSE)
        imp_weights_safe = torch.nan_to_num(imp_weights, nan=1.0, posinf=1.0, neginf=1.0)
        
        loss_Bx = ((residuals_Bx ** 2).mean(dim=-1, keepdim=True) * imp_weights_safe).mean()
        loss_By = ((residuals_By ** 2).mean(dim=-1, keepdim=True) * imp_weights_safe).mean()
        loss_Bz = ((residuals_Bz ** 2).mean(dim=-1, keepdim=True) * imp_weights_safe).mean()
        
        # Weighted combination of component losses
        w_Bx, w_By, w_Bz = self.component_weights
        
        if self.use_uncertainty_weighting and hasattr(self, 'log_var_Bx'):
            # Learned uncertainty weighting (Kendall et al. 2018)
            # L = (1/2σ²) * loss + log(σ) = (1/2) * exp(-log_var) * loss + 0.5 * log_var
            precision_Bx = torch.exp(-self.log_var_Bx)
            precision_By = torch.exp(-self.log_var_By)
            precision_Bz = torch.exp(-self.log_var_Bz)
            
            loss_induction = (
                0.5 * precision_Bx * loss_Bx + 0.5 * self.log_var_Bx +
                0.5 * precision_By * loss_By + 0.5 * self.log_var_By +
                0.5 * precision_Bz * loss_Bz + 0.5 * self.log_var_Bz
            )
        else:
            # Fixed weights with robust normalization
            # Use sqrt for scale compression instead of log1p (preserves gradient flow)
            # sqrt(x) has derivative 1/(2*sqrt(x)) which stays bounded for small x
            # and provides natural scale compression for large x
            eps = 1e-8
            loss_induction = (
                w_Bx * (loss_Bx + eps).sqrt() + 
                w_By * (loss_By + eps).sqrt() + 
                w_Bz * (loss_Bz + eps).sqrt()
            )
        
        # Solenoidal constraint: ∇·B = ∂x·Bx + ∂y·By = 0
        loss_div_B = Bz.new_tensor(0.0)
        if self.enforce_div_free_B:
            div_B = dBx_dx + dBy_dy
            loss_div_B = self.div_B_weight * ((div_B ** 2) * imp_weights_safe).mean()
        
        # Optional: Divergence-free velocity constraint
        loss_div_u = Bz.new_tensor(0.0)
        if self.enforce_div_free_u:
            dux_dx, _, _ = self._compute_spatial_gradients(ux, coords, "ux")
            _, duy_dy, _ = self._compute_spatial_gradients(uy, coords, "uy")
            div_u = dux_dx + duy_dy
            loss_div_u = ((div_u ** 2) * imp_weights_safe).mean()
        
        # SOTA: Force-free constraint (J × B ≈ 0 in low-β corona)
        # This is critical for flare prediction as non-force-free regions indicate
        # magnetic stress that can trigger reconnection.
        # J × B = (Jx*By - Jy*Bx, Jy*Bz - Jz*By, Jz*Bx - Jx*Bz)
        # In 2.5D: Jx = ∂y·Bz, Jy = -∂x·Bz, Jz = ∂x·By - ∂y·Bx
        # Force-free means: α*B = J where α is constant (or slowly varying)
        # Soft constraint: minimize |J × B|² / |B|²
        loss_force_free = Bz.new_tensor(0.0)
        if self.enforce_force_free:
            Jx = dBz_dy
            Jy = -dBz_dx
            # Jz already computed as current density
            
            # Cross product components (J × B)
            JxB_x = Jx * By - Jy * Bx
            JxB_y = Jy * Bz - Jz * By  
            JxB_z = Jz * Bx - Jx * Bz
            
            JxB_sq = JxB_x**2 + JxB_y**2 + JxB_z**2
            B_sq = (Bx**2 + By**2 + Bz**2).clamp(min=1e-8)
            
            # Normalized force-free violation
            ff_violation = JxB_sq / B_sq
            loss_force_free = self.force_free_weight * (ff_violation * imp_weights_safe).mean()
        
        # Total physics loss
        total_loss = loss_induction + loss_div_B + 0.1 * loss_div_u + loss_force_free + tv_reg
        
        # Create diagnostic info
        with torch.no_grad():
            info = VectorPhysicsResidualInfo(
                loss_induction_Bx=float(loss_Bx.detach()),
                loss_induction_By=float(loss_By.detach()),
                loss_induction_Bz=float(loss_Bz.detach()),
                loss_induction_total=float(loss_induction.detach()),
                loss_divergence_B=float(loss_div_B.detach()),
                loss_divergence_u=float(loss_div_u.detach()),
                loss_tv_eta=float(tv_reg.detach()),
                residual_Bx_mean=float(residuals_Bx.abs().mean().detach()),
                residual_By_mean=float(residuals_By.abs().mean().detach()),
                residual_Bz_mean=float(residuals_Bz.abs().mean().detach()),
                residual_Bx_max=float(residuals_Bx.abs().max().detach()),
                residual_By_max=float(residuals_By.abs().max().detach()),
                residual_Bz_max=float(residuals_Bz.abs().max().detach()),
            )
        
        return total_loss, info
    
    def _forward_scalar_fallback(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        imp_weights: torch.Tensor,
        out: dict,
        eta_mode: str,
        eta_scalar: float
    ) -> tuple[torch.Tensor, VectorPhysicsResidualInfo]:
        """
        Fallback to scalar Bz-only physics when Bx, By not available.
        
        This maintains backward compatibility with models that only output Bz.
        """
        Bz = out["B_z"]
        ux = out["u_x"]
        uy = out["u_y"]
        
        # Get resistivity
        if out.get("eta_raw") is not None and eta_mode == "field":
            eta = torch.sigmoid(out["eta_raw"])
            eta = self.eta_min + (self.eta_max - self.eta_min) * eta
            tv_reg = self.tv_eta * (eta ** 2).mean() if self.tv_eta > 0 else Bz.new_tensor(0.0)
        else:
            eta = Bz.new_full(Bz.shape, float(eta_scalar))
            tv_reg = Bz.new_tensor(0.0)
        
        # Compute gradients
        dBz_dx, dBz_dy, dBz_dt = self._compute_spatial_gradients(Bz, coords, "Bz")
        
        # Get test functions
        phis, dphi_dx_list, dphi_dy_list = self.test_fn(coords)
        
        # Compute Bz residuals only
        residuals_Bz = self._compute_residual_Bz(
            Bz, ux, uy, eta, dBz_dx, dBz_dy, dBz_dt,
            phis, dphi_dx_list, dphi_dy_list
        )
        
        # Normalize
        B_abs = Bz.abs()
        u_abs = (ux.abs() + uy.abs()) / 2.0
        residuals_Bz = self._normalize_residuals(residuals_Bz, B_abs, u_abs)
        residuals_Bz = torch.nan_to_num(residuals_Bz.clamp(-10.0, 10.0), nan=0.0)
        
        # Loss
        imp_weights_safe = torch.nan_to_num(imp_weights, nan=1.0, posinf=1.0, neginf=1.0)
        loss_Bz = ((residuals_Bz ** 2).mean(dim=-1, keepdim=True) * imp_weights_safe).mean()
        
        total_loss = loss_Bz + tv_reg
        
        with torch.no_grad():
            info = VectorPhysicsResidualInfo(
                loss_induction_Bx=0.0,
                loss_induction_By=0.0,
                loss_induction_Bz=float(loss_Bz.detach()),
                loss_induction_total=float(loss_Bz.detach()),
                loss_divergence_B=0.0,
                loss_divergence_u=0.0,
                loss_tv_eta=float(tv_reg.detach()),
                residual_Bx_mean=0.0,
                residual_By_mean=0.0,
                residual_Bz_mean=float(residuals_Bz.abs().mean().detach()),
                residual_Bx_max=0.0,
                residual_By_max=0.0,
                residual_Bz_max=float(residuals_Bz.abs().max().detach()),
            )
        
        return total_loss, info


# Backward compatibility alias
class WeakFormInduction2p5D(VectorInduction2p5D):
    """
    Legacy alias for backward compatibility.
    
    This class wraps VectorInduction2p5D but returns the legacy
    PhysicsResidualInfo format for existing training code.
    """
    
    def forward(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        imp_weights: torch.Tensor,
        eta_mode: str = "scalar",
        eta_scalar: float = 0.01
    ) -> tuple[torch.Tensor, PhysicsResidualInfo]:
        """Forward with legacy return type."""
        loss, vector_info = super().forward(model, coords, imp_weights, eta_mode, eta_scalar)
        
        # Convert to legacy format
        legacy_info = PhysicsResidualInfo(
            loss_induction=vector_info.loss_induction_total,
            loss_divergence=vector_info.loss_divergence_B + vector_info.loss_divergence_u,
            loss_tv_eta=vector_info.loss_tv_eta,
            residual_mean=(
                vector_info.residual_Bx_mean + 
                vector_info.residual_By_mean + 
                vector_info.residual_Bz_mean
            ) / 3.0,
            residual_max=max(
                vector_info.residual_Bx_max,
                vector_info.residual_By_max,
                vector_info.residual_Bz_max
            ),
        )
        
        return loss, legacy_info


class FreeEnergyProxy(nn.Module):
    """
    Compute free magnetic energy proxy from PINN outputs.
    
    Free energy E_free ∝ ∫ (B - B_potential)² dV
    
    This is a key predictor of flare potential that can be computed
    from the PINN's magnetic field representation.
    
    Enhanced for vector field inputs.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        B_z: torch.Tensor,
        B_x: Optional[torch.Tensor],
        B_y: Optional[torch.Tensor],
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute free energy proxy.
        
        For PINN outputs, we can approximate this by measuring
        the non-potentiality of the field configuration.
        
        Args:
            B_z: [N, 1] vertical field
            B_x, B_y: [N, 1] horizontal field components (optional)
            coords: [N, 3] coordinates
            
        Returns:
            free_energy: [1] scalar proxy value
        """
        if B_x is None or B_y is None:
            # Fallback: use Bz variance as proxy for complexity
            return B_z.var()
        
        # Compute |∇Bz| as proxy for shear
        grads = torch.autograd.grad(
            B_z, coords,
            grad_outputs=torch.ones_like(B_z),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad_Bz_mag = (grads[..., 0:1]**2 + grads[..., 1:2]**2).sqrt()
        
        # Current helicity proxy: Jz * Bz where Jz = dBy/dx - dBx/dy
        if coords.requires_grad and B_x.requires_grad and B_y.requires_grad:
            grads_Bx = torch.autograd.grad(
                B_x, coords,
                grad_outputs=torch.ones_like(B_x),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            grads_By = torch.autograd.grad(
                B_y, coords,
                grad_outputs=torch.ones_like(B_y),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            Jz = grads_By[..., 0:1] - grads_Bx[..., 1:2]
            helicity_proxy = (Jz * B_z).abs().mean()
            
            # Combined metric
            return grad_Bz_mag.mean() + 0.5 * helicity_proxy
        
        return grad_Bz_mag.mean()


class CurrentHelicityProxy(nn.Module):
    """
    Compute current helicity proxy from PINN outputs.
    
    Current helicity H_c = ∫ B · J dV
    
    where J = ∇×B is the current density.
    This measures the twist and complexity of the magnetic field.
    
    Enhanced for vector field inputs.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        A_z: torch.Tensor,
        B_z: torch.Tensor,
        coords: torch.Tensor,
        B_x: Optional[torch.Tensor] = None,
        B_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute current helicity proxy.
        
        For 2.5D, J = ∇×B has components:
            Jx = ∂y Bz (since ∂z By = 0)
            Jy = -∂x Bz (since ∂z Bx = 0)  
            Jz = ∂x By - ∂y Bx
            
        Helicity ≈ Bz * Jz + Bx * Jx + By * Jy
        
        Args:
            A_z: [N, 1] vector potential
            B_z: [N, 1] vertical field
            coords: [N, 3] coordinates
            B_x, B_y: [N, 1] horizontal field components (optional)
            
        Returns:
            helicity: [1] scalar proxy value
        """
        # Compute ∇Bz for Jx, Jy
        grad_Bz = torch.autograd.grad(
            B_z, coords,
            grad_outputs=torch.ones_like(B_z),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        Jx = grad_Bz[..., 1:2]   # ∂y Bz
        Jy = -grad_Bz[..., 0:1]  # -∂x Bz
        
        if B_x is not None and B_y is not None:
            # Full helicity: B · J = Bx*Jx + By*Jy + Bz*Jz
            
            # Compute Jz = ∂x By - ∂y Bx
            grad_Bx = torch.autograd.grad(
                B_x, coords,
                grad_outputs=torch.ones_like(B_x),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            grad_By = torch.autograd.grad(
                B_y, coords,
                grad_outputs=torch.ones_like(B_y),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            Jz = grad_By[..., 0:1] - grad_Bx[..., 1:2]
            
            helicity = (B_x * Jx + B_y * Jy + B_z * Jz).abs().mean()
        else:
            # Fallback: use |Bz * Jz| where Jz ≈ ∇²Az
            # Compute Laplacian of Az
            grad_Az = torch.autograd.grad(
                A_z, coords,
                grad_outputs=torch.ones_like(A_z),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            dAz_dx = grad_Az[..., 0:1]
            dAz_dy = grad_Az[..., 1:2]
            
            d2Az_dx2 = torch.autograd.grad(
                dAz_dx.sum(), coords,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0][..., 0:1]
            
            d2Az_dy2 = torch.autograd.grad(
                dAz_dy.sum(), coords,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0][..., 1:2]
            
            Jz = d2Az_dx2 + d2Az_dy2
            helicity = (B_z * Jz).abs().mean()
        
        return helicity
