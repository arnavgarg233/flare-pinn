# src/models/pinn/physics.py
"""
Physics-Informed Loss Module for 2.5D MHD Induction Equation.

Implements weak-form residuals with multi-scale test functions for
robust physics enforcement. Designed to beat SOTA by properly
incorporating magnetic field evolution constraints.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class PhysicsResidualInfo:
    """Diagnostic information from physics residual computation."""
    loss_induction: float
    loss_divergence: float
    loss_tv_eta: float
    residual_mean: float
    residual_max: float


def _tv1_1d(x: torch.Tensor, weight: float) -> torch.Tensor:
    """Total variation regularization for spatially-varying eta."""
    if weight <= 0:
        return x.new_tensor(0.0)
    if x.numel() < 2:
        return x.new_tensor(0.0)
    return weight * (x[1:] - x[:-1]).abs().mean()


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
        
    def forward(self, coords: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
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
        # These add stochastic test functions to improve convergence
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


class WeakFormInduction2p5D(nn.Module):
    """
    Weak-form of 2.5D MHD induction equation with multi-scale test functions.
    
    Physics equation:
        ∂Bz/∂t = -∇⊥·(Bz u) + ∇⊥·(η ∇⊥Bz)
        
    Where:
        - Bz: out-of-plane magnetic field
        - u = (ux, uy): in-plane velocity field
        - η: magnetic diffusivity (resistivity)
        
    Weak form (multiply by test φ and integrate by parts):
        ∫ φ ∂Bz/∂t dx = ∫ ∇⊥φ · (Bz u) dx - ∫ ∇⊥φ · (η ∇⊥Bz) dx + boundary
        
    This formulation:
        1. Requires only first derivatives (not second)
        2. Naturally handles discontinuities
        3. Works well with importance-weighted Monte Carlo integration
        
    Key improvements over baseline:
        - Multi-scale test functions for better convergence
        - Adaptive residual scaling based on field magnitudes
        - Optional divergence-free velocity constraint
        - Proper normalization to balance with classification loss
    """
    
    def __init__(
        self, 
        eta_bounds: tuple[float, float] = (1e-4, 1.0),
        use_resistive: bool = True,
        include_boundary: bool = False,
        tv_eta: float = 1e-3,
        n_fourier_modes: int = 3,
        n_random_tests: int = 2,
        enforce_div_free_u: bool = False,
        residual_normalization: str = "adaptive"
    ):
        """
        Args:
            eta_bounds: (min, max) for learned resistivity
            use_resistive: Include resistive diffusion term
            include_boundary: Include boundary integral terms (experimental)
            tv_eta: Total variation weight for eta regularization
            n_fourier_modes: Number of Fourier modes in test functions
            n_random_tests: Number of random linear test functions
            enforce_div_free_u: Add penalty for ∇·u ≠ 0
            residual_normalization: "fixed", "adaptive", or "per_scale"
        """
        super().__init__()
        self.eta_min, self.eta_max = eta_bounds
        self.use_resistive = use_resistive
        self.include_boundary = include_boundary
        self.tv_eta = tv_eta
        self.enforce_div_free_u = enforce_div_free_u
        self.normalization = residual_normalization
        
        self.test_fn = MultiScaleTestFunction(n_fourier_modes, n_random_tests)
        
        # Learnable scaling factors for each loss component
        self.register_buffer("_loss_scale", torch.tensor(1.0))
        
    def forward(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        imp_weights: torch.Tensor,
        eta_mode: str = "scalar",
        eta_scalar: float = 0.01
    ) -> tuple[torch.Tensor, PhysicsResidualInfo]:
        """
        Compute physics loss from weak-form residuals.
        
        Args:
            model: Neural field model that maps coords -> {B_z, u_x, u_y, eta_raw}
            coords: [N, 3] collocation points with requires_grad=True
            imp_weights: [N, 1] importance weights (normalized, mean=1)
            eta_mode: "scalar" (fixed) or "field" (learned spatially-varying)
            eta_scalar: Value of eta when eta_mode="scalar"
            
        Returns:
            loss: Scalar physics loss
            info: Diagnostic information
        """
        # Forward through model to get fields
        out = model(coords)
        B_z = out["B_z"]
        u_x = out["u_x"]
        u_y = out["u_y"]
        
        # Get resistivity
        if out["eta_raw"] is not None and eta_mode == "field":
            eta = torch.sigmoid(out["eta_raw"])
            eta = self.eta_min + (self.eta_max - self.eta_min) * eta
            tv_reg = _tv1_1d(eta.view(-1), self.tv_eta)
        else:
            eta = B_z.new_full(B_z.shape, float(eta_scalar))
            tv_reg = B_z.new_tensor(0.0)
        
        # Compute gradients of B_z
        ones = torch.ones_like(B_z)
        grads = torch.autograd.grad(
            B_z, coords, 
            grad_outputs=ones,
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]
        dBz_dx = grads[..., 0:1]
        dBz_dy = grads[..., 1:2]
        dBz_dt = grads[..., 2:3]
        
        # Get test functions and their gradients
        phis, dphi_dx_list, dphi_dy_list = self.test_fn(coords)
        
        # Compute residuals for each test function
        residuals = []
        for phi, dphi_dx, dphi_dy in zip(phis, dphi_dx_list, dphi_dy_list):
            # Time derivative term: ∫ φ ∂Bz/∂t
            term_time = phi * dBz_dt
            
            # Transport term (integration by parts): ∫ ∇⊥φ · (Bz u)
            # = ∫ (∂φ/∂x * Bz * ux + ∂φ/∂y * Bz * uy)
            term_transport = dphi_dx * (B_z * u_x) + dphi_dy * (B_z * u_y)
            
            # Resistive diffusion term: -∫ ∇⊥φ · (η ∇⊥Bz)
            # = -∫ (∂φ/∂x * η * ∂Bz/∂x + ∂φ/∂y * η * ∂Bz/∂y)
            if self.use_resistive:
                term_resistive = -(dphi_dx * (eta * dBz_dx) + dphi_dy * (eta * dBz_dy))
            else:
                term_resistive = torch.zeros_like(term_time)
            
            # Weak form residual: LHS - RHS = ∂tBz - (transport + resistive) = 0
            residual = term_time - term_transport - term_resistive
            residuals.append(residual)
        
        # Combine residuals from all test functions
        all_residuals = torch.cat(residuals, dim=-1)  # [N, n_tests]
        
        # Adaptive normalization based on field magnitudes
        if self.normalization == "adaptive":
            # Scale by typical field magnitude to make residuals O(1)
            # Use robust statistics to avoid issues with near-zero fields
            B_abs = B_z.abs()
            u_abs = (u_x.abs() + u_y.abs()) / 2.0
            
            # Use median instead of mean for robustness to outliers
            B_scale = B_abs.median().clamp(min=1e-4).detach()
            u_scale = u_abs.median().clamp(min=1e-4).detach()
            
            # Add minimum scale based on data range to prevent division by tiny values
            B_range = (B_abs.max() - B_abs.min()).clamp(min=1e-4).detach()
            u_range = (u_abs.max() - u_abs.min()).clamp(min=1e-4).detach()
            
            # Use geometric mean of scale and range for stability
            B_norm = (B_scale * B_range).sqrt().clamp(min=1e-3)
            u_norm = (u_scale * u_range).sqrt().clamp(min=1e-3)
            
            norm_factor = (B_norm * u_norm).sqrt()
            all_residuals = all_residuals / (norm_factor + 1e-6)
        elif self.normalization == "per_scale":
            # Normalize each test function's residual separately
            with torch.no_grad():
                residual_scales = all_residuals.abs().mean(dim=0, keepdim=True).clamp(min=1e-8)
            all_residuals = all_residuals / residual_scales
        # else "fixed": no normalization
        
        # Importance-weighted mean squared residual
        # Average over test functions, then weighted average over points
        residual_sq = (all_residuals ** 2).mean(dim=-1, keepdim=True)  # [N, 1]
        
        # Safety: clamp residuals to prevent extreme values from destabilizing training
        residual_sq = residual_sq.clamp(max=1e4)
        
        # Handle any NaN values that slipped through (replace with 0 for stability)
        if torch.isnan(residual_sq).any():
            residual_sq = torch.nan_to_num(residual_sq, nan=0.0, posinf=1e4, neginf=0.0)
        
        loss_induction = (residual_sq * imp_weights).mean()
        
        # Optional: Divergence-free velocity constraint
        loss_div = B_z.new_tensor(0.0)
        if self.enforce_div_free_u:
            # Compute ∇·u = ∂ux/∂x + ∂uy/∂y
            du_x_grads = torch.autograd.grad(
                u_x, coords,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            du_y_grads = torch.autograd.grad(
                u_y, coords,
                grad_outputs=torch.ones_like(u_y),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            div_u = du_x_grads[..., 0:1] + du_y_grads[..., 1:2]
            loss_div = ((div_u ** 2) * imp_weights).mean()
        
        # Total physics loss
        total_loss = loss_induction + 0.1 * loss_div + tv_reg
        
        # Create diagnostic info
        with torch.no_grad():
            info = PhysicsResidualInfo(
                loss_induction=float(loss_induction.detach()),
                loss_divergence=float(loss_div.detach()),
                loss_tv_eta=float(tv_reg.detach()),
                residual_mean=float(all_residuals.abs().mean().detach()),
                residual_max=float(all_residuals.abs().max().detach())
            )
        
        return total_loss, info


class FreeEnergyProxy(nn.Module):
    """
    Compute free magnetic energy proxy from PINN outputs.
    
    Free energy E_free ∝ ∫ (B - B_potential)² dV
    
    This is a key predictor of flare potential that can be computed
    from the PINN's magnetic field representation.
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
        
        # High |∇Bz| near PIL indicates free energy
        return grad_Bz_mag.mean()


class CurrentHelicityProxy(nn.Module):
    """
    Compute current helicity proxy from PINN outputs.
    
    Current helicity H_c = ∫ B · J dV
    
    where J = ∇×B is the current density.
    This measures the twist and complexity of the magnetic field.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        A_z: torch.Tensor,
        B_z: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute current helicity proxy.
        
        Using the relation J_z = ∇²A_z (for 2D):
        H_c_z ≈ B_z * J_z = B_z * ∇²A_z
        
        Args:
            A_z: [N, 1] vector potential
            B_z: [N, 1] vertical field
            coords: [N, 3] coordinates
            
        Returns:
            helicity: [1] scalar proxy value
        """
        # Compute ∇A_z
        grad_Az = torch.autograd.grad(
            A_z, coords,
            grad_outputs=torch.ones_like(A_z),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute second derivatives (Laplacian components)
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
        
        # Laplacian = J_z (current density)
        J_z = d2Az_dx2 + d2Az_dy2
        
        # Helicity proxy = mean |B_z * J_z|
        helicity = (B_z * J_z).abs().mean()
        
        return helicity
