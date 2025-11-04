# src/models/pinn/physics.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

def _tv1_1d(x: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0: 
        return x.new_tensor(0.0)
    if x.numel() < 2:
        return x.new_tensor(0.0)
    return weight * (x[1:] - x[:-1]).abs().mean()

class WeakFormInduction2p5D(nn.Module):
    """
    Weak-form of ∂t Bz = -∇⊥·(Bz u) + ∇⊥·(η ∇⊥ Bz)  on observed slab (t <= t0).
    Implements:
       L_phys = E_{(x,y,t)~p}[ w̃ * (  φ ∂t Bz
                                     - ∇⊥φ · (Bz u)
                                     + ∇⊥φ · (η ∇⊥Bz) )^2 ]
    with random low-order test φ(x,y,t) = a0 + a1 x + a2 y + a3 t.
    Boundary terms exist in the full derivation; kept optional/off in starter.
    """
    def __init__(self, eta_bounds=(1e-4, 1.0), use_resistive=False, include_boundary=False, tv_eta: float = 1e-3):
        super().__init__()
        self.eta_min, self.eta_max = eta_bounds
        self.use_resistive = use_resistive
        self.include_boundary = include_boundary
        self.tv_eta = tv_eta

    def forward(
        self,
        model: nn.Module,
        coords: torch.Tensor,              # [N,3], requires_grad=True
        imp_weights: torch.Tensor,         # [N,1] (renormalized importance weights)
        eta_mode: str = "scalar",
        eta_scalar: float = 0.01
    ) -> tuple[torch.Tensor, dict[str, float]]:
        out = model(coords)
        B_z = out["B_z"]; u_x = out["u_x"]; u_y = out["u_y"]
        # eta
        if out["eta_raw"] is not None and eta_mode == "field":
            eta = torch.sigmoid(out["eta_raw"])
            eta = self.eta_min + (self.eta_max - self.eta_min) * eta
            tv_reg = _tv1_1d(eta.view(-1), self.tv_eta)
        else:
            eta = B_z.new_full(B_z.shape, float(eta_scalar))
            tv_reg = B_z.new_tensor(0.0)

        ones = torch.ones_like(B_z)
        grads = torch.autograd.grad(B_z, coords, grad_outputs=ones,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        dBz_dx = grads[..., 0:1]
        dBz_dy = grads[..., 1:2]
        dBz_dt = grads[..., 2:3]

        # random linear test φ
        with torch.no_grad():
            a = torch.randn(4, device=coords.device)
        x, y, t = coords[..., 0:1], coords[..., 1:2], coords[..., 2:3]
        phi = a[0] + a[1]*x + a[2]*y + a[3]*t
        dphi_dx = a[1] + torch.zeros_like(x)
        dphi_dy = a[2] + torch.zeros_like(y)

        # Weak-form residual terms (WITHOUT quad_wts multiplication to avoid tiny values)
        # The quad_wts cause the loss to become ~1/N^3, making it ineffective
        # Instead, we compute the residual at each point and weight by importance
        term_time       = phi * dBz_dt
        term_transport  = -(dphi_dx * (B_z * u_x) + dphi_dy * (B_z * u_y))
        if self.use_resistive:
            term_resistive = (dphi_dx * (eta * dBz_dx) + dphi_dy * (eta * dBz_dy))
        else:
            term_resistive = torch.zeros_like(term_time)

        residual = term_time + term_transport + term_resistive   # [N,1]
        
        # Importance-weighted mean squared residual
        # imp_weights are normalized (mean=1), so this is E_p[residual^2]
        # Scale up by 1e6 to make physics loss competitive with data loss
        # Typical residual ~ 1e-5, so residual^2 ~ 1e-10 without scaling
        loss_phys = ((residual ** 2) * imp_weights).mean() * 1e6
        
        return loss_phys + tv_reg, {
            "phys_loss": float(loss_phys.detach()),
            "tv_eta": float(tv_reg.detach()),
        }

