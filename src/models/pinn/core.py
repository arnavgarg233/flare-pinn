# src/models/pinn/core.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Dict, Tuple

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

    def forward(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
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
def B_perp_from_Az(A_z: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

class ClassifierHead(nn.Module):
    """
    Global spatiotemporal pooling over observed frames only -> MLP -> logits per horizon.
    Expects aggregated per-point features for (A_z, B_z, u_x, u_y).
    """
    def __init__(self, hidden: int = 256, dropout: float = 0.1, horizons=(6,12,24)):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, len(horizons))
        )
        self.horizons = tuple(horizons)

    def forward(self, feats: Dict[str, torch.Tensor], observed_mask: torch.Tensor) -> torch.Tensor:
        """
        feats: dict of tensors shaped [B, T, P, 1] for keys A_z, B_z, u_x, u_y (P = sampled points)
        observed_mask: [B, T] booleans (True for frames t <= t0)
        """
        X = torch.cat([feats["A_z"], feats["B_z"], feats["u_x"], feats["u_y"]], dim=-1)  # [B,T,P,4]
        mask = observed_mask[..., None, None].float()                                     # [B,T,1,1]
        X = X * mask
        denom_t = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        X_t = X.sum(dim=1, keepdim=True) / denom_t                                       # [B,1,P,4]
        X_avg = X_t.mean(dim=2)                                                           # [B,1,4] -> [B,4]
        X_flat = X_avg.squeeze(1)
        return self.mlp(X_flat)  # [B, n_horizons]

