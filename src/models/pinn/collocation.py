# src/models/pinn/collocation.py
from __future__ import annotations
import numpy as np
import torch
from typing import Tuple

def sample_xy_from_mask(mask: np.ndarray, n: int, device: str = "cpu") -> torch.Tensor:
    """mask: [H,W] {0,1}; returns (x,y) in [-1,1]^2"""
    H, W = mask.shape
    idx = np.flatnonzero(mask.reshape(-1) > 0.5)
    if idx.size == 0:
        # uniform fallback
        xy = torch.rand(n, 2, device=device) * 2.0 - 1.0
        return xy
    choice = np.random.choice(idx, size=n, replace=True)
    ys, xs = np.unravel_index(choice, (H, W))
    xs = (xs / (W - 1)) * 2.0 - 1.0
    ys = (ys / (H - 1)) * 2.0 - 1.0
    xy = np.stack([xs, ys], axis=-1).astype(np.float32)
    return torch.from_numpy(xy).to(device)

def mix_pil_uniform(
    H: int, W: int, alpha: float, n_points: int, pil_mask: np.ndarray | None, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      coords: [N,3] in [-1,1]^3 (time uniform over [-1,1])
      p:      [N,1] mixture pdf (spatial * temporal), used for importance weights
    """
    alpha = float(max(0.0, min(1.0, alpha)))
    n_pil = int(round(alpha * n_points))
    n_uni = n_points - n_pil

    xy_pil = sample_xy_from_mask(pil_mask if pil_mask is not None else np.zeros((H,W), np.float32), n_pil, device)
    xy_uni = torch.rand(n_uni, 2, device=device) * 2.0 - 1.0
    xy = torch.cat([xy_pil, xy_uni], dim=0)
    t  = torch.rand(n_points, 1, device=device) * 2.0 - 1.0
    coords = torch.cat([xy, t], dim=-1)

    # pdf: p = α * p_pil + (1-α) * p_uni; treat both approx uniform over their supports
    area_xy = 4.0  # square [-1,1]^2
    p_uni_xy = 1.0 / area_xy
    # conservative fallback -> 1/area if mask empty
    p_pil_xy = p_uni_xy
    p_xy = alpha * p_pil_xy + (1 - alpha) * p_uni_xy
    p_t  = 0.5  # uniform over [-1,1]
    p = torch.full((n_points, 1), p_xy * p_t, device=device)
    return coords, p

def clip_and_renorm_importance(p: torch.Tensor, clip_quantile: float = 0.99) -> Tuple[torch.Tensor, float]:
    inv = 1.0 / p.clamp_min(1e-12)
    with torch.no_grad():
        thr = torch.quantile(inv, float(clip_quantile))
    w = torch.minimum(inv, thr)
    w_tilde = w / w.mean()
    return w_tilde, float(thr.item())

def ess(weights: torch.Tensor) -> float:
    """Effective sample size for normalized weights w̃."""
    w = weights / weights.sum()
    return float(1.0 / (w**2).sum().item())

