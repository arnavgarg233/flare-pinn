# src/models/pinn/losses.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List

def interp_schedule(schedule: List[List[float]], frac: float) -> float:
    """
    schedule = [[x0,y0], [x1,y1], ..., [1.0,y_end]]; piecewise-linear.
    """
    pts = sorted(schedule, key=lambda p: p[0])
    if frac <= pts[0][0]: return float(pts[0][1])
    for i in range(1, len(pts)):
        x0,y0 = pts[i-1]; x1,y1 = pts[i]
        if frac <= x1:
            t = (frac - x0) / max(1e-8, (x1 - x0))
            return float(y0*(1-t) + y1*t)
    return float(pts[-1][1])

def bce_logits(y_hat: torch.Tensor, y: torch.Tensor, pos_weight: float | None = None) -> torch.Tensor:
    """
    Binary cross-entropy with logits. 
    pos_weight: weight for positive class (typically N_neg/N_pos, e.g. 5-20 for solar flares)
    """
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=y_hat.device, dtype=y_hat.dtype)
    return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=pw)

def focal_loss(y_hat: torch.Tensor, y: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss (Lin et al. 2017) for class imbalance.
    Downweights easy examples, focuses on hard negatives/positives.
    
    Args:
        y_hat: logits [N, C]
        y: targets {0,1} [N, C]
        alpha: weight for positive class (0.25 = emphasize positives)
        gamma: focusing parameter (2.0 standard, higher = more focus on hard examples)
    
    Common in solar flare prediction papers for rare event detection.
    """
    bce_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none')
    probs = torch.sigmoid(y_hat)
    
    # p_t = p if y=1, else 1-p
    p_t = probs * y + (1 - probs) * (1 - y)
    
    # alpha_t = alpha if y=1, else 1-alpha
    alpha_t = alpha * y + (1 - alpha) * (1 - y)
    
    # focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    loss = alpha_t * focal_weight * bce_loss
    return loss.mean()

def l1_data(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        w = mask.float()
        return (w * (pred - target).abs()).sum() / w.sum().clamp_min(1.0)
    return (pred - target).abs().mean()

def curl_consistency_l1(B_perp_from_Az_fn, A_z_points, coords_points, Bx_obs=None, By_obs=None, weight: float = 0.1) -> torch.Tensor:
    if Bx_obs is None or By_obs is None or weight <= 0: 
        return A_z_points.new_tensor(0.0)
    Bx, By = B_perp_from_Az_fn(A_z_points, coords_points)
    return weight * (Bx - Bx_obs).abs().mean() + weight * (By - By_obs).abs().mean()

