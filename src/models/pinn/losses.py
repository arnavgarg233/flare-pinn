# src/models/pinn/losses.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional

def interp_schedule(schedule: list[list[float]], frac: float) -> float:
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

def bce_logits(y_hat: torch.Tensor, y: torch.Tensor, pos_weight: Optional[float] = None) -> torch.Tensor:
    """
    Binary cross-entropy with logits. 
    pos_weight: weight for positive class (typically N_neg/N_pos, e.g. 5-20 for solar flares)
    """
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=y_hat.device, dtype=y_hat.dtype)
    return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=pw)

def focal_loss(logits: torch.Tensor, y: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss (Lin et al. 2017) for class imbalance.
    Downweights easy examples, focuses on hard negatives/positives.
    
    Args:
        logits: raw logits [N, C] (NOT probabilities!)
        y: targets {0,1} [N, C]
        alpha: weight for positive class (0.25 = emphasize positives)
        gamma: focusing parameter (2.0 standard, higher = more focus on hard examples)
    
    Common in solar flare prediction papers for rare event detection.
    
    Note: Uses numerically stable implementation with logsigmoid.
    """
    # Use numerically stable BCE with logits
    bce_loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    
    # Compute probabilities for focal weight (clamped for stability)
    probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    
    # p_t = p if y=1, else 1-p
    p_t = probs * y + (1 - probs) * (1 - y)
    
    # alpha_t = alpha if y=1, else 1-alpha
    alpha_t = alpha * y + (1 - alpha) * (1 - y)
    
    # focal weight: (1 - p_t)^gamma (clamped to avoid extreme values)
    focal_weight = ((1 - p_t) ** gamma).clamp(max=100.0)
    
    loss = alpha_t * focal_weight * bce_loss
    return loss.mean()

def l1_data(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        w = mask.float()
        denom = w.sum().clamp_min(1.0)
        # Return 0 if no valid data (all masked), instead of NaN
        if denom == 1.0 and w.sum() == 0.0:
            return pred.new_tensor(0.0)
        return (w * (pred - target).abs()).sum() / denom
    return (pred - target).abs().mean()

def focal_loss_with_label_smoothing(
    logits: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 0.25, 
    gamma: float = 2.0,
    smoothing: float = 0.05
) -> torch.Tensor:
    """
    Focal Loss with label smoothing for improved calibration.
    
    Label smoothing prevents overconfident predictions and
    can improve calibration metrics (ECE, Brier score).
    
    Args:
        logits: raw logits [N, C]
        y: targets {0,1} [N, C]
        alpha: weight for positive class
        gamma: focusing parameter
        smoothing: label smoothing factor (0.0 to 0.1 typical)
    """
    # Apply label smoothing: y_smooth = (1 - smoothing) * y + smoothing * 0.5
    y_smooth = (1 - smoothing) * y + smoothing * 0.5
    
    # Standard focal loss computation with smoothed labels
    bce_loss = F.binary_cross_entropy_with_logits(logits, y_smooth, reduction='none')
    
    probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    p_t = probs * y_smooth + (1 - probs) * (1 - y_smooth)
    alpha_t = alpha * y + (1 - alpha) * (1 - y)  # Use original y for alpha weighting
    focal_weight = ((1 - p_t) ** gamma).clamp(max=100.0)
    
    loss = alpha_t * focal_weight * bce_loss
    return loss.mean()


def curl_consistency_l1(
    B_perp_from_Az_fn,
    A_z_points: torch.Tensor,
    coords_points: torch.Tensor,
    Bx_obs: Optional[torch.Tensor] = None,
    By_obs: Optional[torch.Tensor] = None,
    weight: float = 0.1
) -> torch.Tensor:
    if Bx_obs is None or By_obs is None or weight <= 0: 
        return A_z_points.new_tensor(0.0)
    Bx, By = B_perp_from_Az_fn(A_z_points, coords_points)
    return weight * (Bx - Bx_obs).abs().mean() + weight * (By - By_obs).abs().mean()


# ============================================================================
# SOTA IMPROVEMENTS
# ============================================================================

def mixup_data(
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup data augmentation (Zhang et al., 2018).
    
    Particularly effective for imbalanced classification.
    Creates virtual training examples by interpolating between samples.
    
    Args:
        x: Input features [B, ...]
        y: Labels [B, C]
        alpha: Beta distribution parameter (0.2-0.4 typical)
        
    Returns:
        mixed_x: Interpolated features
        y_a: Original labels
        y_b: Mixed labels  
        lam: Interpolation coefficient
    """
    if alpha <= 0:
        return x, y, y, 1.0
    
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """Mixup loss: weighted combination of losses for mixed targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def asymmetric_focal_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05
) -> torch.Tensor:
    """
    Asymmetric Focal Loss (Ben-Baruch et al., 2020).
    
    Key insight: For imbalanced data, we want different focusing for
    positive vs negative examples. More aggressive focusing on negatives
    helps with class imbalance.
    
    Args:
        logits: Raw predictions [N, C]
        y: Binary targets [N, C]
        gamma_pos: Focusing for positives (0 = no focusing)
        gamma_neg: Focusing for negatives (4 = strong focusing)
        clip: Probability clipping for negatives
        
    Returns:
        loss: Scalar loss
    """
    probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    
    # Positive focal term
    pos_term = y * ((1 - probs) ** gamma_pos) * torch.log(probs)
    
    # Negative focal term with probability clipping
    p_neg = (probs * (1 - y)).clamp(max=1 - clip)
    neg_term = (1 - y) * (p_neg ** gamma_neg) * torch.log(1 - probs)
    
    loss = -(pos_term + neg_term)
    return loss.mean()


class TemperatureScaling(torch.nn.Module):
    """
    Temperature scaling for calibration (Guo et al., 2017).
    
    Post-hoc calibration method that learns a single temperature
    parameter to scale logits. Apply after training on validation set.
    
    Usage:
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(val_logits, val_labels)
        calibrated_probs = temp_scaler(test_logits)
    """
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling and return probabilities."""
        return torch.sigmoid(logits / self.temperature.clamp(min=0.01))
    
    def fit(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """
        Fit temperature on validation set using NLL minimization.
        
        Returns:
            Optimal temperature value
        """
        self.temperature.data.fill_(1.0)
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature.clamp(min=0.01)
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        return float(self.temperature.item())


def gradient_penalty(
    model: torch.nn.Module,
    coords: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    weight: float = 0.1
) -> torch.Tensor:
    """
    Gradient penalty for Lipschitz regularization.
    
    Encourages the model to have bounded gradients, which:
    - Improves generalization
    - Stabilizes physics loss training
    - Reduces overfitting to training distribution
    
    Args:
        model: The neural network
        coords: Input coordinates [N, 3]
        outputs: Model outputs dict with B_z, u_x, u_y
        weight: Regularization weight
        
    Returns:
        Gradient penalty loss
    """
    if weight <= 0:
        return coords.new_tensor(0.0)
    
    # Compute gradient norm of B_z w.r.t inputs
    B_z = outputs.get("B_z")
    if B_z is None or not coords.requires_grad:
        return coords.new_tensor(0.0)
    
    grads = torch.autograd.grad(
        B_z, coords,
        grad_outputs=torch.ones_like(B_z),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Penalize gradient norm exceeding 1 (1-Lipschitz)
    grad_norm = grads.norm(2, dim=-1)
    penalty = ((grad_norm - 1).clamp(min=0) ** 2).mean()
    
    return weight * penalty


def confidence_penalty(
    probs: torch.Tensor,
    weight: float = 0.1
) -> torch.Tensor:
    """
    Confidence penalty to prevent overconfident predictions.
    
    Adds entropy regularization to encourage calibrated predictions,
    especially important for rare event prediction like flares.
    
    Args:
        probs: Predicted probabilities [N, C]
        weight: Regularization weight
        
    Returns:
        Entropy penalty loss (to be added to main loss)
    """
    if weight <= 0:
        return probs.new_tensor(0.0)
    
    probs_clamped = probs.clamp(1e-7, 1 - 1e-7)
    entropy = -(probs_clamped * torch.log(probs_clamped) + 
                (1 - probs_clamped) * torch.log(1 - probs_clamped))
    
    # Negative entropy = penalty for overconfidence
    # We want higher entropy (less confident) predictions
    return -weight * entropy.mean()

