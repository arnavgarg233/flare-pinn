# src/models/pinn/latent_sampling.py
"""
2nd-order differentiable latent sampling for hybrid PINN.

Replaces grid_sample to enable physics loss to train the CNN encoder.
PyTorch's grid_sample doesn't support double backward, which breaks
the weak-form MHD residuals that need ∂²B/∂x∂y.

Usage:
    from models.pinn.latent_sampling import sample_latent_soft_bilinear
    Ls = sample_latent_soft_bilinear(L, xy_norm)  # [N,C,H,W], [N,P,2] -> [N,P,C]
"""
import torch
import torch.nn.functional as F

_EPS = 1e-6

def _to_pix_coords(xy_norm, H, W):
    """Convert normalized coords [-1,1] to pixel coordinates [0, H-1] x [0, W-1]."""
    x = (xy_norm[..., 0].clamp(-1+_EPS, 1-_EPS) + 1.0) * 0.5 * (W - 1)
    y = (xy_norm[..., 1].clamp(-1+_EPS, 1-_EPS) + 1.0) * 0.5 * (H - 1)
    return x, y

@torch.no_grad()
def _safe_shape(L, xy_norm):
    """Validate input shapes."""
    assert L.dim() == 4, f"Expected L [N,C,H,W], got {list(L.shape)}"
    assert xy_norm.shape[-1] == 2, "xy_norm must be [..., 2]"
    return L.shape

def sample_latent_nearest(L: torch.Tensor, xy_norm: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbor latent sampling (piecewise constant w.r.t coords).
    
    - Gradients flow to L (encoder), NOT to coords.
    - 2nd-order grads w.r.t coords avoid sampler path entirely.
    - Fastest option, but may have small artifacts at pixel boundaries.
    
    Args:
        L: [N,C,H,W] - Latent feature map from encoder
        xy_norm: [N,P,2] - Normalized coordinates in [-1,1]
    
    Returns:
        sampled: [N,P,C] - Sampled latent features at each point
    """
    N, C, H, W = _safe_shape(L, xy_norm)
    x, y = _to_pix_coords(xy_norm, H, W)                         # [N,P]
    xi = x.round().long().clamp(0, W-1)                          # [N,P]
    yi = y.round().long().clamp(0, H-1)                          # [N,P]

    # Gather per batch using scatter/bmm trick
    idx = yi * W + xi                                            # [N,P]
    L_flat = L.view(N, C, H*W)                                   # [N,C,HW]
    # Create one-hot encoding for gathering
    one_hot = torch.zeros(N, xy_norm.size(1), H*W, device=L.device, dtype=L.dtype)
    one_hot.scatter_(2, idx.unsqueeze(-1), 1)                    # [N,P,HW]
    # Matrix multiply: (N,P,HW) x (N,HW,C) -> (N,P,C)
    out = torch.bmm(one_hot, L_flat.transpose(1,2))
    return out                                                   # [N,P,C]

def sample_latent_soft_bilinear(L: torch.Tensor, xy_norm: torch.Tensor) -> torch.Tensor:
    """
    Manual bilinear sampler with smooth weights.
    
    - Fully differentiable to 2nd order w.r.t. coords almost everywhere.
    - No autograd limitations of grid_sample double-backward.
    - Smooth interpolation like standard bilinear sampling.
    
    This is the RECOMMENDED sampler for hybrid PINN training.
    
    Args:
        L: [N,C,H,W] - Latent feature map from encoder
        xy_norm: [N,P,2] - Normalized coordinates in [-1,1]
    
    Returns:
        sampled: [N,P,C] - Bilinearly interpolated latent features
        
    Note:
        Gradients flow cleanly to both L (trains encoder) and coords
        (enables physics loss with second-order derivatives).
    """
    N, C, H, W = _safe_shape(L, xy_norm)
    x, y = _to_pix_coords(xy_norm, H, W)                         # [N,P]
    x0 = torch.floor(x).long().clamp(0, W-2); x1 = x0 + 1
    y0 = torch.floor(y).long().clamp(0, H-2); y1 = y0 + 1

    # Fractional parts (gradients flow through x,y; floor has zero grad)
    fx = x - x0.float()                                          # [N,P]
    fy = y - y0.float()                                          # [N,P]

    # Bilinear weights
    w00 = (1 - fx) * (1 - fy)
    w01 = (1 - fx) * fy
    w10 = fx * (1 - fy)
    w11 = fx * fy                                                # all [N,P]

    def gather_xy(xx, yy):
        """Gather values at specific pixel coordinates."""
        idx = yy * W + xx                                        # [N,P]
        L_flat = L.view(N, C, H*W)                               # [N,C,HW]
        one_hot = torch.zeros(N, xy_norm.size(1), H*W, device=L.device, dtype=L.dtype)
        one_hot.scatter_(2, idx.unsqueeze(-1), 1)                # [N,P,HW]
        return torch.bmm(one_hot, L_flat.transpose(1,2))         # [N,P,C]

    # Gather at 4 corners
    v00 = gather_xy(x0, y0)
    v01 = gather_xy(x0, y1)
    v10 = gather_xy(x1, y0)
    v11 = gather_xy(x1, y1)                                      # all [N,P,C]

    # Weighted sum
    out = (w00.unsqueeze(-1) * v00 + w01.unsqueeze(-1) * v01 +
           w10.unsqueeze(-1) * v10 + w11.unsqueeze(-1) * v11)   # [N,P,C]
    return out

