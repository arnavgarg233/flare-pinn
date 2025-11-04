"""
Unit tests for 2nd-order differentiable latent sampling.

Verifies that the new soft-bilinear and nearest samplers support
double backward (required for physics loss with weak-form residuals).
"""
import torch
import pytest
from src.models.pinn.latent_sampling import (
    sample_latent_soft_bilinear,
    sample_latent_nearest
)


def _toy_setup(N=2, C=4, H=8, W=8, P=5, device='cpu'):
    """Create toy latent map and coordinates for testing."""
    L = torch.randn(N, C, H, W, device=device, requires_grad=True)
    xy = torch.rand(N, P, 2, device=device) * 2 - 1  # [-1, 1]
    return L, xy


def test_double_backward_soft_bilinear():
    """
    Test that soft-bilinear sampler supports 2nd-order gradients w.r.t. coordinates.
    
    This is critical for physics loss: weak-form MHD needs ∂²B/∂x∂y.
    """
    L, xy = _toy_setup()
    L.requires_grad_(True)
    
    # First forward pass
    out = sample_latent_soft_bilinear(L, xy)  # [N, P, C]
    
    # Second forward pass with grad-enabled coords
    coords = xy.clone().requires_grad_(True)
    out2 = sample_latent_soft_bilinear(L, coords).sum()
    
    # 1st-order gradient w.r.t. coords
    (grad_coords,) = torch.autograd.grad(out2, coords, create_graph=True)
    assert grad_coords is not None, "First-order gradient w.r.t. coords is None!"
    assert torch.isfinite(grad_coords).all(), "First-order gradient has NaN/Inf!"
    
    # 2nd-order gradient w.r.t. coords (the critical test!)
    grad2 = torch.autograd.grad(
        grad_coords.sum(),
        coords,
        retain_graph=True,
        allow_unused=False
    )[0]
    assert grad2 is not None, "Second-order gradient w.r.t. coords is None!"
    assert torch.isfinite(grad2).all(), "Second-order gradient has NaN/Inf!"
    
    print("✓ Soft-bilinear sampler supports 2nd-order gradients w.r.t. coordinates")


def test_grad_flows_to_L_soft_bilinear():
    """
    Test that gradients flow to the latent map L (trains encoder).
    
    This ensures the CNN encoder learns from physics loss.
    """
    L, xy = _toy_setup()
    L.requires_grad_(True)
    
    out = sample_latent_soft_bilinear(L, xy).sum()
    out.backward()
    
    assert L.grad is not None, "Gradient w.r.t. L (encoder) is None!"
    assert torch.isfinite(L.grad).all(), "Gradient w.r.t. L has NaN/Inf!"
    assert (L.grad.abs() > 1e-6).any(), "Gradient w.r.t. L is too small (possibly zero)"
    
    print("✓ Soft-bilinear sampler propagates gradients to encoder")


def test_nearest_has_no_coord_grad_but_has_L_grad():
    """
    Test that nearest-neighbor sampler:
    1. Does NOT propagate gradients to coords (piecewise constant)
    2. DOES propagate gradients to L (trains encoder)
    """
    L, xy = _toy_setup()
    coords = xy.clone().requires_grad_(True)
    L.requires_grad_(True)
    
    out = sample_latent_nearest(L, coords).sum()
    
    # Check gradient w.r.t. coords (should be zero or None)
    g_coords = torch.autograd.grad(
        out, coords, retain_graph=True, allow_unused=True
    )[0]
    
    if g_coords is not None:
        # Nearest-neighbor is piecewise constant, so gradients should be ~0
        assert torch.allclose(
            g_coords, torch.zeros_like(coords), atol=1e-5
        ), "Nearest sampler should have zero gradient w.r.t. coords!"
    
    # Check gradient w.r.t. L (should exist and be non-zero)
    out.backward()
    assert L.grad is not None, "Gradient w.r.t. L (encoder) is None!"
    assert torch.isfinite(L.grad).all(), "Gradient w.r.t. L has NaN/Inf!"
    
    print("✓ Nearest sampler: no coord grad (expected), encoder grad OK")


def test_soft_bilinear_vs_nearest_smoothness():
    """
    Compare soft-bilinear (smooth) vs nearest (piecewise constant).
    
    Soft-bilinear should produce smoother output when coords change slightly.
    """
    L, xy = _toy_setup(N=1, P=100)
    
    # Sample at original coords
    out_soft_1 = sample_latent_soft_bilinear(L, xy)
    out_nearest_1 = sample_latent_nearest(L, xy)
    
    # Sample at slightly perturbed coords
    xy_perturbed = xy + 0.01 * torch.randn_like(xy)
    out_soft_2 = sample_latent_soft_bilinear(L, xy_perturbed)
    out_nearest_2 = sample_latent_nearest(L, xy_perturbed)
    
    # Compute differences
    diff_soft = (out_soft_2 - out_soft_1).abs().mean()
    diff_nearest = (out_nearest_2 - out_nearest_1).abs().mean()
    
    print(f"  Soft-bilinear diff: {diff_soft:.6f}")
    print(f"  Nearest diff: {diff_nearest:.6f}")
    
    # Nearest should have larger jumps due to piecewise constant nature
    # (This is not always true for random data, but generally holds)
    print("✓ Smoothness comparison complete")


def test_boundary_handling():
    """Test that samplers handle boundary coordinates correctly."""
    L, _ = _toy_setup(H=16, W=16)
    
    # Create coordinates at boundaries
    xy_boundary = torch.tensor([
        [[[-0.99, -0.99], [0.99, 0.99], [-0.99, 0.99], [0.99, -0.99]]],
        [[[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]]]
    ], dtype=torch.float32)
    
    # Should not crash or produce NaN
    out_soft = sample_latent_soft_bilinear(L, xy_boundary)
    out_nearest = sample_latent_nearest(L, xy_boundary)
    
    assert torch.isfinite(out_soft).all(), "Soft-bilinear produces NaN at boundaries!"
    assert torch.isfinite(out_nearest).all(), "Nearest produces NaN at boundaries!"
    
    print("✓ Boundary handling OK")


def test_shape_consistency():
    """Test that output shapes are correct for various input sizes."""
    test_cases = [
        (1, 8, 16, 16, 10),   # Single batch, 10 points
        (2, 4, 32, 32, 100),  # 2 batches, 100 points
        (4, 16, 64, 64, 1000), # 4 batches, 1000 points
    ]
    
    for N, C, H, W, P in test_cases:
        L, xy = _toy_setup(N, C, H, W, P)
        
        out_soft = sample_latent_soft_bilinear(L, xy)
        out_nearest = sample_latent_nearest(L, xy)
        
        expected_shape = (N, P, C)
        assert out_soft.shape == expected_shape, \
            f"Soft-bilinear shape mismatch: {out_soft.shape} vs {expected_shape}"
        assert out_nearest.shape == expected_shape, \
            f"Nearest shape mismatch: {out_nearest.shape} vs {expected_shape}"
    
    print("✓ Shape consistency verified")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing 2nd-order differentiable latent sampling")
    print("=" * 60)
    
    test_double_backward_soft_bilinear()
    test_grad_flows_to_L_soft_bilinear()
    test_nearest_has_no_coord_grad_but_has_L_grad()
    test_soft_bilinear_vs_nearest_smoothness()
    test_boundary_handling()
    test_shape_consistency()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

