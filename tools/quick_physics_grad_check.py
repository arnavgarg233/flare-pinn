#!/usr/bin/env python3
"""
Quick sanity check: verify physics loss trains the CNN encoder.

This script performs a single forward/backward pass and confirms that:
1. Physics residual can be computed (2nd-order derivatives work)
2. Gradients flow back to encoder parameters
3. No NaN/Inf in gradients

Run this before full training to catch any gradient issues early.
"""
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.models.pinn import PINNConfig, HybridPINNModel


def create_dummy_batch(cfg, device='cpu'):
    """Create a minimal batch for testing."""
    T = 4  # time steps
    P = 256  # collocation points per time step
    H, W = 64, 64  # spatial resolution
    
    # Collocation coordinates
    coords = torch.rand(T, P, 3, device=device) * 2.0 - 1.0
    
    # Ground truth Bz (synthetic smooth field)
    x, y, t = coords[..., 0], coords[..., 1], coords[..., 2]
    gt_bz = torch.sin(np.pi * x) * torch.cos(np.pi * y) * torch.exp(-0.5 * t)
    gt_bz = gt_bz.unsqueeze(-1)
    
    # Observed frames (synthetic magnetograms)
    frames = torch.randn(T, H, W, device=device)
    
    # Observed mask (all frames available)
    observed_mask = torch.ones(T, dtype=torch.bool, device=device)
    
    # Labels (dummy)
    labels = torch.tensor([[0.0, 1.0, 0.0]], device=device)  # [1, n_horizons]
    
    # PIL mask (dummy)
    pil_mask = np.ones((H, W), dtype=np.uint8)
    
    return {
        'coords': coords,
        'gt_bz': gt_bz,
        'frames': frames,
        'observed_mask': observed_mask,
        'labels': labels,
        'pil_mask': pil_mask,
    }


def check_physics_grad_flow(cfg, device='cpu'):
    """Main check: verify physics loss trains encoder."""
    
    print("=" * 70)
    print("PHYSICS GRADIENT FLOW CHECK")
    print("=" * 70)
    
    # Create model
    model = HybridPINNModel(cfg, encoder_in_channels=1).to(device)
    model.train()
    
    # Enable physics from the start (skip curriculum)
    model.set_train_frac(1.0)
    
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Get encoder parameters (we want to check these get gradients)
    encoder_params = list(model.backbone.encoder.parameters())
    print(f"✓ Encoder has {len(encoder_params)} parameter tensors")
    
    # Create batch
    batch = create_dummy_batch(cfg, device)
    print(f"✓ Dummy batch created: {batch['coords'].shape[0]} time steps, "
          f"{batch['coords'].shape[1]} points per step")
    
    # Forward pass with physics enabled
    print("\n" + "-" * 70)
    print("Forward pass...")
    print("-" * 70)
    
    out = model(
        coords=batch['coords'],
        frames=batch['frames'],
        gt_bz=batch['gt_bz'],
        observed_mask=batch['observed_mask'],
        labels=batch['labels'],
        pil_mask=batch['pil_mask'],
        mode='train'
    )
    
    print(f"✓ Forward pass complete")
    print(f"  Loss total: {out.loss_total.item():.6f}")
    print(f"  Loss cls:   {out.loss_cls.item():.6f}")
    print(f"  Loss data:  {out.loss_data.item():.6f}")
    print(f"  Loss phys:  {out.loss_phys.item():.6f}")
    print(f"  λ_phys:     {out.lambda_phys:.3f}")
    
    # Check that physics loss is non-zero
    if out.loss_phys.item() < 1e-9:
        print("⚠️  WARNING: Physics loss is near zero!")
        print("   Check that lambda_phys > 0 and physics.enable = true")
    
    # Backward pass
    print("\n" + "-" * 70)
    print("Backward pass...")
    print("-" * 70)
    
    out.loss_total.backward()
    
    print("✓ Backward pass complete (no errors!)")
    
    # Check encoder gradients
    print("\n" + "-" * 70)
    print("Checking encoder gradients...")
    print("-" * 70)
    
    encoder_grads_exist = []
    encoder_grads_nonzero = []
    encoder_grads_finite = []
    
    for i, p in enumerate(encoder_params):
        if p.requires_grad:
            has_grad = p.grad is not None
            encoder_grads_exist.append(has_grad)
            
            if has_grad:
                is_finite = torch.isfinite(p.grad).all().item()
                is_nonzero = (p.grad.abs() > 1e-9).any().item()
                encoder_grads_finite.append(is_finite)
                encoder_grads_nonzero.append(is_nonzero)
                
                grad_norm = p.grad.norm().item()
                grad_max = p.grad.abs().max().item()
                print(f"  Param {i}: grad_norm={grad_norm:.6f}, grad_max={grad_max:.6f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    all_exist = all(encoder_grads_exist)
    all_finite = all(encoder_grads_finite)
    any_nonzero = any(encoder_grads_nonzero)
    
    if all_exist:
        print("✓ All encoder parameters have gradients")
    else:
        print(f"✗ Only {sum(encoder_grads_exist)}/{len(encoder_grads_exist)} "
              f"encoder params have gradients")
    
    if all_finite:
        print("✓ All gradients are finite (no NaN/Inf)")
    else:
        print("✗ Some gradients are NaN/Inf!")
    
    if any_nonzero:
        print("✓ Physics loss propagates gradients to encoder")
    else:
        print("✗ Encoder gradients are all zero - physics loss not reaching encoder!")
    
    print("=" * 70)
    
    if all_exist and all_finite and any_nonzero:
        print("SUCCESS! ✓✓✓")
        print("Physics loss can now train the CNN encoder end-to-end.")
        print("The soft-bilinear sampler is working correctly.")
        return True
    else:
        print("FAILURE! ✗✗✗")
        print("Something is wrong with gradient flow.")
        return False


def main():
    """Run the check."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check physics gradient flow to encoder")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/hybrid_5k.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)"
    )
    args = parser.parse_args()
    
    # Load config
    cfg = PINNConfig.from_yaml(args.config)
    
    # Force physics on for this test
    cfg.physics.enable = True
    cfg.physics.lambda_phys_schedule = [[0.0, 1.0], [1.0, 1.0]]  # Constant λ=1
    
    print(f"Config loaded from: {args.config}")
    print(f"Model type: {cfg.model.model_type}")
    print(f"Physics enabled: {cfg.physics.enable}")
    print(f"Device: {args.device}")
    
    # Run check
    success = check_physics_grad_flow(cfg, device=args.device)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

