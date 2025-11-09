#!/usr/bin/env python3
"""
Quick test of all 4 model variants before GPU training.

Tests:
1. MLP PINN (no physics)
2. MLP PINN (with physics)
3. Hybrid CNN-PINN (no physics)
4. Hybrid CNN-PINN (with physics) <- THE FIX

Each test runs 5 training steps to verify:
- No gradient errors
- Losses are finite
- Model outputs correct shapes
"""
import sys
import torch
import numpy as np
from pathlib import Path

from src.models.pinn import PINNConfig, PINNModel, HybridPINNModel


def create_tiny_batch(device='cpu'):
    """Create minimal test batch."""
    T, P, H, W = 3, 128, 32, 32
    
    coords = torch.rand(T, P, 3, device=device) * 2 - 1
    x, y, t = coords[..., 0], coords[..., 1], coords[..., 2]
    gt_bz = torch.sin(np.pi * x) * torch.cos(np.pi * y)
    gt_bz = gt_bz.unsqueeze(-1)
    
    frames = torch.randn(T, H, W, device=device) * 0.5
    observed_mask = torch.ones(T, dtype=torch.bool, device=device)
    labels = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    pil_mask = np.random.rand(H, W).astype(np.float32)
    pil_mask = (pil_mask > 0.8).astype(np.uint8)
    
    return {
        'coords': coords,
        'gt_bz': gt_bz,
        'frames': frames,
        'observed_mask': observed_mask,
        'labels': labels,
        'pil_mask': pil_mask,
    }


def test_model(model_type, physics_enabled, n_steps=100, device='cpu'):
    """Test a single model variant."""
    
    print(f"\n{'='*70}")
    print(f"Testing: {model_type.upper()} | Physics: {'ON' if physics_enabled else 'OFF'}")
    print(f"{'='*70}")
    
    # Create minimal config
    cfg = PINNConfig(
        seed=42,
        device=device,
        model=dict(
            model_type=model_type,
            hidden=128,
            layers=4,
            learn_eta=False,
            eta_scalar=0.01,
            fourier=dict(max_log2_freq=3, ramp_frac=0.5)
        ),
        classifier=dict(
            hidden=64,
            dropout=0.1,
            horizons=[6, 12, 24],
            loss_type='focal',
            focal_alpha=0.25,
            focal_gamma=2.0
        ),
        physics=dict(
            enable=physics_enabled,
            resistive=False,
            boundary_terms=False,
            lambda_phys_schedule=[[0.0, 1.0], [1.0, 1.0]]
        ),
        loss_weights=dict(cls=1.0, data=1.0, curl_consistency=0.0),
        train=dict(steps=n_steps, batch_size=1, lr=1e-3)
    )
    
    # Create model
    if model_type == 'hybrid':
        model = HybridPINNModel(cfg, encoder_in_channels=1).to(device)
    else:
        model = PINNModel(cfg).to(device)
    
    model.train()
    model.set_train_frac(1.0 if physics_enabled else 0.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Run training steps
    batch = create_tiny_batch(device)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Forward
        if model_type == 'hybrid':
            out = model(
                coords=batch['coords'],
                frames=batch['frames'],
                gt_bz=batch['gt_bz'],
                observed_mask=batch['observed_mask'],
                labels=batch['labels'],
                pil_mask=batch['pil_mask'],
                mode='train'
            )
        else:
            out = model(
                coords=batch['coords'],
                gt_bz=batch['gt_bz'],
                observed_mask=batch['observed_mask'],
                labels=batch['labels'],
                pil_mask=batch['pil_mask'],
                mode='train'
            )
        
        # Check outputs
        assert torch.isfinite(out.loss_total), f"Step {step}: loss is NaN/Inf!"
        assert torch.isfinite(out.probs).all(), f"Step {step}: probs has NaN/Inf!"
        assert out.probs.shape == (1, 3), f"Step {step}: wrong probs shape!"
        
        # Backward
        out.loss_total.backward()
        optimizer.step()
        
        if step == 0 or step == n_steps - 1:
            print(f"  Step {step:3d}: loss={out.loss_total.item():.4f} "
                  f"(cls={out.loss_cls.item():.4f}, "
                  f"data={out.loss_data.item():.4f}, "
                  f"phys={out.loss_phys.item():.4f})")
    
    # Check encoder grads for hybrid+physics
    if model_type == 'hybrid' and physics_enabled:
        encoder_params = [p for p in model.backbone.encoder.parameters() if p.requires_grad]
        has_grads = sum(1 for p in encoder_params if p.grad is not None)
        nonzero_grads = sum(1 for p in encoder_params if p.grad is not None and p.grad.abs().max() > 1e-8)
        
        print(f"  Encoder grads: {has_grads}/{len(encoder_params)} exist, "
              f"{nonzero_grads}/{len(encoder_params)} nonzero")
        
        if nonzero_grads == 0:
            print(f"  ⚠️  WARNING: Physics not reaching encoder!")
            return False
    
    print(f"✓ {n_steps} steps complete, all checks passed")
    return True


def main():
    """Run all 4 tests."""
    
    print("\n" + "="*70)
    print("QUICK MODEL VALIDATION (CPU)")
    print("Testing all 4 variants before GPU training")
    print("="*70)
    
    device = 'cpu'
    results = {}
    
    # Test all 4 combinations
    for model_type in ['mlp', 'hybrid']:
        for physics in [False, True]:
            key = f"{model_type}_physics_{'on' if physics else 'off'}"
            try:
                success = test_model(model_type, physics, n_steps=100, device=device)
                results[key] = 'PASS' if success else 'FAIL'
            except Exception as e:
                print(f"\n✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                results[key] = 'ERROR'
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for key, status in results.items():
        symbol = '✓' if status == 'PASS' else '✗'
        print(f"{symbol} {key:30s}: {status}")
    
    all_pass = all(v == 'PASS' for v in results.values())
    
    print("="*70)
    if all_pass:
        print("SUCCESS! All models working correctly.")
        print("Ready for GPU training.")
        return 0
    else:
        print("FAILURE! Some models have issues.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

