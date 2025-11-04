#!/usr/bin/env python3
"""
Quick integration test - trains for a few steps to verify everything works.
Tests end-to-end pipeline with minimal resources.

Usage:
    python scripts/run_quick_test.py
    python scripts/run_quick_test.py --model hybrid  # Test hybrid model
    python scripts/run_quick_test.py --physics      # Test with physics
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pinn import PINNConfig, PINNModel, HybridPINNModel


def run_quick_test(model_type: str = "mlp", enable_physics: bool = False):
    """Run quick integration test."""
    print(f"\n{'='*60}")
    print(f"Quick Integration Test")
    print(f"Model: {model_type.upper()}, Physics: {enable_physics}")
    print(f"{'='*60}\n")
    
    # Minimal config
    cfg = PINNConfig(
        seed=42,
        device="cpu",  # Use CPU for quick test
        model={
            "model_type": model_type,
            "hidden": 128,
            "layers": 4,
            "fourier": {"max_log2_freq": 3, "ramp_frac": 0.5}
        },
        classifier={
            "hidden": 64,
            "dropout": 0.1,
            "horizons": [6, 12],
            "loss_type": "focal"
        },
        physics={
            "enable": enable_physics,
            "resistive": False,
            "lambda_phys_schedule": [
                [0.0, 0.0],
                [0.5, 0.5],
                [1.0, 1.0]
            ]
        },
        collocation={"n_max": 1024, "alpha_start": 0.5, "alpha_end": 0.8},
        train={"steps": 100, "lr": 1e-3, "batch_size": 1, "amp": False},
        data={"use_real": False, "dummy_T": 8, "dummy_H": 64, "dummy_W": 64}
    )
    
    print(f"Config: {cfg.model_dump_summary()}\n")
    
    # Create model
    print("1. Initializing model...")
    if model_type == "mlp":
        model = PINNModel(cfg)
    else:
        model = HybridPINNModel(cfg, encoder_in_channels=1)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,}")
    
    # Create dummy data
    print("\n2. Creating dummy data...")
    T, H, W = 8, 64, 64
    P_per_t = 128  # Points per time slice
    
    # Dummy frames
    frames = torch.randn(T, H, W) * 100  # Simulated Bz in Gauss
    
    # Collocation points (sample in space-time)
    coords = torch.rand(T, P_per_t, 3) * 2 - 1  # [-1, 1]^3
    
    # Sample Bz at collocation points (for data loss)
    # In reality this comes from interpolating the frames
    gt_bz = torch.randn(T, P_per_t, 1) * 100
    
    # Labels (random flare/no-flare)
    labels = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # No flare
    
    # Observed mask (first 6 frames observed, rest are future)
    observed_mask = torch.zeros(T, dtype=torch.bool)
    observed_mask[:6] = True
    
    # PIL mask (for importance sampling)
    pil_mask = np.random.rand(H, W) > 0.85  # ~15% high gradient regions
    
    print(f"   Frames: {frames.shape}")
    print(f"   Coords: {coords.shape}")
    print(f"   Labels: {labels}")
    print(f"   Observed frames: {observed_mask.sum()}/{T}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    
    # Training loop
    print("\n3. Running training loop...")
    model.train()
    
    for step in range(cfg.train.steps):
        frac = step / cfg.train.steps
        model.set_train_frac(frac)
        
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == "hybrid":
            output = model(
                coords=coords,
                frames=frames,
                gt_bz=gt_bz,
                observed_mask=observed_mask,
                labels=labels,
                pil_mask=pil_mask if enable_physics else None,
                mode="train"
            )
        else:
            output = model(
                coords=coords,
                gt_bz=gt_bz,
                observed_mask=observed_mask,
                labels=labels,
                pil_mask=pil_mask if enable_physics else None,
                mode="train"
            )
        
        # Backward pass
        output.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        
        # Log
        if step % 10 == 0:
            print(f"   Step {step:3d}: "
                  f"loss={output.loss_total.item():.3f} "
                  f"(cls={output.loss_cls.item():.3f}, "
                  f"data={output.loss_data.item():.3f}, "
                  f"phys={output.loss_phys.item():.3e}) "
                  f"λ_phys={output.lambda_phys:.2f}")
    
    # Final evaluation
    print("\n4. Running evaluation...")
    model.eval()
    
    # Create fresh coords with grad for eval (needed for B_perp calculation)
    eval_coords = coords.detach().clone()
    eval_coords.requires_grad_(True)
    
    # Don't use no_grad context for eval - we need gradients for B_perp
    if model_type == "hybrid":
        output = model(eval_coords, frames, gt_bz, observed_mask, labels, mode="eval")
    else:
        output = model(eval_coords, gt_bz, observed_mask, labels, mode="eval")
    
    print(f"   Final predictions: {output.probs.detach().numpy().flatten()}")
    print(f"   B_z range: [{output.B_z.min().item():.1f}, {output.B_z.max().item():.1f}] G")
    print(f"   u_x range: [{output.u_x.min().item():.2f}, {output.u_x.max().item():.2f}]")
    print(f"   u_y range: [{output.u_y.min().item():.2f}, {output.u_y.max().item():.2f}]")
    
    # Check for issues
    print("\n5. Sanity checks...")
    checks = {
        "Predictions in [0,1]": (0 <= output.probs.min() <= 1) and (0 <= output.probs.max() <= 1),
        "No NaN in outputs": not torch.isnan(output.B_z).any(),
        "No Inf in outputs": not torch.isinf(output.B_z).any(),
        "Losses finite": torch.isfinite(output.loss_total),
        "Gradients computed": all(p.grad is not None for p in model.parameters() if p.requires_grad)
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"   {status} {check}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print(f"\n{'='*60}")
        print(f"✓ ALL TESTS PASSED!")
        print(f"{'='*60}\n")
        return True
    else:
        print(f"\n{'='*60}")
        print(f"✗ SOME TESTS FAILED")
        print(f"{'='*60}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick integration test")
    parser.add_argument("--model", choices=["mlp", "hybrid"], default="mlp",
                       help="Model type to test")
    parser.add_argument("--physics", action="store_true",
                       help="Enable physics loss")
    args = parser.parse_args()
    
    success = run_quick_test(
        model_type=args.model,
        enable_physics=args.physics
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

