#!/usr/bin/env python3
"""
Benchmark Comparison: Baseline (data-only) vs Physics-Informed PINN
Compares MLP PINN and Hybrid PINN with/without physics.

Usage:
    # Run all comparisons (takes ~10-15 min on CPU)
    python scripts/benchmark_comparison.py
    
    # Run specific comparison
    python scripts/benchmark_comparison.py --only mlp_baseline
    python scripts/benchmark_comparison.py --only mlp_physics
    python scripts/benchmark_comparison.py --only hybrid_physics
    
    # Use GPU
    python scripts/benchmark_comparison.py --device cuda
    
    # Quick test (fewer steps)
    python scripts/benchmark_comparison.py --steps 100
"""
import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import torch
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pinn import PINNConfig, PINNModel, HybridPINNModel


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    model_type: Literal["mlp", "hybrid"]
    physics_enabled: bool
    training_time: float
    final_loss_total: float
    final_loss_cls: float
    final_loss_data: float
    final_loss_phys: float
    final_probs: np.ndarray
    param_count: int
    ess: float  # Effective sample size


def run_benchmark(
    name: str,
    model_type: Literal["mlp", "hybrid"],
    physics_enabled: bool,
    steps: int = 500,
    device: str = "cpu"
) -> BenchmarkResult:
    """Run single benchmark experiment."""
    print(f"\n{'='*70}")
    print(f"Benchmark: {name}")
    print(f"Model: {model_type.upper()}, Physics: {physics_enabled}, Steps: {steps}")
    print(f"{'='*70}\n")
    
    # Configure model
    cfg = PINNConfig(
        seed=42,
        device=device,
        model={
            "model_type": model_type,
            "hidden": 256,
            "layers": 6,
            "learn_eta": False,
            "fourier": {"max_log2_freq": 4, "ramp_frac": 0.5}
        },
        classifier={
            "hidden": 128,
            "dropout": 0.1,
            "horizons": [6, 12, 24],
            "loss_type": "focal",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0
        },
        physics={
            "enable": physics_enabled,
            "resistive": False,
            "lambda_phys_schedule": [
                [0.0, 0.0],
                [0.3, 0.0],
                [0.6, 1.0],
                [0.8, 2.0],
                [1.0, 3.0]
            ] if physics_enabled else [[0.0, 0.0], [1.0, 0.0]]
        },
        collocation={"n_max": 2048, "alpha_start": 0.5, "alpha_end": 0.8},
        train={"steps": steps, "lr": 1e-3, "batch_size": 1, "amp": False},
        data={"use_real": False, "dummy_T": 8, "dummy_H": 64, "dummy_W": 64}
    )
    
    # Create model
    if model_type == "mlp":
        model = PINNModel(cfg).to(device)
    else:
        # Hybrid model: encoder_in_channels = number of observed frames
        # For 8 frames with 6 observed (t <= t0), encoder sees 6 channels
        T_total = 8
        observed_frames = 6  # First 6 frames are observed
        model = HybridPINNModel(cfg, encoder_in_channels=observed_frames).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}\n")
    
    # Create dummy data
    T, H, W = 8, 64, 64
    P_per_t = 256
    
    frames = torch.randn(T, H, W, device=device) * 100
    coords = torch.rand(T, P_per_t, 3, device=device) * 2 - 1
    gt_bz = torch.randn(T, P_per_t, 1, device=device) * 100
    labels = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)  # Flare at 6h
    observed_mask = torch.zeros(T, dtype=torch.bool, device=device)
    observed_mask[:6] = True
    pil_mask = (np.random.rand(H, W) > 0.85).astype(np.float32)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    
    # Training
    model.train()
    start_time = time.time()
    losses = []
    
    for step in range(steps):
        frac = step / steps
        model.set_train_frac(frac)
        
        optimizer.zero_grad()
        
        if model_type == "hybrid":
            output = model(
                coords=coords,
                frames=frames,
                gt_bz=gt_bz,
                observed_mask=observed_mask,
                labels=labels,
                pil_mask=pil_mask if physics_enabled else None,
                mode="train"
            )
        else:
            output = model(
                coords=coords,
                gt_bz=gt_bz,
                observed_mask=observed_mask,
                labels=labels,
                pil_mask=pil_mask if physics_enabled else None,
                mode="train"
            )
        
        output.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        
        losses.append({
            "step": step,
            "total": output.loss_total.item(),
            "cls": output.loss_cls.item(),
            "data": output.loss_data.item(),
            "phys": output.loss_phys.item()
        })
        
        if step % 50 == 0 or step == steps - 1:
            print(f"Step {step:4d}: "
                  f"loss={output.loss_total.item():.4f} "
                  f"cls={output.loss_cls.item():.4f} "
                  f"data={output.loss_data.item():.4f} "
                  f"phys={output.loss_phys.item():.2e} "
                  f"λ={output.lambda_phys:.2f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
    # Need gradients enabled for B_perp calculation in eval mode
    eval_coords = coords.detach().clone()
    eval_coords.requires_grad_(True)
    if model_type == "hybrid":
        final_output = model(eval_coords, frames, gt_bz, observed_mask, labels, mode="eval")
    else:
        final_output = model(eval_coords, gt_bz, observed_mask, labels, mode="eval")
    
    result = BenchmarkResult(
        name=name,
        model_type=model_type,
        physics_enabled=physics_enabled,
        training_time=training_time,
        final_loss_total=losses[-1]["total"],
        final_loss_cls=losses[-1]["cls"],
        final_loss_data=losses[-1]["data"],
        final_loss_phys=losses[-1]["phys"],
        final_probs=final_output.probs.detach().cpu().numpy().flatten(),
        param_count=param_count,
        ess=final_output.ess
    )
    
    print(f"\n✓ Completed in {training_time:.1f}s")
    print(f"  Final predictions: {result.final_probs}")
    print(f"  ESS: {result.ess:.1f}\n")
    
    return result


def print_comparison_table(results: list[BenchmarkResult]):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("BENCHMARK COMPARISON RESULTS")
    print("="*100 + "\n")
    
    # Create dataframe for easy formatting
    data = []
    for r in results:
        data.append({
            "Experiment": r.name,
            "Model": r.model_type.upper(),
            "Physics": "✓" if r.physics_enabled else "✗",
            "Params": f"{r.param_count:,}",
            "Time (s)": f"{r.training_time:.1f}",
            "Loss (Total)": f"{r.final_loss_total:.4f}",
            "Loss (Cls)": f"{r.final_loss_cls:.4f}",
            "Loss (Data)": f"{r.final_loss_data:.4f}",
            "Loss (Phys)": f"{r.final_loss_phys:.2e}",
            "Pred@6h": f"{r.final_probs[0]:.3f}",
            "ESS": f"{r.ess:.0f}"
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()
    
    # Analysis
    print("="*100)
    print("ANALYSIS")
    print("="*100 + "\n")
    
    baseline = next((r for r in results if not r.physics_enabled and r.model_type == "mlp"), None)
    
    for r in results:
        if r == baseline:
            continue
        
        print(f"{r.name} vs Baseline:")
        if baseline:
            time_ratio = r.training_time / baseline.training_time
            cls_improvement = (baseline.final_loss_cls - r.final_loss_cls) / baseline.final_loss_cls * 100
            data_improvement = (baseline.final_loss_data - r.final_loss_data) / baseline.final_loss_data * 100
            
            print(f"  • Training time: {time_ratio:.2f}x {'slower' if time_ratio > 1 else 'faster'}")
            print(f"  • Classification loss: {cls_improvement:+.1f}% {'better' if cls_improvement > 0 else 'worse'}")
            print(f"  • Data fitting: {data_improvement:+.1f}% {'better' if data_improvement > 0 else 'worse'}")
            
            if r.physics_enabled:
                print(f"  • Physics residual: {r.final_loss_phys:.2e}")
                print(f"  • ESS (sampling quality): {r.ess:.1f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark comparison")
    parser.add_argument("--steps", type=int, default=500,
                       help="Training steps per experiment")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to use")
    parser.add_argument("--only", type=str, default=None,
                       choices=["mlp_baseline", "mlp_physics", "hybrid_baseline", "hybrid_physics"],
                       help="Run only specific experiment")
    args = parser.parse_args()
    
    experiments = {
        "mlp_baseline": ("MLP Baseline (Data Only)", "mlp", False),
        "mlp_physics": ("MLP + Physics", "mlp", True),
        "hybrid_baseline": ("Hybrid CNN Baseline", "hybrid", False),
        "hybrid_physics": ("Hybrid CNN + Physics", "hybrid", True),
    }
    
    # Select experiments to run
    if args.only:
        to_run = {args.only: experiments[args.only]}
    else:
        to_run = experiments
    
    # Run benchmarks
    results = []
    for exp_key, (name, model_type, physics) in to_run.items():
        result = run_benchmark(
            name=name,
            model_type=model_type,
            physics_enabled=physics,
            steps=args.steps,
            device=args.device
        )
        results.append(result)
    
    # Print comparison
    print_comparison_table(results)
    
    # Save results
    output_file = Path("benchmark_results.csv")
    df = pd.DataFrame([{
        "name": r.name,
        "model_type": r.model_type,
        "physics": r.physics_enabled,
        "time_s": r.training_time,
        "loss_total": r.final_loss_total,
        "loss_cls": r.final_loss_cls,
        "loss_data": r.final_loss_data,
        "loss_phys": r.final_loss_phys,
        "params": r.param_count,
        "ess": r.ess
    } for r in results])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

