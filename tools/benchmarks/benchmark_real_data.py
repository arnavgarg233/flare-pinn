#!/usr/bin/env python3
"""
Benchmark on REAL SHARP data - compare baseline vs physics, MLP vs Hybrid.

Usage:
    python scripts/benchmark_real_data.py --steps 300
    python scripts/benchmark_real_data.py --steps 300 --only mlp_physics
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
from data.windows_dataset import WindowsDataset, _coords_grid, _bilinear_sample_bz, pil_mask_from_bz


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
    ess: float


def run_benchmark(
    name: str,
    model_type: Literal["mlp", "hybrid"],
    physics_enabled: bool,
    steps: int = 300,
    device: str = "cpu",
    data_config: dict = None
) -> BenchmarkResult:
    """Run single benchmark experiment on real data."""
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
        data=data_config
    )
    
    # Load real data
    print("Loading real SHARP data...")
    windows_df = pd.read_parquet(cfg.data.windows_parquet)
    print(f"Found {len(windows_df)} windows in parquet")
    
    # Take first 10 windows for quick benchmark
    windows_df = windows_df.head(10)
    
    dataset = WindowsDataset(
        windows_df=windows_df,
        frames_meta_path=cfg.data.frames_meta_parquet,
        npz_root=cfg.data.npz_root,
        target_px=cfg.data.target_size,
        input_hours=cfg.data.input_hours,
        horizons=cfg.classifier.horizons,
        P_per_t=cfg.data.P_per_t,
        pil_top_pct=cfg.data.pil_top_pct
    )
    
    if len(dataset) == 0:
        raise ValueError("No data loaded! Check data paths.")
    
    print(f"Loaded {len(dataset)} windows\n")
    
    # Get one sample
    sample = dataset[0]
    coords = sample["coords"]  # [T, P, 3]
    gt_bz = sample["gt_bz"]  # [T, P, 1]
    frames = sample["frames"]  # [T, H, W] - NEW: full frames for hybrid
    labels = sample["labels"]  # [n_horizons]
    observed_mask = sample["observed_mask"]  # [T]
    pil_mask = sample["pil_mask"]  # [H, W] numpy
    
    T, P = coords.shape[:2]
    H, W = frames.shape[-2:]
    
    print(f"Data shapes:")
    print(f"  Coords: {coords.shape}")
    print(f"  Frames: {frames.shape}")
    print(f"  gt_bz: {gt_bz.shape}")
    print(f"  Labels: {labels}")
    print(f"  Observed frames: {observed_mask.sum()}/{T}")
    
    # Move to device and detach (dataset may have grad enabled)
    coords = coords.detach().to(device)
    gt_bz = gt_bz.detach().to(device)
    frames = frames.detach().to(device)
    labels = labels.detach().unsqueeze(0).to(device)  # Add batch dimension
    observed_mask = observed_mask.detach().to(device)
    
    # Create model
    if model_type == "mlp":
        model = PINNModel(cfg).to(device)
    else:
        # Hybrid: encoder sees number of observed frames
        n_obs = observed_mask.sum().item()
        model = HybridPINNModel(cfg, encoder_in_channels=int(n_obs)).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}\n")
    
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
        
        # Create fresh coords with grad for each step
        step_coords = coords.detach().clone()
        step_coords.requires_grad_(True)
        
        if model_type == "hybrid":
            output = model(
                coords=step_coords,
                frames=frames,
                gt_bz=gt_bz,
                observed_mask=observed_mask,
                labels=labels,
                pil_mask=pil_mask if physics_enabled else None,
                mode="train"
            )
        else:
            output = model(
                coords=step_coords,
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
            print(f"Step {step:4d}: loss={output.loss_total.item():.4f} "
                  f"cls={output.loss_cls.item():.4f} "
                  f"data={output.loss_data.item():.4f} "
                  f"phys={output.loss_phys.item():.2e} "
                  f"λ={output.lambda_phys:.2f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
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
    print("BENCHMARK COMPARISON RESULTS (REAL DATA)")
    print("="*100 + "\n")
    
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
            "Pred@12h": f"{r.final_probs[1]:.3f}",
            "Pred@24h": f"{r.final_probs[2]:.3f}",
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
        
        print(f"{r.name} vs MLP Baseline:")
        if baseline:
            time_ratio = r.training_time / baseline.training_time
            cls_improvement = (baseline.final_loss_cls - r.final_loss_cls) / baseline.final_loss_cls * 100
            data_improvement = (baseline.final_loss_data - r.final_loss_data) / baseline.final_loss_data * 100
            
            print(f"  • Training time: {time_ratio:.2f}x {'slower' if time_ratio > 1 else 'faster'}")
            print(f"  • Classification loss: {cls_improvement:+.1f}%")
            print(f"  • Data fitting: {data_improvement:+.1f}%")
            
            if r.physics_enabled:
                print(f"  • Physics residual: {r.final_loss_phys:.2e}")
                print(f"  • ESS: {r.ess:.1f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark on real SHARP data")
    parser.add_argument("--steps", type=int, default=300,
                       help="Training steps per experiment")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to use")
    parser.add_argument("--only", type=str, default=None,
                       choices=["mlp_baseline", "mlp_physics", "hybrid_baseline", "hybrid_physics"],
                       help="Run only specific experiment")
    args = parser.parse_args()
    
    # Real data config
    data_config = {
        "use_real": True,
        "windows_parquet": "data/interim/windows_train.parquet",
        "frames_meta_parquet": "S:/flare_forecasting/frames_meta.parquet",
        "npz_root": "S:/flare_forecasting",
        "target_size": 128,  # Reduced from 256 for hybrid (memory)
        "input_hours": 8,  # Reduced from 48 for hybrid (memory - encodes all frames)
        "P_per_t": 256,  # Reduced from 512 for hybrid (memory)
        "pil_top_pct": 0.15,
        "dummy_T": 8,
        "dummy_H": 64,
        "dummy_W": 64
    }
    
    experiments = {
        "mlp_baseline": ("MLP Baseline (Data Only)", "mlp", False),
        "mlp_physics": ("MLP + Physics", "mlp", True),
        "hybrid_baseline": ("Hybrid CNN Baseline", "hybrid", False),
        "hybrid_physics": ("Hybrid CNN + Physics", "hybrid", True),
    }
    
    # Select experiments
    if args.only:
        to_run = {args.only: experiments[args.only]}
    else:
        to_run = experiments
    
    # Run benchmarks
    results = []
    for exp_key, (name, model_type, physics) in to_run.items():
        try:
            result = run_benchmark(
                name=name,
                model_type=model_type,
                physics_enabled=physics,
                steps=args.steps,
                device=args.device,
                data_config=data_config
            )
            results.append(result)
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}\n")
            continue
    
    if results:
        # Print comparison
        print_comparison_table(results)
        
        # Save results
        output_file = Path("benchmark_results_real_data.csv")
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
            "ess": r.ess,
            "pred_6h": r.final_probs[0],
            "pred_12h": r.final_probs[1],
            "pred_24h": r.final_probs[2]
        } for r in results])
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

