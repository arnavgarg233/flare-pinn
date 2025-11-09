#!/usr/bin/env python3
"""
Quick benchmark on REAL SHARP data: MLP vs Hybrid (both with physics).

Trains each model for a short run, then evaluates TSS on validation set.
This tests the hybrid CNN-PINN fix on actual solar flare data.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.models.pinn import PINNConfig, PINNModel, HybridPINNModel
from src.data.windows_dataset import WindowsDataset
from src.models.eval.metrics import sweep_tss, pr_auc, brier_score


def load_real_data(cfg, max_samples=200):
    """Load real SHARP data for training and validation."""
    print("Loading real SHARP data...")
    
    # Load windows
    windows_train = pd.read_parquet('data/interim/windows_train.parquet')
    windows_val = pd.read_parquet('data/interim/windows_validation.parquet')
    
    print(f"  Total train windows: {len(windows_train)}")
    print(f"  Total val windows: {len(windows_val)}")
    
    # Subsample for quick test
    if len(windows_train) > max_samples:
        windows_train = windows_train.sample(n=max_samples, random_state=42).reset_index(drop=True)
    if len(windows_val) > max_samples // 2:
        windows_val = windows_val.sample(n=max_samples // 2, random_state=42).reset_index(drop=True)
    
    print(f"  Using {len(windows_train)} train, {len(windows_val)} val samples")
    
    # Create datasets
    train_dataset = WindowsDataset(
        windows_df=windows_train,
        frames_meta_path='S:/flare_forecasting/frames_meta.parquet',
        npz_root='S:/flare_forecasting',
        target_px=128,  # Smaller for speed
        input_hours=24,  # Shorter window for speed
        horizons=[6, 12, 24],
        P_per_t=512,  # Fewer points for speed
        pil_top_pct=0.15,
    )
    
    val_dataset = WindowsDataset(
        windows_df=windows_val,
        frames_meta_path='S:/flare_forecasting/frames_meta.parquet',
        npz_root='S:/flare_forecasting',
        target_px=128,
        input_hours=24,
        horizons=[6, 12, 24],
        P_per_t=512,
        pil_top_pct=0.15,
    )
    
    return train_dataset, val_dataset


def train_model(model, train_loader, cfg, n_steps=500, device='cpu'):
    """Quick training run."""
    print(f"\nTraining for {n_steps} steps...")
    
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr)
    
    step = 0
    epoch = 0
    losses = []
    
    start_time = time.time()
    
    while step < n_steps:
        for batch in train_loader:
            if step >= n_steps:
                break
            
            # Move to device
            coords = batch["coords"].to(device, non_blocking=True)
            gt_bz = batch["gt_bz"].to(device, non_blocking=True)
            observed_mask = batch["observed_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pil_mask = batch.get("pil_mask")
            frames = batch.get("frames")
            if frames is not None:
                frames = frames.to(device, non_blocking=True)
            
            # Update curriculum
            frac = min(1.0, step / max(1, n_steps))
            model.set_train_frac(frac)
            
            # Forward
            optimizer.zero_grad(set_to_none=True)
            
            # Handle batch dimension
            B = coords.shape[0]
            total_loss = 0.0
            
            for i in range(B):
                coords_i = coords[i]
                gt_bz_i = gt_bz[i]
                observed_mask_i = observed_mask[i]
                labels_i = labels[i:i+1]
                pil_mask_i = pil_mask[i].cpu().numpy() if isinstance(pil_mask, torch.Tensor) else pil_mask
                
                forward_kwargs = {
                    "coords": coords_i,
                    "gt_bz": gt_bz_i,
                    "observed_mask": observed_mask_i,
                    "labels": labels_i,
                    "pil_mask": pil_mask_i,
                    "mode": "train"
                }
                
                # Only add frames for hybrid model
                if frames is not None and hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
                    forward_kwargs["frames"] = frames[i]
                
                out = model(**forward_kwargs)
                total_loss += out.loss_total / B
            
            loss = total_loss
            
            # Backward
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            
            losses.append(float(loss.item()))
            
            if step % 50 == 0 or step == n_steps - 1:
                elapsed = time.time() - start_time
                print(f"  Step {step:4d}/{n_steps}: loss={loss.item():.4f}, "
                      f"time={elapsed:.1f}s")
                start_time = time.time()
            
            step += 1
        
        epoch += 1
    
    return losses


@torch.no_grad()
def evaluate_model(model, val_loader, device='cpu'):
    """Evaluate on validation set."""
    print("\nEvaluating...")
    
    model.eval()
    
    all_probs = []
    all_labels = []
    
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= 50:  # Limit evaluation for speed
            break
        
        coords = batch["coords"].to(device)
        observed_mask = batch["observed_mask"].to(device)
        labels = batch["labels"].to(device)
        frames = batch.get("frames")
        if frames is not None:
            frames = frames.to(device)
        
        # Process each sample
        B = coords.shape[0]
        for i in range(B):
            coords_i = coords[i]
            observed_mask_i = observed_mask[i]
            
            # Enable gradients on coords for B_perp computation in eval mode
            coords_i = coords_i.requires_grad_(True)
            
            forward_kwargs = {
                "coords": coords_i,
                "observed_mask": observed_mask_i,
                "mode": "eval"
            }
            
            # Only add frames for hybrid model
            if frames is not None and hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
                forward_kwargs["frames"] = frames[i]
            
            out = model(**forward_kwargs)
            
            all_probs.append(out.probs.cpu().numpy())
            all_labels.append(labels[i:i+1].cpu().numpy())
    
    # Concatenate
    probs = np.concatenate(all_probs, axis=0)  # [N, n_horizons]
    labels = np.concatenate(all_labels, axis=0)  # [N, n_horizons]
    
    # Compute metrics per horizon
    results = {}
    horizons = [6, 12, 24]
    
    for j, h in enumerate(horizons):
        y_true = labels[:, j]
        y_prob = probs[:, j]
        
        if y_true.sum() == 0:
            print(f"  Horizon {h}h: No positive samples")
            continue
        
        thr_tss, tss_val = sweep_tss(y_true, y_prob, n=256)
        prauc = pr_auc(y_true, y_prob)
        bs = brier_score(y_true, y_prob)
        
        results[h] = {
            'tss': tss_val,
            'tss_threshold': thr_tss,
            'pr_auc': prauc,
            'brier': bs,
            'n_pos': int(y_true.sum()),
            'n_total': len(y_true)
        }
        
        print(f"  Horizon {h:2d}h: TSS={tss_val:.3f} @ thr={thr_tss:.2f}, "
              f"PR-AUC={prauc:.3f}, Brier={bs:.3f}, "
              f"pos={int(y_true.sum())}/{len(y_true)}")
    
    return results


def benchmark_model(model_type, device='cpu'):
    """Benchmark a single model."""
    print("\n" + "="*70)
    print(f"BENCHMARK: {model_type.upper()} + PHYSICS")
    print("="*70)
    
    # Create config
    cfg = PINNConfig(
        seed=42,
        device=device,
        model=dict(
            model_type=model_type,
            hidden=256,
            layers=8,
            learn_eta=False,
            eta_scalar=0.01,
            fourier=dict(max_log2_freq=4, ramp_frac=0.5)
        ),
        classifier=dict(
            hidden=128,
            dropout=0.1,
            horizons=[6, 12, 24],
            loss_type='focal',
            focal_alpha=0.25,
            focal_gamma=2.0
        ),
        physics=dict(
            enable=True,
            resistive=False,
            boundary_terms=False,
            # MUCH gentler schedule for short training
            lambda_phys_schedule=[[0.0, 0.0], [0.5, 0.0], [0.8, 0.2], [1.0, 0.5]]
        ),
        loss_weights=dict(cls=1.0, data=1.0, curl_consistency=0.0),
        train=dict(
            steps=500,
            batch_size=1,
            lr=5e-4,  # Lower LR for stability with physics
            grad_clip=0.5,  # Tighter clipping
            amp=False
        )
    )
    
    # Create model
    if model_type == 'hybrid':
        model = HybridPINNModel(cfg, encoder_in_channels=1).to(device)
    else:
        model = PINNModel(cfg).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data
    try:
        train_dataset, val_dataset = load_real_data(cfg, max_samples=200)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        print("Make sure SHARP data is accessible at S:/flare_forecasting/")
        return None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Train
    start_time = time.time()
    losses = train_model(model, train_loader, cfg, n_steps=500, device=device)
    train_time = time.time() - start_time
    
    print(f"\nTraining complete: {train_time:.1f}s")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    # Evaluate
    start_time = time.time()
    results = evaluate_model(model, val_loader, device=device)
    eval_time = time.time() - start_time
    
    print(f"\nEvaluation complete: {eval_time:.1f}s")
    
    return {
        'model_type': model_type,
        'train_time': train_time,
        'eval_time': eval_time,
        'losses': losses,
        'metrics': results
    }


def main():
    """Run benchmark on both models."""
    print("\n" + "="*70)
    print("REAL DATA BENCHMARK: MLP vs HYBRID (both with physics)")
    print("="*70)
    print("\nThis will:")
    print("  1. Load 200 real SHARP magnetogram windows")
    print("  2. Train each model for 500 steps")
    print("  3. Evaluate TSS, PR-AUC, Brier on validation set")
    print("  4. Compare the two models")
    print("\n" + "="*70)
    
    device = 'cpu'
    
    # Benchmark MLP
    mlp_results = benchmark_model('mlp', device=device)
    
    # Benchmark Hybrid
    hybrid_results = benchmark_model('hybrid', device=device)
    
    # Compare
    if mlp_results and hybrid_results:
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        
        print("\nTraining Time:")
        print(f"  MLP:    {mlp_results['train_time']:.1f}s")
        print(f"  Hybrid: {hybrid_results['train_time']:.1f}s")
        
        print("\nTSS Scores (higher is better):")
        for h in [6, 12, 24]:
            if h in mlp_results['metrics'] and h in hybrid_results['metrics']:
                mlp_tss = mlp_results['metrics'][h]['tss']
                hyb_tss = hybrid_results['metrics'][h]['tss']
                delta = hyb_tss - mlp_tss
                winner = "Hybrid" if delta > 0 else "MLP" if delta < 0 else "Tie"
                print(f"  {h:2d}h: MLP={mlp_tss:.3f}, Hybrid={hyb_tss:.3f}, "
                      f"Δ={delta:+.3f} ({winner})")
        
        print("\nPR-AUC (higher is better):")
        for h in [6, 12, 24]:
            if h in mlp_results['metrics'] and h in hybrid_results['metrics']:
                mlp_pr = mlp_results['metrics'][h]['pr_auc']
                hyb_pr = hybrid_results['metrics'][h]['pr_auc']
                delta = hyb_pr - mlp_pr
                print(f"  {h:2d}h: MLP={mlp_pr:.3f}, Hybrid={hyb_pr:.3f}, Δ={delta:+.3f}")
        
        print("\n" + "="*70)
        
        # Overall winner
        mlp_avg_tss = np.mean([mlp_results['metrics'][h]['tss'] 
                               for h in [6, 12, 24] if h in mlp_results['metrics']])
        hyb_avg_tss = np.mean([hybrid_results['metrics'][h]['tss'] 
                               for h in [6, 12, 24] if h in hybrid_results['metrics']])
        
        print(f"\nAverage TSS:")
        print(f"  MLP:    {mlp_avg_tss:.3f}")
        print(f"  Hybrid: {hyb_avg_tss:.3f}")
        
        if hyb_avg_tss > mlp_avg_tss:
            improvement = (hyb_avg_tss - mlp_avg_tss) / mlp_avg_tss * 100
            print(f"\n✓ Hybrid wins by {improvement:.1f}% (fix working!)")
        elif mlp_avg_tss > hyb_avg_tss:
            print(f"\n⚠ MLP wins (hybrid may need more training)")
        else:
            print(f"\n- Tie (both models similar)")
        
        print("="*70)


if __name__ == '__main__':
    main()

