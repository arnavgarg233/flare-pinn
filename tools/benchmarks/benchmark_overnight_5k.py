#!/usr/bin/env python3
"""
Overnight benchmark: MLP vs Hybrid (5K steps each) on real SHARP data.

Compares to SOTA baselines:
- Nishizuka et al. (2021): TSS ~0.50-0.55 @ 24h
- Florios et al. (2018): TSS ~0.45-0.50 @ 12h
- Benchmark goal: TSS > 0.40 for proof-of-concept

This will run ~6-8 hours total on CPU.
"""
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.pinn import PINNConfig, PINNModel, HybridPINNModel
from src.data.windows_dataset import WindowsDataset
from src.models.eval.metrics import sweep_tss, pr_auc, brier_score, adaptive_ece


# SOTA Benchmarks for comparison (realistic values based on literature)
SOTA_BENCHMARKS = {
    # State-of-the-art deep learning performance
    6: {'tss': 0.55, 'source': 'Modern DL SOTA (2021-24)'},
    12: {'tss': 0.60, 'source': 'Modern DL SOTA (2021-24)'},
    24: {'tss': 0.65, 'source': 'Modern DL SOTA (2021-24)'},
}

# Historical baselines for reference
BASELINE_BENCHMARKS = {
    6: {'tss': 0.40, 'source': 'Operational baseline'},
    12: {'tss': 0.48, 'source': 'Florios+ 2018'},
    24: {'tss': 0.53, 'source': 'Nishizuka+ 2021'},
}


def setup_logger(log_path):
    """Just print start time, no file logging."""
    print(f"\n{'='*70}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


def load_real_data(cfg: PINNConfig, use_full_train=False):
    """Load SHARP data with path validation."""
    print("Loading SHARP windows...")
    
    # Validate paths
    train_path = Path('data/interim/windows_train.parquet')
    val_path = Path('data/interim/windows_validation.parquet')
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path.absolute()}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path.absolute()}")
    
    windows_train = pd.read_parquet(train_path)
    windows_val = pd.read_parquet(val_path)
    
    print(f"  Total train: {len(windows_train)}")
    print(f"  Total val: {len(windows_val)}")
    
    # Use subset for speed (still representative)
    if not use_full_train:
        n_train = min(1000, len(windows_train))
        windows_train = windows_train.sample(n=n_train, random_state=42).reset_index(drop=True)
    
    n_val = min(200, len(windows_val))
    windows_val = windows_val.sample(n=n_val, random_state=42).reset_index(drop=True)
    
    print(f"  Using {len(windows_train)} train, {len(windows_val)} val")
    
    # Compute class balance
    for h in cfg.classifier.horizons:
        col = f'y_geq_M_{h}h'
        if col in windows_train.columns:
            pos_rate = windows_train[col].mean()
            print(f"    {h}h positive rate: {pos_rate:.3f}")
    
    # Validate data paths from config
    frames_meta_path = Path(cfg.data.frames_meta_parquet)
    npz_root = Path(cfg.data.npz_root)
    
    if not frames_meta_path.exists():
        raise FileNotFoundError(f"Frames metadata not found: {frames_meta_path.absolute()}")
    if not npz_root.exists():
        raise FileNotFoundError(f"NPZ root directory not found: {npz_root.absolute()}")
    
    # Create datasets using config parameters
    train_dataset = WindowsDataset(
        windows_df=windows_train,
        frames_meta_path=str(frames_meta_path),
        npz_root=str(npz_root),
        target_px=cfg.data.target_size,
        input_hours=cfg.data.input_hours,
        horizons=list(cfg.classifier.horizons),
        P_per_t=cfg.data.P_per_t,
        pil_top_pct=cfg.data.pil_top_pct,
    )
    
    val_dataset = WindowsDataset(
        windows_df=windows_val,
        frames_meta_path=str(frames_meta_path),
        npz_root=str(npz_root),
        target_px=cfg.data.target_size,
        input_hours=cfg.data.input_hours,
        horizons=list(cfg.classifier.horizons),
        P_per_t=cfg.data.P_per_t,
        pil_top_pct=cfg.data.pil_top_pct,
    )
    
    return train_dataset, val_dataset


def train_model(model, train_loader, cfg, n_steps=5000, device='cpu', checkpoint_dir=None):
    """Train model with curriculum and logging."""
    print(f"\nTraining for {n_steps} steps...")
    
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)
    
    step = 0
    losses = {'total': [], 'cls': [], 'data': [], 'phys': []}
    
    start_time = time.time()
    
    while step < n_steps:
        for batch in train_loader:
            if step >= n_steps:
                break
            
            # Move to device and ensure coords has requires_grad
            coords = batch["coords"].to(device).requires_grad_(True)
            gt_bz = batch["gt_bz"].to(device)
            observed_mask = batch["observed_mask"].to(device)
            labels = batch["labels"].to(device)
            pil_mask = batch.get("pil_mask")
            frames = batch.get("frames")
            if frames is not None:
                frames = frames.to(device)
            
            # Curriculum
            frac = step / max(1, n_steps)
            model.set_train_frac(frac)
            
            # Forward
            optimizer.zero_grad(set_to_none=True)
            
            B = coords.shape[0]
            total_loss = 0.0
            
            for i in range(B):
                forward_kwargs = {
                    "coords": coords[i],
                    "gt_bz": gt_bz[i],
                    "observed_mask": observed_mask[i],
                    "labels": labels[i:i+1],
                    "pil_mask": pil_mask[i].cpu().numpy() if isinstance(pil_mask, torch.Tensor) else pil_mask,
                    "mode": "train"
                }
                
                # Frames for hybrid
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
            scheduler.step()
            
            # Check for NaN
            if not torch.isfinite(loss):
                print(f"\n!!! NaN/Inf detected at step {step} !!!")
                print(f"    loss_total: {out.loss_total.item()}")
                print(f"    loss_cls: {out.loss_cls.item()}")
                print(f"    loss_data: {out.loss_data.item()}")
                print(f"    loss_phys: {out.loss_phys.item()}")
                print("    STOPPING TRAINING")
                break
            
            # Log
            losses['total'].append(float(out.loss_total.item()))
            losses['cls'].append(float(out.loss_cls.item()))
            losses['data'].append(float(out.loss_data.item()))
            losses['phys'].append(float(out.loss_phys.item()))
            
            if step % 100 == 0:
                elapsed = time.time() - start_time
                lr = optimizer.param_groups[0]['lr']
                print(f"  [{step:4d}/{n_steps}] loss={loss.item():.4f} "
                      f"(cls={out.loss_cls.item():.4f}, data={out.loss_data.item():.4f}, "
                      f"phys={out.loss_phys.item():.4f}) | "
                      f"λ={out.lambda_phys:.2f}, lr={lr:.2e}, t={elapsed:.1f}s")
                start_time = time.time()
            
            # Checkpoint
            if checkpoint_dir and step % 1000 == 0 and step > 0:
                ckpt_path = Path(checkpoint_dir) / f"step_{step}.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                }, ckpt_path)
                print(f"    Saved checkpoint: {ckpt_path}")
            
            step += 1
    
    return losses


@torch.no_grad()
def evaluate_model(model, val_loader, horizons, device='cpu'):
    """Full evaluation on validation set.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        horizons: List of prediction horizons (e.g., [6, 12, 24])
        device: Device to run on
    """
    print("\nEvaluating on validation set...")
    
    model.eval()
    
    all_probs = []
    all_labels = []
    
    for batch_idx, batch in enumerate(val_loader):
        coords = batch["coords"].to(device)
        observed_mask = batch["observed_mask"].to(device)
        labels = batch["labels"].to(device)
        frames = batch.get("frames")
        if frames is not None:
            frames = frames.to(device)
        
        B = coords.shape[0]
        for i in range(B):
            # Enable gradients for coords even in eval mode (needed for B_perp_from_Az)
            coords_i = coords[i].clone().detach().requires_grad_(True)
            
            forward_kwargs = {
                "coords": coords_i,
                "observed_mask": observed_mask[i],
                "mode": "eval"
            }
            
            if frames is not None and hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
                forward_kwargs["frames"] = frames[i]
            
            out = model(**forward_kwargs)
            
            all_probs.append(out.probs.cpu().numpy())
            all_labels.append(labels[i:i+1].cpu().numpy())
    
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Compute metrics
    results = {}
    
    print(f"\n{'='*70}")
    print("VALIDATION METRICS")
    print(f"{'='*70}")
    
    for j, h in enumerate(horizons):
        y_true = labels[:, j]
        y_prob = probs[:, j]
        
        if y_true.sum() == 0:
            print(f"\nHorizon {h}h: SKIPPED (no positive samples)")
            continue
        
        thr_tss, tss_val = sweep_tss(y_true, y_prob, n=512)
        prauc = pr_auc(y_true, y_prob)
        bs = brier_score(y_true, y_prob)
        ece = adaptive_ece(y_true, y_prob, n_bins=10)
        
        results[h] = {
            'tss': tss_val,
            'threshold': thr_tss,
            'pr_auc': prauc,
            'brier': bs,
            'ece': ece,
            'n_pos': int(y_true.sum()),
            'n_total': len(y_true)
        }
        
        # Compare to SOTA
        sota = SOTA_BENCHMARKS.get(h, {})
        sota_tss = sota.get('tss', 0)
        delta = tss_val - sota_tss
        
        print(f"\n{h}h Horizon:")
        print(f"  TSS:        {tss_val:.4f} @ threshold={thr_tss:.3f}")
        print(f"  SOTA:       {sota_tss:.4f} ({sota.get('source', 'N/A')})")
        print(f"  Delta:      {delta:+.4f} {'✓' if delta > -0.05 else '⚠'}")
        print(f"  PR-AUC:     {prauc:.4f}")
        print(f"  Brier:      {bs:.4f}")
        print(f"  ECE:        {ece:.4f}")
        print(f"  Positives:  {int(y_true.sum())}/{len(y_true)}")
    
    print(f"{'='*70}")
    
    return results


def benchmark_model(model_type, device='cpu', config_override=None):
    """Full 5K step benchmark.
    
    Args:
        model_type: 'mlp' or 'hybrid'
        device: 'cpu' or 'cuda'
        config_override: Optional dict to override config values
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {model_type.upper()} + PHYSICS (5K steps)")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load config from YAML file
    config_path = Path(f"configs/{model_type}_5k.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.absolute()}")
    
    print(f"Loading config from: {config_path}")
    cfg = PINNConfig.from_yaml(config_path)
    
    # Override device
    cfg.device = device
    
    # Apply config overrides (e.g., reduce steps for quick test)
    if config_override:
        for key, value in config_override.items():
            if key == 'train' and isinstance(value, dict):
                # Override train config attributes
                for k, v in value.items():
                    if hasattr(cfg.train, k):
                        setattr(cfg.train, k, v)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
    
    # Print config summary
    print("\nConfiguration:")
    for key, value in cfg.model_dump_summary().items():
        print(f"  {key}: {value}")
    
    # Model
    if model_type == 'hybrid':
        model = HybridPINNModel(cfg, encoder_in_channels=1).to(device)
    else:
        model = PINNModel(cfg).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Data
    train_dataset, val_dataset = load_real_data(cfg, use_full_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Train
    checkpoint_dir = f"checkpoints/overnight_{model_type}_{timestamp}"
    start = time.time()
    losses = train_model(model, train_loader, cfg, n_steps=cfg.train.steps, device=device,
                         checkpoint_dir=checkpoint_dir)
    train_time = time.time() - start
    
    print(f"\nTraining complete: {train_time/3600:.2f} hours")
    print(f"  Loss improvement: {losses['total'][0]:.4f} → {losses['total'][-1]:.4f}")
    
    # Evaluate
    results = evaluate_model(model, val_loader, horizons=list(cfg.classifier.horizons), device=device)
    
    # Save final checkpoint
    final_path = Path(checkpoint_dir) / "final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg.model_dump(),
        'losses': losses,
        'metrics': results,
        'timestamp': timestamp,
    }, final_path)
    print(f"\nSaved final model: {final_path}")
    
    return {
        'model_type': model_type,
        'train_time': train_time,
        'losses': losses,
        'metrics': results,
        'checkpoint': str(final_path),
        'config': cfg.model_dump_summary(),
        'horizons': list(cfg.classifier.horizons),
    }


def main():
    """Run overnight benchmark."""
    device = 'cpu'
    
    setup_logger(None)
    
    print("OVERNIGHT BENCHMARK: 5K Steps Each Model")
    print("Estimated time: 6-8 hours total on CPU")
    print(f"Device: {device}")
    print()
    
    try:
        # Run both models
        mlp_results = benchmark_model('mlp', device=device)
        hybrid_results = benchmark_model('hybrid', device=device)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure:")
        print("  1. Config files exist: configs/mlp_5k.yaml, configs/hybrid_5k.yaml")
        print("  2. Data files exist: data/interim/windows_train.parquet, data/interim/windows_validation.parquet")
        print("  3. SHARP data paths are correct in config files")
        return
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise
    
    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    
    print(f"\nTraining Time:")
    print(f"  MLP:    {mlp_results['train_time']/3600:.2f}h")
    print(f"  Hybrid: {hybrid_results['train_time']/3600:.2f}h")
    
    print(f"\nConfiguration:")
    print(f"  MLP:    {mlp_results['config']}")
    print(f"  Hybrid: {hybrid_results['config']}")
    
    # Use horizons from results (should be same for both)
    horizons = mlp_results.get('horizons', [6, 12, 24])
    
    print(f"\nTSS vs SOTA:")
    for h in horizons:
        if h in mlp_results['metrics'] and h in hybrid_results['metrics']:
            mlp_tss = mlp_results['metrics'][h]['tss']
            hyb_tss = hybrid_results['metrics'][h]['tss']
            sota = SOTA_BENCHMARKS.get(h, {})
            sota_tss = sota.get('tss', 0.0)
            baseline = BASELINE_BENCHMARKS.get(h, {})
            baseline_tss = baseline.get('tss', 0.0)
            
            print(f"\n{h}h Horizon:")
            print(f"  MLP:      {mlp_tss:.4f} ({(mlp_tss/sota_tss-1)*100:+.1f}% vs SOTA)" if sota_tss > 0 else f"  MLP:      {mlp_tss:.4f}")
            print(f"  Hybrid:   {hyb_tss:.4f} ({(hyb_tss/sota_tss-1)*100:+.1f}% vs SOTA)" if sota_tss > 0 else f"  Hybrid:   {hyb_tss:.4f}")
            if sota_tss > 0:
                print(f"  SOTA:     {sota_tss:.4f} ({sota.get('source', 'N/A')})")
            if baseline_tss > 0:
                print(f"  Baseline: {baseline_tss:.4f} ({baseline.get('source', 'N/A')})")
            print(f"  Winner:   {'Hybrid' if hyb_tss > mlp_tss else 'MLP' if mlp_tss > hyb_tss else 'Tie'} (Δ={abs(hyb_tss-mlp_tss):.4f})")
    
    print(f"\nCheckpoints saved:")
    print(f"  MLP:    {mlp_results['checkpoint']}")
    print(f"  Hybrid: {hybrid_results['checkpoint']}")
    
    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

