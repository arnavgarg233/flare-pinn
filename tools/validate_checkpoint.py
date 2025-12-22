#!/usr/bin/env python3
"""
Validate a checkpoint using the EXACT same logic as train.py evaluation.
Supports TTA (Test-Time Augmentation) by injecting noise.
"""
from __future__ import annotations
import argparse
import gc
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore', message='.*MPS autocast.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.amp')

from src.models.pinn import PINNConfig, HybridPINNModel, PINNModel
from src.models.eval.metrics import (
    pr_auc,
    brier_score,
    adaptive_ece,
    sweep_tss,
    tss_at_threshold,
    select_threshold_at_far,
)
from src.utils.memory_optimization import low_memory_mode


def get_device(requested: str) -> torch.device:
    """Get device (exactly as in train.py)."""
    if requested == "mps" and torch.backends.mps.is_available(): 
        return torch.device("mps")
    if requested.startswith("cuda") and torch.cuda.is_available(): 
        return torch.device(requested)
    return torch.device("cpu")


def load_model(cfg: PINNConfig, checkpoint_path: Path, device: torch.device):
    """Load model and checkpoint (exactly as train.py does)."""
    # Create model
    if cfg.model.model_type == "hybrid":
        model = HybridPINNModel(cfg, encoder_in_channels=None).to(device)
    else:
        model = PINNModel(cfg).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {n_params:,}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Load EMA if available
    ema = None
    if 'ema_state_dict' in checkpoint:
        from src.utils.training_utils import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model, decay=cfg.train.ema_decay)
        try:
            ema.load_state_dict(checkpoint['ema_state_dict'])
            print("✅ EMA weights loaded from checkpoint")
        except Exception as e:
            print(f"⚠️  EMA load failed: {e}")
            ema = None
    else:
        print("⚠️  No EMA in checkpoint - using raw model weights")
    
    step = checkpoint.get('step', 0)
    metric = checkpoint.get('metric', 0.0)
    print(f"Checkpoint: step {step}, metric={metric:.4f}")
    
    return model, ema


def create_validation_loader(cfg: PINNConfig) -> DataLoader:
    """Create validation loader (exactly as train.py does)."""
    from src.utils.masked_training import load_windows_with_mask
    from src.data.consolidated_dataset import ConsolidatedWindowsDataset
    from src.data.cached_dataset import CachedWindowsDataset
    
    print("Loading validation data...")
    df, mask = load_windows_with_mask(str(cfg.data.windows_parquet))
    df = df[mask].reset_index(drop=True)
    
    # USE ALL DATA (no train/val split for test set validation)
    df['t0'] = pd.to_datetime(df['t0'])
    df = df.sort_values('t0', kind='stable').reset_index(drop=True)
    
    val_df = df.reset_index(drop=True)
    
    print(f"  Val: {len(val_df)} windows (using all data)")
    
    if cfg.data.use_consolidated:
        val_ds = ConsolidatedWindowsDataset(
            windows_df=val_df,
            consolidated_dir=str(cfg.data.consolidated_dir),
            target_px=cfg.data.target_size,
            input_hours=cfg.data.input_hours,
            horizons=list(cfg.classifier.horizons),
            P_per_t=cfg.data.P_per_t,
            pil_top_pct=cfg.data.pil_top_pct,
            training=False,
            augment=False,
            max_cached_harps=500,
            use_pil_evolution=getattr(cfg.data, 'use_pil_evolution', True),
            use_temporal_statistics=getattr(cfg.data, 'use_temporal_statistics', True),
        )
    else:
        val_ds = CachedWindowsDataset(
            val_df, str(cfg.data.frames_meta_parquet), str(cfg.data.npz_root),
            target_px=cfg.data.target_size, input_hours=cfg.data.input_hours,
            horizons=list(cfg.classifier.horizons), P_per_t=cfg.data.P_per_t,
            pil_top_pct=cfg.data.pil_top_pct,
            training=False, augment=False, preload=False
        )
    
    use_pin_memory = False if cfg.device == "mps" else True
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=use_pin_memory,
    )
    return val_loader


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    horizons: tuple[int, ...],
    device: torch.device,
    ema: Optional["ExponentialMovingAverage"] = None,
    tta_noise: float = 0.0,
    tta_samples: int = 1
) -> dict:
    """
    Run evaluation with optional TTA.
    """
    model.eval()
    all_probs, all_labels = [], []
    
    ema_context = ema.average_parameters() if ema is not None else None
    
    with torch.no_grad(), low_memory_mode():
        if ema_context is not None:
            ema_context.__enter__()
        
        try:
            for batch_idx, batch in enumerate(loader):
                if batch_idx % 100 == 0:
                    print(f"  Batch {batch_idx}/{len(loader)}...")
                
                labels_original = batch["labels"].clone()
                
                # Move inputs to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                
                B = batch["coords"].shape[0]
                batch_probs = []
                
                # Process each sample in batch
                for i in range(B):
                    # Base arguments
                    base_kwargs = {
                        "coords": batch["coords"][i],
                        "gt_bz": batch["gt_bz"][i],
                        "observed_mask": batch["observed_mask"][i],
                        "labels": batch["labels"][i:i+1],
                        "pil_mask": batch["pil_mask"][i],
                        "mode": "eval",
                    }
                    if "scalars" in batch:
                        base_kwargs["scalars"] = batch["scalars"][i]
                    
                    # TTA Loop (1 sample if no TTA)
                    sample_probs_sum = 0.0
                    
                    frames_orig = batch["frames"][i] if "frames" in batch else None
                    
                    for _ in range(tta_samples):
                        sample_kwargs = base_kwargs.copy()
                        
                        if frames_orig is not None:
                            if tta_noise > 0:
                                noise = torch.randn_like(frames_orig) * tta_noise
                                sample_kwargs["frames"] = frames_orig + noise
                            else:
                                sample_kwargs["frames"] = frames_orig
                        
                        out = model(**sample_kwargs)
                        sample_probs_sum += out.probs
                    
                    batch_probs.append(sample_probs_sum / tta_samples)
                
                batch_probs_cat = torch.cat(batch_probs).cpu().numpy()
                batch_labels_np = labels_original.numpy()
                
                all_probs.append(batch_probs_cat)
                all_labels.append(batch_labels_np)
                
                if batch_idx % 100 == 0:
                    gc.collect()
                    if device.type == "mps":
                        torch.mps.empty_cache()
                        
        finally:
            if ema_context is not None:
                ema_context.__exit__(None, None, None)
    
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS (TTA noise={tta_noise}, samples={tta_samples})")
    print(f"{'='*60}")
    
    results = {"max_tss": 0.0, "horizons": {}}
    
    for j, h in enumerate(horizons):
        y_true = labels[:, j]
        y_prob = probs[:, j]
        
        # Filter out NaN AND Inf values (isfinite catches both!)
        valid_mask = np.isfinite(y_true) & np.isfinite(y_prob)
        y_true = y_true[valid_mask]
        y_prob = y_prob[valid_mask]
        
        # Clamp probabilities to valid range [0, 1]
        y_prob = np.clip(y_prob, 0.0, 1.0)
        
        # Ensure labels are binary (0 or 1)
        y_true = np.clip(y_true, 0.0, 1.0)
        
        if len(y_true) == 0 or y_true.sum() == 0: 
            continue
            
        # Check specific threshold from logs
        if h == 24:
            log_thr = 0.24
            tss_log = tss_at_threshold(y_true, y_prob, log_thr)
            print(f"  [Check] TSS@{log_thr:.2f} = {tss_log:.5f}")

        # ✅ FIX: Use default n (training uses sweep_tss with no n param = 1024)
        thr_tss, tss_val = sweep_tss(y_true, y_prob)
        results["max_tss"] = max(results["max_tss"], tss_val)
        prauc = pr_auc(y_true, y_prob)
        bs = brier_score(y_true, y_prob)
        ece = adaptive_ece(y_true, y_prob)
        
        # Count positives correctly - y_true should be 0.0 or 1.0
        n_pos = int((y_true > 0.5).sum())
        
        # ✅ FIX: Safety clamp BS and ECE to reasonable ranges (same as training)
        bs = float(np.clip(bs, 0.0, 1.0))
        ece = float(np.clip(ece, 0.0, 1.0))
        
        results["horizons"][h] = {
            "tss": tss_val,
            "threshold": thr_tss,
            "pr_auc": prauc,
            "brier": bs,
            "ece": ece,
            "n_pos": n_pos,
            "n_total": len(y_true)
        }
        
        print(f"\n{h}h Horizon:")
        print(f"  TSS={tss_val:.3f}@{thr_tss:.2f} | PR={prauc:.3f} | BS={bs:.3f} | ECE={ece:.3f} | Pos={n_pos}/{len(y_true)}")
    
    print(f"\n{'='*60}")
    print(f"[Val] Max TSS: {results['max_tss']:.4f}")
    print(f"{'='*60}")
    
    # Save predictions for confusion matrix analysis
    results['predictions'] = {
        'probs': probs,  # [N, 3] array
        'labels': labels  # [N, 3] array
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate checkpoint using exact training logic")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default=None, help="Override data parquet path (e.g., data/interim/windows_test_15.parquet)")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--tta-noise", type=float, default=0.0, help="Standard deviation of Gaussian noise for TTA")
    parser.add_argument("--tta-samples", type=int, default=1, help="Number of TTA samples per input")
    args = parser.parse_args()
    
    # ✅ FIX: Set seed for deterministic validation
    import random
    SEED = 24
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    use_ema = args.use_ema and not args.no_ema
    
    print(f"{'='*60}")
    print("EXACT TRAINING VALIDATION (with TTA support)")
    print(f"{'='*60}")
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Use EMA:    {use_ema}")
    print(f"TTA:        noise={args.tta_noise}, samples={args.tta_samples}")
    print(f"{'='*60}\n")
    
    cfg = PINNConfig.from_yaml(args.config)
    
    # Override data path if provided
    if args.data:
        cfg.data.windows_parquet = args.data
        print(f"📂 Overriding data path: {args.data}")
    
    device = get_device(cfg.device)
    if device.type == "mps":
        cfg.train.num_workers = 0
    
    checkpoint_path = Path(args.checkpoint)
    model, ema = load_model(cfg, checkpoint_path, device)
    val_loader = create_validation_loader(cfg)
    
    results = evaluate(
        model=model,
        loader=val_loader,
        horizons=cfg.classifier.horizons,
        device=device,
        ema=ema if use_ema else None,
        tta_noise=args.tta_noise,
        tta_samples=args.tta_samples
    )
    
    # Save results for confusion matrix analysis
    output_dir = checkpoint_path.parent / "validation_results"
    output_dir.mkdir(exist_ok=True)
    
    # Save predictions and thresholds
    output_file = output_dir / f"{checkpoint_path.stem}_validation.npz"
    np.savez(
        output_file,
        probs=results['predictions']['probs'],
        labels=results['predictions']['labels'],
        horizons=np.array(cfg.classifier.horizons),
        thresholds=np.array([results['horizons'][h]['threshold'] for h in cfg.classifier.horizons]),
        tss_values=np.array([results['horizons'][h]['tss'] for h in cfg.classifier.horizons])
    )
    
    print(f"\n💾 Saved validation results to: {output_file}")
    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
