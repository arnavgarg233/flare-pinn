#!/usr/bin/env python3
"""
W&B Sweep Training Wrapper for PINN Flare Prediction.

Handles automatic hyperparameter optimization via Weights & Biases sweeps.
Based on best practices from jaxpi and PowerPINN repositories.

Usage:
    # Create sweep:
    wandb sweep wandb_sweep.yaml
    
    # Run agent:
    BASE_CONFIG=src/configs/sota_vector_128.yaml wandb agent <sweep-id>
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.models.pinn import PINNConfig
from src.train import PINNTrainer, setup_logging


def apply_sweep_overrides(cfg: PINNConfig) -> PINNConfig:
    """
    Apply W&B sweep parameters to base config.
    
    Handles both direct overrides (e.g., train.lr) and custom logic
    (e.g., physics schedule construction).
    """
    if wandb.run is None:
        return cfg
    
    sweep_config = wandb.config
    
    # Direct parameter overrides
    direct_overrides = {
        'train.lr': lambda v: setattr(cfg.train, 'lr', v),
        'train.batch_size': lambda v: setattr(cfg.train, 'batch_size', int(v)),
        'classifier.dropout': lambda v: setattr(cfg.classifier, 'dropout', v),
        'classifier.focal_gamma': lambda v: setattr(cfg.classifier, 'focal_gamma', v),
        'classifier.focal_alpha': lambda v: setattr(cfg.classifier, 'focal_alpha', v),
        'classifier.label_smoothing': lambda v: setattr(cfg.classifier, 'label_smoothing', v),
        'train.scheduler.warmup_steps': lambda v: setattr(cfg.train.scheduler, 'warmup_steps', int(v)),
        'train.sampler.positive_multiplier': lambda v: setattr(cfg.train.sampler, 'positive_multiplier', v),
        'collocation.alpha_end': lambda v: setattr(cfg.collocation, 'alpha_end', v),
        'model.fourier.ramp_frac': lambda v: setattr(cfg.model.fourier, 'ramp_frac', v),
        'loss_weights.curl_consistency': lambda v: setattr(cfg.loss_weights, 'curl_consistency', v),
    }
    
    for key, setter in direct_overrides.items():
        if key in sweep_config:
            setter(sweep_config[key])
    
    # Custom: Physics lambda schedule
    if 'physics_lambda_start_frac' in sweep_config and 'physics_lambda_final' in sweep_config:
        start_frac = float(sweep_config['physics_lambda_start_frac'])
        final_weight = float(sweep_config['physics_lambda_final'])
        cfg.physics.lambda_phys_schedule = [
            [0.0, 0.0],
            [start_frac, 0.0],
            [0.8, final_weight * 0.5],
            [1.0, final_weight]
        ]
    
    return cfg


def main():
    # Initialize W&B run
    wandb.init()
    
    # Load base configuration
    base_config_path = os.getenv("BASE_CONFIG", "src/configs/sota_vector_128.yaml")
    cfg = PINNConfig.from_yaml(base_config_path)
    
    # Apply sweep parameter overrides
    cfg = apply_sweep_overrides(cfg)
    
    # Update checkpoint directory for this sweep run
    if wandb.run is not None:
        sweep_id = wandb.run.sweep_id or "manual"
        run_id = wandb.run.id
        cfg.train.checkpoint_dir = Path(f"outputs/checkpoints/sweep_{sweep_id}/{run_id}")
    
    # Setup logging
    log_path = cfg.train.checkpoint_dir / "train.log" if cfg.train.checkpoint_dir else None
    logger = setup_logging(log_path)
    
    logger.info("="*60)
    logger.info("W&B Sweep Training Run")
    logger.info("="*60)
    logger.info(f"Sweep ID: {wandb.run.sweep_id}")
    logger.info(f"Run ID: {wandb.run.id}")
    logger.info(f"Base config: {base_config_path}")
    logger.info("Sweep parameters:")
    for key, value in wandb.config.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60)
    
    # Create trainer
    trainer = PINNTrainer(cfg, logger, use_wandb=True)
    
    # Load data
    if cfg.data.use_real:
        from src.utils.masked_training import load_windows_with_mask
        from src.data.consolidated_dataset import ConsolidatedWindowsDataset
        
        logger.info("Loading real data...")
        df, mask = load_windows_with_mask(str(cfg.data.windows_parquet))
        df = df[mask].reset_index(drop=True)
        logger.info(f"  Loaded {len(df)} windows")
        
        # Train/val split by HARP (prevent temporal leakage)
        unique_harps = df['harpnum'].unique()
        np.random.seed(cfg.seed)
        np.random.shuffle(unique_harps)
        n_val_harps = max(1, int(len(unique_harps) * cfg.data.val_fraction))
        val_harps = set(unique_harps[:n_val_harps])
        
        train_df = df[~df['harpnum'].isin(val_harps)].reset_index(drop=True)
        val_df = df[df['harpnum'].isin(val_harps)].reset_index(drop=True)
        
        logger.info(f"  Train: {len(train_df)} windows ({len(unique_harps) - n_val_harps} HARPs)")
        logger.info(f"  Val: {len(val_df)} windows ({n_val_harps} HARPs)")
        
        # Create datasets
        train_ds = ConsolidatedWindowsDataset(
            windows_df=train_df,
            consolidated_dir=str(cfg.data.consolidated_dir),
            target_px=cfg.data.target_size,
            input_hours=cfg.data.input_hours,
            horizons=list(cfg.classifier.horizons),
            P_per_t=cfg.data.P_per_t,
            pil_top_pct=cfg.data.pil_top_pct,
            training=True,
            augment=True,
            noise_std=0.02,
            max_cached_harps=500
        )
        
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
            max_cached_harps=500
        )
        
        # Class-balanced sampler
        horizon_cols = [f"y_geq_M_{h}h" for h in cfg.classifier.horizons]
        labels = train_df[horizon_cols].to_numpy().sum(axis=1) > 0
        weights = np.full(len(train_df), cfg.train.sampler.smoothing)
        weights[labels] = cfg.train.sampler.positive_multiplier
        sampler = WeightedRandomSampler(torch.tensor(weights), len(weights))
        
        # DataLoaders
        use_pin_memory = False if cfg.device == "mps" else True
        persistent_workers = cfg.train.num_workers > 0
        
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if cfg.train.num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if cfg.train.num_workers > 0 else None
        )
    else:
        raise ValueError("Dummy data not supported in sweep mode. Set data.use_real=true")
    
    # Train model
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader)
    
    # Log final summary
    wandb.summary["best_val_tss"] = trainer.best_val_tss
    logger.info(f"Training complete. Best validation TSS: {trainer.best_val_tss:.4f}")
    
    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()

