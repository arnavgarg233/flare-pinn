#!/usr/bin/env python3
"""
W&B Sweep Agent for PINN Hyperparameter Search.

Run this after creating a sweep with:
    wandb sweep configs/wandb_sweep.yaml

Then run this script with the sweep ID:
    python tools/run_sweep_agent.py --sweep_id YOUR_SWEEP_ID --count 15
"""

import os
import sys
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import wandb


def train_with_sweep_config():
    """Training function called by W&B sweep agent."""
    # Initialize W&B run (sweep provides config overrides)
    # Use flare-pinn-tools project (where the sweep was created)
    run = wandb.init(project="flare-pinn-tools")
    config = wandb.config
    
    # Import training components
    from src.models.pinn.config import PINNConfig
    from src.train import PINNTrainer, setup_logging
    import random
    import numpy as np
    import torch
    
    # Load base config
    cfg = PINNConfig.from_yaml("src/configs/pinn_sota_sweep.yaml")
    
    # Set seeds for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(cfg.seed)
    
    # Apply W&B sweep overrides
    if hasattr(config, 'physics_grad_scale'):
        cfg.physics.physics_grad_scale = config.physics_grad_scale
    
    if hasattr(config, 'lambda_phys_max'):
        # Scale the ENTIRE lambda schedule proportionally to lambda_phys_max
        # Base schedule ends at 0.15 (equivalent to 42k in a 60k run with max=0.50)
        # lambda_phys_max represents the full 60k run target
        scale_factor = config.lambda_phys_max / 0.50  # Scale relative to full run target
        
        new_schedule = []
        for progress, lambda_val in cfg.physics.lambda_phys_schedule:
            # Scale lambda values (but not progress)
            new_schedule.append([progress, lambda_val * scale_factor])
        cfg.physics.lambda_phys_schedule = new_schedule
    
    if hasattr(config, 'causal_decay'):
        cfg.physics.causal_decay = config.causal_decay
    
    if hasattr(config, 'gradnorm_alpha'):
        if config.gradnorm_alpha == 0.0:
            cfg.physics.use_gradnorm = False  # Turn off GradNorm
        else:
            cfg.physics.use_gradnorm = True
            cfg.physics.gradnorm_alpha = config.gradnorm_alpha
    
    if hasattr(config, 'lr'):
        cfg.train.lr = config.lr
    
    # curl_weight fixed at 0.2 (not swept)

    # Unique checkpoint dir per run
    sweep_ckpt_dir = Path(f"outputs/checkpoints/sweep/{run.id}")
    sweep_ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.train.checkpoint_dir = str(sweep_ckpt_dir)  # Config expects string, trainer converts to Path
    
    # Setup logging
    logger = setup_logging(Path(f"outputs/logs/sweep_{run.id}.log"))
    logger.info(f"🎯 Sweep Run: {run.id}")
    logger.info(f"   physics_grad_scale: {cfg.physics.physics_grad_scale}")
    logger.info(f"   lambda_phys_max: {cfg.physics.lambda_phys_schedule[-1][1]}")
    logger.info(f"   causal_decay: {cfg.physics.causal_decay}")
    logger.info(f"   gradnorm_alpha: {cfg.physics.gradnorm_alpha}")
    logger.info(f"   curl_weight: {cfg.loss_weights.curl_consistency}")
    logger.info(f"   lr: {cfg.train.lr}")
    
    # Create trainer with W&B enabled
    trainer = PINNTrainer(cfg, logger, use_wandb=True)
    
    # Resume from 40k CNN baseline checkpoint (80/5/15 split)
    resume_path = Path("outputs/checkpoints/benchmark_classifier/checkpoint_step_0040000.pt")
    
    if resume_path.exists():
        # Load checkpoint
        checkpoint = trainer.checkpoint_mgr.load(resume_path)
        trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Load EMA weights - CRITICAL for good eval!
        if trainer.ema and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            trainer.ema.load_state_dict(checkpoint['ema_state_dict'])
            logger.info("   ✅ EMA weights loaded from checkpoint")
        
        trainer.step = checkpoint.get('step', 0)
        logger.info(f"✅ Loaded checkpoint from step {trainer.step}")
        
        # Fresh optimizer for sweep
        logger.info("   → Using fresh optimizer (sweep)")
    else:
        logger.error(f"❌ Checkpoint not found: {resume_path}")
        wandb.finish(exit_code=1)
        return
    
    # Create data loaders (same as main train.py)
    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from src.data.consolidated_dataset import ConsolidatedWindowsDataset
    
    windows_df = pd.read_parquet(cfg.data.windows_parquet)
    windows_df = windows_df.sort_values('t0').reset_index(drop=True)
    
    split_idx = int(len(windows_df) * (1 - cfg.data.val_fraction))
    train_df = windows_df.iloc[:split_idx].reset_index(drop=True)
    val_df = windows_df.iloc[split_idx:].reset_index(drop=True)
    
    logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}")
    
    consolidated_dir = str(Path(cfg.data.consolidated_dir).expanduser())
    train_ds = ConsolidatedWindowsDataset(
        windows_df=train_df,
        consolidated_dir=consolidated_dir,
        target_px=cfg.data.target_size,
        input_hours=cfg.data.input_hours,
        horizons=list(cfg.classifier.horizons),
        P_per_t=cfg.data.P_per_t,
        pil_top_pct=cfg.data.pil_top_pct,
        training=True, 
        augment=True,
        noise_std=0.05,
        max_cached_harps=500,
        use_pil_evolution=getattr(cfg.data, 'use_pil_evolution', True),
        use_temporal_statistics=getattr(cfg.data, 'use_temporal_statistics', True),
    )
    val_ds = ConsolidatedWindowsDataset(
        windows_df=val_df,
        consolidated_dir=consolidated_dir,
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
    
    # Create samplers
    horizon_cols = [f"y_geq_M_{h}h" for h in cfg.classifier.horizons]
    labels = train_df[horizon_cols].to_numpy().sum(axis=1) > 0
    weights = np.full(len(train_df), cfg.train.sampler.smoothing)
    weights[labels] = cfg.train.sampler.positive_multiplier
    sampler = WeightedRandomSampler(torch.tensor(weights), len(weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"DataLoaders ready: {len(train_loader)} train, {len(val_loader)} val batches")
    
    # Train
    try:
        trainer.train(train_loader, val_loader)
        
        # Get final 24h TSS
        final_tss_24h = trainer.best_val_tss  # This is now 24h TSS specifically
        wandb.log({"final/tss_24h": final_tss_24h})
        logger.info(f"🏁 Final 24h TSS: {final_tss_24h:.4f}")
        
        # Save checkpoint if 24h TSS is good (> 0.795 to catch near-misses too)
        if final_tss_24h > 0.795:
            import shutil
            best_ckpt_dir = Path.home() / "flare-pinn-personal-keep/checkpoints/sweep_best"
            best_ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy best model from this run
            run_best = Path(cfg.train.checkpoint_dir) / "best_model.pt"
            if run_best.exists():
                dest = best_ckpt_dir / f"best_model_tss{final_tss_24h:.4f}_{run.id}.pt"
                shutil.copy(run_best, dest)
                logger.info(f"🌟 SAVED GOOD CHECKPOINT: {dest}")
                logger.info(f"   24h TSS = {final_tss_24h:.4f} (> 0.80 threshold)")
                
                # Also log to W&B as artifact
                artifact = wandb.Artifact(
                    name=f"best-model-{final_tss_24h:.4f}",
                    type="model",
                    description=f"24h TSS = {final_tss_24h:.4f}"
                )
                artifact.add_file(str(dest))
                wandb.log_artifact(artifact)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        wandb.finish(exit_code=1)
        raise
    
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, help="W&B sweep ID")
    parser.add_argument("--count", type=int, default=15, help="Number of runs")
    parser.add_argument("--project", type=str, default="flare-pinn-tools")
    
    # Accept sweep hyperparameters (W&B passes these when running via agent)
    parser.add_argument("--physics_grad_scale", type=float, default=None)
    parser.add_argument("--lambda_phys_max", type=float, default=None)
    parser.add_argument("--causal_decay", type=float, default=None)
    parser.add_argument("--gradnorm_alpha", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    
    args = parser.parse_args()
    
    if args.sweep_id:
        # Run as agent
        wandb.agent(
            args.sweep_id,
            function=train_with_sweep_config,
            project=args.project,
            count=args.count
        )
    else:
        # Single run mode - called by W&B agent with hyperparameters
        os.environ["WANDB_PROJECT"] = args.project
        train_with_sweep_config()


if __name__ == "__main__":
    main()

