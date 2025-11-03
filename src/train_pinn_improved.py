#!/usr/bin/env python3
# src/train_pinn_improved.py
"""
Improved PINN training entrypoint with full integration.

Features:
  - Pydantic config validation
  - Unified PINN model (backbone + classifier + physics)
  - Curriculum learning (P0 → P4s stages)
  - Comprehensive metrics tracking
  - Checkpoint management
  - AMP support

Usage:
    python src/train_pinn_improved.py --config configs/train_pinn_5k.yaml
"""
from __future__ import annotations
import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.models.pinn import PINNConfig, PINNModel, PINNOutput
from src.eval.metrics import (
    pr_auc,
    brier_score,
    adaptive_ece,
    sweep_tss,
    select_threshold_at_far,
)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_path: Optional[Path] = None) -> logging.Logger:
    """Configure structured logging to file and console."""
    logger = logging.getLogger("pinn_train")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# Checkpoint Management
# ============================================================================

@dataclass
class CheckpointManager:
    """Handles model checkpointing with best model tracking."""
    checkpoint_dir: Path
    keep_last_n: int = 5
    
    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric: float = -1e9
        self.best_path: Optional[Path] = None
    
    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        step: int,
        metric_value: float,
        is_best: bool = False
    ) -> Path:
        """Save checkpoint and optionally update best model."""
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric_value,
        }
        
        # Regular checkpoint
        path = self.checkpoint_dir / f"checkpoint_step_{step:07d}.pt"
        torch.save(checkpoint, path)
        
        # Best model
        if is_best or metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, self.best_path)
        
        # Cleanup old checkpoints (keep last N)
        self._cleanup_old_checkpoints()
        
        return path
    
    def _cleanup_old_checkpoints(self):
        """Keep only the N most recent checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in checkpoints[:-self.keep_last_n]:
            old_ckpt.unlink()
    
    def load_best(self, model: nn.Module, optimizer: Optional[optim.Optimizer] = None):
        """Load best checkpoint."""
        if self.best_path is None or not self.best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found at {self.best_path}")
        
        checkpoint = torch.load(self.best_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['step'], checkpoint['metric']


# ============================================================================
# Dummy Dataset (for smoke tests)
# ============================================================================

class DummyPINNDataset(Dataset):
    """
    Synthetic dataset for quick testing.
    Returns collocation points, ground truth, and labels.
    """
    def __init__(self, cfg: PINNConfig, num_samples: int = 256):
        self.cfg = cfg
        self.num_samples = num_samples
        self.T = cfg.data.dummy_T
        self.P = cfg.data.P_per_t
        self.n_horizons = len(cfg.classifier.horizons)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        # Generate random collocation coords [T, P, 3]
        coords = torch.rand(self.T, self.P, 3) * 2.0 - 1.0  # [-1,1]^3
        
        # Dummy ground truth Bz (synthetic smooth field)
        x, y, t = coords[..., 0], coords[..., 1], coords[..., 2]
        gt_bz = torch.sin(np.pi * x) * torch.cos(np.pi * y) * torch.exp(-0.5 * t)
        gt_bz = gt_bz.unsqueeze(-1)  # [T, P, 1]
        
        # Observed mask (simulate some missing frames)
        observed_mask = torch.ones(self.T, dtype=torch.bool)
        if self.T > 4:
            # Randomly drop 10% of frames
            drop_prob = 0.1
            observed_mask = torch.rand(self.T) > drop_prob
        
        # Labels (20% positive rate)
        labels = (torch.rand(self.n_horizons) > 0.8).float()
        
        return {
            "coords": coords,
            "gt_bz": gt_bz,
            "observed_mask": observed_mask,
            "labels": labels,
            "pil_mask": None,
        }


# ============================================================================
# Metrics Collection
# ============================================================================

@dataclass
class MetricsBuffer:
    """Rolling buffer for quick evaluation."""
    max_size: int = 500
    probs: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    
    def add(self, prob: np.ndarray, label: np.ndarray):
        """Add a prediction batch."""
        self.probs.append(prob)
        self.labels.append(label)
        if len(self.probs) > self.max_size:
            self.probs.pop(0)
            self.labels.pop(0)
    
    def compute(self, horizons: list[int], logger: logging.Logger):
        """Compute and log metrics."""
        if not self.probs:
            logger.info("Metrics: No data yet")
            return
        
        probs = np.concatenate(self.probs, axis=0)  # [N, H]
        labels = np.concatenate(self.labels, axis=0)  # [N, H]
        
        for j, h in enumerate(horizons):
            y_true = labels[:, j]
            y_prob = probs[:, j]
            
            # Skip if no positive examples
            if y_true.sum() == 0:
                logger.info(f"  h={h}h: No positive samples")
                continue
            
            thr_tss, tss_val = sweep_tss(y_true, y_prob, n=256)
            thr_far = select_threshold_at_far(y_true, y_prob, max_far=0.05, n=512)
            prauc = pr_auc(y_true, y_prob)
            bs = brier_score(y_true, y_prob)
            ece = adaptive_ece(y_true, y_prob, n_bins=10)
            
            logger.info(
                f"  h={h}h: TSS*={tss_val:.3f}@{thr_tss:.2f} | "
                f"PR-AUC={prauc:.3f} | Brier={bs:.3f} | "
                f"ECE={ece:.3f} | thr@FAR5%={thr_far:.2f}"
            )


# ============================================================================
# Training Loop
# ============================================================================

class PINNTrainer:
    """Encapsulates PINN training logic."""
    
    def __init__(self, cfg: PINNConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        
        # Device setup
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Model
        self.model = PINNModel(cfg).to(self.device)
        self.logger.info(f"Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.train.lr)
        
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=cfg.train.amp and self.device.type == "cuda"
        )
        
        # Checkpointing
        self.checkpoint_mgr: Optional[CheckpointManager] = None
        if cfg.train.checkpoint_dir is not None:
            self.checkpoint_mgr = CheckpointManager(cfg.train.checkpoint_dir)
        
        # Metrics
        self.metrics_buffer = MetricsBuffer()
        
        # State
        self.step = 0
        self.epoch = 0
    
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        # Move to device
        coords = batch["coords"].to(self.device, non_blocking=True)
        gt_bz = batch["gt_bz"].to(self.device, non_blocking=True)
        observed_mask = batch["observed_mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        pil_mask = batch.get("pil_mask")
        
        # Update curriculum progress
        frac = min(1.0, self.step / max(1, self.cfg.train.steps))
        self.model.set_train_frac(frac)
        
        # Forward pass
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.cfg.train.amp and self.device.type == "cuda"
        ):
            # Squeeze batch dim (we typically use batch_size=1 for PINN)
            coords = coords.squeeze(0)  # [T, P, 3]
            gt_bz = gt_bz.squeeze(0)  # [T, P, 1]
            observed_mask = observed_mask.squeeze(0)  # [T]
            
            out: PINNOutput = self.model(
                coords=coords,
                gt_bz=gt_bz,
                observed_mask=observed_mask,
                labels=labels,
                pil_mask=pil_mask,
                mode="train"
            )
            
            loss = out.loss_total
        
        # Backward
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        if self.cfg.train.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.train.grad_clip
            )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Collect metrics
        with torch.no_grad():
            self.metrics_buffer.add(
                out.probs.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
        
        return {
            "loss_total": float(loss.item()),
            "loss_cls": float(out.loss_cls.item()),
            "loss_data": float(out.loss_data.item()),
            "loss_phys": float(out.loss_phys.item()),
            "lambda_phys": out.lambda_phys,
            "fourier_alpha": out.fourier_alpha,
            "ess": out.ess,
        }
    
    def train(self, train_loader: DataLoader):
        """Main training loop."""
        self.logger.info("=" * 80)
        self.logger.info("Starting training")
        self.logger.info(f"Config summary:")
        for k, v in self.cfg.model_dump_summary().items():
            self.logger.info(f"  {k}: {v}")
        self.logger.info("=" * 80)
        
        self.model.train()
        t_start = time.time()
        
        while self.step < self.cfg.train.steps:
            for batch in train_loader:
                self.step += 1
                
                # Train step
                metrics = self.train_step(batch)
                
                # Logging
                if self.step % self.cfg.train.log_every == 0:
                    elapsed = time.time() - t_start
                    lr = self.optimizer.param_groups[0]["lr"]
                    
                    self.logger.info(
                        f"[Step {self.step:05d}/{self.cfg.train.steps}] "
                        f"loss={metrics['loss_total']:.4f} "
                        f"(cls={metrics['loss_cls']:.4f}, "
                        f"data={metrics['loss_data']:.4f}, "
                        f"phys={metrics['loss_phys']:.4f}) | "
                        f"lam_phys={metrics['lambda_phys']:.2f} | "
                        f"a_fourier={metrics['fourier_alpha']:.2f} | "
                        f"ESS={metrics['ess']:.0f} | "
                        f"lr={lr:.2e} | "
                        f"time={elapsed:.1f}s"
                    )
                    t_start = time.time()
                
                # Evaluation
                if self.cfg.train.eval_every > 0 and \
                   (self.step % self.cfg.train.eval_every == 0 or \
                    self.step == self.cfg.train.steps):
                    self.logger.info(f"--- Eval @ step {self.step} ---")
                    self.metrics_buffer.compute(
                        list(self.cfg.classifier.horizons),
                        self.logger
                    )
                
                # Checkpointing
                if self.checkpoint_mgr is not None and \
                   self.step % self.cfg.train.checkpoint_every == 0:
                    # Use average TSS as checkpoint metric (placeholder)
                    ckpt_metric = -metrics['loss_total']
                    path = self.checkpoint_mgr.save(
                        self.model,
                        self.optimizer,
                        self.step,
                        ckpt_metric,
                        is_best=False
                    )
                    self.logger.info(f"Saved checkpoint: {path}")
                
                if self.step >= self.cfg.train.steps:
                    break
        
        self.logger.info("=" * 80)
        self.logger.info(f"Training complete after {self.step} steps")
        self.logger.info("=" * 80)


# ============================================================================
# Main Entrypoint
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train PINN for solar flare prediction"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to log file (optional)"
    )
    args = parser.parse_args()
    
    # Load config
    cfg = PINNConfig.from_yaml(args.config)
    
    # Setup logging
    log_path = Path(args.log) if args.log else None
    logger = setup_logging(log_path)
    
    logger.info(f"Loaded config from: {args.config}")
    
    # Create trainer
    trainer = PINNTrainer(cfg, logger)
    trainer.set_seed(cfg.seed)
    
    # Setup data
    if cfg.data.use_real:
        # Real SHARP data
        from src.data.windows_dataset import WindowsDataset
        from src.utils.masked_training import load_windows_with_mask
        
        logger.info("Loading real SHARP data...")
        windows_df, labeled_mask = load_windows_with_mask(str(cfg.data.windows_parquet))
        windows_df = windows_df[labeled_mask].reset_index(drop=True)
        
        dataset = WindowsDataset(
            windows_df=windows_df,
            frames_meta_path=str(cfg.data.frames_meta_parquet),
            npz_root=str(cfg.data.npz_root),
            target_px=cfg.data.target_size,
            input_hours=cfg.data.input_hours,
            horizons=list(cfg.classifier.horizons),
            P_per_t=cfg.data.P_per_t,
            pil_top_pct=cfg.data.pil_top_pct,
        )
    else:
        # Dummy data
        logger.info("Using dummy synthetic data...")
        dataset = DummyPINNDataset(cfg, num_samples=cfg.data.dummy_num_samples)
    
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging; increase for production
        pin_memory=True
    )
    
    logger.info(f"Dataset: {len(dataset)} samples")
    
    # Train
    trainer.train(train_loader)


if __name__ == "__main__":
    main()


