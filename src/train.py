#!/usr/bin/env python3
# src/train.py
"""
SOTA-Optimized PINN Trainer for Solar Flare Prediction.

Features:
  - Hybrid CNN-PINN training
  - Validation-based checkpointing (saves best TSS model)
  - Physics-informed loss scheduling
  - Comprehensive metrics logging (TSS, PR-AUC, Brier)
  - MPS/CUDA support
"""
from __future__ import annotations
import argparse
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Suppress warnings
warnings.filterwarnings('ignore', message='.*MPS autocast.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.amp')

from src.models.pinn import PINNConfig, PINNModel, HybridPINNModel
from src.models.eval.metrics import (
    pr_auc,
    brier_score,
    adaptive_ece,
    sweep_tss,
    select_threshold_at_far,
)
from src.utils.training_utils import optimize_for_device

# ============================================================================
# Dummy Dataset for Testing
# ============================================================================

class DummyPINNDataset(Dataset):
    """
    Synthetic dataset for testing PINN training without real data.
    Generates random magnetogram-like data with synthetic labels.
    """
    def __init__(self, cfg: PINNConfig):
        self.cfg = cfg
        self.T = cfg.data.dummy_T
        self.H = cfg.data.dummy_H
        self.W = cfg.data.dummy_W
        self.P = cfg.data.P_per_t
        self.num_samples = cfg.data.dummy_num_samples
        self.horizons = list(cfg.classifier.horizons)
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        # Generate random coords
        coords = torch.zeros(self.T, self.P, 3)
        coords[..., :2] = torch.rand(self.T, self.P, 2) * 2.0 - 1.0  # xy in [-1,1]
        t_vals = torch.linspace(-1.0, 1.0, self.T)[:, None, None].expand(self.T, self.P, 1)
        coords[..., 2:3] = t_vals
        coords.requires_grad_(True)
        
        # Generate synthetic Bz field (dipole-like pattern)
        x_grid = torch.linspace(-1, 1, self.W)
        y_grid = torch.linspace(-1, 1, self.H)
        xx, yy = torch.meshgrid(x_grid, y_grid, indexing='xy')
        
        # Create bipolar region (simplified magnetogram)
        bz_base = torch.tanh(5 * (xx - 0.3)) - torch.tanh(5 * (xx + 0.3))
        bz_base = bz_base * torch.exp(-2 * yy**2)
        
        # Add noise and time variation
        frames = torch.zeros(self.T, self.H, self.W)
        for t in range(self.T):
            noise = torch.randn(self.H, self.W) * 0.1
            evolution = 1.0 + 0.1 * t / self.T
            frames[t] = bz_base * evolution + noise
        
        # Sample gt_bz at coords
        gt_bz = torch.zeros(self.T, self.P, 1)
        for t in range(self.T):
            xy = coords[t, :, :2].detach()
            # Simple nearest-neighbor sampling
            x_idx = ((xy[:, 0] + 1) / 2 * (self.W - 1)).long().clamp(0, self.W - 1)
            y_idx = ((xy[:, 1] + 1) / 2 * (self.H - 1)).long().clamp(0, self.H - 1)
            gt_bz[t, :, 0] = frames[t, y_idx, x_idx]
        
        # Observed mask (all observed for dummy data)
        observed_mask = torch.ones(self.T, dtype=torch.bool)
        
        # Random labels (with ~20% positive rate for imbalance)
        labels = (torch.rand(len(self.horizons)) < 0.2).float()
        
        # PIL mask (high gradient regions)
        grad_x = torch.abs(frames[-1, :, 1:] - frames[-1, :, :-1])  # [H, W-1]
        grad_y = torch.abs(frames[-1, 1:, :] - frames[-1, :-1, :])  # [H-1, W]
        # Pad to match size [H, W] using simple concatenation
        grad_x = torch.cat([grad_x, grad_x[:, -1:]], dim=1)  # [H, W]
        grad_y = torch.cat([grad_y, grad_y[-1:, :]], dim=0)  # [H, W]
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        pil_mask = (grad_mag > grad_mag.quantile(0.85)).numpy().astype(np.uint8)
        
        return {
            "coords": coords,
            "gt_bz": gt_bz,
            "frames": frames,
            "observed_mask": observed_mask,
            "labels": labels,
            "pil_mask": pil_mask,
        }

# ============================================================================
# Logging
# ============================================================================

def setup_logging(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("pinn_train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# ============================================================================
# Checkpointing (Optimized for TSS)
# ============================================================================

@dataclass
class CheckpointManager:
    checkpoint_dir: Path
    keep_last_n: int = 3
    
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
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric_value,
        }
        
        path = self.checkpoint_dir / f"checkpoint_step_{step:07d}.pt"
        torch.save(checkpoint, path)
        
        if is_best or metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, self.best_path)
            print(f"🌟 NEW BEST MODEL! TSS={metric_value:.4f}")
        
        self._cleanup_old_checkpoints()
        return path
    
    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in checkpoints[:-self.keep_last_n]:
            old_ckpt.unlink()
            
    def load_best(self, model: nn.Module, optimizer: Optional[optim.Optimizer] = None):
        if self.best_path is None or not self.best_path.exists():
            return 0, 0.0
        checkpoint = torch.load(self.best_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step'], checkpoint['metric']

# ============================================================================
# Metrics
# ============================================================================

@dataclass
class MetricsBuffer:
    max_size: int = 500
    probs: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    
    def add(self, prob: np.ndarray, label: np.ndarray):
        self.probs.append(prob)
        self.labels.append(label)
        if len(self.probs) > self.max_size:
            self.probs.pop(0)
            self.labels.pop(0)
    
    def compute(self, horizons: list[int], logger: logging.Logger, label_prefix: str = ""):
        if not self.probs: return
        probs = np.concatenate(self.probs, axis=0)
        labels = np.concatenate(self.labels, axis=0)
        
        max_tss = 0.0
        
        for j, h in enumerate(horizons):
            y_true = labels[:, j]
            y_prob = probs[:, j]
            if y_true.sum() == 0: continue
            
            thr_tss, tss_val = sweep_tss(y_true, y_prob, n=256)
            max_tss = max(max_tss, tss_val)
            prauc = pr_auc(y_true, y_prob)
            bs = brier_score(y_true, y_prob)
            thr_far = select_threshold_at_far(y_true, y_prob, max_far=0.05)
            
            logger.info(
                f"{label_prefix}h={h}h: TSS={tss_val:.3f}@{thr_tss:.2f} | "
                f"PR={prauc:.3f} | BS={bs:.3f} | FAR5%@{thr_far:.2f}"
            )
        return max_tss

# ============================================================================
# Trainer
# ============================================================================

class PINNTrainer:
    def __init__(self, cfg: PINNConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.device = self._get_device(cfg.device)
        optimize_for_device(self.device)
        
        # Model
        if cfg.model.model_type == "hybrid":
            self.model = HybridPINNModel(cfg, encoder_in_channels=1).to(self.device)
        else:
            self.model = PINNModel(cfg).to(self.device)
            
        self.logger.info(f"Model Params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Optimizer & Scheduler with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=cfg.train.lr,
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.999)
        )
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.train.steps, eta_min=cfg.train.scheduler.min_lr
        )
        
        # EMA for better generalization
        self.ema = None
        if cfg.train.use_ema:
            from src.utils.training_utils import ExponentialMovingAverage
            self.ema = ExponentialMovingAverage(self.model, decay=cfg.train.ema_decay)
            self.logger.info(f"EMA enabled with decay={cfg.train.ema_decay}")
        
        # Early stopping
        from src.utils.training_utils import EarlyStopping
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='max')
        
        self.checkpoint_mgr = None
        if cfg.train.checkpoint_dir:
            self.checkpoint_mgr = CheckpointManager(cfg.train.checkpoint_dir)
            
        self.metrics_buffer = MetricsBuffer()
        self.step = 0
        self.best_val_tss = 0.0

    def _get_device(self, requested: str) -> torch.device:
        if requested == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
        if requested.startswith("cuda") and torch.cuda.is_available(): return torch.device(requested)
        return torch.device("cpu")

    def train_step(self, batch: dict) -> dict:
        # Data to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        
        # Update curriculum
        frac = min(1.0, self.step / self.cfg.train.steps)
        self.model.set_train_frac(frac)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward
        autocast_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        use_autocast = self.use_amp or (self.device.type == "mps" and self.cfg.train.amp)
        
        with torch.autocast(self.device.type, dtype=autocast_dtype, enabled=use_autocast):
            # Unpack batch manually to handle list inputs like pil_mask if needed
            B = batch["coords"].shape[0]
            total_loss = 0.0
            batch_probs, batch_labels = [], []
            
            # Process samples (PINN loop)
            for i in range(B):
                sample_kwargs = {
                    "coords": batch["coords"][i],
                    "gt_bz": batch["gt_bz"][i],
                    "observed_mask": batch["observed_mask"][i],
                    "labels": batch["labels"][i:i+1],
                    "pil_mask": batch["pil_mask"][i].cpu().numpy() if isinstance(batch["pil_mask"], torch.Tensor) else batch["pil_mask"][i],
                    "mode": "train"
                }
                if "frames" in batch:
                    sample_kwargs["frames"] = batch["frames"][i]
                if "scalars" in batch:
                    sample_kwargs["scalars"] = batch["scalars"][i]
                
                out = self.model(**sample_kwargs)
                total_loss += out.loss_total / B
                batch_probs.append(out.probs)
                batch_labels.append(batch["labels"][i:i+1])
            
            loss = total_loss
            probs = torch.cat(batch_probs)
            labels = torch.cat(batch_labels)

        # Backward
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.cfg.train.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            else:
                grad_norm = 0.0
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.cfg.train.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            else:
                grad_norm = 0.0
            self.optimizer.step()
        
        # Update EMA
        if self.ema is not None:
            self.ema.update()
            
        # Scheduler (warmup manual)
        if self.step < self.cfg.train.scheduler.warmup_steps:
            lr_scale = min(1.0, float(self.step + 1) / self.cfg.train.scheduler.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.cfg.train.lr * lr_scale
        else:
            self.scheduler.step()
            
        with torch.no_grad():
            self.metrics_buffer.add(probs.detach().cpu().numpy(), labels.detach().cpu().numpy())
            
        return {
            "loss": loss.item(),
            "phys": out.loss_phys.item(),
            "cls": out.loss_cls.item(),
            "lam": out.lambda_phys,
            "alpha": out.fourier_alpha,
            "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
        }

    def evaluate(self, loader: DataLoader, use_ema: bool = True) -> dict:
        """
        Run full validation and return comprehensive metrics.
        
        Args:
            loader: Validation DataLoader
            use_ema: Whether to use EMA weights for evaluation
            
        Returns:
            Dictionary with TSS, PR-AUC, Brier scores per horizon
        """
        if loader is None: 
            return {"max_tss": 0.0}
        
        self.model.eval()
        all_probs, all_labels = [], []
        
        # Use EMA weights if available (with proper context manager)
        ema_context = self.ema.average_parameters() if (use_ema and self.ema is not None) else None
        
        with torch.no_grad():
            # Enter EMA context if available
            if ema_context is not None:
                ema_context.__enter__()
            
            try:
                for batch in loader:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device, non_blocking=True)
                    
                    B = batch["coords"].shape[0]
                    batch_probs = []
                    for i in range(B):
                        sample_kwargs = {
                            "coords": batch["coords"][i],
                            "gt_bz": batch["gt_bz"][i],
                            "observed_mask": batch["observed_mask"][i],
                            "labels": batch["labels"][i:i+1],
                            "pil_mask": batch["pil_mask"][i].cpu().numpy() if isinstance(batch["pil_mask"], torch.Tensor) else batch["pil_mask"][i],
                            "mode": "eval"
                        }
                        if "frames" in batch:
                            sample_kwargs["frames"] = batch["frames"][i]
                        if "scalars" in batch:
                            sample_kwargs["scalars"] = batch["scalars"][i]
                        
                        out = self.model(**sample_kwargs)
                        batch_probs.append(out.probs)
                    
                    all_probs.append(torch.cat(batch_probs).cpu().numpy())
                    all_labels.append(batch["labels"].cpu().numpy())
            finally:
                # Exit EMA context
                if ema_context is not None:
                    ema_context.__exit__(None, None, None)
        
        self.model.train()
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        
        # Compute comprehensive metrics
        results = {"max_tss": 0.0, "horizons": {}}
        
        for j, h in enumerate(self.cfg.classifier.horizons):
            y_true = labels[:, j]
            y_prob = probs[:, j]
            
            if y_true.sum() == 0: 
                continue
                
            # TSS
            thr_tss, tss_val = sweep_tss(y_true, y_prob)
            results["max_tss"] = max(results["max_tss"], tss_val)
            
            # PR-AUC
            prauc = pr_auc(y_true, y_prob)
            
            # Brier score
            bs = brier_score(y_true, y_prob)
            
            # ECE
            ece = adaptive_ece(y_true, y_prob)
            
            results["horizons"][h] = {
                "tss": tss_val,
                "threshold": thr_tss,
                "pr_auc": prauc,
                "brier": bs,
                "ece": ece,
                "n_pos": int(y_true.sum()),
                "n_total": len(y_true)
            }
            
        return results

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        self.model.train()
        t_start = time.time()
        stopped_early = False
        
        while self.step < self.cfg.train.steps and not stopped_early:
            for batch in train_loader:
                self.step += 1
                metrics = self.train_step(batch)
                
                if self.step % self.cfg.train.log_every == 0:
                    elapsed = time.time() - t_start
                    lr = self.optimizer.param_groups[0]['lr']
                    grad_info = f"GradNorm={metrics.get('grad_norm', 0):.2f} | " if 'grad_norm' in metrics else ""
                    self.logger.info(
                        f"Step {self.step:05d} | Loss={metrics['loss']:.4f} "
                        f"(Phys={metrics['phys']:.4f}, Cls={metrics['cls']:.4f}) | "
                        f"{grad_info}"
                        f"Lam={metrics['lam']:.2f} | LR={lr:.2e} | T={elapsed:.1f}s"
                    )
                    t_start = time.time()
                
                if self.step % self.cfg.train.eval_every == 0:
                    self.metrics_buffer.compute(self.cfg.classifier.horizons, self.logger, "[Train] ")
                    if val_loader:
                        val_results = self.evaluate(val_loader)
                        val_tss = val_results["max_tss"]
                        
                        # Log detailed metrics
                        self.logger.info(f"[Val] Max TSS: {val_tss:.4f}")
                        for h, h_metrics in val_results.get("horizons", {}).items():
                            self.logger.info(
                                f"  {h}h: TSS={h_metrics['tss']:.3f}@{h_metrics['threshold']:.2f} | "
                                f"PR={h_metrics['pr_auc']:.3f} | BS={h_metrics['brier']:.3f} | "
                                f"ECE={h_metrics['ece']:.3f} | Pos={h_metrics['n_pos']}/{h_metrics['n_total']}"
                            )
                        
                        # Track best
                        if val_tss > self.best_val_tss:
                            self.best_val_tss = val_tss
                            self.logger.info(f"🌟 New best validation TSS: {val_tss:.4f}")
                        
                        # Save checkpoint
                        if self.checkpoint_mgr:
                            self.checkpoint_mgr.save(self.model, self.optimizer, self.step, val_tss)
                        
                        # Early stopping check
                        if self.early_stopping(val_tss, self.step):
                            self.logger.info(f"🛑 Early stopping triggered at step {self.step}")
                            self.logger.info(f"   Best TSS was {self.early_stopping.best_metric:.4f} at step {self.early_stopping.best_step}")
                            stopped_early = True
                            break
                
                if self.step >= self.cfg.train.steps: 
                    break
        
        # Final summary
        self.logger.info("="*60)
        self.logger.info("Training Complete!")
        self.logger.info(f"Final best validation TSS: {self.best_val_tss:.4f}")
        self.logger.info("="*60)

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    cfg = PINNConfig.from_yaml(args.config)
    logger = setup_logging(Path(f"outputs/logs/{Path(args.config).stem}.log"))
    trainer = PINNTrainer(cfg, logger)
    
    # Load Data
    if cfg.data.use_real:
        from src.data.windows_dataset import WindowsDataset
        from src.utils.masked_training import load_windows_with_mask
        
        logger.info("Loading real data...")
        df, mask = load_windows_with_mask(str(cfg.data.windows_parquet))
        df = df[mask].reset_index(drop=True)
        
        # Split
        val_idx = int(len(df) * (1 - cfg.data.val_fraction))
        train_df = df.iloc[:val_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx:].reset_index(drop=True)
        
        # Datasets
        train_ds = WindowsDataset(train_df, str(cfg.data.frames_meta_parquet), str(cfg.data.npz_root), 
                                  target_px=cfg.data.target_size, input_hours=cfg.data.input_hours,
                                  horizons=list(cfg.classifier.horizons), P_per_t=cfg.data.P_per_t, 
                                  pil_top_pct=cfg.data.pil_top_pct, scalar_features=cfg.data.scalar_features)
        
        val_ds = WindowsDataset(val_df, str(cfg.data.frames_meta_parquet), str(cfg.data.npz_root),
                                target_px=cfg.data.target_size, input_hours=cfg.data.input_hours,
                                horizons=list(cfg.classifier.horizons), P_per_t=cfg.data.P_per_t,
                                pil_top_pct=cfg.data.pil_top_pct, scalar_features=cfg.data.scalar_features)
        
        # Sampler
        horizon_cols = [f"y_geq_M_{h}h" for h in cfg.classifier.horizons]
        labels = train_df[horizon_cols].to_numpy().sum(axis=1) > 0
        weights = np.full(len(train_df), cfg.train.sampler.smoothing)
        weights[labels] = cfg.train.sampler.positive_multiplier
        sampler = WeightedRandomSampler(torch.tensor(weights), len(weights))
        
        train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    else:
        # Dummy fallback - use DummyPINNDataset defined in this module
        ds = DummyPINNDataset(cfg)
        train_loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True)
        val_loader = None

    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
