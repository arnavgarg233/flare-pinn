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
import gc
import logging
import time
import warnings
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Suppress warnings
warnings.filterwarnings('ignore', message='.*MPS autocast.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.amp')

# W&B import (optional dependency)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from src.models.pinn import PINNConfig, PINNModel, HybridPINNModel
from src.models.eval.metrics import (
    pr_auc,
    brier_score,
    adaptive_ece,
    sweep_tss,
    select_threshold_at_far,
)
from src.utils.training_utils import optimize_for_device, clear_memory_cache
from src.utils.memory_optimization import (
    MPSGradientAccumulator,
    aggressive_memory_cleanup,
    low_memory_mode,
)

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
        # Don't set requires_grad here - will be set in training loop
        
        # Generate synthetic Bz field (dipole-like pattern)
        x_grid = torch.linspace(-1, 1, self.W)
        y_grid = torch.linspace(-1, 1, self.H)
        xx, yy = torch.meshgrid(x_grid, y_grid, indexing='xy')
        
        # Create bipolar region (simplified magnetogram)
        bz_base = torch.tanh(5 * (xx - 0.3)) - torch.tanh(5 * (xx + 0.3))
        bz_base = bz_base * torch.exp(-2 * yy**2)
        
        # Add noise and time variation
        C = self.cfg.data.n_components
        frames = torch.zeros(self.T, C, self.H, self.W)
        for t in range(self.T):
            noise = torch.randn(C, self.H, self.W) * 0.1
            evolution = 1.0 + 0.1 * t / self.T
            # Replicate base for all channels or create variations
            for c in range(C):
                frames[t, c] = bz_base * evolution + noise[c]
        
        # Sample gt_bz at coords
        gt_bz = torch.zeros(self.T, self.P, C)
        for t in range(self.T):
            xy = coords[t, :, :2].detach()
            # Simple nearest-neighbor sampling
            x_idx = ((xy[:, 0] + 1) / 2 * (self.W - 1)).long().clamp(0, self.W - 1)
            y_idx = ((xy[:, 1] + 1) / 2 * (self.H - 1)).long().clamp(0, self.H - 1)
            for c in range(C):
                gt_bz[t, :, c] = frames[t, c, y_idx, x_idx]
        
        # Observed mask (all observed for dummy data)
        observed_mask = torch.ones(self.T, dtype=torch.bool)
        
        # Random labels (with ~20% positive rate for imbalance)
        labels = (torch.rand(len(self.horizons)) < 0.2).float()
        
        # PIL mask (high gradient regions) - as tensor for type consistency
        # Use Bz (channel 0) for PIL
        frame_bz = frames[-1, -1] if C > 0 else frames[-1] # Last channel usually Bz? Or first? Config says ["Bx", "By", "Bz"]
        # Actually check config order. Assuming index -1 is Bz.
        frame_bz = frames[-1, -1] 
        
        grad_x = torch.abs(frame_bz[:, 1:] - frame_bz[:, :-1])  # [H, W-1]
        grad_y = torch.abs(frame_bz[1:, :] - frame_bz[:-1, :])  # [H-1, W]
        # Pad to match size [H, W] using simple concatenation
        grad_x = torch.cat([grad_x, grad_x[:, -1:]], dim=1)  # [H, W]
        grad_y = torch.cat([grad_y, grad_y[-1:, :]], dim=0)  # [H, W]
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        pil_mask = (grad_mag > grad_mag.quantile(0.85)).float()  # Keep as tensor
        
        # Scalar features (dummy values for testing)
        # [r_value, gwpil, obs_coverage, frame_count]
        base_scalars = torch.tensor([
            2.5,   # Dummy R-value (typical range: 1-5)
            100.0, # Dummy GWPIL
            1.0,   # All frames observed
            float(self.T),  # Frame count
        ], dtype=torch.float32)
        
        # Pad to match n_scalar_features
        n_features = self.cfg.data.n_scalar_features
        if n_features > 4:
            padding = torch.zeros(n_features - 4, dtype=torch.float32)
            scalars = torch.cat([base_scalars, padding])
        else:
            scalars = base_scalars
        
        return {
            "coords": coords,
            "gt_bz": gt_bz,
            "frames": frames,
            "observed_mask": observed_mask,
            "labels": labels,
            "pil_mask": pil_mask,
            "scalars": scalars,
        }

# ============================================================================
# Logging
# ============================================================================

def setup_logging(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("pinn_train")
    logger.setLevel(logging.INFO)
    
    # FIXED: Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
    
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
class CheckpointState:
    """Container for all training state that should be checkpointed."""
    step: int
    metric: float
    model_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: Optional[dict] = None
    ema_state_dict: Optional[dict] = None
    early_stopping_state: Optional[dict] = None
    best_val_tss: float = 0.0


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
        is_best: bool = False,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        ema: Optional["ExponentialMovingAverage"] = None,
        early_stopping: Optional["EarlyStopping"] = None,
        best_val_tss: float = 0.0,
    ) -> Path:
        """
        Save checkpoint with full training state.
        
        Now includes EMA, scheduler, and early stopping state for proper resumption.
        """
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric_value,
            'best_val_tss': best_val_tss,
        }
        
        # Save scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save EMA state (CRITICAL for proper resumption!)
        if ema is not None:
            checkpoint['ema_state_dict'] = ema.state_dict()
        
        # Save early stopping state (including config for consistency check on resume)
        if early_stopping is not None:
            checkpoint['early_stopping_state'] = {
                'best_metric': early_stopping.best_metric,
                'best_step': early_stopping.best_step,
                'counter': early_stopping.counter,
                'patience': early_stopping.patience,
                'min_delta': early_stopping.min_delta,
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
    
    def load(self, path: Path) -> dict:
        """Load checkpoint from path."""
        if not path.exists():
            return {}
        return torch.load(path, map_location='cpu', weights_only=False)
            
    def load_best(
        self, 
        model: nn.Module, 
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        ema: Optional["ExponentialMovingAverage"] = None,
        early_stopping: Optional["EarlyStopping"] = None,
    ) -> tuple[int, float, float]:
        """
        Load best checkpoint with full state restoration.
        
        Returns:
            Tuple of (step, metric, best_val_tss)
        """
        if self.best_path is None or not self.best_path.exists():
            return 0, 0.0, 0.0
        
        checkpoint = self.load(self.best_path)
        
        # Core state - strict=False handles new buffers added after checkpoint was saved
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                # Optimizer state mismatch (model architecture changed)
                # Skip optimizer restore - will use fresh optimizer with loaded model weights
                import warnings
                warnings.warn(
                    f"⚠️  Optimizer state mismatch, starting fresh optimizer: {e}\n"
                    "   This is normal if model architecture changed since checkpoint."
                )
        
        # Scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except (ValueError, KeyError) as e:
                import warnings
                warnings.warn(f"⚠️  Scheduler state mismatch, using fresh scheduler: {e}")
        
        # EMA state (CRITICAL!)
        if ema is not None and 'ema_state_dict' in checkpoint:
            try:
                ema.load_state_dict(checkpoint['ema_state_dict'])
            except (ValueError, KeyError) as e:
                # EMA state mismatch (model architecture changed)
                import warnings
                warnings.warn(
                    f"⚠️  EMA state mismatch, starting fresh EMA: {e}\n"
                    "   This is normal if model architecture changed since checkpoint."
                )
        
        # Early stopping state
        if early_stopping is not None and 'early_stopping_state' in checkpoint:
            es_state = checkpoint['early_stopping_state']
            early_stopping.best_metric = es_state.get('best_metric', early_stopping.best_metric)
            early_stopping.best_step = es_state.get('best_step', 0)
            early_stopping.counter = es_state.get('counter', 0)
            
            # ✅ FIX #16: Warn if patience/min_delta changed since checkpoint
            saved_patience = es_state.get('patience')
            saved_min_delta = es_state.get('min_delta')
            if saved_patience is not None and saved_patience != early_stopping.patience:
                import warnings
                warnings.warn(
                    f"EarlyStopping patience changed: checkpoint={saved_patience}, config={early_stopping.patience}"
                )
            if saved_min_delta is not None and saved_min_delta != early_stopping.min_delta:
                import warnings
                warnings.warn(
                    f"EarlyStopping min_delta changed: checkpoint={saved_min_delta}, config={early_stopping.min_delta}"
                )
        
        best_val_tss = checkpoint.get('best_val_tss', checkpoint.get('metric', 0.0))
        
        return checkpoint['step'], checkpoint['metric'], best_val_tss

# ============================================================================
# Metrics
# ============================================================================

@dataclass
class MetricsBuffer:
    max_size: int = 100  # REDUCED from 500 to prevent memory leaks
    # ✅ FIX: Use deque for O(1) pop from left (was O(n) with list.pop(0))
    probs: deque = field(default_factory=lambda: deque(maxlen=100))
    labels: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add(self, prob: np.ndarray, label: np.ndarray):
        # deque automatically drops oldest when maxlen exceeded
        self.probs.append(prob)
        self.labels.append(label)
    
    def clear(self):
        """Clear all stored predictions to free memory."""
        self.probs.clear()
        self.labels.clear()
    
    def compute(self, horizons: list[int], logger: logging.Logger, label_prefix: str = ""):
        if not self.probs: return
        probs = np.concatenate(self.probs, axis=0)
        labels = np.concatenate(self.labels, axis=0)
        
        # SAFETY: Filter NaN/Inf and clamp probabilities
        valid_mask = np.isfinite(probs).all(axis=1) & np.isfinite(labels).all(axis=1)
        if valid_mask.sum() == 0:
            logger.warning(f"{label_prefix}No valid samples for metrics!")
            return 0.0
        probs = probs[valid_mask]
        labels = labels[valid_mask]
        probs = np.clip(probs, 0.0, 1.0)
        
        max_tss = 0.0
        
        for j, h in enumerate(horizons):
            y_true = labels[:, j]
            y_prob = probs[:, j]
            
            # Filter NaN AND Inf, clamp to valid ranges
            valid = np.isfinite(y_true) & np.isfinite(y_prob)
            y_true = np.clip(y_true[valid], 0.0, 1.0)
            y_prob = np.clip(y_prob[valid], 0.0, 1.0)
            
            if len(y_true) == 0 or y_true.sum() == 0: continue
            
            thr_tss, tss_val = sweep_tss(y_true, y_prob, n=256)
            max_tss = max(max_tss, tss_val)
            prauc = pr_auc(y_true, y_prob)
            bs = brier_score(y_true, y_prob)
            thr_far = select_threshold_at_far(y_true, y_prob, max_far=0.05)
            
            # Safety: clamp BS to [0, 1] range
            bs = float(np.clip(bs, 0.0, 1.0))
            
            logger.info(
                f"{label_prefix}h={h}h: TSS={tss_val:.3f}@{thr_tss:.2f} | "
                f"PR={prauc:.3f} | BS={bs:.3f} | FAR5%@{thr_far:.2f}"
            )
        return max_tss

# ============================================================================
# Trainer
# ============================================================================

class PINNTrainer:
    def __init__(self, cfg: PINNConfig, logger: logging.Logger, use_wandb: bool = False):
        self.cfg = cfg
        self.logger = logger
        self.device = self._get_device(cfg.device)
        
        # CRITICAL: Force num_workers=0 on MPS to prevent memory leaks
        if self.device.type == "mps" and cfg.train.num_workers > 0:
            self.logger.warning(f"⚠️  Forcing num_workers=0 on MPS (was {cfg.train.num_workers}) to prevent memory leaks")
            cfg.train.num_workers = 0
        
        optimize_for_device(self.device)
        
        # W&B initialization
        self.use_wandb = use_wandb and WANDB_AVAILABLE and not os.getenv("WANDB_DISABLED")
        if self.use_wandb and wandb.run is None:
            # Only init if not already initialized by sweep
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "flare-pinn-sota"),
                config=cfg.model_dump(),
                name=f"{cfg.model.model_type}_lr{cfg.train.lr}"
            )
        
        # Model
        if cfg.model.model_type == "hybrid":
            self.model = HybridPINNModel(cfg, encoder_in_channels=None).to(self.device)
        else:
            self.model = PINNModel(cfg).to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model Params: {n_params:,} ({n_trainable:,} trainable)")
        
        # Memory estimation for MPS
        if self.device.type == "mps":
            param_memory_mb = n_params * 4 / 1e6  # float32
            self.logger.info(f"Estimated model memory: {param_memory_mb:.1f} MB")
            self.logger.info("⚠️  MPS mode: Using gradient accumulation for larger effective batch")
        
        # Optimizer & Scheduler with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=cfg.train.lr,
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.999)
        )
        
        # AMP setup - MPS uses different approach
        self.use_amp = cfg.train.amp
        if self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        else:
            self.scaler = None  # MPS doesn't use GradScaler
        
        # Gradient accumulation for effective larger batches
        accum_steps = getattr(cfg.train, 'gradient_accumulation_steps', 1)
        if accum_steps > 1:
            self.grad_accumulator = MPSGradientAccumulator(
                self.model,
                self.optimizer,
                accum_steps=accum_steps,
                grad_clip=cfg.train.grad_clip,
                cleanup_every=getattr(cfg.train, 'memory_cleanup_every', 50)
            )
            self.logger.info(f"Gradient accumulation: {accum_steps} steps (effective batch = {cfg.train.batch_size * accum_steps})")
        else:
            self.grad_accumulator = None
        
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
        
        # Early stopping - FIXED: Balance patience for physics training
        # With eval_every=2000, patience=12 means 24k steps without improvement
        # This allows physics to stabilize while catching genuine plateaus
        from src.utils.training_utils import EarlyStopping
        self.early_stopping = EarlyStopping(patience=12, min_delta=0.003, mode='max')
        
        self.checkpoint_mgr = None
        if cfg.train.checkpoint_dir:
            self.checkpoint_mgr = CheckpointManager(cfg.train.checkpoint_dir)
            
        self.metrics_buffer = MetricsBuffer()
        self.step = 0
        self.best_val_tss = 0.0
        
        # MPS-specific settings
        self.memory_cleanup_every = getattr(cfg.train, 'memory_cleanup_every', 50)
        self._accumulated_loss = 0.0
        self._accumulated_phys = 0.0
        self._accumulated_cls = 0.0
        self._accumulated_count = 0

    def _get_device(self, requested: str) -> torch.device:
        if requested == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
        if requested.startswith("cuda") and torch.cuda.is_available(): return torch.device(requested)
        return torch.device("cpu")

    def train_step(self, batch: dict) -> dict:
        # Move all tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        
        # ⚡ SAFETY: Validate inputs and sanitize NaN/Inf (with logging)
        for k in ["frames", "coords", "scalar_features"]:
            if k in batch and batch[k] is not None:
                has_nan = torch.isnan(batch[k]).any()
                has_inf = torch.isinf(batch[k]).any()
                if has_nan or has_inf:
                    self.logger.warning(f"Step {self.step}: NaN/Inf in input '{k}', sanitizing")
                    batch[k] = torch.nan_to_num(batch[k], nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Update curriculum
        frac = min(1.0, self.step / self.cfg.train.steps)
        self.model.set_train_frac(frac)
        
        # Clear gradients if not using accumulator
        if self.grad_accumulator is None:
            self.optimizer.zero_grad(set_to_none=True)
        
        # Forward
        autocast_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        # MPS autocast with bfloat16 is more stable (if available)
        if self.device.type == "mps":
            autocast_dtype = torch.float32  # MPS float16 has issues, use float32
            use_autocast = False  # Disable autocast on MPS for stability
        else:
            use_autocast = self.use_amp
        
        with torch.autocast(self.device.type, dtype=autocast_dtype, enabled=use_autocast):
            B = batch["coords"].shape[0]
            
            # Use BATCHED forward for HybridPINNModel (MAJOR SPEEDUP)
            if hasattr(self.model, 'forward_batched') and "frames" in batch:
                out = self.model.forward_batched(
                    coords=batch["coords"],
                    frames=batch["frames"],
                    gt_bz=batch["gt_bz"],
                    observed_mask=batch["observed_mask"],
                    labels=batch["labels"],
                    pil_mask=batch["pil_mask"],
                    scalars=batch.get("scalars"),
                    mode="train",
                )
                
                loss = out.loss_total
                probs = out.probs
                labels = batch["labels"]
                total_phys = float(out.loss_phys.detach().cpu())
                total_cls = float(out.loss_cls.detach().cpu())
            else:
                # Fallback: per-sample processing (for non-hybrid models)
                total_loss = None
                total_phys = 0.0
                total_cls = 0.0
                batch_probs, batch_labels = [], []
                
                for i in range(B):
                    sample_kwargs = {
                        "coords": batch["coords"][i],
                        "gt_bz": batch["gt_bz"][i],
                        "observed_mask": batch["observed_mask"][i],
                        "labels": batch["labels"][i:i+1],
                        "pil_mask": batch["pil_mask"][i],
                        "mode": "train",
                    }
                    if "frames" in batch:
                        sample_kwargs["frames"] = batch["frames"][i]
                    if "scalars" in batch:
                        sample_kwargs["scalars"] = batch["scalars"][i]
                    
                    out = self.model(**sample_kwargs)
                    
                    if torch.isnan(out.loss_total) or torch.isinf(out.loss_total):
                        self.logger.warning(f"Step {self.step}: NaN/Inf in loss_total from sample {i}, skipping batch")
                        return {"loss": 0.0, "phys": 0.0, "cls": 0.0, "lam": 0.0, "grad_norm": 0.0}
                    
                    if out.probs is not None and (torch.isnan(out.probs).any() or torch.isinf(out.probs).any()):
                        out.probs.data.copy_(torch.nan_to_num(out.probs, nan=0.5, posinf=1.0, neginf=0.0))
                    
                    if total_loss is None:
                        total_loss = out.loss_total / B
                    else:
                        total_loss = total_loss + (out.loss_total / B)
                    
                    total_phys += float(out.loss_phys.detach().cpu()) / B
                    total_cls += float(out.loss_cls.detach().cpu()) / B
                    batch_probs.append(out.probs)
                    batch_labels.append(batch["labels"][i:i+1])
                
                loss = total_loss
                probs = torch.cat(batch_probs)
                labels = torch.cat(batch_labels)

        # NaN check - skip update if loss is NaN
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.warning(f"Step {self.step}: NaN/Inf loss detected, skipping update")
            if self.grad_accumulator:
                self.grad_accumulator.zero_grad()
            return {
                "loss": 0.0, "phys": 0.0, "cls": 0.0, "lam": 0.0
            }
        
        # Check that loss requires gradients (critical for backward pass)
        if not loss.requires_grad:
            self.logger.error(
                f"Step {self.step}: Loss does not require gradients! "
                f"loss.requires_grad={loss.requires_grad}, "
                f"loss.grad_fn={loss.grad_fn}, "
                f"loss.is_leaf={loss.is_leaf}. "
                f"This usually means the loss computation was disconnected from the model parameters."
            )
            # Try to reconnect by adding a zero term connected to model parameters
            # Get a parameter to ensure connection - use the first one that requires grad
            for param in self.model.parameters():
                if param.requires_grad:
                    # Convert loss to tensor if needed and add connected zero term
                    if isinstance(loss, (int, float)):
                        loss = torch.tensor(float(loss), device=param.device, dtype=param.dtype)
                    loss = loss + (param * 0.0).sum()
                    break
            
            if not loss.requires_grad:
                raise RuntimeError(
                    f"Failed to reconnect loss to computation graph at step {self.step}. "
                    f"loss.requires_grad={loss.requires_grad}, loss.grad_fn={loss.grad_fn}. "
                    f"Check that model parameters require gradients and forward pass is correct."
                )
            self.logger.warning(f"Step {self.step}: Successfully reconnected loss to computation graph.")

        # Backward with gradient accumulation support
        did_step = False
        grad_norm = 0.0
        
        if self.grad_accumulator is not None:
            # Use gradient accumulator (handles scaling and stepping)
            did_step = self.grad_accumulator.step(loss, self.step)
            if did_step:
                grad_norm = self.cfg.train.grad_clip  # Approximate
        elif self.scaler is not None:
            # CUDA with AMP
            self.scaler.scale(loss).backward()
            if self.cfg.train.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            did_step = True
        else:
            # Standard backward (MPS or CPU)
            loss.backward()
            if self.cfg.train.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            
            # Check for NaN gradients
            has_nan_grad = False
            for p in self.model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                self.logger.warning(f"Step {self.step}: NaN/Inf gradients detected, zeroing gradients")
                self.optimizer.zero_grad(set_to_none=True)
                return {"loss": 0.0, "phys": 0.0, "cls": 0.0, "lam": 0.0, "grad_norm": 0.0}
            
            self.optimizer.step()
            did_step = True
        
        # Update EMA only when optimizer stepped
        if did_step and self.ema is not None:
            self.ema.update()
            
        # Scheduler (warmup manual) - only when optimizer stepped
        if did_step:
            if self.step < self.cfg.train.scheduler.warmup_steps:
                lr_scale = min(1.0, float(self.step + 1) / self.cfg.train.scheduler.warmup_steps)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.cfg.train.lr * lr_scale
            else:
                self.scheduler.step()
        
        # FIXED: Aggressive MPS memory cleanup with gc
        if self.device.type == "mps" and self.step % self.memory_cleanup_every == 0:
            aggressive_memory_cleanup()
            gc.collect()  # Force Python garbage collection
            torch.mps.empty_cache()  # Explicitly clear MPS cache
            
        with torch.no_grad():
            self.metrics_buffer.add(probs.detach().cpu().numpy(), labels.detach().cpu().numpy())
        
        # Track accumulated stats for logging
        # FIXED: Avoid .item() which can hang on MPS - use detach().cpu() instead
        self._accumulated_loss += float(loss.detach().cpu())
        self._accumulated_phys += total_phys
        self._accumulated_cls += total_cls
        self._accumulated_count += 1
        
        # Return averaged stats
        if did_step and self._accumulated_count > 0:
            avg_loss = self._accumulated_loss / self._accumulated_count
            avg_phys = self._accumulated_phys / self._accumulated_count
            avg_cls = self._accumulated_cls / self._accumulated_count
            self._accumulated_loss = 0.0
            self._accumulated_phys = 0.0
            self._accumulated_cls = 0.0
            self._accumulated_count = 0
        else:
            # FIXED: Avoid .item() which can hang on MPS - use detach().cpu() instead
            avg_loss = float(loss.detach().cpu())
            avg_phys = total_phys
            avg_cls = total_cls
        
        metrics_dict = {
            "loss": avg_loss,
            "phys": avg_phys,
            "cls": avg_cls,
            "lam": out.lambda_phys,
            "alpha": out.fourier_alpha,
            "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "did_step": did_step,
        }
        
        # Log to W&B
        if self.use_wandb and self.step % self.cfg.train.log_every == 0:
            log_dict = {
                "train/loss_total": metrics_dict["loss"],
                "train/loss_cls": metrics_dict["cls"],
                "train/loss_phys": metrics_dict["phys"],
                "train/grad_norm": metrics_dict["grad_norm"],
                "physics/lambda": metrics_dict["lam"],
                "physics/fourier_alpha": metrics_dict["alpha"],
                "optimizer/lr": self.optimizer.param_groups[0]['lr'],
                "step": self.step
            }
            
            # ✅ FIX: Log GradNorm weights for debugging
            if hasattr(self.model, '_gradnorm_weights') and self.cfg.physics.enable:
                use_gradnorm = getattr(self.cfg.physics, 'use_gradnorm', False)
                if use_gradnorm:
                    log_dict["gradnorm/w_cls"] = float(self.model._gradnorm_weights[0].detach().cpu())
                    log_dict["gradnorm/w_phys"] = float(self.model._gradnorm_weights[1].detach().cpu())
            
            # Log LRA lambda if enabled
            if hasattr(self.model, '_lra_lambda') and getattr(self.cfg.physics, 'use_lra', False):
                log_dict["lra/lambda"] = float(self.model._lra_lambda.detach().cpu())
            
            wandb.log(log_dict, step=self.step)
        
        return metrics_dict

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
        
        # Use low memory mode for MPS
        with torch.no_grad(), low_memory_mode():
            # Enter EMA context if available
            if ema_context is not None:
                ema_context.__enter__()
            
            try:
                for batch in loader:
                    # Clone labels BEFORE moving to device
                    labels_original = batch["labels"].clone()
                    
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
                            "pil_mask": batch["pil_mask"][i],
                            "mode": "eval",
                        }
                        if "frames" in batch:
                            sample_kwargs["frames"] = batch["frames"][i]
                        if "scalars" in batch:
                            sample_kwargs["scalars"] = batch["scalars"][i]
                        
                        out = self.model(**sample_kwargs)
                        batch_probs.append(out.probs)
                    
                    batch_probs_cat = torch.cat(batch_probs).cpu().numpy()
                    # Use ORIGINAL labels (before device transfer) for metrics
                    batch_labels_np = labels_original.numpy()
                    
                    all_probs.append(batch_probs_cat)
                    all_labels.append(batch_labels_np)
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
            
            # DEBUG: Count before filtering
            n_before = len(y_true)
            n_pos_before = int((y_true == 1.0).sum()) if np.isfinite(y_true).all() else "?"
            n_prob_nan = int((~np.isfinite(y_prob)).sum())
            n_true_nan = int((~np.isfinite(y_true)).sum())
            
            # Filter out NaN AND Inf values (isfinite catches both!)
            valid_mask = np.isfinite(y_true) & np.isfinite(y_prob)
            y_true = y_true[valid_mask]
            y_prob = y_prob[valid_mask]
            
            # DEBUG: Log filtering stats
            n_after = len(y_true)
            n_filtered = n_before - n_after
            if n_filtered > 0:
                self.logger.warning(f"  {h}h: Filtered {n_filtered}/{n_before} samples (prob_nan={n_prob_nan}, true_nan={n_true_nan})")
            
            # Clamp probabilities to valid range [0, 1]
            y_prob = np.clip(y_prob, 0.0, 1.0)
            
            # Ensure labels are binary (0 or 1)
            y_true = np.clip(y_true, 0.0, 1.0)
            
            if len(y_true) == 0 or y_true.sum() == 0: 
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
            
            # Ensure y_true is binary (0 or 1) before counting
            # DEBUG: Check what's actually in y_true
            unique_vals = np.unique(y_true)
            if len(unique_vals) > 2 or (len(unique_vals) == 2 and not np.allclose(unique_vals, [0.0, 1.0])):
                self.logger.warning(f"  {h}h DEBUG: y_true has unexpected values: {unique_vals[:10]}")
            
            # Count positives correctly - y_true should be 0.0 or 1.0
            n_pos = int((y_true > 0.5).sum())  # More robust than .astype(int32)
            
            # Safety: clamp BS and ECE to reasonable ranges
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
            
        return results

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        self.model.train()
        t_start = time.time()
        stopped_early = False
        
        self.logger.info(f"Starting training loop... (target: {self.cfg.train.steps} steps)")
        
        while self.step < self.cfg.train.steps and not stopped_early:
            for batch in train_loader:
                if self.step == 0:
                    self.logger.info("First batch loaded from disk!")
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
                    # FIXED: Clear metrics buffer after logging to prevent memory leak
                    self.metrics_buffer.clear()
                    gc.collect()  # Force garbage collection
                    if val_loader:
                        val_results = self.evaluate(val_loader)
                        val_tss = val_results["max_tss"]
                        
                        # FIXED: Aggressive cleanup after validation to prevent leaks
                        clear_memory_cache(self.device)
                        gc.collect()  # Force Python garbage collection
                        if self.device.type == "mps":
                            torch.mps.empty_cache()  # Clear MPS cache
                        
                        # Log detailed metrics
                        self.logger.info(f"[Val] Max TSS: {val_tss:.4f}")
                        
                        # W&B logging
                        if self.use_wandb:
                            log_dict = {"val/tss_max": val_tss, "step": self.step}
                            for h, h_metrics in val_results.get("horizons", {}).items():
                                log_dict.update({
                                    f"val/tss_{h}h": h_metrics["tss"],
                                    f"val/pr_auc_{h}h": h_metrics["pr_auc"],
                                    f"val/brier_{h}h": h_metrics["brier"],
                                    f"val/ece_{h}h": h_metrics["ece"],
                                })
                            wandb.log(log_dict, step=self.step)
                        
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
                        
                        # Save checkpoint with FULL state (EMA, scheduler, early stopping)
                        if self.checkpoint_mgr:
                            self.checkpoint_mgr.save(
                                self.model, 
                                self.optimizer, 
                                self.step, 
                                val_tss,
                                scheduler=self.scheduler,
                                ema=self.ema,
                                early_stopping=self.early_stopping,
                                best_val_tss=self.best_val_tss,
                            )
                        
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
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    cfg = PINNConfig.from_yaml(args.config)
    logger = setup_logging(Path(f"outputs/logs/{Path(args.config).stem}.log"))
    # Enable wandb if requested or if WANDB_PROJECT env var is set
    use_wandb = args.wandb or bool(os.getenv("WANDB_PROJECT"))
    trainer = PINNTrainer(cfg, logger, use_wandb=use_wandb)
    
    # Resume from checkpoint (explicit --resume or auto from best_model.pt)
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Resume checkpoint not found: {resume_path}")
            return
    elif trainer.checkpoint_mgr:
        best_ckpt_path = trainer.checkpoint_mgr.checkpoint_dir / "best_model.pt"
        if best_ckpt_path.exists():
            resume_path = best_ckpt_path
    
    if resume_path is not None and trainer.checkpoint_mgr:
        # Set the path so load_best can find it
        trainer.checkpoint_mgr.best_path = resume_path
        
        # Load full state including EMA, early stopping
        # NOTE: Don't restore scheduler when using --resume (allows fine-tuning with new LR)
        resume_step, resume_metric, best_val_tss = trainer.checkpoint_mgr.load_best(
            trainer.model, 
            trainer.optimizer,
            scheduler=None if args.resume else trainer.scheduler,  # Fresh scheduler for fine-tune
            ema=trainer.ema,
            early_stopping=trainer.early_stopping,
        )
        trainer.step = resume_step
        trainer.best_val_tss = best_val_tss if best_val_tss > 0 else resume_metric
        
        # If using --resume, reset optimizer LR to config value and step scheduler
        if args.resume:
            # Reset optimizer LR to config value (checkpoint had old LR)
            new_lr = cfg.train.lr
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = new_lr
                param_group['initial_lr'] = new_lr
            
            # Step scheduler to current position
            if trainer.scheduler is not None:
                for _ in range(resume_step):
                    trainer.scheduler.step()
                current_lr = trainer.scheduler.get_last_lr()[0]
            else:
                current_lr = new_lr
            logger.info(f"   📊 LR reset to config value, now at {current_lr:.2e} (step {resume_step})")
        
        # Check if EMA was restored
        ema_restored = 'ema_state_dict' in trainer.checkpoint_mgr.load(resume_path)
        ema_status = "✅ EMA restored" if ema_restored else "⚠️ EMA not in checkpoint (starting fresh)"
        
        logger.info(f"✅ Resumed from checkpoint at step {resume_step} (best TSS: {resume_metric:.4f})")
        logger.info(f"   {ema_status}")
    
    # Load Data
    if cfg.data.use_real:
        from src.utils.masked_training import load_windows_with_mask
        
        logger.info("Loading real data...")
        logger.info("  Reading windows parquet...")
        df, mask = load_windows_with_mask(str(cfg.data.windows_parquet))
        df = df[mask].reset_index(drop=True)
        logger.info(f"  Loaded {len(df)} windows")
        
        # FIXED: Split by HARP number to prevent temporal leakage
        # Windows from same HARP should all be in train OR val, not both
        unique_harps = df['harpnum'].unique()
        np.random.seed(cfg.seed)  # Reproducible split
        np.random.shuffle(unique_harps)
        n_val_harps = max(1, int(len(unique_harps) * cfg.data.val_fraction))
        val_harps = set(unique_harps[:n_val_harps])
        
        train_df = df[~df['harpnum'].isin(val_harps)].reset_index(drop=True)
        val_df = df[df['harpnum'].isin(val_harps)].reset_index(drop=True)
        
        # Log class balance PER HORIZON
        logger.info(f"  Train: {len(train_df)} windows ({len(unique_harps) - n_val_harps} HARPs)")
        logger.info(f"  Val: {len(val_df)} windows ({n_val_harps} HARPs)")
        logger.info("  Ground truth positives:")
        for h in cfg.classifier.horizons:
            train_h = train_df[f"y_geq_M_{h}h"].sum()
            val_h = val_df[f"y_geq_M_{h}h"].sum()
            logger.info(f"    {h}h: Train={train_h:.0f}, Val={val_h:.0f}")
        
        # Choose dataset based on config
        if cfg.data.use_consolidated:
            from src.data.consolidated_dataset import ConsolidatedWindowsDataset
            logger.info("  Using CONSOLIDATED dataset (fast I/O)...")
            
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
                max_cached_harps=500,
                # SOTA features
                use_pil_evolution=getattr(cfg.data, 'use_pil_evolution', True),
                use_temporal_statistics=getattr(cfg.data, 'use_temporal_statistics', True),
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
                max_cached_harps=500,
                # SOTA features
                use_pil_evolution=getattr(cfg.data, 'use_pil_evolution', True),
                use_temporal_statistics=getattr(cfg.data, 'use_temporal_statistics', True),
            )
        else:
            from src.data.cached_dataset import CachedWindowsDataset
            logger.info("  Using CACHED dataset (slower I/O, higher quality)...")
            
            train_ds = CachedWindowsDataset(
                train_df, str(cfg.data.frames_meta_parquet), str(cfg.data.npz_root), 
                target_px=cfg.data.target_size, input_hours=cfg.data.input_hours,
                horizons=list(cfg.classifier.horizons), P_per_t=cfg.data.P_per_t, 
                pil_top_pct=cfg.data.pil_top_pct,
                training=True, augment=True, noise_std=0.02, preload=False
            )
            
            val_ds = CachedWindowsDataset(
                val_df, str(cfg.data.frames_meta_parquet), str(cfg.data.npz_root),
                target_px=cfg.data.target_size, input_hours=cfg.data.input_hours,
                horizons=list(cfg.classifier.horizons), P_per_t=cfg.data.P_per_t,
                pil_top_pct=cfg.data.pil_top_pct,
                training=False, augment=False, preload=False
            )
        
        logger.info("  Datasets created!")
        
        # Sampler
        horizon_cols = [f"y_geq_M_{h}h" for h in cfg.classifier.horizons]
        labels = train_df[horizon_cols].to_numpy().sum(axis=1) > 0
        weights = np.full(len(train_df), cfg.train.sampler.smoothing)
        weights[labels] = cfg.train.sampler.positive_multiplier
        sampler = WeightedRandomSampler(torch.tensor(weights), len(weights))
        
        logger.info("  Creating DataLoaders...")
        # MPS/Mac optimization: pin_memory can cause hangs
        use_pin_memory = False if cfg.device == "mps" else True
        
        # FIXED: Disable persistent workers on MPS - they leak memory!
        # Use persistent workers only on CUDA to avoid respawn overhead
        persistent_workers = cfg.train.num_workers > 0 and cfg.device != "mps"
        
        # ✅ FIX #17: Seed numpy RNG per worker to avoid identical augmentations
        def worker_init_fn(worker_id: int):
            worker_seed = cfg.seed + worker_id
            np.random.seed(worker_seed)
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=cfg.train.batch_size, 
            sampler=sampler, 
            num_workers=cfg.train.num_workers, 
            pin_memory=use_pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if cfg.train.num_workers > 0 else None,
            worker_init_fn=worker_init_fn if cfg.train.num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=cfg.train.batch_size, 
            shuffle=False, 
            num_workers=cfg.train.num_workers, 
            pin_memory=use_pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if cfg.train.num_workers > 0 else None,
            worker_init_fn=worker_init_fn if cfg.train.num_workers > 0 else None
        )
        logger.info(f"  DataLoaders ready! Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    else:
        # Dummy fallback - use DummyPINNDataset defined in this module
        ds = DummyPINNDataset(cfg)
        train_loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True)
        val_loader = None

    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
