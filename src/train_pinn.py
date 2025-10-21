#!/usr/bin/env python3
# src/train_pinn.py
"""
Training entrypoint for the 2.5D PINN (P0→P4s curriculum).

Supports two data modes:
 - Dummy MMS-style synthetic sequences (for smoke tests; default)
 - Real SHARP data windows (windows_*.parquet + frames_meta.parquet + NPZ frames)

Usage
-----
python src/train_pinn.py --cfg configs/train_pinn_smoke_test.yaml   # dummy data
python src/train_pinn.py --cfg configs/train_pinn_real.yaml         # real SHARP data
"""

from __future__ import annotations
import argparse
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Eval metrics
from src.eval.metrics import (
    pr_auc,
    brier_score,
    adaptive_ece,
    sweep_tss,
    select_threshold_at_far,
)


# -----------------------------
# Focal Loss for class imbalance
# -----------------------------
def focal_loss(probs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss (Lin et al. 2017) - downweights easy examples, focuses on hard cases.
    
    Args:
        probs: predicted probabilities [B, H] (after sigmoid)
        targets: ground truth {0,1} [B, H]
        alpha: weight for positive class (0.25 typical for rare events)
        gamma: focusing parameter (2.0 standard; higher = more focus on hard examples)
    """
    bce = nn.functional.binary_cross_entropy(probs, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    return (alpha_t * focal_weight * bce).mean()


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    horizons: Tuple[int, ...] = (6, 12, 24)
    steps: int = 50000
    batch_size: int = 1
    lr: float = 1e-3
    grad_clip: float = 1.0
    amp: bool = True
    log_every: int = 25
    eval_every: int = 500
    loss_type: str = "focal"  # "bce" or "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # Data config
    use_real: bool = False
    windows_parquet: Optional[str] = None
    frames_meta_parquet: Optional[str] = None
    npz_root: Optional[str] = None
    target_size: int = 256
    input_hours: int = 48


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    train = raw.get("train", {})
    classifier = raw.get("classifier", {})
    data = raw.get("data", {})
    return TrainConfig(
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "cuda")),
        horizons=tuple(classifier.get("horizons", [6, 12, 24])),
        steps=int(train.get("steps", 50000)),
        batch_size=int(train.get("batch_size", 1)),
        lr=float(train.get("lr", 1e-3)),
        grad_clip=float(train.get("grad_clip", 1.0)),
        amp=bool(train.get("amp", True)),
        log_every=int(train.get("log_every", 25)),
        eval_every=int(train.get("eval_every", 500)),
        loss_type=str(classifier.get("loss_type", "focal")),
        focal_alpha=float(classifier.get("focal_alpha", 0.25)),
        focal_gamma=float(classifier.get("focal_gamma", 2.0)),
        use_real=bool(data.get("use_real", False)),
        windows_parquet=data.get("windows_parquet"),
        frames_meta_parquet=data.get("frames_meta_parquet"),
        npz_root=data.get("npz_root"),
        target_size=int(data.get("target_size", 256)),
        input_hours=int(data.get("input_hours", 48)),
    )


# -----------------------------
# Dummy Dataset (for smoke tests)
# -----------------------------
class DummyDataset(torch.utils.data.Dataset):
    """
    Dummy dataset for quick testing (MMS-style synthetic).
    Returns (inputs, labels) where labels = [B, H].
    """
    def __init__(self, num_samples: int = 64, horizons: int = 3):
        self.num = num_samples
        self.h = horizons

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int):
        # Dummy inputs & labels
        x = torch.randn(1, 8, 64, 64)          # e.g., [C,T,H,W]
        y = (torch.rand(self.h) > 0.8).float() # ~20% positive rate
        return x, y


# -----------------------------
# Real SHARP Dataset Loader
# -----------------------------
def load_windows_with_mask(parquet_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load windows parquet and return (df, mask) where mask indicates has_noaa_mapping==True.
    """
    df = pd.read_parquet(parquet_path)
    if "has_noaa_mapping" in df.columns:
        mask = df["has_noaa_mapping"].fillna(False).astype(bool).to_numpy()
    else:
        mask = np.ones(len(df), dtype=bool)
    return df, mask


# -----------------------------
# Dummy Model (for smoke tests)
# -----------------------------
class DummyPINNModel(nn.Module):
    """
    Minimal placeholder model for testing.
    forward(...) returns: {"logits": logits} with logits shape [B, H].
    """
    def __init__(self, horizons: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 8 * 64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, horizons)

    def forward(self, inputs: torch.Tensor):
        z = self.encoder(inputs)
        logits = self.head(z)
        return {"logits": logits}


# -----------------------------
# Training + Eval
# -----------------------------
def train_loop(cfg: TrainConfig, model: nn.Module, loader: torch.utils.data.DataLoader) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    horizons = list(cfg.horizons)
    step = 0
    t0 = time.time()

    # small rolling buffers for quick eval
    recent_probs: List[np.ndarray] = []
    recent_labels: List[np.ndarray] = []

    model.train()
    while step < cfg.steps:
        for batch in loader:
            step += 1
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()  # [B, H]

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
                out = model(inputs)
                logits = out["logits"]                  # [B, H]
                probs = torch.sigmoid(logits)
                
                # Loss: Focal (for class imbalance) or BCE
                if cfg.loss_type == "focal":
                    loss = focal_loss(probs, labels, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
                else:
                    loss = torch.nn.functional.binary_cross_entropy(probs, labels, reduction="mean")

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            if (step % cfg.log_every == 0) or (step == 1):
                dt = time.time() - t0
                lr = opt.param_groups[0]["lr"]
                print(f"[{step:05d}/{cfg.steps}] loss={loss.item():.4f}  lr={lr:.2e}  dt={dt:.1f}s")
                t0 = time.time()

            # Collect recent predictions for quick eval
            with torch.no_grad():
                recent_probs.append(probs.detach().cpu().numpy())   # [B, H]
                recent_labels.append(labels.detach().cpu().numpy()) # [B, H]
                if len(recent_probs) > 50:  # keep memory bounded
                    recent_probs.pop(0); recent_labels.pop(0)

            # Periodic evaluation
            if (cfg.eval_every > 0) and ((step % cfg.eval_every == 0) or (step == cfg.steps)):
                evaluate_window(horizons, recent_probs, recent_labels)

            if step >= cfg.steps:
                break

    print("Training complete.")


def evaluate_window(horizons: List[int], prob_buf: List[np.ndarray], label_buf: List[np.ndarray]) -> None:
    """
    Quick research eval over a recent-window buffer.
    Reports TSS, PR-AUC, Brier, ECE per horizon.
    """
    if not prob_buf:
        print("Eval: no data collected yet.")
        return
    probs = np.concatenate(prob_buf, axis=0)   # [N, H]
    labels = np.concatenate(label_buf, axis=0) # [N, H]

    logs = []
    H = probs.shape[1]
    assert H == len(horizons), f"Mismatch: probs has H={H}, horizons={horizons}"
    for j, h in enumerate(horizons):
        yj = labels[:, j].astype(float)
        pj = probs[:, j].astype(float)
        thr, tss_val = sweep_tss(yj, pj, n=256)
        thr_far = select_threshold_at_far(yj, pj, max_far=0.05, n=512)
        pr = pr_auc(yj, pj)
        bs = brier_score(yj, pj)
        ece = adaptive_ece(yj, pj, n_bins=10)
        logs.append(
            f"h={h}  TSS*={tss_val:.3f}@{thr:.2f}  PR-AUC={pr:.3f}  "
            f"Brier={bs:.3f}  ECE~{ece:.3f}  thr@FAR5={thr_far:.2f}"
        )
    print("Eval:", " | ".join(logs))


# -----------------------------
# Entrypoint
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", "--config", type=str, required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    set_seed(cfg.seed)

    # Data setup: real SHARP or dummy
    if cfg.use_real:
        # TODO: Import WindowsDataset when ready
        # from src.data.windows_dataset import WindowsDataset
        # 
        # windows_df, labeled_mask = load_windows_with_mask(cfg.windows_parquet)
        # windows_df = windows_df[labeled_mask].reset_index(drop=True)
        # 
        # dataset = WindowsDataset(
        #     windows_df=windows_df,
        #     frames_meta_path=cfg.frames_meta_parquet,
        #     npz_root=cfg.npz_root,
        #     target_px=cfg.target_size,
        #     input_hours=cfg.input_hours,
        #     horizons=list(cfg.horizons),
        # )
        raise NotImplementedError("Real SHARP data integration is not yet complete. Use use_real: false for dummy data.")
    else:
        # Dummy data
        dataset = DummyDataset(num_samples=2048, horizons=len(cfg.horizons))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    model = DummyPINNModel(horizons=len(cfg.horizons))
    train_loop(cfg, model, loader)


if __name__ == "__main__":
    main()
