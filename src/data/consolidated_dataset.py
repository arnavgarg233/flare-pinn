# src/data/consolidated_dataset.py
"""
Dataset for loading consolidated per-HARP frame bundles.

MUCH faster than loading individual NPZ files:
- 1 file open per HARP instead of ~1000
- Pre-processed and normalized
- Float16 storage (smaller, faster to load)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .pil_mask import (
    pil_mask_from_bz,
    compute_r_value,
    compute_pil_gradient_weighted_length,
)


# =============================================================================
# Helper Functions
# =============================================================================

def _linspace_times(t0_utc: pd.Timestamp, hours: int) -> list[pd.Timestamp]:
    """Generate hourly timestamps from (t0 - hours) to t0."""
    start = t0_utc - pd.Timedelta(hours=hours)
    return list(pd.date_range(start, t0_utc, freq="1h", inclusive="both"))


def _coords_grid(T: int, P: int, device: torch.device) -> torch.Tensor:
    """Generate random collocation coordinates."""
    xy = torch.rand(T, P, 2, device=device) * 2.0 - 1.0
    t = torch.linspace(-1.0, 1.0, T, device=device)[:, None, None].expand(T, P, 1)
    coords = torch.cat([xy, t], dim=-1)
    return coords


def _bilinear_sample(img: torch.Tensor, xy_norm: torch.Tensor) -> torch.Tensor:
    """
    Bilinear interpolation of field at coordinates using F.grid_sample.
    
    IMPROVED: Uses PyTorch's optimized grid_sample for better performance
    and proper handling of edge cases.
    
    Args:
        img: [C, H, W] or [H, W] field
        xy_norm: [P, 2] normalized coordinates in [-1, 1]
    Returns:
        [P, C] sampled values
    """
    if img.ndim == 2:
        img = img.unsqueeze(0)  # [1, H, W]
    
    C, H, W = img.shape
    P = xy_norm.shape[0]
    
    # F.grid_sample expects [N, C, H_in, W_in] and grid [N, H_out, W_out, 2]
    # For our case: N=1, H_out=1, W_out=P
    img_batch = img.unsqueeze(0)  # [1, C, H, W]
    
    # grid_sample expects (x, y) order where x is along W and y is along H
    # Our xy_norm is already in (x, y) order and normalized to [-1, 1]
    grid = xy_norm.view(1, 1, P, 2)  # [1, 1, P, 2]
    
    # Sample using bilinear interpolation with zero padding for out-of-bounds
    sampled = F.grid_sample(
        img_batch, 
        grid, 
        mode='bilinear', 
        padding_mode='border',  # Use border values for out-of-bounds
        align_corners=True  # Match our coordinate convention
    )  # [1, C, 1, P]
    
    return sampled.squeeze(0).squeeze(1).permute(1, 0)  # [P, C]


def _resample_gt_from_frames(frames: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Recompute ground-truth samples after geometric augmentations.
    
    Args:
        frames: [T, C, H, W] frames
        coords: [T, P, 3] normalized coordinates
    Returns:
        gt: [T, P, C]
    """
    T, P = coords.shape[0], coords.shape[1]
    C = frames.shape[1]
    device = frames.device
    gt = torch.zeros(T, P, C, device=device, dtype=frames.dtype)
    for t in range(T):
        gt[t] = _bilinear_sample(frames[t], coords[t, :, 0:2])
    return gt


# =============================================================================
# Consolidated HARP Cache
# =============================================================================

class HARPCache:
    """
    In-memory LRU cache for consolidated HARP files.
    
    Loads entire HARP bundles into memory for fast access.
    Uses OrderedDict for O(1) LRU eviction.
    
    FIXED: Thread-safe for DataLoader with num_workers > 0.
    Uses RLock to allow recursive locking within same thread.
    """
    def __init__(self, consolidated_dir: Path, max_harps: int = 500):
        from collections import OrderedDict
        import threading
        
        self.consolidated_dir = consolidated_dir
        self.max_harps = max_harps
        self._cache: OrderedDict[int, tuple[np.ndarray, dict[str, int]]] = OrderedDict()
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Load manifest
        manifest_path = consolidated_dir / "manifest.pkl"
        if manifest_path.exists():
            with open(manifest_path, "rb") as f:
                self.manifest = pickle.load(f)
        else:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    def get_frame(self, harpnum: int, timestamp_iso: str) -> Optional[np.ndarray]:
        """Get a single frame from cache or disk (thread-safe)."""
        with self._lock:
            if harpnum not in self._cache:
                self._load_harp(harpnum)
            else:
                # Move to end (most recently used) - O(1) with OrderedDict
                self._cache.move_to_end(harpnum)
            
            if harpnum not in self._cache:
                return None
            
            frames, ts_index = self._cache[harpnum]
            idx = ts_index.get(timestamp_iso)
            if idx is None:
                return None
            
            # Return a copy to avoid issues with concurrent access to numpy arrays
            return frames[idx].astype(np.float32).copy()
    
    def _load_harp(self, harpnum: int) -> None:
        """Load a HARP bundle into cache (must be called with lock held)."""
        if harpnum not in self.manifest:
            return
        
        # Evict oldest (first) if at capacity - O(1) with OrderedDict
        while len(self._cache) >= self.max_harps:
            self._cache.popitem(last=False)  # Remove oldest (first item)
        
        # Load from disk
        harp_file = self.consolidated_dir / f"H{harpnum}.npz"
        if not harp_file.exists():
            return
        
        try:
            data = np.load(harp_file, allow_pickle=True)
            frames = data["frames"]  # [T, H, W] float16
            timestamps = data["timestamps"]  # array of ISO strings
            
            # Build timestamp -> index mapping
            ts_index = {str(ts): i for i, ts in enumerate(timestamps)}
            
            self._cache[harpnum] = (frames, ts_index)
        except Exception:
            pass
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# =============================================================================
# Main Dataset
# =============================================================================

class ConsolidatedWindowsDataset(Dataset):
    """
    Fast dataset using consolidated per-HARP frame bundles.
    
    Advantages over CachedWindowsDataset:
    - 1 file open per HARP instead of ~1000 individual NPZs
    - Pre-processed and normalized data
    - Float16 storage (smaller, faster I/O)
    - Much faster on external drives
    
    Usage:
        dataset = ConsolidatedWindowsDataset(
            windows_df=train_df,
            consolidated_dir="~/flare_data/consolidated",
            target_px=64,
            input_hours=48,
        )
    """
    
    def __init__(
        self,
        windows_df: pd.DataFrame,
        consolidated_dir: str | Path,
        target_px: int = 64,
        input_hours: int = 48,
        horizons: list[int] | None = None,
        P_per_t: int = 512,
        pil_top_pct: float = 0.15,
        training: bool = True,
        augment: bool = True,
        noise_std: float = 0.02,
        max_cached_harps: int = 500,
    ):
        if horizons is None:
            horizons = [6, 12, 24]
        
        self.df = windows_df.reset_index(drop=True)
        self.consolidated_dir = Path(consolidated_dir).expanduser()
        self.target_px = int(target_px)
        self.input_hours = int(input_hours)
        self.horizons = list(horizons)
        self.P_per_t = int(P_per_t)
        self.pil_top_pct = float(pil_top_pct)
        self.training = training
        self.augment = augment and training
        self.noise_std = noise_std
        
        # Initialize HARP cache
        self.cache = HARPCache(self.consolidated_dir, max_harps=max_cached_harps)
        
        print(f"ConsolidatedDataset ready: {len(self.df)} windows, {len(self.cache.manifest)} HARPs available")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        harp = int(row["harpnum"])
        t0 = pd.to_datetime(row["t0"], utc=True)
        times = _linspace_times(t0, self.input_hours)
        T = len(times)
        
        device = torch.device("cpu")
        coords = _coords_grid(T, self.P_per_t, device)
        
        observed = np.zeros((T,), np.bool_)
        gt_b = torch.zeros(T, self.P_per_t, 3, device=device)
        frames = torch.zeros(T, 3, self.target_px, self.target_px, device=device)
        last_obs_bz: Optional[torch.Tensor] = None
        
        for ti, ts in enumerate(times):
            ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            frame_np = self.cache.get_frame(harp, ts_iso)
            
            if frame_np is None:
                continue
            
            observed[ti] = True
            frame_tensor = torch.from_numpy(frame_np).to(device)
            
            # Handle old 2D format (Bz only) vs new 3D format (Bx, By, Bz)
            if frame_tensor.ndim == 2:
                # Back-compat: treat as Bz, fill Bx/By with zeros
                h, w = frame_tensor.shape
                bz = frame_tensor.unsqueeze(0)
                bx = torch.zeros_like(bz)
                by = torch.zeros_like(bz)
                frame_tensor = torch.cat([bx, by, bz], dim=0)
            
            # Resize if consolidated size differs from target_px
            # frame_tensor is [3, H, W]
            if frame_tensor.shape[-1] != self.target_px:
                frame_tensor = F.interpolate(
                    frame_tensor.unsqueeze(0),
                    size=(self.target_px, self.target_px),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            if not torch.isfinite(frame_tensor).all():
                frame_tensor = torch.nan_to_num(frame_tensor, nan=0.0, posinf=3.0, neginf=-3.0)
            
            # Sample at collocation points
            xy = coords[ti, :, 0:2]
            gt = _bilinear_sample(frame_tensor, xy) # [P, 3]
            if not torch.isfinite(gt).all():
                gt = torch.nan_to_num(gt, nan=0.0, posinf=3.0, neginf=-3.0)
            gt_b[ti] = gt
            
            frames[ti] = frame_tensor
            last_obs_bz = frame_tensor[2] # Bz is index 2 (Bx, By, Bz)
        
        observed_mask = torch.from_numpy(observed)
        
        if not observed.any():
            observed[0] = True
            observed_mask = torch.from_numpy(observed)
        
        # Labels
        labels = []
        for H in self.horizons:
            col = f"y_geq_M_{H}h"
            labels.append(float(bool(row[col])) if col in row.index else 0.0)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # PIL mask and physics-based scalar features
        if last_obs_bz is not None:
            bz_np = last_obs_bz.numpy()
            pil_mask_np = pil_mask_from_bz(bz_np, top_percent=self.pil_top_pct)
            
            # Compute R-value and GWPIL (key flare predictors from literature)
            r_value = compute_r_value(bz_np, pil_mask_np)
            gwpil = compute_pil_gradient_weighted_length(bz_np, pil_mask_np)
        else:
            pil_mask_np = np.zeros((self.target_px, self.target_px), dtype=np.uint8)
            r_value = 0.0
            gwpil = 0.0
        pil_mask_tensor = torch.from_numpy(pil_mask_np.astype(np.float32))
        
        # Compute observation coverage
        obs_coverage = float(observed.sum()) / T
        
        # Build scalar features tensor
        scalars = torch.tensor([
            r_value,
            gwpil,
            obs_coverage,
            float(observed.sum()),  # frame_count_in_window
        ], dtype=torch.float32)
        
        # Augmentation
        if self.augment:
            geom_changed = False
            if np.random.rand() > 0.5:
                frames = torch.flip(frames, dims=[-1])
                pil_mask_tensor = torch.flip(pil_mask_tensor, dims=[-1])
                coords = coords.clone()
                coords[..., 0] = -coords[..., 0]
                geom_changed = True
            
            if np.random.rand() > 0.5:
                frames = torch.flip(frames, dims=[-2])
                pil_mask_tensor = torch.flip(pil_mask_tensor, dims=[-2])
                coords = coords.clone()
                coords[..., 1] = -coords[..., 1]
                geom_changed = True
            
            k = np.random.randint(0, 4)
            if k > 0:
                frames = torch.rot90(frames, k, dims=[-2, -1])
                pil_mask_tensor = torch.rot90(pil_mask_tensor, k, dims=[-2, -1])
                coords = coords.clone()
                for _ in range(k):
                    old_x, old_y = coords[..., 0].clone(), coords[..., 1].clone()
                    coords[..., 0] = -old_y
                    coords[..., 1] = old_x
                geom_changed = True
            
            if geom_changed:
                gt_b = _resample_gt_from_frames(frames, coords)
            
            if self.noise_std > 0:
                frames = frames + torch.randn_like(frames) * self.noise_std
                gt_b = gt_b + torch.randn_like(gt_b) * (self.noise_std * 0.5)
        
        return {
            "coords": coords,
            "gt_bz": gt_b,  # Keep key name for backward compatibility with train.py
            "frames": frames,
            "observed_mask": observed_mask,
            "labels": labels_tensor,
            "pil_mask": pil_mask_tensor,
            "scalars": scalars,
        }

