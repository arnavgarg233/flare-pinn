#!/usr/bin/env python3
"""
PyTorch Dataset for SHARP CEA magnetogram windows.

Loads preprocessed windows from parquet + NPZ magnetograms.
Computes PIL masks on-the-fly from |∇Bz|.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
from torch.utils.data import Dataset
from scipy.ndimage import binary_closing, binary_erosion
from skimage.morphology import skeletonize


# ===================== PIL Mask Generation (Blueprint Recipe) =====================

def compute_pil_mask(
    Bz: np.ndarray, 
    top_pct: float = 0.15,
    closing_radius: int = 2,
    do_skeleton: bool = True
) -> np.ndarray:
    """
    Compute PIL (Polarity Inversion Line) mask from Bz using blueprint recipe.
    
    Recipe (Section 3.3):
    1. Compute |∇Bz| via Sobel (3×3)
    2. Take top 15% pixels by magnitude
    3. Morphological closing (disk radius 2px)
    4. Binary skeletonize
    
    Args:
        Bz: [H, W] magnetic field component
        top_pct: Top percentile threshold (default 0.15 = 15%)
        closing_radius: Radius for morphological closing
        do_skeleton: Whether to skeletonize
        
    Returns:
        pil_mask: [H, W] binary mask {0,1}
    """
    from scipy.ndimage import sobel
    
    # Handle NaNs
    Bz = np.nan_to_num(Bz, nan=0.0)
    
    # Sobel gradients
    grad_x = sobel(Bz, axis=1, mode='constant', cval=0.0)
    grad_y = sobel(Bz, axis=0, mode='constant', cval=0.0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Top percentile threshold
    if grad_mag.max() == 0:
        return np.zeros_like(Bz, dtype=np.uint8)
    
    threshold = np.percentile(grad_mag[grad_mag > 0], 100 * (1 - top_pct))
    mask = (grad_mag >= threshold).astype(np.uint8)
    
    # Morphological closing (connect nearby regions)
    if closing_radius > 0:
        # Create disk structuring element
        y, x = np.ogrid[-closing_radius:closing_radius+1, -closing_radius:closing_radius+1]
        disk = (x**2 + y**2 <= closing_radius**2).astype(np.uint8)
        mask = binary_closing(mask, structure=disk).astype(np.uint8)
    
    # Skeletonize (thin to 1-pixel lines)
    if do_skeleton and mask.sum() > 0:
        mask = skeletonize(mask).astype(np.uint8)
    
    return mask


# ===================== SHARP Windows Dataset =====================

class SHARPWindowsDataset(Dataset):
    """
    PyTorch Dataset for SHARP CEA magnetogram windows.
    
    Each sample is a 48h window of magnetogram frames + labels for 6/12/24h horizons.
    
    Args:
        windows_parquet: Path to windows_{train,val,test}.parquet
        frames_meta_parquet: Path to frames metadata (optional, for paths)
        npz_root: Root directory containing processed NPZ files
        filter_labeled: If True, only return windows with has_noaa_mapping==True
        target_size: Spatial resolution (default 256×256)
        horizons: Prediction horizons in hours (default [6,12,24])
    """
    
    def __init__(
        self,
        windows_parquet: str | Path,
        npz_root: str | Path,
        filter_labeled: bool = True,
        target_size: int = 256,
        horizons: Tuple[int, ...] = (6, 12, 24),
        frames_meta_parquet: Optional[str | Path] = None,
    ):
        self.npz_root = Path(npz_root)
        self.target_size = target_size
        self.horizons = tuple(horizons)
        
        # Load windows
        print(f"[SHARPWindowsDataset] Loading windows from {windows_parquet}...")
        self.windows = pd.read_parquet(windows_parquet)
        
        # Filter to labeled windows only
        if filter_labeled and 'has_noaa_mapping' in self.windows.columns:
            n_before = len(self.windows)
            self.windows = self.windows[self.windows['has_noaa_mapping']].reset_index(drop=True)
            n_after = len(self.windows)
            print(f"  Filtered to labeled: {n_after}/{n_before} ({100*n_after/n_before:.1f}%)")
        
        # Load frame metadata (REQUIRED for finding NPZ paths)
        self.frames_meta = None
        self.frames_lookup = {}
        if frames_meta_parquet is not None:
            self.frames_meta = pd.read_parquet(frames_meta_parquet)
            # Build lookup: (harpnum, timestamp) -> path
            for _, row in self.frames_meta.iterrows():
                ts = pd.Timestamp(row['date_obs'])
                key = (int(row['harpnum']), ts)
                self.frames_lookup[key] = self.npz_root / row['frame_path']
            print(f"  Loaded {len(self.frames_meta)} frame metadata entries")
        
        print(f"  Dataset ready: {len(self.windows)} windows")
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def _find_npz_path(self, harpnum: int, date_obs: pd.Timestamp) -> Optional[Path]:
        """
        Find NPZ file for given HARP + timestamp using frames_lookup.
        """
        # Use lookup table if available
        if self.frames_lookup:
            key = (harpnum, date_obs)
            if key in self.frames_lookup:
                return self.frames_lookup[key]
        
        # Fallback: try common patterns
        date_str = date_obs.strftime("%Y-%m-%d")
        time_str = date_obs.strftime("T%H-%M-%S")
        
        patterns = [
            self.npz_root / f"frames/H{harpnum}_{date_str}{time_str}.npz",
            self.npz_root / f"H{harpnum}_{date_str}{time_str}.npz",
        ]
        
        for candidate in patterns:
            if candidate.exists():
                return candidate
        
        return None
    
    def _load_frame(self, harpnum: int, date_obs: pd.Timestamp) -> Optional[Dict[str, np.ndarray]]:
        """Load single NPZ frame. Returns dict with Bx, By, Bz, mask."""
        npz_path = self._find_npz_path(harpnum, date_obs)
        if npz_path is None:
            return None
        
        try:
            data = np.load(npz_path)
            
            # Expected keys: Bx, By, Bz, mask (or similar)
            # Handle both processed (Bx/By/Bz) and raw (need to check keys)
            out = {}
            
            # Try direct keys
            for k in ['Bx', 'By', 'Bz']:
                if k in data:
                    out[k] = data[k]
            
            # Mask
            if 'mask' in data:
                out['mask'] = data['mask']
            elif 'valid_mask' in data:
                out['mask'] = data['valid_mask']
            else:
                # Create mask from NaNs
                if 'Bz' in out:
                    out['mask'] = ~np.isnan(out['Bz'])
            
            # Validate we got the essentials
            if 'Bz' not in out:
                return None
            
            return out
            
        except Exception as e:
            # print(f"Warning: Failed to load {npz_path}: {e}")
            return None
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            coords_grid: [T, H, W, 3] in [-1, 1]^3
            Bx_obs: [T, H, W, 1]
            By_obs: [T, H, W, 1]
            Bz_obs: [T, H, W, 1]
            observed_mask: [T] bool (True for t <= t0)
            pil_mask: [H, W] float {0,1}
            labels: [len(horizons)] float {0,1}
            meta: dict with harpnum, t0, window_uid
        """
        row = self.windows.iloc[idx]
        
        harpnum = int(row['harpnum'])
        t0 = pd.Timestamp(row['t0'])
        input_hours = int(row['input_span_hours'])
        
        # Build time grid for input window [t0 - input_hours, t0]
        # Assume 1h cadence (adjust if needed)
        n_frames = input_hours + 1  # inclusive
        times = pd.date_range(t0 - pd.Timedelta(hours=input_hours), t0, freq='1h')
        
        # Preallocate arrays
        H, W = self.target_size, self.target_size
        T = len(times)
        
        Bx_stack = np.zeros((T, H, W, 1), dtype=np.float32)
        By_stack = np.zeros((T, H, W, 1), dtype=np.float32)
        Bz_stack = np.zeros((T, H, W, 1), dtype=np.float32)
        mask_stack = np.zeros((T, H, W), dtype=np.uint8)
        
        # Load frames
        for i, ts in enumerate(times):
            frame = self._load_frame(harpnum, ts)
            if frame is None:
                # Missing frame - leave as zeros, mask will be 0
                continue
            
            # Resize if needed
            if 'Bz' in frame:
                Bz = frame['Bz']
                if Bz.shape[:2] != (H, W):
                    import cv2
                    Bz = cv2.resize(Bz, (W, H), interpolation=cv2.INTER_LINEAR)
                Bz_stack[i, :, :, 0] = Bz
            
            if 'Bx' in frame:
                Bx = frame['Bx']
                if Bx.shape[:2] != (H, W):
                    import cv2
                    Bx = cv2.resize(Bx, (W, H), interpolation=cv2.INTER_LINEAR)
                Bx_stack[i, :, :, 0] = Bx
            
            if 'By' in frame:
                By = frame['By']
                if By.shape[:2] != (H, W):
                    import cv2
                    By = cv2.resize(By, (W, H), interpolation=cv2.INTER_LINEAR)
                By_stack[i, :, :, 0] = By
            
            if 'mask' in frame:
                m = frame['mask']
                if m.shape[:2] != (H, W):
                    import cv2
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                mask_stack[i] = m
        
        # Coordinate grid [-1, 1]^3
        xs = np.linspace(-1, 1, W, dtype=np.float32)
        ys = np.linspace(-1, 1, H, dtype=np.float32)
        ts_norm = np.linspace(-1, 1, T, dtype=np.float32)  # -1 = earliest, +1 = t0
        
        X, Y, Tt = np.meshgrid(xs, ys, ts_norm, indexing='xy')
        coords_grid = np.stack([X, Y, Tt], axis=-1)  # [W, H, T, 3]
        coords_grid = coords_grid.transpose(2, 1, 0, 3)  # [T, H, W, 3]
        
        # Observed mask (all frames are observed in input window)
        observed_mask = np.ones(T, dtype=bool)
        
        # PIL mask from last observed frame
        pil_mask = compute_pil_mask(Bz_stack[-1, :, :, 0], top_pct=0.15)
        
        # Labels for each horizon
        labels = np.zeros(len(self.horizons), dtype=np.float32)
        for i, h in enumerate(self.horizons):
            col = f'y_geq_M_{h}h'
            if col in row:
                labels[i] = float(row[col])
        
        # Convert to tensors
        return {
            'coords_grid': torch.from_numpy(coords_grid),
            'Bx_obs': torch.from_numpy(Bx_stack),
            'By_obs': torch.from_numpy(By_stack),
            'Bz_obs': torch.from_numpy(Bz_stack),
            'observed_mask': torch.from_numpy(observed_mask),
            'pil_mask': torch.from_numpy(pil_mask.astype(np.float32)),
            'labels': torch.from_numpy(labels),
            'meta': {
                'harpnum': harpnum,
                't0': str(t0),
                'window_uid': str(row.get('window_uid', f'{harpnum}_{t0}')),
            }
        }


# ===================== Quick Test =====================

if __name__ == "__main__":
    # Quick smoke test
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python sharp_dataset.py <windows.parquet> <npz_root>")
        sys.exit(1)
    
    windows_path = sys.argv[1]
    npz_root = sys.argv[2]
    
    ds = SHARPWindowsDataset(
        windows_parquet=windows_path,
        npz_root=npz_root,
        filter_labeled=True,
    )
    
    print(f"\n[Test] Dataset length: {len(ds)}")
    
    if len(ds) > 0:
        print(f"[Test] Loading sample 0...")
        sample = ds[0]
        
        print(f"  coords_grid: {sample['coords_grid'].shape}")
        print(f"  Bz_obs: {sample['Bz_obs'].shape}, range [{sample['Bz_obs'].min():.2f}, {sample['Bz_obs'].max():.2f}]")
        print(f"  pil_mask: {sample['pil_mask'].shape}, sum={sample['pil_mask'].sum()}")
        print(f"  labels: {sample['labels']}")
        print(f"  meta: {sample['meta']}")
        print(f"\n[Test] [OK] Dataset working!")

