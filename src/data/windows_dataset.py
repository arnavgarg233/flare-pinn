# src/data/windows_dataset.py
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .pil_mask import pil_mask_from_bz

def _linspace_times(t0_utc: pd.Timestamp, hours: int) -> List[pd.Timestamp]:
    # Inclusive window: [t0 - hours, t0] at 1h cadence → hours+1 frames expected
    start = t0_utc - pd.Timedelta(hours=hours)
    return list(pd.date_range(start, t0_utc, freq="1h", inclusive="both"))

def make_expected_grid_times(t0: pd.Timestamp, input_hours: int) -> np.ndarray:
    return np.array(_linspace_times(t0, input_hours), dtype="datetime64[ns]")

def _coords_grid(T: int, P: int, device: torch.device) -> torch.Tensor:
    xy = torch.rand(T, P, 2, device=device) * 2.0 - 1.0  # [-1,1]^2
    t = torch.linspace(-1.0, 1.0, T, device=device)[:, None, None].expand(T, P, 1)
    coords = torch.cat([xy, t], dim=-1)  # [T,P,3]
    coords.requires_grad_(True)
    return coords

def _bilinear_sample_bz(bz: np.ndarray, xy_norm: torch.Tensor) -> torch.Tensor:
    """
    bz: [H,W] numpy
    xy_norm: [N,2] torch in [-1,1]^2
    returns [N,1] torch
    """
    H, W = bz.shape
    x = (xy_norm[:, 0] + 1.0) * 0.5 * (W - 1)
    y = (xy_norm[:, 1] + 1.0) * 0.5 * (H - 1)
    x0 = torch.clamp(x.floor().long(), 0, W - 2)
    y0 = torch.clamp(y.floor().long(), 0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (x - x0.float()).unsqueeze(-1)
    wy = (y - y0.float()).unsqueeze(-1)
    # gather 4 neighbors
    def gather(ix, iy):
        return torch.from_numpy(bz[iy.cpu().numpy(), ix.cpu().numpy()]).to(xy_norm.device).float().unsqueeze(-1)
    v00 = gather(x0, y0); v01 = gather(x0, y1)
    v10 = gather(x1, y0); v11 = gather(x1, y1)
    v0 = v00 * (1 - wy) + v01 * wy
    v1 = v10 * (1 - wy) + v11 * wy
    v = v0 * (1 - wx) + v1 * wx
    return v  # [N,1]

class WindowsDataset(Dataset):
    """
    Yields batches for PINN:
      - coords [T,P,3] in [-1,1]^3
      - gt_bz [T,P,1] sampled from frames on observed times only (mask provided)
      - observed_mask [T]
      - labels [n_horizons] (y_geq_M_*h)
      - pil_mask [H,W] (binary) computed from ∣∇Bz∣ on the last observed frame
    """
    def __init__(
        self,
        windows_df: pd.DataFrame,
        frames_meta_path: str,
        npz_root: str,
        target_px: int = 256,
        input_hours: int = 48,
        horizons: List[int] = [6,12,24],
        P_per_t: int = 1024,
        pil_top_pct: float = 0.15,
    ):
        self.df = windows_df.reset_index(drop=True)
        self.meta = pd.read_parquet(frames_meta_path)
        self.root = Path(npz_root)
        self.target_px = int(target_px)
        self.input_hours = int(input_hours)
        self.horizons = list(horizons)
        self.P_per_t = int(P_per_t)
        self.pil_top_pct = float(pil_top_pct)

        # Minimal index for faster lookup
        self.meta["date_obs"] = pd.to_datetime(self.meta["date_obs"], utc=True)
        self.meta = self.meta.sort_values(["harpnum","date_obs"]).reset_index(drop=True)
        
        # Suppress timezone warnings
        import warnings
        warnings.filterwarnings('ignore', message='.*timezones available.*')

    def __len__(self) -> int:
        return len(self.df)

    def _frames_for_window(self, harpnum: int, t0: pd.Timestamp) -> Tuple[List[pd.Timestamp], List[Optional[str]]]:
        expect_times = _linspace_times(t0, self.input_hours)
        g = self.meta[self.meta["harpnum"]==int(harpnum)]
        g_times = g["date_obs"].to_numpy().astype("datetime64[ns]")
        paths = []
        for ts in expect_times:
            # nearest neighbor within 90 minutes, else None
            ts_np = np.datetime64(ts)
            if g_times.size == 0:
                paths.append(None)
                continue
            idx = np.searchsorted(g_times, ts_np, side="left")
            idx = np.clip(idx, 0, len(g_times)-1)
            nearest = g.iloc[int(idx)]
            dt = abs(pd.Timestamp(nearest["date_obs"]) - ts)
            if dt <= pd.Timedelta(minutes=90):
                paths.append(str(nearest["frame_path"]))
            else:
                paths.append(None)
        return expect_times, paths

    def _load_bz(self, frame_path_rel: str) -> Optional[np.ndarray]:
        try:
            p = self.root / frame_path_rel
            npz = np.load(p, allow_pickle=False)
            # Expect a processed local Bz (post sign convention), else fallback
            if "Bz" in npz:
                bz = npz["Bz"].astype(np.float32)
            elif "Br" in npz:
                bz = npz["Br"].astype(np.float32)  # fallback; sign handled in preprocessing
            else:
                return None
            # Resize on the fly if needed (nearest/bilinear simple)
            if bz.shape != (self.target_px, self.target_px):
                # simple center crop/pad to target size
                H, W = bz.shape
                out = np.zeros((self.target_px, self.target_px), np.float32)
                y0 = max(0, (self.target_px - H)//2); x0 = max(0, (self.target_px - W)//2)
                y1 = y0 + min(H, self.target_px); x1 = x0 + min(W, self.target_px)
                sy0 = max(0, (H - self.target_px)//2); sx0 = max(0, (W - self.target_px)//2)
                sy1 = sy0 + (y1 - y0); sx1 = sx0 + (x1 - x0)
                out[y0:y1, x0:x1] = bz[sy0:sy1, sx0:sx1]
                bz = out
            
            # Replace NaN/Inf with zeros (better than letting them propagate)
            if not np.isfinite(bz).all():
                nan_mask = ~np.isfinite(bz)
                bz[nan_mask] = 0.0
            
            # Normalization: robust clipping + optional z-score
            # SHARP Bz typical range: [-3000, +3000] Gauss
            # We clip outliers then normalize to reasonable range for neural network
            valid_values = bz[np.abs(bz) > 1e-10]  # Exclude near-zero values
            if len(valid_values) > 10:
                p01, p99 = np.percentile(valid_values, [1, 99])
                bz = np.clip(bz, p01, p99)
            else:
                # Not enough valid data - just clip to reasonable range
                bz = np.clip(bz, -3000.0, 3000.0)
            
            # Scale to approximately [-1, 1] range for better gradient flow
            # Typical AR has std ~ 500-1000 G after clipping
            bz_std = np.std(bz)
            if bz_std < 1e-6:
                # Constant or near-constant field - use fallback scaling
                bz_std = max(np.abs(bz).max(), 1000.0)
            bz = bz / (3.0 * bz_std)  # 3-sigma scaling
            bz = np.clip(bz, -3.0, 3.0)  # Final safety clip
            
            # Final check for NaN/Inf after normalization
            if not np.isfinite(bz).all():
                bz = np.nan_to_num(bz, nan=0.0, posinf=3.0, neginf=-3.0)
            
            return bz
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        harp = int(row["harpnum"])
        t0 = pd.to_datetime(row["t0"], utc=True)
        times, paths = self._frames_for_window(harp, t0)
        T = len(times)

        device = torch.device("cpu")  # tensors are moved in train loop
        coords = _coords_grid(T, self.P_per_t, device)  # [T,P,3]

        # Build observed mask & sample gt Bz & load full frames
        observed = np.zeros((T,), np.bool_)
        gt_bz = torch.zeros(T, self.P_per_t, 1, device=device)
        frames = torch.zeros(T, self.target_px, self.target_px, device=device)  # NEW: full frames
        last_obs_bz = None
        
        for ti, p in enumerate(paths):
            if p is None:
                continue
            bz = self._load_bz(p)
            if bz is None:
                continue
            observed[ti] = True
            
            # Sample P points at this time index for gt_bz
            xy = coords[ti, :, 0:2]  # [P,2]
            gt = _bilinear_sample_bz(bz, xy)  # [P,1]
            
            # Safety check: replace any NaN/Inf in sampled values
            if not torch.isfinite(gt).all():
                gt = torch.nan_to_num(gt, nan=0.0, posinf=3.0, neginf=-3.0)
            
            gt_bz[ti] = gt
            
            # Store full frame for hybrid model
            frame_tensor = torch.from_numpy(bz).to(device)
            # Safety check for frames too
            if not torch.isfinite(frame_tensor).all():
                frame_tensor = torch.nan_to_num(frame_tensor, nan=0.0, posinf=3.0, neginf=-3.0)
            frames[ti] = frame_tensor
            last_obs_bz = bz

        observed_mask = torch.from_numpy(observed)
        
        # Skip samples with NO observed frames (would cause NaN in training)
        if not observed.any():
            # Return None to signal DataLoader to skip this sample
            # Actually, we can't return None from __getitem__, so we need to handle this differently
            # For now, synthesize a dummy observed frame to avoid NaN
            observed[0] = True  # Mark first frame as observed
            gt_bz[0] = torch.zeros_like(gt_bz[0])  # Fill with zeros
            frames[0] = torch.zeros_like(frames[0])  # Fill with zeros
            observed_mask = torch.from_numpy(observed)

        # Labels for horizons
        labels = []
        for H in self.horizons:
            col = f"y_geq_M_{H}h"
            labels.append(float(bool(row[col])) if col in row else 0.0)
        labels = torch.tensor(labels).float()

        # PIL mask from last observed frame (optional bias)
        if last_obs_bz is not None:
            pil_mask = pil_mask_from_bz(last_obs_bz, top_percent=self.pil_top_pct)
        else:
            # Return zero mask if no observed frames (shouldn't happen but be safe)
            pil_mask = np.zeros((self.target_px, self.target_px), dtype=np.uint8)

        return {
            "coords": coords,                  # [T,P,3]
            "gt_bz": gt_bz,                    # [T,P,1]
            "frames": frames,                  # [T,H,W] NEW!
            "observed_mask": observed_mask,    # [T]
            "labels": labels,                  # [n_horizons]
            "pil_mask": pil_mask,              # np.ndarray[H,W]
        }

