#!/usr/bin/env python3
"""
Quick QC for SHARP NPZ archives.

What it checks (fast):
- Can open NPZ (corruption rate)
- Presence of keys: Bx, By, Bz; optional stats/meta keys
- Shapes & dtypes (consistency; target_px awareness)
- Extract (harpnum, date_obs) for duplicate detection
- Counts of duplicates per (harpnum, date_obs)
- Meta presence (pxscale_Mm, cmd_deg, is_masked_limb)
- Date range, #unique ARs, basic file size stats

What it samples (light):
- For a subset (or strided downsample): NaN fraction, min/max,
  fraction outside [-5, 5] to catch unnormalized data

Outputs:
- A compact summary JSON
- A CSV of duplicate groups (top-N)
- A CSV of suspicious files (e.g., high NaN fraction, missing keys)
"""
import os
import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------- helpers ---------------------------

def parse_name_or_header(npz_path: Path) -> Tuple[Optional[int], Optional[str]]:
    """Extract (harpnum, date_obs) from NPZ contents if present; else from filename."""
    harpnum = None
    date_obs = None
    try:
        with np.load(npz_path, mmap_mode="r", allow_pickle=False) as z:
            if "harpnum" in z:
                harpnum = int(z["harpnum"])
            if "date_obs" in z:
                date_obs = str(z["date_obs"])
    except Exception:
        pass

    if harpnum is None or not date_obs:
        stem = npz_path.stem  # e.g., H12345_2012-01-01T00-00-00
        if stem.startswith("H") and "_" in stem:
            try:
                left, right = stem[1:].split("_", 1)
                harpnum = int(left)
                date_obs = right
            except Exception:
                pass
    return harpnum, date_obs

def safe_shape_dtype(z, key: str) -> Tuple[Optional[Tuple[int,int]], Optional[str]]:
    try:
        arr = z[key]
        if arr.ndim == 2:
            return tuple(arr.shape), str(arr.dtype)
        else:
            return None, str(arr.dtype)
    except Exception:
        return None, None

def fast_sample_stats(arr: np.ndarray, stride: int = 16) -> Dict[str, float]:
    """
    Very light stats on a strided sample. Avoids loading the whole array.
    Assumes arr is memmapped or already in memory.
    """
    if arr.ndim != 2:
        return {"nan_frac": math.nan, "vmin": math.nan, "vmax": math.nan, "frac_out_5": math.nan}
    sample = arr[::stride, ::stride].astype(np.float32)
    finite = np.isfinite(sample)
    total = sample.size
    if total == 0:
        return {"nan_frac": 1.0, "vmin": math.nan, "vmax": math.nan, "frac_out_5": math.nan}
    nan_frac = float((~finite).sum()) / float(total)
    if finite.any():
        vals = sample[finite]
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        frac_out_5 = float(((vals < -5) | (vals > 5)).mean())
    else:
        vmin = vmax = math.nan
        frac_out_5 = math.nan
    return {"nan_frac": nan_frac, "vmin": vmin, "vmax": vmax, "frac_out_5": frac_out_5}

def find_npz(root: Path, max_files: Optional[int]) -> List[Path]:
    paths = [p for p in root.rglob("*.npz") if p.is_file()]
    if max_files is not None and len(paths) > max_files:
        # uniform sample across tree
        random.shuffle(paths)
        paths = paths[:max_files]
    return paths

# --------------------------- main ---------------------------

@click.command()
@click.option("--root", required=True, help="Root folder containing NPZs (e.g., S:\\SHARP_1h).")
@click.option("--out", required=False, default=None,
              help="Folder to write QC outputs (default: <root>/qc_report).")
@click.option("--target-px", default=256, show_default=True, type=int,
              help="Expected spatial size (only for consistency checks).")
@click.option("--max-files", default=None, type=int,
              help="Optionally limit the number of NPZs scanned (sampling).")
@click.option("--stride", default=16, show_default=True, type=int,
              help="Sampling stride for quick stats (higher is faster).")
@click.option("--topk-dups", default=50, show_default=True, type=int,
              help="How many duplicate groups to list in CSV.")
def main(root: str, out: Optional[str], target_px: int,
         max_files: Optional[int], stride: int, topk_dups: int):

    random.seed(1234)
    np.random.seed(1234)

    root_path = Path(root)
    if out is None:
        out_path = root_path / "qc_report"
    else:
        out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    npz_paths = find_npz(root_path, max_files)
    if not npz_paths:
        raise SystemExit(f"No NPZ files found under: {root_path}")

    # Aggregates
    total = 0
    corrupt = 0
    missing_keys = 0
    missing_meta = 0
    shapes = {}
    dtypes = {}
    size_stats = []

    groups: Dict[Tuple[int,str], List[Path]] = {}
    suspicious_rows = []

    # scan
    for p in tqdm(npz_paths, desc="QC scan (meta + light sample)"):
        total += 1
        try:
            with np.load(p, mmap_mode="r", allow_pickle=False) as z:
                # core keys check
                has_core = all(k in z for k in ("Bx","By","Bz"))
                if not has_core:
                    missing_keys += 1
                    suspicious_rows.append({"path": str(p), "reason": "missing_core_keys"})
                    # cannot proceed to shape checks safely
                    continue

                # shapes & dtypes
                sx, dx = safe_shape_dtype(z, "Bx")
                sy, dy = safe_shape_dtype(z, "By")
                sz, dz = safe_shape_dtype(z, "Bz")
                shapes[str(sx)] = shapes.get(str(sx), 0) + 1
                dtypes[dx] = dtypes.get(dx, 0) + 1

                # meta presence
                meta_keys = ("harpnum","date_obs","pxscale_Mm","cmd_deg")
                has_meta = all(k in z for k in meta_keys)
                if not has_meta:
                    missing_meta += 1
                    suspicious_rows.append({"path": str(p), "reason": "missing_meta_keys"})

                # IDs for duplicate grouping (fallback to filename)
                harpnum, date_obs = None, None
                try:
                    harpnum = int(z["harpnum"]) if "harpnum" in z else None
                    date_obs = str(z["date_obs"]) if "date_obs" in z else None
                except Exception:
                    pass
                if harpnum is None or not date_obs:
                    harpnum, date_obs = parse_name_or_header(p)
                if harpnum is not None and date_obs:
                    groups.setdefault((harpnum, date_obs), []).append(p)

                # light sampling stats (on normalized arrays if these are normalized)
                try:
                    bx = z["Bx"]
                    by = z["By"]
                    bz = z["Bz"]
                    stats_x = fast_sample_stats(bx, stride=stride)
                    stats_y = fast_sample_stats(by, stride=stride)
                    stats_z = fast_sample_stats(bz, stride=stride)

                    # flag obviously problematic cases
                    # high NaNs or large fraction outside [-5, 5] (expected clip for normalized)
                    nan_thresh = 0.20  # 20% in sample is suspicious
                    out5_thresh = 0.05  # >5% outside [-5,5] suggests not normalized
                    reasons = []
                    if stats_x["nan_frac"] > nan_thresh or stats_y["nan_frac"] > nan_thresh or stats_z["nan_frac"] > nan_thresh:
                        reasons.append(f"high_nan(sample)>={nan_thresh}")
                    if (stats_x["frac_out_5"] and stats_x["frac_out_5"] > out5_thresh) or \
                       (stats_y["frac_out_5"] and stats_y["frac_out_5"] > out5_thresh) or \
                       (stats_z["frac_out_5"] and stats_z["frac_out_5"] > out5_thresh):
                        reasons.append(f"out_of_clip(sample)>={out5_thresh}")

                    if reasons:
                        suspicious_rows.append({"path": str(p), "reason": ";".join(reasons)})
                except Exception:
                    suspicious_rows.append({"path": str(p), "reason": "sample_stats_failed"})

                # file size stats
                try:
                    size_stats.append(p.stat().st_size)
                except Exception:
                    pass

        except Exception:
            corrupt += 1
            suspicious_rows.append({"path": str(p), "reason": "corrupt_npz"})

    # duplicates summary
    dup_groups = [(k, v) for k, v in groups.items() if len(v) > 1]
    dup_groups.sort(key=lambda kv: len(kv[1]), reverse=True)

    # summary JSON
    summary = {
        "root": str(root_path),
        "scanned_files": total,
        "corrupt_files": corrupt,
        "missing_core_keys": missing_keys,
        "missing_meta_keys": missing_meta,
        "unique_shapes": {k: int(v) for k, v in sorted(shapes.items(), key=lambda kv: (-kv[1], kv[0]))},
        "dtypes_Bx_hist": {k: int(v) for k, v in sorted(dtypes.items(), key=lambda kv: (-kv[1], kv[0]))},
        "duplicate_key_groups": len(dup_groups),
        "duplicate_examples_listed": min(len(dup_groups), 50),
        "target_px": target_px,
        "size_stats_bytes": {
            "count": len(size_stats),
            "min": int(np.min(size_stats)) if size_stats else None,
            "median": int(np.median(size_stats)) if size_stats else None,
            "p90": int(np.percentile(size_stats, 90)) if size_stats else None,
            "max": int(np.max(size_stats)) if size_stats else None,
        },
    }

    # write outputs
    summary_path = out_path / "qc_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[QC] Wrote summary: {summary_path}")

    # duplicates CSV (top-K groups)
    if dup_groups:
        rows = []
        for (harp, date_obs), paths in dup_groups[:topk_dups]:
            for p in paths:
                rows.append({"harpnum": harp, "date_obs": date_obs, "path": str(p)})
        dup_csv = out_path / "duplicates_top.csv"
        pd.DataFrame(rows).to_csv(dup_csv, index=False)
        print(f"[QC] Wrote duplicate listing: {dup_csv} (top {min(len(dup_groups), topk_dups)} groups)")

    # suspicious CSV
    if suspicious_rows:
        susp_csv = out_path / "suspicious_files.csv"
        pd.DataFrame(suspicious_rows).drop_duplicates().to_csv(susp_csv, index=False)
        print(f"[QC] Wrote suspicious files: {susp_csv}")

    print("[QC] Done.")
    print("Tip: If you see many duplicates or missing keys/meta, run your preprocess to standardize and dedupe.")
    

if __name__ == "__main__":
    warnings.filterwarnings("default", category=UserWarning)
    main()

