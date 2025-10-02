#!/usr/bin/env python3
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import click
import cv2
import math
import shutil
import time

# ------------------------- helpers -------------------------

def parse_name_or_header(npz_path: Path) -> Tuple[Optional[int], Optional[str]]:
    """
    Extract (harpnum, date_obs) from NPZ content if present,
    otherwise parse from filename: H{harpnum}_{DATE}.npz
    """
    harpnum = None
    date_obs = None
    try:
        with np.load(npz_path, mmap_mode="r") as z:
            harpnum = int(z["harpnum"]) if "harpnum" in z else None
            date_obs = str(z["date_obs"]) if "date_obs" in z else None
    except Exception:
        pass

    if harpnum is None or not date_obs:
        stem = npz_path.stem  # e.g., H12345_2012-01-01T00-00-00
        if stem.startswith("H"):
            try:
                left, right = stem[1:].split("_", 1)
                harpnum = int(left)
                date_obs = right.replace("_", "T")
            except Exception:
                pass
    return harpnum, date_obs

def sanitize_timestamp(ts: str) -> str:
    return ts.replace(" ", "T").replace(":", "-").replace("/", "-")

def resize_with_nan(img: np.ndarray, size: int) -> np.ndarray:
    """NaN-preserving resize."""
    if img.ndim != 2:
        raise ValueError("Expected 2D array for resize.")
    mask = np.isfinite(img).astype(np.uint8)
    if mask.sum() == 0:
        return np.full((size, size), np.nan, dtype=np.float32)
    med = float(np.nanmedian(img))
    tmp = img.copy()
    tmp[~np.isfinite(tmp)] = med
    resized = cv2.resize(tmp, (size, size), interpolation=cv2.INTER_CUBIC)
    mask_r = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    out = resized.astype(np.float32)
    out[mask_r == 0] = np.nan
    return out

def robust_norm_nan(a: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """Robust normalize allowing NaNs; returns (z, med, iqr, nan_fraction)."""
    nan_frac = float(np.isnan(a).mean())
    med = float(np.nanmedian(a))
    q1, q3 = np.nanpercentile(a, [25, 75])
    iqr = float(max(q3 - q1, 1e-6))
    denom = iqr / 1.349  # robust sigma
    z = (a - med) / denom
    z = np.clip(z, -5, 5).astype(np.float32)
    return z, med, iqr, nan_frac

def shape_of(z, key: str) -> Optional[Tuple[int, int]]:
    try:
        arr = z[key]
        if arr.ndim == 2:
            return tuple(arr.shape)
    except Exception:
        pass
    return None

def nan_fractions_from_npz(z) -> Optional[float]:
    """Prefer stored nan fractions; fallback None. Returns average NaN fraction."""
    if all(k in z for k in ("Bx_nan","By_nan","Bz_nan")):
        return float((z["Bx_nan"] + z["By_nan"] + z["Bz_nan"]) / 3.0)
    return None

def file_quality_tuple(p: Path, target_px: int) -> Tuple:
    """
    Higher tuple is better.
    (has_stats, has_target_shape, -nan_sum, mtime, size)
    """
    has_stats = False
    has_target_shape = False
    nan_sum = math.inf
    mtime = p.stat().st_mtime
    size = p.stat().st_size

    try:
        with np.load(p, mmap_mode="r") as z:
            has_stats = all(k in z for k in (
                "Bx_med","Bx_iqr","By_med","By_iqr","Bz_med","Bz_iqr"
            ))
            sBx = shape_of(z, "Bx")
            sBy = shape_of(z, "By")
            sBz = shape_of(z, "Bz")
            has_target_shape = (sBx == (target_px, target_px)
                                and sBy == (target_px, target_px)
                                and sBz == (target_px, target_px))
            ns = nan_fractions_from_npz(z)
            if ns is not None:
                nan_sum = ns
    except Exception:
        pass

    # invert nan_sum for sorting: we want lower nan_sum preferred
    inv_nan = -nan_sum if np.isfinite(nan_sum) else -1e9
    return (int(has_stats), int(has_target_shape), inv_nan, mtime, size)

def group_npz(npz_paths: List[Path]) -> Dict[Tuple[int,str], List[Path]]:
    groups: Dict[Tuple[int,str], List[Path]] = {}
    for p in npz_paths:
        harp, date_obs = parse_name_or_header(p)
        if harp is None or not date_obs:
            warnings.warn(f"skip (cannot parse key) {p}")
            continue
        key = (harp, date_obs)
        groups.setdefault(key, []).append(p)
    return groups

def write_clean_npz(out_npz: Path,
                    Bx: np.ndarray, By: np.ndarray, Bz: np.ndarray,
                    stats: Dict, meta: Dict):
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        Bx=Bx.astype(np.float16),
        By=By.astype(np.float16),
        Bz=Bz.astype(np.float16),
        Bx_med=np.float32(stats["Bx_med"]), Bx_iqr=np.float32(stats["Bx_iqr"]), Bx_nan=np.float32(stats["Bx_nan"]),
        By_med=np.float32(stats["By_med"]), By_iqr=np.float32(stats["By_iqr"]), By_nan=np.float32(stats["By_nan"]),
        Bz_med=np.float32(stats["Bz_med"]), Bz_iqr=np.float32(stats["Bz_iqr"]), Bz_nan=np.float32(stats["Bz_nan"]),
        pxscale_Mm=np.float32(meta.get("pxscale_Mm", np.nan)),
        cmd_deg=np.float32(meta.get("cmd_deg", np.nan)),
        harpnum=np.int32(meta["harpnum"]),
        date_obs=str(meta["date_obs"]),
        is_masked_limb=np.int8(meta.get("is_masked_limb", 0)),
    )

def find_files(root: Path, pattern: str) -> List[Path]:
    return [p for p in root.rglob(pattern) if p.is_file()]

# ------------------------- main -------------------------

@click.command()
@click.option("--npz-root", required=True, help="Root with raw NPZs (e.g., S:\\SHARP_1h)")
@click.option("--out-root", required=True, help="Project root on D: (outputs go under this).")
@click.option("--frames-subdir", default="data/interim/frames", show_default=True,
              help="Relative subdir under out-root for NPZ frames.")
@click.option("--meta-name", default="frames_meta.parquet", show_default=True,
              help="Filename for meta parquet (written next to frames dir).")
@click.option("--target-px", default=256, show_default=True, type=int,
              help="Target spatial size for Bx/By/Bz (square).")
@click.option("--delete-old/--keep-old", default=True, show_default=True,
              help="Delete source NPZs (and duplicates) in npz-root after successful write.")
@click.option("--purge-fits/--keep-fits", default=True, show_default=True,
              help="Delete any stray FITS under npz-root (if found).")
def main(npz_root: str, out_root: str, frames_subdir: str, meta_name: str,
         target_px: int, delete_old: bool, purge_fits: bool):
    npz_root = Path(npz_root)
    out_root = Path(out_root)
    frames_dir = out_root / frames_subdir
    frames_dir.mkdir(parents=True, exist_ok=True)
    meta_out = frames_dir.parent / meta_name

    # 1) Collect NPZs from S:
    print(f"Scanning for NPZ files under: {npz_root}")
    npz_paths = find_files(npz_root, "*.npz")
    if not npz_paths:
        raise SystemExit(f"No NPZ files found under: {npz_root}")
    print(f"Found {len(npz_paths):,} NPZ files.")

    # 2) Group and dedupe
    print("Grouping by (harpnum, date_obs)...")
    groups = group_npz(npz_paths)
    duplicates = len(npz_paths) - len(groups)
    print(f"→ {len(groups):,} unique frames ({duplicates:,} duplicates to remove).")

    meta_rows: List[Dict] = []
    to_delete: List[Path] = []
    processed = 0
    skipped = 0

    pbar = tqdm(groups.items(), total=len(groups), desc="Processing", unit="frame")
    for key, paths in pbar:
        harpnum, date_obs = key

        # choose best candidate
        qualities = [(file_quality_tuple(p, target_px), p) for p in paths]
        qualities.sort(reverse=True)
        best_path = qualities[0][1]
        dup_paths = [p for (_, p) in qualities[1:]]

        # delete duplicates immediately if requested
        if delete_old and dup_paths:
            for dp in dup_paths:
                try:
                    os.remove(dp)
                except Exception as e:
                    print(f"warn: could not delete duplicate {dp}: {e}")

        # 3) Load best NPZ and (re)format
        try:
            with np.load(best_path, allow_pickle=False) as z:
                # read arrays (may be float16/32)
                Bx = z["Bx"].astype(np.float32)
                By = z["By"].astype(np.float32)
                Bz = z["Bz"].astype(np.float32)

                # ensure shape = (target_px, target_px)
                if Bx.shape != (target_px, target_px):
                    Bx = resize_with_nan(Bx, target_px)
                if By.shape != (target_px, target_px):
                    By = resize_with_nan(By, target_px)
                if Bz.shape != (target_px, target_px):
                    Bz = resize_with_nan(Bz, target_px)

                # (re)normalize robustly to standardize scale/clip
                Bx_n, Bx_med, Bx_iqr, Bx_nan = robust_norm_nan(Bx)
                By_n, By_med, By_iqr, By_nan = robust_norm_nan(By)
                Bz_n, Bz_med, Bz_iqr, Bz_nan = robust_norm_nan(Bz)

                # meta
                pxscale_Mm = float(z["pxscale_Mm"]) if "pxscale_Mm" in z else np.nan
                cmd_deg = float(z["cmd_deg"]) if "cmd_deg" in z else np.nan
                is_masked_limb = int(z["is_masked_limb"]) if "is_masked_limb" in z else 0
        except Exception as e:
            warnings.warn(f"skip (bad NPZ): {best_path} ({e})")
            skipped += 1
            pbar.set_postfix({"✓": processed, "✗": skipped})
            continue

        # sanitize filename
        out_name = f"H{int(harpnum)}_{sanitize_timestamp(str(date_obs))}.npz"
        out_npz = frames_dir / out_name

        # write clean NPZ
        stats = dict(Bx_med=Bx_med, Bx_iqr=Bx_iqr, Bx_nan=Bx_nan,
                     By_med=By_med, By_iqr=By_iqr, By_nan=By_nan,
                     Bz_med=Bz_med, Bz_iqr=Bz_iqr, Bz_nan=Bz_nan)
        meta = dict(pxscale_Mm=pxscale_Mm, cmd_deg=cmd_deg, harpnum=int(harpnum),
                    date_obs=str(date_obs), is_masked_limb=is_masked_limb)

        write_clean_npz(out_npz, Bx_n, By_n, Bz_n, stats, meta)

        # verify write, then mark best_path to delete
        if delete_old:
            to_delete.append(best_path)

        # meta row
        meta_rows.append({
            "npz": str(out_npz),
            "harpnum": int(harpnum),
            "date_obs": str(date_obs),
            "cmd_deg": float(cmd_deg) if not np.isnan(cmd_deg) else np.nan,
            "pxscale_Mm": float(pxscale_Mm) if not np.isnan(pxscale_Mm) else np.nan,
            "is_masked_limb": bool(is_masked_limb),
            "Bx_nan": Bx_nan, "By_nan": By_nan, "Bz_nan": Bz_nan,
        })
        
        processed += 1
        pbar.set_postfix({"✓": processed, "✗": skipped})

    # 4) Write meta parquet
    if not meta_rows:
        raise SystemExit("No frames were written.")
    meta_df = pd.DataFrame(meta_rows).sort_values(["harpnum", "date_obs"])
    meta_df.to_parquet(meta_out, index=False)
    print(f"Wrote frames: {frames_dir}")
    print(f"Wrote meta:   {meta_out} (n={len(meta_df)})")

    # 5) Delete chosen source NPZs after success
    if delete_old and to_delete:
        print(f"\nDeleting {len(to_delete):,} source NPZ files from {npz_root}...")
        deleted = 0
        for p in tqdm(to_delete, desc="Deleting NPZs", unit="file"):
            try:
                os.remove(p)
                deleted += 1
            except Exception as e:
                print(f"warn: could not delete source {p}: {e}")
        print(f"✓ Deleted {deleted:,} source NPZ files.")

    # 6) Purge any stray FITS under S:\SHARP_1h (just in case)
    if purge_fits:
        print(f"\nScanning for FITS files to purge under {npz_root}...")
        fits_paths = find_files(npz_root, "*.fits")
        if fits_paths:
            print(f"Found {len(fits_paths):,} FITS files to delete...")
            purged = 0
            for fp in tqdm(fits_paths, desc="Purging FITS", unit="file"):
                try:
                    os.remove(fp)
                    purged += 1
                except Exception as e:
                    print(f"warn: could not delete FITS {fp}: {e}")
            print(f"✓ Purged {purged:,} FITS files.")
        else:
            print("No FITS files found.")

    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"✓ Processed:  {processed:,} frames")
    print(f"✗ Skipped:    {skipped:,} frames")
    print(f"📂 Output:    {frames_dir}")
    print(f"📊 Metadata:  {meta_out} ({len(meta_df):,} records)")
    print("="*60)

if __name__ == "__main__":
    # Show normal warnings but keep going on non-fatal issues.
    warnings.filterwarnings("default", category=UserWarning)
    main()
