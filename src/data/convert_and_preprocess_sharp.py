#!/usr/bin/env python3
"""
Self-contained SHARP CEA converter for Windows (robust atomic saves + failure CSVs).

Converts component NPZs (Br, Bt, Bp) → local Cartesian (Bx, By, Bz) with:
    Bx = -Bp  (east-positive)
    By = -Bt  (north-positive)
    Bz =  Br  (outward-positive)
Resizes to TARGET_PX, robust-normalizes per channel (median/IQR, ±CLIP_SIGMA),
writes one compressed NPZ per frame with per-channel stats & minimal metadata,
and maintains a Parquet (CSV fallback) metadata table. Idempotent & safe.

New:
- Validates Br/Bt/Bp NPZs before loading (ZIP + CRC)
- Logs all bad components to <OUT_ROOT>\bad_components.csv
- Writes grouped <OUT_ROOT>\failed_timestamps.csv listing timestamps to re-fetch

Requires: numpy, pandas, tqdm, opencv-python (cv2)
"""

import os
import time
import uuid
import zipfile
import warnings
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ===================== CONFIG =====================

CONFIG = dict(
    NPZ_ROOT=r"S:\SHARP_1h",                # <-- input components on S:\
    OUT_ROOT=r"S:\flare_forecasting",       # <-- outputs written directly to S:\
    FRAMES_SUBDIR="frames",                 # e.g., S:\flare_forecasting\frames
    META_NAME="frames_meta.parquet",        # metadata table at S:\flare_forecasting\frames_meta.parquet
    TARGET_PX=256,                          # output size (square)
    DELETE_OLD=False,                       # delete Br/Bt/Bp NPZs after success
    SIGN_MODE="plan",                       # "plan" (Bx=-Bp, By=-Bt, Bz=Br) or "raw" (Bx=+Bp, By=+Bt, Bz=Br)
    CLIP_SIGMA=5.0,                         # clamp normalized values to ±CLIP_SIGMA
)

# ==================================================


# -------------------- I/O helpers --------------------

def atomic_save_npz(out_path: Path, **arrays) -> None:
    """
    Robust atomic save for Windows:
    - write a unique .partial file next to the final file
    - flush + fsync to ensure bytes are on disk
    - os.replace() (atomic on same volume)
    - retry with backoff on transient sharing violations
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_tries = 10
    for attempt in range(max_tries):
        tmp_name = f"{out_path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
        tmp_path = out_path.parent / tmp_name

        try:
            with open(tmp_path, "wb") as f:
                np.savez_compressed(f, **arrays)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp_path, out_path)
            return

        except (FileNotFoundError, PermissionError, OSError) as e:
            # winerror 32/33 = sharing violations
            if isinstance(e, OSError) and getattr(e, "winerror", None) not in (32, 33):
                raise
            time.sleep(0.15 * (attempt + 1))
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    raise RuntimeError(f"atomic_save_npz: failed after {max_tries} attempts: {out_path}")


def cleanup_orphan_tmps(folder: Path) -> int:
    """Remove orphan '.*.tmp' and '.*.partial' files left from previous crashes."""
    n = 0
    for pat in ("*.npz.*.tmp", "*.partial"):
        for p in folder.glob(pat):
            try:
                p.unlink()
                n += 1
            except Exception:
                pass
    return n


def find_files(root: Path, pattern: str) -> List[Path]:
    """Recursive glob, case-friendly for NPZ on some filesystems."""
    paths = list(root.rglob(pattern))
    if not paths and pattern == "*.npz":
        paths = list(root.rglob("*.NPZ"))
    return sorted(paths)


# -------------------- Parsing / grouping --------------------

def parse_sharp_filename(p: Path) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Parse filenames like: hmi.sharp_cea_720s.7240.20180226_010000_TAI.Br.npz
    Returns: (harpnum, date_obs, component) where date_obs = 'YYYY-MM-DDTHH:MM:SS'
    """
    stem = p.stem  # drops .npz
    parts = stem.split(".")
    harpnum = None
    date_obs = None
    component = parts[-1] if len(parts) >= 2 else None

    # HARP number
    for tok in parts:
        if tok.isdigit():
            try:
                val = int(tok)
                if 1 <= val <= 999999:
                    harpnum = val
                    break
            except Exception:
                pass

    # Timestamp YYYYMMDD_HHMMSS(_TAI)
    for tok in parts:
        if len(tok) >= 15 and "_" in tok:
            cand = tok.replace("_TAI", "")
            if "_" in cand:
                d, t = cand.split("_", 1)
                if len(d) == 8 and len(t) >= 6 and d.isdigit() and t[:6].isdigit():
                    date_obs = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}"
                    break
    return harpnum, date_obs, component


def group_component_files(npz_paths: List[Path]) -> Dict[Tuple[int, str], Dict[str, Path]]:
    """
    Group Br/Bt/Bp by (harpnum, date_obs).
    Returns: {(harpnum, date_obs): {"Br": path, "Bt": path, "Bp": path}}
    """
    groups: Dict[Tuple[int, str], Dict[str, Path]] = {}
    for p in npz_paths:
        harpnum, date_obs, component = parse_sharp_filename(p)
        if harpnum is None or date_obs is None or component is None:
            continue
        key = (harpnum, date_obs)
        groups.setdefault(key, {})[component] = p
    return groups


# -------------------- Loading / transforms --------------------

def is_valid_npz_file(path: Path) -> bool:
    """Check that file is a valid NPZ (ZIP) and passes CRC."""
    try:
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except Exception:
        return False


def load_component_npz(path: Path) -> Tuple[np.ndarray, dict]:
    """
    Robustly load a component NPZ saved as:
      - z['data'] : ndarray (numeric or sometimes object-wrapped)
      - optional z['meta'] : 0-d object array holding a dict
    Returns (array_float32, meta_dict).
    """
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        arr = None
        if "data" in keys:
            arr = z["data"]
        else:
            for k in ("Br", "Bt", "Bp", "B", "arr_0", "array"):
                if k in keys:
                    arr = z[k]
                    break
        if arr is None:
            raise ValueError(f"{path.name}: no numeric data array found")

        if getattr(arr, "dtype", None) == object:
            try:
                arr = np.array(arr.item(), dtype=np.float32, copy=False)
            except Exception:
                arr = np.asarray(arr, dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)

        meta = {}
        if "meta" in keys:
            try:
                m = z["meta"]
                meta = dict(m.item()) if getattr(m, "dtype", None) == object else {}
            except Exception:
                meta = {}
        return arr, meta


def cea_to_local_components(Br: np.ndarray, Bt: np.ndarray, Bp: np.ndarray, signs: str):
    """
    Map CEA spherical (Br,Bt,Bp) to local Cartesian (Bx,By,Bz).
    signs='plan'  -> Bx=-Bp, By=-Bt, Bz=Br  (east/north/outward positive)
    signs='raw'   -> Bx=+Bp, By=+Bt, Bz=Br  (west/south/outward positive)
    """
    if signs == "plan":
        Bx_raw = -Bp
        By_raw = -Bt
        Bz_raw = Br
    elif signs == "raw":
        Bx_raw = Bp
        By_raw = Bt
        Bz_raw = Br
    else:
        raise ValueError(f"Unknown signs option: {signs}")
    return Bx_raw, By_raw, Bz_raw


def resize_with_nan(arr: np.ndarray, target_px: int) -> np.ndarray:
    """Resize 2D arrays while preserving NaNs (mask back after resize)."""
    if arr.shape == (target_px, target_px):
        return arr
    mask = ~np.isfinite(arr)
    filled = np.where(mask, 0.0, arr)
    resized = cv2.resize(filled, (target_px, target_px), interpolation=cv2.INTER_AREA)
    if mask.any():
        m = cv2.resize(mask.astype(np.float32), (target_px, target_px), interpolation=cv2.INTER_NEAREST)
        resized[m > 0.5] = np.nan
    return resized


def robust_norm_nan(arr: np.ndarray, clip_sigma: float = 5.0):
    """
    Robust normalization preserving NaNs.
    Returns: (normalized_array, median, iqr, nan_fraction)
    """
    finite = np.isfinite(arr)
    nan_frac = float((~finite).sum()) / float(arr.size)

    if not finite.any():
        return np.full_like(arr, np.nan, dtype=np.float32), np.nan, np.nan, nan_frac

    vals = arr[finite]
    med = float(np.median(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = float(q3 - q1) + 1e-8

    normed = (arr - med) / iqr
    np.clip(normed, -clip_sigma, clip_sigma, out=normed)
    return normed.astype(np.float32), med, iqr, nan_frac


# -------------------- Metadata helpers --------------------

def read_existing_meta(meta_out: Path) -> Optional[pd.DataFrame]:
    """Read existing metadata (Parquet preferred, CSV fallback)."""
    if meta_out.exists():
        try:
            return pd.read_parquet(meta_out)
        except Exception:
            pass
    csv_path = meta_out.with_suffix(".csv")
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return None


def write_meta(meta_df: pd.DataFrame, meta_out: Path) -> None:
    """Write metadata (Parquet with CSV fallback)."""
    try:
        meta_df.to_parquet(meta_out, index=False)
        print(f"✓ Saved metadata (Parquet): {meta_out}")
    except Exception as e:
        csv_fallback = meta_out.with_suffix(".csv")
        meta_df.to_csv(csv_fallback, index=False)
        print(f"⚠ Parquet failed ({e}). Wrote CSV fallback: {csv_fallback}")


# -------------------- Main processing --------------------

def main():
    # Resolve paths
    npz_root = Path(CONFIG["NPZ_ROOT"]).resolve()
    out_root = Path(CONFIG["OUT_ROOT"]).resolve()
    frames_dir = (out_root / CONFIG["FRAMES_SUBDIR"]).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    meta_out = (out_root / CONFIG["META_NAME"]).resolve()
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    bad_csv = out_root / "bad_components.csv"
    failed_ts_csv = out_root / "failed_timestamps.csv"

    target_px = int(CONFIG["TARGET_PX"])
    delete_old = bool(CONFIG["DELETE_OLD"])
    signs = str(CONFIG["SIGN_MODE"])
    clip_sigma = float(CONFIG["CLIP_SIGMA"])

    print("=" * 60)
    print("SHARP CEA → Local (Bx, By, Bz) Converter (Windows-robust)")
    print("=" * 60)
    print(f"Input (components): {npz_root}")
    print(f"Output frames dir:  {frames_dir}")
    print(f"Metadata path:      {meta_out}")
    print(f"Bad components log: {bad_csv}")
    print(f"Target size:        {target_px}×{target_px}")
    print(f"Sign convention:    {signs}  (plan: Bx=-Bp, By=-Bt, Bz=Br)")
    print("=" * 60)

    # Clean orphaned temp files
    n_tmp = cleanup_orphan_tmps(frames_dir)
    if n_tmp:
        print(f"Cleaned {n_tmp} orphan temp files in {frames_dir}")

    # Discover component NPZs
    print("Scanning for NPZ component files...")
    npz_paths = find_files(npz_root, "*.npz")
    if not npz_paths:
        raise SystemExit(f"No NPZ files found under: {npz_root}")
    print(f"Found {len(npz_paths):,} NPZ files.")

    # Group into complete triplets
    print("Grouping by (harpnum, date_obs)...")
    groups = group_component_files(npz_paths)
    print(f"→ {len(groups):,} unique timestamps (any components).")

    complete = {k: v for k, v in groups.items() if all(c in v for c in ("Br", "Bt", "Bp"))}
    print(f"→ {len(complete):,} complete sets (Br+Bt+Bp).")
    if not complete:
        raise SystemExit("No complete Br/Bt/Bp sets found.")

    # Prepare to append to existing metadata (dedup later)
    existing_meta = read_existing_meta(meta_out)
    meta_rows = []
    to_delete: List[Path] = []

    processed = 0
    already = 0
    skipped = 0

    # collect bad components and per-timestamp failures
    bad_rows: List[dict] = []

    pbar = tqdm(complete.items(), desc="Assemble frames", unit="frame")

    for (harpnum, date_obs), paths in pbar:
        # Idempotent skip if frame already exists
        out_name = f"H{harpnum}_{date_obs.replace(':', '-')}.npz"
        out_path = (frames_dir / out_name).resolve()
        if out_path.exists():
            already += 1
            pbar.set_postfix({"✓": processed, "•already": already, "↻skipped": skipped})
            continue

        # Pre-check NPZ validity for Br/Bt/Bp
        precheck_ok = True
        for comp in ("Br", "Bt", "Bp"):
            fp = paths[comp]
            if not is_valid_npz_file(fp):
                bad_rows.append({
                    "harpnum": int(harpnum),
                    "date_obs": str(date_obs),
                    "component": comp,
                    "path": str(fp),
                    "reason": "invalid_npz_zip_or_crc",
                })
                precheck_ok = False
        if not precheck_ok:
            skipped += 1
            pbar.set_postfix({"✓": processed, "•already": already, "↻skipped": skipped})
            continue

        # Load each component with its own try/except so we can log which one failed
        try:
            Br, meta_br = load_component_npz(paths["Br"])
        except Exception as e:
            bad_rows.append({
                "harpnum": int(harpnum),
                "date_obs": str(date_obs),
                "component": "Br",
                "path": str(paths["Br"]),
                "reason": f"load_error: {type(e).__name__}",
            })
            skipped += 1
            pbar.set_postfix({"✓": processed, "•already": already, "↻skipped": skipped})
            continue

        try:
            Bt, _ = load_component_npz(paths["Bt"])
        except Exception as e:
            bad_rows.append({
                "harpnum": int(harpnum),
                "date_obs": str(date_obs),
                "component": "Bt",
                "path": str(paths["Bt"]),
                "reason": f"load_error: {type(e).__name__}",
            })
            skipped += 1
            pbar.set_postfix({"✓": processed, "•already": already, "↻skipped": skipped})
            continue

        try:
            Bp, _ = load_component_npz(paths["Bp"])
        except Exception as e:
            bad_rows.append({
                "harpnum": int(harpnum),
                "date_obs": str(date_obs),
                "component": "Bp",
                "path": str(paths["Bp"]),
                "reason": f"load_error: {type(e).__name__}",
            })
            skipped += 1
            pbar.set_postfix({"✓": processed, "•already": already, "↻skipped": skipped})
            continue

        # Basic shape sanity
        if not (Br.shape == Bt.shape == Bp.shape and Br.ndim == 2):
            bad_rows.append({
                "harpnum": int(harpnum),
                "date_obs": str(date_obs),
                "component": "triplet",
                "path": f"{paths['Br']}|{paths['Bt']}|{paths['Bp']}",
                "reason": f"shape_mismatch: Br{Br.shape},Bt{Bt.shape},Bp{Bp.shape}",
            })
            skipped += 1
            pbar.set_postfix({"✓": processed, "•already": already, "↻skipped": skipped})
            continue

        try:
            # Map to local Cartesian per config
            Bx_raw, By_raw, Bz_raw = cea_to_local_components(Br, Bt, Bp, signs)

            # Resize to target (NaNs preserved)
            if Bx_raw.shape != (target_px, target_px):
                Bx_raw = resize_with_nan(Bx_raw, target_px)
                By_raw = resize_with_nan(By_raw, target_px)
                Bz_raw = resize_with_nan(Bz_raw, target_px)

            # Robust per-channel normalization
            Bx, Bx_med, Bx_iqr, Bx_nan = robust_norm_nan(Bx_raw, clip_sigma)
            By, By_med, By_iqr, By_nan = robust_norm_nan(By_raw, clip_sigma)
            Bz, Bz_med, Bz_iqr, Bz_nan = robust_norm_nan(Bz_raw, clip_sigma)

            # Compact storage
            Bx = Bx.astype(np.float16)
            By = By.astype(np.float16)
            Bz = Bz.astype(np.float16)

            # Minimal meta (pixel scale & unit if available; cmd if present)
            pxscale = meta_br.get("CDELT1", np.nan)
            try:
                pxscale = float(pxscale)
            except Exception:
                pxscale = np.nan
            pxunit = str(meta_br.get("CUNIT1", "")) if "CUNIT1" in meta_br else ""
            cmd_deg = meta_br.get("CRVAL1", np.nan)
            try:
                cmd_deg = float(cmd_deg)
            except Exception:
                cmd_deg = np.nan

            # Atomic frame save (Windows-robust)
            atomic_save_npz(
                out_path,
                Bx=Bx, By=By, Bz=Bz,
                Bx_median=float(Bx_med), By_median=float(By_med), Bz_median=float(Bz_med),
                Bx_iqr=float(Bx_iqr), By_iqr=float(By_iqr), Bz_iqr=float(Bz_iqr),
                Bx_nan=float(Bx_nan), By_nan=float(By_nan), Bz_nan=float(Bz_nan),
                harpnum=np.int32(harpnum),
                date_obs=str(date_obs),
                pxscale=float(pxscale),
                pxunit=str(pxunit),
                cmd_deg=float(cmd_deg),
                signs=str(signs),
            )

            # Relative path for portability
            rel_path = str(out_path.relative_to(out_root))
            meta_rows.append({
                "harpnum": int(harpnum),
                "date_obs": str(date_obs),
                "frame_path": rel_path,
                "Bx_median": float(Bx_med), "By_median": float(By_med), "Bz_median": float(Bz_med),
                "Bx_iqr": float(Bx_iqr), "By_iqr": float(By_iqr), "Bz_iqr": float(Bz_iqr),
                "Bx_nan": float(Bx_nan), "By_nan": float(By_nan), "Bz_nan": float(Bz_nan),
                "pxscale": float(pxscale),
                "pxunit": str(pxunit),
                "cmd_deg": float(cmd_deg),
                "signs": str(signs),
            })

            if delete_old:
                to_delete.extend([paths["Br"], paths["Bt"], paths["Bp"]])

            processed += 1
            pbar.set_postfix({"✓": processed, "•already": already, "↻skipped": skipped})

        except Exception as e:
            # catch-all for any unexpected failure in transform/save
            bad_rows.append({
                "harpnum": int(harpnum),
                "date_obs": str(date_obs),
                "component": "triplet",
                "path": f"{paths['Br']}|{paths['Bt']}|{paths['Bp']}",
                "reason": f"transform_or_save_error: {type(e).__name__}",
            })
            skipped += 1
            pbar.set_postfix({"✓": processed, "•already": already, "↻skipped": skipped})
            continue

    # Merge metadata with existing (dedup)
    if meta_rows:
        new_df = pd.DataFrame(meta_rows).sort_values(["harpnum", "date_obs"])
        existing_meta = existing_meta if existing_meta is not None else pd.DataFrame(columns=new_df.columns)
        meta_all = pd.concat([existing_meta, new_df], ignore_index=True)
        if {"harpnum", "date_obs", "frame_path"}.issubset(meta_all.columns):
            meta_all = meta_all.sort_values(["harpnum", "date_obs"]).drop_duplicates(
                subset=["harpnum", "date_obs", "frame_path"], keep="last"
            )
        else:
            meta_all = meta_all.drop_duplicates(subset=["frame_path"], keep="last")

        write_meta(meta_all, meta_out)
        print(f"Rows (new): {len(new_df):,} | Rows (total): {len(meta_all):,}")
    else:
        if existing_meta is not None:
            print(f"No new frames. Existing metadata rows: {len(existing_meta):,}")
        else:
            print("No new frames and no existing metadata to write.")

    # Write bad_components.csv (append-safe + dedup by path)
    if bad_rows:
        bad_df_new = pd.DataFrame(bad_rows)
        if bad_csv.exists():
            try:
                bad_df_old = pd.read_csv(bad_csv)
            except Exception:
                bad_df_old = pd.DataFrame(columns=bad_df_new.columns)
        else:
            bad_df_old = pd.DataFrame(columns=bad_df_new.columns)

        bad_all = pd.concat([bad_df_old, bad_df_new], ignore_index=True)
        bad_all = bad_all.drop_duplicates(subset=["path", "reason"], keep="last")
        bad_all.to_csv(bad_csv, index=False)
        print(f"✗ Logged bad components: {len(bad_df_new):,} new (total {len(bad_all):,}) → {bad_csv}")

        # Group into failed_timestamps.csv
        grp = (bad_all.groupby(["harpnum", "date_obs"])
               .agg(n_bad=("component", "count"),
                    components=("component", lambda s: ",".join(sorted(set(s)))))
               .reset_index()
               .sort_values(["harpnum", "date_obs"]))
        grp.to_csv(failed_ts_csv, index=False)
        print(f"✗ Wrote grouped timestamps to: {failed_ts_csv} (rows: {len(grp):,})")
    else:
        print("✓ No bad components encountered in this run.")

    # Optional deletion
    if delete_old and to_delete:
        print(f"\nDeleting {len(to_delete):,} source component files...")
        deleted = 0
        for p in tqdm(to_delete, desc="Deleting", unit="file"):
            try:
                Path(p).unlink(missing_ok=True)
                deleted += 1
            except Exception as e:
                warnings.warn(f"Could not delete {p}: {e}")
        print(f"✓ Deleted {deleted:,} files.")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"✓ New frames: {processed:,}")
    print(f"• Already had: {already:,}")
    print(f"↻ Skipped:     {skipped:,}")
    out_meta = meta_out if meta_out.exists() else meta_out.with_suffix(".csv")
    print(f"📂 Output dir:  {frames_dir}")
    print(f"📊 Metadata:    {out_meta}")
    print(f"🧾 Fail logs:   {bad_csv}  |  {failed_ts_csv}")
    print("=" * 60)


if __name__ == "__main__":
    warnings.filterwarnings("default", category=UserWarning)
    main()
