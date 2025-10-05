#!/usr/bin/env python3
# Fast metadata rebuild for SHARP frames
# - Parallel I/O
# - Avoids loading large arrays
# - Keeps same columns as your original script
#
# Env knobs:
#   REBUILD_WORKERS: override thread count (e.g., set to 16/24/32)
#   REBUILD_SKIP_SORT=1: skip final sort for a tiny extra speed-up

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

OUT_ROOT = Path(r"S:\flare_forecasting")
FRAMES   = OUT_ROOT / "frames"
META_OUT = OUT_ROOT / "frames_meta.parquet"   # CSV fallback if Parquet fails

# --- Tuning ---
CPU = os.cpu_count() or 8
MAX_WORKERS = int(os.environ.get("REBUILD_WORKERS", min(32, max(8, CPU * 2))))
SKIP_SORT = os.environ.get("REBUILD_SKIP_SORT", "0") == "1"

# tiny helper
def _restore_date_from_name(stem: str):
    # Expect: H{harp}_{YYYY-MM-DDTHH-MM-SS}
    try:
        _, right = stem.split("_", 1)
        ymd, hms = right.split("T", 1)
        # turn 01-00-00 -> 01:00:00
        hms = hms.replace("-", ":", 2)
        return f"{ymd}T{hms}"
    except Exception:
        return None

def parse_frame_filename(p: Path):
    name = p.stem  # H{harp}_{YYYY-MM-DDTHH-MM-SS}
    if not name.startswith("H"):
        return None, None
    try:
        harp = int(name[1:name.index("_")])
    except Exception:
        harp = None
    return harp, _restore_date_from_name(name)

SCALAR_KEYS = [
    "Bx_median","By_median","Bz_median",
    "Bx_iqr","By_iqr","Bz_iqr",
    "Bx_nan","By_nan","Bz_nan",
    "harpnum","date_obs","pxscale","pxunit","cmd_deg","signs"
]

def process_one(p: Path):
    harp, date_from_name = parse_frame_filename(p)
    rel = str(p.relative_to(OUT_ROOT))
    row = {
        "frame_path": rel,
        "harpnum": harp,
        "date_obs": date_from_name,
        "has_Bx": False, "has_By": False, "has_Bz": False,
    }
    try:
        # np.load on NPZ reads only the ZIP central directory until you index a key.
        with np.load(p, allow_pickle=False) as z:
            files = set(z.files)
            row["has_Bx"] = "Bx" in files
            row["has_By"] = "By" in files
            row["has_Bz"] = "Bz" in files

            # grab tiny scalars if present (cheap); never touch big arrays
            for k in SCALAR_KEYS:
                if k in files:
                    v = z[k]
                    # scalar array -> .item(); small strings stay as strings
                    if v.shape == ():
                        try:
                            row[k] = v.item()
                        except Exception:
                            row[k] = str(v)
                    else:
                        # if someone accidentally stored non-scalar, keep a safe representation
                        try:
                            row[k] = v.tolist()
                        except Exception:
                            row[k] = np.nan
    except Exception as e:
        # unreadable NPZ—rare after your converter, but keep trace
        row["error"] = f"load_error:{type(e).__name__}"
    # filename fallback if NPZ didn’t carry harp/date
    if row.get("harpnum") in (None, "", np.nan):
        row["harpnum"] = harp
    if not row.get("date_obs"):
        row["date_obs"] = date_from_name
    return row

def save_meta(df: pd.DataFrame, path: Path):
    try:
        df.to_parquet(path, index=False)
        print(f"✓ Saved Parquet meta: {path}")
    except Exception as e:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"⚠ Parquet failed ({e}). Wrote CSV: {csv_path}")

def main():
    if not FRAMES.exists():
        print(f"Missing frames dir: {FRAMES}")
        sys.exit(2)

    files = list(FRAMES.glob("H*.npz"))  # don’t sort: saves time on huge dirs
    if not files:
        print("No frame NPZs found. Abort.")
        sys.exit(2)

    print(f"Rebuilding metadata from {len(files):,} frames with {MAX_WORKERS} threads...")

    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_one, p): p for p in files}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="file", desc="Reading frames"):
            rows.append(fut.result())

    meta = pd.DataFrame.from_records(rows)

    # Coerce types / dedup
    if "harpnum" in meta.columns:
        meta["harpnum"] = pd.to_numeric(meta["harpnum"], errors="coerce").astype("Int32")
    if "date_obs" in meta.columns:
        # ensure string (don’t parse to datetime; faster & keeps exact T format)
        meta["date_obs"] = meta["date_obs"].astype(str)

    meta = meta.drop_duplicates(subset=["frame_path"], keep="last")

    # quick counts
    has_cols = [c for c in ("has_Bx","has_By","has_Bz") if c in meta.columns]
    if has_cols:
        ok_B = int((meta[has_cols].all(axis=1)).sum())
        print(f"Frames indexed: {len(meta):,} | with Bx/By/Bz present: {ok_B:,}")
    else:
        print(f"Frames indexed: {len(meta):,}")

    if not SKIP_SORT and {"harpnum","date_obs"}.issubset(meta.columns):
        meta = meta.sort_values(["harpnum", "date_obs"])

    save_meta(meta, META_OUT)

    # Optional: per-AR stats (fast—only groups scalars already read)
    stat_cols = [c for c in [
        "Bx_median","By_median","Bz_median",
        "Bx_iqr","By_iqr","Bz_iqr",
        "Bx_nan","By_nan","Bz_nan"
    ] if c in meta.columns]

    if stat_cols and "harpnum" in meta.columns:
        per_ar = (meta.dropna(subset=["harpnum"])
                      .groupby("harpnum", as_index=False)[stat_cols]
                      .median(numeric_only=True))
        save_meta(per_ar, OUT_ROOT / "per_ar_stats.parquet")
        print(f"✓ Wrote per-AR stats for {len(per_ar):,} HARPs")
    else:
        print("Per-AR stats skipped (missing columns).")

    print("DONE.")

if __name__ == "__main__":
    main()
