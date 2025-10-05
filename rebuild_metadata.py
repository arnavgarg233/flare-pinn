#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

OUT_ROOT = Path(r"S:\flare_forecasting")
FRAMES   = OUT_ROOT / "frames"
META_OUT = OUT_ROOT / "frames_meta.parquet"   # CSV fallback if Parquet fails

def parse_frame_filename(p: Path):
    # H{harpnum}_{YYYY-MM-DDTHH-MM-SS}.npz
    stem = p.stem
    if not stem.startswith("H") or "_" not in stem:
        return None, None
    left, right = stem.split("_", 1)
    try:
        harp = int(left[1:])
    except Exception:
        harp = None
    date_obs = right.replace("-", ":", 2).replace("-", ":", 1)  # put back colons only for HH:MM:SS
    # The filename used "-" everywhere; restore time colons safely:
    # e.g. 2018-02-26T01-00-00 -> 2018-02-26T01:00:00
    if "T" in date_obs and date_obs.count("-") >= 2:
        ymd, hms = date_obs.split("T", 1)
        hms = hms.replace("-", ":", 2)
        date_obs = f"{ymd}T{hms}"
    return harp, date_obs

def read_frame_npz(p: Path):
    with np.load(p, allow_pickle=False) as z:
        row = {
            "frame_path": str(p.relative_to(OUT_ROOT)),
            "has_Bx": "Bx" in z.files,
            "has_By": "By" in z.files,
            "has_Bz": "Bz" in z.files,
        }
        # Pull stored stats/metadata if present
        for k in [
            "Bx_median","By_median","Bz_median",
            "Bx_iqr","By_iqr","Bz_iqr",
            "Bx_nan","By_nan","Bz_nan",
            "harpnum","date_obs","pxscale","pxunit","cmd_deg","signs"
        ]:
            row[k] = (z[k].item() if k in z.files and z[k].shape == () else
                      (str(z[k]) if (k in z.files and z[k].dtype.type is np.str_) else
                       (z[k].tolist() if k in z.files else np.nan)))
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
    assert FRAMES.exists(), f"Missing frames dir: {FRAMES}"
    files = sorted(FRAMES.glob("H*.npz"))
    if not files:
        print("No frame NPZs found. Abort.")
        sys.exit(2)
    
    print(f"Rebuilding metadata from {len(files):,} frames...")

    rows = []
    for p in tqdm(files, desc="Reading frames", unit="file"):
        harp, date_obs = parse_frame_filename(p)
        try:
            r = read_frame_npz(p)
        except Exception as e:
            # Minimal row if NPZ is unreadable (should be rare at this stage)
            r = {"frame_path": str(p.relative_to(OUT_ROOT)),
                 "error": f"load_error:{type(e).__name__}"}
        if "harpnum" not in r or pd.isna(r["harpnum"]) or r["harpnum"] in ("", None):
            r["harpnum"] = harp
        if "date_obs" not in r or pd.isna(r["date_obs"]) or r["date_obs"] in ("", None):
            r["date_obs"] = date_obs
        rows.append(r)

    meta = pd.DataFrame(rows)

    # Coerce types and clean
    meta["harpnum"] = pd.to_numeric(meta["harpnum"], errors="coerce").astype("Int32")
    meta["date_obs"] = meta["date_obs"].astype(str)
    # Drop obvious dups (same file path)
    meta = meta.drop_duplicates(subset=["frame_path"], keep="last")

    # Basic sanity
    n = len(meta)
    ok_B = int((meta.get("has_Bx", True) & meta.get("has_By", True) & meta.get("has_Bz", True)).sum())
    print(f"Frames indexed: {n:,} | with Bx/By/Bz present: {ok_B:,}")

    # Save meta
    save_meta(meta.sort_values(["harpnum", "date_obs"]), META_OUT)

    # Optional: per-AR stats table (using stored per-frame stats; compute medians)
    stat_cols = ["Bx_median","By_median","Bz_median","Bx_iqr","By_iqr","Bz_iqr","Bx_nan","By_nan","Bz_nan"]
    have_stats = [c for c in stat_cols if c in meta.columns]
    if have_stats:
        per_ar = (meta.dropna(subset=["harpnum"])
                       .groupby("harpnum", as_index=False)[have_stats]
                       .median(numeric_only=True))
        save_meta(per_ar, OUT_ROOT / "per_ar_stats.parquet")
        print(f"✓ Wrote per-AR stats for {len(per_ar):,} HARPs")
    else:
        print("No stored per-frame stats found; per-AR table skipped.")

    print("DONE.")

if __name__ == "__main__":
    main()
