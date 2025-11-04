#!/usr/bin/env python3
"""
audit_sharp_tree.py
Day-by-day audit of SHARP downloads (.npz). Compatible with the summary
returned by src/data/test_sharp_quality.py (the NPZ-only checker).

Outputs:
  - audit_per_day.csv  (status: ok / fail / no_data)
  - audit_by_month.csv
"""

from __future__ import annotations
import argparse, math, sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict
import pandas as pd
from tqdm import tqdm

# --- Import summarize_quality robustly (works from repo root or inside src/data)
try:
    from src.data.test_sharp_quality import summarize_quality  # preferred explicit path
except Exception:
    from test_sharp_quality import summarize_quality  # fallback if script is run from src/data directly


def looks_like_day_dir(p: Path) -> bool:
    n = p.name
    return p.is_dir() and len(n) == 10 and n[4] == "-" and n[7] == "-"

def find_day_dirs(root: Path, years: List[int]) -> List[Path]:
    out: List[Path] = []
    for y in years:
        ydir = root / str(y)
        if not ydir.exists():
            continue
        for mdir in sorted(ydir.iterdir()):
            if not mdir.is_dir():
                continue
            for ddir in sorted(mdir.iterdir()):
                if looks_like_day_dir(ddir):
                    out.append(ddir)
    return out

def _first_key(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def main():
    ap = argparse.ArgumentParser(description="Audit SHARP downloads per day for multiple years.")
    ap.add_argument("--root", required=True, help=r"Root of downloads (e.g. S:\SHARP_1h)")
    ap.add_argument("--years", nargs="+", type=int, required=True, help="Years to audit, e.g. 2011 2012 2013")
    ap.add_argument("--sample-size", type=int, default=30, help="Frames sampled per day")
    ap.add_argument("--divergence-threshold", type=float, default=1e-3)
    ap.add_argument("--finite-min-frac", type=float, default=0.7)
    ap.add_argument("--min-frames", type=int, default=10, help="Require at least N frames to judge a day")
    ap.add_argument("--min-readability", type=float, default=0.95, help="Readable fraction needed to pass")
    ap.add_argument("--min-allthree", type=float, default=0.95, help="Fraction of frames with Br/Bt/Br needed to pass")
    ap.add_argument("--out", default="data/logs/audit", help="Output directory")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "audit_per_day.csv"
    out_month_csv = out_dir / "audit_by_month.csv"

    day_dirs = find_day_dirs(root, args.years)
    print(f"Auditing {len(day_dirs)} day folders under {root} ...")

    rows: List[Dict[str, Any]] = []
    for ddir in tqdm(day_dirs, unit="day"):
        # quick device hiccup guard + count .npz
        try:
            npz_count = sum(1 for _ in ddir.glob("*.npz"))
        except OSError:
            # device temporarily unavailable
            continue
        has_npz = npz_count > 0

        year = None
        try:
            year = int(ddir.parts[-3]) if ddir.parts[-3].isdigit() else None
        except Exception:
            pass

        if not has_npz:
            # no files at all → mark as no_data
            rows.append(dict(
                day_path=str(ddir), year=year, month=ddir.parts[-2], day=ddir.name,
                has_npz=False,
                total_npz=0, field_total=0, field_readable=0, field_corrupt=0,
                unique_frames=0, frames_with_all_three=0,
                field_readable_rate=math.nan, all_three_rate=math.nan,
                status="no_data", pass_field=False
            ))
            continue

        # run summary
        try:
            summary, _ = summarize_quality(
                root=ddir,
                sample_size=args.sample_size,
                divergence_threshold=args.divergence_threshold,
                finite_min_frac=args.finite_min_frac,
            )
            s = asdict(summary)
        except Exception as e:
            rows.append(dict(
                day_path=str(ddir), year=year, month=ddir.parts[-2], day=ddir.name,
                has_npz=True,
                total_npz=npz_count, field_total=0, field_readable=0, field_corrupt=0,
                unique_frames=0, frames_with_all_three=0,
                field_readable_rate=math.nan, all_three_rate=math.nan,
                status=f"error:{e}", pass_field=False
            ))
            continue

        # tolerate old/new field names
        total_npz = int(_first_key(s, "total_npz", default=0) or 0)
        field_total = int(_first_key(s, "field_total", default=0) or 0)
        field_readable = int(_first_key(s, "field_readable", default=0) or 0)
        field_corrupt = int(_first_key(s, "field_corrupt", default=0) or 0)
        unique_frames = int(_first_key(s, "unique_frames", default=0) or 0)
        frames_with_all_three = int(_first_key(s, "frames_with_all_three", default=0) or 0)

        # if field_total missing (older summary), approximate: 3 files per complete frame
        if field_total == 0 and frames_with_all_three > 0:
            field_total = 3 * frames_with_all_three
            if field_readable == 0:
                field_readable = field_total

        # derive rates
        field_readable_rate = (field_readable / field_total) if field_total else math.nan
        all_three_rate = (frames_with_all_three / unique_frames) if unique_frames else math.nan

        # decide status
        if unique_frames < args.min_frames or field_total == 0:
            status = "no_data"
            ok = False
        else:
            ok = ( (field_readable_rate >= args.min_readability) and
                   (all_three_rate >= args.min_allthree) )
            status = "ok" if ok else "fail"

        rows.append(dict(
            day_path=str(ddir),
            year=year,
            month=ddir.parts[-2],
            day=ddir.name,
            has_npz=True,
            total_npz=total_npz,
            field_total=field_total,
            field_readable=field_readable,
            field_corrupt=field_corrupt,
            unique_frames=unique_frames,
            frames_with_all_three=frames_with_all_three,
            field_readable_rate=field_readable_rate,
            all_three_rate=all_three_rate,
            status=status,
            pass_field=ok,
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows collected (no .npz found or drive not accessible).")
        return

    # save per-day
    df.to_csv(out_csv, index=False)
    print(f"\nWrote per-day audit to: {out_csv}")

    # console summary (exclude no_data from pass rate)
    data_df = df[df["status"] != "no_data"]
    total_days = len(data_df)
    passed = int(data_df["pass_field"].sum())
    print(f"\nDays with data: {total_days}")
    print(f"Days passing:   {passed} ({(passed/total_days*100 if total_days else 0):.1f}%)")

    # by-month rollup
    df["ym"] = df["year"].astype(str) + "-" + df["month"]
    month = (df[df["status"]!="no_data"]
             .groupby("ym")
             .agg(days_with_data=("day", "count"),
                  pass_days=("pass_field", "sum"),
                  avg_readable_rate=("field_readable_rate", "mean"),
                  avg_allthree_rate=("all_three_rate", "mean"))
             .reset_index()
             .sort_values("ym"))
    month.to_csv(out_month_csv, index=False)
    print(f"Wrote monthly rollup to: {out_month_csv}")

    # show worst 10 days with data
    if not data_df.empty:
        bad = data_df.sort_values(["pass_field","field_readable_rate","all_three_rate"]).head(10)
        print("\nWorst 10 days (with data):")
        with pd.option_context("display.width", 160):
            print(bad[["day_path","field_readable_rate","all_three_rate","field_total","field_corrupt","unique_frames","frames_with_all_three","status"]].to_string(index=False))

if __name__ == "__main__":
    main()
