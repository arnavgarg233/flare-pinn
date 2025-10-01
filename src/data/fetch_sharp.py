#!/usr/bin/env python3
"""
download_sharp_cea.py
---------------------
Bulk-download HMI SHARP CEA data (hmi.sharp_cea_720s) from JSOC using the SunPy
`drms` client. Designed for long spans.

Highlights
- Default 1-hour cadence (override with --step 12m for full 12-min).
- Chunked exports (day/month) to keep requests manageable & resumable.
- Parallel workers; safe resume: converted .npz prevents rework.
- Auto-pause when offline; auto-resume when network returns (--retry-forever).
- Optional HARPNUM filtering; estimate mode to count records first.
- Compact postprocess: saves Br/Bt/Bp as float32, bitmap as uint8, + metadata.
- Optional deletion of FITS after successful convert to save disk space.
- CSV manifest of everything fetched/converted.

Requirements
    pip install drms pandas tqdm tenacity astropy numpy

Usage (PowerShell example)
    $env:JSOC_EMAIL="arnavgarg888@gmail.com"
    python .\download_sharp_cea.py --start 2011-01-01 --end 2019-01-01 `
      --step 1h --out D:\SHARP_1h --segments Br Bt Bp bitmap `
      --chunk day --workers 6 --delete-after --retry-forever --verbose
"""

import argparse
import os
import sys
import time
import math
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import drms
from astropy.time import Time
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

SERIES = "hmi.sharp_cea_720s"
DEFAULT_SEGMENTS = ["Br", "Bt", "Bp", "bitmap"]  # include bitmap -> mask
DEFAULT_STEP = "1h"  # default cadence (override with --step 12m for 12-min)
TIME_FMT = "%Y.%m.%d_%H:%M:%S_TAI"  # JSOC time string format

def parse_args():
    p = argparse.ArgumentParser(description="Bulk download HMI SHARP CEA data from JSOC.")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD) in UTC (inclusive).")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD) in UTC (exclusive).")
    p.add_argument("--out", default="./sharp_download", help="Output directory.")
    p.add_argument("--segments", nargs="+", default=DEFAULT_SEGMENTS, help="Segments to download.")
    p.add_argument("--chunk", choices=["day", "month"], default="day", help="Chunking granularity.")
    p.add_argument("--step", default=DEFAULT_STEP,
                   help="Cadence for T_REC selection, e.g. '1h', '12m'. "
                        "Leave empty to use native 12-min cadence.")
    p.add_argument("--harpnums", nargs="*", default=None,
                   help="Optional list of HARPNUMs to include (space/comma separated).")
    p.add_argument("--estimate", action="store_true",
                   help="Only estimate record counts and size; do not download.")
    p.add_argument("--fits-headers", action="store_true",
                   help="Request FITS with full headers (protocol=fits). If unset, use 'as-is' for speed.")
    p.add_argument("--max-retries", type=int, default=5, help="Max retries per export on errors.")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between chunks.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers (2–6 recommended).")
    p.add_argument("--delete-after", action="store_true",
                   help="Delete downloaded FITS after successful postprocess.")
    # Resilience to Wi-Fi drops
    p.add_argument("--retry-forever", action="store_true",
                   help="Keep retrying on network/server errors indefinitely.")
    p.add_argument("--net-poll", type=int, default=30,
                   help="Seconds between network checks when offline.")
    return p.parse_args()

def jsoc_email_or_die() -> str:
    email = os.environ.get("JSOC_EMAIL", "").strip()
    if not email:
        print("ERROR: JSOC_EMAIL environment variable is not set. "
              "Register your email with JSOC and set JSOC_EMAIL=you@example.com",
              file=sys.stderr)
        sys.exit(2)
    return email

def have_network(host="171.64.103.244", port=80, timeout=5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def daterange_chunks(start: datetime, end: datetime, mode: str) -> Iterable[Tuple[datetime, datetime]]:
    cur = start
    if mode == "day":
        step = timedelta(days=1)
        while cur < end:
            nxt = min(cur + step, end)
            yield cur, nxt
            cur = nxt
    else:
        def add_month(d: datetime) -> datetime:
            y, m = d.year, d.month
            return datetime(y + (m // 12), (m % 12) + 1, 1)
        while cur < end:
            nxt = add_month(cur)
            yield cur, min(nxt, end)
            cur = nxt

def make_recordset(start_dt: datetime, end_dt: datetime,
                   harpnums: Optional[List[str]], step: Optional[str]) -> List[str]:
    """
    Build recordset strings for this chunk.

    SHARP CEA prime keys are HARPNUM, T_REC (in that order). If you select by time
    across *all* HARPs, include an empty HARPNUM bracket: SERIES[][time].
    If you filter HARPNUMs explicitly, use SERIES[<HARPNUM>][time].
    """
    t0 = start_dt.strftime("%Y.%m.%d_%H:%M:%S_UTC")  # JSOC accepts _UTC (clean vs TAI)
    dur_days = (end_dt - start_dt).days + (end_dt - start_dt).seconds / 86400.0
    ndays = max(1, math.ceil(dur_days))
    step_part = f"@{step}" if step else ""  # empty -> native 12-min
    if not harpnums:
        return [f"{SERIES}[][{t0}/{ndays}d{step_part}]"]
    return [f"{SERIES}[{int(h)}][{t0}/{ndays}d{step_part}]" for h in harpnums]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def segment_brace(segments: List[str]) -> str:
    return "{" + ",".join(segments) + "}"

def csv_append(path: Path, df: pd.DataFrame):
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)

class NoDataError(Exception):
    pass

def postprocess_one_file(path: Path) -> Path:
    """
    Convert one SHARP FITS file to a compact .npz:
      - Br/Bt/Bp saved as float32 'data'
      - bitmap saved as uint8 'mask'
      - key metadata saved in 'meta' dict
    Returns the .npz path if successful.
    """
    out_path = path.with_suffix(".npz")
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        hdr = hdu.header
        arr = np.array(hdu.data)

        seg = (hdr.get("SEGMENT") or hdr.get("EXTNAME") or hdr.get("CONTENT") or "").lower()
        meta = {
            "segment": seg,
            "FILENAME": path.name,
            "T_REC": hdr.get("T_REC") or hdr.get("DATE-OBS"),
            "HARPNUM": int(hdr.get("HARPNUM") or hdr.get("HARP_NUM") or -1),
            "CDELT1": hdr.get("CDELT1"), "CDELT2": hdr.get("CDELT2"),
            "CRVAL1": hdr.get("CRVAL1"), "CRVAL2": hdr.get("CRVAL2"),
            "CRPIX1": hdr.get("CRPIX1"), "CRPIX2": hdr.get("CRPIX2"),
            "RSUN_OBS": hdr.get("RSUN_OBS"),
        }

        if "bitmap" in seg:
            mask = (arr > 0).astype(np.uint8)
            np.savez_compressed(out_path, mask=mask, meta=meta)
        else:
            data = arr.astype(np.float32, copy=False)
            np.savez_compressed(out_path, data=data, meta=meta)

    return out_path

@retry(reraise=True, stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=2, max=60),
       retry=retry_if_exception_type((drms.DrmsError, TimeoutError)))
def submit_export(client: drms.Client, rec: str, segments: List[str], fits_headers: bool) -> drms.ExportRequest:
    method = "url" if fits_headers else "url_quick"
    protocol = "fits" if fits_headers else "as-is"
    return client.export(f"{rec}{segment_brace(segments)}", method=method, protocol=protocol)

def _download_task(args_tuple):
    """
    Worker task: export+download for one recordset. Will:
      - Wait while offline (polling --net-poll)
      - Retry on errors with capped exponential backoff
      - Optionally retry forever (--retry-forever)
      - Postprocess to .npz and (optionally) delete FITS
    """
    (rec, subdir, segments, fits_headers, sleep_s, verbose,
     delete_after, max_retries, retry_forever, net_poll) = args_tuple

    email = os.environ.get("JSOC_EMAIL", "").strip()
    rows = []
    attempt = 0

    while True:
        # If offline, wait and poll
        if not have_network():
            if verbose:
                print(f"[INFO] No network for {rec}. Rechecking in {net_poll}s...")
            time.sleep(net_poll)
            continue

        try:
            client = drms.Client(email=email)  # fresh client per attempt
            exp = submit_export(client, rec, segments, fits_headers)
            dl_df = exp.download(str(subdir))
        except Exception as e:
            attempt += 1
            if not retry_forever and attempt >= max_retries:
                return [], f"{rec} :: {e}"
            backoff = min(60, 2 ** min(attempt, 6))  # 2,4,8,...,60s
            if verbose:
                print(f"[WARN] {rec} failed (attempt {attempt}): {e} -> retrying in {backoff}s")
            time.sleep(backoff)
            continue  # retry

        # Success: convert & (maybe) delete
        if dl_df is not None and "download" in dl_df.columns:
            for _, row in dl_df.iterrows():
                local = row.get("download")
                url = row.get("url")
                if not local:
                    continue
                path = Path(local)
                if not path.exists():
                    continue

                ok = True
                out_npz = None
                try:
                    out_npz = postprocess_one_file(path)
                except Exception as e:
                    ok = False
                    if verbose:
                        print(f"[WARN] postprocess failed: {path.name}: {e}")

                rows.append({
                    "recordset": rec,
                    "url": url,
                    "fits_path": os.path.abspath(local),
                    "npz_path": (str(out_npz) if out_npz else ""),
                    "npz_exists": bool(out_npz and Path(out_npz).exists()),
                    "postprocessed": ok,
                })

                if delete_after and ok:
                    try:
                        # Python 3.8+: missing_ok supported
                        path.unlink(missing_ok=True)
                        if verbose:
                            print(f"[INFO] Deleted FITS after convert: {path.name}")
                    except Exception as e:
                        print(f"[WARN] could not delete {path}: {e}")

        if sleep_s > 0:
            time.sleep(sleep_s)
        return rows, None

def main():
    args = parse_args()
    _ = jsoc_email_or_die()

    out_root = Path(args.out).expanduser().resolve()
    ensure_dir(out_root)

    # Parse dates
    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    if end_dt <= start_dt:
        print("ERROR: end must be after start.", file=sys.stderr)
        sys.exit(2)

    # Parse HARPNUMs if provided
    harpnums = None
    if args.harpnums:
        join = " ".join(args.harpnums).replace(",", " ").split()
        harpnums = [s.strip() for s in join if s.strip().isdigit()]
        if args.verbose:
            print(f"Filtering to HARPNUMs: {harpnums}")

    manifest_path = out_root / "manifest.csv"

    # Estimation-only pass (counts only; no downloads)
    if args.estimate:
        rows = []
        chunks = list(daterange_chunks(start_dt, end_dt, args.chunk))
        for c_start, c_end in tqdm(chunks, desc="Estimating", unit=args.chunk):
            recsets = make_recordset(c_start, c_end, harpnums, args.step)
            client = drms.Client(email=os.environ["JSOC_EMAIL"])
            for rec in recsets:
                try:
                    df = client.query(rec, key="HARPNUM,T_REC")
                    n = 0 if df is None else len(df)
                    rows.append({
                        "recordset": rec,
                        "start": c_start.isoformat(),
                        "end": c_end.isoformat(),
                        "count": int(n)
                    })
                except Exception as e:
                    rows.append({
                        "recordset": rec,
                        "start": c_start.isoformat(),
                        "end": c_end.isoformat(),
                        "count": -1,
                        "error": str(e)
                    })
                time.sleep(args.sleep)
        est = pd.DataFrame(rows)
        est["cum_count"] = est["count"].clip(lower=0).cumsum()
        est_path = out_root / "estimate.csv"
        est.to_csv(est_path, index=False)
        total = int(est["count"].clip(lower=0).sum())
        print(f"\nEstimated records across chunks: {total}")
        print(f"Wrote per-chunk details to: {est_path}")
        print("Note: long spans at 12-min cadence can be very large; consider 1h if bandwidth-limited.")
        return

    # ---- Parallel download loop ----
    tasks = []
    chunk_info = []
    for c_start, c_end in daterange_chunks(start_dt, end_dt, args.chunk):
        # Subdirectory per chunk (YYYY/MM or YYYY-MM-DD)
        subdir = out_root / f"{c_start:%Y}" / f"{c_start:%m}" / (f"{c_start:%Y-%m-%d}" if args.chunk == "day" else "")
        ensure_dir(subdir)
        recsets = make_recordset(c_start, c_end, harpnums, args.step)
        for rec in recsets:
            tasks.append((
                rec, subdir, args.segments, args.fits_headers,
                args.sleep, args.verbose, args.delete_after,
                args.max_retries, args.retry_forever, args.net_poll
            ))
            chunk_info.append((rec, c_start.isoformat(), c_end.isoformat()))

    if not tasks:
        print("No tasks to run for the given arguments.")
        return

    rec_to_bounds = {rec: (start, end) for rec, start, end in chunk_info}
    manifest_lock = Lock()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(_download_task, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            rows, err = fut.result()
            if err:
                print(f"[WARN] {err}", file=sys.stderr)
            if rows:
                # add start/end to each row and append to manifest
                for r in rows:
                    se = rec_to_bounds.get(r["recordset"])
                    if se:
                        r["start"], r["end"] = se
                with manifest_lock:
                    csv_append(manifest_path, pd.DataFrame(rows))

    print(f"\nDone. Manifest at: {manifest_path}")
    print("Tip: keep the manifest under version control to track exactly what was fetched.")

if __name__ == "__main__":
    main()
