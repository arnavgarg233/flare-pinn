# src/data/fetch_sharp.py
"""
Downloader for HMI SHARP CEA vector components (Br, Bt, Bp).

Key behavior
------------
- DRMS "all HARPNUMs" by default (fast): series[][{START_TAI}-{END_TAI}][? QUALITY >= 0 ?]{Br,Bt,Bp}
- Auto-splits big spans recursively on errors/timeouts down to --min-days.
- Long HTTP timeout is configurable (default 300s) to avoid JSON timeouts.
- Optional single-HARPNUM mode via --harp (uses JSOCClient with FITS).

Requires: sunpy>=5, drms, astropy
"""

from __future__ import annotations

import os
import re
import sys
import time
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Optional

import click
from astropy.time import Time
import drms
from drms import DrmsExportError, DrmsQueryError
from sunpy.net import jsoc
from sunpy.net import attrs as a


# ------------------------------ helpers ------------------------------------- #

def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)

def _tai(ts: str) -> str:
    """DRMS-friendly TAI timestamp like 2014.10.01_00:00:00_TAI."""
    return Time(ts).tai.strftime("%Y.%m.%d_%H:%M:%S_TAI")

def month_chunks(start: str, end: str) -> Iterable[Tuple[str, str]]:
    d0, d1 = _dt(start), _dt(end)
    d = d0.replace(day=1)
    while d < d1:
        n = (d.replace(day=28) + timedelta(days=4)).replace(day=1)  # next month
        yield d.strftime("%Y-%m-%d"), min(n, d1).strftime("%Y-%m-%d")
        d = n

def _segment_attrs(segments: Sequence[str]):
    """Return one JSOC Segment attr per segment (e.g., Segment('Br'), ...)."""
    return [a.jsoc.Segment(seg) for seg in segments]

_REQID_RE = re.compile(r"(JSOC_\d{8}_[0-9]+)")

def _resume_pending_if_any(client: jsoc.JSOCClient, ex: BaseException, out: Path,
                           poll_sleep: int, max_conn: int) -> bool:
    """If a JSOC export error mentions a request ID, resume it."""
    m = _REQID_RE.search(str(ex))
    if not m:
        return False
    reqid = m.group(1)
    try:
        print(f"    ↪ found pending export {reqid}; resuming download …")
        res = client.get_request(reqid, path=str(out), progress=True, sleep=poll_sleep, max_conn=max_conn)
        res.wait(progress=True)
        print("    ✓ resumed & downloaded")
        return True
    except Exception as rex:
        print(f"    ! failed to resume {reqid}: {type(rex).__name__}: {rex}")
        return False


# ------------------------------ CLI ----------------------------------------- #

@click.command()
@click.option("--start", default="2010-05-01", show_default=True, help="inclusive start date (YYYY-MM-DD)")
@click.option("--end",   default="2025-09-23", show_default=True, help="exclusive end date (YYYY-MM-DD)")
@click.option("--out",   default="data/raw/sharp_cea", show_default=True, type=click.Path(path_type=Path))
@click.option("--sleep", default=10, show_default=True, help="seconds between top-level chunks")
@click.option("--retries", default=3, show_default=True, help="retries per span")
@click.option("--retry-sleep", default=30, show_default=True, help="base seconds between retries within a span")
@click.option("--min-days", default=3, show_default=True, help="smallest chunk size when auto-splitting")
@click.option("--max-conn", default=1, show_default=True, help="parallel connections for downloads (JSOC default is 1)")
@click.option("--poll-sleep", default=15, show_default=True, help="seconds between JSOC poll attempts")
@click.option("--http-timeout", default=300, show_default=True, help="HTTP timeout (seconds) for DRMS JSON calls")
@click.option("--series", default="hmi.sharp_cea_720s", show_default=True, help="Series to download")
@click.option("--segments", multiple=True, default=("Br", "Bt", "Bp"), show_default=True,
              help="Segments to download; pass multiple like --segments Br --segments Bt --segments Bp")
@click.option("--email", default=None, help="JSOC/DRMS email (must be registered at JSOC)")
@click.option("--harp", default=None, help="HARPNUM (e.g., 377). If omitted, downloads ALL HARPNUMs via DRMS.")
@click.option("--test", is_flag=True, help="Run only the first month chunk (smoke test)")
@click.option("--force", is_flag=True, help="Ignore .done_ flags and re-run chunks")
def main(start: str, end: str, out: Path, sleep: int, retries: int, retry_sleep: int,
         min_days: int, max_conn: int, poll_sleep: int, http_timeout: int, series: str,
         segments: Sequence[str], email: Optional[str], harp: Optional[str], test: bool, force: bool):
    out.mkdir(parents=True, exist_ok=True)

    # Email is required for DRMS and JSOC exports
    email = (email or os.environ.get("SUNPY_JSOC_EMAIL") or os.environ.get("JSOC_EMAIL") or "").strip()
    if not email:
        sys.exit("Set --email or SUNPY_JSOC_EMAIL (registered at JSOC) before running.")

    # Set a long default HTTP timeout for DRMS JSON roundtrips
    socket.setdefaulttimeout(int(http_timeout))

    client = jsoc.JSOCClient()

    chunks = list(month_chunks(start, end))
    if test and chunks:
        chunks = chunks[:1]
        print(f"[TEST MODE] Limited to 1 month: {chunks[0][0]}→{chunks[0][1]}")

    print(f"[fetch_sharp] span={start}→{end} into {out}")
    print(f"  top-level chunks: {len(chunks)}")
    mode = "JSOC (single HARPNUM)" if harp else "DRMS (all HARPNUMs)"
    print(f"  mode: {mode}")

    for i, (s, e) in enumerate(chunks, 1):
        tag = f"{s}_{e}".replace("-", "")
        done_flag = out / f".done_{tag}"
        print(f"\n[{i}/{len(chunks)}] {s}→{e}")
        if done_flag.exists() and not force:
            print("  (skipped, done)")
            continue

        ok = try_fetch_span(
            client=client,
            out=out,
            s=s, e=e,
            email=email,
            series=series,
            segments=list(segments),
            retries=retries,
            retry_sleep=retry_sleep,
            poll_sleep=poll_sleep,
            max_conn=max_conn,
            min_days=min_days,
            http_timeout=http_timeout,
            harp=harp,
        )
        if ok:
            done_flag.touch()
        time.sleep(max(0, sleep))


# --------------------------- core fetch logic -------------------------------- #

def try_fetch_span(*, client: jsoc.JSOCClient, out: Path, s: str, e: str, email: str,
                   series: str, segments: Sequence[str], retries: int, retry_sleep: int,
                   poll_sleep: int, max_conn: int, min_days: int, http_timeout: int,
                   harp: Optional[str]) -> bool:
    """
    If --harp is provided: use JSOCClient with PrimeKey(HARPNUM=harp).
    Else: use DRMS recordset for ALL HARPNUMs in the window.
    Both paths auto-split spans on repeated errors/timeouts down to --min-days.
    """
    delta_days = (_dt(e) - _dt(s)).days

    if harp:
        # --- JSOC path: single HARPNUM (FITS) ---
        try:
            resp = client.search(
                a.Time(s, e),
                a.jsoc.Series(series),
                a.jsoc.PrimeKey("HARPNUM", str(harp)),
                *_segment_attrs(segments),
                a.jsoc.Notify(email),
            )
        except Exception as ex:
            print(f"  · JSOC search error ({ex}); falling back to DRMS for this span …")
            return _drms_span(out, s, e, series, segments, retries, retry_sleep, min_days, http_timeout)

        if len(resp) == 0:
            print("  · empty JSOC result; falling back to DRMS for this span …")
            return _drms_span(out, s, e, series, segments, retries, retry_sleep, min_days, http_timeout)

        for attempt in range(1, retries + 1):
            try:
                print(f"  fetching {s}→{e} (Δ={delta_days}d) via JSOCClient, HARPNUM={harp}")
                client.fetch(resp, path=str(out), progress=True, sleep=poll_sleep, max_conn=max_conn, wait=True)
                print("  ✓ downloaded")
                return True
            except DrmsExportError as ex:
                if "status=7" in str(ex) and _resume_pending_if_any(client, ex, out, poll_sleep, max_conn):
                    return True
                delay = max(5, retry_sleep * attempt)
                print(f"  ! export error (try {attempt}/{retries}): {ex} — retrying in {delay}s …")
                time.sleep(delay)
            except TimeoutError as ex:
                delay = max(5, retry_sleep * attempt)
                print(f"  ! timeout (try {attempt}/{retries}): {ex} — retrying in {delay}s …")
                time.sleep(delay)
            except Exception as ex:
                delay = max(5, retry_sleep * attempt)
                print(f"  ! unexpected error (try {attempt}/{retries}): {type(ex).__name__}: {ex} — retrying in {delay}s …")
                time.sleep(delay)

        print("  · JSOCClient fetch failed; trying DRMS for this span …")
        return _drms_span(out, s, e, series, segments, retries, retry_sleep, min_days, http_timeout)

    else:
        # --- DRMS path: ALL HARPNUMs ---
        return _drms_span(out, s, e, series, segments, retries, retry_sleep, min_days, http_timeout)


def _drms_span(out: Path, s: str, e: str, series: str, segments: Sequence[str],
               retries: int, retry_sleep: int, min_days: int, http_timeout: int) -> bool:
    """
    DRMS 'url_quick/as-is' export with QUALITY filter and long JSON timeout.
    Auto-splits span on failure down to --min-days.
    """
    delta_days = (_dt(e) - _dt(s)).days
    # set (again) in case caller changed it; defensive
    socket.setdefaulttimeout(int(http_timeout))

    c = drms.Client()  # server defaults to JSOC
    rec = f"{series}[][{_tai(s)}-{_tai(e)}][? QUALITY >= 0 ?]{{{','.join(segments)}}}"

    for attempt in range(1, retries + 1):
        try:
            print(f"  exporting via DRMS url_quick/as-is: {rec}")
            req = c.export(rec, method="url_quick", protocol="as-is", email=os.environ.get("SUNPY_JSOC_EMAIL"))
            # url_quick has direct URLs; just download
            req.download(str(out))
            print("  ✓ downloaded (DRMS)")
            return True

        except TimeoutError as ex:
            if attempt < retries:
                delay = 30 * attempt
                print(f"  ! JSON timeout (try {attempt}/{retries}): {ex} — retrying in {delay}s …")
                time.sleep(delay)
                continue
            print("  ! JSON timeout — no retries left")

        except (DrmsExportError, DrmsQueryError) as ex:
            if attempt < retries:
                delay = 30 * attempt
                print(f"  ! DRMS error (try {attempt}/{retries}): {ex} — retrying in {delay}s …")
                time.sleep(delay)
                continue

    # If we reach here, split the span if large enough
    if delta_days > min_days:
        mid = _dt(s) + timedelta(days=delta_days // 2)
        left_s, left_e = s, mid.strftime("%Y-%m-%d")
        right_s, right_e = mid.strftime("%Y-%m-%d"), e
        print(f"    ↪ splitting {s}→{e} into {left_s}→{left_e} and {right_s}→{right_e}")
        ok1 = _drms_span(out, left_s, left_e, series, segments, retries, retry_sleep, min_days, http_timeout)
        ok2 = _drms_span(out, right_s, right_e, series, segments, retries, retry_sleep, min_days, http_timeout)
        return ok1 and ok2

    print("  × giving up on this span (reached minimum chunk size)")
    return False


# ------------------------------- main ---------------------------------------- #

if __name__ == "__main__":
    main()
