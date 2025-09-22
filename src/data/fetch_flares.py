from pathlib import Path
import pandas as pd
from sunpy.net import hek
import click

from utils.common import load_cfg, ensure_dirs

@click.command()
@click.option("--cfg", default="configs/data_train.yaml")
@click.option("--class_min", default="M", type=click.Choice(["C","M","X"]))
def main(cfg, class_min):
    cfg = load_cfg(cfg); ensure_dirs(cfg)
    start, end = cfg["span"]["start"], cfg["span"]["end"]

    hc = hek.HEKClient()
    results = hc.search(
        hek.attrs.Time(start, end),
        hek.attrs.EventType("FL"),
        hek.attrs.OBS.Observatory == "GOES",
        hek.attrs.FL.GOESCls >= class_min
    )
    if len(results) == 0:
        raise SystemExit("HEK returned 0 flares for the span.")

    df = results.to_pandas()
    keep = {
        "event_starttime": "start",
        "event_peaktime": "peak",
        "event_endtime": "end",
        "fl_goescls": "class",
        "ar_noaanum": "noaa_ar",
    }
    out = (
        df[list(keep.keys())]
        .rename(columns=keep)
        .assign(
            noaa_ar=lambda d: pd.to_numeric(d["noaa_ar"], errors="coerce"),
            start=lambda d: pd.to_datetime(d["start"]),
            end=lambda d: pd.to_datetime(d["end"])
        )
        .sort_values("start")
        .reset_index(drop=True)
    )
    out_path = Path(cfg["paths"]["interim_dir"]) / "flares_hek.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path} (rows={len(out)})")

if __name__ == "__main__":
    main()
