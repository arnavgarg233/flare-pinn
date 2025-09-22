import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import click
import drms

from utils.common import load_cfg, ensure_dirs

@click.command()
@click.option("--cfg", default="configs/data_train.yaml", help="Config path.")
def main(cfg):
    cfg = load_cfg(cfg)
    ensure_dirs(cfg)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    start, end = cfg["span"]["start"], cfg["span"]["end"]
    cadence_min = int(cfg["cadence_minutes"])
    jsoc_email = os.environ.get("JSOC_EMAIL", "you@example.com")

    client = drms.Client(email=jsoc_email)

    print("Querying JSOC index ...")
    qstr = f"hmi.sharp_cea_720s[{start}/{end}@{cadence_min}m]"
    keys = ["HARPNUM", "T_REC", "T_START", "T_STOP"]
    q = client.query(qstr, key=keys)
    if len(q) == 0:
        raise SystemExit("No SHARP records for the given span/cadence.")

    df = pd.DataFrame(q)
    harp_nums = sorted(df["HARPNUM"].unique())
    print(f"HARPs in span: {len(harp_nums)}")

    segs = ["Br","Bt","Bp"]
    all_records = []

    for harp in tqdm(harp_nums, desc="Export HARPs"):
        harp_df = df[df["HARPNUM"] == harp].sort_values("T_REC")
        t0 = harp_df["T_REC"].iloc[0]
        t1 = harp_df["T_REC"].iloc[-1]
        sel = f"hmi.sharp_cea_720s[{t0}-{t1}@{cadence_min}m][? HARPNUM={harp} ?]{{{','.join(segs)}}}"
        exp = client.export(sel, method="url", protocol="fits")
        if exp.status != 0:
            print(f"Export error for HARPNUM={harp} (status {exp.status}); skipping.")
            continue
        outdir = raw_dir / f"HARP_{harp}"
        outdir.mkdir(parents=True, exist_ok=True)
        files = exp.download(dir=str(outdir))
        for f in files.downloaded:
            all_records.append({"harpnum": int(harp), "fits": f})

    idx = pd.DataFrame(all_records)
    idx_path = Path(cfg["paths"]["interim_dir"]) / "sharp_files_index.parquet"
    idx.to_parquet(idx_path, index=False)
    print(f"Wrote index: {idx_path} (files={len(idx)})")

if __name__ == "__main__":
    main()
