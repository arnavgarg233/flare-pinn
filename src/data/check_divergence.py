from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
import click

from utils.common import load_cfg, finite_diff_divergence

@click.command()
@click.option("--cfg", default="configs/data_train.yaml")
@click.option("--n", default=200, help="Random sample size.")
def main(cfg, n):
    cfg = load_cfg(cfg)
    meta = pd.read_parquet(Path(cfg["paths"]["interim_dir"]) / "frames_meta.parquet")
    if len(meta) == 0:
        raise SystemExit("frames_meta.parquet missing.")
    sample = meta.sample(min(n, len(meta)), random_state=0)

    vals = []
    for row in tqdm(sample.itertuples(), total=len(sample), desc="Divergence check"):
        arr = np.load(row.npz)
        Bx, By = arr["Bx"].astype(np.float32), arr["By"].astype(np.float32)
        div = finite_diff_divergence(Bx, By, dx=1.0, dy=1.0)
        vals.append(div.mean())

    vals = np.array(vals, dtype=np.float32)
    print(f"Mean(div) over {len(vals)} frames: {vals.mean():.3e} ± {vals.std():.3e}")

if __name__ == "__main__":
    main()
