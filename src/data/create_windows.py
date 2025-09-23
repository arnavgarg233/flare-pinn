from pathlib import Path
import pandas as pd
from tqdm import tqdm
import click

from src.utils.common import load_cfg, ensure_dirs

@click.command()
@click.option("--cfg", default="configs/data_train.yaml")
def main(cfg):
    cfg = load_cfg(cfg); ensure_dirs(cfg)

    frames = pd.read_parquet(Path(cfg["paths"]["interim_dir"]) / "frames_meta.parquet")
    flares = pd.read_parquet(Path(cfg["paths"]["interim_dir"]) / "flares_hek.parquet")

    frames["date_obs"] = pd.to_datetime(frames["date_obs"])
    flares["start"] = pd.to_datetime(flares["start"])
    flares["end"]   = pd.to_datetime(flares["end"])

    window_h = int(cfg["window_hours"])
    stride_h = int(cfg["window_stride_hours"])
    targets_h = list(cfg["targets_hours"])
    cmd_thr = float(cfg["censoring"]["cmd_deg_threshold"])

    windows = []
    for harp, grp in tqdm(frames.groupby("harpnum"), total=frames["harpnum"].nunique(), desc="Windowing"):
        grp = grp.sort_values("date_obs").reset_index(drop=True)
        if len(grp) == 0: continue
        tmin, tmax = grp["date_obs"].min(), grp["date_obs"].max()
        times = pd.date_range(tmin + pd.Timedelta(hours=window_h), tmax, freq=f"{stride_h}H")
        if len(times) == 0: continue

        # nearest-row helper
        def nearest_row(ts):
            i = (grp["date_obs"]-ts).abs().argsort().iloc[0]
            return grp.iloc[i]

        for t0 in times:
            row_t0 = nearest_row(t0)
            out = {"harpnum": harp, "t0": t0, "npz_t0": row_t0["npz"]}

            for H in targets_h:
                t1 = t0 + pd.Timedelta(hours=H)
                row_t1 = nearest_row(t1)

                # Mask if either end of horizon is beyond limb threshold
                is_masked = bool((abs(row_t0["cmd_deg"]) > cmd_thr) or (abs(row_t1["cmd_deg"]) > cmd_thr))

                # Label positive if any flare overlaps [t0, t1]
                m = (flares["start"] < t1) & (flares["end"] >= t0)
                y = int(m.any())

                out[f"is_masked_{H}h"] = is_masked
                out[f"y_{H}h_any"] = y

            windows.append(out)

    win = pd.DataFrame(windows).sort_values(["harpnum","t0"])

    # Prevalence after censoring (per horizon)
    prev_rows = []
    for H in targets_h:
        m = win.loc[~win[f"is_masked_{H}h"], f"y_{H}h_any"].mean()
        prev_rows.append({"h": H, "prevalence": float(m)})
    preval = pd.DataFrame(prev_rows)

    out_path = Path(cfg["paths"]["processed_dir"]) / "windows.parquet"
    win.to_parquet(out_path, index=False)
    stats_path = Path(cfg["paths"]["logs_dir"]) / "prevalence_after_censoring.csv"
    preval.to_csv(stats_path, index=False)

    print(f"Wrote windows: {out_path} (n={len(win)})")
    print("Prevalence after censoring:")
    print(preval)

if __name__ == "__main__":
    main()
