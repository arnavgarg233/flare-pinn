import os
from pathlib import Path
import numpy as np, pandas as pd
from astropy.io import fits
from tqdm import tqdm
import click, cv2

from src.utils.common import (
    load_cfg, ensure_dirs, to_local_fields,
    pixel_scale_Mm_per_pix_from_header, cmd_abs_deg_from_header
)

def seg_from_filename(name: str):
    n = name.lower()
    if n.endswith(".br.fits") or "_br.fits" in n: return "Br"
    if n.endswith(".bt.fits") or "_bt.fits" in n: return "Bt"
    if n.endswith(".bp.fits") or "_bp.fits" in n: return "Bp"
    return None

@click.command()
@click.option("--cfg", default="configs/data_train.yaml")
@click.option("--delete-raw/--keep-raw", default=True, help="Delete FITS after writing NPZ.")
def main(cfg, delete_raw):
    cfg = load_cfg(cfg); ensure_dirs(cfg)
    idx_path = Path(cfg["paths"]["interim_dir"]) / "sharp_files_index.parquet"
    if not idx_path.exists():
        raise SystemExit(f"Missing index: {idx_path}. Run fetch_sharp first.")
    df = pd.read_parquet(idx_path)

    # Build triplets (Br,Bt,Bp) keyed by (harpnum, DATE-OBS)
    recs = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Indexing FITS"):
        fpath = Path(row.fits)
        seg = seg_from_filename(fpath.name)
        with fits.open(fpath) as hdul:
            hdr = hdul[1].header if len(hdul)>1 else hdul[0].header
            date_obs = hdr.get("DATE-OBS") or hdr.get("T_REC")
        if seg in {"Br","Bt","Bp"} and date_obs:
            recs.append({"harpnum": int(row.harpnum), "date_obs": str(date_obs), "seg": seg, "path": str(fpath)})

    if not recs:
        raise SystemExit("No recognizable Br/Bt/Bp FITS files found in raw.")
    tab = pd.DataFrame(recs)
    trip = tab.pivot_table(index=["harpnum","date_obs"], columns="seg", values="path", aggfunc="first").reset_index()
    trip = trip.dropna(subset=["Br","Bt","Bp"]).sort_values(["harpnum","date_obs"])

    out_dir = Path(cfg["paths"]["interim_dir"]) / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_px = int(cfg["resize"]["target_px"])

    def robust_norm(a: np.ndarray):
        med = np.median(a); q1,q3 = np.percentile(a, [25,75]); iqr = max(q3-q1, 1e-6)
        z = (a - med) / (iqr / 1.349)
        return np.clip(z, -5, 5).astype(np.float32), float(med), float(iqr)

    meta_rows = []
    for r in tqdm(trip.itertuples(), total=len(trip), desc="Preprocess triplets"):
        fBr, fBt, fBp = Path(r.Br), Path(r.Bt), Path(r.Bp)
        with fits.open(fBr) as hBr, fits.open(fBt) as hBt, fits.open(fBp) as hBp:
            Br = (hBr[1].data if len(hBr)>1 else hBr[0].data).astype(np.float32)
            Bt = (hBt[1].data if len(hBt)>1 else hBt[0].data).astype(np.float32)
            Bp = (hBp[1].data if len(hBp)>1 else hBp[0].data).astype(np.float32)
            hdr = hBr[1].header if len(hBr)>1 else hBr[0].header

        # Convert to local fields and resize
        Bx, By, Bz = to_local_fields(Br, Bt, Bp)
        Bx = cv2.resize(Bx, (target_px, target_px), interpolation=cv2.INTER_CUBIC)
        By = cv2.resize(By, (target_px, target_px), interpolation=cv2.INTER_CUBIC)
        Bz = cv2.resize(Bz, (target_px, target_px), interpolation=cv2.INTER_CUBIC)

        # Robust normalize -> clip -> store as fp16 (compact)
        Bx_n, Bx_med, Bx_iqr = robust_norm(Bx)
        By_n, By_med, By_iqr = robust_norm(By)
        Bz_n, Bz_med, Bz_iqr = robust_norm(Bz)

        # Cast to float16 for disk; keep stats as float32 scalars
        Bx_n = Bx_n.astype(np.float16)
        By_n = By_n.astype(np.float16)
        Bz_n = Bz_n.astype(np.float16)

        pxscale_Mm = pixel_scale_Mm_per_pix_from_header(hdr)
        cmd_deg = cmd_abs_deg_from_header(hdr)

        out_npz = out_dir / f"H{r.harpnum}_{r.date_obs.replace(':','-').replace(' ','T')}.npz"
        np.savez_compressed(
            out_npz,
            Bx=Bx_n, By=By_n, Bz=Bz_n,
            Bx_med=np.float32(Bx_med), Bx_iqr=np.float32(Bx_iqr),
            By_med=np.float32(By_med), By_iqr=np.float32(By_iqr),
            Bz_med=np.float32(Bz_med), Bz_iqr=np.float32(Bz_iqr),
            pxscale_Mm=np.float32(pxscale_Mm),
            cmd_deg=np.float32(cmd_deg),
            harpnum=np.int32(r.harpnum),
            date_obs=str(r.date_obs),
        )

        # Optional: delete raw FITS right away to save disk
        if delete_raw:
            for f in (fBr, fBt, fBp):
                try: os.remove(f)
                except Exception as e: print(f"warn: could not delete {f}: {e}")

        meta_rows.append({
            "npz": str(out_npz),
            "harpnum": int(r.harpnum),
            "date_obs": str(r.date_obs),
            "cmd_deg": float(cmd_deg),
            "pxscale_Mm": float(pxscale_Mm)
        })

    meta = pd.DataFrame(meta_rows).sort_values(["harpnum","date_obs"])
    meta_out = Path(cfg["paths"]["interim_dir"]) / "frames_meta.parquet"
    meta.to_parquet(meta_out, index=False)
    print(f"Wrote frames + meta: {meta_out} (n={len(meta)})")

if __name__ == "__main__":
    main()
