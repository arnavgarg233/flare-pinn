import math
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

RSUN_Mm = 696.34  # solar radius in megameters

def load_cfg(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Create necessary directories from config."""
    for k in ["raw_dir", "interim_dir", "processed_dir", "cache_dir", "logs_dir"]:
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)

def pixel_scale_Mm_per_pix_from_header(hdr: Dict[str, Any]) -> float:
    """
    SHARP CEA: CDELT1 (deg/pix). Convert deg->Mm at photosphere.
    Returns Mm/pixel.
    """
    cdelt1 = float(hdr.get("CDELT1", "nan"))
    if math.isnan(cdelt1):
        return float("nan")
    return (cdelt1 * math.pi / 180.0) * RSUN_Mm

def cmd_abs_deg_from_header(hdr: Dict[str, Any]) -> float:
    """
    Approx |CMD| (deg) for patch center using CRLN_OBS and a center-long key.
    """
    cln_obs = hdr.get("CRLN_OBS")
    lon_center = None
    for k in ["LON_FWT", "CRLN", "LON_CEA", "CRLN_CNTR", "HGLOBS"]:
        if k in hdr: 
            lon_center = hdr[k]
            break
    if cln_obs is None or lon_center is None:
        return float("nan")
    cmd = (float(lon_center) - float(cln_obs)) % 360.0
    if cmd > 180.0: 
        cmd -= 360.0
    return abs(cmd)

def finite_diff_divergence(Bx: np.ndarray, By: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Compute divergence using finite differences."""
    dBx_dx = np.gradient(Bx, axis=1) / dx
    dBy_dy = np.gradient(By, axis=0) / dy
    return dBx_dx + dBy_dy

def to_local_fields(Br: np.ndarray, Bt: np.ndarray, Bp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert SHARP CEA coordinates to local field components.
    
    Plan's convention:
      Bz = Br
      Bx = -Bp  
      By = -Bt
    """
    Bz = Br
    Bx = -Bp
    By = -Bt
    return Bx, By, Bz

def validate_divergence(Bx: np.ndarray, By: np.ndarray, threshold: float = 1e-3) -> bool:
    """
    Validate that magnetic field is approximately divergence-free.
    
    Args:
        Bx, By: Magnetic field components
        threshold: Maximum allowed divergence magnitude
        
    Returns:
        True if divergence is below threshold
    """
    div = finite_diff_divergence(Bx, By)
    max_div = np.nanmax(np.abs(div))
    return max_div < threshold

def robust_normalize(data: np.ndarray, clip_range: Tuple[float, float] = (-5.0, 5.0)) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Robust normalization using median and IQR.
    
    Args:
        data: Input array
        clip_range: Range to clip normalized data
        
    Returns:
        Normalized data and statistics dictionary
    """
    median = np.nanmedian(data)
    q75, q25 = np.nanpercentile(data, [75, 25])
    iqr = q75 - q25
    
    if iqr == 0:
        normalized = np.zeros_like(data)
    else:
        normalized = (data - median) / iqr
    
    # Clip outliers
    normalized = np.clip(normalized, clip_range[0], clip_range[1])
    
    stats = {
        'median': median,
        'q25': q25,
        'q75': q75,
        'iqr': iqr
    }
    
    return normalized, stats
