from __future__ import annotations
import numpy as np

SOBEL_X = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T.copy()

def _conv2d_valid(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    H, W = img.shape
    kh, kw = k.shape
    out = np.zeros((H- kh + 1, W - kw + 1), dtype=np.float32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i+kh, j:j+kw] * k)
    return out

def pil_mask_from_bz(
    Bz: np.ndarray,
    top_percent: float = 15.0,
    close_radius_px: int = 2,
    do_skeletonize: bool = True
) -> np.ndarray:
    Bz = Bz.astype(np.float32)
    pad = 1
    P = np.pad(Bz, pad, mode="reflect")
    gx = _conv2d_valid(P, SOBEL_X)
    gy = _conv2d_valid(P, SOBEL_Y)
    gmag = np.hypot(gx, gy)
    # gmag is already the right shape after valid convolution
    g = gmag
    thr = np.percentile(g, 100.0 - float(top_percent))
    mask = (g >= thr).astype(np.uint8)
    try:
        import scipy.ndimage as ndi
        if close_radius_px > 0:
            str_el = ndi.generate_binary_structure(2, 1)
            m = ndi.binary_dilation(mask, iterations=close_radius_px, structure=str_el)
            m = ndi.binary_erosion(m, iterations=close_radius_px, structure=str_el)
            mask = m.astype(np.uint8)
    except Exception:
        pass
    if do_skeletonize:
        try:
            from skimage.morphology import skeletonize
            mask = skeletonize(mask > 0).astype(np.uint8)
        except Exception:
            pass
    return mask
