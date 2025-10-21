from __future__ import annotations
import numpy as np
from typing import Callable, Sequence

def paired_block_bootstrap(
    groups: Sequence[int],
    y_true_A: np.ndarray, y_prob_A: np.ndarray,
    y_true_B: np.ndarray, y_prob_B: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = rng or np.random.default_rng(12345)
    def _metric(y_true, y_prob):
        return float(metric_fn(y_true, y_prob))
    deltas = []
    for _ in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sample])
        mA = _metric(y_true_A[idx], y_prob_A[idx])
        mB = _metric(y_true_B[idx], y_prob_B[idx])
        deltas.append(mA - mB)
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_mean": float(deltas.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}
