from __future__ import annotations
import numpy as np
from typing import Tuple

def _safe_eps(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(x, eps, 1.0 - eps)

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_prob = _safe_eps(y_prob.astype(np.float64))
    return float(np.mean((y_prob - y_true) ** 2))

def confusion_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> tuple[int,int,int,int]:
    y_pred = (y_prob >= thr).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn

def tpr_fpr(tp:int, fp:int, fn:int, tn:int) -> Tuple[float,float]:
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    return tpr, fpr

def tss_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    tp, fp, fn, tn = confusion_at_threshold(y_true, y_prob, thr)
    tpr, fpr = tpr_fpr(tp, fp, fn, tn)
    return float(tpr - fpr)

def sweep_tss(y_true: np.ndarray, y_prob: np.ndarray, n: int = 512) -> Tuple[float, float]:
    if len(y_prob) <= n:
        thrs = np.unique(np.sort(y_prob))
    else:
        thrs = np.linspace(0, 1, n)
    best = -1.0
    best_thr = 0.5
    for thr in thrs:
        s = tss_at_threshold(y_true, y_prob, float(thr))
        if s > best:
            best = s
            best_thr = float(thr)
    return best_thr, best

def select_threshold_at_far(y_true: np.ndarray, y_prob: np.ndarray, max_far: float = 0.05, n: int = 2048) -> float:
    thrs = np.linspace(0, 1, n)
    chosen = 1.0
    for thr in thrs:
        _, fp, _, tn = confusion_at_threshold(y_true, y_prob, float(thr))
        fpr = fp / max(1, fp + tn)
        if fpr <= max_far:
            chosen = float(thr)
            break
    return chosen

def precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, n: int = 512) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    thrs = np.linspace(0, 1, n)[::-1]
    P, R = [], []
    for thr in thrs:
        tp, fp, fn, _ = confusion_at_threshold(y_true, y_prob, float(thr))
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        P.append(prec); R.append(rec)
    return np.asarray(R), np.asarray(P), thrs

def pr_auc(y_true: np.ndarray, y_prob: np.ndarray, n: int = 512) -> float:
    R, P, _ = precision_recall_curve(y_true, y_prob, n=n)
    idx = np.argsort(R)
    R, P = R[idx], P[idx]
    return float(np.trapz(P, R))

def adaptive_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = y_true.astype(np.float64)
    y_prob = _safe_eps(y_prob.astype(np.float64))
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(y_prob, qs)
    edges[0], edges[-1] = 0.0, 1.0
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1] + 1e-12
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0: 
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        w = float(mask.mean())
        ece += w * abs(acc - conf)
    return float(ece)

def reliability_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[float,float]:
    y_true = y_true.astype(np.float64)
    y_prob = _safe_eps(y_prob.astype(np.float64))
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(y_prob, qs); edges[0], edges[-1] = 0.0, 1.0
    xs, ys, ws = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1] + 1e-12
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0: continue
        xs.append(float(y_prob[m].mean()))
        ys.append(float(y_true[m].mean()))
        ws.append(float(m.sum()))
    X = np.vstack([np.ones(len(xs)), np.asarray(xs)]).T
    y = np.asarray(ys)
    W = np.diag(np.asarray(ws))
    XtWX = X.T @ W @ X
    beta = np.linalg.pinv(XtWX) @ X.T @ W @ y
    intercept, slope = float(beta[0]), float(beta[1])
    return slope, intercept
