#!/usr/bin/env python3
# src/train_pinn.py
"""
Minimal, reproducible training entrypoint for the 2.5D PINN (P0→P4s curriculum-ready).

It supports two data modes:
 - Dummy MMS-style synthetic sequences (for smoke tests; default)
 - Real SHARP data (stub hooks; fill TODOs once your windows parquet is finalized)

Usage
-----
python -m src.train_pinn --cfg configs/train_pinn_smoke_test.yaml
# or
python -m src.train_pinn --cfg configs/train_pinn.yaml
# switch to GPU by setting device: "cuda" in the config.
"""
from __future__ import annotations

import argparse, math, random, time, os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local modules
from src.models.pinn.core import PINNBackbone, ClassifierHead, B_perp_from_Az
from src.models.pinn.physics import WeakFormInduction2p5D
from src.models.pinn.collocation import mix_pil_uniform, clip_and_renorm_importance, ess


# ------------------------------- utils -------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ramp_value(step_frac: float, schedule: List[Tuple[float, float]]) -> float:
    """
    Piecewise-linear schedule lookup.
    schedule: list of [x, y] points with x in [0,1] sorted ascending.
    """
    xs = [p[0] for p in schedule]
    ys = [p[1] for p in schedule]
    if step_frac <= xs[0]:
        return ys[0]
    if step_frac >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if step_frac <= xs[i]:
            # linear interpolate
            t = (step_frac - xs[i-1]) / (xs[i] - xs[i-1] + 1e-12)
            return ys[i-1] * (1 - t) + ys[i] * t
    return ys[-1]


# ----------------------- Dummy MMS-style data ------------------------

@dataclass
class DummyConfig:
    T: int = 8
    H: int = 64
    W: int = 64


def mms_fields(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, eta0: float = 0.01):
    """
    Manufactured solution from the plan appendix:
      A_z* = sin(pi x) sin(pi y) cos(pi t)
      B_z* = cos(pi x) cos(pi y) sin(pi t)
      u_x* = 0.3 sin(pi x/2) cos(pi y/2)
      u_y* = -0.3 cos(pi x/2) sin(pi y/2)
      eta* = eta0 (1 + 0.2 sin(pi x) sin(pi y))
    """
    Az = torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.cos(math.pi * t)
    Bz = torch.cos(math.pi * x) * torch.cos(math.pi * y) * torch.sin(math.pi * t)
    ux = 0.3 * torch.sin(0.5 * math.pi * x) * torch.cos(0.5 * math.pi * y)
    uy = -0.3 * torch.cos(0.5 * math.pi * x) * torch.sin(0.5 * math.pi * y)
    eta = eta0 * (1.0 + 0.2 * torch.sin(math.pi * x) * torch.sin(math.pi * y))
    return Az, Bz, ux, uy, eta


def sample_coords_grid(T: int, P: int, device: torch.device) -> torch.Tensor:
    """
    Sample T time-slices with P spatial points each. Return coords [T,P,3] in [-1,1]^3.
    """
    xy = torch.rand(T, P, 2, device=device) * 2.0 - 1.0  # [-1,1]^2
    t = torch.linspace(-1.0, 1.0, T, device=device)[:, None, None].expand(T, P, 1)
    coords = torch.cat([xy, t], dim=-1)  # [T,P,3]
    coords.requires_grad_(True)
    return coords


def make_dummy_batch(cfg_dummy: DummyConfig, P: int, device: torch.device):
    """
    Returns:
      coords_bt: [B(=1), T, P, 3]
      feats_star: dict of tensors with ground truth fields [B, T, P, 1]
      labels: tensor [B, n_horizons] with synthetic labels
      observed_mask: [B, T] (all True)
      pil_mask_np: numpy array [H,W] for collocation biasing (here None to use uniform)
    """
    T, H, W = cfg_dummy.T, cfg_dummy.H, cfg_dummy.W
    coords = sample_coords_grid(T, P, device)  # [T,P,3]
    x, y, t = coords[..., 0:1], coords[..., 1:2], coords[..., 2:3]
    Azs, Bzs, uxs, uys, etas = mms_fields(x, y, t)  # [T,P,1] each (broadcasted)

    # Pack shapes
    pack = lambda z: z.unsqueeze(0)  # [1,T,P,1]
    feats_star = {
        "A_z": pack(Azs),
        "B_z": pack(Bzs),
        "u_x": pack(uxs),
        "u_y": pack(uys),
        "eta": pack(etas),
    }
    coords_bt = coords.unsqueeze(0)  # [1,T,P,3]
    observed_mask = torch.ones(1, T, dtype=torch.bool, device=device)

    # Synthetic labels: use mean |Bz| over observed frames as proxy → > thresh => positive
    bz_mean = Bzs.abs().mean().detach()
    prob = (bz_mean / (bz_mean + 1.0)).clamp(0.05, 0.95).item()
    rng = np.random.RandomState(1234)
    y = torch.tensor([[
        float(rng.rand() < prob),
        float(rng.rand() < prob),
        float(rng.rand() < prob),
    ]], device=device)

    # No PIL mask for dummy
    pil_mask_np = None
    return coords_bt, feats_star, y, observed_mask, pil_mask_np


# ------------------------------- Training -------------------------------

def build_model(cfg: dict, device: torch.device):
    model = PINNBackbone(
        hidden=int(cfg["model"]["hidden"]),
        layers=int(cfg["model"]["layers"]),
        max_log2_freq=int(cfg["model"]["fourier"]["max_log2_freq"]),
        learn_eta=bool(cfg["model"]["learn_eta"]),
    ).to(device)
    clf = ClassifierHead(
        hidden=int(cfg["classifier"]["hidden"]),
        dropout=float(cfg["classifier"]["dropout"]),
        horizons=tuple(cfg["classifier"]["horizons"]),
    ).to(device)
    return model, clf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, help="Path to training YAML (e.g., configs/train_pinn_smoke_test.yaml)")
    args = p.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cpu"))
    use_amp = bool(cfg["train"].get("amp", False)) and (device.type == "cuda")

    # Models
    model, clf = build_model(cfg, device)

    # Physics module (can be disabled via cfg.physics.enable)
    physics_cfg = cfg.get("physics", {})
    eta_cfg = cfg.get("eta", {})
    phys_mod = WeakFormInduction2p5D(
        eta_bounds=(float(eta_cfg.get("min", 1e-4)), float(eta_cfg.get("max", 1.0))),
        use_resistive=bool(physics_cfg.get("resistive", False)),
        include_boundary=bool(physics_cfg.get("boundary_terms", False)),
        tv_eta=float(eta_cfg.get("tv_weight", 1e-3)),
    ).to(device)

    # Optimizer
    opt = torch.optim.AdamW(list(model.parameters()) + list(clf.parameters()), lr=float(cfg["train"]["lr"]))

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Config shorthands
    steps = int(cfg["train"]["steps"])
    batch_size = int(cfg["train"]["batch_size"])
    grad_clip = float(cfg["train"]["grad_clip"])
    log_every = int(cfg["train"].get("log_every", 20))
    eval_every = int(cfg["train"].get("eval_every", 0)) or None

    # Fourier ramp settings
    ff_max_log2 = int(cfg["model"]["fourier"]["max_log2_freq"])
    ff_ramp_frac = float(cfg["model"]["fourier"]["ramp_frac"])

    # Collocation
    n_colloc = int(cfg["collocation"]["n_max"])
    alpha_start = float(cfg["collocation"]["alpha_start"])
    alpha_end = float(cfg["collocation"]["alpha_end"])
    clip_q = float(cfg["collocation"]["impw_clip_quantile"])

    # Loss weights
    w_cls = float(cfg["loss_weights"]["cls"])
    w_data = float(cfg["loss_weights"]["data"])

    # Dummy data cfg (default path)
    dummy_cfg = DummyConfig(
        T=int(cfg["data"].get("T", 8)),
        H=int(cfg["data"].get("H", 64)),
        W=int(cfg["data"].get("W", 64)),
    )

    # Main loop
    model.train(); clf.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        step_frac = step / float(max(1, steps))
        # Fourier feature annealing: map step_frac∈[0,ff_ramp_frac] → alpha∈[0,1]
        alpha = min(1.0, step_frac / max(1e-6, ff_ramp_frac))
        model.set_fourier_alpha(alpha)

        # Lambda_phys schedule
        lam_phys = ramp_value(step_frac, physics_cfg.get("lambda_phys_schedule", [
            [0.00, 0.0], [0.30, 0.0], [0.80, 3.0], [1.00, 5.0]
        ]))
        physics_enabled = bool(physics_cfg.get("enable", False)) and lam_phys > 0.0

        # Mix PIL/uniform schedule alpha
        alpha_colloc = alpha_start + (alpha_end - alpha_start) * step_frac

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            # -------- batch (dummy) --------
            # We keep B=1 for now; configs use small batch_size anyway.
            coords_bt, feats_star, labels, observed_mask, pil_mask_np = make_dummy_batch(dummy_cfg, P=1024, device=device)
            B, T, Pts, _ = coords_bt.shape

            # Forward backbone on classifier sampling points
            coords_flat = coords_bt.reshape(B * T * Pts, 3)
            out = model(coords_flat)
            # unpack to [B,T,P,1]
            def rs(z): return z.reshape(B, T, Pts, 1)
            feats = {k: rs(v) for k, v in out.items() if k in ("A_z","B_z","u_x","u_y")}

            # --- classification loss (multi-horizon logits) ---
            logits = clf(feats, observed_mask=observed_mask)   # [B, n_hor]
            # naive BCE with hard labels; for real data, swap to focal/weighted
            loss_cls = F.binary_cross_entropy_with_logits(logits, labels)

            # --- data loss (L1 on Bz vs GT on observed frames only) ---
            bz_pred = feats["B_z"]  # [B,T,P,1]
            bz_star = feats_star["B_z"].to(device)
            mask = observed_mask[..., None, None].float()  # [B,T,1,1]
            loss_data = torch.mean(torch.abs((bz_pred - bz_star) * mask))

            # --- physics loss (weak form on collocation points) ---
            if physics_enabled:
                coords_c, p_pdf = mix_pil_uniform(dummy_cfg.H, dummy_cfg.W, alpha_colloc, n_colloc, pil_mask_np, device=str(device))
                coords_c.requires_grad_(True)
                w_tilde, clip_thr = clip_and_renorm_importance(p_pdf, clip_quantile=clip_q)
                quad_wts = torch.ones_like(w_tilde)  # GL weights ~1 for random sampling
                eta_mode = "field" if bool(cfg["model"]["learn_eta"]) else "scalar"
                loss_phys, extras = phys_mod(
                    model=model,
                    coords=coords_c,
                    quad_wts=quad_wts,
                    imp_weights=w_tilde,
                    eta_mode=eta_mode,
                    eta_scalar=float(cfg["model"].get("eta_scalar", 0.01)),
                )
            else:
                loss_phys, clip_thr = torch.tensor(0.0, device=device), 0.0

            loss = w_cls * loss_cls + w_data * loss_data + float(lam_phys) * loss_phys

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(clf.parameters()), grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % log_every == 0 or step == 1:
            dt = time.time() - t0
            lr = opt.param_groups[0]["lr"]
            msg = (f"[{step:05d}/{steps}] loss={loss.item():.4f} "
                   f"(cls={loss_cls.item():.4f}, data={loss_data.item():.4f}, phys={float(loss_phys):.4f}) "
                   f"alpha_ff={alpha:.2f} lam_phys={lam_phys:.2f} lr={lr:.2e} dt={dt:.1f}s")
            if physics_enabled:
                msg += f" imp_clip≤{clip_thr:.2e} ESS≈{ess(torch.ones(n_colloc,1,device=device)):.1f}"
            print(msg)
            t0 = time.time()

        # (Optional) cheap eval hooks could go here; for now keep loop tight.

    print("Training run complete. Tip: switch to configs/train_pinn_real.yaml once your data windows are ready.")


if __name__ == "__main__":
    main()


