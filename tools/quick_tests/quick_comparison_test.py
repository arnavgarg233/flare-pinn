#!/usr/bin/env python3
"""
Quick comparison test for 4 PINN configurations:
1. MLP + No Physics
2. MLP + Physics
3. Hybrid + No Physics
4. Hybrid + Physics

Runs short CPU tests (~3-5 min each) to verify everything works.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time
from dataclasses import dataclass
import numpy as np
import torch
import yaml

from src.models.pinn import PINNConfig, PINNModel, HybridPINNModel
from src.train import DummyPINNDataset, setup_logging
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comparison_test")


@dataclass
class TestResult:
    """Results from a quick test run."""
    config_name: str
    model_type: str
    physics_enabled: bool
    steps: int
    final_loss: float
    final_loss_cls: float
    final_loss_data: float
    final_loss_phys: float
    tss_6h: float
    tss_12h: float
    tss_24h: float
    prauc_6h: float
    prauc_12h: float
    prauc_24h: float
    train_time_sec: float
    success: bool
    error_msg: str = ""


def create_test_config(model_type: str, physics_enabled: bool) -> PINNConfig:
    """Create a minimal config for quick testing."""
    config_dict = {
        "seed": 42,
        "device": "cpu",
        "model": {
            "model_type": model_type,
            "hidden": 128,
            "layers": 4,
            "learn_eta": False,
            "eta_scalar": 0.01,
            "fourier": {
                "max_log2_freq": 3,
                "ramp_frac": 0.5
            }
        },
        "classifier": {
            "hidden": 64,
            "dropout": 0.1,
            "horizons": [6, 12, 24],
            "loss_type": "focal",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "pos_weight": None
        },
        "physics": {
            "enable": physics_enabled,
            "resistive": False,
            "boundary_terms": False,
            "lambda_phys_schedule": [
                [0.0, 0.0],
                [0.5, 0.0],
                [0.7, 0.5],
                [1.0, 1.0]
            ]
        },
        "eta": {
            "min": 1.0e-4,
            "max": 1.0,
            "tv_weight": 1.0e-3
        },
        "loss_weights": {
            "cls": 1.0,
            "data": 1.0,
            "curl_consistency": 0.0
        },
        "collocation": {
            "n_max": 2048,
            "alpha_start": 0.5,
            "alpha_end": 0.8,
            "impw_clip_quantile": 0.99
        },
        "train": {
            "steps": 150,  # Quick test
            "batch_size": 1,
            "lr": 1.0e-3,
            "grad_clip": 1.0,
            "amp": False,
            "log_every": 50,
            "eval_every": 150,
            "checkpoint_every": 999999,
            "checkpoint_dir": None
        },
        "data": {
            "use_real": False,
            "target_size": 64,
            "input_hours": 48,
            "P_per_t": 256,
            "pil_top_pct": 0.15,
            "dummy_T": 4,
            "dummy_H": 64,
            "dummy_W": 64,
            "dummy_num_samples": 64
        }
    }
    
    return PINNConfig(**config_dict)


def compute_metrics(probs: np.ndarray, labels: np.ndarray):
    """Compute TSS and PR-AUC for each horizon."""
    from src.models.eval.metrics import sweep_tss, pr_auc
    
    n_horizons = probs.shape[1]
    tss_vals = []
    prauc_vals = []
    
    for h in range(n_horizons):
        y_true = labels[:, h]
        y_prob = probs[:, h]
        
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            tss_vals.append(0.0)
            prauc_vals.append(0.0)
        else:
            _, tss = sweep_tss(y_true, y_prob, n=128)
            pa = pr_auc(y_true, y_prob)
            tss_vals.append(tss)
            prauc_vals.append(pa)
    
    return tss_vals, prauc_vals


def run_single_test(config_name: str, model_type: str, physics_enabled: bool) -> TestResult:
    """Run a single configuration test."""
    logger.info("=" * 80)
    logger.info(f"Testing: {config_name}")
    logger.info(f"  Model: {model_type}, Physics: {physics_enabled}")
    logger.info("=" * 80)
    
    try:
        # Create config
        cfg = create_test_config(model_type, physics_enabled)
        
        # Create model
        device = torch.device("cpu")
        if model_type == "hybrid":
            model = HybridPINNModel(cfg, encoder_in_channels=1).to(device)
        else:
            model = PINNModel(cfg).to(device)
        
        # Create dataset and dataloader
        dataset = DummyPINNDataset(cfg, num_samples=cfg.data.dummy_num_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
        
        # Training loop
        model.train()
        t_start = time.time()
        
        all_probs = []
        all_labels = []
        final_metrics = {}
        
        step = 0
        for epoch in range(100):  # Max epochs
            for batch in dataloader:
                if step >= cfg.train.steps:
                    break
                
                step += 1
                
                # Move to device
                coords = batch["coords"].to(device)
                gt_bz = batch["gt_bz"].to(device)
                observed_mask = batch["observed_mask"].to(device)
                labels = batch["labels"].to(device)
                pil_mask = batch.get("pil_mask")
                frames = batch.get("frames")
                if frames is not None:
                    frames = frames.to(device)
                
                # Update curriculum
                frac = min(1.0, step / max(1, cfg.train.steps))
                model.set_train_frac(frac)
                
                # Forward pass
                optimizer.zero_grad(set_to_none=True)
                
                # Process batch (batch_size=1)
                coords_i = coords[0]
                gt_bz_i = gt_bz[0]
                observed_mask_i = observed_mask[0]
                labels_i = labels[0:1]
                pil_mask_i = pil_mask[0].cpu().numpy() if isinstance(pil_mask, torch.Tensor) else pil_mask
                
                forward_kwargs = {
                    "coords": coords_i,
                    "gt_bz": gt_bz_i,
                    "observed_mask": observed_mask_i,
                    "labels": labels_i,
                    "pil_mask": pil_mask_i,
                    "mode": "train"
                }
                
                if model_type == "hybrid" and frames is not None:
                    forward_kwargs["frames"] = frames[0]
                
                out = model(**forward_kwargs)
                
                # Backward pass
                if model_type == "hybrid" and cfg.physics.enable and out.lambda_phys > 0:
                    # Separate backward for hybrid with physics
                    data_cls_loss = (
                        cfg.loss_weights.cls * out.loss_cls +
                        cfg.loss_weights.data * out.loss_data
                    )
                    data_cls_loss.backward(retain_graph=True)
                    out.loss_phys.backward()
                else:
                    out.loss_total.backward()
                
                # Gradient clipping
                if cfg.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                
                optimizer.step()
                
                # Collect metrics
                all_probs.append(out.probs.detach().cpu().numpy())
                all_labels.append(labels_i.detach().cpu().numpy())
                
                # Log
                if step % cfg.train.log_every == 0:
                    logger.info(
                        f"  [Step {step:03d}/{cfg.train.steps}] "
                        f"loss={out.loss_total.item():.4f} "
                        f"(cls={out.loss_cls.item():.4f}, "
                        f"data={out.loss_data.item():.4f}, "
                        f"phys={out.loss_phys.item():.4f}) | "
                        f"lam_phys={out.lambda_phys:.2f}"
                    )
                
                # Store final metrics
                if step == cfg.train.steps:
                    final_metrics = {
                        "loss_total": float(out.loss_total.item()),
                        "loss_cls": float(out.loss_cls.item()),
                        "loss_data": float(out.loss_data.item()),
                        "loss_phys": float(out.loss_phys.item())
                    }
            
            if step >= cfg.train.steps:
                break
        
        train_time = time.time() - t_start
        
        # Compute final metrics
        probs_all = np.concatenate(all_probs, axis=0)
        labels_all = np.concatenate(all_labels, axis=0)
        tss_vals, prauc_vals = compute_metrics(probs_all, labels_all)
        
        logger.info(f"  Final Metrics:")
        logger.info(f"    6h:  TSS={tss_vals[0]:.3f}, PR-AUC={prauc_vals[0]:.3f}")
        logger.info(f"    12h: TSS={tss_vals[1]:.3f}, PR-AUC={prauc_vals[1]:.3f}")
        logger.info(f"    24h: TSS={tss_vals[2]:.3f}, PR-AUC={prauc_vals[2]:.3f}")
        logger.info(f"  Training time: {train_time:.1f}s")
        logger.info("  ✓ SUCCESS")
        
        return TestResult(
            config_name=config_name,
            model_type=model_type,
            physics_enabled=physics_enabled,
            steps=cfg.train.steps,
            final_loss=final_metrics["loss_total"],
            final_loss_cls=final_metrics["loss_cls"],
            final_loss_data=final_metrics["loss_data"],
            final_loss_phys=final_metrics["loss_phys"],
            tss_6h=tss_vals[0],
            tss_12h=tss_vals[1],
            tss_24h=tss_vals[2],
            prauc_6h=prauc_vals[0],
            prauc_12h=prauc_vals[1],
            prauc_24h=prauc_vals[2],
            train_time_sec=train_time,
            success=True
        )
    
    except Exception as e:
        logger.error(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        return TestResult(
            config_name=config_name,
            model_type=model_type,
            physics_enabled=physics_enabled,
            steps=0,
            final_loss=0.0,
            final_loss_cls=0.0,
            final_loss_data=0.0,
            final_loss_phys=0.0,
            tss_6h=0.0,
            tss_12h=0.0,
            tss_24h=0.0,
            prauc_6h=0.0,
            prauc_12h=0.0,
            prauc_24h=0.0,
            train_time_sec=0.0,
            success=False,
            error_msg=str(e)
        )


def print_summary_table(results: list[TestResult]):
    """Print a nice summary table."""
    print("\n" + "=" * 120)
    print("COMPARISON TEST RESULTS")
    print("=" * 120)
    print(f"{'Config':<25} {'Status':<10} {'Loss':<10} {'TSS@6h':<10} {'TSS@12h':<10} {'TSS@24h':<10} {'Time(s)':<10}")
    print("-" * 120)
    
    for r in results:
        status = "✓ PASS" if r.success else "✗ FAIL"
        print(
            f"{r.config_name:<25} {status:<10} {r.final_loss:<10.4f} "
            f"{r.tss_6h:<10.3f} {r.tss_12h:<10.3f} {r.tss_24h:<10.3f} {r.train_time_sec:<10.1f}"
        )
    
    print("=" * 120)
    
    # Summary
    passed = sum(1 for r in results if r.success)
    total = len(results)
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All configurations working! Ready for GPU training.")
    else:
        print("✗ Some configurations failed. Review errors above.")
    
    print()


def main():
    """Run all 4 comparison tests."""
    logger.info("Starting quick comparison test on CPU")
    logger.info("Testing 4 configurations (150 steps each, ~3-5 min per test)")
    
    test_configs = [
        ("1. MLP (No Physics)", "mlp", False),
        ("2. MLP + Physics", "mlp", True),
        ("3. Hybrid (No Physics)", "hybrid", False),
        ("4. Hybrid + Physics", "hybrid", True),
    ]
    
    results = []
    total_start = time.time()
    
    for config_name, model_type, physics_enabled in test_configs:
        result = run_single_test(config_name, model_type, physics_enabled)
        results.append(result)
        print()  # Spacing
    
    total_time = time.time() - total_start
    
    # Print summary
    print_summary_table(results)
    logger.info(f"Total test time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()

