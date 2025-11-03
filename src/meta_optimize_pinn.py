#!/usr/bin/env python3
"""
Meta-optimization for PINN loss weights and hyperparameters.

Outer loop: Bayesian optimization (Optuna) searches hyperparameter space
Inner loop: PINN trains with those hyperparameters for N steps

Goal: Maximize validation TSS@12h while maintaining physical consistency
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import json

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
import torch

from src.models.pinn import PINNConfig, PINNModel
from src.data.windows_dataset import WindowsDataset
from src.utils.masked_training import load_windows_with_mask
from torch.utils.data import DataLoader


logger = logging.getLogger("meta_pinn")


class PINNObjective:
    """
    Optuna objective for PINN hyperparameter optimization.
    
    Each trial:
    1. Samples hyperparameters from search space
    2. Trains PINN for N steps (inner loop)
    3. Evaluates on validation set
    4. Returns composite metric (TSS + physics penalty)
    """
    
    def __init__(
        self,
        base_config_path: str,
        train_dataset: WindowsDataset,
        val_dataset: WindowsDataset,
        inner_steps: int = 2000,
        device: str = "cuda"
    ):
        """
        Args:
            base_config_path: Path to base YAML config (will be overridden)
            train_dataset: Training dataset
            val_dataset: Validation dataset  
            inner_steps: Number of training steps per trial
            device: Device to train on
        """
        self.base_config = PINNConfig.from_yaml(base_config_path)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.inner_steps = inner_steps
        self.device = device
        
        logger.info(f"Meta-optimization: {len(train_dataset)} train, {len(val_dataset)} val samples")
        logger.info(f"Inner loop: {inner_steps} steps per trial")
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Run one hyperparameter trial.
        
        Returns:
            Composite score (higher is better): TSS@12h - penalty*physics_residual
        """
        # Sample hyperparameters
        config = self._sample_config(trial)
        
        # Create model
        model = PINNModel(config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=config.train.amp)
        
        # Data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        # Training loop (inner loop)
        model.train()
        train_iter = iter(train_loader)
        
        for step in range(self.inner_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            # Training step
            loss_info = self._train_step(model, optimizer, scaler, batch, config, step)
            
            # Intermediate pruning (every 200 steps)
            if step % 200 == 0 and step > 0:
                val_score = self._evaluate(model, config)
                trial.report(val_score, step)
                
                # Prune if not promising
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                logger.info(f"Trial {trial.number} step {step}/{self.inner_steps}: score={val_score:.4f}")
        
        # Final evaluation
        final_score = self._evaluate(model, config)
        
        # Log hyperparameters for this trial
        trial.set_user_attr("final_tss", final_score)
        trial.set_user_attr("config", config.model_dump(mode='json'))
        
        logger.info(f"Trial {trial.number} complete: final_score={final_score:.4f}")
        
        return final_score
    
    def _sample_config(self, trial: optuna.Trial) -> PINNConfig:
        """Sample hyperparameters and create config."""
        config = self.base_config.model_copy(deep=True)
        
        # === Loss Weights (Primary search space) ===
        config.loss_weights.cls = trial.suggest_float("lambda_cls", 0.5, 2.0, log=True)
        config.loss_weights.data = trial.suggest_float("lambda_data", 0.5, 2.0, log=True)
        config.loss_weights.curl_consistency = trial.suggest_float("lambda_curl", 0.01, 0.5, log=True)
        
        # Physics weight schedule (final value)
        lambda_phys_final = trial.suggest_float("lambda_phys_final", 1.0, 10.0, log=True)
        config.physics.lambda_phys_schedule = [
            [0.0, 0.0],
            [0.3, 0.0],
            [0.8, lambda_phys_final * 0.6],
            [1.0, lambda_phys_final]
        ]
        
        # === Collocation Strategy ===
        config.collocation.alpha_start = trial.suggest_float("alpha_start", 0.3, 0.7)
        config.collocation.alpha_end = trial.suggest_float("alpha_end", 0.6, 0.9)
        config.collocation.impw_clip_quantile = trial.suggest_float("impw_clip_quantile", 0.95, 0.995)
        config.collocation.n_max = trial.suggest_int("n_max", 10000, 40000, step=5000)
        
        # === Architecture ===
        config.model.hidden = trial.suggest_categorical("hidden", [256, 384, 512])
        config.model.layers = trial.suggest_int("layers", 8, 12)
        config.model.fourier.max_log2_freq = trial.suggest_int("fourier_freq", 4, 6)
        config.model.fourier.ramp_frac = trial.suggest_float("fourier_ramp", 0.4, 0.7)
        
        # === Training ===
        config.train.lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True)
        config.train.grad_clip = trial.suggest_float("grad_clip", 0.5, 2.0)
        
        # === Regularization ===
        config.eta.tv_weight = trial.suggest_float("tv_weight", 1e-4, 1e-2, log=True)
        
        # === Classifier ===
        config.classifier.dropout = trial.suggest_float("dropout", 0.0, 0.3)
        config.classifier.focal_alpha = trial.suggest_float("focal_alpha", 0.1, 0.5)
        config.classifier.focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
        
        return config
    
    def _train_step(
        self,
        model: PINNModel,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        batch: Dict[str, Any],
        config: PINNConfig,
        step: int
    ) -> Dict[str, float]:
        """Execute one training step."""
        # Move batch to device
        coords = batch["coords"].to(self.device, non_blocking=True).squeeze(0)
        gt_bz = batch["gt_bz"].to(self.device, non_blocking=True).squeeze(0)
        observed_mask = batch["observed_mask"].to(self.device, non_blocking=True).squeeze(0)
        labels = batch["labels"].to(self.device, non_blocking=True)
        pil_mask = batch.get("pil_mask")
        
        # Update progress
        frac = step / max(1, self.inner_steps)
        model.set_train_frac(frac)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=config.train.amp):
            out = model(
                coords=coords,
                gt_bz=gt_bz,
                observed_mask=observed_mask,
                labels=labels,
                pil_mask=pil_mask,
                mode="train"
            )
            loss = out.loss_total
        
        scaler.scale(loss).backward()
        
        if config.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        return {
            "loss_total": loss.item(),
            "loss_cls": out.loss_cls.item(),
            "loss_data": out.loss_data.item(),
            "loss_phys": out.loss_phys.item(),
        }
    
    def _evaluate(self, model: PINNModel, config: PINNConfig) -> float:
        """
        Evaluate model on validation set.
        
        Returns composite score: TSS@12h - alpha * mean_physics_residual
        where alpha balances discrimination vs physical consistency.
        """
        model.eval()
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        all_probs = []
        all_labels = []
        phys_residuals = []
        
        with torch.no_grad():
            for batch in val_loader:
                coords = batch["coords"].to(self.device).squeeze(0)
                gt_bz = batch["gt_bz"].to(self.device).squeeze(0)
                observed_mask = batch["observed_mask"].to(self.device).squeeze(0)
                labels = batch["labels"].to(self.device)
                pil_mask = batch.get("pil_mask")
                
                out = model(
                    coords=coords,
                    gt_bz=gt_bz,
                    observed_mask=observed_mask,
                    labels=labels,
                    pil_mask=pil_mask,
                    mode="eval"
                )
                
                all_probs.append(out.probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                phys_residuals.append(out.loss_phys.item())
        
        # Compute TSS at 12h horizon (index 1)
        probs = np.concatenate(all_probs, axis=0)[:, 1]  # 12h predictions
        labels = np.concatenate(all_labels, axis=0)[:, 1]
        
        tss = self._compute_tss(labels, probs)
        mean_phys = np.mean(phys_residuals)
        
        # Composite score: TSS - penalty*physics
        # Penalty weight: we want physics residual to be ~0, but TSS is primary
        alpha_penalty = 0.1  # Tune this based on residual scale
        score = tss - alpha_penalty * mean_phys
        
        logger.debug(f"Eval: TSS={tss:.4f}, phys={mean_phys:.4f}, score={score:.4f}")
        
        model.train()
        return score
    
    @staticmethod
    def _compute_tss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute TSS (True Skill Statistic) by sweeping thresholds."""
        from src.eval.metrics import sweep_tss
        
        if len(y_true) < 2:
            return 0.0
        
        _, tss = sweep_tss(y_true, y_pred, n=100)
        return tss


def main():
    parser = argparse.ArgumentParser(
        description="Meta-optimize PINN hyperparameters with Bayesian optimization"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Base config YAML (will be overridden by optimization)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=2000,
        help="Training steps per trial (inner loop)"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="pinn_meta_opt",
        help="Optuna study name"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_pinn.db",
        help="Optuna storage (database URL)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    logger.info("=" * 80)
    logger.info("PINN Meta-Optimization (Outer Loop)")
    logger.info("=" * 80)
    logger.info(f"Base config: {args.config}")
    logger.info(f"Trials: {args.n_trials}, Inner steps: {args.inner_steps}")
    logger.info(f"Study: {args.study_name}")
    
    # Load base config
    base_cfg = PINNConfig.from_yaml(args.config)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_df, train_mask = load_windows_with_mask(str(base_cfg.data.windows_parquet))
    train_df = train_df[train_mask].reset_index(drop=True)
    
    # For meta-optimization, split train into train/val (e.g., 80/20)
    n_train = int(0.8 * len(train_df))
    meta_train_df = train_df.iloc[:n_train]
    meta_val_df = train_df.iloc[n_train:]
    
    train_dataset = WindowsDataset(
        windows_df=meta_train_df,
        frames_meta_path=str(base_cfg.data.frames_meta_parquet),
        npz_root=str(base_cfg.data.npz_root),
        target_px=base_cfg.data.target_size,
        input_hours=base_cfg.data.input_hours,
        horizons=list(base_cfg.classifier.horizons),
        P_per_t=base_cfg.data.P_per_t,
        pil_top_pct=base_cfg.data.pil_top_pct,
    )
    
    val_dataset = WindowsDataset(
        windows_df=meta_val_df,
        frames_meta_path=str(base_cfg.data.frames_meta_parquet),
        npz_root=str(base_cfg.data.npz_root),
        target_px=base_cfg.data.target_size,
        input_hours=base_cfg.data.input_hours,
        horizons=list(base_cfg.classifier.horizons),
        P_per_t=base_cfg.data.P_per_t,
        pil_top_pct=base_cfg.data.pil_top_pct,
    )
    
    logger.info(f"Meta-train: {len(train_dataset)}, Meta-val: {len(val_dataset)}")
    
    # Create objective
    objective = PINNObjective(
        base_config_path=args.config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        inner_steps=args.inner_steps,
        device=args.device
    )
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",  # Maximize TSS - physics_penalty
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=400)
    )
    
    logger.info("Starting optimization...")
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Results
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    
    best_trial = study.best_trial
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best score: {best_trial.value:.4f}")
    logger.info(f"Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Save best config
    best_config_path = Path("configs") / f"train_pinn_optimized_{args.study_name}.yaml"
    best_config_path.parent.mkdir(exist_ok=True)
    
    best_config_dict = best_trial.user_attrs.get("config")
    if best_config_dict:
        best_config = PINNConfig(**best_config_dict)
        best_config.to_yaml(best_config_path)
        logger.info(f"Saved best config to: {best_config_path}")
    
    # Save study summary
    df = study.trials_dataframe()
    summary_path = Path("meta_optimization_results") / f"{args.study_name}_trials.csv"
    summary_path.parent.mkdir(exist_ok=True)
    df.to_csv(summary_path, index=False)
    logger.info(f"Saved trial history to: {summary_path}")
    
    # Optuna visualization (if installed)
    try:
        import optuna.visualization as vis
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(summary_path.parent / f"{args.study_name}_history.html")
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(summary_path.parent / f"{args.study_name}_importance.html")
        
        logger.info(f"Saved visualizations to: {summary_path.parent}")
    except ImportError:
        logger.warning("optuna[visualization] not installed; skipping plots")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

