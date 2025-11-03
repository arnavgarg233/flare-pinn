# src/utils/training_utils.py
"""
Training utilities: optimizations, error handling, and helpers.
"""
from __future__ import annotations
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Error Handling & Validation
# ============================================================================

class TrainingError(Exception):
    """Base exception for training-related errors."""
    pass


class ConfigurationError(TrainingError):
    """Invalid configuration."""
    pass


class DataError(TrainingError):
    """Data loading or processing error."""
    pass


class NumericalInstabilityError(TrainingError):
    """Numerical instability detected (NaN, Inf, etc.)."""
    pass


def validate_tensor(
    tensor: torch.Tensor,
    name: str,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> None:
    """
    Validate tensor for NaN/Inf values.
    
    Args:
        tensor: Tensor to validate
        name: Tensor name for error messages
        allow_nan: Whether NaN values are acceptable
        allow_inf: Whether Inf values are acceptable
    
    Raises:
        NumericalInstabilityError: If validation fails
    """
    if not allow_nan and torch.isnan(tensor).any():
        raise NumericalInstabilityError(f"{name} contains NaN values")
    
    if not allow_inf and torch.isinf(tensor).any():
        raise NumericalInstabilityError(f"{name} contains Inf values")


def check_gradient_health(
    model: nn.Module,
    logger: Optional[logging.Logger] = None
) -> dict[str, float]:
    """
    Check gradient health (norm, NaN, etc.).
    
    Returns:
        Dictionary with gradient statistics
    """
    total_norm = 0.0
    num_params = 0
    num_nan = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            num_params += 1
            
            if torch.isnan(p.grad).any():
                num_nan += 1
    
    total_norm = total_norm ** 0.5
    
    stats = {
        'grad_norm': total_norm,
        'num_params': num_params,
        'num_nan': num_nan,
    }
    
    if logger and num_nan > 0:
        logger.warning(f"Found NaN gradients in {num_nan}/{num_params} parameters")
    
    return stats


@contextmanager
def catch_cuda_oom(logger: Optional[logging.Logger] = None):
    """
    Context manager to catch and handle CUDA OOM errors gracefully.
    
    Usage:
        with catch_cuda_oom(logger):
            # training code
    """
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if logger:
                logger.error("CUDA out of memory!")
                logger.error("Try reducing: batch_size, hidden_dim, n_collocation_points")
            torch.cuda.empty_cache()
            raise TrainingError("CUDA OOM - reduce model size or batch size") from e
        else:
            raise


# ============================================================================
# Optimization Utilities
# ============================================================================

class GradientAccumulator:
    """
    Gradient accumulation for effective larger batch sizes.
    
    Usage:
        accumulator = GradientAccumulator(model, optimizer, accum_steps=4)
        
        for step, batch in enumerate(dataloader):
            loss = compute_loss(batch)
            accumulator.step(loss, step)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accum_steps: int = 1,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.accum_steps = accum_steps
        self.scaler = scaler
        self._accum_count = 0
    
    def step(self, loss: torch.Tensor, global_step: int) -> bool:
        """
        Accumulate gradients and step optimizer when ready.
        
        Args:
            loss: Loss tensor (already scaled by 1/accum_steps if needed)
            global_step: Global training step
        
        Returns:
            True if optimizer stepped, False otherwise
        """
        # Scale loss by accumulation steps
        loss = loss / self.accum_steps
        
        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self._accum_count += 1
        
        # Step optimizer when accumulated enough
        if self._accum_count >= self.accum_steps:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_count = 0
            return True
        
        return False


class EarlyStopping:
    """
    Early stopping based on validation metric.
    
    Usage:
        early_stop = EarlyStopping(patience=10, mode='max')
        
        for epoch in range(epochs):
            metric = evaluate()
            if early_stop(metric, epoch):
                break
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_metric = -float('inf') if mode == 'max' else float('inf')
        self.best_step = 0
        self.counter = 0
    
    def __call__(self, metric: float, step: int) -> bool:
        """
        Check if should stop.
        
        Args:
            metric: Current metric value
            step: Current training step
        
        Returns:
            True if should stop, False otherwise
        """
        improved = False
        
        if self.mode == 'max':
            improved = metric > (self.best_metric + self.min_delta)
        else:
            improved = metric < (self.best_metric - self.min_delta)
        
        if improved:
            self.best_metric = metric
            self.best_step = step
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


class ExponentialMovingAverage:
    """
    Maintains exponential moving average of model parameters.
    Often improves generalization for PINNs.
    
    Usage:
        ema = ExponentialMovingAverage(model, decay=0.999)
        
        for batch in dataloader:
            loss.backward()
            optimizer.step()
            ema.update()
        
        # For evaluation
        with ema.average_parameters():
            evaluate()
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    @contextmanager
    def average_parameters(self):
        """Context manager to temporarily use EMA parameters."""
        # Backup current parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        
        try:
            yield
        finally:
            # Restore original parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.backup[name])
            self.backup = {}


# ============================================================================
# Learning Rate Schedulers
# ============================================================================

class WarmupCosineSchedule:
    """
    Learning rate schedule with linear warmup and cosine decay.
    
    Commonly used for transformers and PINNs.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, step: int):
        """Update learning rate."""
        if step < self.warmup_steps:
            # Linear warmup
            lr_mult = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr_mult = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * (
                0.5 * (1.0 + np.cos(np.pi * progress))
            )
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_mult


# ============================================================================
# Memory Optimization
# ============================================================================

def optimize_memory():
    """Apply PyTorch memory optimizations."""
    # Enable TF32 on Ampere GPUs (A100, RTX 3090, etc.)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True
    
    # Enable gradient checkpointing for large models (manual in model code)
    # This trades compute for memory by recomputing activations


def get_memory_stats() -> dict[str, float]:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
    }


def clear_memory_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()



