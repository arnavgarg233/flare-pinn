# PINN models package
from .config import (
    PINNConfig,
    ModelConfig,
    ClassifierConfig,
    PhysicsConfig,
    EtaConfig,
    LossWeightsConfig,
    CollocationConfig,
    TrainConfig,
    DataConfig,
    FourierConfig,
)
from .core import PINNBackbone, ClassifierHead, B_perp_from_Az, FourierFeatures
from .losses import focal_loss, bce_logits, l1_data, curl_consistency_l1, interp_schedule
from .physics import WeakFormInduction2p5D
from .collocation import mix_pil_uniform, clip_and_renorm_importance, ess, sample_xy_from_mask
from .model import PINNModel, PINNOutput
from .hybrid_model import HybridPINNModel, HybridPINNOutput

__all__ = [
    # Config
    "PINNConfig",
    "ModelConfig", 
    "ClassifierConfig",
    "PhysicsConfig",
    "EtaConfig",
    "LossWeightsConfig",
    "CollocationConfig",
    "TrainConfig",
    "DataConfig",
    "FourierConfig",
    # Core models
    "PINNBackbone",
    "ClassifierHead",
    "B_perp_from_Az",
    "FourierFeatures",
    "PINNModel",
    "PINNOutput",
    # Hybrid CNN-PINN
    "HybridPINNModel",
    "HybridPINNOutput",
    # Losses
    "focal_loss",
    "bce_logits",
    "l1_data",
    "curl_consistency_l1",
    "interp_schedule",
    # Physics
    "WeakFormInduction2p5D",
    # Collocation
    "mix_pil_uniform",
    "clip_and_renorm_importance",
    "ess",
    "sample_xy_from_mask",
]
