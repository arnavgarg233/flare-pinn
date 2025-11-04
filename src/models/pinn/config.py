# src/models/pinn/config.py
"""
Pydantic configuration models for PINN training.
Provides type-safe, validated configuration management.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class FourierConfig(BaseModel):
    """Fourier feature configuration."""
    max_log2_freq: int = Field(default=5, ge=1, le=10, description="Maximum log2 frequency for Fourier features")
    ramp_frac: float = Field(default=0.5, ge=0.0, le=1.0, description="Fraction of training for frequency ramping")
    
    @field_validator('max_log2_freq')
    @classmethod
    def validate_freq(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_log2_freq must be at least 1")
        return v


class ModelConfig(BaseModel):
    """PINN backbone configuration."""
    model_type: Literal["mlp", "hybrid"] = Field(default="mlp", description="Model architecture: 'mlp' (pure coordinate) or 'hybrid' (CNN+MLP)")
    hidden: int = Field(default=384, ge=64, le=2048, description="Hidden dimension for MLP")
    layers: int = Field(default=10, ge=2, le=20, description="Number of hidden layers")
    learn_eta: bool = Field(default=False, description="Learn spatially-varying resistivity η(x,y,t)")
    eta_scalar: float = Field(default=0.01, ge=1e-6, le=1.0, description="Fixed scalar resistivity if not learning")
    fourier: FourierConfig = Field(default_factory=FourierConfig)


class ClassifierConfig(BaseModel):
    """Flare prediction classifier head configuration."""
    hidden: int = Field(default=256, ge=64, le=1024)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    horizons: tuple[int, ...] = Field(default=(6, 12, 24), description="Prediction horizons in hours")
    loss_type: Literal["bce", "focal"] = Field(default="focal")
    focal_alpha: float = Field(default=0.25, ge=0.0, le=1.0, description="Focal loss alpha (weight for positive class)")
    focal_gamma: float = Field(default=2.0, ge=0.0, le=5.0, description="Focal loss gamma (focusing parameter)")
    pos_weight: Optional[float] = Field(default=None, ge=1.0, description="Positive class weight for BCE")


class PhysicsConfig(BaseModel):
    """Physics-informed loss configuration."""
    enable: bool = Field(default=True, description="Enable physics loss")
    resistive: bool = Field(default=False, description="Include resistive (diffusion) terms")
    boundary_terms: bool = Field(default=False, description="Include boundary integration terms")
    lambda_phys_schedule: list[list[float]] = Field(
        default=[[0.0, 0.0], [0.3, 0.0], [0.8, 2.0], [1.0, 3.0]],
        description="Piecewise-linear schedule [[frac, weight], ...] for physics loss ramp-up"
    )
    
    @field_validator('lambda_phys_schedule')
    @classmethod
    def validate_schedule(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            raise ValueError("lambda_phys_schedule cannot be empty")
        for point in v:
            if len(point) != 2:
                raise ValueError(f"Each schedule point must be [frac, weight], got {point}")
            if not (0.0 <= point[0] <= 1.0):
                raise ValueError(f"Schedule fraction must be in [0,1], got {point[0]}")
        return sorted(v, key=lambda p: p[0])


class EtaConfig(BaseModel):
    """Resistivity configuration."""
    min: float = Field(default=1e-4, ge=1e-10, le=1e-2)
    max: float = Field(default=1.0, ge=1e-2, le=10.0)
    tv_weight: float = Field(default=1e-3, ge=0.0, le=1.0, description="Total variation regularization weight")
    
    @model_validator(mode='after')
    def validate_bounds(self) -> EtaConfig:
        if self.min >= self.max:
            raise ValueError(f"eta.min ({self.min}) must be < eta.max ({self.max})")
        return self


class LossWeightsConfig(BaseModel):
    """Loss term weights."""
    cls: float = Field(default=1.0, ge=0.0, description="Classification loss weight")
    data: float = Field(default=1.0, ge=0.0, description="Data fitting (L1 on Bz) weight")
    curl_consistency: float = Field(default=0.1, ge=0.0, description="Curl consistency weight (optional)")


class CollocationConfig(BaseModel):
    """Collocation point sampling configuration."""
    n_max: int = Field(default=20000, ge=1000, le=100000, description="Maximum collocation points per batch")
    alpha_start: float = Field(default=0.5, ge=0.0, le=1.0, description="Initial PIL bias weight")
    alpha_end: float = Field(default=0.8, ge=0.0, le=1.0, description="Final PIL bias weight")
    impw_clip_quantile: float = Field(default=0.99, ge=0.5, le=0.999, description="Importance weight clipping quantile")
    
    @model_validator(mode='after')
    def validate_alpha_progression(self) -> CollocationConfig:
        if self.alpha_start > self.alpha_end:
            raise ValueError(f"alpha_start ({self.alpha_start}) should be <= alpha_end ({self.alpha_end})")
        return self


class TrainConfig(BaseModel):
    """Training hyperparameters."""
    steps: int = Field(default=50000, ge=100, le=1000000)
    batch_size: int = Field(default=1, ge=1, le=64)
    lr: float = Field(default=1e-3, ge=1e-6, le=1e-1)
    grad_clip: float = Field(default=1.0, ge=0.0, le=10.0)
    amp: bool = Field(default=True, description="Automatic mixed precision")
    log_every: int = Field(default=25, ge=1)
    eval_every: int = Field(default=500, ge=10)
    checkpoint_every: int = Field(default=5000, ge=100)
    checkpoint_dir: Optional[Path] = Field(default=None, description="Directory for checkpoints")
    
    @field_validator('checkpoint_dir', mode='before')
    @classmethod
    def validate_checkpoint_dir(cls, v):
        if v is not None:
            return Path(v)
        return v


class DataConfig(BaseModel):
    """Dataset configuration."""
    use_real: bool = Field(default=False, description="Use real SHARP data (True) or dummy data (False)")
    windows_parquet: Optional[Path] = None
    frames_meta_parquet: Optional[Path] = None
    npz_root: Optional[Path] = None
    target_size: int = Field(default=256, ge=64, le=1024, description="Spatial resolution (pixels)")
    input_hours: int = Field(default=48, ge=6, le=120, description="Input time window (hours)")
    P_per_t: int = Field(default=1024, ge=256, le=8192, description="Points sampled per time slice")
    pil_top_pct: float = Field(default=0.15, ge=0.01, le=0.5, description="Top % of |∇Bz| for PIL mask")
    
    # Dummy data settings
    dummy_T: int = Field(default=8, ge=2, le=48, description="Dummy time steps")
    dummy_H: int = Field(default=64, ge=32, le=256, description="Dummy spatial height")
    dummy_W: int = Field(default=64, ge=32, le=256, description="Dummy spatial width")
    dummy_num_samples: int = Field(default=2048, ge=64, le=10000, description="Dummy dataset size")
    
    @model_validator(mode='after')
    def validate_real_data_paths(self) -> DataConfig:
        if self.use_real:
            if self.windows_parquet is None:
                raise ValueError("windows_parquet is required when use_real=True")
            if self.frames_meta_parquet is None:
                raise ValueError("frames_meta_parquet is required when use_real=True")
            if self.npz_root is None:
                raise ValueError("npz_root is required when use_real=True")
        return self
    
    @field_validator('windows_parquet', 'frames_meta_parquet', 'npz_root', mode='before')
    @classmethod
    def validate_paths(cls, v):
        if v is not None:
            return Path(v)
        return v


class PINNConfig(BaseModel):
    """Complete PINN training configuration."""
    seed: int = Field(default=42, ge=0)
    device: str = Field(default="cuda", pattern="^(cuda|cpu|cuda:[0-9]+)$")
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    eta: EtaConfig = Field(default_factory=EtaConfig)
    loss_weights: LossWeightsConfig = Field(default_factory=LossWeightsConfig)
    collocation: CollocationConfig = Field(default_factory=CollocationConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> PINNConfig:
        """Load configuration from YAML file."""
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                self.model_dump(mode='json', exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False
            )
    
    def model_dump_summary(self) -> dict:
        """Generate human-readable summary of key settings."""
        return {
            "model": f"{self.model.layers}x{self.model.hidden} (fourier={self.model.fourier.max_log2_freq})",
            "classifier": f"{len(self.classifier.horizons)} horizons, {self.classifier.loss_type}",
            "physics": f"enabled={self.physics.enable}, resistive={self.physics.resistive}",
            "training": f"{self.train.steps} steps @ lr={self.train.lr}, batch={self.train.batch_size}",
            "data": f"real={self.data.use_real}, size={self.data.target_size}x{self.data.target_size}, hours={self.data.input_hours}",
        }


