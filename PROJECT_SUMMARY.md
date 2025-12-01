# Physics-Informed Neural Networks for Solar Flare Prediction

## Project Overview

**Objective**: Develop a hybrid deep learning model for predicting M/X-class solar flare occurrence at 6, 12, and 24-hour horizons using physics-informed neural networks (PINNs).


---

## Data Consolidation Status

**Current Progress**: 3,291 / 3,892 HARPs (84.6%) ✓  
**Data Size**: 31 GB (estimated final: ~37 GB)  
**Format**: Vector magnetogram (Bx, By, Bz) at 128×128 resolution  
**Status**: In progress (~20 minutes remaining)

---

## Dataset

### Source Data
- **Instrument**: SDO/HMI SHARP magnetograms
- **Active Regions**: 3,892 HARPs (unique active region patches)
- **Temporal Coverage**: Multi-day sequences per HARP (13-264 timesteps)
- **Spatial Resolution**: 128×128 pixels per frame
- **Magnetic Field Components**: 3 channels (Bx, By, Bz)
- **Total Size**: ~37 GB (consolidated format)

### Input Windows
- **Sequence Length**: 48 hours (49 timesteps at ~1h cadence)
- **Prediction Horizons**: 6h, 12h, 24h ahead
- **Target Events**: M/X-class flare occurrence (binary classification)

### Data Split
- **Method**: HARP-level split (prevents temporal leakage)
- **Validation Fraction**: 15% of HARPs
- **Class Balance**: ~20:1 imbalance (handled via sampling + focal loss)

---

## Model Architecture

### Overview
Hybrid CNN-PINN architecture combining:
1. Convolutional encoder for spatial feature extraction
2. Temporal attention for sequence aggregation
3. Physics-informed coordinate network
4. Multi-task classifier head

### Component Details

#### 1. Spatial Encoder (TinyEncoder)
- **Type**: 6-layer convolutional network
- **Input**: [B, T, 3, 128, 128] (batch, time, channels, height, width)
- **Architecture**:
  - Conv layers: 3→32→64→64→128→128→128 channels
  - Normalization: GroupNorm
  - Activation: GELU
  - Regularization: Dropout (0.1 at layers 2, 4, 6)
- **Output**: [B, T, 128, 32, 32] spatial features per timestep

#### 2. Temporal Encoder
- **Type**: Multi-head attention with positional encoding
- **Components**:
  - Per-frame global pooling
  - Sinusoidal position encoding
  - 4-head attention mechanism
  - Temporal importance weighting
- **Outputs**:
  - Global temporal code: [B, 64]
  - Aggregated spatial features: [B, 256, 32, 32]
  - Attention weights: [T] (interpretable temporal importance)

#### 3. Physics-Informed Network (PINN Backbone)
- **Type**: Fourier-enhanced coordinate MLP
- **Input**: (x, y, t) coordinates + conditioning features
- **Architecture**:
  - Fourier feature encoding (max frequency: 2^5)
  - 10-layer MLP (hidden dimension: 384)
  - FiLM conditioning layers (3, 6, 9)
- **Outputs**:
  - Vector potential: A_z
  - Velocity field: (u_x, u_y)
  - Vertical magnetic field: B_z
- **Derived quantities** (via automatic differentiation):
  - Horizontal field: B_x = -∂A_z/∂y, B_y = +∂A_z/∂x

#### 4. Physics Module
- **Equations**: 2.5D resistive MHD induction equation
  ```
  ∂B_x/∂t = ∂_y(u_x·B_y - u_y·B_x) - ∂_y[η·J_z]
  ∂B_y/∂t = -∂_x(u_x·B_y - u_y·B_x) + ∂_x[η·J_z]
  ∂B_z/∂t = -∇·(B_z·u) + ∇·(η·∇B_z)
  ```
- **Constraints**:
  - Solenoidal condition: ∇·B = 0
  - Curl consistency: ∇×A = B
- **Implementation**: Weak-form residuals with multi-scale test functions
- **Sampling**: Importance-weighted collocation (1,024 points/timestep, weighted toward polarity inversion lines)

#### 5. Feature Extraction
- **Physics Features** (18 total):
  - Base statistics: mean, std, max, polarity balance
  - Flow dynamics: velocity statistics, flux transport
  - Temporal evolution: variance, acceleration, recent changes
  - Field complexity: potential structure, kurtosis
  - Vector quantities: horizontal field, shear, free energy, current helicity
- **Scalar Features** (4 total):
  - R-value, GOES weighted PIL, observation coverage, frame count

#### 6. Classifier Head
- **Input Dimensions**:
  - CNN features: 256
  - Physics features: 18
  - Scalar features: 4
  - Total: 278 features
- **Architecture**:
  - Spatial attention layer
  - MLP: 278 → 256 → 128 → 3
  - Dropout: 0.15
  - Batch normalization
- **Output**: 3 logits (6h, 12h, 24h horizons)

### Model Capacity
- **Total Parameters**: ~5 million
- **Trainable**: ~5 million

---

## Loss Functions

### Multi-Objective Training
```
L_total = λ_cls · L_classification 
        + λ_data · L_data_fitting
        + λ_phys · L_physics
        + λ_curl · L_curl_consistency
```

### Loss Components

#### 1. Classification Loss (L_cls)
- **Type**: Focal loss
- **Formula**: FL(p,y) = -α(1-p)^γ log(p) for y=1
- **Parameters**: α=0.25, γ=2.0
- **Purpose**: Handles severe class imbalance (20:1)

#### 2. Data Fitting Loss (L_data)
- **Type**: L1 norm
- **Formula**: ||B_predicted - B_observed||_1
- **Scope**: Bz component at collocation points

#### 3. Physics Loss (L_phys)
- **Type**: MHD residual minimization
- **Components**: Induction equation residuals for Bx, By, Bz
- **Implementation**: Weak-form with test functions

#### 4. Curl Consistency Loss (L_curl)
- **Type**: L1 norm
- **Formula**: ||∇×A - B_observed||_1
- **Purpose**: Enforce Maxwell's equations

### Loss Scheduling
- **Classification weight**: 1.0 (constant)
- **Data weight**: 1.0 (constant)
- **Physics weight**: 0 → 2.0 (curriculum, ramped over training)
- **Curl weight**: 0.5 (constant)

### Additional Regularization
- **Label smoothing**: 0.05
- **Confidence penalty**: 0.05
- **Weight decay**: 0.01 (AdamW)
- **Dropout**: 0.15 (classifier)

---

## Training Configuration

### Optimization
- **Optimizer**: AdamW
- **Learning rate**: 1e-3 (subject to hyperparameter tuning)
- **Scheduler**: Cosine annealing with warmup
  - Warmup: 2,000 steps
  - Minimum LR: 1e-6
- **Gradient clipping**: 1.0
- **Mixed precision**: Enabled (AMP)

### Training Strategy
- **Batch size**: 8
- **Training steps**: 60,000 (initial), 120,000 (final)
- **Validation frequency**: Every 500 steps
- **Checkpoint frequency**: Every 5,000 steps

### Data Augmentation
- Random horizontal/vertical flips
- Random 90° rotations
- Gaussian noise injection (σ=0.02)

### Sampling Strategy
- **Method**: Class-balanced weighted sampling
- **Positive class multiplier**: 10.0
- **Smoothing factor**: 0.5

### Stabilization Techniques
- Exponential moving average (EMA) of weights (decay=0.999)
- Early stopping (patience=25 epochs, min_delta=0.005)
- Gradient accumulation (for memory-constrained hardware)

---

## Hyperparameter Optimization

### Strategy
- **Method**: Bayesian Optimization with Hyperband early termination (BOHB)
- **Framework**: Weights & Biases Sweeps
- **Trials**: ~39 configurations
- **Trial length**: 3,000-15,000 steps (adaptive)

### Parameters Under Optimization (13 total)

| Parameter | Type | Search Space |
|-----------|------|--------------|
| Learning rate | Continuous | [3e-4, 3e-3] log-uniform |
| Physics λ final | Continuous | [0.5, 3.0] |
| Physics ramp start | Continuous | [0.3, 0.6] |
| Curl consistency weight | Discrete | {0.0, 0.1, 0.2, 0.3, 0.5} |
| Batch size | Discrete | {4, 8, 16} |
| Dropout | Continuous | [0.1, 0.3] |
| Focal gamma | Continuous | [1.5, 3.0] |
| Focal alpha | Continuous | [0.2, 0.35] |
| Warmup steps | Discrete | {500, 1000, 2000, 3000} |
| Label smoothing | Discrete | {0.0, 0.05, 0.1} |
| Positive multiplier | Continuous | [5.0, 20.0] |
| PIL alpha (end) | Continuous | [0.75, 0.95] |
| Fourier ramp fraction | Discrete | {0.3, 0.5, 0.7} |

### Optimization Efficiency
- **Hyperband schedule**: 3k → 9k → 15k steps (successive halving, η=3)
- **Early termination**: Bottom 2/3 of trials stopped at each bracket
- **Expected compute**: ~207,000 total training steps across all trials
- **Expected duration**: 2-4 days (hardware-dependent)

---

## Evaluation Metrics

### Primary Metric
- **TSS** (True Skill Statistic): Primary metric for model selection and comparison

### Secondary Metrics
- **PR-AUC** (Precision-Recall Area Under Curve): Appropriate for imbalanced data
- **Brier Score**: Probabilistic calibration quality
- **ECE** (Expected Calibration Error): Adaptive binning-free calibration measure
- **Threshold Analysis**: TSS at fixed FAR (5%)

### Evaluation Protocol
- Per-horizon metrics (6h, 12h, 24h)
- HARP-level validation split (prevents data leakage)
- Threshold sweeping for optimal TSS

---

## Key Innovations

### 1. Vector Magnetogram Input
- Full 3-component magnetic field (Bx, By, Bz)
- Enables computation of current density, magnetic shear, free energy
- Matches input used by current state-of-the-art models

### 2. Physics-Informed Learning
- First application of PINNs to solar flare prediction
- Enforces 2.5D MHD evolution equations during training
- Weak-form formulation for numerical stability
- Importance-weighted sampling favoring polarity inversion lines

### 3. Hybrid Architecture
- Combines strengths of data-driven (CNN) and physics-based (PINN) approaches
- Temporal attention for adaptive sequence weighting
- Multi-scale feature extraction (learned + physics-derived)

### 4. Comprehensive Feature Set
- 256 learned CNN features
- 18 physics-derived features
- 4 domain-knowledge scalar features
- Total: 278 features for classification

### 5. Production-Ready Training
- Automated hyperparameter optimization
- Calibration techniques (label smoothing, focal loss)
- Stability mechanisms (EMA, gradient clipping)
- Mixed-precision training for efficiency

---

## Implementation Details

### Software Stack
- **Framework**: PyTorch 2.x
- **Data**: NumPy, Pandas, PyArrow (Parquet)
- **Optimization**: W&B Sweeps
- **Automatic Differentiation**: torch.autograd
- **Mixed Precision**: torch.cuda.amp / torch.cpu.amp

### Hardware Considerations
- **Training**: Apple M1/M2 (MPS) or NVIDIA GPU (CUDA)
- **Memory**: 16+ GB recommended
- **Storage**: ~40 GB for consolidated data + checkpoints

### Code Organization
```
src/
├── configs/          # YAML configuration files
├── data/             # Dataset classes and utilities
├── models/pinn/      # Model architecture
│   ├── core.py       # PINN backbone, feature extractors
│   ├── encoder.py    # CNN and temporal encoders
│   ├── physics.py    # MHD physics module
│   ├── losses.py     # Loss functions
│   ├── model.py      # Complete model assembly
│   └── config.py     # Pydantic configuration schemas
├── train.py          # Training script
└── utils/            # Utilities (metrics, training helpers)
tools/
├── consolidate_frames.py  # Data preprocessing
└── wandb_sweep_train.py   # HPO wrapper
```

---

## Experimental Plan

### Phase 1: Data Preparation ✓
- Vector magnetogram consolidation (3-channel format)
- Quality validation (no NaN/Inf, correct shapes)
- Dataset pipeline implementation

### Phase 2: Baseline Testing (Pending)
- Hardware performance measurement
- Initial training run (2,000 steps)
- Convergence and stability verification

### Phase 3: Hyperparameter Optimization (Pending)
- W&B sweep execution (~39 trials)
- Identification of optimal configuration
- Expected duration: 2-4 days

### Phase 4: Final Training (Pending)
- Full training run (120,000 steps) with best hyperparameters
- Model checkpointing and validation
- Expected duration: 1 day

### Phase 5: Evaluation & Analysis (Pending)
- Performance metrics on held-out test set
- Comparison to baseline and SOTA models
- Ablation studies (physics contribution, vector vs scalar input)
- Interpretability analysis (attention weights, physics residuals)

---

## Expected Contributions

1. **Methodological**: First application of physics-informed neural networks to solar flare forecasting
2. **Architectural**: Novel hybrid CNN-PINN design for spatiotemporal prediction
3. **Performance**: Competitive results with state-of-the-art (target: TSS > 0.80)
4. **Interpretability**: Physics-based features and attention mechanisms for explainability

---

## Results

*To be added after experimental completion*

---

## Current Status

**Data Preparation**: 84.6% complete (3,291/3,892 HARPs consolidated)  
**Model Implementation**: Complete ✓  
**Training Infrastructure**: Complete ✓  
**HPO Configuration**: Complete ✓  
**Baseline Testing**: Pending (awaiting data consolidation)  
**Full Experiments**: Pending

**Estimated Time to Results**: 5-7 days from consolidation completion

---

## References

### Methodology
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. Journal of Computational Physics.
- Li, L., et al. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. JMLR.
- Falkner, S., et al. (2018). BOHB: Robust and efficient hyperparameter optimization at scale. ICML.

### Domain Background
- Bobra, M. G., & Couvidat, S. (2015). Solar flare prediction using SDO/HMI vector magnetic field data. ApJ.
- Sun, X., et al. (2022). Predicting solar flares using a deep temporal convolutional network. ApJ.

### Physics
- Priest, E. (2014). Magnetohydrodynamics of the Sun. Cambridge University Press.

