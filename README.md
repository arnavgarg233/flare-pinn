Physics-Informed Neural Network for Solar Flare Prediction

A deep learning framework combining physics-informed neural networks (PINNs) with magnetohydrodynamics (MHD) constraints for operational solar flare forecasting.

## Overview

This repository implements a hybrid architecture that integrates data-driven learning with physical constraints from solar magnetohydrodynamics. The model forecasts flare probability at 6h, 12h, and 24h horizons using HMI/SHARP magnetogram data.

**Key Features:**
- Physics-informed loss terms enforcing MHD equations (∇·B = 0, ∇×B = μ₀J)
- Temporal encoder with GRU architecture for time-series modeling
- Class-balanced focal loss for extreme class imbalance
- Deterministic training and evaluation pipeline
- Chronological train/validation/test split preventing temporal leakage

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU) or MPS (for Apple Silicon)

### Environment Setup

```bash
# Create conda environment
conda create -n flare-pinn python=3.9
conda activate flare-pinn

# Install PyTorch (adjust for your system)
# For CUDA:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# For Apple Silicon:
conda install pytorch torchvision torchaudio -c pytorch

# Install dependencies
pip install -r requirements.txt
```

### JSOC Account Setup

Data download requires a JSOC account (free registration at http://jsoc.stanford.edu/ajax/register_email.html):

```bash
export JSOC_EMAIL="your_email@institution.edu"
```

## Data Pipeline

The data pipeline consists of five sequential scripts that download and process HMI/SHARP magnetogram data:

### 1. Fetch Flare Events

Download GOES flare catalog from HEK:

```bash
python data_scripts/fetch_flares.py
```

**Output:** `data/raw/goes_flares_hek.csv` (GOES X/M/C class events with timestamps and NOAA active regions)

### 2. Bootstrap HARP-NOAA Mapping

Create mapping between HARP numbers and NOAA active regions:

```bash
python data_scripts/bootstrap_harp_mapping.py
```

**Output:** `data/interim/harp_to_noaa.parquet` (temporal mapping linking HARP IDs to NOAA ARs)

### 3. Download SHARP Magnetograms

Download HMI SHARP CEA magnetogram data from JSOC (requires `JSOC_EMAIL`):

```bash
python data_scripts/download_sharp_cea.py
```

**Output:** `data/raw/sharp_cea/` (NPZ files containing Bx, By, Bz components at 12-minute cadence)

**Note:** This step is slow (~several hours for full 2011-2017 dataset). Progress is logged to `data/raw/download_progress.txt`.

### 4. Create Rolling Windows

Generate 48-hour rolling windows with flare labels:

```bash
python data_scripts/create_windows.py
```

**Output:** `data/interim/windows_all_2011_2017.parquet` (time windows with labels for 6h/12h/24h forecasting horizons)

### 5. Consolidate Frames

Bundle NPZ files per HARP for fast I/O during training:

```bash
python tools/consolidate_frames.py
```

**Output:** `~/flare_data/consolidated/` (per-HARP bundles with all temporal frames)

### Data Split

After running the pipeline, split the data chronologically:

```python
import pandas as pd

# Load full dataset
df = pd.read_parquet("data/interim/windows_all_2011_2017.parquet")
df = df.sort_values('t0', kind='stable').reset_index(drop=True)

# 80/5/15 split (optimized for M-class flare statistical power)
n_train = int(len(df) * 0.80)
n_val = int(len(df) * 0.05)

train_df = df.iloc[:n_train]
val_df = df.iloc[n_train:n_train+n_val]
test_df = df.iloc[n_train+n_val:]

# Save splits
train_df.to_parquet("data/interim/windows_train_80.parquet")
val_df.to_parquet("data/interim/windows_val_5.parquet")
test_df.to_parquet("data/interim/windows_test_15.parquet")

# Combined train+val for training with internal validation split
train_val_df = pd.concat([train_df, val_df])
train_val_df.to_parquet("data/interim/windows_train_val_8005.parquet")
```

## Training

### CNN Baseline (40k steps)

Train the CNN baseline model (no physics constraints):

```bash
./run_benchmark.sh
```

Expected: **TSS=0.84** at 40k (5% validation set)

**Hyperparameters:**
- Learning rate: 2e-4 (cosine decay, 1000 step warmup)
- Batch size: 2 (effective 16 with gradient accumulation = 8)
- Training steps: 40,000
- Encoder dropout: 0.15
- Classifier dropout: 0.35
- Label smoothing: 0.0
- Class-balanced focal loss (β=0.999, γ=1.5)

**Data configuration:**
- Uses `windows_train_val_8005.parquet` with `val_fraction=0.0588` (internal 80/5 split)
- Chronological split: Train (80%, 30,481 windows), Validation (5%, 1,905 windows, 98 M-class)
- Test set (`windows_test_15.parquet`, 15%, 5,716 windows, 91 M-class) held out completely
- **Rationale:** 80/5/15 split maximizes test set statistical power given limited M-class flares

---

### Physics-Informed Model (PINN)

**Stage 1: CNN Baseline (40k steps)**

Train the baseline CNN without physics constraints:

```bash
./run_benchmark.sh
```

**Expected:** TSS=0.84 at 40k (validation)  
**Checkpoint:** `outputs/checkpoints/benchmark_classifier/checkpoint_step_0040000.pt`

---

**Stage 2: Hyperparameter Sweep (40k → 44k)**

From the 40k CNN baseline, perform Bayesian hyperparameter optimization using W&B sweeps to find optimal physics loss settings. The sweep evaluates on the **validation set only** (no test set access during sweep).

**Sweep parameters explored:**
- `physics_grad_scale`: [0.08, 0.30]
- `lambda_phys_max`: [0.2, 0.4, 0.6] (ramped from 0 at 40k)
- `causal_decay`: [0.85, 1.0, 1.15]
- `gradnorm_alpha`: [0.0, 1.0]
- `lr`: [5e-6, 1e-5, 2e-5]

**Best hyperparameters (found via sweep):**
- `physics_grad_scale`: 0.166
- `lambda_phys`: ramped from 0 at 40k → 0.06 at 44k
- `causal_decay`: 1.0
- `gradnorm_alpha`: 1.0
- `lr`: 1e-5
- `seed`: 1234

**Sweep command:**
```bash
wandb sweep configs/wandb_sweep.yaml
wandb agent <sweep_id>
```

---

**Stage 3: Final Model Training (40k → 46k)**

Using the best hyperparameters from the sweep, train the final model with seed 1234:

```bash
python src/train.py \
  --config src/configs/pinn_40k_to_60k.yaml \
  --resume outputs/checkpoints/benchmark_classifier/checkpoint_step_0040000.pt
```

**Final checkpoint:** `outputs/checkpoints/pinn_40k_to_60k/checkpoint_step_0046000.pt`

---

**Test Set Evaluation:**

After training, evaluate the final 46k checkpoint on the held-out test set:

```bash
python tools/validate_checkpoint.py \
  --config src/configs/pinn_40k_to_60k.yaml \
  --checkpoint outputs/checkpoints/pinn_40k_to_60k/checkpoint_step_0046000.pt \
  --data data/interim/windows_test_15.parquet \
  --use-ema
```

**Final Performance:**
- **24h TSS: 0.812** (test set)
- **Improvement: +3.0%** over CNN baseline (0.782)
- **Beats SOTA** (0.801) by +1.1%

## Evaluation

### Final Model Results

**CNN Baseline (40k steps):**
- Checkpoint: `outputs/checkpoints/benchmark_classifier/checkpoint_step_0040000.pt`
- Test results: `outputs/checkpoints/benchmark_classifier/validation_results/checkpoint_step_0040000_test_15pct.npz`
- Test TSS (24h): 0.782
- Confusion matrices: `final_results/confusion_matrices/cm_CNN_Baseline_{6h,12h,24h}.png`

**PINN Final (46k steps):**
- Checkpoint: `outputs/checkpoints/pinn_40k_to_60k/checkpoint_step_0046000.pt`
- Test results: `outputs/checkpoints/pinn_40k_to_60k/validation_results/checkpoint_step_0046000_validation.npz`
- **Test TSS (24h): 0.812** 🎯
- **Improvement: +3.0% over CNN baseline**
- Confusion matrices: `final_results/confusion_matrices/cm_PINN_46k_{6h,12h,24h}.png`

---

### Validate Checkpoint

Evaluate a trained checkpoint on validation or test set:

```bash
python tools/validate_checkpoint.py \
  --config src/configs/benchmark_classifier.yaml \
  --checkpoint outputs/checkpoints/benchmark_classifier/checkpoint_step_0040000.pt \
  --use-ema
```

**Notes:**
- Modify the config's `windows_parquet` field to point to test set for final evaluation
- Use `--use-ema` flag to evaluate exponential moving average weights
- Outputs detailed metrics including TSS, PR-AUC, Brier Score, ECE at optimal thresholds

### Confusion Matrix

Generate confusion matrix and diagnostic plots:

```bash
python tools/compute_confusion_matrix.py \
  --config src/configs/benchmark_classifier.yaml \
  --checkpoint outputs/checkpoints/benchmark_classifier/checkpoint_step_0040000.pt \
  --output-dir final_results/confusion_matrices/
```

## Reproducibility

### Determinism

Training uses fixed random seeds for reproducibility:

```python
seed = 1234  # Best seed from hyperparameter sweep
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.mps.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The final model checkpoint provides **exact reproducibility of predictions**.

**Key implementation details:**
- Stable sorting for chronological splits (`df.sort_values('t0', kind='stable')`)
- Per-sample coordinate sampling with deterministic RNG state management
- Consistent validation metrics using `sweep_tss(n=1024)` in both training and validation scripts

### Checkpoint Naming Convention

Checkpoints are saved with step numbers:
- `checkpoint_step_0040000.pt` - Full checkpoint at 40k steps (includes optimizer state)
- `checkpoint_step_0044000.pt` - Full checkpoint at 44k steps (includes optimizer state) result of best W&B sweep
- `checkpoint_step_0046000.pt` - Full checkpoint at 46k steps (includes optimizer state) result of early stopping at ~46k in 44k to 60k training
- Validation results: `validation_results/checkpoint_step_0040000_test_15pct.npz`

## Directory Structure

```
flare-pinn/
├── data/
│   ├── raw/                    # Downloaded SHARP magnetograms
│   └── interim/                # Processed windows and features
├── data_scripts/               # Data pipeline scripts
├── src/
│   ├── configs/                # Model and training configurations
│   ├── data/                   # Dataset implementations
│   ├── models/pinn/            # PINN architecture
│   └── train.py                # Training script
├── tools/                      # Evaluation and utilities
├── outputs/
│   └── checkpoints/            # Saved model checkpoints
└── final_results/              # Published metrics and figures
```

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{yourlastname2025flare,
  title={Physics-Informed Deep Learning for Solar Flare Forecasting},
  author={Your Name and Collaborators},
  journal={TBD},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- HMI/SHARP data provided by NASA/SDO and the HMI science team
- GOES flare catalog from NOAA/SWPC via Heliophysics Event Knowledgebase (HEK)
- JSOC data access infrastructure at Stanford University

