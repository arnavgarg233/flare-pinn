#!/usr/bin/env python3
"""
Visualize inferred photospheric velocity fields for publication figures.

Creates:
1. Case study comparison (flaring vs non-flaring AR)
2. Temporal evolution strip
3. Statistical analysis plots
"""
import argparse
from pathlib import Path
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.ndimage import gaussian_filter

from src.models.pinn import PINNConfig, HybridPINNModel
from src.data.consolidated_dataset import ConsolidatedWindowsDataset
from src.utils.masked_training import load_windows_with_mask


def get_device(requested: str) -> torch.device:
    """Get computing device."""
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def load_model_and_data(
    config_path: Path,
    checkpoint_path: Path,
    data_path: Path,
    device: torch.device,
    use_ema: bool = True,
) -> Tuple[HybridPINNModel, ConsolidatedWindowsDataset]:
    """Load trained model and dataset."""
    print(f"📦 Loading config from {config_path}")
    cfg = PINNConfig.from_yaml(config_path)
    
    print(f"🔥 Loading model from {checkpoint_path}")
    model = HybridPINNModel(cfg)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    if use_ema and "ema_state_dict" in ckpt:
        print("   Using EMA weights (inference mode, strict=False)")
        # EMA state dict contains {'decay': X, 'shadow': {...}}
        if isinstance(ckpt["ema_state_dict"], dict) and "shadow" in ckpt["ema_state_dict"]:
            model.load_state_dict(ckpt["ema_state_dict"]["shadow"], strict=False)
        else:
            model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    elif "model_state_dict" in ckpt:
        print("   Using model weights (inference mode, strict=False)")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        # Direct state dict (no wrapper)
        print("   Loading direct state dict (inference mode, strict=False)")
        model.load_state_dict(ckpt, strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"📊 Loading data from {data_path}")
    windows_df, _ = load_windows_with_mask(data_path)  # Returns (df, mask)
    dataset = ConsolidatedWindowsDataset(
        windows_df=windows_df,
        consolidated_dir=str(cfg.data.consolidated_dir),
        target_px=cfg.data.target_size,
        input_hours=cfg.data.input_hours,
        horizons=list(cfg.classifier.horizons),
        P_per_t=512,
        training=False,
        augment=False,  # No augmentation for visualization
        max_cached_harps=100,
    )
    
    return model, dataset


def extract_physics_fields(
    model: HybridPINNModel,
    sample: dict,
    device: torch.device,
    t_idx: int = -1,  # Default to final timestep
    n_points: int = 2048,
) -> dict:
    """
    Extract physical fields (u, B, eta) at specified timestep.
    
    Returns:
        dict with keys: 'coords', 'u_x', 'u_y', 'B_x', 'B_y', 'B_z', 'eta', 
                        'u_mag', 'B_mag'
    """
    with torch.no_grad():
        # Move sample to device
        frames = sample["frames"].to(device)  # [T, C, H, W]
        scalars = sample["scalars"].unsqueeze(0).to(device)  # [1, F]
        observed_mask = sample["observed_mask"].to(device)  # [T]
        
        T, C, H, W = frames.shape
        if t_idx < 0:
            t_idx = T + t_idx  # Convert negative index
        
        # Extract input Bz magnetogram at this timestep
        # Assuming Bz is the 3rd channel (index 2)
        input_bz = frames[t_idx, 2, :, :].cpu().numpy()  # [H, W]
        
        # Generate dense spatial grid at specified timestep
        x = torch.linspace(-1, 1, int(np.sqrt(n_points)), device=device)
        y = torch.linspace(-1, 1, int(np.sqrt(n_points)), device=device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Normalized time coordinate
        t_norm = 2 * t_idx / (T - 1) - 1  # Map to [-1, 1]
        tt = torch.full_like(xx, t_norm)
        
        # Reshape for model: [T=1, P=N, 3]
        coords_flat = torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3)  # [N, 3]
        coords = coords_flat.unsqueeze(0)  # [T=1, P=N, 3]
        
        # Forward pass - model expects NO batch dimension
        out = model(
            coords=coords,  # [T=1, P=N, 3]
            frames=frames,  # [T, C, H, W]
            scalars=scalars,  # [1, F]
            observed_mask=observed_mask,  # [T]
        )
        
        # Extract fields from model output
        # out.A: [1, N, 3], out.B: [1, N, 3], out.u: [1, N, 2]
        A = out.A.squeeze(0).cpu().numpy()  # [N, 3]
        B = out.B.squeeze(0).cpu().numpy()  # [N, 3]
        u = out.u.squeeze(0).cpu().numpy()  # [N, 2]
        
        # Note: eta is not directly available in output, would need to access field object
        fields = {
            "coords": coords.squeeze(0).cpu().numpy(),  # [N, 3]
            "A_x": A[:, 0],
            "A_y": A[:, 1],
            "A_z": A[:, 2],
            "B_x": B[:, 0],
            "B_y": B[:, 1],
            "B_z": B[:, 2],
            "u_x": u[:, 0],
            "u_y": u[:, 1],
            "eta": np.ones(A.shape[0]) * 0.1,  # Placeholder - not in output
            "input_bz": input_bz,  # Input magnetogram from frames
        }
        
        # Compute magnitudes
        fields["u_mag"] = np.sqrt(fields["u_x"]**2 + fields["u_y"]**2)
        fields["B_mag"] = np.sqrt(fields["B_x"]**2 + fields["B_y"]**2 + fields["B_z"]**2)
        
        # Reshape to 2D grid
        grid_size = int(np.sqrt(n_points))
        for key in ["B_x", "B_y", "B_z", "u_x", "u_y", "eta", "u_mag", "B_mag"]:
            fields[key] = fields[key].reshape(grid_size, grid_size)
        
        fields["x_grid"] = xx.cpu().numpy()
        fields["y_grid"] = yy.cpu().numpy()
        
        return fields


def compute_pil_mask(B_z: np.ndarray, threshold: float = 300.0, width: int = 3) -> np.ndarray:
    """
    Compute Polarity Inversion Line mask.
    
    Args:
        B_z: Vertical magnetic field [H, W]
        threshold: Gauss threshold for PIL (|Bz| < threshold)
        width: Dilation width in pixels
    
    Returns:
        Boolean mask [H, W]
    """
    from scipy.ndimage import binary_dilation
    
    # Denormalize if needed (assuming z-score normalization)
    # Typical Bz std ~ 500 G, so threshold / 500 in normalized units
    if np.abs(B_z).max() < 10:  # Likely normalized
        threshold = threshold / 500.0
    
    pil_core = np.abs(B_z) < threshold
    pil_mask = binary_dilation(pil_core, iterations=width)
    return pil_mask


def plot_case_study_comparison(
    flaring_fields: dict,
    nonflaring_fields: dict,
    output_path: Path,
    dpi: int = 300,
):
    """
    Create 2-panel comparison figure (flaring vs non-flaring).
    
    Figure layout:
    [Panel A: Flaring AR] | [Panel B: Non-Flaring AR]
    - Background: Bz magnetogram
    - Overlay: Velocity quiver + magnitude colormap
    - Contour: PIL boundaries
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (fields, title) in enumerate([
        (flaring_fields, "(a) Flaring Active Region\n24h before M5+ event"),
        (nonflaring_fields, "(b) Non-Flaring Active Region\nComplex but stable")
    ]):
        ax = axes[idx]
        
        # Background: Input Bz magnetogram (grayscale)
        im_bz = ax.imshow(
            fields["input_bz"],
            extent=[-1, 1, -1, 1],
            origin="lower",
            cmap="gray",
            alpha=0.7,
            vmin=-2, vmax=2,  # Normalized z-score
        )
        
        # Overlay: Velocity magnitude (hot colormap) - use adaptive scaling
        u_mag_smooth = gaussian_filter(fields["u_mag"], sigma=1.5)
        # Normalize to show relative patterns
        u_mag_norm = (u_mag_smooth - u_mag_smooth.min()) / (u_mag_smooth.max() - u_mag_smooth.min() + 1e-10)
        im_u = ax.imshow(
            u_mag_norm,
            extent=[-1, 1, -1, 1],
            origin="lower",
            cmap="hot",
            alpha=0.6,
            vmin=0, vmax=1,
        )
        
        # Quiver: Velocity vectors (subsample for clarity) - normalize for visualization
        stride = 4
        x_sub = fields["x_grid"][::stride, ::stride]
        y_sub = fields["y_grid"][::stride, ::stride]
        u_x_sub = fields["u_x"][::stride, ::stride]
        u_y_sub = fields["u_y"][::stride, ::stride]
        
        # Normalize velocity vectors to show direction clearly
        u_mag_sub = np.sqrt(u_x_sub**2 + u_y_sub**2) + 1e-10
        u_x_norm = u_x_sub / u_mag_sub
        u_y_norm = u_y_sub / u_mag_sub
        
        ax.quiver(
            x_sub, y_sub, u_x_norm, u_y_norm,
            color='cyan', alpha=0.7, width=0.003,
            scale=25, scale_units='xy',
        )
        
        # PIL contours removed for cleaner visualization
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("x (normalized)", fontsize=10)
        ax.set_ylabel("y (normalized)", fontsize=10)
        ax.set_aspect('equal')
        
        # Add colorbar for velocity magnitude
        cbar = plt.colorbar(im_u, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$|\mathbf{u}|$ (norm. units)", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved case study figure: {output_path}")
    plt.close()


def plot_temporal_evolution(
    model: HybridPINNModel,
    sample: dict,
    device: torch.device,
    output_path: Path,
    t_indices: list = [-36, -24, -12, -6],  # Hours before t0
    dpi: int = 300,
):
    """
    Create temporal evolution strip showing 4 timesteps.
    
    Top row: Bz magnetograms
    Bottom row: Velocity fields
    """
    T = sample["frames"].shape[0]  # Total timesteps
    
    # Find timesteps with actual data (non-zero Bz)
    valid_indices = []
    for t in range(T):
        bz = sample["frames"][t, 2, :, :]  # Bz component
        if (bz.abs() > 0.01).any():
            valid_indices.append(t)
    
    if len(valid_indices) < 4:
        print(f"⚠️  Only {len(valid_indices)} valid frames, cannot create temporal evolution")
        return
    
    # Select 4 evenly spaced valid timesteps
    step = len(valid_indices) // 4
    t_idx_list = [valid_indices[i * step] for i in range(4)]
    t_hour_list = [t - T for t in t_idx_list]  # Convert to hours before t0
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for col_idx, (t_hour, t_idx) in enumerate(zip(t_hour_list, t_idx_list)):
        fields = extract_physics_fields(model, sample, device, t_idx=t_idx, n_points=4096)
        
        # Top row: Input Bz magnetogram
        ax_bz = axes[0, col_idx]
        ax_bz.imshow(
            fields["input_bz"],
            extent=[-1, 1, -1, 1],
            origin="lower",
            cmap="RdBu_r",
            vmin=-2, vmax=2,
        )
        ax_bz.set_title(f"t = {t_hour:+d}h", fontsize=11, fontweight='bold')
        ax_bz.set_xticks([])
        ax_bz.set_yticks([])
        if col_idx == 0:
            ax_bz.set_ylabel(r"$B_z$ [G]", fontsize=10)
        
        # Bottom row: Velocity field (normalized)
        ax_u = axes[1, col_idx]
        
        # Normalize velocity magnitude for better visualization
        u_mag_norm = (fields["u_mag"] - fields["u_mag"].min()) / (fields["u_mag"].max() - fields["u_mag"].min() + 1e-10)
        
        # Background: velocity magnitude
        im = ax_u.imshow(
            u_mag_norm,
            extent=[-1, 1, -1, 1],
            origin="lower",
            cmap="viridis",
            alpha=0.8,
            vmin=0, vmax=1,
        )
        
        # Quiver (normalized for direction)
        stride = 6
        u_x_sub = fields["u_x"][::stride, ::stride]
        u_y_sub = fields["u_y"][::stride, ::stride]
        u_mag_sub = np.sqrt(u_x_sub**2 + u_y_sub**2) + 1e-10
        
        ax_u.quiver(
            fields["x_grid"][::stride, ::stride],
            fields["y_grid"][::stride, ::stride],
            u_x_sub / u_mag_sub,
            u_y_sub / u_mag_sub,
            color='white', alpha=0.8, width=0.003,
            scale=20, scale_units='xy',
        )
        
        ax_u.set_xticks([])
        ax_u.set_yticks([])
        if col_idx == 0:
            ax_u.set_ylabel(r"$\mathbf{u}$ [km/s]", fontsize=10)
        
        # Colorbar for last column
        if col_idx == 3:
            cbar = plt.colorbar(im, ax=ax_u, fraction=0.046, pad=0.04)
            cbar.set_label(r"$|\mathbf{u}|$", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved temporal evolution figure: {output_path}")
    plt.close()


def plot_statistical_analysis(
    all_flaring_fields: list,
    all_nonflaring_fields: list,
    output_path: Path,
    dpi: int = 300,
):
    """
    Create 3-panel statistical comparison:
    (a) Velocity magnitude distribution
    (b) Flow convergence at PILs
    (c) Temporal acceleration
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Extract statistics
    flaring_u_mag = [f["u_mag"].flatten() for f in all_flaring_fields]
    nonflaring_u_mag = [f["u_mag"].flatten() for f in all_nonflaring_fields]
    
    # Panel (a): Velocity magnitude distributions
    ax = axes[0]
    ax.hist(
        np.concatenate(flaring_u_mag), bins=50, alpha=0.6,
        color='red', label='Flaring', density=True, range=(0, 2)
    )
    ax.hist(
        np.concatenate(nonflaring_u_mag), bins=50, alpha=0.6,
        color='blue', label='Non-Flaring', density=True, range=(0, 2)
    )
    ax.set_xlabel(r"$|\mathbf{u}|$ (normalized units)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(a) Velocity Magnitude Distribution", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Compute medians and statistical test
    med_flaring = np.median(np.concatenate(flaring_u_mag))
    med_nonflaring = np.median(np.concatenate(nonflaring_u_mag))
    stat, pval = stats.mannwhitneyu(
        np.concatenate(flaring_u_mag),
        np.concatenate(nonflaring_u_mag),
        alternative='greater'
    )
    ax.text(
        0.95, 0.95, 
        f"Median (flaring): {med_flaring:.3f}\n"
        f"Median (non-flaring): {med_nonflaring:.3f}\n"
        f"p = {pval:.2e}",
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Panel (b): Flow convergence (placeholder - requires divergence computation)
    ax = axes[1]
    ax.text(
        0.5, 0.5, "Flow Convergence Analysis\n(Requires gradient computation)",
        transform=ax.transAxes, ha='center', va='center', fontsize=10
    )
    ax.set_title("(b) Flow Convergence at PILs", fontsize=12, fontweight='bold')
    
    # Panel (c): Velocity magnitude boxplot comparison
    ax = axes[2]
    bp = ax.boxplot(
        [np.concatenate(nonflaring_u_mag), np.concatenate(flaring_u_mag)],
        labels=['Non-Flaring', 'Flaring'],
        patch_artist=True,
        showfliers=False,
    )
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel(r"$|\mathbf{u}|$ (normalized units)", fontsize=11)
    ax.set_title("(c) Velocity Distribution", fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved statistical analysis figure: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize inferred velocity fields")
    parser.add_argument("--config", type=Path, required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Path to test data parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/velocity_viz"), help="Output directory")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"], help="Device")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use EMA weights")
    parser.add_argument("--flaring-idx", type=int, default=None, help="Index of flaring example")
    parser.add_argument("--nonflaring-idx", type=int, default=None, help="Index of non-flaring example")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of samples for statistics")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device(args.device)
    print(f"🖥️  Using device: {device}")
    
    # Load model and data
    model, dataset = load_model_and_data(
        args.config, args.checkpoint, args.data, device, args.use_ema
    )
    
    print(f"\n📊 Dataset contains {len(dataset)} windows")
    
    # Find good examples if not specified
    if args.flaring_idx is None or args.nonflaring_idx is None:
        print("🔍 Searching for good case study examples...")
        
        # Find flaring and non-flaring indices
        flaring_indices = []
        nonflaring_indices = []
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            # labels tensor: [6h, 12h, 24h]
            label_24h = sample["labels"][2].item()  # 24h is index 2
            if label_24h == 1:
                flaring_indices.append(idx)
            else:
                nonflaring_indices.append(idx)
            
            if len(flaring_indices) >= 10 and len(nonflaring_indices) >= 10:
                break
        
        # Use middle examples (avoid edge cases)
        args.flaring_idx = flaring_indices[len(flaring_indices) // 2] if flaring_indices else 0
        args.nonflaring_idx = nonflaring_indices[len(nonflaring_indices) // 2] if nonflaring_indices else 1
    
    print(f"   Flaring example: index {args.flaring_idx}")
    print(f"   Non-flaring example: index {args.nonflaring_idx}")
    
    # ========== Figure 1: Case Study Comparison ==========
    print("\n🎨 Generating case study comparison...")
    flaring_sample = dataset[args.flaring_idx]
    nonflaring_sample = dataset[args.nonflaring_idx]
    
    flaring_fields = extract_physics_fields(model, flaring_sample, device, t_idx=-1, n_points=4096)
    nonflaring_fields = extract_physics_fields(model, nonflaring_sample, device, t_idx=-1, n_points=4096)
    
    plot_case_study_comparison(
        flaring_fields,
        nonflaring_fields,
        args.output_dir / "velocity_case_study.png"
    )
    
    # ========== Figure 2: Temporal Evolution ==========
    print("\n🎨 Generating temporal evolution...")
    plot_temporal_evolution(
        model,
        flaring_sample,
        device,
        args.output_dir / "velocity_temporal_evolution.png"
    )
    
    # ========== Figure 3: Statistical Analysis ==========
    print("\n🎨 Generating statistical analysis (sampling windows)...")
    
    # Sample fields from multiple windows
    n_samples = min(args.n_samples, len(dataset))
    flaring_fields_list = []
    nonflaring_fields_list = []
    
    for idx in range(n_samples):
        sample = dataset[idx]
        fields = extract_physics_fields(model, sample, device, t_idx=-1, n_points=1024)
        
        # labels tensor: [6h, 12h, 24h]
        label_24h = sample["labels"][2].item()
        if label_24h == 1:
            flaring_fields_list.append(fields)
        else:
            nonflaring_fields_list.append(fields)
        
        if idx % 10 == 0:
            print(f"   Processed {idx}/{n_samples} samples...")
    
    print(f"   Collected {len(flaring_fields_list)} flaring, {len(nonflaring_fields_list)} non-flaring")
    
    if flaring_fields_list and nonflaring_fields_list:
        plot_statistical_analysis(
            flaring_fields_list,
            nonflaring_fields_list,
            args.output_dir / "velocity_statistics.png"
        )
    else:
        print("⚠️  Not enough samples for statistical analysis")
    
    print(f"\n✅ All visualizations saved to: {args.output_dir}")
    print("\n📝 Add these figures to your paper in Section 4.4 or Discussion!")


if __name__ == "__main__":
    main()

