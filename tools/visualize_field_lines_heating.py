#!/usr/bin/env python3
"""
Visualize magnetic field lines and Ohmic heating from physics-informed model.

Field lines: Traced from B field using streamplot
Ohmic heating: η|J|² (energy dissipation rate)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

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


def load_model_and_data(config_path, checkpoint_path, data_path, device):
    """Load trained model and test dataset."""
    cfg = PINNConfig.from_yaml(config_path)
    
    # Initialize model
    model = HybridPINNModel(cfg)
    model = model.to(device)
    model.eval()
    
    # Load checkpoint (EMA weights)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["ema_state_dict"]["shadow"], strict=False)
    
    print(f"✓ Loaded checkpoint from step {ckpt['step']}")
    
    # Load test data
    windows_df, _ = load_windows_with_mask(data_path)
    dataset = ConsolidatedWindowsDataset(
        windows_df=windows_df,
        consolidated_dir=Path("~/flare_data/consolidated").expanduser(),
        target_px=128,
        input_hours=48,
        horizons=[6, 12, 24],
        P_per_t=512,
        training=False,
        augment=False,
    )
    
    print(f"✓ Loaded test set: {len(dataset)} windows")
    
    return model, dataset


def extract_physics_fields(
    model: HybridPINNModel,
    sample: dict,
    device: torch.device,
    t_idx: int = -1,
    n_points: int = 4096,
) -> dict:
    """
    Extract physical fields at specified timestep.
    """
    with torch.no_grad():
        # Move sample to device
        frames = sample["frames"].to(device)  # [T, C, H, W]
        scalars = sample["scalars"].to(device)  # [F]
        observed_mask = sample["observed_mask"].to(device)  # [T]
        
        T, C, H, W = frames.shape
        
        # Create dense grid for field line tracing
        grid_size = int(np.sqrt(n_points))
        x = torch.linspace(-1, 1, grid_size, device=device)
        y = torch.linspace(-1, 1, grid_size, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Coordinates at final timestep ONLY (not all timesteps)
        t_val = float(t_idx) / (T - 1) * 2 - 1  # Normalize to [-1, 1]
        coords = torch.stack([
            xx.flatten(),
            yy.flatten(),
            torch.full_like(xx.flatten(), t_val)
        ], dim=-1)  # [P, 3]
        
        # For single timestep: [1, P, 3] not [T, P, 3]
        coords_single = coords.unsqueeze(0)  # [1, P, 3]
        
        # Forward pass with SINGLE timestep coords
        out = model(
            coords=coords_single,
            frames=frames,
            scalars=scalars,
            observed_mask=observed_mask
        )
        
        # Extract fields (should now be [P, 3] or [1, P, 3])
        B = out.B.cpu().numpy()
        u = out.u.cpu().numpy()
        
        print(f"  DEBUG: B shape = {B.shape}, expected [P, 3] or [1, P, 3]")
        print(f"  DEBUG: P = {coords_single.shape[1]}")
        
        # Get eta (assuming constant)
        eta_val = 0.01  # Default from config
        
        # Reshape to 2D grids
        if B.ndim == 2:
            # [P, 3]
            B_x = B[:, 0].reshape(grid_size, grid_size)
            B_y = B[:, 1].reshape(grid_size, grid_size)
            B_z = B[:, 2].reshape(grid_size, grid_size)
        elif B.ndim == 3 and B.shape[0] == 1:
            # [1, P, 3]
            B_x = B[0, :, 0].reshape(grid_size, grid_size)
            B_y = B[0, :, 1].reshape(grid_size, grid_size)
            B_z = B[0, :, 2].reshape(grid_size, grid_size)
        else:
            raise ValueError(f"Unexpected B shape: {B.shape}")
        
        # Compute current density J_z = ∂B_y/∂x - ∂B_x/∂y
        # Match the current density script exactly
        dBy_dx = np.gradient(B_y, axis=1)  # x is axis=1
        dBx_dy = np.gradient(B_x, axis=0)  # y is axis=0
        J_z = dBy_dx - dBx_dy
        
        # Ohmic heating: η|J|²
        heating = eta_val * J_z**2
        
        # Get input magnetogram
        input_bz = frames[t_idx, 2, :, :].cpu().numpy()  # [H, W]
        
    return {
        'B_x': B_x,
        'B_y': B_y,
        'B_z': B_z,
        'J_z': J_z,
        'heating': heating,
        'input_bz': input_bz,
        'x_1d': x.cpu().numpy(),  # 1D array for streamplot
        'y_1d': y.cpu().numpy(),  # 1D array for streamplot
    }


def plot_field_lines_heating(
    flaring_fields: dict,
    nonflaring_fields: dict,
    output_path: Path,
    dpi: int = 300,
):
    """
    Create 2-panel comparison with field lines and Ohmic heating.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (fields, title) in enumerate([
        (flaring_fields, "(a) Flaring Active Region\n24h before M-class event"),
        (nonflaring_fields, "(b) Non-Flaring Active Region\nMagnetically complex but stable")
    ]):
        ax = axes[idx]
        
        # Background: Input Bz magnetogram
        extent = [-1, 1, -1, 1]
        ax.imshow(
            fields["input_bz"],
            extent=extent,
            origin="lower",
            cmap="gray",
            alpha=0.4,
            vmin=-2, vmax=2,
        )
        
        # Overlay: Ohmic heating (η|J|²)
        heating_normalized = fields["heating"] / fields["heating"].max()
        im = ax.imshow(
            heating_normalized,
            extent=extent,
            origin="lower",
            cmap="hot",
            alpha=0.6,
            vmin=0, vmax=1,
        )
        
        # Overlay: Magnetic field lines (streamlines)
        # streamplot needs 1D x, y arrays
        x_1d = fields["x_1d"]
        y_1d = fields["y_1d"]
        B_x = fields["B_x"]
        B_y = fields["B_y"]
        
        # Compute field line density (proportional to |B|)
        B_mag = np.sqrt(B_x**2 + B_y**2)
        density = 1.5  # Base density
        
        # Plot streamlines
        strm = ax.streamplot(
            x_1d, y_1d,
            B_x, B_y,
            color='cyan',
            linewidth=1.5,
            density=density,
            arrowsize=1.2,
            arrowstyle='->',
        )
        
        # Colorbar for heating
        if idx == 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Normalized Ohmic Heating\n' + r'$\eta |J_z|^2$', 
                          fontsize=11, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
        
        ax.set_xlabel('x [normalized]', fontsize=12, fontweight='bold')
        ax.set_ylabel('y [normalized]', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.tick_params(labelsize=10)
        ax.set_aspect('equal')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, label='Bz Magnetogram'),
            Line2D([0], [0], color='cyan', lw=2, label='Field Lines'),
            Line2D([0], [0], color='red', lw=4, label='Ohmic Heating'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\n✓ Saved field line + heating plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize magnetic field lines and Ohmic heating")
    parser.add_argument("--config", type=Path, default=Path("src/configs/pinn_40k_to_60k.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/Final model PINN/best_model.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/interim/windows_test_15.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("final_results/field_lines"))
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--flaring-idx", type=int, default=42, help="Test set index for flaring case")
    parser.add_argument("--nonflaring-idx", type=int, default=1500, help="Test set index for non-flaring case")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    device = get_device(args.device)
    model, dataset = load_model_and_data(args.config, args.checkpoint, args.data, device)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTING PHYSICS FIELDS")
    print(f"{'='*60}")
    
    # Extract fields for flaring case
    print(f"\nProcessing flaring sample {args.flaring_idx}...")
    sample_flaring = dataset[args.flaring_idx]
    fields_flaring = extract_physics_fields(model, sample_flaring, device, t_idx=-1, n_points=4096)
    
    print(f"  Max heating: {fields_flaring['heating'].max():.6f}")
    print(f"  Mean |Jz|: {np.abs(fields_flaring['J_z']).mean():.4f}")
    print(f"  Mean |B|: {np.sqrt(fields_flaring['B_x']**2 + fields_flaring['B_y']**2).mean():.4f}")
    
    # Extract fields for non-flaring case
    print(f"\nProcessing non-flaring sample {args.nonflaring_idx}...")
    sample_nonflaring = dataset[args.nonflaring_idx]
    fields_nonflaring = extract_physics_fields(model, sample_nonflaring, device, t_idx=-1, n_points=4096)
    
    print(f"  Max heating: {fields_nonflaring['heating'].max():.6f}")
    print(f"  Mean |Jz|: {np.abs(fields_nonflaring['J_z']).mean():.4f}")
    print(f"  Mean |B|: {np.sqrt(fields_nonflaring['B_x']**2 + fields_nonflaring['B_y']**2).mean():.4f}")
    
    # Create visualization
    plot_field_lines_heating(
        fields_flaring,
        fields_nonflaring,
        args.output_dir / "field_lines_heating.png",
        dpi=300
    )
    
    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

