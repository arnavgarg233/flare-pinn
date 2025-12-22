#!/usr/bin/env python3
"""
Visualize current density patterns to show physics constraint learns energy concentration.

Current density J_z = ∂B_y/∂x - ∂B_x/∂y is embedded in the resistive MHD term.
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


def load_model_and_data(
    config_path: Path,
    checkpoint_path: Path,
    data_path: Path,
    device: torch.device,
):
    """Load trained model and dataset."""
    print(f"📦 Loading config from {config_path}")
    cfg = PINNConfig.from_yaml(config_path)
    
    print(f"🔥 Loading model from {checkpoint_path}")
    model = HybridPINNModel(cfg)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "ema_state_dict" in ckpt and "shadow" in ckpt["ema_state_dict"]:
        print("   Using EMA weights")
        model.load_state_dict(ckpt["ema_state_dict"]["shadow"], strict=False)
    else:
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"📊 Loading data from {data_path}")
    windows_df, _ = load_windows_with_mask(data_path)
    dataset = ConsolidatedWindowsDataset(
        windows_df=windows_df,
        consolidated_dir=str(cfg.data.consolidated_dir),
        target_px=cfg.data.target_size,
        input_hours=cfg.data.input_hours,
        horizons=list(cfg.classifier.horizons),
        P_per_t=512,
        training=False,
        augment=False,
        max_cached_harps=100,
    )
    
    return model, dataset


def extract_current_density(
    model: HybridPINNModel,
    sample: dict,
    device: torch.device,
    n_points: int = 4096,
) -> dict:
    """
    Extract current density J_z = ∂B_y/∂x - ∂B_x/∂y.
    
    Returns dict with: J_z, B_x, B_y, B_z, input_bz
    """
    with torch.no_grad():
        frames = sample["frames"].to(device)  # [T, C, H, W]
        scalars = sample["scalars"].unsqueeze(0).to(device)
        observed_mask = sample["observed_mask"].to(device)
        
        T, C, H, W = frames.shape
        t_idx = T - 1  # Final timestep
        
        # Input magnetogram
        input_bz = frames[t_idx, 2, :, :].cpu().numpy()
        
        # Dense spatial grid
        grid_size = int(np.sqrt(n_points))
        x = torch.linspace(-1, 1, grid_size, device=device)
        y = torch.linspace(-1, 1, grid_size, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        t_norm = 2 * t_idx / (T - 1) - 1
        tt = torch.full_like(xx, t_norm)
        
        coords_flat = torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3)
        coords = coords_flat.unsqueeze(0)  # [1, N, 3]
        
        # Enable gradients for derivative computation
        coords.requires_grad_(True)
        
        # Forward pass
        out = model(
            coords=coords,
            frames=frames,
            scalars=scalars,
            observed_mask=observed_mask,
        )
        
        B = out.B  # [N, 3]
        B_x = B[:, 0].reshape(grid_size, grid_size)
        B_y = B[:, 1].reshape(grid_size, grid_size)
        B_z = B[:, 2].reshape(grid_size, grid_size)
        
        # Compute current density: J_z = ∂B_y/∂x - ∂B_x/∂y
        # Use numpy gradient (more stable than autograd for visualization)
        B_x_np = B_x.detach().cpu().numpy()
        B_y_np = B_y.detach().cpu().numpy()
        B_z_np = B_z.detach().cpu().numpy()
        
        dBy_dx = np.gradient(B_y_np, axis=1)  # x is axis=1
        dBx_dy = np.gradient(B_x_np, axis=0)  # y is axis=0
        
        J_z = dBy_dx - dBx_dy
        
        return {
            "J_z": J_z,
            "B_x": B_x_np,
            "B_y": B_y_np,
            "B_z": B_z_np,
            "input_bz": input_bz,
            "x_grid": xx.cpu().numpy(),
            "y_grid": yy.cpu().numpy(),
        }


def plot_current_density_comparison(
    flaring_fields: dict,
    nonflaring_fields: dict,
    output_path: Path,
    dpi: int = 300,
):
    """
    Create 2-panel comparison of current density (clean, no background).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (fields, title) in enumerate([
        (flaring_fields, "(a) Flaring Active Region\n24h before M-class event"),
        (nonflaring_fields, "(b) Non-Flaring Active Region\nComplex but stable")
    ]):
        ax = axes[idx]
        
        # Current density |J_z| - clean display, no background
        J_z_abs = np.abs(fields["J_z"])
        
        # Normalize per-panel to show relative patterns
        J_z_norm = (J_z_abs - J_z_abs.min()) / (J_z_abs.max() - J_z_abs.min() + 1e-10)
        
        im_j = ax.imshow(
            J_z_norm,
            extent=[-1, 1, -1, 1],
            origin="lower",
            cmap="inferno",
            vmin=0, vmax=1,
        )
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("x (normalized)", fontsize=10)
        ax.set_ylabel("y (normalized)", fontsize=10)
        ax.set_aspect('equal')
        
        # Colorbar
        cbar = plt.colorbar(im_j, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$|J_z|$ (normalized)", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved current density figure: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize current density patterns")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--data", type=str, required=True, help="Test data parquet path")
    parser.add_argument("--output-dir", type=str, default="final_results/current_density", help="Output directory")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--flaring-idx", type=int, default=42, help="Flaring sample index")
    parser.add_argument("--nonflaring-idx", type=int, default=1500, help="Non-flaring sample index")
    
    args = parser.parse_args()
    
    device = get_device(args.device)
    print(f"🖥️  Using device: {device}")
    
    # Load model and data
    model, dataset = load_model_and_data(
        Path(args.config),
        Path(args.checkpoint),
        Path(args.data),
        device,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Extracting current density patterns...")
    
    # Extract fields
    flaring_sample = dataset[args.flaring_idx]
    nonflaring_sample = dataset[args.nonflaring_idx]
    
    flaring_fields = extract_current_density(model, flaring_sample, device, n_points=4096)
    nonflaring_fields = extract_current_density(model, nonflaring_sample, device, n_points=4096)
    
    # Print stats
    print(f"\nFlaring region (idx={args.flaring_idx}):")
    print(f"  |J_z|: mean={np.abs(flaring_fields['J_z']).mean():.4f}, max={np.abs(flaring_fields['J_z']).max():.4f}")
    
    print(f"\nNon-flaring region (idx={args.nonflaring_idx}):")
    print(f"  |J_z|: mean={np.abs(nonflaring_fields['J_z']).mean():.4f}, max={np.abs(nonflaring_fields['J_z']).max():.4f}")
    
    # Generate figure
    print(f"\n🎨 Generating current density comparison...")
    plot_current_density_comparison(
        flaring_fields,
        nonflaring_fields,
        output_dir / "current_density_comparison.png",
        dpi=300,
    )
    
    print(f"\n✅ Figure saved to: {output_dir}")
    print(f"\n📝 This shows current density (J_z = ∂B_y/∂x - ∂B_x/∂y)")
    print(f"   embedded in the resistive MHD term η∇×(∇×B)")


if __name__ == "__main__":
    main()

