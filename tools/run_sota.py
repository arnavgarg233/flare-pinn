#!/usr/bin/env python3
"""
Run SOTA Physics-Informed Neural Network Training

This script runs the optimized PINN configuration designed to beat
current SOTA benchmarks (TSS > 0.8 @ 24h) for solar flare prediction.

Key innovations:
1. Multi-scale physics constraints with weak-form MHD induction
2. Attention-based spatial pooling for PIL focus
3. Physics-derived features (polarity balance, flux transport)
4. Temporal modeling of field evolution
5. EMA and aggressive class balancing

Usage:
    python tools/run_sota.py [--config CONFIG_PATH] [--device DEVICE]
"""
import sys
import argparse
import subprocess
from pathlib import Path


def check_dependencies():
    """Verify all required packages are available."""
    required = ['torch', 'numpy', 'pandas', 'pydantic', 'scipy']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def check_data_paths(config_path: Path):
    """Verify data paths in config exist."""
    import yaml
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    data_cfg = cfg.get('data', {})
    
    if not data_cfg.get('use_real', False):
        print("⚠️  Config set to use dummy data - will not achieve SOTA")
        return True
    
    paths_to_check = [
        ('windows_parquet', data_cfg.get('windows_parquet')),
        ('frames_meta_parquet', data_cfg.get('frames_meta_parquet')),
        ('npz_root', data_cfg.get('npz_root')),
    ]
    
    all_exist = True
    for name, path in paths_to_check:
        if path:
            p = Path(path)
            if not p.exists():
                print(f"❌ {name} not found: {path}")
                all_exist = False
            else:
                print(f"✓ {name}: {path}")
    
    return all_exist


def run_sota_training(config_path: str, device: str = None):
    """Run SOTA training with comprehensive monitoring."""
    print("=" * 80)
    print("🚀 STARTING SOTA PHYSICS-INFORMED NEURAL NETWORK TRAINING")
    print("=" * 80)
    print()
    
    config_path = Path(config_path)
    
    # Verify config exists
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("Available configs:")
        for cfg in Path("src/configs").glob("*.yaml"):
            print(f"  - {cfg}")
        sys.exit(1)
    
    print(f"Config: {config_path}")
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ All dependencies available")
    print()
    
    # Check data paths
    print("Checking data paths...")
    if not check_data_paths(config_path):
        print()
        print("⚠️  Some data paths are missing. Training may fail.")
        response = input("Continue anyway? [y/N] ")
        if response.lower() != 'y':
            sys.exit(1)
    print()
    
    # Prepare output directories
    Path("outputs/checkpoints/sota_physics").mkdir(parents=True, exist_ok=True)
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        "src/train.py",
        "--config", str(config_path)
    ]
    
    # Print training info
    print("Target Performance:")
    print("  - TSS @ 24h: > 0.80 (Current SOTA: ~0.65-0.75)")
    print("  - TSS @ 12h: > 0.70")
    print("  - TSS @ 6h:  > 0.60")
    print()
    print("Key Techniques for SOTA:")
    print("  1. Multi-scale physics (weak-form MHD induction)")
    print("  2. Attention-based PIL focus")
    print("  3. Physics-derived features")
    print("  4. Temporal evolution modeling")
    print("  5. EMA + aggressive class balancing")
    print()
    print("-" * 80)
    print("Starting training...")
    print("-" * 80)
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("=" * 80)
        print("✅ Training Complete!")
        print("=" * 80)
        print()
        print("Best model saved to: outputs/checkpoints/sota_physics/best_model.pt")
        print()
        print("Next steps:")
        print("  1. Run evaluation: python tools/evaluate_model.py --checkpoint outputs/checkpoints/sota_physics/best_model.pt")
        print("  2. Compare with baselines: python tools/benchmarks/benchmark_comparison.py")
        print()
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print("❌ Training Failed!")
        print("=" * 80)
        print(f"Error code: {e.returncode}")
        print("Check logs at: outputs/logs/sota_physics.log")
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("🛑 Training Interrupted by User")
        print("Checkpoints saved in: outputs/checkpoints/sota_physics/")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Run SOTA Physics-Informed Neural Network Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/sota_physics.yaml",
        help="Path to config file (default: src/configs/sota_physics.yaml)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detected if not specified."
    )
    
    args = parser.parse_args()
    run_sota_training(args.config, args.device)


if __name__ == "__main__":
    main()
