#!/usr/bin/env python3
"""
Quick comparison of baseline vs physics-informed results.
Run after smoke test completes.
"""
import re
from pathlib import Path

def extract_metrics_from_log(log_path: Path):
    """Extract TSS, PR-AUC, Brier from log file."""
    if not log_path.exists():
        return None
    
    metrics = {"TSS": [], "PR-AUC": [], "Brier": []}
    
    with open(log_path, 'r') as f:
        for line in f:
            # Look for eval lines like: "h=12  TSS*=0.456@0.32  PR-AUC=0.678  Brier=0.123"
            if "TSS*=" in line and "h=12" in line:
                # Extract TSS
                tss_match = re.search(r"TSS\*=([\d.]+)", line)
                if tss_match:
                    metrics["TSS"].append(float(tss_match.group(1)))
                
                # Extract PR-AUC
                pr_match = re.search(r"PR-AUC=([\d.]+)", line)
                if pr_match:
                    metrics["PR-AUC"].append(float(pr_match.group(1)))
                
                # Extract Brier
                brier_match = re.search(r"Brier=([\d.]+)", line)
                if brier_match:
                    metrics["Brier"].append(float(brier_match.group(1)))
    
    if metrics["TSS"]:
        return {
            "TSS_12h": max(metrics["TSS"]),
            "PR-AUC_12h": max(metrics["PR-AUC"]) if metrics["PR-AUC"] else 0.0,
            "Brier_12h": min(metrics["Brier"]) if metrics["Brier"] else 1.0,
        }
    return None


def main():
    print("=" * 60)
    print("SMOKE TEST RESULTS COMPARISON")
    print("=" * 60)
    print()
    
    # Load logs
    baseline_log = Path("logs/smoke_test_baseline.log")
    physics_log = Path("logs/smoke_test_physics.log")
    
    baseline_metrics = extract_metrics_from_log(baseline_log)
    physics_metrics = extract_metrics_from_log(physics_log)
    
    if not baseline_metrics:
        print("⚠ Baseline log not found or incomplete")
        print(f"   Expected: {baseline_log}")
        return
    
    if not physics_metrics:
        print("⚠ Physics log not found or incomplete")
        print(f"   Expected: {physics_log}")
        return
    
    # Compare
    print("📊 Metrics at 12h Horizon:")
    print()
    print(f"                    Baseline    Physics    Δ (improvement)")
    print("-" * 60)
    
    tss_delta = physics_metrics["TSS_12h"] - baseline_metrics["TSS_12h"]
    tss_pct = 100 * tss_delta / max(baseline_metrics["TSS_12h"], 0.01)
    print(f"TSS (higher better)  {baseline_metrics['TSS_12h']:.4f}      {physics_metrics['TSS_12h']:.4f}     {tss_delta:+.4f} ({tss_pct:+.1f}%)")
    
    pr_delta = physics_metrics["PR-AUC_12h"] - baseline_metrics["PR-AUC_12h"]
    pr_pct = 100 * pr_delta / max(baseline_metrics["PR-AUC_12h"], 0.01)
    print(f"PR-AUC (higher)      {baseline_metrics['PR-AUC_12h']:.4f}      {physics_metrics['PR-AUC_12h']:.4f}     {pr_delta:+.4f} ({pr_pct:+.1f}%)")
    
    brier_delta = physics_metrics["Brier_12h"] - baseline_metrics["Brier_12h"]
    brier_pct = 100 * brier_delta / max(baseline_metrics["Brier_12h"], 0.01)
    print(f"Brier (lower)        {baseline_metrics['Brier_12h']:.4f}      {physics_metrics['Brier_12h']:.4f}     {brier_delta:+.4f} ({brier_pct:+.1f}%)")
    
    print()
    print("=" * 60)
    
    # Verdict
    if tss_delta > 0.02:
        print("✅ PHYSICS WINS! TSS improvement > 0.02")
        print("   → Physics constraints are helping discrimination!")
    elif tss_delta > 0:
        print("✓ Physics slightly better (marginal improvement)")
        print("   → May need more training steps or hyperparameter tuning")
    else:
        print("⚠ Physics not helping yet")
        print("   → Check: physics weight schedule, collocation, or data quality")
    
    print()
    print("Next steps:")
    if tss_delta > 0:
        print("  1. ✅ Code works! Ready for GPU training")
        print("  2. Run full training (5K steps) on RunPod")
        print("  3. Expect bigger improvements with more compute")
    else:
        print("  1. Check logs for errors or NaNs")
        print("  2. Try adjusting lambda_phys_schedule")
        print("  3. Verify data quality (divergence-free check)")
    print()


if __name__ == "__main__":
    main()

