#!/usr/bin/env python3
"""
Quick 2K benchmark test - runs both MLP and Hybrid models for 2000 steps each.
Should take ~2-3 hours total.
"""
import sys
sys.path.insert(0, '.')

from benchmark_overnight_5k import benchmark_model, main as orig_main, setup_logger, SOTA_BENCHMARKS
from datetime import datetime

def main():
    """Run quick 2K benchmark."""
    device = 'cpu'
    
    setup_logger(None)
    
    print("QUICK BENCHMARK: 2K Steps Each Model")
    print("Estimated time: 2-3 hours total on CPU")
    print(f"Device: {device}")
    print()
    
    # Override config to use 2000 steps and 128x128 images (for memory)
    config_override = {
        'train': {'steps': 2000},
        'data': {'target_size': 128, 'P_per_t': 512}
    }
    
    try:
        # Run both models with 2K steps
        print("\n" + "="*70)
        print("Running MLP model (2K steps)...")
        print("="*70)
        mlp_results = benchmark_model('mlp', device=device, config_override=config_override)
        
        print("\n" + "="*70)
        print("Running Hybrid model (2K steps)...")
        print("="*70)
        hybrid_results = benchmark_model('hybrid', device=device, config_override=config_override)
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure:")
        print("  1. Config files exist: configs/mlp_5k.yaml, configs/hybrid_5k.yaml")
        print("  2. Data files exist: data/interim/windows_train.parquet, etc.")
        return
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise
    
    # Final comparison
    print(f"\n{'='*70}")
    print("QUICK TEST RESULTS (2K steps)")
    print(f"{'='*70}")
    
    print(f"\nTraining Time:")
    print(f"  MLP:    {mlp_results['train_time']/3600:.2f}h")
    print(f"  Hybrid: {hybrid_results['train_time']/3600:.2f}h")
    
    # Use horizons from results
    horizons = mlp_results.get('horizons', [6, 12, 24])
    
    print(f"\nTSS Comparison:")
    for h in horizons:
        if h in mlp_results['metrics'] and h in hybrid_results['metrics']:
            mlp_tss = mlp_results['metrics'][h]['tss']
            hyb_tss = hybrid_results['metrics'][h]['tss']
            sota = SOTA_BENCHMARKS.get(h, {})
            sota_tss = sota.get('tss', 0.0)
            
            print(f"\n{h}h Horizon:")
            print(f"  MLP:      {mlp_tss:.4f}")
            print(f"  Hybrid:   {hyb_tss:.4f}")
            if sota_tss > 0:
                print(f"  SOTA:     {sota_tss:.4f} ({sota.get('source', 'N/A')})")
            print(f"  Winner:   {'Hybrid' if hyb_tss > mlp_tss else 'MLP' if mlp_tss > hyb_tss else 'Tie'}")
    
    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    print(f"\n💡 Note: This is a quick 2K test. Full 5K training recommended for final results.")


if __name__ == '__main__':
    main()

