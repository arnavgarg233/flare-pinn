#!/usr/bin/env python3
"""
Quick 2K benchmark test with PARALLEL execution - runs both models simultaneously.
Should take ~1-1.5 hours total (half the time of sequential).
"""
import sys
sys.path.insert(0, '.')

from benchmark_overnight_5k import benchmark_model, setup_logger, SOTA_BENCHMARKS, BASELINE_BENCHMARKS
from datetime import datetime
import multiprocessing as mp
from pathlib import Path

def run_model_worker(model_type, device, config_override, result_queue):
    """Worker function to run a single model and put results in queue."""
    try:
        print(f"\n[{model_type.upper()}] Starting training...")
        result = benchmark_model(model_type, device=device, config_override=config_override)
        result_queue.put((model_type, result, None))
        print(f"\n[{model_type.upper()}] Training complete!")
    except Exception as e:
        result_queue.put((model_type, None, str(e)))
        print(f"\n[{model_type.upper()}] ERROR: {e}")


def main():
    """Run parallel 2K benchmark."""
    device = 'cpu'
    
    setup_logger(None)
    
    print("="*70)
    print("PARALLEL QUICK BENCHMARK: 2K Steps Per Model")
    print("="*70)
    print(f"Running MLP and Hybrid in PARALLEL")
    print(f"Estimated time: 1-1.5 hours total on CPU")
    print(f"Device: {device}")
    print()
    
    # Override config to use 2000 steps
    config_override = {'train': {'steps': 2000}}
    
    # Create result queue for inter-process communication
    result_queue = mp.Queue()
    
    # Create processes for both models
    mlp_process = mp.Process(
        target=run_model_worker,
        args=('mlp', device, config_override, result_queue)
    )
    
    hybrid_process = mp.Process(
        target=run_model_worker,
        args=('hybrid', device, config_override, result_queue)
    )
    
    print("\n" + "="*70)
    print("Starting parallel training...")
    print("="*70)
    print("\n⚠️  Note: Output from both models will be interleaved.")
    print("    Check checkpoint directories for individual logs.")
    print()
    
    # Start both processes
    start_time = datetime.now()
    mlp_process.start()
    hybrid_process.start()
    
    # Wait for both to complete
    mlp_process.join()
    hybrid_process.join()
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Collect results from queue
    results = {}
    errors = {}
    while not result_queue.empty():
        model_type, result, error = result_queue.get()
        if error:
            errors[model_type] = error
        else:
            results[model_type] = result
    
    # Check for errors
    if errors:
        print(f"\n{'='*70}")
        print("ERRORS OCCURRED")
        print(f"{'='*70}")
        for model_type, error in errors.items():
            print(f"\n{model_type.upper()}: {error}")
        return
    
    if 'mlp' not in results or 'hybrid' not in results:
        print("\n❌ Both models did not complete successfully.")
        return
    
    mlp_results = results['mlp']
    hybrid_results = results['hybrid']
    
    # Final comparison
    print(f"\n{'='*70}")
    print("PARALLEL TEST RESULTS (2K steps)")
    print(f"{'='*70}")
    
    print(f"\nTotal Wall Time: {total_time/3600:.2f}h (both ran in parallel!)")
    print(f"\nIndividual Training Times:")
    print(f"  MLP:    {mlp_results['train_time']/3600:.2f}h")
    print(f"  Hybrid: {hybrid_results['train_time']/3600:.2f}h")
    print(f"\nSpeedup: {(mlp_results['train_time'] + hybrid_results['train_time'])/total_time:.1f}x faster than sequential")
    
    # Use horizons from results
    horizons = mlp_results.get('horizons', [6, 12, 24])
    
    print(f"\n{'='*70}")
    print("TSS COMPARISON")
    print(f"{'='*70}")
    
    for h in horizons:
        if h in mlp_results['metrics'] and h in hybrid_results['metrics']:
            mlp_tss = mlp_results['metrics'][h]['tss']
            hyb_tss = hybrid_results['metrics'][h]['tss']
            sota = SOTA_BENCHMARKS.get(h, {})
            sota_tss = sota.get('tss', 0.0)
            baseline = BASELINE_BENCHMARKS.get(h, {})
            baseline_tss = baseline.get('tss', 0.0)
            
            print(f"\n{h}h Horizon:")
            print(f"  MLP:      {mlp_tss:.4f} ({(mlp_tss/sota_tss-1)*100:+.1f}% vs SOTA)" if sota_tss > 0 else f"  MLP:      {mlp_tss:.4f}")
            print(f"  Hybrid:   {hyb_tss:.4f} ({(hyb_tss/sota_tss-1)*100:+.1f}% vs SOTA)" if sota_tss > 0 else f"  Hybrid:   {hyb_tss:.4f}")
            if sota_tss > 0:
                print(f"  SOTA:     {sota_tss:.4f} ({sota.get('source', 'N/A')})")
            if baseline_tss > 0:
                print(f"  Baseline: {baseline_tss:.4f} ({baseline.get('source', 'N/A')})")
            
            winner = 'Hybrid' if hyb_tss > mlp_tss else 'MLP' if mlp_tss > hyb_tss else 'Tie'
            delta = abs(hyb_tss - mlp_tss)
            print(f"  Winner:   {winner} (Δ={delta:.4f})")
    
    print(f"\n{'='*70}")
    print("CHECKPOINTS")
    print(f"{'='*70}")
    print(f"  MLP:    {mlp_results['checkpoint']}")
    print(f"  Hybrid: {hybrid_results['checkpoint']}")
    
    print(f"\n{'='*70}")
    print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    print(f"\n💡 Note: This is a quick 2K test. Full 5K training recommended for final results.")


if __name__ == '__main__':
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()

