#!/usr/bin/env python3
"""
Quick test script to verify Apple Silicon GPU (MPS) support is working.
Run this before training to ensure your M2 Pro GPU is accessible.
"""
import torch
import sys

def test_mps_availability():
    """Test if MPS backend is available and working."""
    print("=" * 60)
    print("Testing Apple Silicon GPU (MPS) Support")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Check MPS availability
    print(f"\nMPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if not torch.backends.mps.is_available():
        print("\n❌ MPS is NOT available!")
        print("   Make sure you're running on an Apple Silicon Mac (M1/M2/M3)")
        print("   and have PyTorch 1.12+ installed")
        return False
    
    # Test basic tensor operations on MPS
    print("\n" + "=" * 60)
    print("Testing MPS Operations")
    print("=" * 60)
    
    try:
        device = torch.device("mps")
        
        # Test 1: Basic tensor creation
        print("\n✓ Test 1: Creating tensor on MPS...")
        x = torch.randn(1000, 1000, device=device)
        print(f"  Created tensor with shape {x.shape} on {x.device}")
        
        # Test 2: Matrix multiplication
        print("\n✓ Test 2: Matrix multiplication on MPS...")
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        print(f"  Matrix multiplication successful: {z.shape}")
        
        # Test 3: Neural network operations
        print("\n✓ Test 3: Neural network layer on MPS...")
        linear = torch.nn.Linear(1000, 500).to(device)
        out = linear(x)
        print(f"  Linear layer output: {out.shape}")
        
        # Test 4: Backward pass
        print("\n✓ Test 4: Backward pass on MPS...")
        loss = out.sum()
        loss.backward()
        print(f"  Backward pass successful, grad shape: {linear.weight.grad.shape}")
        
        # Test 5: Data transfer
        print("\n✓ Test 5: Data transfer CPU <-> MPS...")
        cpu_tensor = torch.randn(100, 100)
        mps_tensor = cpu_tensor.to(device)
        back_to_cpu = mps_tensor.cpu()
        print(f"  Transfer successful, data preserved: {torch.allclose(cpu_tensor, back_to_cpu)}")
        
        print("\n" + "=" * 60)
        print("✅ All MPS tests passed!")
        print("=" * 60)
        print("\nYour M2 Pro GPU is ready for training!")
        print("You can now run training with device='mps' in your config.")
        return True
        
    except Exception as e:
        print(f"\n❌ MPS test failed with error: {e}")
        print("   There may be an issue with your MPS backend.")
        return False


def benchmark_devices():
    """Quick benchmark comparing CPU vs MPS performance."""
    print("\n" + "=" * 60)
    print("Performance Benchmark: CPU vs MPS")
    print("=" * 60)
    
    import time
    
    size = 2000
    iterations = 10
    
    # CPU benchmark
    print(f"\nBenchmarking CPU (matrix multiply {size}x{size}, {iterations} iterations)...")
    cpu_times = []
    for i in range(iterations):
        x = torch.randn(size, size)
        y = torch.randn(size, size)
        start = time.time()
        z = torch.mm(x, y)
        end = time.time()
        cpu_times.append(end - start)
    cpu_avg = sum(cpu_times) / len(cpu_times)
    print(f"  CPU average time: {cpu_avg*1000:.2f} ms")
    
    # MPS benchmark
    if torch.backends.mps.is_available():
        print(f"\nBenchmarking MPS (matrix multiply {size}x{size}, {iterations} iterations)...")
        device = torch.device("mps")
        mps_times = []
        
        # Warmup
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        _ = torch.mm(x, y)
        
        for i in range(iterations):
            x = torch.randn(size, size, device=device)
            y = torch.randn(size, size, device=device)
            start = time.time()
            z = torch.mm(x, y)
            torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
            end = time.time()
            mps_times.append(end - start)
        
        mps_avg = sum(mps_times) / len(mps_times)
        print(f"  MPS average time: {mps_avg*1000:.2f} ms")
        
        speedup = cpu_avg / mps_avg
        print(f"\n🚀 Speedup: {speedup:.2f}x faster on MPS!")
        
        if speedup < 1.5:
            print("\n⚠️  Warning: MPS speedup is less than expected.")
            print("   This could be due to:")
            print("   - Small tensor sizes (GPU overhead)")
            print("   - System load or thermal throttling")
            print("   - MPS backend version")
    else:
        print("\n⚠️  MPS not available, skipping MPS benchmark")


if __name__ == "__main__":
    success = test_mps_availability()
    
    if success:
        print("\n" + "=" * 60)
        response = input("\nRun performance benchmark? (y/n): ")
        if response.lower() == 'y':
            benchmark_devices()
    
    sys.exit(0 if success else 1)

