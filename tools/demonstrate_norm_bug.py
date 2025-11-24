
import numpy as np

def test_normalization_issue():
    # Simulate a growing active region
    print("--- Simulating Flux Emergence Normalization Bug ---")
    
    # Case 1: Quiet Sun (noise)
    # Random noise ~ 10 Gauss
    bz_quiet = np.random.randn(100, 100) * 10.0
    
    # Case 2: Emerging Region (peak 1000 Gauss)
    # Gaussian blob
    x = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(x, x)
    blob = 1000.0 * np.exp(-(xx**2 + yy**2)/0.1)
    bz_active = blob + np.random.randn(100, 100) * 10.0
    
    # Case 3: Pre-Flare Region (peak 3000 Gauss)
    bz_flare = 3000.0 * np.exp(-(xx**2 + yy**2)/0.1) + np.random.randn(100, 100) * 10.0
    
    frames = [("Quiet", bz_quiet), ("Active", bz_active), ("Flare", bz_flare)]
    
    for name, bz in frames:
        # Original logic from windows_dataset.py
        bz_std = np.std(bz)
        # Fallback in code
        if bz_std < 1e-6: bz_std = 1000.0
        
        # Normalization
        bz_norm = bz / (3.0 * bz_std)
        bz_norm = np.clip(bz_norm, -3.0, 3.0)
        
        print(f"State: {name:10s} | Real Peak: {bz.max():7.1f} G | Std: {bz_std:7.1f} | Norm Peak: {bz_norm.max():.4f}")

if __name__ == "__main__":
    test_normalization_issue()

