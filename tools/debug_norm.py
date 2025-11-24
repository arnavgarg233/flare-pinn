
import numpy as np
import torch

def test_normalization_issue():
    # Simulate a growing active region
    # Frame 0: Peak 500G
    # Frame 1: Peak 1000G
    # Frame 2: Peak 2000G
    
    frames = []
    peaks = [500.0, 1000.0, 2000.0]
    
    print("--- Simulating Flux Emergence ---")
    for i, peak in enumerate(peaks):
        # Create gaussian blob
        x = np.linspace(-1, 1, 100)
        xx, yy = np.meshgrid(x, x)
        bz = peak * np.exp(-(xx**2 + yy**2)/0.1)
        
        # Replicate logic from WindowsDataset._load_bz
        # 1. Clip (dummy range for this test)
        bz = np.clip(bz, -3000, 3000)
        
        # 2. Calculate std
        bz_std = np.std(bz)
        if bz_std < 1e-6: bz_std = 1000.0
        
        # 3. Normalize
        bz_norm = bz / (3.0 * bz_std)
        bz_norm = np.clip(bz_norm, -3.0, 3.0)
        
        print(f"Frame {i}: Real Peak={peak:.1f}, Std={bz_std:.1f}, Norm Peak={bz_norm.max():.4f}")

if __name__ == "__main__":
    test_normalization_issue()

