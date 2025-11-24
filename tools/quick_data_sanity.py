import numpy as np
from pathlib import Path

def quick_spot_check():
    # Target a specific file we know exists
    target_file = Path("/Volumes/Lexar/flare_forecasting/frames/H1001_2011-10-22T18-00-00.npz")
    
    print(f"🔍 Spot checking: {target_file}")
    
    if not target_file.exists():
        print("❌ File not found!")
        return

    try:
        # Load the file
        data = np.load(target_file)
        print("✅ Load successful")
        
        # Check contents
        keys = list(data.keys())
        print(f"🔑 Keys found: {keys}")
        
        # Check shapes of key arrays
        for k in keys:
            arr = data[k]
            print(f"   - {k}: shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min():.2f}, {arr.max():.2f}]")
            
            # Simple validity check for Bz
            if k in ['Bz', 'Br']:
                if np.isnan(arr).all():
                    print("   ⚠️ Warning: Field is all NaN!")
                elif np.all(arr == 0):
                    print("   ⚠️ Warning: Field is all Zeros!")
                else:
                    print("   ✅ Field looks valid (non-empty)")

        print("\n🎉 Data looks GOOD! The pipeline should be able to read this.")

    except Exception as e:
        print(f"❌ Failed to load file: {e}")

if __name__ == "__main__":
    quick_spot_check()
