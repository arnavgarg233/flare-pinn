import numpy as np
import pandas as pd

frames = pd.read_parquet('S:/flare_forecasting/frames_meta.parquet')
sample = frames.iloc[0]

print(f'Sample frame: {sample["frame_path"]}')
print(f'HARP: {sample["harpnum"]}')
print(f'Date: {sample["date_obs"]}')

npz_path = 'S:/flare_forecasting/' + sample['frame_path']
npz = np.load(npz_path)

print(f'\nNPZ arrays: {list(npz.keys())}')

# Check if there are any coordinate arrays
for key in npz.keys():
    arr = npz[key]
    print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}')

