"""Debug script to check if physics loss is actually being computed."""
import torch
import sys
sys.path.insert(0, "src")

from models.pinn.model import PINNModel
from models.pinn.config import PINNConfig
import yaml

# Load config
with open("configs/train_pinn_cpu_smoke_test.yaml") as f:
    cfg_dict = yaml.safe_load(f)
cfg = PINNConfig(**cfg_dict)

# Create model
model = PINNModel(cfg)
model.eval()

# Create dummy input
T, P = 8, 512
coords = torch.randn(T, P, 3)  # Random coordinates
gt_bz = torch.randn(T, P, 1)   # Random Bz values
observed_mask = torch.ones(T, dtype=torch.bool)
labels = torch.randint(0, 2, (1, 3)).float()

# Set physics lambda to non-zero
model.step = 500  # Force lambda_phys > 0
frac = 500 / 500  # End of training
model.set_train_frac(frac)
print(f"Training fraction: {frac}")
print(f"Lambda phys at step {model.step}: {model.get_lambda_phys()}")

# Forward pass
print("\n=== Running forward pass ===")
out = model(
    coords=coords,
    gt_bz=gt_bz,
    observed_mask=observed_mask,
    labels=labels,
    mode="train"
)

print(f"\n=== Results ===")
print(f"Loss total: {out.loss_total.item():.6f}")
print(f"Loss cls: {out.loss_cls.item():.6f}")
print(f"Loss data: {out.loss_data.item():.6f}")
print(f"Loss phys: {out.loss_phys.item():.6f}")
print(f"Lambda phys: {out.lambda_phys}")
print(f"ESS: {out.ess}")

if out.loss_phys.item() == 0:
    print("\n❌ WARNING: Physics loss is ZERO! This is the bug!")
else:
    print(f"\n✅ Physics loss is non-zero: {out.loss_phys.item():.6f}")

