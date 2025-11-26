
import torch
import pytest
from src.models.pinn.config import PINNConfig, DataConfig, ModelConfig
from src.models.pinn.hybrid_model import HybridPINNModel
from src.data.windows_dataset import WindowsDataset

def test_config_components():
    cfg = PINNConfig()
    cfg.data.components = ["Bx", "By", "Bz"]
    assert cfg.data.n_components == 3
    
    cfg.data.components = ["Bz"]
    assert cfg.data.n_components == 1

def test_backbone_shapes():
    # Test 1-component (legacy)
    cfg = PINNConfig()
    cfg.data.components = ["Bz"]
    model = HybridPINNModel(cfg)
    
    # Batch size 2, Time 4, Points 10
    B, T, P = 2, 4, 10
    coords = torch.randn(T, P, 3)
    frames = torch.randn(T, 1, 64, 64)
    observed_mask = torch.ones(T, dtype=torch.bool)
    
    out = model(coords, frames=frames, observed_mask=observed_mask)
    assert out.B_z.shape == (1, 10) # wait, model output B_z is [N, 1] in backbone, but reshaped in forward?
    # In forward: field.B is [N, C]
    # HybridPINNOutput stores A_z, B_z... derived from field.A, field.B
    # Let's check the internal tensors in FieldOutputs if accessible, or just the output
    
    # If 1 component, B has shape [N, 1]
    # HybridPINNOutput.B is field.B -> [N, 1]
    assert out.B.shape == (T*P, 1)
    
    # Test 3-component
    cfg.data.components = ["Bx", "By", "Bz"]
    model = HybridPINNModel(cfg)
    
    frames = torch.randn(T, 3, 64, 64)
    out = model(coords, frames=frames, observed_mask=observed_mask)
    
    assert out.B.shape == (T*P, 3)
    # Check legacy mapping
    assert out.B_z.shape == (T*P, 3) # It maps to B
    
if __name__ == "__main__":
    test_config_components()
    test_backbone_shapes()
    print("All tests passed!")

