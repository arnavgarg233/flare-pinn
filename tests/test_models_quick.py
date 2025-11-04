"""
Quick unit tests for PINN models.
Tests model initialization, forward pass, and basic functionality.

Usage:
    pytest tests/test_models_quick.py -v
    # or
    python -m pytest tests/test_models_quick.py -v
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pinn import (
    PINNConfig,
    PINNModel,
    HybridPINNModel,
)


@pytest.fixture
def minimal_config():
    """Minimal config for fast testing."""
    return PINNConfig(
        seed=42,
        device="cpu",
        model={"hidden": 64, "layers": 3, "fourier": {"max_log2_freq": 2}},
        classifier={"hidden": 32, "horizons": [6, 12]},
        physics={"enable": False},  # Disable for speed
        train={"steps": 100, "batch_size": 1},
        data={"use_real": False, "dummy_T": 4, "dummy_H": 32, "dummy_W": 32}
    )


@pytest.fixture
def physics_config():
    """Config with physics enabled."""
    cfg = PINNConfig(
        seed=42,
        device="cpu",
        model={"hidden": 64, "layers": 3, "fourier": {"max_log2_freq": 2}},
        classifier={"hidden": 32, "horizons": [6, 12]},
        physics={
            "enable": True,
            "lambda_phys_schedule": [[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]]
        },
        train={"steps": 100},
        data={"use_real": False}
    )
    return cfg


class TestPINNModel:
    """Test pure MLP PINN model."""
    
    def test_initialization(self, minimal_config):
        """Test model can be initialized."""
        model = PINNModel(minimal_config)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self, minimal_config):
        """Test forward pass with dummy data."""
        model = PINNModel(minimal_config)
        model.eval()
        
        # Create dummy inputs
        T, P = 4, 64
        coords = torch.randn(T, P, 3)  # (x, y, t) coordinates
        gt_bz = torch.randn(T, P, 1)
        observed_mask = torch.ones(T, dtype=torch.bool)
        labels = torch.randint(0, 2, (1, 2)).float()
        
        # Forward pass
        with torch.no_grad():
            output = model(
                coords=coords,
                gt_bz=gt_bz,
                observed_mask=observed_mask,
                labels=labels,
                mode="eval"
            )
        
        # Check outputs
        assert output.probs.shape == (1, 2)  # 2 horizons
        assert output.B_z.shape == (T * P, 1)
        assert output.logits is not None
        assert 0.0 <= output.probs.min() <= 1.0
        assert 0.0 <= output.probs.max() <= 1.0
    
    def test_training_mode(self, minimal_config):
        """Test forward pass in training mode computes losses."""
        model = PINNModel(minimal_config)
        model.train()
        model.set_train_frac(0.5)
        
        T, P = 4, 64
        coords = torch.randn(T, P, 3)
        coords.requires_grad_(True)
        gt_bz = torch.randn(T, P, 1)
        observed_mask = torch.ones(T, dtype=torch.bool)
        labels = torch.randint(0, 2, (1, 2)).float()
        
        output = model(
            coords=coords,
            gt_bz=gt_bz,
            observed_mask=observed_mask,
            labels=labels,
            mode="train"
        )
        
        # Check losses are computed
        assert output.loss_cls.item() >= 0
        assert output.loss_data.item() >= 0
        assert output.loss_total.item() >= 0
        
        # Check can backprop
        output.loss_total.backward()
    
    def test_physics_loss(self, physics_config):
        """Test physics loss is computed when enabled."""
        model = PINNModel(physics_config)
        model.train()
        model.set_train_frac(1.0)  # End of training (max physics weight)
        
        T, P = 4, 64
        coords = torch.randn(T, P, 3)
        coords.requires_grad_(True)
        gt_bz = torch.randn(T, P, 1)
        observed_mask = torch.ones(T, dtype=torch.bool)
        labels = torch.randint(0, 2, (1, 2)).float()
        
        # Create dummy PIL mask
        H, W = 32, 32
        pil_mask = np.random.rand(H, W) > 0.8
        
        output = model(
            coords=coords,
            gt_bz=gt_bz,
            observed_mask=observed_mask,
            labels=labels,
            pil_mask=pil_mask,
            mode="train"
        )
        
        # Physics loss should be > 0 when enabled
        assert output.lambda_phys > 0
        assert output.loss_phys.item() > 0
        print(f"Physics loss: {output.loss_phys.item():.4e}")
        print(f"Lambda phys: {output.lambda_phys:.2f}")
    
    def test_curriculum_scheduling(self, physics_config):
        """Test curriculum learning adjusts physics weight."""
        model = PINNModel(physics_config)
        
        # Check lambda_phys increases over training
        fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
        lambdas = []
        
        for frac in fracs:
            model.set_train_frac(frac)
            lambdas.append(model.get_lambda_phys())
        
        print(f"Lambda schedule: {lambdas}")
        assert lambdas[0] == 0.0  # Start at 0
        assert lambdas[-1] > lambdas[0]  # Increases
        assert all(lambdas[i] <= lambdas[i+1] for i in range(len(lambdas)-1))  # Monotonic


class TestHybridPINNModel:
    """Test hybrid CNN-conditioned PINN model."""
    
    def test_initialization(self, minimal_config):
        """Test hybrid model can be initialized."""
        # Update config for hybrid
        minimal_config.model.model_type = "hybrid"
        model = HybridPINNModel(minimal_config, encoder_in_channels=4)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_encoder(self, minimal_config):
        """Test CNN encoder works."""
        minimal_config.model.model_type = "hybrid"
        model = HybridPINNModel(minimal_config, encoder_in_channels=4)
        
        # Create dummy frames
        T_obs, H, W = 4, 32, 32
        frames_obs = torch.randn(T_obs, H, W)
        observed_mask = torch.ones(T_obs, dtype=torch.bool)
        
        # Encode
        L, g = model.encode_frames(frames_obs, observed_mask)
        
        assert L.shape == (1, 32, H, W)  # Latent feature map
        assert g.shape == (1, 64)  # Global code
        print(f"Latent shape: {L.shape}, Global shape: {g.shape}")
    
    def test_forward_with_frames(self, minimal_config):
        """Test forward pass with frame encoding."""
        minimal_config.model.model_type = "hybrid"
        model = HybridPINNModel(minimal_config, encoder_in_channels=1)
        model.eval()
        
        T, P = 4, 64
        H, W = 32, 32
        coords = torch.randn(T, P, 3)
        frames = torch.randn(T, H, W)  # Full frames
        gt_bz = torch.randn(T, P, 1)
        observed_mask = torch.ones(T, dtype=torch.bool)
        labels = torch.randint(0, 2, (1, 2)).float()
        
        with torch.no_grad():
            output = model(
                coords=coords,
                frames=frames,  # NEW: pass frames for encoding
                gt_bz=gt_bz,
                observed_mask=observed_mask,
                labels=labels,
                mode="eval"
            )
        
        assert output.probs.shape == (1, 2)
        assert output.B_z.shape == (T * P, 1)
        print(f"Hybrid model output: probs={output.probs}")
    
    def test_hybrid_vs_mlp_shapes(self, minimal_config):
        """Test hybrid and MLP models produce same output shapes."""
        # MLP model
        mlp_cfg = minimal_config
        mlp_cfg.model.model_type = "mlp"
        mlp_model = PINNModel(mlp_cfg)
        
        # Hybrid model
        hybrid_cfg = minimal_config
        hybrid_cfg.model.model_type = "hybrid"
        hybrid_model = HybridPINNModel(hybrid_cfg, encoder_in_channels=1)
        
        # Same inputs
        T, P = 4, 64
        H, W = 32, 32
        coords = torch.randn(T, P, 3)
        frames = torch.randn(T, H, W)
        gt_bz = torch.randn(T, P, 1)
        observed_mask = torch.ones(T, dtype=torch.bool)
        labels = torch.randint(0, 2, (1, 2)).float()
        
        with torch.no_grad():
            mlp_out = mlp_model(coords, gt_bz, observed_mask, labels, mode="eval")
            hybrid_out = hybrid_model(coords, frames, gt_bz, observed_mask, labels, mode="eval")
        
        # Check shapes match
        assert mlp_out.probs.shape == hybrid_out.probs.shape
        assert mlp_out.B_z.shape == hybrid_out.B_z.shape
        assert mlp_out.u_x.shape == hybrid_out.u_x.shape
        print("✓ MLP and Hybrid produce compatible output shapes")


class TestCollocation:
    """Test collocation point sampling and importance weighting."""
    
    def test_pil_mask_sampling(self, minimal_config):
        """Test PIL mask importance weighting."""
        model = PINNModel(minimal_config)
        model.train()
        
        T, P = 4, 128
        H, W = 32, 32
        coords = torch.randn(T, P, 3)
        coords.requires_grad_(True)
        
        # Create strong PIL mask (concentrated region)
        pil_mask = np.zeros((H, W), dtype=np.float32)
        pil_mask[10:20, 10:20] = 1.0  # Only 25% of area
        
        gt_bz = torch.randn(T, P, 1)
        observed_mask = torch.ones(T, dtype=torch.bool)
        labels = torch.randint(0, 2, (1, 2)).float()
        
        # Enable physics to test collocation
        minimal_config.physics.enable = True
        model.set_train_frac(1.0)
        
        output = model(
            coords=coords,
            gt_bz=gt_bz,
            observed_mask=observed_mask,
            labels=labels,
            pil_mask=pil_mask,
            mode="train"
        )
        
        # ESS should be reasonable (> 0.1 * N)
        assert output.ess > 0.1 * T * P
        print(f"ESS: {output.ess:.1f} / {T*P} = {output.ess/(T*P):.2%}")


@pytest.mark.parametrize("model_type", ["mlp", "hybrid"])
def test_model_parameters_finite(model_type, minimal_config):
    """Test all model parameters are finite after initialization."""
    if model_type == "mlp":
        model = PINNModel(minimal_config)
    else:
        model = HybridPINNModel(minimal_config, encoder_in_channels=1)
    
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"Non-finite param: {name}"
    
    print(f"✓ All {model_type.upper()} parameters are finite")


if __name__ == "__main__":
    # Quick manual test
    print("Running quick model tests...")
    cfg = PINNConfig(
        seed=42,
        device="cpu",
        model={"hidden": 64, "layers": 3},
        classifier={"hidden": 32, "horizons": [6, 12]},
        physics={"enable": False},
        data={"use_real": False}
    )
    
    print("\n1. Testing MLP PINN...")
    mlp = PINNModel(cfg)
    coords = torch.randn(4, 64, 3)
    gt_bz = torch.randn(4, 64, 1)
    labels = torch.randint(0, 2, (1, 2)).float()
    out = mlp(coords, gt_bz, labels=labels, mode="eval")
    print(f"   Probs: {out.probs}")
    print(f"   ✓ MLP PINN works!")
    
    print("\n2. Testing Hybrid PINN...")
    cfg.model.model_type = "hybrid"
    hybrid = HybridPINNModel(cfg, encoder_in_channels=1)
    frames = torch.randn(4, 32, 32)
    observed_mask = torch.ones(4, dtype=torch.bool)
    out = hybrid(coords, frames, gt_bz, observed_mask, labels, mode="eval")
    print(f"   Probs: {out.probs}")
    print(f"   ✓ Hybrid PINN works!")
    
    print("\n✓ All quick tests passed!")

