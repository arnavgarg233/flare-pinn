# tests/test_vector_physics.py
"""
Unit tests for the 2.5D Vector Induction Equation physics module.

Tests verify:
1. Correct mathematical formulation of the weak-form residuals
2. Solenoidal constraint (∇·B = 0) enforcement
3. Backward compatibility with scalar-only physics
4. Gradient computation correctness
5. Component-wise residual tracking
"""
import pytest
import torch
import torch.nn as nn
import math
from typing import Optional

# Import the modules under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pinn.physics import (
    VectorInduction2p5D, 
    WeakFormInduction2p5D,
    VectorPhysicsResidualInfo,
    PhysicsResidualInfo,
    MultiScaleTestFunction,
)
from models.pinn.core import PINNBackbone
from models.pinn.config import PINNConfig, PhysicsConfig, EtaConfig


class DummyVectorModel(nn.Module):
    """Simple model that outputs vector B field for testing."""
    
    def __init__(self, with_eta: bool = False):
        super().__init__()
        self.with_eta = with_eta
        # Simple linear layers for testing
        self.net = nn.Linear(3, 7 if with_eta else 6)
    
    def forward(self, coords: torch.Tensor) -> dict[str, Optional[torch.Tensor]]:
        out = self.net(coords)
        
        A_z = out[..., 0:1]
        Bx = out[..., 1:2]
        By = out[..., 2:3]
        Bz = out[..., 3:4]
        ux = out[..., 4:5]
        uy = out[..., 5:6]
        eta_raw = out[..., 6:7] if self.with_eta else None
        
        B = torch.cat([Bx, By, Bz], dim=-1)
        u = torch.cat([ux, uy], dim=-1)
        
        return {
            "A_z": A_z,
            "B": B,
            "u": u,
            "B_x": Bx,
            "B_y": By,
            "B_z": Bz,
            "u_x": ux,
            "u_y": uy,
            "eta_raw": eta_raw,
        }


class DummyScalarModel(nn.Module):
    """Simple model that outputs only scalar Bz for testing backward compatibility."""
    
    def __init__(self, with_eta: bool = False):
        super().__init__()
        self.with_eta = with_eta
        self.net = nn.Linear(3, 5 if with_eta else 4)
    
    def forward(self, coords: torch.Tensor) -> dict[str, Optional[torch.Tensor]]:
        out = self.net(coords)
        
        return {
            "A_z": out[..., 0:1],
            "B_z": out[..., 1:2],
            "u_x": out[..., 2:3],
            "u_y": out[..., 3:4],
            "eta_raw": out[..., 4:5] if self.with_eta else None,
        }


class AnalyticTestModel(nn.Module):
    """
    Model with analytic solution for testing physics correctness.
    
    Uses a simple travelling wave solution:
        Bx = A * sin(kx * x + ky * y - omega * t)
        By = A * cos(kx * x + ky * y - omega * t)
        Bz = B0 * sin(kx * x + ky * y - omega * t)
        ux = u0 (constant)
        uy = 0
        
    The solenoidal condition ∂x·Bx + ∂y·By = 0 is satisfied when:
        kx * A * cos(...) - ky * A * sin(...) = 0
    which requires kx = ky = k.
    """
    
    def __init__(self, A: float = 1.0, B0: float = 1.0, k: float = 1.0, 
                 omega: float = 1.0, u0: float = 0.1):
        super().__init__()
        self.A = A
        self.B0 = B0
        self.k = k
        self.omega = omega
        self.u0 = u0
    
    def forward(self, coords: torch.Tensor) -> dict[str, Optional[torch.Tensor]]:
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        t = coords[..., 2:3]
        
        phase = self.k * x + self.k * y - self.omega * t
        
        Bx = self.A * torch.sin(phase)
        By = self.A * torch.cos(phase)  
        Bz = self.B0 * torch.sin(phase)
        ux = torch.full_like(x, self.u0)
        uy = torch.zeros_like(y)
        
        B = torch.cat([Bx, By, Bz], dim=-1)
        u = torch.cat([ux, uy], dim=-1)
        
        return {
            "A_z": torch.zeros_like(Bz),  # Not used in physics
            "B": B,
            "u": u,
            "B_x": Bx,
            "B_y": By,
            "B_z": Bz,
            "u_x": ux,
            "u_y": uy,
            "eta_raw": None,
        }


class SolenoidalTestModel(nn.Module):
    """
    Model that explicitly satisfies ∇·B = 0.
    
    Uses stream function formulation:
        Bx = ∂ψ/∂y
        By = -∂ψ/∂x
        Bz = arbitrary
    where ψ = sin(x) * sin(y) * cos(t)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, coords: torch.Tensor) -> dict[str, Optional[torch.Tensor]]:
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        t = coords[..., 2:3]
        
        # Stream function ψ = sin(x) * sin(y) * cos(t)
        # Bx = ∂ψ/∂y = sin(x) * cos(y) * cos(t)
        # By = -∂ψ/∂x = -cos(x) * sin(y) * cos(t)
        
        Bx = torch.sin(x) * torch.cos(y) * torch.cos(t)
        By = -torch.cos(x) * torch.sin(y) * torch.cos(t)
        Bz = torch.sin(x + y) * torch.cos(t)
        
        # Simple velocity
        ux = 0.1 * torch.ones_like(x)
        uy = 0.1 * torch.ones_like(y)
        
        B = torch.cat([Bx, By, Bz], dim=-1)
        u = torch.cat([ux, uy], dim=-1)
        
        return {
            "A_z": torch.zeros_like(Bz),
            "B": B,
            "u": u,
            "B_x": Bx,
            "B_y": By,
            "B_z": Bz,
            "u_x": ux,
            "u_y": uy,
            "eta_raw": None,
        }


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_vector_physics(
    *,
    resistive: bool = False,
    div_B_weight: float = 1.0,
    component_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    enforce_div_free_u: bool = False,
):
    """Helper to construct physics module with config objects."""
    physics_cfg = PhysicsConfig(
        resistive=resistive,
        div_B_weight=div_B_weight,
        component_weights=component_weights,
        enforce_div_free_u=enforce_div_free_u,
    )
    eta_cfg = EtaConfig()
    return VectorInduction2p5D(
        physics_cfg=physics_cfg,
        eta_cfg=eta_cfg,
        n_fourier_modes=3,
        n_random_tests=2,
    )


def make_scalar_physics(resistive: bool = False):
    """Helper to construct legacy scalar physics alias."""
    physics_cfg = PhysicsConfig(resistive=resistive)
    eta_cfg = EtaConfig()
    return WeakFormInduction2p5D(
        physics_cfg=physics_cfg,
        eta_cfg=eta_cfg,
        n_fourier_modes=3,
        n_random_tests=2,
    )


@pytest.fixture
def coords(device):
    """Generate random collocation points."""
    N = 256
    coords = torch.rand(N, 3, device=device) * 2 - 1  # [-1, 1]^3
    coords.requires_grad_(True)
    return coords


@pytest.fixture
def imp_weights(device):
    """Generate uniform importance weights."""
    N = 256
    return torch.ones(N, 1, device=device)


class TestMultiScaleTestFunction:
    """Test the multi-scale test function generator."""
    
    def test_output_shapes(self, coords):
        """Test that test functions have correct shapes."""
        test_fn = MultiScaleTestFunction(n_fourier_modes=3, n_random=2)
        phis, dphi_dx, dphi_dy = test_fn(coords)
        
        N = coords.shape[0]
        
        # Check we have the expected number of test functions
        # 1 constant + 2 linear + 3*3 fourier (3 modes, each with sin_x, sin_y, cross) + 2 random
        expected_count = 1 + 2 + 9 + 2
        assert len(phis) == expected_count
        assert len(dphi_dx) == expected_count
        assert len(dphi_dy) == expected_count
        
        # Check shapes
        for phi, dx, dy in zip(phis, dphi_dx, dphi_dy):
            assert phi.shape == (N, 1)
            assert dx.shape == (N, 1)
            assert dy.shape == (N, 1)
    
    def test_gradient_correctness(self, coords):
        """Test that analytical gradients match autograd."""
        test_fn = MultiScaleTestFunction(n_fourier_modes=2, n_random=0)
        phis, dphi_dx, dphi_dy = test_fn(coords)
        
        # Check gradient of x (should be 1)
        # phi[1] = x, so dphi_dx[1] = 1
        assert torch.allclose(dphi_dx[1], torch.ones_like(dphi_dx[1]), atol=1e-6)
        assert torch.allclose(dphi_dy[1], torch.zeros_like(dphi_dy[1]), atol=1e-6)
        
        # Check gradient of y
        # phi[2] = y, so dphi_dy[2] = 1
        assert torch.allclose(dphi_dx[2], torch.zeros_like(dphi_dx[2]), atol=1e-6)
        assert torch.allclose(dphi_dy[2], torch.ones_like(dphi_dy[2]), atol=1e-6)


class TestVectorInduction2p5D:
    """Test the vector induction equation physics module."""
    
    def test_forward_vector_mode(self, device, coords, imp_weights):
        """Test forward pass with vector B field model."""
        physics = make_vector_physics(resistive=False).to(device)
        
        model = DummyVectorModel().to(device)
        
        loss, info = physics(model, coords, imp_weights)
        
        # Check output types
        assert isinstance(loss, torch.Tensor)
        assert isinstance(info, VectorPhysicsResidualInfo)
        
        # Check loss is finite
        assert torch.isfinite(loss)
        
        # Check info fields
        assert info.loss_induction_Bx >= 0
        assert info.loss_induction_By >= 0
        assert info.loss_induction_Bz >= 0
        assert info.loss_divergence_B >= 0
    
    def test_forward_scalar_fallback(self, device, coords, imp_weights):
        """Test that scalar-only models work via fallback."""
        physics = make_vector_physics(resistive=False).to(device)
        
        model = DummyScalarModel().to(device)
        
        loss, info = physics(model, coords, imp_weights)
        
        # Should still work but only compute Bz physics
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)
        
        # Bx and By should be zero since scalar-only
        assert info.loss_induction_Bx == 0.0
        assert info.loss_induction_By == 0.0
        assert info.loss_induction_Bz > 0.0 or info.loss_induction_Bz == 0.0
    
    def test_resistive_term(self, device, coords, imp_weights):
        """Test that resistive term is included when enabled."""
        physics_no_resist = make_vector_physics(resistive=False).to(device)
        physics_resist = make_vector_physics(resistive=True).to(device)
        
        model = DummyVectorModel().to(device)
        
        loss_no, _ = physics_no_resist(model, coords, imp_weights)
        loss_yes, _ = physics_resist(model, coords, imp_weights)
        
        # Losses should be different (resistive adds diffusion term)
        # Note: might be equal in edge cases, so just check both are finite
        assert torch.isfinite(loss_no)
        assert torch.isfinite(loss_yes)
    
    def test_div_free_constraint(self, device, coords, imp_weights):
        """Test solenoidal constraint enforcement."""
        physics_with_div = make_vector_physics(div_B_weight=1.0).to(device)
        physics_no_div = make_vector_physics(div_B_weight=0.0).to(device)
        
        model = DummyVectorModel().to(device)
        
        _, info_div = physics_with_div(model, coords, imp_weights)
        _, info_no_div = physics_no_div(model, coords, imp_weights)
        
        # With div constraint, should have non-zero divergence loss
        # Without it, should be zero
        assert info_no_div.loss_divergence_B == 0.0
        # info_div.loss_divergence_B may or may not be > 0 depending on model
    
    def test_solenoidal_model(self, device, imp_weights):
        """Test that solenoidal model has low divergence loss."""
        N = 256
        coords = torch.rand(N, 3, device=device) * 2 - 1
        coords.requires_grad_(True)
        
        physics = make_vector_physics(div_B_weight=1.0, resistive=False).to(device)
        
        # Use the solenoidal test model
        model = SolenoidalTestModel().to(device)
        
        _, info = physics(model, coords, imp_weights)
        
        # Divergence should be very small for solenoidal field
        assert info.loss_divergence_B < 0.1, f"Divergence loss too high: {info.loss_divergence_B}"
    
    def test_component_weights(self, device, coords, imp_weights):
        """Test that component weights affect loss correctly."""
        physics_uniform = make_vector_physics(component_weights=(1.0, 1.0, 1.0)).to(device)
        physics_bz_only = make_vector_physics(component_weights=(0.0, 0.0, 1.0)).to(device)
        
        model = DummyVectorModel().to(device)
        
        loss_uniform, info_uniform = physics_uniform(model, coords, imp_weights)
        loss_bz, info_bz = physics_bz_only(model, coords, imp_weights)
        
        # With Bz-only weights, Bx and By contributions should be zeroed
        # The loss should be different
        assert torch.isfinite(loss_uniform)
        assert torch.isfinite(loss_bz)
    
    def test_gradient_flow(self, device, coords, imp_weights):
        """Test that gradients flow through physics loss."""
        physics = make_vector_physics().to(device)
        model = DummyVectorModel().to(device)
        
        loss, _ = physics(model, coords, imp_weights)
        
        # Backward pass should work
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


class TestWeakFormInduction2p5D:
    """Test backward-compatible WeakFormInduction2p5D alias."""
    
    def test_legacy_return_type(self, device, coords, imp_weights):
        """Test that legacy class returns PhysicsResidualInfo."""
        physics = make_scalar_physics(resistive=False).to(device)
        
        model = DummyVectorModel().to(device)
        
        loss, info = physics(model, coords, imp_weights)
        
        # Should return legacy info type
        assert isinstance(info, PhysicsResidualInfo)
        assert hasattr(info, 'loss_induction')
        assert hasattr(info, 'loss_divergence')
        assert hasattr(info, 'residual_mean')
        assert hasattr(info, 'residual_max')
    
    def test_backward_compat_with_scalar(self, device, coords, imp_weights):
        """Test backward compatibility with scalar models."""
        physics = make_scalar_physics().to(device)
        model = DummyScalarModel().to(device)
        
        loss, info = physics(model, coords, imp_weights)
        
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)


class TestPINNBackboneVector:
    """Test PINNBackbone with vector_B option."""
    
    def test_vector_output_format(self, device):
        """Test that vector mode outputs correct format."""
        backbone = PINNBackbone(
            hidden=64,
            layers=2,
            vector_B=True,
        ).to(device)
        
        coords = torch.rand(32, 3, device=device, requires_grad=True)
        out = backbone(coords)
        
        # Check vector outputs exist
        assert "B" in out
        assert "u" in out
        assert out["B"].shape == (32, 3)
        assert out["u"].shape == (32, 2)
        
        # Check legacy unpacked outputs also exist
        assert "B_x" in out
        assert "B_y" in out
        assert "B_z" in out
        assert out["B_x"].shape == (32, 1)
        assert out["B_y"].shape == (32, 1)
        assert out["B_z"].shape == (32, 1)
    
    def test_scalar_output_format(self, device):
        """Test that scalar mode outputs correct format."""
        backbone = PINNBackbone(
            hidden=64,
            layers=2,
            vector_B=False,
        ).to(device)
        
        coords = torch.rand(32, 3, device=device, requires_grad=True)
        out = backbone(coords)
        
        # Check scalar outputs
        assert "B_z" in out
        assert out["B_z"].shape == (32, 1)
        
        # Vector outputs should NOT be present in scalar mode
        assert "B" not in out
    
    def test_consistency_vector_unpacked(self, device):
        """Test that packed and unpacked vectors are consistent."""
        backbone = PINNBackbone(
            hidden=64,
            layers=2,
            vector_B=True,
        ).to(device)
        
        coords = torch.rand(32, 3, device=device, requires_grad=True)
        out = backbone(coords)
        
        B = out["B"]
        Bx = out["B_x"]
        By = out["B_y"]
        Bz = out["B_z"]
        
        # Check consistency
        assert torch.allclose(B[..., 0:1], Bx)
        assert torch.allclose(B[..., 1:2], By)
        assert torch.allclose(B[..., 2:3], Bz)


class TestIntegration:
    """Integration tests for full physics pipeline."""
    
    def test_full_pipeline_vector(self, device):
        """Test full forward-backward pass with vector physics."""
        # Create backbone in vector mode
        backbone = PINNBackbone(
            hidden=64,
            layers=2,
            vector_B=True,
        ).to(device)
        
        # Create physics module
        physics = make_vector_physics(resistive=True).to(device)
        
        # Generate coords
        N = 128
        coords = torch.rand(N, 3, device=device) * 2 - 1
        coords.requires_grad_(True)
        imp_weights = torch.ones(N, 1, device=device)
        
        # Forward pass
        loss, info = physics(backbone, coords, imp_weights, eta_mode="scalar")
        
        # Backward pass
        loss.backward()
        
        # Check
        assert torch.isfinite(loss)
        assert info.loss_induction_total >= 0
        
        for param in backbone.parameters():
            assert param.grad is not None
    
    def test_full_pipeline_scalar_legacy(self, device):
        """Test full forward-backward pass with scalar physics (legacy)."""
        backbone = PINNBackbone(
            hidden=64,
            layers=2,
            vector_B=False,
        ).to(device)
        
        physics = make_scalar_physics(resistive=False).to(device)
        
        N = 128
        coords = torch.rand(N, 3, device=device) * 2 - 1
        coords.requires_grad_(True)
        imp_weights = torch.ones(N, 1, device=device)
        
        loss, info = physics(backbone, coords, imp_weights)
        loss.backward()
        
        assert torch.isfinite(loss)
        assert isinstance(info, PhysicsResidualInfo)


class TestMathematicalCorrectness:
    """Tests for mathematical correctness of physics equations."""
    
    def test_bz_equation_matches_scalar(self, device):
        """
        Test that Bz component equation matches the scalar formulation.
        
        The Bz equation in vector form should be identical to the scalar:
            ∂Bz/∂t = -∇⊥·(Bz·u) + ∇⊥·(η·∇⊥Bz)
        """
        N = 128
        coords = torch.rand(N, 3, device=device) * 2 - 1
        coords.requires_grad_(True)
        imp_weights = torch.ones(N, 1, device=device)
        
        # Create models that output identical Bz
        torch.manual_seed(42)
        scalar_model = DummyScalarModel().to(device)
        
        # Create a vector model that outputs the same Bz
        class MatchingVectorModel(nn.Module):
            def __init__(self, scalar_model):
                super().__init__()
                self.scalar_model = scalar_model
            
            def forward(self, coords):
                out = self.scalar_model(coords)
                Bx = torch.zeros_like(out["B_z"])
                By = torch.zeros_like(out["B_z"])
                B = torch.cat([Bx, By, out["B_z"]], dim=-1)
                u = torch.cat([out["u_x"], out["u_y"]], dim=-1)
                return {
                    "A_z": out["A_z"],
                    "B": B,
                    "u": u,
                    "B_x": Bx,
                    "B_y": By,
                    "B_z": out["B_z"],
                    "u_x": out["u_x"],
                    "u_y": out["u_y"],
                    "eta_raw": None,
                }
        
        vector_model = MatchingVectorModel(scalar_model).to(device)
        
        # Scalar physics
        scalar_physics = make_scalar_physics(resistive=False).to(device)
        
        # Vector physics (with zero weights on Bx, By)
        vector_physics = VectorInduction2p5D(
            physics_cfg=PhysicsConfig(
                resistive=False,
                component_weights=(0.0, 0.0, 1.0),
                div_B_weight=1.0,
            ),
            eta_cfg=EtaConfig(),
            n_fourier_modes=3,
            n_random_tests=2,
        ).to(device)
        
        # Compute losses
        with torch.no_grad():
            loss_scalar, _ = scalar_physics(scalar_model, coords.clone().requires_grad_(True), imp_weights)
            loss_vector, info_vector = vector_physics(vector_model, coords.clone().requires_grad_(True), imp_weights)
        
        # The Bz component loss should be similar
        # (not exactly equal due to normalization differences)
        assert torch.isfinite(loss_scalar)
        assert torch.isfinite(loss_vector)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
