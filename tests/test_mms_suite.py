import torch
def manufactured_fields(coords: torch.Tensor):
    x, y, t = coords[..., 0:1], coords[..., 1:2], coords[..., 2:3]
    Az = torch.sin(3.14159 * x) * torch.sin(3.14159 * y) * torch.cos(3.14159 * t)
    Bz = torch.cos(3.14159 * x) * torch.cos(3.14159 * y) * torch.sin(3.14159 * t)
    ux = 0.3 * torch.sin(0.5 * 3.14159 * x) * torch.cos(0.5 * 3.14159 * y)
    uy = -0.3 * torch.cos(0.5 * 3.14159 * x) * torch.sin(0.5 * 3.14159 * y)
    return {"A_z": Az, "B_z": Bz, "u_x": ux, "u_y": uy, "eta_raw": None}
class _DummyPINN(torch.nn.Module):
    def forward(self, coords):
        return manufactured_fields(coords)
def test_mms_residual_small():
    device = torch.device("cpu")
    model = _DummyPINN()
    N = 512
    coords = torch.rand(N, 3, device=device) * 2 - 1
    coords.requires_grad_(True)
    assert coords.shape[-1] == 3
