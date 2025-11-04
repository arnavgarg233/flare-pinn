import torch
def B_perp_from_Az(Az, coords):
    ones = torch.ones_like(Az)
    dA_dx = torch.autograd.grad(Az, coords, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0][...,0:1]
    dA_dy = torch.autograd.grad(Az, coords, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0][...,1:2]
    Bx = -dA_dy
    By = dA_dx
    return Bx, By
def test_in_plane_divergence_zero():
    N = 1024
    coords = torch.rand(N,3,requires_grad=True)
    Az = torch.sin(3.1415*coords[...,0:1]) * torch.cos(3.1415*coords[...,1:2])
    Bx, By = B_perp_from_Az(Az, coords)
    ones = torch.ones_like(Bx)
    gx = torch.autograd.grad(Bx, coords, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0][...,0:1]
    gy = torch.autograd.grad(By, coords, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0][...,1:2]
    divp = gx + gy
    assert float(divp.abs().mean().item()) < 1e-2
