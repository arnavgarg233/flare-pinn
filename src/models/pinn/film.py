# src/models/pinn/film.py
"""
Feature-wise Linear Modulation (FiLM) for conditioning neural networks.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    FiLM: Feature-wise Linear Modulation.
    
    Applies affine transformation to hidden activations:
        FiLM(h; gamma, beta) = gamma ⊙ h + beta
    
    where gamma and beta are predicted from conditioning code.
    
    References:
        Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)
    """
    def __init__(self, hidden_dim: int, code_dim: int):
        super().__init__()
        # Predict both scale (gamma) and shift (beta)
        self.mlp = nn.Linear(code_dim, hidden_dim * 2)
        
        # Initialize with identity-like behavior
        with torch.no_grad():
            self.mlp.weight.fill_(0.0)
            self.mlp.bias[:hidden_dim].fill_(1.0)  # gamma = 1
            self.mlp.bias[hidden_dim:].fill_(0.0)  # beta = 0
    
    def forward(self, h: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [N, hidden_dim] - Hidden activations
            code: [N, code_dim] - Conditioning code (e.g., global features)
        
        Returns:
            modulated: [N, hidden_dim] - FiLM-modulated activations
        """
        gam_beta = self.mlp(code)  # [N, 2*hidden_dim]
        gamma, beta = gam_beta.chunk(2, dim=-1)
        
        # Optional: clamp gamma to prevent instability
        gamma = torch.tanh(gamma) * 2.0 + 1.0  # Range: [-1, 3]
        
        return gamma * h + beta

