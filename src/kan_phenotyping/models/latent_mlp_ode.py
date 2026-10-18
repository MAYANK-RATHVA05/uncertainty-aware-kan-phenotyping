"""Replace only the latent derivative of the existing KAN-encoded ODE model."""

from __future__ import annotations

import torch
from torch import nn

from .latent_kan_ode import LatentKANODE


class LatentMLPODE(LatentKANODE):
    """Same KAN encoders/head, GRU updates and RK4 schedule; MLP dynamics only.

    Shared components are initialized before the MLP consumes random draws.
    Standard PyTorch Linear initialization is retained except that the final
    layer's weights are scaled by 0.01 and its bias is set to zero. This does
    not match the KAN derivative's function-space initialization.
    """

    def __init__(self, *, step_size: float = 0.25) -> None:
        # A parameter-free, RNG-free placeholder lets the inherited constructor
        # initialize shared components without first constructing KAN dynamics.
        # It is replaced before this constructor returns and is never evaluated.
        super().__init__(step_size=step_size, dynamics=nn.Identity())
        self.dynamics = nn.Sequential(nn.Linear(16, 93), nn.Tanh(), nn.Linear(93, 16))
        with torch.no_grad():
            self.dynamics[2].weight.mul_(0.01)
            self.dynamics[2].bias.zero_()
