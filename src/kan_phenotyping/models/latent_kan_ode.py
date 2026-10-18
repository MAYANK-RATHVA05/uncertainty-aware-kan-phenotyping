"""A paper-informed latent KAN-ODE extension of the recurrent baseline.

The learned hidden coordinates are not observed clinical states. A shared
autonomous KAN derivative evolves those coordinates between relative study
months; the two observed wearable packets still enter conventional GRU updates.
This is an adaptation of the KAN-ODE idea, not a reproduction of its numerical
solver, basis functions, or dynamical-system experiments.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..numerics.rk4 import RK4Integrator
from .kan_layer import KANLinear
from .sequential_kan import KANEncodedRecurrentBaseline


class LatentKANODE(KANEncodedRecurrentBaseline):
    """Evolve a latent state, assimilate SM1/SM2, then forecast measured SM3.

    The inherited encoders, shared GRUCell, and regression head retain the
    recurrent control's initialization order and implementation. The dynamics
    layer is initialized only after those components: its base and spline
    weights are scaled by 0.01, and its bias starts at zero. Default dimensions
    are static 63, wearable 48, and hidden 16.

    ``dh/dt = dynamics(h)`` is integrated over [0, 1], [1, 2], and [2, 3]. These
    are relative study-month coordinates, not verified dates or 30-day periods.
    No observations arrive after SM2. Static context initializes the state once;
    it is neither a wearable packet nor a recurrent time step.

    ``dynamics`` may be supplied as an autonomous ``nn.Module`` with the same
    input/output hidden shape. This interface separates the derivative from the
    integrator without implementing or fitting an additional model family.
    """

    TRACE_STATE_NAMES = (
        "h0", "h1_before", "h1_after", "h2_before", "h2_after", "h3"
    )

    def __init__(
        self,
        n_wearable_features: int = 48,
        n_static_features: int = 63,
        hidden_width: int = 16,
        *,
        grid_intervals: int = 8,
        spline_degree: int = 3,
        grid_range: tuple[float, float] = (-3.0, 3.0),
        step_size: float = 0.25,
        dynamics: nn.Module | None = None,
    ) -> None:
        super().__init__(
            n_wearable_features=n_wearable_features,
            n_static_features=n_static_features,
            hidden_width=hidden_width,
            grid_intervals=grid_intervals,
            spline_degree=spline_degree,
            grid_range=grid_range,
        )
        self.integrator = RK4Integrator(step_size=step_size)
        if dynamics is None:
            self.dynamics = KANLinear(
                hidden_width,
                hidden_width,
                grid_intervals=grid_intervals,
                spline_degree=spline_degree,
                grid_range=grid_range,
            )
            # Initialization is outside integration. The solver itself retains
            # the complete autograd graph through every RK stage and interval.
            with torch.no_grad():
                self.dynamics.base_weight.mul_(0.01)
                self.dynamics.spline_weight.mul_(0.01)
                self.dynamics.bias.zero_()
        else:
            if not isinstance(dynamics, nn.Module):
                raise TypeError("dynamics must be a torch nn.Module")
            self.dynamics = dynamics

    @property
    def step_size(self) -> float:
        """Configured fixed RK4 step in relative study-month coordinates."""
        return self.integrator.step_size

    @step_size.setter
    def step_size(self, value: float) -> None:
        # RK4Integrator validates finite, strictly positive ordinary reals. A
        # diagnostic may change resolution while preserving solver counters
        # and model parameters; it must explicitly restore the training step.
        self.integrator.step_size = value

    def trace_states(self, wearable: Tensor, static: Tensor) -> Tensor:
        """Return ``[h0,h1_before,h1_after,h2_before,h2_after,h3]``.

        The result has shape ``[N, 6, hidden_width]``. Every call starts afresh
        from that call's static inputs. The SM1 update encodes only SM1; SM2 is
        not passed to the packet encoder until the later update.
        """
        if not isinstance(wearable, Tensor):
            raise TypeError("wearable must be a torch Tensor")
        if wearable.ndim != 3 or wearable.shape[1:] != (2, self.n_wearable_features):
            raise ValueError(
                f"wearable must have shape [batch, 2, {self.n_wearable_features}]"
            )
        self._matrix(static, self.n_static_features, "static")
        self._matching_tensors(wearable, static, "wearable and static")
        h0 = self.initial_state(static)
        h1_before = self.integrator.integrate(self.dynamics, h0, 0.0, 1.0)
        h1_after = self.step(wearable[:, 0, :], h1_before)
        h2_before = self.integrator.integrate(self.dynamics, h1_after, 1.0, 2.0)
        h2_after = self.step(wearable[:, 1, :], h2_before)
        h3 = self.integrator.integrate(self.dynamics, h2_after, 2.0, 3.0)
        return torch.stack((h0, h1_before, h1_after, h2_before, h2_after, h3), dim=1)

    def forward(self, wearable: Tensor, static: Tensor) -> Tensor:
        """Return unbounded scalar SM3 estimates; no intermediate PHQ labels."""
        return self.head(self.trace_states(wearable, static)[:, -1, :]).squeeze(-1)
