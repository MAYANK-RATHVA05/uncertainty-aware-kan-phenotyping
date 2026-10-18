"""A KAN-encoded recurrent control for conditional relative-month forecasting.

Static context initializes the hidden state once. Two observed wearable packets
then enter the same KAN encoder and conventional GRU cell in SM1, SM2 order.
This is an observation-by-observation recurrent model, not KAN-ODE or a
replication of a multi-scale temporal convolution architecture.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .kan_layer import KANLinear


class KANEncodedRecurrentBaseline(nn.Module):
    """Encode context and packets with KAN layers, update state with a GRU.

    The forecasting experiment uses wearable ``[N, 2, 48]`` and static
    ``[N, 63]`` inputs with hidden width 16. The three KAN layers inherit the
    existing fixed-grid spline implementation without extra activations,
    normalization, clipping, or output constraints. Each example starts from
    its own static context; hidden state is never retained between calls.

    ``initial_state`` and ``step`` expose the prefix computation directly. The
    first update receives only SM1 and its static-derived hidden state; SM2 is
    passed to the shared encoder only when computing the second update.
    """

    def __init__(
        self,
        n_wearable_features: int = 48,
        n_static_features: int = 63,
        hidden_width: int = 16,
        *,
        grid_intervals: int = 8,
        spline_degree: int = 3,
        grid_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        for name, value in (
            ("n_wearable_features", n_wearable_features),
            ("n_static_features", n_static_features),
            ("hidden_width", hidden_width),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.n_wearable_features = n_wearable_features
        self.n_static_features = n_static_features
        self.hidden_width = hidden_width
        grid_options = dict(
            grid_intervals=grid_intervals,
            spline_degree=spline_degree,
            grid_range=grid_range,
        )
        self.static_encoder = KANLinear(n_static_features, hidden_width, **grid_options)
        self.wearable_encoder = KANLinear(n_wearable_features, hidden_width, **grid_options)
        self.gru = nn.GRUCell(hidden_width, hidden_width)
        self.head = KANLinear(hidden_width, 1, **grid_options)

    @staticmethod
    def _matrix(value: Tensor, width: int, name: str) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch Tensor")
        if value.ndim != 2 or value.shape[1] != width or value.shape[0] < 1:
            raise ValueError(f"{name} must have shape [nonempty batch, {width}]")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")

    @staticmethod
    def _matching_tensors(first: Tensor, second: Tensor, names: str) -> None:
        if first.shape[0] != second.shape[0]:
            raise ValueError(f"{names} must have the same batch size")
        if first.device != second.device or first.dtype != second.dtype:
            raise ValueError(f"{names} must have matching device and dtype")

    def initial_state(self, static: Tensor) -> Tensor:
        """Map enrollment demographics and measured SM0 to the initial state."""
        self._matrix(static, self.n_static_features, "static")
        return self.static_encoder(static)

    def step(self, packet: Tensor, hidden: Tensor) -> Tensor:
        """Consume one observed packet with the shared encoder and GRU cell."""
        self._matrix(packet, self.n_wearable_features, "packet")
        self._matrix(hidden, self.hidden_width, "hidden")
        self._matching_tensors(packet, hidden, "packet and hidden")
        return self.gru(self.wearable_encoder(packet), hidden)

    def trace_states(self, wearable: Tensor, static: Tensor) -> Tensor:
        """Return ``[h0, h_after_SM1, h_after_SM2]`` as ``[N, 3, hidden]``."""
        if not isinstance(wearable, Tensor):
            raise TypeError("wearable must be a torch Tensor")
        if wearable.ndim != 3 or wearable.shape[1:] != (2, self.n_wearable_features):
            raise ValueError(
                f"wearable must have shape [batch, 2, {self.n_wearable_features}]"
            )
        self._matrix(static, self.n_static_features, "static")
        self._matching_tensors(wearable, static, "wearable and static")
        # Do not pre-encode the entire sequence: SM2 is only encoded after the
        # first GRU update is complete. Static context is not a recurrent step.
        hidden = self.initial_state(static)
        states = [hidden]
        for month_position in range(2):
            hidden = self.step(wearable[:, month_position, :], hidden)
            states.append(hidden)
        return torch.stack(states, dim=1)

    def forward(self, wearable: Tensor, static: Tensor) -> Tensor:
        """Return one unbounded measured-SM3 PHQ-9 severity estimate per sample."""
        final_hidden = self.trace_states(wearable, static)[:, -1, :]
        return self.head(final_hidden).squeeze(-1)
