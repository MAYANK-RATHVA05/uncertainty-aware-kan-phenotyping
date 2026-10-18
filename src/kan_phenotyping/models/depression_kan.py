"""Temporal and explicitly tabular depression prediction models.

All models return one raw scalar per example. With task='regression' it is an
unbounded PHQ change estimate. With task='binary' it is a logit, suitable for
BCEWithLogitsLoss; callers may apply sigmoid when reporting probabilities.
The model never thresholds or converts a continuous target into a class.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .kan_convolution import KANConv1d, validate_temporal_inputs
from .kan_layer import KANLinear


def _validate_task(task: str) -> None:
    if task not in {"regression", "binary"}:
        raise ValueError("task must be 'regression' or 'binary'")


def _mixing_layer(in_features: int, out_features: int, mixing_type: str, grid_options: dict) -> nn.Module:
    if mixing_type == "kan":
        return KANLinear(in_features, out_features, **grid_options)
    if mixing_type == "mlp":
        return nn.Sequential(nn.Linear(in_features, out_features), nn.SiLU())
    raise ValueError("mixing_type must be 'kan' or 'mlp'")


def _join_static(dynamic: Tensor, static: Tensor | None, n_static_features: int) -> Tensor:
    if n_static_features:
        if static is None or static.shape != (dynamic.shape[0], n_static_features):
            raise ValueError(f"static must have shape [batch, {n_static_features}]")
        return torch.cat((dynamic, static), dim=-1)
    if static is not None and static.shape != (dynamic.shape[0], 0):
        raise ValueError("this model was configured without static covariates")
    return dynamic


def _sequence_input(
    dynamic: Tensor, mask: Tensor | None, n_dynamic_features: int
) -> tuple[Tensor, Tensor]:
    if dynamic.ndim != 3 or dynamic.shape[-1] != n_dynamic_features:
        raise ValueError(f"dynamic must have shape [batch, time, {n_dynamic_features}]")
    channel_first = dynamic.transpose(1, 2)
    mask = validate_temporal_inputs(channel_first, mask)
    if not mask.any(dim=1).all():
        raise ValueError("each example must contain at least one valid timestep")
    return channel_first, mask


def _masked_average(features: Tensor, mask: Tensor) -> Tensor:
    features = features.masked_fill(~mask.unsqueeze(1), 0.0)
    return features.sum(dim=-1) / mask.sum(dim=-1, keepdim=True).to(features.dtype)


class TemporalDepressionKAN(nn.Module):
    """Three causal KAN branches, masked GAP, KAN mixing, and a scalar KAN head.

    The default kernels 3/5/7 follow the supplied architecture. Allocating the
    requested 64 total branch channels as 22/21/21 is an explicit implementation
    choice. Mixing has width 64. Static covariates enter once, after mixing.
    Grid settings and initial weights are implementation choices, not reported
    paper defaults. Real temporal training requires confirmed sequence identity
    and ordering; this class does not infer them from row indices.
    """

    def __init__(
        self,
        n_dynamic_features: int,
        n_static_features: int = 0,
        *,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        branch_channels: tuple[int, ...] = (22, 21, 21),
        mixing_width: int = 64,
        grid_intervals: int = 8,
        spline_degree: int = 3,
        grid_range: tuple[float, float] = (-3.0, 3.0),
        mixing_type: str = "kan",
        task: str = "regression",
    ) -> None:
        super().__init__()
        _validate_task(task)
        if n_dynamic_features < 1 or n_static_features < 0:
            raise ValueError("dynamic feature count must be positive; static count must be nonnegative")
        if not kernel_sizes or len(kernel_sizes) != len(branch_channels):
            raise ValueError("kernel_sizes and branch_channels must be nonempty and have equal length")
        self.n_dynamic_features = n_dynamic_features
        self.n_static_features = n_static_features
        self.task = task
        grid_options = dict(
            grid_intervals=grid_intervals,
            spline_degree=spline_degree,
            grid_range=grid_range,
        )
        self.branches = nn.ModuleList(
            KANConv1d(n_dynamic_features, channels, kernel, **grid_options)
            for kernel, channels in zip(kernel_sizes, branch_channels)
        )
        self.mixing = _mixing_layer(sum(branch_channels), mixing_width, mixing_type, grid_options)
        self.head = KANLinear(mixing_width + n_static_features, 1, **grid_options)

    def forward(
        self,
        dynamic: Tensor,
        static: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        x, mask = _sequence_input(dynamic, mask, self.n_dynamic_features)
        pooled = torch.cat([_masked_average(branch(x, mask), mask) for branch in self.branches], dim=-1)
        mixed = self.mixing(pooled)
        return self.head(_join_static(mixed, static, self.n_static_features)).squeeze(-1)


class TabularDepressionKAN(nn.Module):
    """Separate tabular KAN with default widths n_features -> 64 -> 1.

    It makes no temporal-convolution claim and expects an already prepared
    feature matrix; preprocessing must be fitted on training participants only.
    """

    def __init__(
        self,
        n_features: int,
        hidden_width: int = 64,
        *,
        mixing_type: str = "kan",
        task: str = "regression",
        **grid_options: object,
    ) -> None:
        super().__init__()
        _validate_task(task)
        self.task = task
        self.layers = nn.Sequential(
            _mixing_layer(n_features, hidden_width, mixing_type, grid_options),
            KANLinear(hidden_width, 1, **grid_options),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x).squeeze(-1)


class MLPBaseline(nn.Module):
    """Conventional tabular MLP baseline returning raw scalars."""

    def __init__(self, n_features: int, hidden_width: int = 64, *, task: str = "regression") -> None:
        super().__init__()
        _validate_task(task)
        self.task = task
        self.layers = nn.Sequential(nn.Linear(n_features, hidden_width), nn.SiLU(), nn.Linear(hidden_width, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x).squeeze(-1)


class TemporalCNNBaseline(nn.Module):
    """Conventional causal CNN with the same branch/pooling/static interface."""

    def __init__(
        self,
        n_dynamic_features: int,
        n_static_features: int = 0,
        *,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        branch_channels: tuple[int, ...] = (22, 21, 21),
        mixing_width: int = 64,
        task: str = "regression",
    ) -> None:
        super().__init__()
        _validate_task(task)
        if n_dynamic_features < 1 or n_static_features < 0:
            raise ValueError("dynamic feature count must be positive; static count must be nonnegative")
        if not kernel_sizes or len(kernel_sizes) != len(branch_channels):
            raise ValueError("kernel_sizes and branch_channels must be nonempty and have equal length")
        self.n_dynamic_features = n_dynamic_features
        self.n_static_features = n_static_features
        self.task = task
        self.kernel_sizes = kernel_sizes
        self.branches = nn.ModuleList(
            nn.Conv1d(n_dynamic_features, channels, kernel)
            for kernel, channels in zip(kernel_sizes, branch_channels)
        )
        self.mixing = nn.Sequential(nn.Linear(sum(branch_channels), mixing_width), nn.SiLU())
        self.head = nn.Linear(mixing_width + n_static_features, 1)

    def forward(self, dynamic: Tensor, static: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        x, mask = _sequence_input(dynamic, mask, self.n_dynamic_features)
        x = x.masked_fill(~mask.unsqueeze(1), 0.0)
        branch_features = [
            _masked_average(F.silu(branch(F.pad(x, (kernel - 1, 0)))), mask)
            for branch, kernel in zip(self.branches, self.kernel_sizes)
        ]
        mixed = self.mixing(torch.cat(branch_features, dim=-1))
        return self.head(_join_static(mixed, static, self.n_static_features)).squeeze(-1)
