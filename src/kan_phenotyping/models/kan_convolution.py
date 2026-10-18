"""Causal temporal convolution whose channel/lag edges use KAN functions."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .kan_layer import KANLinear


def validate_temporal_inputs(x: Tensor, mask: Tensor | None) -> Tensor:
    """Validate channel-first sequences and return their boolean time mask."""
    if x.ndim != 3 or x.shape[-1] == 0:
        raise ValueError("expected nonempty temporal input shaped [batch, channels, time]")
    if not x.is_floating_point():
        raise TypeError("temporal inputs must be floating point")
    expected_shape = (x.shape[0], x.shape[-1])
    if mask is None:
        return torch.ones(expected_shape, dtype=torch.bool, device=x.device)
    if mask.shape != expected_shape or mask.dtype != torch.bool:
        raise ValueError(f"mask must be boolean with shape {expected_shape}")
    if mask.device != x.device:
        raise ValueError("mask and x must be on the same device")
    return mask


class KANConv1d(nn.Module):
    """Stride-one causal KAN convolution with preserved sequence length.

    Input/output layout is [batch, channels, time]. Left padding and masked
    timesteps contribute no edge functions, and outputs at invalid timesteps
    are zero. The mask describes observed timesteps, not individual features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        grid_intervals: int = 8,
        spline_degree: int = 3,
        grid_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.edge_functions = KANLinear(
            in_channels * kernel_size,
            out_channels,
            grid_intervals=grid_intervals,
            spline_degree=spline_degree,
            grid_range=grid_range,
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        mask = validate_temporal_inputs(x, mask)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels")
        batch, channels, length = x.shape
        x = x.masked_fill(~mask.unsqueeze(1), 0.0)
        padded = F.pad(x, (self.kernel_size - 1, 0))
        patches = padded.unfold(-1, self.kernel_size, 1)
        patches = patches.permute(0, 2, 1, 3).reshape(batch, length, -1)
        padded_mask = F.pad(mask, (self.kernel_size - 1, 0), value=False)
        patch_mask = padded_mask.unfold(-1, self.kernel_size, 1)
        patch_mask = patch_mask.unsqueeze(2).expand(-1, -1, channels, -1)
        patch_mask = patch_mask.reshape(batch, length, -1)
        output = self.edge_functions(patches, feature_mask=patch_mask)
        return output.transpose(1, 2).masked_fill(~mask.unsqueeze(1), 0.0)
