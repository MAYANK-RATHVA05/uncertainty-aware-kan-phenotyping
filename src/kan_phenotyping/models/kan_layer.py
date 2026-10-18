"""Fixed-grid Kolmogorov-Arnold layers implemented with PyTorch operations.

Each edge has a SiLU base term and a learned B-spline expansion. These are
implementation choices, not a claim to reproduce adaptive-grid KAN training.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class KANLinear(nn.Module):
    """Compute ``y_j = b_j + sum_i [w_ji SiLU(x_i) + sum_r c_jir B_r(x_i)]``.

    ``grid_intervals`` intervals cover ``grid_range``. Extending this grid by
    ``spline_degree`` knots at each end gives ``grid_intervals + spline_degree``
    basis functions per input. For the defaults this is 11 cubic basis functions,
    with partition of unity on [-3, 3], including the endpoints. Outside that
    interval the finite spline expansion tapers to zero while the base term
    continues to extrapolate. There is no input clipping or adaptive grid update.

    ``feature_mask`` excludes missing/padded input contributions, including their
    spline values at zero; a zero-valued input alone does not exclude an edge.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_intervals: int = 8,
        spline_degree: int = 3,
        grid_range: tuple[float, float] = (-3.0, 3.0),
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")
        if grid_intervals < 1 or spline_degree < 0:
            raise ValueError("grid_intervals must be positive and spline_degree nonnegative")
        lo, hi = grid_range
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise ValueError("grid_range must contain finite, increasing bounds")
        self.in_features = in_features
        self.out_features = out_features
        self.grid_intervals = grid_intervals
        self.spline_degree = spline_degree
        self.grid_range = (float(lo), float(hi))
        self.n_basis = grid_intervals + spline_degree
        spacing = (hi - lo) / grid_intervals
        knots = lo + torch.arange(-spline_degree, grid_intervals + spline_degree + 1) * spacing
        self.register_buffer("knots", knots)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, self.n_basis))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        # Keep the initially random spline contribution modest relative to the base.
        scale = 0.1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.spline_weight, -scale, scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def b_spline_basis(self, x: Tensor) -> Tensor:
        """Evaluate Cox-de Boor bases, returning ``[..., in_features, n_basis]``."""
        self._check_input(x)
        return self._basis_values(x)

    def _basis_values(self, x: Tensor) -> Tensor:
        values = x.unsqueeze(-1)
        knots = self.knots
        basis = ((values >= knots[:-1]) & (values < knots[1:])).to(x.dtype)
        for degree in range(1, self.spline_degree + 1):
            left = (values - knots[: -(degree + 1)]) / (
                knots[degree:-1] - knots[: -(degree + 1)]
            )
            right = (knots[degree + 1 :] - values) / (
                knots[degree + 1 :] - knots[1:-degree]
            )
            basis = left * basis[..., :-1] + right * basis[..., 1:]
        return basis

    def edge_response(self, feature_index: int, output_index: int, values: Tensor) -> Tensor:
        """Evaluate one full learned edge on arbitrary-shaped floating values.

        Includes the SiLU base and spline terms, excluding the neuron bias,
        which is added once after summing all input-edge responses.
        """
        return self.edge_components(feature_index, output_index, values)["total"]

    def edge_components(
        self, feature_index: int, output_index: int, values: Tensor
    ) -> dict[str, Tensor]:
        """Return separate learned ``base``, ``spline``, and ``total`` edge curves.

        Neuron bias is excluded from all three components.
        """
        if not 0 <= feature_index < self.in_features:
            raise IndexError("feature_index is out of range")
        if not 0 <= output_index < self.out_features:
            raise IndexError("output_index is out of range")
        if not values.is_floating_point():
            raise TypeError("edge response inputs must be floating point")
        base = self.base_weight[output_index, feature_index] * F.silu(values)
        splines = (
            self._basis_values(values) * self.spline_weight[output_index, feature_index]
        ).sum(dim=-1)
        return {"base": base, "spline": splines, "total": base + splines}

    def _check_input(self, x: Tensor) -> None:
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"expected last input dimension {self.in_features}, got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("KAN inputs must be floating point")

    def forward(self, x: Tensor, feature_mask: Tensor | None = None) -> Tensor:
        self._check_input(x)
        if feature_mask is not None:
            if feature_mask.shape != x.shape or feature_mask.dtype != torch.bool:
                raise ValueError("feature_mask must be boolean and have the same shape as x")
            # masked_fill also removes NaN sentinels in padded inputs.
            x = x.masked_fill(~feature_mask, 0.0)
        base_inputs = F.silu(x)
        basis = self.b_spline_basis(x)
        if feature_mask is not None:
            base_inputs = base_inputs * feature_mask
            basis = basis * feature_mask.unsqueeze(-1)
        base = F.linear(base_inputs, self.base_weight, self.bias)
        splines = F.linear(
            basis.flatten(start_dim=-2), self.spline_weight.flatten(start_dim=1)
        )
        return base + splines

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"grid_intervals={self.grid_intervals}, spline_degree={self.spline_degree}, "
            f"grid_range={self.grid_range}, bias={self.bias is not None}"
        )
