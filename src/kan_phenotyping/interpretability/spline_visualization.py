"""Selected learned KAN edge curves, distinct from input-feature importance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .feature_analysis import (
    ASSOCIATION_WARNING,
    _export_directory,
    _save_figure,
    _save_metadata,
)


@dataclass
class SplineVisualizationResult:
    curves: pd.DataFrame
    metadata: dict[str, Any]
    paths: dict[str, Path]


def plot_selected_spline_edges(
    layer: Any,
    selected_edges: Sequence[tuple[int, int]],
    feature_names: Sequence[str],
    output_dir: str | Path,
    *,
    feature_ranges: Mapping[str, tuple[float, float]] | None = None,
    transformation_metadata: Mapping[str, Any] | None = None,
    output_names: Sequence[str] | None = None,
    n_points: int = 200,
    prefix: str = "learned_kan_edges",
    dpi: int = 300,
) -> SplineVisualizationResult:
    """Plot selected ``(input_index, output_index)`` learned spline+SiLU edges.

    Shared output-neuron bias is recorded separately and excluded from every
    edge, because adding it to every edge would count it repeatedly. A learned
    edge is not a feature-importance estimate and, in deeper layers, its input
    need not correspond to a raw observed feature. Supply actual layer-input
    names. Input values remain in transformed units; no inverse map is assumed.
    When no explicit range is provided, use the layer's declared spline range.
    """
    import matplotlib.pyplot as plt

    if not hasattr(layer, "edge_components"):
        raise TypeError("layer must expose edge_components(input_index, output_index, values).")
    if len(feature_names) != layer.in_features:
        raise ValueError("feature_names must name every actual layer input.")
    names = list(feature_names)
    if len(set(names)) != len(names):
        raise ValueError("Layer-input feature names must be unique.")
    outputs = list(output_names) if output_names is not None else [f"unit_{i}" for i in range(layer.out_features)]
    if len(outputs) != layer.out_features:
        raise ValueError("output_names must name every layer output.")
    edges = list(selected_edges)
    if not 1 <= len(edges) <= 12 or len(set(edges)) != len(edges):
        raise ValueError("Select 1--12 distinct learned edges per figure.")
    for edge in edges:
        if len(edge) != 2 or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in edge):
            raise ValueError("Each edge is an (input_index, output_index) integer pair.")
        if not 0 <= edge[0] < layer.in_features or not 0 <= edge[1] < layer.out_features:
            raise ValueError("A selected edge index is outside the layer dimensions.")
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 2 or dpi < 300:
        raise ValueError("n_points must be >= 2 and dpi must be >= 300.")
    parameter = next(layer.parameters())
    records: list[dict[str, Any]] = []
    edge_metadata: list[dict[str, Any]] = []
    degree = int(layer.spline_degree)
    default_range = layer.grid_range
    for input_index, output_index in edges:
        low, high = (feature_ranges or {}).get(names[input_index], default_range)
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("Feature ranges must be finite (low, high) pairs with low < high.")
        values = torch.linspace(low, high, n_points, dtype=parameter.dtype, device=parameter.device)
        with torch.no_grad():
            components = layer.edge_components(input_index, output_index, values)
            total, spline, base = components["total"], components["spline"], components["base"]
        if not all(torch.isfinite(curve).all() for curve in (total, spline, base)):
            raise ValueError("Learned edge produced non-finite responses.")
        for x, edge, spl, bas in zip(
            values.detach().cpu().numpy(), total.detach().cpu().numpy(),
            spline.detach().cpu().numpy(), base.detach().cpu().numpy(),
        ):
            records.append(
                {
                    "feature": names[input_index], "feature_index": int(input_index),
                    "output_unit": outputs[output_index], "output_index": int(output_index),
                    "input_value": float(x), "edge_response": float(edge),
                    "spline_component": float(spl), "silu_component": float(bas),
                }
            )
        bias = getattr(layer, "bias", None)
        edge_metadata.append(
            {
                "input_index": int(input_index), "input_name": names[input_index],
                "output_index": int(output_index), "output_name": outputs[output_index],
                "input_range": [float(low), float(high)],
                "shared_neuron_bias_excluded_from_curve": None if bias is None else float(bias[output_index].detach().cpu()),
            }
        )
    curves = pd.DataFrame.from_records(records)
    metadata = {
        "method": "selected learned univariate KAN edge curves: B-spline plus weighted SiLU",
        "feature_names": names,
        "selected_edges": edge_metadata,
        "degree": degree,
        "n_basis": int(layer.spline_weight.shape[-1]),
        "knots": layer.knots.detach().cpu().tolist(),
        "input_units": "transformed layer-input units; inverse mapping not assumed",
        "output_units": "layer edge contribution units, excluding shared neuron bias",
        "transformations": dict(transformation_metadata or {}),
        "limitations": [
            ASSOCIATION_WARNING,
            "Spline shape is distinct from permutation feature importance and does not establish clinical significance.",
            "Curves across unsupported or categorical intermediate values need not describe observed data.",
            "Deeper-layer coordinates are learned representations, not automatically raw feature effects.",
        ],
    }
    directory = _export_directory(output_dir, prefix)
    paths = {"curves": directory / f"{prefix}.csv", "metadata": directory / f"{prefix}.json"}
    curves.to_csv(paths["curves"], index=False)
    _save_metadata(paths["metadata"], metadata)
    n_columns = min(2, len(edges))
    n_rows = (len(edges) + n_columns - 1) // n_columns
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axes = plt.subplots(n_rows, n_columns, figsize=(6 * n_columns, 3.5 * n_rows), squeeze=False)
        try:
            for axis, (input_index, output_index) in zip(axes.ravel(), edges):
                curve = curves[(curves["feature_index"] == input_index) & (curves["output_index"] == output_index)]
                axis.plot(curve["input_value"], curve["edge_response"], color="#246a8d", label="Spline + SiLU edge")
                axis.plot(curve["input_value"], curve["spline_component"], color="#a34522", linestyle="--", label="Spline component")
                axis.set_xlabel(f"{names[input_index]} (transformed units)")
                axis.set_ylabel("Learned edge contribution")
                axis.set_title(f"{names[input_index]} → {outputs[output_index]}")
                axis.spines[["top", "right"]].set_visible(False)
                axis.legend(frameon=False, fontsize=8)
            for axis in axes.ravel()[len(edges):]:
                axis.set_visible(False)
            figure.suptitle("Selected learned KAN edge functions")
            figure.text(0.01, 0.01, "Associations only; no causal, clinical-significance, or diagnostic interpretation. Shared neuron bias excluded.", fontsize=8)
            figure.tight_layout(rect=(0, 0.04, 1, 0.96))
            paths.update(_save_figure(figure, directory, prefix, dpi))
        finally:
            plt.close(figure)
    return SplineVisualizationResult(curves, metadata, paths)
