"""Export validation-only retrospective model associations from a trusted run.

Learned KAN edges, feature responses, and permutation importance are separate
analyses. No output establishes causal effects, clinical validity, or diagnosis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import build_model, load_run, sha256_file
from ..interpretability import (
    conditional_feature_responses,
    export_feature_responses,
    export_permutation_importance,
    plot_selected_spline_edges,
    validation_permutation_importance,
)
from ..models import KANLinear
from ..training.trainer import write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", required=True, type=Path, help="Trusted local run with best.pt and preprocessor.joblib")
    result.add_argument("--data", required=True, type=Path)
    result.add_argument("--dictionary", required=True, type=Path)
    result.add_argument("--output", type=Path, help="New empty directory; default: RUN_DIR/interpretation")
    result.add_argument("--n-repeats", type=int, default=5)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--max-features", type=int, help="Describe the first N original feature groups only; ordering is fixed before interpretation")
    result.add_argument("--feature", action="append", help="Exact transformed numeric feature to plot; repeat up to four times. Default: top four numeric features by validation permutation score")
    result.add_argument("--n-points", type=int, default=30)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--num-threads", type=int, default=2)
    return result


def main(argv: list[str] | None = None) -> dict:
    args = parser().parse_args(argv)
    if any(value < 1 for value in (args.n_repeats, args.batch_size, args.num_threads)):
        raise ValueError("n-repeats, batch-size, and num-threads must be positive.")
    if args.n_points < 2 or (args.max_features is not None and args.max_features < 1):
        raise ValueError("n-points must be >= 2 and max-features must be positive.")
    output = args.output or args.run_dir / "interpretation"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new, empty interpretation output directory: {output}")
    # Fingerprints and saved splits are checked before trusted joblib is loaded.
    prepared = load_run(args.run_dir, args.data, args.dictionary)
    saved = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    if saved["model_metadata"] != prepared.metadata["model"]:
        raise ValueError("Best-checkpoint model specification differs from the saved run.")
    model = build_model(saved["model_metadata"])
    model.load_state_dict(saved["model_state"])
    model.eval()
    torch.set_num_threads(args.num_threads)
    features = prepared.features["validation"].tabular
    names = list(prepared.features["validation"].feature_names["tabular"])
    target = prepared.examples.y[prepared.split.validation]
    parameter = next(model.parameters())

    def predict(values: np.ndarray) -> np.ndarray:
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(values), args.batch_size):
                batch = torch.as_tensor(values[start:start + args.batch_size], dtype=parameter.dtype, device=parameter.device)
                predictions.append(model(batch).detach().cpu().numpy())
        return np.concatenate(predictions)

    # Re-export fitted transformation metadata so older runs also receive the
    # current, exact raw-to-transformed column mapping without refitting.
    transformations = prepared.preprocessor.export_metadata()
    all_groups = transformations["permutation_groups"]
    group_items = list(all_groups.items())
    selected_groups = dict(group_items[:args.max_features] if args.max_features else group_items)
    selected = [index for group in selected_groups.values() for index in group]
    importance = validation_permutation_importance(
        predict, features, target, names, n_repeats=args.n_repeats, seed=args.seed,
        permutation_groups=selected_groups, transformation_metadata=transformations,
    )
    # Individual dummy-column replacement can invalidate categorical encodings.
    # Restrict automatic response/edge selection to genuine numeric coordinates.
    numeric_names = {
        name for name in names
        if name.startswith("numeric__") and not name.startswith("numeric__missingindicator_")
    }
    if args.feature:
        requested = list(dict.fromkeys(args.feature))
        if len(requested) > 4 or any(name not in numeric_names for name in requested):
            raise ValueError("Select at most four exact transformed numeric feature names (excluding indicators).")
        plotted = [names.index(name) for name in requested]
    else:
        plotted = [int(row.feature_index) for row in importance.summary.itertuples() if names[int(row.feature_index)] in numeric_names][:4]
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, object] = {
        "permutation_importance": {key: str(value) for key, value in export_permutation_importance(importance, output).items()}
    }
    if plotted:
        responses = conditional_feature_responses(
            predict, features, names, plotted, n_points=args.n_points,
            transformation_metadata=transformations,
        )
        artifacts["feature_responses"] = {key: str(value) for key, value in export_feature_responses(responses, output).items()}
    first_layer = model.layers[0]
    edge_indices = []
    ranges = {}
    for index in plotted:
        low, high = np.quantile(features[:, index], [0.05, 0.95])
        if low < high:
            edge_indices.append((index, 0))
            ranges[names[index]] = (float(low), float(high))
    if isinstance(first_layer, KANLinear) and edge_indices:
        edges = plot_selected_spline_edges(
            first_layer, edge_indices, names, output, feature_ranges=ranges,
            transformation_metadata=transformations,
        )
        artifacts["learned_edges"] = {key: str(value) for key, value in edges.paths.items()}
    manifest = {
        "analysis_type": "exploratory validation-only retrospective associations",
        "split": "validation",
        "checkpoint_sha256": sha256_file(args.run_dir / "best.pt"),
        "input_fingerprints": prepared.metadata["inputs"],
        "model": prepared.metadata["model"],
        "seed": args.seed,
        "n_repeats": args.n_repeats,
        "permutation_feature_indices": selected,
        "permutation_groups": selected_groups,
        "importance_unit": "original feature: all one-hot columns or numeric value plus missingness indicator shuffled together",
        "all_feature_count": len(names),
        "response_feature_indices": plotted,
        "edge_output_unit": "first hidden unit, fixed in advance; not selected using test data",
        "selection_rule": "explicit numeric coordinates if supplied; otherwise top four numeric features among evaluated validation grouped-permutation scores",
        "units": "input coordinates transformed; regression output and RMSE in PHQ-9 change points",
        "artifacts": artifacts,
        "limitations": [
            *prepared.metadata["limitations"],
            "Permutation shuffles observed rows; repeated-row/provisional-group dependence is not preserved and participant-level inference is unsupported.",
            "A shared permutation preserves one-hot categories and numeric/missingness pairs within each original feature.",
            "Individual transformed coordinates are varied with others fixed; correlated features may leave joint data support.",
            "Permutation importance is a separate method from spline shape; neither implies causality or clinical significance.",
            "No held-out test predictions or test-derived feature selections are used in interpretation.",
            "Missing learned-edge output means the first layer is not KAN or selected numeric features are constant over the plotted range.",
        ],
    }
    write_json(output / "interpretation.json", manifest)
    print(f"Saved validation-only model associations and publication exports to {output}")
    return manifest


if __name__ == "__main__":
    main()
