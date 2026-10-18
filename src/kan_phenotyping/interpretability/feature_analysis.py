"""Validation-set associations, kept separate from learned KAN edge curves.

These methods describe a fitted model. They do not identify causal effects,
clinical significance, or a depression diagnosis. Correlation, confounding,
sample selection, and evaluations outside the observed joint feature support
can affect every interpretation here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


ASSOCIATION_WARNING = (
    "Fitted-model associations only; these outputs establish neither causal "
    "effects, clinical significance, nor diagnosis. Correlation, confounding, "
    "sample selection, and model limitations affect interpretation."
)


@dataclass
class PermutationImportanceResult:
    """Aggregate RMSE deterioration and repeated permutation measurements."""

    summary: pd.DataFrame
    repeats: pd.DataFrame
    metadata: dict[str, Any]


@dataclass
class FeatureResponseResult:
    """Average model responses, with no participant-level predictions."""

    curves: pd.DataFrame
    metadata: dict[str, Any]


def _validation_only(split: str) -> None:
    if split != "validation":
        raise ValueError(
            "Exploratory interpretation and feature selection require the "
            "validation split; the held-out test set is reserved for final evaluation."
        )


def _features_and_names(
    features: np.ndarray, feature_names: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    array = np.asarray(features)
    if array.ndim not in (2, 3) or array.shape[0] < 2 or array.shape[-1] == 0:
        raise ValueError("features must have shape (n>=2, p) or (n>=2, time, p).")
    if array.ndim == 3 and array.shape[1] == 0:
        raise ValueError("The temporal dimension must be nonempty.")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("features must contain finite, preprocessed numeric values.")
    names = list(feature_names)
    if len(names) != array.shape[-1] or any(not isinstance(n, str) for n in names):
        raise ValueError("feature_names must label every column on the last axis.")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique transformed-column names.")
    return array, names


def _indices(selected: Sequence[int] | None, n_features: int) -> list[int]:
    indices = list(range(n_features)) if selected is None else list(selected)
    if not indices or any(
        isinstance(i, (bool, np.bool_))
        or not isinstance(i, (int, np.integer))
        or i < 0
        or i >= n_features
        for i in indices
    ):
        raise ValueError("Select at least one valid feature index.")
    if len(set(indices)) != len(indices):
        raise ValueError("Selected feature indices must not be repeated.")
    return [int(i) for i in indices]


def _predict(predict_fn: Callable[[np.ndarray], Any], features: np.ndarray) -> np.ndarray:
    prediction = predict_fn(features)
    if hasattr(prediction, "detach"):
        prediction = prediction.detach().cpu().numpy()
    prediction = np.asarray(prediction)
    if prediction.shape == (len(features), 1):
        prediction = prediction[:, 0]
    if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
        raise ValueError("predict_fn must return one finite scalar per observation.")
    return prediction.astype(np.float64, copy=False)


def validation_permutation_importance(
    predict_fn: Callable[[np.ndarray], Any],
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str],
    *,
    split: str = "validation",
    n_repeats: int = 10,
    seed: int = 42,
    selected_features: Sequence[int] | None = None,
    permutation_groups: Mapping[str, Sequence[int]] | None = None,
    transformation_metadata: Mapping[str, Any] | None = None,
) -> PermutationImportanceResult:
    """Measure validation RMSE increase when one transformed feature is shuffled.

    Tabular inputs use a descriptive observed-row shuffle, including when rows
    are correlated or belong to repeated provisional groups. Such row shuffling
    does not establish participant-level importance or sampling uncertainty.
    For a 3-D input, one row permutation is applied to all time steps of a
    feature, preserving each supplied trajectory's internal time profile. This
    function cannot verify participant identifiers or independent observations.
    It never splits or refits a model.

    ``permutation_groups`` maps original feature names to transformed-column
    indices. Shuffling each group's columns together preserves all one-hot
    coordinates or numeric-value/missingness-indicator pairs. Without grouping,
    encoded columns are perturbed separately and may produce invalid category
    combinations. Groups must be disjoint; selected_features is available only
    when groups are absent (subset the mapping to select original features).
    Correlated predictors can mask each other's importance.
    Scores are model reliance, not causal or clinical feature significance.
    Use a deterministic, evaluation-mode predict_fn with a frozen checkpoint.
    """
    _validation_only(split)
    array, names = _features_and_names(features, feature_names)
    if permutation_groups is None:
        indices = _indices(selected_features, len(names))
        groups = {names[index]: [index] for index in indices}
    else:
        if selected_features is not None:
            raise ValueError("Subset permutation_groups directly instead of combining it with selected_features.")
        if not permutation_groups or any(not isinstance(name, str) or not name for name in permutation_groups):
            raise ValueError("permutation_groups must map nonempty original feature names to columns.")
        groups = {name: _indices(columns, len(names)) for name, columns in permutation_groups.items()}
        indices = [index for columns in groups.values() for index in columns]
        if len(set(indices)) != len(indices):
            raise ValueError("permutation_groups must contain disjoint transformed columns.")
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer.")
    target = np.asarray(targets, dtype=np.float64)
    if target.shape == (len(array), 1):
        target = target[:, 0]
    if target.shape != (len(array),) or not np.isfinite(target).all():
        raise ValueError("targets must contain one finite regression target per row.")

    baseline = float(np.sqrt(np.mean(np.square(_predict(predict_fn, array) - target))))
    generator = np.random.default_rng(seed)
    # Common permutations across features make the repeat comparison reproducible.
    permutations = [generator.permutation(len(array)) for _ in range(n_repeats)]
    records: list[dict[str, Any]] = []
    for name, columns in groups.items():
        for repeat, permutation in enumerate(permutations):
            changed = array.copy()
            changed[..., columns] = array[permutation][..., columns]
            rmse = float(np.sqrt(np.mean(np.square(_predict(predict_fn, changed) - target))))
            records.append(
                {
                    "feature": name,
                    "feature_index": columns[0],
                    "transformed_indices": json.dumps(columns),
                    "repeat": repeat,
                    "baseline_rmse": baseline,
                    "permuted_rmse": rmse,
                    "delta_rmse": rmse - baseline,
                }
            )
    repeats = pd.DataFrame.from_records(records)
    summaries = []
    for name, columns in groups.items():
        values = repeats.loc[repeats["feature"] == name, "delta_rmse"].to_numpy()
        summaries.append(
            {
                "feature": name,
                "feature_index": columns[0],
                "transformed_indices": json.dumps(columns),
                "mean_delta_rmse": float(np.mean(values)),
                "std_delta_rmse": float(np.std(values, ddof=0)),
                "n_repeats": n_repeats,
                "baseline_rmse": baseline,
            }
        )
    summary = pd.DataFrame.from_records(summaries).sort_values(
        ["mean_delta_rmse", "feature_index"], ascending=[False, True], ignore_index=True
    )
    metadata = {
        "method": "validation permutation importance: permuted RMSE minus baseline RMSE",
        "split": split,
        "seed": seed,
        "n_repeats": n_repeats,
        "n_evaluation_units": len(array),
        "input_shape": list(array.shape),
        "feature_names": names,
        "selected_feature_indices": indices,
        "permutation_groups": groups,
        "importance_unit": "original feature with its transformed columns shuffled together" if permutation_groups is not None else "individual transformed coordinate",
        "permutation_unit": (
            "whole supplied feature trajectory across all time steps"
            if array.ndim == 3 else "observed validation rows (not verified independent participants)"
        ),
        "aggregation": "population standard deviation across repeated permutations (ddof=0)",
        "units": "target units for RMSE and delta RMSE; input coordinates are transformed",
        "transformations": dict(transformation_metadata or {}),
        "limitations": [
            ASSOCIATION_WARNING,
            "Correlated features can mask reliance; shuffling can leave joint support across different original features.",
            "All supplied group columns share one row permutation, preserving encoded categories and missingness pairs." if permutation_groups is not None else "Encoded columns are shuffled individually and may produce invalid category combinations; supply coherent permutation_groups when available.",
            "Uncertainty across repeated shuffles is not a sampling confidence interval.",
            "Row-level shuffling does not preserve repeated-observation or provisional-group dependence; scores are descriptive, not participant-level inference.",
        ],
    }
    return PermutationImportanceResult(summary, repeats, metadata)


def conditional_feature_responses(
    predict_fn: Callable[[np.ndarray], Any],
    features: np.ndarray,
    feature_names: Sequence[str],
    selected_features: Sequence[int],
    *,
    split: str = "validation",
    n_points: int = 50,
    quantile_range: tuple[float, float] = (0.05, 0.95),
    feature_values: Mapping[str, Sequence[float]] | None = None,
    temporal_step: int | None = None,
    transformation_metadata: Mapping[str, Any] | None = None,
) -> FeatureResponseResult:
    """Average conditional model responses while all other inputs remain fixed.

    For each validation row, replace one transformed feature coordinate and
    average the resulting predictions. This is a partial-dependence-style model
    response, not an observational conditional expectation or causal effect.
    A temporal input requires one explicit time index; no arbitrary columns are
    interpreted as time steps. No other coordinate is altered.

    The default grid spans observed 5th--95th percentiles. Features with at most
    ten observed values use those discrete values instead of interpolating.
    For encoded categorical features callers should select valid joint inputs
    separately: replacing one dummy coordinate may leave the data manifold.
    """
    _validation_only(split)
    array, names = _features_and_names(features, feature_names)
    indices = _indices(selected_features, len(names))
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 2:
        raise ValueError("n_points must be an integer of at least two.")
    if len(quantile_range) != 2 or not 0 <= quantile_range[0] < quantile_range[1] <= 1:
        raise ValueError("quantile_range must satisfy 0 <= lower < upper <= 1.")
    if array.ndim == 3:
        if (
            isinstance(temporal_step, bool)
            or not isinstance(temporal_step, (int, np.integer))
            or not 0 <= temporal_step < array.shape[1]
        ):
            raise ValueError("Temporal responses require an explicit valid temporal_step.")
    elif temporal_step is not None:
        raise ValueError("temporal_step applies only to genuine 3-D temporal inputs.")

    records: list[dict[str, Any]] = []
    grids: dict[str, list[float]] = {}
    for index in indices:
        observed = array[:, index] if array.ndim == 2 else array[:, temporal_step, index]
        if feature_values is not None and names[index] in feature_values:
            grid = np.asarray(feature_values[names[index]], dtype=np.float64)
            if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all():
                raise ValueError("Explicit feature_values must be nonempty finite 1-D grids.")
            grid = np.unique(grid)
        else:
            distinct = np.unique(observed)
            if len(distinct) <= 10:
                grid = distinct
            else:
                lower, upper = np.quantile(observed, quantile_range)
                grid = np.unique(np.linspace(lower, upper, n_points))
        grids[names[index]] = [float(value) for value in grid]
        for value in grid:
            # Floating copy avoids truncation if an integer-valued array is supplied.
            changed = array.astype(np.result_type(array.dtype, np.float32), copy=True)
            if array.ndim == 2:
                changed[:, index] = value
            else:
                changed[:, temporal_step, index] = value
            prediction = _predict(predict_fn, changed)
            records.append(
                {
                    "feature": names[index],
                    "feature_index": index,
                    "input_value": float(value),
                    "prediction_mean": float(np.mean(prediction)),
                    "prediction_std": float(np.std(prediction, ddof=0)),
                }
            )
    metadata = {
        "method": "average model response to one replaced transformed feature coordinate",
        "split": split,
        "feature_names": names,
        "selected_feature_indices": indices,
        "n_evaluation_units": len(array),
        "input_shape": list(array.shape),
        "temporal_step": None if temporal_step is None else int(temporal_step),
        "default_quantile_range": list(quantile_range),
        "feature_grids": grids,
        "input_units": "transformed feature units; no inverse transformation is implied",
        "output_units": "model regression target units",
        "transformations": dict(transformation_metadata or {}),
        "limitations": [
            ASSOCIATION_WARNING,
            "Average conditional model response is partial dependence, not a causal effect or observed conditional expectation.",
            "Other transformed coordinates are held fixed per row; encoded categories and correlated features may leave joint support.",
            "prediction_std describes between-row model-response dispersion, not uncertainty or a confidence interval.",
        ],
    }
    return FeatureResponseResult(pd.DataFrame.from_records(records), metadata)


def _export_directory(output_dir: str | Path, prefix: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError("prefix must be a filename stem without directory separators.")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _save_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(metadata, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_figure(figure: Any, directory: Path, prefix: str, dpi: int) -> dict[str, Path]:
    if dpi < 300:
        raise ValueError("Publication exports require dpi >= 300.")
    paths = {kind: directory / f"{prefix}.{kind}" for kind in ("png", "pdf")}
    for kind, path in paths.items():
        figure.savefig(path, format=kind, dpi=dpi, bbox_inches="tight", facecolor="white")
    return paths


def export_permutation_importance(
    result: PermutationImportanceResult,
    output_dir: str | Path,
    *,
    prefix: str = "permutation_importance",
    max_features: int = 20,
    dpi: int = 300,
) -> dict[str, Path]:
    """Save aggregate CSV/JSON plus publication PNG/PDF; never individual records."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if dpi < 300 or max_features < 1:
        raise ValueError("dpi must be >= 300 and max_features must be positive.")
    directory = _export_directory(output_dir, prefix)
    paths = {
        "summary": directory / f"{prefix}.csv",
        "repeats": directory / f"{prefix}_repeats.csv",
        "metadata": directory / f"{prefix}.json",
    }
    result.summary.to_csv(paths["summary"], index=False)
    result.repeats.to_csv(paths["repeats"], index=False)
    _save_metadata(paths["metadata"], result.metadata)
    shown = result.summary.head(max_features).iloc[::-1]
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axis = plt.subplots(figsize=(8, max(3, 0.32 * len(shown) + 1.6)))
        try:
            axis.barh(
                shown["feature"], shown["mean_delta_rmse"],
                xerr=shown["std_delta_rmse"], color="#246a8d", capsize=3,
            )
            axis.axvline(0, color="black", linewidth=0.7)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useMathText=True)
            axis.set_xlabel("Validation RMSE increase after permutation (target units)")
            axis.set_title("Model reliance: permutation importance")
            axis.spines[["top", "right"]].set_visible(False)
            figure.text(0.01, 0.01, "Error bars: repeat SD; associations only, no causal or diagnostic interpretation.", fontsize=8)
            figure.tight_layout(rect=(0, 0.04, 1, 1))
            paths.update(_save_figure(figure, directory, prefix, dpi))
        finally:
            plt.close(figure)
    return paths


def export_feature_responses(
    result: FeatureResponseResult,
    output_dir: str | Path,
    *,
    prefix: str = "feature_responses",
    dpi: int = 300,
) -> dict[str, Path]:
    """Save mean-response panels, transformed-coordinate grids, and metadata."""
    import matplotlib.pyplot as plt

    groups = list(result.curves.groupby("feature_index", sort=False))
    if not groups or len(groups) > 12 or dpi < 300:
        raise ValueError("Export 1--12 selected response curves at dpi >= 300.")
    directory = _export_directory(output_dir, prefix)
    paths = {"curves": directory / f"{prefix}.csv", "metadata": directory / f"{prefix}.json"}
    result.curves.to_csv(paths["curves"], index=False)
    _save_metadata(paths["metadata"], result.metadata)
    n_columns = min(2, len(groups))
    n_rows = (len(groups) + n_columns - 1) // n_columns
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axes = plt.subplots(n_rows, n_columns, figsize=(6 * n_columns, 3.5 * n_rows), squeeze=False)
        try:
            for axis, (_, group) in zip(axes.ravel(), groups):
                axis.plot(group["input_value"], group["prediction_mean"], color="#246a8d", marker="o" if len(group) <= 10 else None)
                axis.set_xlabel(f"{group['feature'].iloc[0]} (transformed units)")
                axis.set_ylabel("Mean model response (target units)")
                axis.spines[["top", "right"]].set_visible(False)
            for axis in axes.ravel()[len(groups):]:
                axis.set_visible(False)
            figure.suptitle("Feature responses of the fitted model")
            figure.text(0.01, 0.01, "Other coordinates fixed per row; associations only. Replaced coordinates can leave joint data support.", fontsize=8)
            figure.tight_layout(rect=(0, 0.04, 1, 0.96))
            paths.update(_save_figure(figure, directory, prefix, dpi))
        finally:
            plt.close(figure)
    return paths
