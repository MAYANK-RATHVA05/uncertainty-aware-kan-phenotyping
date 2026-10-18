"""Synthetic tests for model associations; no sensitive records or study claims."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from kan_phenotyping.interpretability import (
    conditional_feature_responses,
    export_feature_responses,
    export_permutation_importance,
    plot_selected_spline_edges,
    validation_permutation_importance,
)
from kan_phenotyping.models.kan_layer import KANLinear


def test_permutation_importance_identifies_used_feature_and_is_reproducible():
    generator = np.random.default_rng(9)
    features = generator.normal(size=(100, 3))
    original = features.copy()
    prediction = lambda values: 3 * values[:, 0]
    target = prediction(features)
    arguments = dict(n_repeats=4, seed=13, transformation_metadata={"scale": "training-only standardization"})
    first = validation_permutation_importance(prediction, features, target, ["used", "unused", "also_unused"], **arguments)
    second = validation_permutation_importance(prediction, features, target, ["used", "unused", "also_unused"], **arguments)

    assert first.summary.iloc[0]["feature"] == "used"
    assert first.summary.iloc[0]["mean_delta_rmse"] > 2
    assert np.all(first.summary.iloc[1:]["mean_delta_rmse"] == 0)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.repeats, second.repeats)
    np.testing.assert_array_equal(features, original)
    assert first.metadata["split"] == "validation"
    assert "observed validation rows" in first.metadata["permutation_unit"]
    assert "training-only" in first.metadata["transformations"]["scale"]


def test_temporal_permutation_preserves_whole_supplied_feature_trajectory():
    row_level = np.arange(12, dtype=np.float64)[:, None]
    features = np.zeros((12, 7, 2))
    features[:, :, 0] = row_level * 100 + np.arange(7)[None, :]
    features[:, :, 1] = row_level * 11
    calls = []

    def prediction(values):
        calls.append(values.copy())
        return values[:, :, 0].mean(axis=1)

    result = validation_permutation_importance(
        prediction, features, features[:, :, 0].mean(axis=1), ["trajectory", "fixed"],
        selected_features=[0], n_repeats=3, seed=7,
    )
    for permuted in calls[1:]:
        # A time-wise shuffle would break the known within-trajectory increments.
        np.testing.assert_array_equal(np.diff(permuted[:, :, 0], axis=1), np.ones((12, 6)))
        np.testing.assert_array_equal(permuted[:, :, 1], features[:, :, 1])
        assert sorted(permuted[:, 0, 0]) == sorted(features[:, 0, 0])
    assert len(result.repeats) == 3
    assert "whole supplied feature trajectory" in result.metadata["permutation_unit"]


def test_grouped_permutation_preserves_categories_and_value_missingness_pairs():
    features = np.array([
        [5, 0, 1, 0, 0], [7, 0, 0, 1, 0], [6, 1, 0, 0, 1],
        [9, 0, 1, 0, 0], [6, 1, 0, 1, 0], [8, 0, 0, 0, 1],
    ], dtype=float)
    valid_numeric_pairs = set(map(tuple, features[:, :2]))
    calls = []

    def prediction(values):
        calls.append(values.copy())
        assert all(tuple(pair) in valid_numeric_pairs for pair in values[:, :2])
        np.testing.assert_array_equal(values[:, 2:].sum(axis=1), np.ones(len(values)))
        return values[:, 0] + 2 * values[:, 2]

    result = validation_permutation_importance(
        prediction, features, prediction(features), ["value", "missing", "cat_a", "cat_b", "cat_c"],
        n_repeats=3, seed=1, permutation_groups={"original_numeric": [0, 1], "original_category": [2, 3, 4]},
    )
    assert set(result.summary["feature"]) == {"original_numeric", "original_category"}
    assert len(result.repeats) == 6
    assert "original feature" in result.metadata["importance_unit"]
    assert result.metadata["permutation_groups"]["original_category"] == [2, 3, 4]
    for values in calls[2:5]:
        np.testing.assert_array_equal(values[:, 2:], features[:, 2:])
    for values in calls[5:]:
        np.testing.assert_array_equal(values[:, :2], features[:, :2])


def test_grouped_permutation_rejects_overlapping_groups():
    with pytest.raises(ValueError, match="disjoint"):
        validation_permutation_importance(
            lambda values: values[:, 0], np.ones((4, 2)), np.ones(4), ["a", "b"],
            permutation_groups={"first": [0], "second": [0, 1]},
        )


@pytest.mark.parametrize("split", ["train", "test", "heldout"])
def test_exploratory_methods_reject_non_validation_split(split):
    features = np.ones((5, 2))
    with pytest.raises(ValueError, match="validation split"):
        validation_permutation_importance(lambda values: values[:, 0], features, np.ones(5), ["a", "b"], split=split)
    with pytest.raises(ValueError, match="validation split"):
        conditional_feature_responses(lambda values: values[:, 0], features, ["a", "b"], [0], split=split)


def test_feature_response_holds_other_coordinates_fixed_and_averages_only():
    features = np.array([[0, 4], [1, 6], [2, 8]], dtype=np.int64)
    original = features.copy()
    observed = []

    def prediction(values):
        observed.append(values.copy())
        return 2 * values[:, 0] + 3 * values[:, 1]

    result = conditional_feature_responses(
        prediction, features, ["activity", "other"], [0],
        feature_values={"activity": [0.25, 1.5, 2.75]},
        transformation_metadata={"activity": "standardized by training mean and scale"},
    )
    np.testing.assert_allclose(result.curves["prediction_mean"], [18.5, 21, 23.5])
    for array in observed:
        np.testing.assert_array_equal(array[:, 1], features[:, 1])
    np.testing.assert_array_equal(features, original)
    assert "transformed" in result.metadata["input_units"]
    assert result.curves.shape == (3, 5)
    assert "participant" not in " ".join(result.curves.columns)


def test_discrete_feature_response_does_not_interpolate_unobserved_levels():
    features = np.array([[0.0], [1.0], [0.0], [1.0]])
    result = conditional_feature_responses(lambda values: values[:, 0], features, ["flag"], [0])
    assert result.curves["input_value"].tolist() == [0.0, 1.0]


def test_temporal_response_requires_explicit_time_and_leaves_other_times_fixed():
    features = np.arange(30, dtype=np.float64).reshape(5, 3, 2)
    with pytest.raises(ValueError, match="explicit valid temporal_step"):
        conditional_feature_responses(lambda values: values.mean(axis=(1, 2)), features, ["a", "b"], [0])

    def prediction(values):
        np.testing.assert_array_equal(values[:, 0], features[:, 0])
        np.testing.assert_array_equal(values[:, 2], features[:, 2])
        np.testing.assert_array_equal(values[:, 1, 1], features[:, 1, 1])
        return values[:, 1, 0]

    result = conditional_feature_responses(
        prediction, features, ["a", "b"], [0], temporal_step=1, feature_values={"a": [-1, 2]},
    )
    np.testing.assert_array_equal(result.curves["prediction_mean"], [-1, 2])


def test_spline_export_matches_learned_components_and_excludes_shared_bias(tmp_path):
    layer = KANLinear(2, 1)
    with torch.no_grad():
        layer.base_weight.zero_()
        layer.spline_weight.fill_(2)
        layer.bias.fill_(17)
    result = plot_selected_spline_edges(
        layer, [(0, 0)], ["numeric__activity", "numeric__sleep"], tmp_path,
        feature_ranges={"numeric__activity": (-2.0, 2.0)}, n_points=25,
        transformation_metadata={"numeric_units": "standardized"},
    )
    np.testing.assert_allclose(result.curves["edge_response"], 2, atol=1e-6)
    np.testing.assert_allclose(result.curves["spline_component"], 2, atol=1e-6)
    np.testing.assert_array_equal(result.curves["silu_component"], 0)
    assert result.metadata["selected_edges"][0]["shared_neuron_bias_excluded_from_curve"] == 17
    assert result.metadata["transformations"]["numeric_units"] == "standardized"
    _assert_publication_outputs(result.paths)
    saved = json.loads(result.paths["metadata"].read_text())
    assert saved["feature_names"][0] == "numeric__activity"
    assert "clinical" in " ".join(saved["limitations"])


def _assert_publication_outputs(paths):
    assert paths["pdf"].read_bytes().startswith(b"%PDF")
    with Image.open(paths["png"]) as image:
        assert image.info["dpi"][0] >= 299
    assert all(path.stat().st_size > 10 for path in paths.values())


def test_feature_and_importance_exports_contain_aggregate_metadata(tmp_path):
    features = np.arange(24, dtype=np.float64).reshape(12, 2)
    predictor = lambda values: values[:, 0]
    importance = validation_permutation_importance(predictor, features, features[:, 0], ["a", "b"], n_repeats=2)
    response = conditional_feature_responses(predictor, features, ["a", "b"], [0], n_points=3)
    paths = export_permutation_importance(importance, tmp_path)
    _assert_publication_outputs(paths)
    paths = export_feature_responses(response, tmp_path)
    _assert_publication_outputs(paths)
    assert pd.read_csv(paths["curves"]).shape[0] == 3
    assert "transformed" in json.loads(paths["metadata"].read_text())["input_units"]


@pytest.mark.parametrize("bad_features", [np.array([[1.0], [np.nan]]), np.ones(3), np.zeros((0, 2))])
def test_feature_validation_rejects_invalid_inputs(bad_features):
    with pytest.raises(ValueError):
        validation_permutation_importance(lambda values: values[:, 0], bad_features, np.ones(2), ["a"])


def test_nonfinite_predictions_and_invalid_export_paths_are_rejected(tmp_path):
    features = np.ones((4, 1))
    with pytest.raises(ValueError, match="one finite scalar"):
        validation_permutation_importance(lambda values: np.full(len(values), np.nan), features, np.ones(4), ["a"])
    result = conditional_feature_responses(lambda values: values[:, 0], features, ["a"], [0])
    with pytest.raises(ValueError, match="filename stem"):
        export_feature_responses(result, tmp_path, prefix="../outside")
    with pytest.raises(ValueError, match="dpi"):
        export_feature_responses(result, tmp_path, dpi=100)


def test_interpretation_cli_uses_validation_only_and_frozen_checkpoint(tmp_path, monkeypatch):
    from kan_phenotyping.experiments import interpret_depression as command
    from kan_phenotyping.experiments.common import build_model, model_spec

    features = np.random.default_rng(5).normal(size=(12, 3)).astype(np.float32)
    names = ["numeric__activity", "numeric__sleep", "numeric__steps"]
    spec = model_spec(3, hidden_width=4)
    model = build_model(spec)
    run = tmp_path / "trusted_run"
    run.mkdir()
    torch.save({"model_metadata": spec, "model_state": model.state_dict()}, run / "best.pt")
    # No train/test partition is exposed; accessing either would fail this test.
    prepared = SimpleNamespace(
        metadata={
            "model": spec,
            "inputs": {"data_sha256": "synthetic", "dictionary_sha256": "synthetic"},
            "preprocessing": {"numeric_units": "standardized"},
            "limitations": ["Synthetic pipeline check, not research performance."],
        },
        features={"validation": SimpleNamespace(tabular=features, feature_names={"tabular": names})},
        preprocessor=SimpleNamespace(export_metadata=lambda: {
            "numeric_units": "standardized",
            "permutation_groups": {"activity": [0], "sleep": [1], "steps": [2]},
        }),
        examples=SimpleNamespace(y=np.arange(12, dtype=float)),
        split=SimpleNamespace(validation=np.arange(12)),
    )
    monkeypatch.setattr(command, "load_run", lambda *args: prepared)
    output = tmp_path / "interpretation"
    manifest = command.main([
        "--run-dir", str(run), "--data", "not-read-directly", "--dictionary", "not-read-directly",
        "--output", str(output), "--n-repeats", "2", "--max-features", "2", "--n-points", "3",
    ])
    assert manifest["split"] == "validation"
    assert manifest["permutation_feature_indices"] == [0, 1]
    assert manifest["permutation_groups"] == {"activity": [0], "sleep": [1]}
    assert len(manifest["response_feature_indices"]) == 2
    assert "learned_edges" in manifest["artifacts"]
    assert (output / "interpretation.json").exists()
    assert "first hidden unit" in manifest["edge_output_unit"]
    assert "participant-level inference is unsupported" in " ".join(manifest["limitations"])
