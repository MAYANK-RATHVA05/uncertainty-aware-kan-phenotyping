"""Independent baseline checks using synthetic saved forecasting artifacts.

The fixture preserves the real builder's schema and split/preprocessing path.
No uploaded calibration/test outcomes or neural models are used in these tests.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from kan_phenotyping.data.conditional_forecasting import (
    BASELINE_DEMOGRAPHIC_FEATURES,
    WEARABLE_STATISTICS_FEATURES,
    ConditionalForecastAssumptions,
    build_conditional_forecasting_dataset,
)
from kan_phenotyping.data.forecasting_preprocessing import ConditionalForecastPreprocessor
from kan_phenotyping.experiments.build_forecasting_dataset import (
    prepare_forecasting_dataset,
    sha256_file,
)
from kan_phenotyping.experiments import forecasting_baselines as baselines


@pytest.fixture
def saved_forecasting_data(tmp_path):
    rows = []
    for person in range(20):
        for month in range(1, 7):
            quarter = (month - 1) // 3
            endpoint = month % 3 == 0
            row = {
                "__index_level_0__": f"{1000 + person}_{month}",
                "phq9_score_start": (person + 2 * quarter) % 28 if endpoint else np.nan,
                "phq9_score_end": (3 * person + 5 * quarter) % 28 if endpoint else np.nan,
            }
            row.update({
                name: float(person * 7 + month * 2 + index / 10)
                for index, name in enumerate(WEARABLE_STATISTICS_FEATURES)
            })
            row.update({
                name: float((person + index) % 3)
                for index, name in enumerate(BASELINE_DEMOGRAPHIC_FEATURES)
            })
            rows.append(row)
    frame = pd.DataFrame(rows).sample(frac=1, random_state=91).reset_index(drop=True)
    frame.loc[frame["__index_level_0__"] == "1000_1", WEARABLE_STATISTICS_FEATURES[0]] = np.nan
    entries = [
        (name, "Steps" if name.startswith("steps_") else "Sleep", "Statistics", f"Observed {name}", "")
        for name in WEARABLE_STATISTICS_FEATURES
    ] + [
        (name, "Demographic", "Static", f"Enrollment {name}", "")
        for name in BASELINE_DEMOGRAPHIC_FEATURES
    ] + [
        ("phq9_score_start", "PHQ-9", "PHQ-9 score", "PHQ-9 at start of three-month period", "0-27"),
        ("phq9_score_end", "PHQ-9", "PHQ-9 score", "PHQ-9 at end of three-month period", "0-27"),
    ]
    dictionary = pd.DataFrame(entries, columns=["Feature name", "Category", "Subcategory", "Description", "Notes"])
    data_path = tmp_path / "source.csv"
    dictionary_path = tmp_path / "dictionary.tsv"
    frame.to_csv(data_path, index=False)
    dictionary.to_csv(dictionary_path, sep="\t", index=False)
    assumptions = ConditionalForecastAssumptions(
        True, True, True, True, "Synthetic known relative-month generator, not verified availability."
    )
    directory = tmp_path / "prepared"
    manifest = prepare_forecasting_dataset(data_path, dictionary_path, directory, assumptions, seed=17)
    dataset = build_conditional_forecasting_dataset(frame, dictionary, assumptions)
    return directory, manifest, dataset


def test_persistence_is_exact_measured_sm0_and_not_change():
    measured_sm0 = np.array([0.0, 1.0, 8.0, 19.0, 27.0])
    predicted = baselines.PersistenceBaseline().predict(measured_sm0)
    np.testing.assert_array_equal(predicted, measured_sm0)
    assert predicted.shape == measured_sm0.shape


def test_recover_measured_sm0_undoes_training_scaler_without_roundoff_drift():
    measured = np.arange(28, dtype=float).reshape(-1, 1)
    scaler = StandardScaler().fit(np.array([[1.0], [4.0], [10.0], [26.0]]))
    standardized = scaler.transform(measured).astype(np.float32)
    recovered = baselines.recover_measured_sm0(standardized, scaler)
    assert recovered.shape == (28,)
    np.testing.assert_array_equal(recovered, measured[:, 0])


@pytest.mark.parametrize("bad_measured", [-1.0, 28.0, 4.2, np.nan, np.inf])
def test_recovery_rejects_invalid_measured_scores_instead_of_clipping(bad_measured):
    scaler = StandardScaler().fit(np.array([[0.0], [27.0]]))
    # Form the inverse coordinates explicitly so NaN/inf reach the guard too.
    standardized = np.array([[(bad_measured - scaler.mean_[0]) / scaler.scale_[0]]])
    with pytest.raises(ValueError):
        baselines.recover_measured_sm0(standardized, scaler)


def test_training_mean_uses_only_supplied_training_outcomes():
    training = np.array([1.0, 2.0, 8.0, 17.0])
    model = baselines.TrainingMeanBaseline().fit(training)
    expected = np.full(5, training.mean())
    np.testing.assert_array_equal(model.predict(5), expected)
    np.testing.assert_array_equal(model.predict(5), expected)


def test_metrics_use_original_units_and_unclipped_predictions():
    measured = np.array([0.0, 9.0, 18.0, 27.0])
    predicted = np.array([-5.0, 10.0, 16.0, 35.0])
    metrics = baselines.regression_metrics(measured, predicted)
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["mae"] == pytest.approx(mean_absolute_error(measured, predicted))
    assert metrics["rmse"] == pytest.approx(np.sqrt(mean_squared_error(measured, predicted)))
    assert metrics["r2"] == pytest.approx(r2_score(measured, predicted))
    assert metrics != baselines.regression_metrics(measured, np.clip(predicted, 0, 27))


def test_loader_reuses_saved_sample_order_targets_and_month_identity(saved_forecasting_data):
    directory, manifest, dataset = saved_forecasting_data
    data = baselines.load_baseline_data(directory)
    with np.load(directory / "splits.npz", allow_pickle=False) as split_archive:
        for name in ("train", "validation"):
            partition = getattr(data, name)
            indices = split_archive[name]
            np.testing.assert_array_equal(partition.sample_indices, indices)
            np.testing.assert_array_equal(partition.participants, dataset.participants[indices])
            np.testing.assert_array_equal(partition.targets, dataset.targets[indices])
            np.testing.assert_array_equal(partition.sm0_phq, dataset.predictors.sm0_phq[indices, 0])
            restored = data.preprocessor.transform(dataset.predictors, indices)
            np.testing.assert_array_equal(partition.tabular, restored.tabular)
            dimension = restored.wearable.shape[-1]
            np.testing.assert_array_equal(partition.tabular[:, :dimension], restored.wearable[:, 0])
            np.testing.assert_array_equal(partition.tabular[:, dimension:2 * dimension], restored.wearable[:, 1])
            names = data.preprocessor.feature_names["tabular"]
            assert all(name.startswith("SM1__") for name in names[:dimension])
            assert all(name.startswith("SM2__") for name in names[dimension:2 * dimension])
    assert set(data.train.participants).isdisjoint(data.validation.participants)
    assert data.manifest == manifest
    assert data.preprocessor.n_training_samples_ == len(data.train.targets)
    assert data.preprocessor.n_training_wearable_rows_ == 2 * len(data.train.targets)
    assert data.preprocessor.baseline_scaler.n_samples_seen_ == len(data.train.targets)
    np.testing.assert_allclose(data.preprocessor.baseline_scaler.mean_, data.train.sm0_phq.mean())


def test_ridge_fits_each_candidate_on_training_only_and_selects_validation_rmse(
    saved_forecasting_data, monkeypatch
):
    directory, _, _ = saved_forecasting_data
    data = baselines.load_baseline_data(directory)
    calls = []
    original_fit = Ridge.fit

    def observe_fit(model, x, y, *args, **kwargs):
        calls.append((float(model.alpha), np.array(x, copy=True), np.array(y, copy=True)))
        return original_fit(model, x, y, *args, **kwargs)

    monkeypatch.setattr(Ridge, "fit", observe_fit)
    result = baselines.fit_baselines(data)
    assert tuple(baselines.ALPHAS) == (0.01, 0.1, 1.0, 10.0, 100.0)
    assert [call[0] for call in calls] == list(baselines.ALPHAS)
    assert len(calls) == 5, "The selected candidate must not be refitted on validation."
    for _, predictors, targets in calls:
        np.testing.assert_array_equal(predictors, data.train.tabular)
        np.testing.assert_array_equal(targets, data.train.targets)
    assert set(result.models) == set(result.validation_predictions) == set(result.metrics) == {
        "persistence", "training_mean", "ridge",
    }
    expected_candidates = []
    for alpha in baselines.ALPHAS:
        model = Ridge(alpha=alpha, solver="svd", fit_intercept=True)
        original_fit(model, data.train.tabular.astype(np.float64), data.train.targets.astype(np.float64))
        prediction = model.predict(data.validation.tabular.astype(np.float64))
        expected_candidates.append((float(np.sqrt(mean_squared_error(data.validation.targets, prediction))), alpha))
    best_alpha = min(expected_candidates, key=lambda pair: pair[0])[1]
    assert result.selected_alpha == best_alpha
    assert result.models["ridge"].alpha == best_alpha
    assert [entry["alpha"] for entry in result.ridge_candidates] == list(baselines.ALPHAS)
    assert result.selected_alpha == min(
        result.ridge_candidates, key=lambda entry: entry["validation_metrics"]["rmse"]
    )["alpha"]
    np.testing.assert_array_equal(result.validation_predictions["persistence"], data.validation.sm0_phq)
    np.testing.assert_array_equal(
        result.validation_predictions["training_mean"],
        np.full(len(data.validation.targets), data.train.targets.mean(dtype=np.float64)),
    )
    for name, predictions in result.validation_predictions.items():
        assert predictions.shape == data.validation.targets.shape
        assert result.metrics[name] == baselines.regression_metrics(data.validation.targets, predictions)


def test_validation_changes_cannot_change_candidate_coefficients_or_training_mean(saved_forecasting_data):
    directory, _, _ = saved_forecasting_data
    data = baselines.load_baseline_data(directory)
    first = baselines.fit_baselines(data)
    changed = deepcopy(data)
    changed_validation = replace(
        changed.validation,
        targets=np.full_like(changed.validation.targets, 27),
    )
    changed = replace(changed, validation=changed_validation)
    second = baselines.fit_baselines(changed)
    np.testing.assert_array_equal(
        first.models["training_mean"].predict(len(data.validation.targets)),
        second.models["training_mean"].predict(len(data.validation.targets)),
    )
    # Selection may change; every fitted candidate must remain identical.
    for first_candidate, second_candidate in zip(first.fitted_candidates, second.fitted_candidates, strict=True):
        assert first_candidate.alpha == second_candidate.alpha
        np.testing.assert_array_equal(first_candidate.coef_, second_candidate.coef_)
        np.testing.assert_array_equal(first_candidate.intercept_, second_candidate.intercept_)


def test_predictions_are_not_clipped_when_ridge_extrapolates(saved_forecasting_data):
    directory, _, _ = saved_forecasting_data
    data = baselines.load_baseline_data(directory)
    # Preserve a valid scaled input layout while making the SM0 predictor
    # extrapolate far outside training support. This is a counterfactual test.
    extreme_x = data.validation.tabular.copy()
    extreme_x[:, -1] = 1e6
    data = replace(data, validation=replace(data.validation, tabular=extreme_x))
    result = baselines.fit_baselines(data)
    expected = result.models["ridge"].predict(extreme_x)
    assert ((expected < 0) | (expected > 27)).any()
    np.testing.assert_array_equal(result.validation_predictions["ridge"], expected)


def test_fixed_saved_inputs_give_reproducible_models_predictions_and_selection(saved_forecasting_data):
    directory, _, _ = saved_forecasting_data
    first = baselines.fit_baselines(baselines.load_baseline_data(directory))
    second = baselines.fit_baselines(baselines.load_baseline_data(directory))
    assert first.selected_alpha == second.selected_alpha
    assert first.metrics == second.metrics
    assert first.ridge_candidates == second.ridge_candidates
    np.testing.assert_array_equal(first.models["ridge"].coef_, second.models["ridge"].coef_)
    np.testing.assert_array_equal(first.models["ridge"].intercept_, second.models["ridge"].intercept_)
    for name in first.validation_predictions:
        np.testing.assert_array_equal(first.validation_predictions[name], second.validation_predictions[name])


def test_baseline_loader_and_runner_never_read_reserved_partition_values_or_refit_preprocessing(
    saved_forecasting_data, tmp_path, monkeypatch
):
    directory, _, _ = saved_forecasting_data
    before = {path.name: sha256_file(path) for path in directory.iterdir()}
    forbidden = {f"{partition}_{kind}.npz" for partition in ("calibration", "test")
                 for kind in ("inputs", "targets")}
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.name not in forbidden, f"Reserved values were accessed: {path.name}"
        return original_open(path, *args, **kwargs)

    original_np_load = np.load

    def guarded_load(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            assert Path(path).name not in forbidden
        return original_np_load(path, *args, **kwargs)

    def deny_preprocessor_refit(*_args, **_kwargs):
        raise AssertionError("Baseline execution must reuse saved training-fitted preprocessing.")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(np, "load", guarded_load)
    monkeypatch.setattr(ConditionalForecastPreprocessor, "fit", deny_preprocessor_refit)
    data = baselines.load_baseline_data(directory)
    assert not {Path(path).name for path in data.read_files} & forbidden
    report = baselines.run_forecasting_baselines(directory, tmp_path / "baseline_run")
    assert isinstance(report, dict)
    assert not any("calibration_metrics" in key or "test_metrics" in key for key in report)
    # Restore guard before hashing reserved files: immutability verification is
    # not part of the baseline runner and does not inspect any outcomes.
    monkeypatch.undo()
    assert before == {path.name: sha256_file(path) for path in directory.iterdir()}


def test_loader_detects_tampered_validation_inputs(saved_forecasting_data):
    directory, _, _ = saved_forecasting_data
    path = directory / "validation_inputs.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["tabular"] = arrays["tabular"][::-1]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="hash|SHA|sha|integrity"):
        baselines.load_baseline_data(directory)


def test_runner_saves_original_unit_predictions_and_fitted_model_artifacts(
    saved_forecasting_data, tmp_path
):
    directory, _, _ = saved_forecasting_data
    output = tmp_path / "baseline_run"
    report = baselines.run_forecasting_baselines(directory, output)
    config = json.loads((output / "config.json").read_text())
    run = json.loads((output / "run.json").read_text())
    assert config["workflow"] == "conditional_relative_month_forecasting_baselines"
    assert config["target"] == "measured SM3 PHQ-9 severity (phq9_score_end), not change"
    assert config["prediction_clipping"] is False
    assert config["target_transformation"] == "none"
    assert report["selected_alpha"] in baselines.ALPHAS
    assert config["ridge"]["selected_alpha"] == report["selected_alpha"]
    assert config["feature_names"]["tabular"][-1] == "measured_sm0_phq9_standardized"
    assert run["original_input_hashes"]
    saved_report = json.loads((output / "validation_metrics.json").read_text())
    assert saved_report == report
    models = joblib.load(output / "models.joblib")
    data = baselines.load_baseline_data(directory)
    assert set(models) == {"persistence", "training_mean", "ridge"}
    np.testing.assert_array_equal(models["persistence"].predict(data.validation.sm0_phq), data.validation.sm0_phq)
    np.testing.assert_array_equal(models["training_mean"].predict(len(data.validation.targets)),
                                  np.full(len(data.validation.targets), data.train.targets.mean(dtype=np.float64)))
    expected = baselines.fit_baselines(data)
    np.testing.assert_array_equal(models["ridge"].predict(data.validation.tabular),
                                  expected.validation_predictions["ridge"])
    with np.load(output / "validation_predictions.npz", allow_pickle=False) as predictions:
        np.testing.assert_array_equal(predictions["sample_indices"], data.validation.sample_indices)
        np.testing.assert_array_equal(predictions["measured_sm0"], data.validation.sm0_phq)
        np.testing.assert_array_equal(predictions["measured_sm3"], data.validation.targets)
        for name, values in expected.validation_predictions.items():
            np.testing.assert_array_equal(predictions[name], values)
    with np.load(output / "used_split_indices.npz", allow_pickle=False) as splits:
        assert set(splits.files) == {"train", "validation"}
        np.testing.assert_array_equal(splits["train"], data.train.sample_indices)
        np.testing.assert_array_equal(splits["validation"], data.validation.sample_indices)
    assert sha256_file(output / "preprocessor.joblib") == sha256_file(directory / "preprocessor.joblib")
    for name, digest in run["consumed_sha256"].items():
        assert digest == sha256_file(directory / name)
    for name, digest in run["artifact_sha256"].items():
        assert digest == sha256_file(output / name)


def test_module_cli_models_reload_and_reconstruct_predictions_in_a_fresh_process(
    saved_forecasting_data, tmp_path
):
    """A module CLI must not serialize classes under ephemeral ``__main__``."""
    directory, _, _ = saved_forecasting_data
    output = tmp_path / "cli_run"
    environment = os.environ.copy()
    source_root = str(Path(baselines.__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        source_root, environment.get("PYTHONPATH", ""),
    )))
    cli = subprocess.run(
        [sys.executable, "-m", "kan_phenotyping.experiments.forecasting_baselines",
         "--dataset-dir", str(directory), "--output", str(output)],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
        check=True, timeout=30,
    )
    cli_report = json.loads(cli.stdout)
    assert cli_report == json.loads((output / "validation_metrics.json").read_text())
    # Load the serialized models before explicitly importing our implementation:
    # joblib must discover their canonical module by itself in a new interpreter.
    reconstruction = subprocess.run(
        [sys.executable, "-c", """
import json
from pathlib import Path
import sys

import joblib
import numpy as np

output = Path(sys.argv[1])
models = joblib.load(output / "models.joblib")
assert models["persistence"].__class__.__module__ == "kan_phenotyping.experiments.forecasting_baselines"
assert models["training_mean"].__class__.__module__ == "kan_phenotyping.experiments.forecasting_baselines"

from kan_phenotyping.experiments.forecasting_baselines import load_baseline_data

data = load_baseline_data(sys.argv[2])
predictions = {
    "persistence": models["persistence"].predict(data.validation.sm0_phq),
    "training_mean": models["training_mean"].predict(len(data.validation.targets)),
    "ridge": models["ridge"].predict(data.validation.tabular),
}
with np.load(output / "validation_predictions.npz", allow_pickle=False) as saved:
    for name, values in predictions.items():
        np.testing.assert_array_equal(values, saved[name])
print(json.dumps({"models_reloaded": sorted(models), "predictions_reconstructed": True}))
""", str(output), str(directory)],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
        check=True, timeout=30,
    )
    assert json.loads(reconstruction.stdout) == {
        "models_reloaded": ["persistence", "ridge", "training_mean"],
        "predictions_reconstructed": True,
    }
