"""Synthetic protocol checks for the fixed full latent KAN-ODE experiment.

No uploaded records or calibration/test payloads are used. Tiny two-epoch fits
exercise the real runner's update, diagnostics, restoration, and failure paths.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from kan_phenotyping.experiments import forecasting_kan_ode as implementation
from kan_phenotyping.experiments import forecasting_kan_ode_full as forecasting
from kan_phenotyping.experiments.build_forecasting_dataset import sha256_file
from kan_phenotyping.experiments.forecasting_baselines import BaselinePartition, SavedBaselineData
from kan_phenotyping.experiments.forecasting_sequential_kan import prepare_sequential_data
from kan_phenotyping.training.trainer import seed_everything


class FrozenPreprocessor:
    """Reject refitting or transformations: saved arrays are already fitted."""

    def __init__(self, n_training_samples):
        wearable = [f"wearable_{index}" for index in range(48)]
        demographics = [f"demographic_{index}" for index in range(62)]
        self.feature_names = {
            "wearable": wearable, "demographics": demographics,
            "sm0_phq": ["measured_sm0_phq9_standardized"],
            "tabular": [f"SM{month}__{name}" for month in (1, 2) for name in wearable]
            + [f"baseline_demographic__{name}" for name in demographics]
            + ["measured_sm0_phq9_standardized"],
        }
        self.metadata = {"fitted_partition": "training_only", "training_samples": n_training_samples,
                         "validation_used_for_fit": False, "calibration_used_for_fit": False,
                         "test_used_for_fit": False, "targets_used_for_fit": False}

    def export_metadata(self):
        return deepcopy(self.metadata)

    def fit(self, *args, **kwargs):
        raise AssertionError("Saved preprocessing must never be fitted again.")

    def transform(self, *args, **kwargs):
        raise AssertionError("Saved arrays must never be transformed again.")


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(174)

    def partition(n, offset, prefix):
        features = rng.normal(0, .4, (n, 159)).astype(np.float32).astype(np.float64)
        sm0 = ((np.arange(n) * 5 + 3) % 28).astype(np.float64)
        features[:, -1] = (sm0 - 13.5) / 8
        targets = (2 + np.arange(n) % 4).astype(np.float64)
        return BaselinePartition(
            sample_indices=rng.permutation(np.arange(offset, offset + n)),
            participants=np.array([f"{prefix}{index // 3}" for index in range(n)]),
            sm0_phq=sm0, tabular=features, targets=targets,
        )

    train, validation = partition(24, 1000, "train-"), partition(8, 2000, "validation-")
    preprocessing = FrozenPreprocessor(len(train.targets))
    root = tmp_path / "synthetic_saved_dataset"
    root.mkdir()
    source = SavedBaselineData(
        train=train, validation=validation, preprocessor=preprocessing,
        manifest={"preprocessing": preprocessing.export_metadata(), "dataset": {"conditional_timing": True},
                  "split": {"seed": 42}, "inputs": {"synthetic": {"sha256": "not-real-data"}},
                  "source_sha256": {}},
        dataset_dir=root, read_files=(), manifest_sha256="synthetic-cohort-manifest",
    )
    return prepare_sequential_data(source)


@pytest.fixture
def tiny_config(monkeypatch):
    original = forecasting.experiment_config

    def config(seed):
        return replace(original(seed), max_epochs=2, batch_size=8, num_threads=1)

    monkeypatch.setattr(forecasting, "experiment_config", config)
    return config


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_fixed_full_training_settings_and_unchanged_architecture(seed):
    config = forecasting.experiment_config(seed)
    assert config.seed == seed and config.max_epochs == 50
    assert config.batch_size == 128 and config.patience == 10
    assert config.learning_rate == .001 and config.weight_decay == .0001
    assert config.gradient_clip == 5 and config.regression_loss == "mse"
    assert config.min_delta == 0 and config.task == "regression"
    assert config.device == "cpu" and config.num_threads == 2
    actual, existing = forecasting.model_spec(), implementation.model_spec()
    assert actual.pop("workflow") == "conditional_relative_month_latent_KAN_ODE_full_training"
    existing.pop("workflow")
    assert actual == existing
    assert actual["kwargs"]["step_size"] == .25
    assert actual["dynamics_initialization"] == {"base_and_spline_scale": .01, "bias": 0.}


@pytest.mark.parametrize("seed", [0, 41, 45, None, True, "42"])
def test_fixed_experiment_rejects_unapproved_seeds(seed):
    with pytest.raises(ValueError, match="42, 43, 44"):
        forecasting.experiment_config(seed)


def test_numerical_diagnostic_subset_matches_smoke_and_ignores_outcomes(data):
    expected = implementation.participant_subset(data.train)
    expected = implementation.subset_rows(expected, np.arange(8))
    actual = forecasting.diagnostic_partition(data.train)
    changed = replace(data.train, targets=27 - data.train.targets)
    altered = forecasting.diagnostic_partition(changed)
    for name in ("sample_indices", "participants", "wearable", "static", "targets"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    np.testing.assert_array_equal(actual.sample_indices, altered.sample_indices)


def test_preprocessing_guard_stops_before_model_initialization(data, tmp_path, monkeypatch):
    data.source.preprocessor.metadata["training_samples"] += 1
    monkeypatch.setattr(forecasting, "build_model", lambda: pytest.fail("Reject changed preprocessing first."))
    with pytest.raises(ValueError, match="unchanged saved training-only preprocessing"):
        forecasting.train_seed(data, tmp_path / "changed_preprocessing", 42)


def test_full_protocol_tiny_fit_is_fresh_reproducible_and_records_preclip_gradients(
    data, tmp_path, tiny_config, monkeypatch,
):
    before_metadata = data.source.preprocessor.export_metadata()
    before_arrays = {(name, field): getattr(getattr(data, name), field).copy()
                     for name in ("train", "validation")
                     for field in ("sample_indices", "participants", "wearable", "static", "targets")}
    original_clip = torch.nn.utils.clip_grad_norm_
    observed_preclip, options = [], []

    def record_clip(parameters, max_norm, **kwargs):
        parameters = list(parameters)
        observed_preclip.append(float(torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(parameter.grad.detach().double())
            for parameter in parameters if parameter.grad is not None
        ]))))
        options.append((max_norm, kwargs))
        return original_clip(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    results, predictions, states, initial_states = [], [], [], []
    for repetition, seed in enumerate((42, 42, 43)):
        # A fresh seed, rather than the preceding experiment's RNG or weights,
        # determines every initialization even after unrelated RNG use.
        seed_everything(800 + repetition, 1)
        torch.randn(29)
        output = tmp_path / f"tiny_seed_{seed}_repeat_{repetition}"
        result = forecasting.train_seed(data, output, seed)
        results.append(result)
        initial = torch.load(output / "initial.pt", map_location="cpu", weights_only=True)
        initial_states.append(initial["model_state"])
        seed_everything(seed, 1)
        expected = forecasting.build_model().state_dict()
        for name, value in expected.items():
            torch.testing.assert_close(initial["model_state"][name], value, rtol=0, atol=0)
        assert initial["initialization_seed"] == seed
        assert result["fresh_initialization"] is True
        assert result["warm_start_from_smoke_or_sanity"] is False
        assert result["parameter_count"] == 26257
        assert result["best_epoch"] in (1, 2) and result["stopped_epoch"] == 2
        assert result["stop_reason"] == "epoch_budget"
        assert result["restored_predictions_and_metrics_exact"] is True
        assert result["preprocessing_unchanged"] is True and result["numerical_failures"] == []
        assert result["dynamics_change"]["every_dynamics_parameter_tensor_changed"] is True
        assert all(value > 0 for value in result["dynamics_change"]["l2_change_from_initial_by_parameter"].values())
        checks = result["gradient_checks"]
        assert checks["completed_backward_updates"] == 6
        assert checks["all_observed_gradients_finite"] is True
        assert checks["all_parameters_received_nonzero_gradients"] is True
        assert len(checks["backward_calls_by_parameter"]) == 16
        assert set(checks["backward_calls_by_parameter"].values()) == {6}
        assert checks["pending_parameters"] == []
        rows = json.loads((output / "gradient_norms.json").read_text())
        assert len(rows) == 6
        for row in rows:
            assert row["epoch"] in (1, 2)
            assert set(row["component_norms"]) == {"static_encoder", "wearable_encoder", "gru", "head", "dynamics"}
            assert all(value > 0 for value in row["dynamics_l2_update_by_parameter"].values())
        np.testing.assert_allclose([row["preclip_global_norm"] for row in rows],
                                   observed_preclip[repetition * 6:(repetition + 1) * 6], rtol=1e-12)
        assert result["solver_calls_during_fit_including_epoch_validation"] == {
            "integrate_calls": 24, "steps": 96, "rhs_evaluations": 384,
        }
        assert result["solver_calls_final_validation"]["rhs_evaluations"] == 48
        epochs = json.loads((output / "epoch_diagnostics.json").read_text())
        assert [row["optimizer_updates"] for row in epochs] == [3, 3]
        assert [row["validation_solver_calls"]["rhs_evaluations"] for row in epochs] == [48, 48]
        assert [row["training_solver_calls"]["rhs_evaluations"] for row in epochs] == [144, 144]
        for key in ("step_size_initial", "step_size_restored"):
            assert result[key]["training_step_size_unchanged"] is True
            assert result[key]["validation_used_for_step_selection"] is False
            assert result[key]["instability_warning"] is False
        metrics, restored = forecasting.restore_validation(output / "best.pt", data)
        assert metrics == result["validation"]
        with np.load(output / "validation_predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["sample_indices"], data.validation.sample_indices)
            np.testing.assert_array_equal(saved["measured_sm3"], data.validation.targets)
            np.testing.assert_array_equal(saved["prediction"], restored)
            predictions.append(saved["prediction"].copy())
        states.append(torch.load(output / "best.pt", map_location="cpu", weights_only=True)["model_state"])
        assert json.loads((output / "training_config.json").read_text()) == asdict(tiny_config(seed))
        history = json.loads((output / "history.json").read_text())
        assert result["validation"]["rmse"] == min(row["validation"]["rmse"] for row in history)
    assert results[0]["validation"] == results[1]["validation"]
    np.testing.assert_array_equal(predictions[0], predictions[1])
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    assert any(not torch.equal(initial_states[0][key], initial_states[2][key]) for key in initial_states[0])
    assert options == [(5., {"error_if_nonfinite": True})] * 18
    assert data.source.preprocessor.export_metadata() == before_metadata
    for (name, field), values in before_arrays.items():
        np.testing.assert_array_equal(getattr(getattr(data, name), field), values)


@pytest.mark.parametrize("diagnostic", [
    {"all_states_and_predictions_finite": False, "instability_warning": False},
    {"all_states_and_predictions_finite": True, "instability_warning": True},
])
def test_stability_guard_requires_both_finiteness_and_small_step_difference(diagnostic):
    with pytest.raises(FloatingPointError, match="stop and diagnose"):
        forecasting.assert_stable(diagnostic)


def test_restored_instability_stops_before_final_predictions_and_preserves_evidence(
    data, tmp_path, tiny_config, monkeypatch,
):
    original = forecasting.step_size_diagnostic
    calls = []

    def unstable_only_after_training(model, diagnostic):
        result = original(model, diagnostic)
        calls.append(result)
        if len(calls) == 2:
            result["instability_warning"] = True
            result["prediction_max_absolute_difference"] = .1
        return result

    monkeypatch.setattr(forecasting, "step_size_diagnostic", unstable_only_after_training)
    monkeypatch.setattr(forecasting.DiagnosticTrainer, "predict",
                        lambda *args: pytest.fail("No final performance after material solver instability."))
    output = tmp_path / "unstable_restored"
    with pytest.raises(FloatingPointError, match="stop and diagnose"):
        forecasting.train_seed(data, output, 42)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["phase"] == "restored numerical diagnostic"
    assert failure["performance_interpretation_permitted"] is False
    assert failure["silent_solver_changes"] is False
    assert (output / "failed_state.pt").exists() and (output / "best.pt").exists()
    assert (output / "gradient_norms.json").exists() and (output / "history.json").exists()
    assert not (output / "validation_predictions.npz").exists()
    assert not (output / "validation_metrics.json").exists()


def test_nonfinite_training_gradient_stops_and_preserves_failure_evidence(
    data, tmp_path, tiny_config, monkeypatch,
):
    original = forecasting.build_model

    def model_with_nonfinite_gradient():
        model = original()
        model.dynamics.base_weight.register_hook(lambda value: torch.full_like(value, float("nan")))
        return model

    monkeypatch.setattr(forecasting, "build_model", model_with_nonfinite_gradient)
    output = tmp_path / "failed_gradient"
    with pytest.raises(FloatingPointError, match="Nonfinite backward gradient"):
        forecasting.train_seed(data, output, 42)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["phase"] == "training and validation checkpoint selection"
    assert failure["performance_interpretation_permitted"] is False
    assert failure["gradient_checks"]["all_observed_gradients_finite"] is False
    assert failure["gradient_checks"]["numerical_failures"][0]["parameter"] == "dynamics.base_weight"
    assert (output / "failed_state.pt").exists() and (output / "initial.pt").exists()
    assert not (output / "validation_predictions.npz").exists()


@pytest.mark.parametrize("change", ["architecture", "solver_step", "workflow", "input_kind", "epochs", "clip"])
def test_restore_rejects_altered_fixed_protocol_before_validation(data, tmp_path, monkeypatch, change):
    saved = {"model_metadata": forecasting.model_spec(), "input_kind": "temporal",
             "training_config": asdict(forecasting.experiment_config(42)), "model_state": {}}
    if change == "architecture":
        saved["model_metadata"]["kwargs"]["hidden_width"] = 8
    elif change == "solver_step":
        saved["model_metadata"]["kwargs"]["step_size"] = .125
    elif change == "workflow":
        saved["model_metadata"]["workflow"] = "smoke"
    elif change == "input_kind":
        saved["input_kind"] = "tabular"
    else:
        key, value = {"epochs": ("max_epochs", 2), "clip": ("gradient_clip", None)}[change]
        saved["training_config"][key] = value
    checkpoint = tmp_path / "wrong_protocol.pt"
    torch.save(saved, checkpoint)
    monkeypatch.setattr(forecasting, "build_model", lambda: pytest.fail("Reject metadata before inference."))
    with pytest.raises(ValueError, match="fixed full-training"):
        forecasting.restore_validation(checkpoint, data)


@pytest.fixture
def recurrent_directory(data, tmp_path, monkeypatch):
    monkeypatch.setattr(forecasting, "_comparisons", lambda *args: {"provenance": {}})
    root = tmp_path / "synthetic_recurrent_control"
    root.mkdir()
    summary = {"mean_and_sample_sd": {"rmse": {"mean": 1., "sample_sd": .1}},
               "seeds": [{"seed": seed, "validation": {"mae": 1., "rmse": 1., "r2": .1}}
                         for seed in (42, 43, 44)]}
    (root / "summary.json").write_text(json.dumps(summary))
    for seed in (42, 43, 44):
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir()
        config = asdict(forecasting.experiment_config(seed))
        config["gradient_clip"] = None
        (seed_dir / "training_config.json").write_text(json.dumps(config))
        np.savez_compressed(seed_dir / "validation_predictions.npz",
                            sample_indices=data.validation.sample_indices,
                            measured_sm3=data.validation.targets, prediction=data.validation.targets + .5)
    files = [path for path in root.rglob("*") if path.is_file()]
    (root / "run.json").write_text(json.dumps({
        "dataset_manifest_sha256": data.source.manifest_sha256,
        "artifact_sha256": {str(path.relative_to(root)): sha256_file(path) for path in files},
    }))
    return root


def test_recurrent_comparison_reports_actual_gradient_clipping_difference(data, recurrent_directory):
    result = forecasting.load_comparisons(data.source, ".", ".", ".", recurrent_directory)
    assert result["comparison_limitations"]["not_a_perfectly_matched_dynamics_control"] is True
    assert result["comparison_limitations"]["parameters"] == {"recurrent": 23169, "latent_kan_ode": 26257}
    for seed in ("42", "43", "44"):
        assert result["training_config_differences"][seed] == {
            "gradient_clip": {"recurrent": None, "latent_kan_ode": 5.},
        }
    assert set(result["provenance"]["recurrent"]["sha256"]) == {
        "run.json", "summary.json", *[f"seed_{seed}/{name}" for seed in (42, 43, 44)
                                     for name in ("training_config.json", "validation_predictions.npz")],
    }


@pytest.mark.parametrize("change", ["learning_rate", "gradient_clip", "max_epochs"])
def test_unexpected_recurrent_control_configuration_stops_comparison(
    data, recurrent_directory, change,
):
    path = recurrent_directory / "seed_42/training_config.json"
    config = json.loads(path.read_text())
    config[change] = {"learning_rate": .002, "gradient_clip": 1., "max_epochs": 20}[change]
    path.write_text(json.dumps(config))
    manifest_path = recurrent_directory / "run.json"
    run = json.loads(manifest_path.read_text())
    run["artifact_sha256"][str(path.relative_to(recurrent_directory))] = sha256_file(path)
    manifest_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="Unexpected recurrent-control training configuration"):
        forecasting.load_comparisons(data.source, ".", ".", ".", recurrent_directory)
