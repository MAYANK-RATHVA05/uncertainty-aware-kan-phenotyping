"""Synthetic checks for the optimization-matched recurrent control.

These fixtures never load participant data or trained ODE checkpoints. Fresh
ODE construction is used only to establish the saved initialization contract.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from kan_phenotyping.evaluation.metrics import regression_metrics
from kan_phenotyping.experiments import forecasting_matched_recurrent as forecasting
from kan_phenotyping.experiments.build_forecasting_dataset import sha256_file
from kan_phenotyping.experiments.forecasting_baselines import BaselinePartition, SavedBaselineData
from kan_phenotyping.experiments.forecasting_kan import aggregate_seed_metrics
from kan_phenotyping.experiments.forecasting_sequential_kan import prepare_sequential_data
from kan_phenotyping.models.latent_kan_ode import LatentKANODE
from kan_phenotyping.training.trainer import seed_everything


class FrozenPreprocessor:
    def __init__(self, n):
        wearable = [f"wearable_{index}" for index in range(48)]
        demographics = [f"demographic_{index}" for index in range(62)]
        self.feature_names = {
            "wearable": wearable, "demographics": demographics,
            "sm0_phq": ["measured_sm0_phq9_standardized"],
            "tabular": [f"SM{month}__{name}" for month in (1, 2) for name in wearable]
            + [f"baseline_demographic__{name}" for name in demographics]
            + ["measured_sm0_phq9_standardized"],
        }
        self.metadata = {"fitted_partition": "training_only", "training_samples": n,
                         "validation_used_for_fit": False, "calibration_used_for_fit": False,
                         "test_used_for_fit": False, "targets_used_for_fit": False}

    def export_metadata(self):
        return deepcopy(self.metadata)

    def fit(self, *args, **kwargs):
        raise AssertionError("Never refit saved preprocessing.")

    def transform(self, *args, **kwargs):
        raise AssertionError("Never transform the already transformed arrays again.")


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(174)

    def partition(n, offset, prefix):
        x = rng.normal(0, .4, (n, 159)).astype(np.float32).astype(np.float64)
        sm0 = ((np.arange(n) * 5 + 3) % 28).astype(np.float64)
        x[:, -1] = (sm0 - 13.5) / 8
        # This unique saved static value lets tests recover minibatch order.
        x[:, 96] = np.arange(n)
        return BaselinePartition(
            sample_indices=rng.permutation(np.arange(offset, offset + n)),
            participants=np.array([f"{prefix}{index // 3}" for index in range(n)]),
            sm0_phq=sm0, tabular=x, targets=(2 + np.arange(n) % 4).astype(np.float64),
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


def write_initial(root, seed):
    path = root / f"seed_{seed}" / "initial.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, 1)
    model = LatentKANODE(**forecasting.ode_spec()["kwargs"])
    torch.save({"model_state": model.state_dict(), "initialization_seed": seed,
                "model_metadata": forecasting.ode_spec()}, path)
    return path


@pytest.fixture
def initial_paths(tmp_path):
    return {seed: write_initial(tmp_path / "synthetic_ode_initial", seed) for seed in (42, 43, 44)}


@pytest.fixture
def tiny_config(monkeypatch):
    original = forecasting.experiment_config

    def config(seed):
        return replace(original(seed), max_epochs=2, batch_size=8, num_threads=1)

    monkeypatch.setattr(forecasting, "experiment_config", config)
    return config


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_fixed_matched_settings_and_original_recurrent_architecture(seed):
    config = forecasting.experiment_config(seed)
    assert config.seed == seed and config.max_epochs == 50 and config.patience == 10
    assert config.batch_size == 128 and config.learning_rate == .001 and config.weight_decay == .0001
    assert config.gradient_clip == 5 and config.regression_loss == "mse"
    assert config.device == "cpu" and config.num_threads == 2 and config.min_delta == 0
    actual, original = forecasting.model_spec(), forecasting.recurrent_spec()
    assert actual.pop("workflow") == "conditional_relative_month_optimization_matched_clip5_recurrent_control"
    original.pop("workflow")
    assert actual == original
    assert sum(p.numel() for p in forecasting.build_sequential_model().parameters()) == 23169


@pytest.mark.parametrize("seed", [0, 41, 45, None, True, "42"])
def test_unapproved_seeds_rejected(seed):
    with pytest.raises(ValueError, match="42, 43, 44"):
        forecasting.experiment_config(seed)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_fresh_recurrent_components_exactly_match_saved_ode_initialization(seed, initial_paths):
    seed_everything(seed, 1)
    recurrent = forecasting.build_sequential_model()
    before = {name: value.clone() for name, value in recurrent.state_dict().items()}
    result = forecasting.initialization_check(recurrent, initial_paths[seed], seed)
    assert result["all_shared_state_tensors_exact"] is True
    assert result["component_matches"] == {name: True for name in forecasting.COMPONENTS}
    assert result["shared_state_tensors"] == 16 and result["shared_parameter_tensors"] == 13
    assert len(result["state_tensor_matches"]) == 16
    assert result["weights_copied_from_ode"] is False and result["trained_ode_weights_consumed"] is False
    assert result["ode_initial_sha256"] == sha256_file(initial_paths[seed])
    for name, value in recurrent.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["seed", "weight", "dtype", "metadata", "extra"])
def test_initialization_mismatch_rejected(initial_paths, mutation):
    path = initial_paths[42]
    initial = torch.load(path, map_location="cpu", weights_only=True)
    if mutation == "seed":
        initial["initialization_seed"] = 43
    elif mutation == "weight":
        initial["model_state"]["gru.weight_ih"][0, 0] += .01
    elif mutation == "dtype":
        initial["model_state"]["head.base_weight"] = initial["model_state"]["head.base_weight"].double()
    elif mutation == "metadata":
        initial["model_metadata"]["workflow"] = "smoke"
    else:
        initial["model_state"]["unrecognized.weight"] = torch.zeros(1)
    torch.save(initial, path)
    seed_everything(42, 1)
    with pytest.raises(ValueError):
        forecasting.initialization_check(forecasting.build_sequential_model(), path, 42)


@pytest.mark.parametrize("relative_path", ["seed_42/best.pt", "seed_43/initial.pt", "seed_42/sanity.pt"])
def test_trained_or_wrong_seed_path_rejected_before_file_open(tmp_path, monkeypatch, relative_path):
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Reject forbidden paths before reading."))
    with pytest.raises(ValueError, match="matching seed's initial.pt"):
        forecasting.initialization_check(forecasting.build_sequential_model(), tmp_path / relative_path, 42)


def test_minibatch_order_matches_ode_seed42_protocol_for_every_initialization(data):
    orders = []
    for seed in (42, 43, 44):
        config = replace(forecasting.experiment_config(seed), batch_size=8)
        seed_everything(seed + 100, 1)
        loader = forecasting.make_sequential_loader(data.train, config, shuffle=True)
        orders.append([torch.cat([batch["static"][:, 0] for batch in loader]).numpy() for _ in range(2)])
    for candidate in orders[1:]:
        np.testing.assert_array_equal(candidate, orders[0])
    assert not np.array_equal(orders[0][0], orders[0][1])
    for epoch in orders[0]:
        np.testing.assert_array_equal(np.sort(epoch), np.arange(24))


def test_changed_preprocessing_rejected_before_model_initialization(data, tmp_path, monkeypatch):
    data.source.preprocessor.metadata["training_samples"] += 1
    monkeypatch.setattr(forecasting, "build_sequential_model", lambda: pytest.fail("Reject preprocessing first."))
    with pytest.raises(ValueError, match="unchanged saved training-only preprocessing"):
        forecasting.train_seed(data, tmp_path / "changed_preprocessing", 42, tmp_path / "unused")


def test_real_tiny_fits_are_fresh_reproducible_aligned_and_record_unclipped_norms(
    data, tmp_path, initial_paths, tiny_config, monkeypatch,
):
    before_metadata = data.source.preprocessor.export_metadata()
    before_arrays = {(name, field): getattr(getattr(data, name), field).copy()
                     for name in ("train", "validation")
                     for field in ("sample_indices", "participants", "wearable", "static", "targets")}
    original_clip = torch.nn.utils.clip_grad_norm_
    independently_observed, clip_options = [], []

    def observe_clip(parameters, max_norm, **kwargs):
        parameters = list(parameters)
        independently_observed.append(float(torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(parameter.grad.detach().double())
            for parameter in parameters if parameter.grad is not None
        ]))))
        clip_options.append((max_norm, kwargs))
        return original_clip(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observe_clip)
    results, predictions, initial_states, restored_states = [], [], [], []
    for repetition, seed in enumerate((42, 42, 43)):
        seed_everything(700 + repetition, 1)
        torch.randn(29)
        output = tmp_path / f"tiny_seed_{seed}_repeat_{repetition}"
        result = forecasting.train_seed(data, output, seed, initial_paths[seed])
        results.append(result)
        initial = torch.load(output / "initial.pt", map_location="cpu", weights_only=True)
        initial_states.append(initial["model_state"])
        seed_everything(seed, 1)
        expected = forecasting.build_sequential_model().state_dict()
        for name, value in expected.items():
            torch.testing.assert_close(initial["model_state"][name], value, rtol=0, atol=0)
        assert initial["initialization_seed"] == seed and initial["model_metadata"] == forecasting.model_spec()
        assert result["fresh_initialization"] is True and result["trained_ode_weights_consumed"] is False
        assert result["parameter_count"] == 23169 and result["best_epoch"] in (1, 2)
        assert result["stopped_epoch"] == 2 and result["stop_reason"] == "epoch_budget"
        assert result["restored_predictions_and_metrics_exact"] is True
        assert result["preprocessing_unchanged"] is True and result["numerical_failures"] == []
        checks = result["gradient_checks"]
        assert checks["completed_backward_updates"] == 6 and checks["all_observed_gradients_finite"] is True
        assert checks["all_parameters_received_nonzero_gradients"] is True
        assert len(checks["backward_calls_by_parameter"]) == 13
        assert set(checks["backward_calls_by_parameter"].values()) == {6}
        assert checks["pending_parameters"] == []
        rows = json.loads((output / "gradient_norms.json").read_text())
        assert len(rows) == 6
        for row in rows:
            assert row["epoch"] in (1, 2)
            assert set(row["component_norms"]) == {"static_encoder", "wearable_encoder", "gru", "head"}
        norms = [row["preclip_global_norm"] for row in rows]
        np.testing.assert_allclose(norms, independently_observed[repetition * 6:(repetition + 1) * 6], rtol=1e-12)
        count = sum(norm > 5 for norm in norms)
        assert count > 0  # This fixture actually exercises clipping.
        assert result["clipping_frequency"] == {
            "updates_above_threshold": count, "updates": 6, "fraction": count / 6, "threshold": 5.,
        }
        epochs = json.loads((output / "epoch_diagnostics.json").read_text())
        assert [row["optimizer_updates"] for row in epochs] == [3, 3]
        assert sum(row["updates_above_clip5"] for row in epochs) == count
        assert result["prefix_checks"]["sm1_gradient_wrt_sm2_exactly_zero"] is True
        metrics, restored = forecasting.restore_validation(output / "best.pt", data)
        assert metrics == result["validation"]
        with np.load(output / "validation_predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["sample_indices"], data.validation.sample_indices)
            np.testing.assert_array_equal(saved["measured_sm3"], data.validation.targets)
            np.testing.assert_array_equal(saved["prediction"], restored)
            predictions.append(saved["prediction"].copy())
        restored_states.append(torch.load(output / "best.pt", map_location="cpu", weights_only=True)["model_state"])
        assert json.loads((output / "training_config.json").read_text()) == asdict(tiny_config(seed))
        history = json.loads((output / "history.json").read_text())
        assert result["validation"]["rmse"] == min(row["validation"]["rmse"] for row in history)
    assert results[0]["validation"] == results[1]["validation"]
    np.testing.assert_array_equal(predictions[0], predictions[1])
    for name in restored_states[0]:
        torch.testing.assert_close(restored_states[0][name], restored_states[1][name], rtol=0, atol=0)
    assert any(not torch.equal(initial_states[0][name], initial_states[2][name]) for name in initial_states[0])
    assert clip_options == [(5., {"error_if_nonfinite": True})] * 18
    assert data.source.preprocessor.export_metadata() == before_metadata
    for (name, field), values in before_arrays.items():
        np.testing.assert_array_equal(getattr(getattr(data, name), field), values)


def test_nonfinite_gradient_preserves_failure_state_and_stops_before_evaluation(
    data, tmp_path, initial_paths, tiny_config, monkeypatch,
):
    original = forecasting.build_sequential_model

    def invalid_gradient_model():
        model = original()
        model.head.base_weight.register_hook(lambda value: torch.full_like(value, float("nan")))
        return model

    monkeypatch.setattr(forecasting, "build_sequential_model", invalid_gradient_model)
    output = tmp_path / "nonfinite_gradient"
    with pytest.raises(FloatingPointError, match="Nonfinite backward gradient"):
        forecasting.train_seed(data, output, 42, initial_paths[42])
    failure = json.loads((output / "failure.json").read_text())
    assert failure["phase"] == "training and checkpoint selection"
    assert failure["performance_interpretation_permitted"] is False
    assert failure["gradient_checks"]["all_observed_gradients_finite"] is False
    assert failure["gradient_checks"]["numerical_failures"][0]["parameter"] == "head.base_weight"
    assert (output / "failed_state.pt").exists() and (output / "initial.pt").exists()
    assert not (output / "validation_predictions.npz").exists()
    assert not (output / "validation_metrics.json").exists()


@pytest.mark.parametrize("change", ["architecture", "workflow", "input_kind", "epochs", "clip"])
def test_restore_rejects_altered_protocol_before_inference(data, tmp_path, monkeypatch, change):
    saved = {"model_metadata": forecasting.model_spec(), "input_kind": "temporal",
             "training_config": asdict(forecasting.experiment_config(42)), "model_state": {}}
    if change == "architecture":
        saved["model_metadata"]["kwargs"]["hidden_width"] = 8
    elif change == "workflow":
        saved["model_metadata"]["workflow"] = "original_unclipped"
    elif change == "input_kind":
        saved["input_kind"] = "tabular"
    else:
        key, value = {"epochs": ("max_epochs", 2), "clip": ("gradient_clip", None)}[change]
        saved["training_config"][key] = value
    path = tmp_path / "wrong_protocol.pt"
    torch.save(saved, path)
    monkeypatch.setattr(forecasting, "build_sequential_model", lambda: pytest.fail("Reject metadata first."))
    with pytest.raises(ValueError, match="optimization-matched clip-5"):
        forecasting.restore_validation(path, data)


@pytest.fixture
def ode_reference(data, tmp_path):
    root = tmp_path / "synthetic_ode_reference"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"minibatch_order_seed": 42, "model": forecasting.ode_spec()}))
    rows = []
    for seed in (42, 43, 44):
        initial = write_initial(root, seed)
        (initial.parent / "training_config.json").write_text(json.dumps(asdict(forecasting.experiment_config(seed))))
        prediction = data.validation.targets + (seed - 40) / 10
        metrics = regression_metrics(data.validation.targets, prediction)
        rows.append({"seed": seed, "validation": metrics})
        (initial.parent / "validation_metrics.json").write_text(json.dumps({"validation": metrics}))
        np.savez_compressed(initial.parent / "validation_predictions.npz", sample_indices=data.validation.sample_indices,
                            measured_sm3=data.validation.targets, prediction=prediction)
    (root / "summary.json").write_text(json.dumps({"seeds": rows, "mean_and_sample_sd": aggregate_seed_metrics(rows),
        "parameter_count": 26257, "comparisons": {"recurrent_KAN": {"preserved": True}}}))
    files = [path for path in root.rglob("*") if path.is_file()]
    (root / "run.json").write_text(json.dumps({"dataset_manifest_sha256": data.source.manifest_sha256,
        "device": "cpu", "num_threads": 2, "deterministic_algorithms": True,
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
        "source_sha256": {}, "consumed_sha256": {},
        "artifact_sha256": {str(path.relative_to(root)): sha256_file(path) for path in files}}))
    return root


def test_reference_validates_initial_artifact_hashes_and_aligned_metrics_without_trained_weights(
    data, ode_reference, monkeypatch,
):
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Reference loading needs no trained weights."))
    result = forecasting.load_ode_reference(data, ode_reference)
    assert result["trained_ode_weights_consumed"] is False and result["parameter_count"] == 26257
    assert result["minibatch_order_seed"] == 42 and result["precision"] == "float32"
    for seed in (42, 43, 44):
        assert result["training_by_seed"][str(seed)] == asdict(forecasting.experiment_config(seed))
        assert f"seed_{seed}/initial.pt" in result["consumed_sha256"]
    assert all("best.pt" not in name for name in result["consumed_sha256"])


@pytest.mark.parametrize("mutation", ["manifest", "version", "hash", "config", "alignment", "target", "metric"])
def test_reference_rejects_provenance_configuration_and_alignment_mismatches(data, ode_reference, mutation):
    run_path = ode_reference / "run.json"
    run = json.loads(run_path.read_text())
    if mutation == "manifest":
        run["dataset_manifest_sha256"] = "different-cohort"
    elif mutation == "version":
        run["versions"]["torch"] = "different-runtime"
    elif mutation == "hash":
        run["artifact_sha256"]["seed_42/initial.pt"] = "wrong-hash"
    else:
        if mutation == "config":
            path = ode_reference / "seed_42/training_config.json"
            config = json.loads(path.read_text())
            config["gradient_clip"] = None
            path.write_text(json.dumps(config))
        elif mutation == "metric":
            path = ode_reference / "seed_42/validation_metrics.json"
            metrics = json.loads(path.read_text())
            metrics["validation"]["rmse"] += .1
            path.write_text(json.dumps(metrics))
        else:
            path = ode_reference / "seed_42/validation_predictions.npz"
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            arrays["sample_indices" if mutation == "alignment" else "measured_sm3"][0] += 1
            np.savez_compressed(path, **arrays)
        run["artifact_sha256"][str(path.relative_to(ode_reference))] = sha256_file(path)
    run_path.write_text(json.dumps(run))
    with pytest.raises((ValueError, AssertionError)):
        forecasting.load_ode_reference(data, ode_reference)


def test_paired_differences_are_ode_minus_matched_in_original_metric_direction():
    records = [{"seed": seed, "validation": {"mae": 2., "rmse": 3., "r2": .6}} for seed in (42, 43, 44)]
    reference = {"seed_metrics": {str(seed): {"mae": 2.1, "rmse": 2.9, "r2": .61} for seed in (42, 43, 44)}}
    result = forecasting.paired_differences(records, reference)
    for row in result:
        np.testing.assert_allclose([row["ode_minus_matched"][name] for name in ("mae", "rmse", "r2")], [.1, -.1, .01])
    with pytest.raises(ValueError, match="canonical order"):
        forecasting.paired_differences(list(reversed(records)), reference)
