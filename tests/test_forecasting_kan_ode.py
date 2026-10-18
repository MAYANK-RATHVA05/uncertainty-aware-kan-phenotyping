"""Synthetic protocol checks for the limited conditional latent KAN-ODE smoke.

These tests use no uploaded observations or reserved partition payloads. Tiny
two-epoch fits exercise the real implementation without running the separate
overfit diagnostic or conducting a forecasting experiment.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from kan_phenotyping.experiments import forecasting_kan_ode as forecasting
from kan_phenotyping.experiments.forecasting_baselines import (
    BaselinePartition, SavedBaselineData,
)
from kan_phenotyping.training.trainer import seed_everything


class FrozenPreprocessor:
    """Saved metadata only: any attempt to fit/transform is a test failure."""

    def __init__(self, n_training_samples):
        wearable = [f"numeric__wearable_{index}" for index in range(27)]
        wearable += [f"numeric__missingindicator_wearable_{index}" for index in range(21)]
        demographics = [f"demographic__feature_{index}" for index in range(62)]
        self.feature_names = {
            "wearable": wearable, "demographics": demographics,
            "sm0_phq": ["measured_sm0_phq9_standardized"],
            "tabular": [f"SM{month}__{name}" for month in (1, 2) for name in wearable]
            + [f"baseline_demographic__{name}" for name in demographics]
            + ["measured_sm0_phq9_standardized"],
        }
        self.metadata = {
            "fitted_partition": "training_only", "training_samples": n_training_samples,
            "validation_used_for_fit": False, "calibration_used_for_fit": False,
            "test_used_for_fit": False, "targets_used_for_fit": False,
        }

    def export_metadata(self):
        return deepcopy(self.metadata)

    def fit(self, *args, **kwargs):
        raise AssertionError("Saved preprocessing must never be fitted again.")

    def transform(self, *args, **kwargs):
        raise AssertionError("Reuse saved arrays without further transformation.")


@pytest.fixture
def saved_data(tmp_path):
    rng = np.random.default_rng(74)

    def partition(n, offset, prefix, repeats):
        features = rng.normal(0, .4, (n, 159)).astype(np.float32).astype(np.float64)
        measured_sm0 = ((np.arange(n) * 5 + 3) % 28).astype(np.float64)
        features[:, -1] = (measured_sm0 - 13.5) / 8
        target = (2 + np.arange(n) % 4).astype(np.float64)
        features[:, 96] = (target - 3.5) / 2
        return BaselinePartition(
            sample_indices=rng.permutation(np.arange(offset, offset + n)),
            participants=np.array([f"{prefix}{index // repeats}" for index in range(n)]),
            sm0_phq=measured_sm0, tabular=features, targets=target,
        )

    train = partition(24, 1000, "training-person-", 3)
    validation = partition(8, 2000, "validation-person-", 2)
    preprocessor = FrozenPreprocessor(len(train.targets))
    root = tmp_path / "saved_dataset"
    root.mkdir()
    consumed = ("manifest.json", "preprocessor.joblib", "train_inputs.npz",
                "train_targets.npz", "validation_inputs.npz", "validation_targets.npz")
    # Only the mocked run_smoke boundary reads these bytes for provenance. No
    # test deserializes them or creates calibration/test payloads.
    for filename in consumed:
        (root / filename).write_bytes(f"synthetic {filename}\n".encode())
    return SavedBaselineData(
        train=train, validation=validation, preprocessor=preprocessor,
        manifest={"preprocessing": preprocessor.export_metadata(),
                  "dataset": {"conditional_timing": True}, "split": {"seed": 42},
                  "inputs": {"data": {"sha256": "synthetic-input"}},
                  "source_sha256": {"synthetic-builder": "not-a-real-source-hash"}},
        dataset_dir=root, read_files=consumed,
        manifest_sha256="synthetic-fixture-not-a-real-input-hash",
    )


@pytest.fixture
def data(saved_data):
    return forecasting.prepare_sequential_data(saved_data)


def test_smoke_settings_are_fixed_two_epochs_and_gradient_clip_five():
    config = forecasting.smoke_config()
    assert config.seed == 42 and config.max_epochs == 2
    assert config.learning_rate == .001 and config.weight_decay == .0001
    assert config.batch_size == 128 and config.patience == 10
    assert config.gradient_clip == 5 and config.regression_loss == "mse"
    assert config.task == "regression" and config.min_delta == 0
    assert config.device == "cpu" and config.num_threads == 2
    spec = forecasting.model_spec()
    assert spec["kwargs"]["step_size"] == .25
    assert spec["times"] == [0, 1, 2, 3]
    assert spec["dynamics_initialization"] == {"base_and_spline_scale": .01, "bias": 0.}
    assert spec["solver"]["differentiation"] == "direct PyTorch autograd, not adjoint"


def test_participant_subset_is_outcome_independent_deterministic_and_retains_all_rows(data):
    selected = forecasting.participant_subset(data.train, count=3, seed=42)
    repeated = forecasting.participant_subset(data.train, count=3, seed=42)
    altered = replace(data.train, targets=27 - data.train.targets,
                      wearable=data.train.wearable * -11, static=data.train.static + 10)
    counterfactual = forecasting.participant_subset(altered, count=3, seed=42)
    expected_ids = np.random.default_rng(42).permutation(np.unique(data.train.participants))[:3]
    expected_rows = np.flatnonzero(np.isin(data.train.participants, expected_ids))
    assert len(selected.targets) == 9
    assert set(selected.participants) == set(expected_ids)
    for field in ("sample_indices", "participants", "wearable", "static", "targets"):
        np.testing.assert_array_equal(getattr(selected, field), getattr(repeated, field))
        np.testing.assert_array_equal(getattr(selected, field), getattr(data.train, field)[expected_rows])
    np.testing.assert_array_equal(selected.sample_indices, counterfactual.sample_indices)
    np.testing.assert_array_equal(selected.participants, counterfactual.participants)
    for participant in expected_ids:
        np.testing.assert_array_equal(
            selected.sample_indices[selected.participants == participant],
            data.train.sample_indices[data.train.participants == participant],
        )
    assert not np.shares_memory(selected.wearable, data.train.wearable)
    # A budget exceeding available groups retains the full saved cohort.
    np.testing.assert_array_equal(forecasting.participant_subset(data.train).sample_indices,
                                  data.train.sample_indices)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2", None])
def test_participant_subset_rejects_invalid_budgets(data, count):
    with pytest.raises(ValueError, match="positive integer"):
        forecasting.participant_subset(data.train, count=count)


def test_saved_transformed_columns_and_loader_keep_month_and_sample_alignment(data, saved_data):
    assert data.source is saved_data
    for name in ("train", "validation"):
        original, prepared = getattr(saved_data, name), getattr(data, name)
        np.testing.assert_array_equal(prepared.wearable[:, 0], original.tabular[:, :48])
        np.testing.assert_array_equal(prepared.wearable[:, 1], original.tabular[:, 48:96])
        np.testing.assert_array_equal(prepared.static, original.tabular[:, 96:])
        loader = forecasting.make_sequential_loader(prepared, forecasting.smoke_config(), shuffle=False)
        batches = list(loader)
        for key, expected in (("temporal", prepared.wearable), ("static", prepared.static),
                              ("target", prepared.targets)):
            np.testing.assert_array_equal(torch.cat([batch[key] for batch in batches]).numpy(),
                                          expected.astype(np.float32))
        for field in ("sample_indices", "participants", "targets"):
            np.testing.assert_array_equal(getattr(prepared, field), getattr(original, field))


def test_step_diagnostic_uses_same_training_prefix_and_preserves_parameters_and_step(data):
    seed_everything(42, 2)
    model = forecasting.build_model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    contexts, packets = [], []
    handles = [
        model.static_encoder.register_forward_pre_hook(
            lambda module, args: contexts.append(args[0].clone())),
        model.wearable_encoder.register_forward_pre_hook(
            lambda module, args: packets.append(args[0].clone())),
    ]
    try:
        diagnostic = forecasting.step_size_diagnostic(model, data.train)
    finally:
        for handle in handles:
            handle.remove()
    assert len(contexts) == 2 and len(packets) == 4
    for context in contexts:
        np.testing.assert_array_equal(context.numpy(), data.train.static[:8])
    for index, packet in enumerate(packets):
        np.testing.assert_array_equal(packet.numpy(), data.train.wearable[:8, index % 2])
    assert diagnostic["steps_compared"] == [.25, .125]
    assert diagnostic["training_only"] is True and diagnostic["samples"] == 8
    assert diagnostic["validation_used_for_step_selection"] is False
    assert diagnostic["parameters_unchanged"] is True
    assert diagnostic["all_states_and_predictions_finite"] is True
    assert diagnostic["training_step_size_unchanged"] is True and model.step_size == .25
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert diagnostic["solver_calls"]["0.25"] == {
        "integrate_calls": 3, "steps": 12, "rhs_evaluations": 48,
    }
    assert diagnostic["solver_calls"]["0.125"] == {
        "integrate_calls": 3, "steps": 24, "rhs_evaluations": 96,
    }


def test_nonfinite_diagnostic_is_reported_and_restores_default_step(data, monkeypatch):
    model = forecasting.build_model()
    monkeypatch.setattr(model.dynamics, "forward", lambda state: torch.full_like(state, float("nan")))
    with pytest.raises(FloatingPointError, match="non-finite"):
        forecasting.step_size_diagnostic(model, data.train)
    assert model.step_size == .25


def test_prefix_diagnostic_proves_sm1_has_no_access_to_sm2(data):
    seed_everything(42, 2)
    check = forecasting.prefix_check(forecasting.build_model(), data.train)
    assert check == {"sm1_gradient_wrt_sm2_exactly_zero": True,
                     "sm1_state_unchanged_when_sm2_changes": True,
                     "prefix_without_sm2_matches": True,
                     "sm1_and_context_gradients_nonzero": True}


@pytest.mark.parametrize("field", ["sample_indices", "participants", "wearable", "static", "targets", "partial_history"])
def test_smoke_rejects_substituted_subset_content_before_training(data, tmp_path, monkeypatch, field):
    selected = forecasting.participant_subset(data.train)
    if field == "partial_history":
        changed = forecasting.subset_rows(selected, np.arange(1, len(selected.targets)))
    else:
        values = getattr(selected, field).copy()
        if field == "participants":
            values[0] = "wrong-person"
        elif field in {"sample_indices", "targets"}:
            values[0] += 1
        else:
            values.flat[0] += .5
        changed = replace(selected, **{field: values})
    monkeypatch.setattr(forecasting, "build_model", lambda: pytest.fail("Invalid subset must fail before building/fitting."))
    with pytest.raises(AssertionError):
        forecasting.train_smoke(data, changed, tmp_path / f"tampered_{field}")


def test_smoke_rejects_changed_preprocessing_metadata(data, tmp_path, monkeypatch):
    data.source.preprocessor.metadata["training_samples"] += 1
    monkeypatch.setattr(forecasting, "build_model", lambda: pytest.fail("Invalid preprocessing must fail before fitting."))
    with pytest.raises(ValueError, match="preprocessing"):
        forecasting.train_smoke(data, forecasting.participant_subset(data.train), tmp_path / "changed_preprocessor")


def test_real_two_epoch_smoke_is_fresh_reproducible_and_restores_aligned_predictions(data, tmp_path, monkeypatch):
    before_metadata = data.source.preprocessor.export_metadata()
    before_names = deepcopy(data.source.preprocessor.feature_names)
    before_arrays = {(name, field): getattr(getattr(data, name), field).copy()
                     for name in ("train", "validation")
                     for field in ("wearable", "static", "targets", "participants", "sample_indices")}
    adamw_options, clip_options = [], []
    original_adamw, original_clip = torch.optim.AdamW, torch.nn.utils.clip_grad_norm_

    def record_adamw(parameters, **kwargs):
        adamw_options.append(kwargs.copy())
        return original_adamw(parameters, **kwargs)

    def record_clip(parameters, max_norm, **kwargs):
        clip_options.append((max_norm, kwargs.copy()))
        return original_clip(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", record_adamw)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    reports, states, predictions = [], [], []
    for index in (0, 1):
        # Unrelated RNG consumption must not warm-start or change smoke init.
        seed_everything(900 + index, 2)
        torch.randn(20)
        output = tmp_path / f"repeat_{index}"
        report = forecasting.train_smoke(data, forecasting.participant_subset(data.train), output)
        reports.append(report)
        states.append(torch.load(output / "best.pt", map_location="cpu", weights_only=True)["model_state"])
        initial = torch.load(output / "initial.pt", map_location="cpu", weights_only=True)
        seed_everything(42, 2)
        expected_initial = forecasting.build_model().state_dict()
        assert initial["initialization_seed"] == 42
        for key, value in expected_initial.items():
            torch.testing.assert_close(initial["model_state"][key], value, rtol=0, atol=0)
        assert report["preliminary_smoke_only"] is True and report["epochs_run"] == 2
        assert 1 <= report["best_epoch"] <= 2
        assert report["parameter_count"] == 26257
        assert report["restored_predictions_match"] is True
        assert report["preprocessing_unchanged"] is True
        assert report["fresh_initialization_not_sanity_weights"] is True
        assert report["numerical_failures"] == []
        assert report["gradient_checks"]["all_observed_gradients_finite"] is True
        assert all(report["gradient_checks"]["nonzero_gradient_seen_by_parameter"].values())
        assert len(report["gradient_checks"]["backward_calls_by_parameter"]) == 16
        metrics, restored = forecasting.restore_validation(output / "best.pt", data)
        assert metrics == report["validation"]
        with np.load(output / "validation_predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["sample_indices"], data.validation.sample_indices)
            np.testing.assert_array_equal(saved["measured_sm3"], data.validation.targets)
            np.testing.assert_array_equal(saved["prediction"], restored)
            predictions.append(saved["prediction"].copy())
        assert json.loads((output / "training_config.json").read_text()) == asdict(forecasting.smoke_config())
        history = json.loads((output / "history.json").read_text())
        assert [row["epoch"] for row in history] == [1, 2]
        assert report["validation"]["rmse"] == min(row["validation"]["rmse"] for row in history)
    assert reports[0]["validation"] == reports[1]["validation"]
    assert reports[0]["best_epoch"] == reports[1]["best_epoch"]
    np.testing.assert_array_equal(*predictions)
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    assert adamw_options == [{"lr": .001, "weight_decay": .0001}] * 2
    assert clip_options == [(5., {"error_if_nonfinite": True})] * 4
    assert data.source.preprocessor.export_metadata() == before_metadata
    assert data.source.preprocessor.feature_names == before_names
    for (name, field), value in before_arrays.items():
        np.testing.assert_array_equal(getattr(getattr(data, name), field), value)


@pytest.mark.parametrize("change", ["architecture", "solver_step", "workflow", "input_kind", "training_seed", "epochs", "clip"])
def test_restoration_rejects_wrong_checkpoint_metadata_before_consuming_validation(data, tmp_path, monkeypatch, change):
    checkpoint = {"model_metadata": forecasting.model_spec(), "input_kind": "temporal",
                  "training_config": asdict(forecasting.smoke_config()), "model_state": {}}
    if change == "architecture":
        checkpoint["model_metadata"]["kwargs"]["hidden_width"] = 8
    elif change == "solver_step":
        checkpoint["model_metadata"]["kwargs"]["step_size"] = .125
    elif change == "workflow":
        checkpoint["model_metadata"]["workflow"] = "some-other-task"
    elif change == "input_kind":
        checkpoint["input_kind"] = "tabular"
    else:
        key, value = {"training_seed": ("seed", 43), "epochs": ("max_epochs", 50),
                      "clip": ("gradient_clip", None)}[change]
        checkpoint["training_config"][key] = value
    path = tmp_path / "wrong.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr(forecasting, "build_model", lambda: pytest.fail("Reject metadata before inference."))
    with pytest.raises(ValueError, match="fixed two-epoch"):
        forecasting.restore_validation(path, data)


def test_runner_orders_separate_sanity_then_smoke_and_records_only_train_validation(data, tmp_path, monkeypatch):
    phases = []
    original = data.source
    monkeypatch.setattr(forecasting, "load_baseline_data", lambda path: original)

    def sanity(train, output):
        phases.append("sanity")
        np.testing.assert_array_equal(train.sample_indices, data.train.sample_indices)
        assert not set(train.participants) & set(data.validation.participants)
        return {"passed": True, "steps_run": 1, "final_rmse": .1, "warm_start_smoke": False}

    def smoke(frozen, train, output):
        phases.append("smoke")
        assert phases == ["sanity", "smoke"]
        np.testing.assert_array_equal(frozen.validation.sample_indices, data.validation.sample_indices)
        np.testing.assert_array_equal(train.sample_indices, data.train.sample_indices)
        return {"validation": {"mae": 1., "rmse": 1.2, "r2": .1}, "training_seconds": .1}

    monkeypatch.setattr(forecasting, "run_overfit_sanity", sanity)
    monkeypatch.setattr(forecasting, "train_smoke", smoke)
    monkeypatch.setattr(forecasting, "_plot", lambda output: None)
    output = tmp_path / "mocked_boundary"
    result = forecasting.run_smoke(original.dataset_dir, output)
    assert phases == ["sanity", "smoke"]
    assert result["counts"] == {
        "saved_training": {"samples": 24, "participants": 8},
        "smoke_training": {"samples": 24, "participants": 8},
        "validation": {"samples": 8, "participants": 4},
    }
    config = json.loads((output / "config.json").read_text())
    assert config["calibration_or_test_payloads_consumed"] is False
    assert config["raw_dataset_reopened"] is False and config["preprocessing_refit"] is False
    assert config["hyperparameter_search"] is False and config["full_three_seed_training"] is False
    provenance = json.loads((output / "run.json").read_text())
    assert provenance["dataset_files_consumed"] == list(original.read_files)
    assert not any("calibration" in name or "test_" in name for name in provenance["dataset_files_consumed"])
    with np.load(output / "used_split_indices.npz", allow_pickle=False) as selected:
        np.testing.assert_array_equal(selected["diagnostic_train"], data.train.sample_indices[:8])
        np.testing.assert_array_equal(selected["smoke_train"], data.train.sample_indices)
        np.testing.assert_array_equal(selected["validation"], data.validation.sample_indices)


def test_runner_records_sanity_failure_and_does_not_start_smoke(data, tmp_path, monkeypatch):
    monkeypatch.setattr(forecasting, "load_baseline_data", lambda path: data.source)

    def fail_sanity(train, output):
        raise FloatingPointError("synthetic nonfinite dynamics")

    monkeypatch.setattr(forecasting, "run_overfit_sanity", fail_sanity)
    monkeypatch.setattr(forecasting, "train_smoke", lambda *args: pytest.fail("Failed sanity must stop smoke."))
    output = tmp_path / "failed_boundary"
    with pytest.raises(FloatingPointError, match="nonfinite dynamics"):
        forecasting.run_smoke(data.source.dataset_dir, output)
    failure = json.loads((output / "failure.json").read_text())
    assert failure == {"phase": "training-only overfit diagnostic", "type": "FloatingPointError",
                       "message": "synthetic nonfinite dynamics", "silent_solver_changes": False}
