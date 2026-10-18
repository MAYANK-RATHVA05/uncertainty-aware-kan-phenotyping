"""Synthetic checks for the conditional flattened-history forecasting KAN.

No uploaded observations, reserved partition arrays, or baseline artifacts are
opened here. Small one/two-epoch fits test the experiment plumbing separately
from the fixed three-seed research run.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from kan_phenotyping.experiments.forecasting_baselines import (
    BaselinePartition,
    SavedBaselineData,
    regression_metrics,
)
from kan_phenotyping.experiments import forecasting_kan as forecasting
from kan_phenotyping.models.depression_kan import TabularDepressionKAN
from kan_phenotyping.models.kan_layer import KANLinear
from kan_phenotyping.training.trainer import seed_everything


class AlreadyFittedPreprocessor:
    """Detect any attempted preprocessing fit/transform during model training."""

    def __init__(self, n_training_samples: int):
        self.feature_names = {
            "tabular": [f"SM{month}__wearable_{feature}" for month in (1, 2)
                        for feature in range(54)]
            + [f"baseline_demographic__feature_{feature}" for feature in range(50)]
            + ["measured_sm0_phq"],
        }
        self.metadata = {
            "fitted_partition": "training_only",
            "training_samples": n_training_samples,
            "validation_used_for_fit": False,
            "calibration_used_for_fit": False,
            "test_used_for_fit": False,
            "targets_used_for_fit": False,
        }

    def export_metadata(self):
        return deepcopy(self.metadata)

    def fit(self, *args, **kwargs):
        raise AssertionError("Forecasting KAN must reuse the saved preprocessing fit.")

    def transform(self, *args, **kwargs):
        raise AssertionError("Forecasting KAN must consume the saved transformed inputs.")


@pytest.fixture
def synthetic_data(tmp_path):
    generator = np.random.default_rng(71)

    def partition(n, first_index, prefix):
        measured_sm0 = ((np.arange(n) * 5 + 3) % 28).astype(np.float64)
        features = generator.normal(0, .5, (n, 159)).astype(np.float64)
        features[:, -1] = (measured_sm0 - 13.5) / 8
        measured_sm3 = ((np.arange(n) * 3 + 5) % 28).astype(np.float64)
        return BaselinePartition(
            sample_indices=generator.permutation(np.arange(first_index, first_index + n)),
            participants=np.array([f"{prefix}{index // 2}" for index in range(n)]),
            sm0_phq=measured_sm0,
            tabular=features,
            targets=measured_sm3,
        )

    training = partition(48, 1000, "training-person-")
    validation = partition(16, 2000, "validation-person-")
    preprocessor = AlreadyFittedPreprocessor(len(training.targets))
    root = tmp_path / "synthetic_saved_dataset"
    root.mkdir()
    return SavedBaselineData(
        train=training,
        validation=validation,
        preprocessor=preprocessor,
        manifest={"preprocessing": preprocessor.export_metadata()},
        dataset_dir=root,
        read_files=("manifest.json", "train_inputs.npz", "train_targets.npz",
                    "validation_inputs.npz", "validation_targets.npz"),
        manifest_sha256="synthetic-provenance-not-a-real-input-hash",
    )


def short_config(seed=42, epochs=2):
    return replace(forecasting.experiment_config(seed), max_epochs=epochs, patience=epochs,
                   batch_size=16, num_threads=1)


def test_architecture_reuses_spline_kan_with_exact_requested_widths():
    spec = forecasting.model_spec()
    model = forecasting.build_forecasting_model()
    assert isinstance(model, TabularDepressionKAN)
    assert spec["input_kind"] == "tabular"
    assert spec["kwargs"]["n_features"] == 159
    assert spec["kwargs"]["hidden_width"] == 64
    assert spec["kwargs"]["task"] == "regression"
    layers = [layer for layer in model.modules() if isinstance(layer, KANLinear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [(159, 64), (64, 1)]
    assert all(layer.grid_intervals == 8 and layer.spline_degree == 3 for layer in layers)
    assert all(layer.grid_range == (-3.0, 3.0) for layer in layers)
    assert sum(parameter.numel() for parameter in model.parameters()) == 122945
    assert not any(isinstance(layer, (nn.Sigmoid, nn.Softmax)) for layer in model.modules())


@pytest.mark.parametrize("value", [-7.0, 34.0, 4.125])
def test_output_is_raw_unbounded_and_unrounded(value):
    model = forecasting.build_forecasting_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.layers[-1].bias.fill_(value)
    predictions = model(torch.zeros(5, 159))
    assert predictions.shape == (5,)
    torch.testing.assert_close(predictions, torch.full((5,), value), rtol=0, atol=0)


def test_experiment_settings_are_fixed_except_initialization_seed():
    configurations = [forecasting.experiment_config(seed) for seed in (42, 43, 44)]
    comparable = []
    for seed, configuration in zip((42, 43, 44), configurations, strict=True):
        assert configuration.seed == seed
        assert configuration.task == "regression"
        assert configuration.regression_loss == "mse"
        assert configuration.max_epochs == 50
        assert configuration.patience == 10
        assert configuration.batch_size == 128
        assert configuration.learning_rate == .001
        assert configuration.weight_decay == .0001
        assert configuration.gradient_clip is None
        assert configuration.min_delta == 0
        assert configuration.device == "cpu"
        configuration_without_seed = asdict(configuration)
        configuration_without_seed.pop("seed")
        comparable.append(configuration_without_seed)
    assert comparable[0] == comparable[1] == comparable[2]


def test_forward_backward_has_correct_shapes_and_finite_nonzero_gradients(synthetic_data):
    seed_everything(42, num_threads=1)
    model = forecasting.build_forecasting_model()
    inputs = torch.as_tensor(synthetic_data.train.tabular[:8], dtype=torch.float32)
    targets = torch.as_tensor(synthetic_data.train.targets[:8], dtype=torch.float32)
    predictions = model(inputs)
    assert predictions.shape == targets.shape == (8,)
    loss = nn.MSELoss()(predictions, targets)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_backward_gradient_checks_reject_nonfinite_values():
    model = forecasting.build_forecasting_model()
    checks = forecasting.FiniteGradientChecks(model)
    try:
        nonfinite_loss = model(torch.zeros(2, 159)).sum() * float("inf")
        with pytest.raises(FloatingPointError, match="gradient"):
            nonfinite_loss.backward()
    finally:
        checks.close()


def test_initialization_seeds_share_identical_batch_order(synthetic_data):
    orders = []
    for seed in (42, 43, 44):
        loader = forecasting.make_forecast_loader(synthetic_data.train, short_config(seed), shuffle=True)
        # Outcomes identify rows independently of the feature matrix; the loader
        # generator must use the fixed data-order seed rather than model seed.
        orders.append(torch.cat([batch["target"] for batch in loader]).numpy())
    np.testing.assert_array_equal(orders[0], orders[1])
    np.testing.assert_array_equal(orders[0], orders[2])


def test_checkpoint_restoration_and_saved_validation_alignment(synthetic_data, tmp_path):
    output = tmp_path / "one_seed"
    report = forecasting.train_seed(synthetic_data, output, 42, config=short_config())
    assert report["seed"] == 42
    assert report["parameter_count"] == 122945
    assert report["restored_predictions_match"] is True
    assert 1 <= report["best_epoch"] <= report["epochs_run"] <= 2
    assert report["training_seconds"] >= 0
    gradients = report["gradient_checks"]
    assert gradients["all_observed_gradients_finite"] is True
    assert len(gradients["backward_calls_by_parameter"]) == 6
    assert all(count > 0 for count in gradients["backward_calls_by_parameter"].values())
    assert all(gradients["nonzero_gradient_seen_by_parameter"].values())
    assert (output / "history.json").is_file()
    assert (output / "training_config.json").is_file()
    assert json.loads((output / "validation_metrics.json").read_text())["validation"] == report["validation"]
    restored_metrics, restored_predictions = forecasting.restore_validation(output / "best.pt", synthetic_data)
    assert restored_metrics == report["validation"]
    with np.load(output / "validation_predictions.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["sample_indices"], synthetic_data.validation.sample_indices)
        np.testing.assert_array_equal(archive["measured_sm3"], synthetic_data.validation.targets)
        np.testing.assert_array_equal(archive["prediction"], restored_predictions)
        assert archive["prediction"].shape == synthetic_data.validation.targets.shape
        independent_metrics = regression_metrics(archive["measured_sm3"], archive["prediction"])
        for name, value in independent_metrics.items():
            assert restored_metrics[name] == pytest.approx(value)
        assert restored_metrics["n"] == len(synthetic_data.validation.targets)


def test_same_seed_inputs_and_configuration_reproduce_weights_and_predictions(synthetic_data, tmp_path):
    reports = []
    states = []
    predictions = []
    for run in (1, 2):
        output = tmp_path / f"repeated_run_{run}"
        reports.append(forecasting.train_seed(synthetic_data, output, 43, config=short_config(43)))
        states.append(torch.load(output / "best.pt", map_location="cpu", weights_only=True)["model_state"])
        with np.load(output / "validation_predictions.npz", allow_pickle=False) as archive:
            predictions.append(archive["prediction"].copy())
    assert reports[0]["validation"] == reports[1]["validation"]
    assert reports[0]["best_epoch"] == reports[1]["best_epoch"]
    np.testing.assert_array_equal(*predictions)
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)


def test_validation_targets_cannot_affect_one_epoch_training_weights(synthetic_data, tmp_path):
    altered = replace(synthetic_data, validation=replace(
        synthetic_data.validation, targets=27 - synthetic_data.validation.targets,
    ))
    states = []
    metrics = []
    for number, data in enumerate((synthetic_data, altered)):
        output = tmp_path / f"validation_counterfactual_{number}"
        report = forecasting.train_seed(data, output, 42, config=short_config(epochs=1))
        states.append(torch.load(output / "best.pt", map_location="cpu", weights_only=True)["model_state"])
        metrics.append(report["validation"])
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
    assert metrics[0] != metrics[1]


def test_training_reuses_preprocessing_and_does_not_mutate_saved_arrays(
    synthetic_data, tmp_path, monkeypatch
):
    before_metadata = synthetic_data.preprocessor.export_metadata()
    before_features = deepcopy(synthetic_data.preprocessor.feature_names)
    original_arrays = {
        (name, field): np.array(getattr(getattr(synthetic_data, name), field), copy=True)
        for name in ("train", "validation")
        for field in ("tabular", "targets", "sample_indices", "sm0_phq", "participants")
    }
    optimizer_options = []
    original_adamw = torch.optim.AdamW

    def record_adamw(parameters, **options):
        optimizer_options.append(options.copy())
        return original_adamw(parameters, **options)

    monkeypatch.setattr(torch.optim, "AdamW", record_adamw)
    forecasting.train_seed(synthetic_data, tmp_path / "reuse_fit", 44,
                           config=short_config(44, epochs=1))
    assert optimizer_options == [{"lr": .001, "weight_decay": .0001}]
    assert synthetic_data.preprocessor.export_metadata() == before_metadata
    assert synthetic_data.preprocessor.feature_names == before_features
    for (name, field), before in original_arrays.items():
        np.testing.assert_array_equal(getattr(getattr(synthetic_data, name), field), before)


@pytest.mark.parametrize("field,change", [
    ("tabular", lambda values: values[:, :-1]),
    ("tabular", lambda values: np.full_like(values, np.nan)),
    ("tabular", lambda values: np.full_like(values, np.inf)),
    ("targets", lambda values: values[:-1]),
    ("targets", lambda values: values[:, None]),
    ("targets", lambda values: np.full_like(values, np.nan)),
    ("sample_indices", lambda values: values[:-1]),
    ("participants", lambda values: values[:-1]),
])
def test_tensor_dataset_rejects_misaligned_or_nonfinite_inputs(synthetic_data, field, change):
    invalid = replace(synthetic_data.train, **{field: change(getattr(synthetic_data.train, field))})
    with pytest.raises(ValueError):
        forecasting.ForecastTensorDataset(invalid)


def test_seed_aggregation_uses_sample_sd_rather_than_population_sd():
    records = [
        {"seed": seed, "validation": {"mae": value, "rmse": value + 1, "r2": value / 10}}
        for seed, value in zip((42, 43, 44), (2.0, 3.0, 6.0), strict=True)
    ]
    summary = forecasting.aggregate_seed_metrics(records)
    for key in ("mae", "rmse", "r2"):
        values = np.array([record["validation"][key] for record in records])
        assert summary[key]["mean"] == pytest.approx(values.mean())
        assert summary[key]["sample_sd"] == pytest.approx(values.std(ddof=1))
        assert summary[key]["sample_sd"] != pytest.approx(values.std(ddof=0))


def test_sample_sd_requires_more_than_one_initialization():
    with pytest.raises(ValueError, match="at least two"):
        forecasting.aggregate_seed_metrics([
            {"seed": 42, "validation": {"mae": 2., "rmse": 3., "r2": .4}},
        ])


def test_training_only_overfit_sanity_is_separate_and_reduces_error(synthetic_data, tmp_path):
    # The sanity API accepts only a training partition: no validation object is
    # passed, so a successful check cannot inspect validation/reserved outcomes.
    output = tmp_path / "training_only_sanity"
    report = forecasting.run_overfit_sanity(synthetic_data.train, output)
    assert report["passed"] is True
    assert report["samples"] == 32
    assert report["final_rmse"] <= .25
    assert report["final_mse"] <= .01 * report["initial_mse"]
    assert (output / "sanity.json").is_file()
    assert not (output / "validation_predictions.npz").exists()
