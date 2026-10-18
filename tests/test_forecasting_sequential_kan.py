"""Synthetic checks for the conditional KAN-encoded recurrent baseline.

No uploaded observations or reserved partition payloads are read. Short fits
exercise reproducibility and the experiment boundary; causal-prefix checks
exercise the actual recurrent computation rather than a performance claim.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch
from torch import nn

from kan_phenotyping.experiments.forecasting_baselines import (
    BaselinePartition,
    SavedBaselineData,
    regression_metrics,
)
from kan_phenotyping.experiments import forecasting_sequential_kan as forecasting
from kan_phenotyping.models.kan_layer import KANLinear
from kan_phenotyping.models.sequential_kan import KANEncodedRecurrentBaseline
from kan_phenotyping.training.trainer import seed_everything


class FrozenPreprocessor:
    """Expose saved metadata while rejecting any preprocessing operation."""

    def __init__(self, n_training_samples):
        wearable = [f"numeric__wearable_{index}" for index in range(27)]
        wearable += [f"numeric__missingindicator_wearable_{index}" for index in range(21)]
        demographics = [f"demographic__feature_{index}" for index in range(62)]
        self.feature_names = {
            "wearable": wearable,
            "demographics": demographics,
            "sm0_phq": ["measured_sm0_phq9_standardized"],
            "tabular": [f"SM{month}__{name}" for month in (1, 2) for name in wearable]
            + [f"baseline_demographic__{name}" for name in demographics]
            + ["measured_sm0_phq9_standardized"],
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
        raise AssertionError("Reuse saved training-fitted preprocessing without fitting.")

    def transform(self, *args, **kwargs):
        raise AssertionError("Reuse saved transformed arrays without transforming.")


@pytest.fixture
def saved_data(tmp_path):
    rng = np.random.default_rng(74)

    def partition(n, offset, prefix):
        # The saved bundle is float32, and its baseline loader promotes those
        # exact values to float64. Match that lossless round-trip here.
        features = rng.normal(0, .4, (n, 159)).astype(np.float32).astype(np.float64)
        measured_sm0 = ((np.arange(n) * 5 + 3) % 28).astype(np.float64)
        features[:, -1] = (measured_sm0 - 13.5) / 8
        target = (2 + np.arange(n) % 4).astype(np.float64)
        # Make the sanity problem modest while preserving nonconstant measured
        # integer outcomes and distinct observations at the two real positions.
        features[:, 96] = (target - 3.5) / 2
        return BaselinePartition(
            sample_indices=rng.permutation(np.arange(offset, offset + n)),
            participants=np.array([f"{prefix}{index // 2}" for index in range(n)]),
            sm0_phq=measured_sm0,
            tabular=features,
            targets=target,
        )

    train = partition(48, 1000, "training-person-")
    validation = partition(16, 2000, "validation-person-")
    preprocessor = FrozenPreprocessor(len(train.targets))
    root = tmp_path / "saved_dataset"
    root.mkdir()
    return SavedBaselineData(
        train=train, validation=validation, preprocessor=preprocessor,
        manifest={"preprocessing": preprocessor.export_metadata()},
        dataset_dir=root,
        read_files=("manifest.json", "train_inputs.npz", "train_targets.npz",
                    "validation_inputs.npz", "validation_targets.npz"),
        manifest_sha256="synthetic-fixture-not-a-real-input-hash",
    )


@pytest.fixture
def data(saved_data):
    return forecasting.prepare_sequential_data(saved_data)


def short_config(seed=42, epochs=2):
    return replace(forecasting.experiment_config(seed), max_epochs=epochs,
                   patience=epochs, batch_size=16, num_threads=1)


def test_architecture_is_three_spline_encoders_and_one_shared_gru():
    model = KANEncodedRecurrentBaseline()
    spline_layers = [layer for layer in model.modules() if isinstance(layer, KANLinear)]
    assert spline_layers == [model.static_encoder, model.wearable_encoder, model.head]
    assert [(layer.in_features, layer.out_features) for layer in spline_layers] == [
        (63, 16), (48, 16), (16, 1),
    ]
    assert all(layer.grid_intervals == 8 and layer.spline_degree == 3 for layer in spline_layers)
    assert all(layer.grid_range == (-3., 3.) for layer in spline_layers)
    assert isinstance(model.gru, nn.GRUCell)
    assert model.gru.input_size == model.gru.hidden_size == 16
    assert len([layer for layer in model.modules() if isinstance(layer, nn.GRUCell)]) == 1
    assert sum(parameter.numel() for parameter in model.parameters()) == 23169
    assert not any(isinstance(layer, (nn.Sigmoid, nn.Softmax)) for layer in model.modules())


def test_shared_encoder_and_gru_consume_sm1_then_sm2_and_static_once():
    seed_everything(42, num_threads=1)
    model = KANEncodedRecurrentBaseline()
    wearable = torch.randn(5, 2, 48)
    static = torch.randn(5, 63)
    packet_calls, static_calls, gru_calls = [], [], []
    handles = [
        model.wearable_encoder.register_forward_pre_hook(
            lambda module, args: packet_calls.append((id(module), args[0].detach().clone()))),
        model.static_encoder.register_forward_pre_hook(
            lambda module, args: static_calls.append(args[0].detach().clone())),
        model.gru.register_forward_pre_hook(
            lambda module, args: gru_calls.append((id(module), args[0].detach().clone(),
                                                   args[1].detach().clone()))),
    ]
    try:
        states = model.trace_states(wearable, static)
    finally:
        for handle in handles:
            handle.remove()
    assert states.shape == (5, 3, 16)
    assert len(static_calls) == 1
    assert len(packet_calls) == len(gru_calls) == 2
    assert [item[0] for item in packet_calls] == [id(model.wearable_encoder)] * 2
    assert [item[0] for item in gru_calls] == [id(model.gru)] * 2
    torch.testing.assert_close(static_calls[0], static, rtol=0, atol=0)
    for position in range(2):
        torch.testing.assert_close(packet_calls[position][1], wearable[:, position], rtol=0, atol=0)
        torch.testing.assert_close(gru_calls[position][2], states[:, position], rtol=0, atol=0)
    # Independently reconstruct the equations, including the initial state.
    h0 = model.static_encoder(static)
    h1 = model.gru(model.wearable_encoder(wearable[:, 0]), h0)
    h2 = model.gru(model.wearable_encoder(wearable[:, 1]), h1)
    torch.testing.assert_close(states, torch.stack([h0, h1, h2], dim=1), rtol=0, atol=0)
    torch.testing.assert_close(model(wearable, static), model.head(h2).squeeze(-1), rtol=0, atol=0)


def test_sm1_state_has_no_computational_access_to_sm2():
    seed_everything(42, num_threads=1)
    model = KANEncodedRecurrentBaseline()
    wearable = torch.randn(4, 2, 48, requires_grad=True)
    static = torch.randn(4, 63, requires_grad=True)
    states = model.trace_states(wearable, static)
    altered = wearable.detach().clone()
    altered[:, 1] = torch.linspace(-100., 100., 48)
    altered_states = model.trace_states(altered, static.detach())
    torch.testing.assert_close(states[:, :2], altered_states[:, :2], rtol=0, atol=0)
    packet_gradient, static_gradient = torch.autograd.grad(states[:, 1].square().sum(), (wearable, static))
    assert torch.count_nonzero(packet_gradient[:, 1]).item() == 0
    assert torch.count_nonzero(packet_gradient[:, 0]).item() > 0
    assert torch.count_nonzero(static_gradient).item() > 0
    assert torch.isfinite(packet_gradient).all() and torch.isfinite(static_gradient).all()


def test_repeated_calls_do_not_retain_hidden_state_between_samples():
    seed_everything(42, num_threads=1)
    model = KANEncodedRecurrentBaseline()
    wearable, static = torch.randn(6, 2, 48), torch.randn(6, 63)
    first = model(wearable, static)
    model(torch.randn(3, 2, 48), torch.randn(3, 63))
    repeated = model(wearable, static)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    reverse = torch.arange(5, -1, -1)
    reordered = model(wearable[reverse], static[reverse])
    torch.testing.assert_close(reordered, first[reverse], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("value", [-8., 34., 4.125])
def test_output_is_raw_unbounded_scalar(value):
    model = KANEncodedRecurrentBaseline()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias.fill_(value)
    prediction = model(torch.zeros(5, 2, 48), torch.zeros(5, 63))
    assert prediction.shape == (5,)
    torch.testing.assert_close(prediction, torch.full((5,), value), rtol=0, atol=0)


def test_forward_backward_gradients_are_finite_and_nonzero_for_every_parameter(data):
    seed_everything(42, num_threads=1)
    model = KANEncodedRecurrentBaseline()
    wearable = torch.tensor(data.train.wearable[:8], dtype=torch.float32)
    static = torch.tensor(data.train.static[:8], dtype=torch.float32)
    targets = torch.tensor(data.train.targets[:8], dtype=torch.float32)
    prediction = model(wearable, static)
    assert prediction.shape == targets.shape == (8,)
    loss = nn.MSELoss()(prediction, targets)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad).item() > 0, name


@pytest.mark.parametrize("wearable,static", [
    (torch.zeros(3, 1, 48), torch.zeros(3, 63)),
    (torch.zeros(3, 3, 48), torch.zeros(3, 63)),
    (torch.zeros(3, 2, 47), torch.zeros(3, 63)),
    (torch.zeros(3, 2, 48), torch.zeros(3, 62)),
    (torch.zeros(3, 2, 48), torch.zeros(4, 63)),
    (torch.full((3, 2, 48), float("nan")), torch.zeros(3, 63)),
    (torch.zeros(3, 2, 48), torch.full((3, 63), float("inf"))),
])
def test_model_rejects_extra_packets_wrong_shapes_and_nonfinite_inputs(wearable, static):
    with pytest.raises(ValueError):
        KANEncodedRecurrentBaseline()(wearable, static)


def test_fixed_training_settings_differ_only_in_initialization_seed():
    comparable = []
    for seed in (42, 43, 44):
        configuration = forecasting.experiment_config(seed)
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
        options = asdict(configuration)
        options.pop("seed")
        comparable.append(options)
    assert comparable[0] == comparable[1] == comparable[2]


def test_prepare_sequence_preserves_all_transformed_columns_month_identity_and_alignment(saved_data, data):
    assert data.source is saved_data
    for partition_name in ("train", "validation"):
        original, prepared = getattr(saved_data, partition_name), getattr(data, partition_name)
        n = len(original.targets)
        assert prepared.wearable.shape == (n, 2, 48)
        assert prepared.static.shape == (n, 63)
        np.testing.assert_array_equal(prepared.wearable[:, 0], original.tabular[:, :48])
        np.testing.assert_array_equal(prepared.wearable[:, 1], original.tabular[:, 48:96])
        np.testing.assert_array_equal(prepared.static, original.tabular[:, 96:])
        np.testing.assert_array_equal(np.concatenate([prepared.wearable.reshape(n, -1), prepared.static], axis=1),
                                      original.tabular)
        for field in ("sample_indices", "participants", "targets"):
            np.testing.assert_array_equal(getattr(prepared, field), getattr(original, field))


@pytest.mark.parametrize("tampering", ["reversed_names", "wrong_month", "wrong_width", "wrong_static"])
def test_preparation_rejects_mislabeled_or_changed_frozen_features(saved_data, tampering):
    saved = deepcopy(saved_data)
    if tampering == "reversed_names":
        saved.preprocessor.feature_names["tabular"][:48] = list(reversed(saved.preprocessor.feature_names["tabular"][:48]))
    elif tampering == "wrong_month":
        saved.preprocessor.feature_names["tabular"][0] = "SM2__unexpected_feature"
    elif tampering == "wrong_width":
        saved.preprocessor.feature_names["wearable"].pop()
    else:
        saved.preprocessor.feature_names["demographics"].pop()
    with pytest.raises(ValueError):
        forecasting.prepare_sequential_data(saved)


def test_training_rejects_preprocessing_metadata_changed_after_loading(data, tmp_path):
    data.source.preprocessor.metadata["training_samples"] += 1
    with pytest.raises(ValueError, match="preprocessing"):
        forecasting.train_seed(data, tmp_path / "changed_metadata", 42, config=short_config(epochs=1))


def test_initialization_seeds_use_identical_minibatch_orders(data):
    orders = []
    for seed in (42, 43, 44):
        loader = forecasting.make_sequential_loader(data.train, short_config(seed), shuffle=True)
        # A continuous saved predictor identifies each synthetic observation;
        # the repeated integer outcomes cannot establish exact row order.
        orders.append(torch.cat([batch["temporal"][:, 0, 0] for batch in loader]).numpy())
    np.testing.assert_array_equal(orders[0], orders[1])
    np.testing.assert_array_equal(orders[0], orders[2])


def test_unshuffled_loader_keeps_sequence_static_context_and_targets_aligned(data):
    batches = list(forecasting.make_sequential_loader(data.validation, short_config(), shuffle=False))
    for key, expected in (("temporal", data.validation.wearable),
                          ("static", data.validation.static),
                          ("target", data.validation.targets)):
        values = torch.cat([batch[key] for batch in batches]).numpy()
        np.testing.assert_array_equal(values, expected.astype(np.float32))


def test_restoration_reproduces_aligned_predictions_and_original_scale_metrics(data, tmp_path):
    output = tmp_path / "one_seed"
    report = forecasting.train_seed(data, output, 42, config=short_config())
    assert report["seed"] == 42 and report["parameter_count"] == 23169
    assert report["restored_predictions_match"] is True
    assert report["preprocessing_unchanged"] is True
    assert 1 <= report["best_epoch"] <= report["epochs_run"] <= 2
    assert report["training_seconds"] >= 0
    gradients = report["gradient_checks"]
    assert gradients["all_observed_gradients_finite"] is True
    assert len(gradients["backward_calls_by_parameter"]) == 13
    assert all(count > 0 for count in gradients["backward_calls_by_parameter"].values())
    assert all(gradients["nonzero_gradient_seen_by_parameter"].values())
    prefix = report["prefix_checks"]
    assert prefix["events"] == ["initialize_static", "encode_SM1", "update_SM1", "encode_SM2", "update_SM2"]
    for key in ("wearable_encoder_shared", "gru_shared", "sm1_state_unchanged_when_sm2_changes",
                "prefix_only_api_matches_first_update", "sm1_gradient_wrt_sm2_exactly_zero",
                "sm1_gradient_wrt_sm1_nonzero", "sm1_gradient_wrt_static_nonzero"):
        assert prefix[key] is True
    assert prefix["reversal_performance_requirement"] is False
    assert (output / "history.json").is_file()
    assert (output / "training_config.json").is_file()
    assert json.loads((output / "validation_metrics.json").read_text())["validation"] == report["validation"]
    restored_metrics, restored_prediction = forecasting.restore_validation(output / "best.pt", data)
    assert restored_metrics == report["validation"]
    with np.load(output / "validation_predictions.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["sample_indices"], data.validation.sample_indices)
        np.testing.assert_array_equal(archive["measured_sm3"], data.validation.targets)
        np.testing.assert_array_equal(archive["prediction"], restored_prediction)
        independent_metrics = regression_metrics(archive["measured_sm3"], archive["prediction"])
        for name, value in independent_metrics.items():
            assert restored_metrics[name] == pytest.approx(value)
        assert restored_metrics["n"] == len(data.validation.targets)


def test_same_seed_reproduces_checkpoint_weights_and_validation_predictions(data, tmp_path):
    states, predictions, reports = [], [], []
    for run in (1, 2):
        output = tmp_path / f"repeat_{run}"
        reports.append(forecasting.train_seed(data, output, 43, config=short_config(43)))
        states.append(torch.load(output / "best.pt", map_location="cpu", weights_only=True)["model_state"])
        with np.load(output / "validation_predictions.npz", allow_pickle=False) as archive:
            predictions.append(archive["prediction"].copy())
    assert reports[0]["validation"] == reports[1]["validation"]
    assert reports[0]["best_epoch"] == reports[1]["best_epoch"]
    np.testing.assert_array_equal(*predictions)
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)


def test_validation_outcomes_do_not_affect_one_epoch_training_weights(data, tmp_path):
    altered = replace(data, validation=replace(data.validation, targets=27 - data.validation.targets))
    states, metrics = [], []
    for index, saved in enumerate((data, altered)):
        output = tmp_path / f"validation_counterfactual_{index}"
        report = forecasting.train_seed(saved, output, 42, config=short_config(epochs=1))
        states.append(torch.load(output / "best.pt", map_location="cpu", weights_only=True)["model_state"])
        metrics.append(report["validation"])
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
    assert metrics[0] != metrics[1]


def test_training_reuses_preprocessing_and_preserves_saved_arrays(data, tmp_path, monkeypatch):
    before_metadata = data.source.preprocessor.export_metadata()
    before_names = deepcopy(data.source.preprocessor.feature_names)
    before_arrays = {(partition, field): np.array(getattr(getattr(data, partition), field), copy=True)
                     for partition in ("train", "validation")
                     for field in ("wearable", "static", "targets", "sample_indices", "participants")}
    options = []
    adamw = torch.optim.AdamW

    def record_adamw(parameters, **kwargs):
        options.append(kwargs.copy())
        return adamw(parameters, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", record_adamw)
    forecasting.train_seed(data, tmp_path / "frozen_preprocessing", 44,
                           config=short_config(44, epochs=1))
    assert options == [{"lr": .001, "weight_decay": .0001}]
    assert data.source.preprocessor.export_metadata() == before_metadata
    assert data.source.preprocessor.feature_names == before_names
    for (partition, field), original in before_arrays.items():
        np.testing.assert_array_equal(getattr(getattr(data, partition), field), original)


@pytest.mark.parametrize("field,change", [
    ("wearable", lambda values: values[:, ::-1, :47]),
    ("wearable", lambda values: values[:, :1]),
    ("wearable", lambda values: np.full_like(values, np.nan)),
    ("static", lambda values: values[:, :-1]),
    ("static", lambda values: np.full_like(values, np.inf)),
    ("targets", lambda values: values[:-1]),
    ("targets", lambda values: values[:, None]),
    ("targets", lambda values: np.full_like(values, np.nan)),
    ("sample_indices", lambda values: values[:-1]),
    ("participants", lambda values: values[:-1]),
])
def test_tensor_dataset_rejects_wrong_shapes_or_misalignment(data, field, change):
    invalid = replace(data.train, **{field: change(getattr(data.train, field))})
    with pytest.raises(ValueError):
        forecasting.SequentialTensorDataset(invalid)


def test_training_only_overfit_sanity_is_separate_and_reduces_error(data, tmp_path):
    output = tmp_path / "training_only_sanity"
    result = forecasting.run_overfit_sanity(data.train, output)
    assert result["training_only"] is True and result["passed"] is True
    assert result["samples"] == 32
    assert result["final_rmse"] <= .25
    assert result["final_mse"] <= .01 * result["initial_mse"]
    assert result["warm_start_experiment"] is False
    assert (output / "sanity.json").is_file()
    assert not (output / "validation_predictions.npz").exists()
