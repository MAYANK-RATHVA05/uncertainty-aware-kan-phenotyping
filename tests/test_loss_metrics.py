"""Mathematical checks and invalid-input tests for public loss/metric APIs."""

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from kan_phenotyping.evaluation.metrics import (
    binary_classification_metrics,
    regression_metrics,
)
from kan_phenotyping.training.focal_loss import BinaryFocalLoss


def test_focal_loss_known_values_and_reductions():
    logits = torch.zeros(2, dtype=torch.float64)
    labels = torch.tensor([0.0, 1.0], dtype=torch.float64)
    expected = torch.tensor([0.75, 0.25], dtype=torch.float64) * 0.25 * np.log(2)
    assert torch.allclose(BinaryFocalLoss(reduction="none")(logits, labels), expected)
    assert BinaryFocalLoss()(logits, labels).item() == pytest.approx(expected.mean().item())
    assert BinaryFocalLoss(reduction="sum")(logits, labels).item() == pytest.approx(expected.sum().item())


def test_gamma_zero_matches_weighted_bce():
    logits = torch.tensor([-2.0, 0.1, 0.8, 4.0])
    labels = torch.tensor([0, 1, 0, 1])
    expected = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    expected *= torch.where(labels == 1, 0.25, 0.75)
    assert torch.allclose(BinaryFocalLoss(gamma=0, reduction="none")(logits, labels), expected)


def test_focal_loss_extreme_logits_have_finite_loss_and_gradients():
    logits = torch.tensor([-10000.0, 10000.0, -10000.0, 10000.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    loss = BinaryFocalLoss()(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert loss.item() == pytest.approx(2500.0)


@pytest.mark.parametrize("target", [0.2, -1.0, 2.0, float("nan"), float("inf")])
def test_focal_loss_rejects_invalid_labels(target):
    with pytest.raises(ValueError):
        BinaryFocalLoss()(torch.tensor([0.0]), torch.tensor([target]))


@pytest.mark.parametrize("logit", [float("nan"), float("inf"), -float("inf")])
def test_focal_loss_rejects_nonfinite_logits(logit):
    with pytest.raises(ValueError):
        BinaryFocalLoss()(torch.tensor([logit]), torch.tensor([0.0]))


@pytest.mark.parametrize("kwargs", [{"alpha": -0.1}, {"alpha": 1.1}, {"gamma": -1}, {"gamma": float("inf")}, {"reduction": "median"}])
def test_focal_loss_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BinaryFocalLoss(**kwargs)


def test_focal_loss_rejects_shape_mismatch_and_empty_tensors():
    with pytest.raises(ValueError):
        BinaryFocalLoss()(torch.zeros(2, 1), torch.zeros(2))
    with pytest.raises(ValueError):
        BinaryFocalLoss()(torch.empty(0), torch.empty(0))


def test_regression_metrics_known_values():
    result = regression_metrics([1, 2, 3], [2, 2, 2])
    assert result == {"mae": pytest.approx(2 / 3), "rmse": pytest.approx(np.sqrt(2 / 3)), "r2": 0.0, "n": 3}


@pytest.mark.parametrize("truth,prediction", [([2], [2]), ([2, 2], [1, 3])])
def test_regression_constant_or_single_truth_has_undefined_r2(truth, prediction):
    assert regression_metrics(truth, prediction)["r2"] is None


@pytest.mark.parametrize("truth,prediction", [([], []), ([[1]], [1]), ([1], [1, 2]), ([np.nan], [1]), ([1], [np.inf])])
def test_metrics_reject_nonfinite_empty_and_mismatched_vectors(truth, prediction):
    with pytest.raises(ValueError):
        regression_metrics(truth, prediction)
    with pytest.raises(ValueError):
        binary_classification_metrics(truth, prediction)


def test_binary_metrics_and_probability_calibration_known_values():
    result = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], n_bins=2)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["auroc"] == 0.75
    assert result["auprc"] == pytest.approx(5 / 6)
    for name in ["precision", "recall", "f1", "specificity", "balanced_accuracy", "prevalence"]:
        assert result[name] == 0.5
    assert result["brier"] == pytest.approx(0.185)
    assert result["ece"] == pytest.approx(0.25)
    assert result["calibration"][0]["count"] == 2
    assert result["calibration"][0]["mean_probability"] == 0.25
    assert result["calibration"][0]["fraction_positive"] == 0.5
    json.dumps(result, allow_nan=False)


def test_threshold_boundary_and_explicit_threshold():
    result = binary_classification_metrics([0, 1], [0.499, 0.5])
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]
    result = binary_classification_metrics([0, 1], [0.6, 0.9], threshold=0.8)
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]
    assert result["threshold"] == 0.8


def test_single_class_metrics_are_explicitly_undefined():
    result = binary_classification_metrics([0, 0], [0.0, 0.1])
    for name in ["auroc", "auprc", "precision", "recall", "f1", "balanced_accuracy"]:
        assert result[name] is None
    assert result["specificity"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 0]]
    json.dumps(result, allow_nan=False)
    positive = binary_classification_metrics([1, 1], [0.1, 0.9])
    assert positive["specificity"] is None
    assert positive["recall"] == 0.5
    assert positive["f1"] == pytest.approx(2 / 3)


def test_reliability_endpoints_empty_bins_and_perfect_predictions():
    result = binary_classification_metrics([0, 1], [0.0, 1.0])
    assert result["calibration"][0]["count"] == 1
    assert result["calibration"][-1]["count"] == 1
    assert result["calibration"][1]["mean_probability"] is None
    assert result["calibration"][1]["fraction_positive"] is None
    assert result["brier"] == result["ece"] == 0.0
    assert sum(item["count"] for item in result["calibration"]) == 2


@pytest.mark.parametrize("truth,probability", [([0, 0.5], [0.1, 0.9]), ([0, 2], [0.1, 0.9]), ([0, 1], [-0.1, 0.5]), ([0, 1], [0.5, 1.1])])
def test_binary_metrics_reject_continuous_targets_and_logits(truth, probability):
    with pytest.raises(ValueError):
        binary_classification_metrics(truth, probability)


@pytest.mark.parametrize("kwargs", [{"threshold": float("nan")}, {"threshold": 1.1}, {"n_bins": 0}, {"n_bins": 1.5}, {"n_bins": True}])
def test_binary_metrics_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        binary_classification_metrics([0, 1], [0.1, 0.9], **kwargs)
