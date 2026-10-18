"""Strict regression and binary metrics with JSON-compatible results.

Undefined statistics are returned as ``None``, never NaN. Functions require
nonempty, equally sized one-dimensional vectors containing finite numbers.
Classification labels must already be exactly 0/1; no target binarization or
threshold optimization is performed here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score


def _finite_vector(values: ArrayLike, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric vector") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _pair(y_true: ArrayLike, prediction: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    truth = _finite_vector(y_true, "y_true")
    predicted = _finite_vector(prediction, "prediction")
    if truth.shape != predicted.shape:
        raise ValueError("y_true and prediction must have the same length")
    return truth, predicted


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, Any]:
    """Return ``mae``, ``rmse``, ``r2`` and ``n`` for continuous predictions.

    R-squared is undefined (``None``) for fewer than two observations or constant
    ground-truth values, including when their predictions are perfect.
    """
    truth, prediction = _pair(y_true, y_pred)
    residual = prediction - truth
    mse = float(np.mean(np.square(residual)))
    total = float(np.sum(np.square(truth - np.mean(truth))))
    r2 = None
    if truth.size >= 2 and total > 0.0:
        r2 = float(1.0 - np.sum(np.square(residual)) / total)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
        "n": int(truth.size),
    }


def binary_classification_metrics(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate 0/1 targets against positive-class probabilities in [0, 1].

    Predictions are positive when ``probability >= threshold``. The threshold
    defaults to 0.5; callers may explicitly supply a threshold selected using
    validation data. This function never selects or changes that threshold.

    Returned keys are ``auroc``, ``auprc`` (average precision), ``precision``,
    ``recall``, ``f1``, ``balanced_accuracy``, ``specificity``,
    ``confusion_matrix`` (rows true labels 0/1, columns predicted labels 0/1),
    ``brier``, ``prevalence``, ``calibration``, ``ece``, ``threshold`` and ``n``.
    AUROC, AUPRC and balanced accuracy are ``None`` for single-class targets;
    other undefined ratios are ``None`` when their denominator is zero.

    ``calibration`` contains ``n_bins`` equal-width reliability bins. Each bin
    records ``lower``, ``upper``, ``count``, ``mean_probability`` and
    ``fraction_positive``. Bins include their lower bound and exclude their
    upper bound, except the final bin includes 1. Empty-bin means are ``None``.
    ``ece`` is the sample-weighted absolute difference of the two bin means.
    """
    truth, probability = _pair(y_true, y_probability)
    if not np.isin(truth, [0.0, 1.0]).all():
        raise ValueError("y_true must contain binary labels 0 and 1")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("y_probability must be in [0, 1], not logits")
    if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    positive = truth == 1
    predicted_positive = probability >= threshold
    tp = int(np.sum(positive & predicted_positive))
    tn = int(np.sum(~positive & ~predicted_positive))
    fp = int(np.sum(~positive & predicted_positive))
    fn = int(np.sum(positive & ~predicted_positive))

    def ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    has_both_classes = bool(positive.any() and (~positive).any())
    balanced_accuracy = (
        float((recall + specificity) / 2.0) if has_both_classes else None
    )

    bin_ids = np.minimum((probability * n_bins).astype(np.int64), n_bins - 1)
    calibration = []
    ece = 0.0
    for index in range(n_bins):
        mask = bin_ids == index
        count = int(np.sum(mask))
        confidence = float(np.mean(probability[mask])) if count else None
        fraction_positive = float(np.mean(truth[mask])) if count else None
        calibration.append(
            {
                "lower": float(index / n_bins),
                "upper": float((index + 1) / n_bins),
                "count": count,
                "mean_probability": confidence,
                "fraction_positive": fraction_positive,
            }
        )
        if count:
            ece += count / truth.size * abs(confidence - fraction_positive)

    return {
        "auroc": float(roc_auc_score(truth, probability)) if has_both_classes else None,
        "auprc": float(average_precision_score(truth, probability)) if has_both_classes else None,
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": balanced_accuracy,
        "specificity": specificity,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "brier": float(np.mean(np.square(probability - truth))),
        "prevalence": float(np.mean(truth)),
        "calibration": calibration,
        "ece": float(ece),
        "threshold": float(threshold),
        "n": int(truth.size),
    }
