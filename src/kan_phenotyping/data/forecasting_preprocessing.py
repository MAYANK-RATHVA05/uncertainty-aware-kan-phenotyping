"""Training-only predictor transforms for conditional relative-month forecasting.

Unlike the retrospective preprocessor, the tabular representation retains both
SM1 and SM2. This API accepts a predictor payload only: targets are not stored
on, passed to, transformed by, or imputed by this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from .conditional_forecasting import (
    BASELINE_DEMOGRAPHIC_FEATURES, DEMOGRAPHIC_CATEGORICAL_FEATURES,
    WEARABLE_STATISTICS_FEATURES, ForecastPredictors,
)
from .preprocessing import _indices, _normalise_block, _transformer


@dataclass(frozen=True)
class ForecastTransformedPredictors:
    wearable: np.ndarray
    demographics: np.ndarray
    sm0_phq: np.ndarray
    relative_months: np.ndarray
    study_months: np.ndarray
    tabular: np.ndarray
    feature_names: dict[str, list[str]]


class ConditionalForecastPreprocessor:
    """Fit medians, scales and category vocabulary on training predictors only.

    The two actual monthly wearable packets share one fitted transformation;
    demographics and the measured SM0 severity have separate transforms. Infinite
    numeric predictor values are treated as missing, as in the existing workflow.
    Missing numeric values use training medians plus training-discovered missing
    indicators. All-missing training columns remain present (imputed zero).
    Categorical missingness gets an explicit category, unseen categories map to
    all-zero one-hot values. Measured SM0 severity is scaled and never imputed.
    """

    def __init__(self) -> None:
        self._fitted = False

    @staticmethod
    def _layout(predictors: ForecastPredictors) -> None:
        if not isinstance(predictors, ForecastPredictors):
            raise TypeError("Pass only ForecastPredictors, never a target-bearing dataset.")
        size = len(predictors)
        if (size == 0 or len(predictors.wearable) != size * 2
                or len(predictors.demographics) != size
                or np.asarray(predictors.sm0_phq).shape != (size, 1)
                or np.asarray(predictors.relative_months).shape != (size, 2)
                or np.asarray(predictors.study_months).shape != (size, 2)):
            raise ValueError("Predictor blocks must have matching N, two actual monthly rows, and one SM0 score.")
        if predictors.wearable.columns.duplicated().any() or predictors.demographics.columns.duplicated().any():
            raise ValueError("Predictor feature columns must be unique.")
        if (not set(predictors.wearable.columns).issubset(WEARABLE_STATISTICS_FEATURES)
                or not set(predictors.demographics.columns).issubset(BASELINE_DEMOGRAPHIC_FEATURES)):
            raise ValueError("Predictor columns must belong to the reviewed forecasting feature allowlist.")
        if not np.array_equal(predictors.relative_months, np.tile([1, 2], (size, 1))):
            raise ValueError("Relative history positions must remain exactly SM1 and SM2.")
        actual = np.asarray(predictors.study_months)
        if (not np.issubdtype(actual.dtype, np.integer) or not np.isin(actual[:, 0], [1, 4, 7, 10]).all()
                or not np.array_equal(actual[:, 1], actual[:, 0] + 1)):
            raise ValueError("Study positions must preserve same-quarter SM1/SM2 suffixes.")

    @staticmethod
    def _baseline(predictors: ForecastPredictors, indices: np.ndarray) -> np.ndarray:
        try:
            values = np.asarray(predictors.sm0_phq[indices], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Measured SM0 PHQ-9 must be numeric.") from exc
        if (not np.isfinite(values).all() or not ((values >= 0) & (values <= 27)).all()
                or not np.equal(values, np.floor(values)).all()):
            raise ValueError("Measured SM0 PHQ-9 must be observed integer scores in 0..27; never imputed.")
        return values

    def fit(self, predictors: ForecastPredictors, train_indices: np.ndarray) -> "ConditionalForecastPreprocessor":
        self._fitted = False
        self._layout(predictors)
        indices = _indices(train_indices, len(predictors))
        rows = (indices[:, None] * 2 + np.arange(2)[None, :]).ravel()
        self.wearable_columns_ = tuple(predictors.wearable.columns)
        self.demographic_columns_ = tuple(predictors.demographics.columns)
        categorical = tuple(name for name in self.demographic_columns_
                            if name in DEMOGRAPHIC_CATEGORICAL_FEATURES)
        self.categorical_columns_ = categorical
        self.wearable_transformer = _transformer(self.wearable_columns_, ())
        self.demographic_transformer = _transformer(self.demographic_columns_, categorical)
        if self.wearable_transformer is None:
            raise ValueError("At least one wearable Statistics predictor is required.")
        self.wearable_transformer.fit(_normalise_block(predictors.wearable.iloc[rows], ()))
        if self.demographic_transformer is not None:
            self.demographic_transformer.fit(_normalise_block(predictors.demographics.iloc[indices], categorical))
        self.baseline_scaler = StandardScaler().fit(self._baseline(predictors, indices))
        self.n_training_samples_ = len(indices)
        self.n_training_wearable_rows_ = len(rows)
        self._fitted = True
        return self

    @property
    def feature_names(self) -> dict[str, list[str]]:
        if not self._fitted:
            raise RuntimeError("Preprocessing must be fitted on the training partition first.")
        wearable = self.wearable_transformer.get_feature_names_out().tolist()
        demographics = (self.demographic_transformer.get_feature_names_out().tolist()
                        if self.demographic_transformer is not None else [])
        baseline = ["measured_sm0_phq9_standardized"]
        tabular = ([f"SM{month}__{name}" for month in (1, 2) for name in wearable]
                   + [f"baseline_demographic__{name}" for name in demographics] + baseline)
        return {"wearable": wearable, "demographics": demographics,
                "sm0_phq": baseline, "tabular": tabular}

    def transform(self, predictors: ForecastPredictors, indices: np.ndarray) -> ForecastTransformedPredictors:
        if not self._fitted:
            raise RuntimeError("Preprocessing must be fitted on the training partition first.")
        self._layout(predictors)
        if (tuple(predictors.wearable.columns) != self.wearable_columns_
                or tuple(predictors.demographics.columns) != self.demographic_columns_):
            raise ValueError("Predictor columns and order must match the fitted training schema.")
        indices = _indices(indices, len(predictors))
        rows = (indices[:, None] * 2 + np.arange(2)[None, :]).ravel()
        wearable = self.wearable_transformer.transform(
            _normalise_block(predictors.wearable.iloc[rows], ())
        ).astype(np.float32).reshape(len(indices), 2, -1)
        demographics = (self.demographic_transformer.transform(
            _normalise_block(predictors.demographics.iloc[indices], self.categorical_columns_)
        ).astype(np.float32) if self.demographic_transformer is not None
            else np.empty((len(indices), 0), dtype=np.float32))
        baseline = self.baseline_scaler.transform(self._baseline(predictors, indices)).astype(np.float32)
        tabular = np.concatenate([wearable.reshape(len(indices), -1), demographics, baseline], axis=1)
        if not np.isfinite(tabular).all():
            raise ValueError("Preprocessing produced nonfinite predictor values.")
        return ForecastTransformedPredictors(
            wearable, demographics, baseline,
            np.asarray(predictors.relative_months)[indices].copy(),
            np.asarray(predictors.study_months)[indices].copy(),
            tabular, self.feature_names,
        )

    def export_metadata(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Preprocessing must be fitted on the training partition first.")
        return {
            "study_type": "conditional_relative_month_forecasting",
            "fitted_partition": "training_only",
            "training_samples": self.n_training_samples_,
            "training_wearable_rows": self.n_training_wearable_rows_,
            "calibration_used_for_fit": False,
            "validation_used_for_fit": False,
            "test_used_for_fit": False,
            "targets_used_for_fit": False,
            "feature_names": self.feature_names,
            "raw_wearable_features": list(self.wearable_columns_),
            "raw_demographic_features": list(self.demographic_columns_),
            "categorical_features": list(self.categorical_columns_),
            "numeric_transform": "training median imputation, fitted missingness indicators, training StandardScaler",
            "empty_numeric_features": "retained; zero imputation and training missingness indicator",
            "nonfinite_numeric_predictors": "infinite values treated as missing",
            "categorical_transform": "missing category; training one-hot vocabulary; unseen categories ignored",
            "baseline_transform": "training StandardScaler only; measured SM0 score never imputed",
            "tabular_layout": "flattened SM1 then SM2 wearable features, baseline demographics, measured SM0 PHQ",
            "time_positions": "unaltered integer relative/study month positions; no daily upsampling",
            "available_at_timestamps_verified": False,
        }
