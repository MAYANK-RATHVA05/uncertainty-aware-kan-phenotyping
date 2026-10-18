"""Participant exclusivity and held-out invariance for the new dataset workflow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kan_phenotyping.data.conditional_forecasting import ForecastPredictors
from kan_phenotyping.data.forecasting_preprocessing import ConditionalForecastPreprocessor
from kan_phenotyping.data.forecasting_splits import ForecastPartitions, participant_forecast_split


@pytest.fixture
def predictors():
    count = 40
    wearable = pd.DataFrame({
        "steps_awake_mean": np.arange(count * 2, dtype=float),
        "sleep_asleep_weekday_mean": np.arange(count * 2, dtype=float) + 300,
        "steps_mvpa_iqr": np.full(count * 2, np.nan),
    })
    wearable.loc[::7, "steps_awake_mean"] = np.nan
    demographics = pd.DataFrame({
        "birthyear": np.arange(count, dtype=float) + 1960,
        "sex": [0, 1] * (count // 2),
        "educ": [1, 2, 3, np.nan] * (count // 4),
    })
    demographics.loc[::5, "birthyear"] = np.nan
    return ForecastPredictors(
        wearable=wearable,
        demographics=demographics,
        sm0_phq=np.arange(count, dtype=np.float32).reshape(-1, 1) % 28,
        relative_months=np.tile([1, 2], (count, 1)),
        study_months=np.tile([[1, 2], [4, 5], [7, 8], [10, 11]], (count // 4, 1)),
    )


def _fit_state(preprocessor):
    """Inspect the actual fitted medians/scales/category vocabularies."""
    state = {}
    for name in ("wearable", "demographic"):
        transformer = getattr(preprocessor, f"{name}_transformer")
        for kind, pipeline, _ in transformer.transformers_:
            if kind == "remainder":
                continue
            state[f"{name}_{kind}_imputation"] = pipeline.named_steps["imputer"].statistics_.copy()
            if kind == "numeric":
                scaler = pipeline.named_steps["scaler"]
                state[f"{name}_mean"] = scaler.mean_.copy()
                state[f"{name}_scale"] = scaler.scale_.copy()
                state[f"{name}_missing_indicators"] = pipeline.named_steps["imputer"].indicator_.features_.copy()
            else:
                for position, values in enumerate(pipeline.named_steps["encoder"].categories_):
                    state[f"{name}_categories_{position}"] = values.copy()
    state["sm0_mean"] = preprocessor.baseline_scaler.mean_.copy()
    state["sm0_scale"] = preprocessor.baseline_scaler.scale_.copy()
    return state


def test_four_way_split_keeps_all_quarters_together_and_is_reproducible():
    participants = np.repeat([f"p{i:03d}" for i in range(100)], 4)
    first = participant_forecast_split(participants)
    again = participant_forecast_split(participants)
    first.validate(participants)
    for name, expected in (("train", 60), ("validation", 10), ("calibration", 10), ("test", 20)):
        np.testing.assert_array_equal(getattr(first, name), getattr(again, name))
        selected = participants[getattr(first, name)]
        assert len(np.unique(selected)) == expected
        assert np.all(np.unique(selected, return_counts=True)[1] == 4)
    assert not np.array_equal(first.train, participant_forecast_split(participants, seed=43).train)
    # Input row order does not change a participant's assigned partition.
    permutation = np.random.default_rng(7).permutation(len(participants))
    shuffled = participant_forecast_split(participants[permutation])
    for name in ("train", "validation", "calibration", "test"):
        assert set(participants[getattr(first, name)]) == set(participants[permutation][getattr(shuffled, name)])
    metadata = first.metadata(participants)
    assert metadata["calibration"] == {"samples": 40, "participants": 10}
    assert metadata["participant_overlap"] is False
    assert "reserved" in metadata["calibration_role"]


def test_rounding_uses_stable_largest_remainders_and_nonempty_small_partitions():
    split = participant_forecast_split(np.arange(11))
    assert [len(getattr(split, name)) for name in ("train", "validation", "calibration", "test")] == [7, 1, 1, 2]
    tie = participant_forecast_split(np.arange(6), fractions=(0.25,) * 4)
    assert [len(getattr(tie, name)) for name in ("train", "validation", "calibration", "test")] == [2, 2, 1, 1]
    for count in (4, 5):
        participant_forecast_split(np.arange(count)).validate(np.arange(count))


@pytest.mark.parametrize("participants", [
    np.array([]), np.array([["a", "b"]]), np.array(["a", "b", "c"]),
    np.array(["a", "b", "c", ""]), np.array(["a", "b", "c", None], dtype=object),
    np.array([1, 2, 3, np.nan]), np.array([1, 2, 3, np.inf]),
    np.array(["a", "b", "c", 4], dtype=object),
])
def test_split_rejects_missing_invalid_or_insufficient_participants(participants):
    with pytest.raises(ValueError):
        participant_forecast_split(participants)


@pytest.mark.parametrize("fractions", [(0.6, 0.1, 0.3), (0.6, 0.1, 0, 0.3),
                                         (0.6, 0.1, 0.1, 0.3), (np.nan, 0.1, 0.1, 0.2)])
def test_split_rejects_invalid_fractions(fractions):
    with pytest.raises(ValueError, match="fractions"):
        participant_forecast_split(np.arange(10), fractions=fractions)


def test_partition_validation_detects_cross_participant_duplicates_and_invalid_indices():
    with pytest.raises(ValueError, match="Participant leakage"):
        ForecastPartitions(*(np.array([value]) for value in range(4))).validate(np.array(["a", "a", "b", "c"]))
    participants = np.arange(5)
    for train in (np.array([0, 0]), np.array([0.0, 1.0]), np.array([-1, 1]), np.array([0, 5]), np.array([[0, 1]])):
        with pytest.raises(ValueError):
            ForecastPartitions(train, np.array([2]), np.array([3]), np.array([4])).validate(participants)


@pytest.mark.parametrize("partition", ["validation", "calibration", "test"])
def test_held_out_values_cannot_affect_fitted_medians_scales_categories_or_baseline(predictors, partition):
    participants = np.repeat(np.arange(10).astype(str), 4)
    split = participant_forecast_split(participants)
    original = ConditionalForecastPreprocessor().fit(predictors, split.train)
    altered = deepcopy(predictors)
    held_out = getattr(split, partition)
    rows = (held_out[:, None] * 2 + np.arange(2)).ravel()
    altered.wearable.iloc[rows, 0] = 1e15
    altered.wearable.iloc[rows, 1] = np.nan
    # Previously all-missing training feature acquires observations only outside train.
    altered.wearable.iloc[rows, 2] = -1e15
    altered.demographics.loc[held_out, "birthyear"] = -1e10
    altered.demographics.loc[held_out, "sex"] = 99
    altered.demographics.loc[held_out, "educ"] = 99
    altered.sm0_phq[held_out] = 27
    refitted = ConditionalForecastPreprocessor().fit(altered, split.train)
    for name, value in _fit_state(original).items():
        np.testing.assert_array_equal(value, _fit_state(refitted)[name], err_msg=name)
    assert original.feature_names == refitted.feature_names
    np.testing.assert_array_equal(original.transform(predictors, split.train).tabular,
                                  refitted.transform(altered, split.train).tabular)
    assert all("99" not in name for name in original.feature_names["demographics"])
    transformed = refitted.transform(altered, held_out)
    assert np.isfinite(transformed.tabular).all()
    sex_columns = [i for i, name in enumerate(transformed.feature_names["demographics"])
                   if name.startswith("categorical__sex_")]
    assert sex_columns
    assert (transformed.demographics[:, sex_columns] == 0).all()


def test_preprocessor_retains_both_months_missing_features_and_actual_month_positions(predictors):
    train = np.arange(20)
    preprocessor = ConditionalForecastPreprocessor().fit(predictors, train)
    indices = np.array([2, 3, 21])
    transformed = preprocessor.transform(predictors, indices)
    assert transformed.wearable.shape[:2] == (3, 2)
    assert transformed.wearable.dtype == transformed.demographics.dtype == transformed.sm0_phq.dtype == np.float32
    names = transformed.feature_names
    assert "numeric__steps_mvpa_iqr" in names["wearable"]
    assert "numeric__missingindicator_steps_mvpa_iqr" in names["wearable"]
    assert names["tabular"][0].startswith("SM1__")
    assert names["tabular"][transformed.wearable.shape[2]].startswith("SM2__")
    np.testing.assert_array_equal(transformed.relative_months, predictors.relative_months[indices])
    np.testing.assert_array_equal(transformed.study_months, [[7, 8], [10, 11], [4, 5]])
    np.testing.assert_array_equal(transformed.tabular, np.concatenate([
        transformed.wearable.reshape(3, -1), transformed.demographics, transformed.sm0_phq], axis=1))
    assert len(names["tabular"]) == transformed.tabular.shape[1]
    metadata = preprocessor.export_metadata()
    assert metadata["fitted_partition"] == "training_only"
    assert metadata["training_wearable_rows"] == 40
    assert metadata["calibration_used_for_fit"] is metadata["targets_used_for_fit"] is False
    assert not hasattr(preprocessor, "targets")


def test_preprocessor_cannot_accept_future_targets_or_impute_measured_baseline(predictors):
    with pytest.raises(TypeError, match="never a target-bearing dataset"):
        ConditionalForecastPreprocessor().fit(SimpleNamespace(predictors=predictors, targets=np.ones(40)), np.arange(20))
    predictors.sm0_phq[0] = np.nan
    with pytest.raises(ValueError, match="never imputed"):
        ConditionalForecastPreprocessor().fit(predictors, np.arange(20))


def test_preprocessor_rejects_unsafe_schema_invalid_positions_and_invalid_indices(predictors):
    with pytest.raises(RuntimeError, match="training partition"):
        ConditionalForecastPreprocessor().transform(predictors, np.arange(5))
    for indices in (np.array([0, 0]), np.array([-1]), np.array([40]), np.array([0.0]), np.array([], dtype=int)):
        with pytest.raises(ValueError):
            ConditionalForecastPreprocessor().fit(predictors, indices)
    unsafe = deepcopy(predictors)
    unsafe.wearable["phq9_score_end"] = 20
    with pytest.raises(ValueError, match="allowlist"):
        ConditionalForecastPreprocessor().fit(unsafe, np.arange(20))
    invalid = deepcopy(predictors)
    invalid.study_months[0] = [2, 4]
    with pytest.raises(ValueError, match="same-quarter"):
        ConditionalForecastPreprocessor().fit(invalid, np.arange(20))
    preprocessor = ConditionalForecastPreprocessor().fit(predictors, np.arange(20))
    reordered = replace(predictors, wearable=predictors.wearable.iloc[:, ::-1])
    with pytest.raises(ValueError, match="match the fitted training schema"):
        preprocessor.transform(reordered, np.arange(5))
