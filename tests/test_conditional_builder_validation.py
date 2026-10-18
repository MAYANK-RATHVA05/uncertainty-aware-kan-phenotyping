"""Structural/validation tests for the separate conditional dataset builder."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from kan_phenotyping.data.conditional_forecasting import (
    BASELINE_DEMOGRAPHIC_FEATURES,
    WEARABLE_STATISTICS_FEATURES,
    ConditionalForecastAssumptions,
    build_conditional_forecasting_dataset,
)


def _assumptions():
    return ConditionalForecastAssumptions(
        assume_release_index_mapping=True,
        assume_historical_features_available_by_sm2=True,
        assume_sm0_phq_available_by_sm2=True,
        assume_enrollment_demographics=True,
        evidence_reference="docs/temporal_feasibility.md; conditional test fixture",
    )


def _dictionary():
    rows = []
    for name in BASELINE_DEMOGRAPHIC_FEATURES + WEARABLE_STATISTICS_FEATURES:
        static = name in BASELINE_DEMOGRAPHIC_FEATURES
        rows.append({
            "Feature name": name,
            "Category": "Demographic" if static else "Steps" if name.startswith("steps_") else "Sleep",
            "Subcategory": "Static" if static else "Statistics",
            "Description": f"Documented fixture {name}",
            "Notes": "",
        })
    for name in ("phq9_score_start", "phq9_score_end"):
        rows.append({"Feature name": name, "Category": "PHQ-9", "Subcategory": "PHQ-9 score", "Description": "Measured endpoint", "Notes": ""})
    return pd.DataFrame(rows)


def _frame():
    rows = []
    for participant in ("10", "20"):
        for month in range(1, 7):
            row = {name: float(month) for name in BASELINE_DEMOGRAPHIC_FEATURES + WEARABLE_STATISTICS_FEATURES}
            row["__index_level_0__"] = f"{participant}_{month}"
            row["phq9_score_start"] = float(month + 2) if month % 3 == 0 else np.nan
            row["phq9_score_end"] = float(month + 3) if month % 3 == 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _build(frame=None, dictionary=None, assumptions=None):
    return build_conditional_forecasting_dataset(
        _frame() if frame is None else frame,
        _dictionary() if dictionary is None else dictionary,
        _assumptions() if assumptions is None else assumptions,
    )


@pytest.mark.parametrize("flag", [
    "assume_release_index_mapping",
    "assume_historical_features_available_by_sm2",
    "assume_sm0_phq_available_by_sm2",
    "assume_enrollment_demographics",
])
def test_each_conditional_assumption_requires_explicit_opt_in(flag):
    with pytest.raises(ValueError, match="explicit opt-in"):
        _build(assumptions=replace(_assumptions(), **{flag: False}))


def test_default_guard_and_evidence_reference_required():
    with pytest.raises(ValueError, match="explicit opt-in"):
        build_conditional_forecasting_dataset(_frame(), _dictionary())
    with pytest.raises(ValueError, match="evidence_reference"):
        _build(assumptions=replace(_assumptions(), evidence_reference="  "))


def test_minimal_columns_need_neither_generated_categories_nor_row_zero():
    result = _build()
    assert len(result) == 4
    assert result.predictors.wearable.shape == (8, 27)
    assert result.predictors.demographics.shape == (4, 29)
    assert result.predictors.sm0_phq.shape == (4, 1)
    assert result.targets.shape == (4,)
    assert not hasattr(result.predictors, "targets")
    assert not hasattr(result.predictors, "participants")
    assert not hasattr(result.predictors, "target_row_positions")
    np.testing.assert_array_equal(result.quarters, [0, 1, 0, 1])
    np.testing.assert_array_equal(result.predictors.relative_months, [[1, 2]] * 4)
    np.testing.assert_array_equal(result.predictors.study_months, [[1, 2], [4, 5], [1, 2], [4, 5]])
    np.testing.assert_array_equal(result.predictors.sm0_phq[:, 0], [5, 8, 5, 8])
    np.testing.assert_array_equal(result.targets, [6, 9, 6, 9])
    metadata = result.export_metadata()
    assert metadata["study_type"] == "conditional_relative_month_forecasting"
    assert metadata["available_at_timestamps_verified"] is False
    assert "relative study-month" in metadata["horizon"]


@pytest.mark.parametrize("bad_key", [None, "x_1", "10_no", "10_1\n", " 10_1", "10_0", "10_13", "10_999999999999999999999"])
def test_malformed_or_out_of_range_indices_rejected(bad_key):
    frame = _frame()
    frame.loc[0, "__index_level_0__"] = bad_key
    with pytest.raises(ValueError, match="indices|suffix"):
        _build(frame)


def test_duplicates_are_rejected_after_month_normalization():
    frame = _frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate["__index_level_0__"] = "10_01"
    with pytest.raises(ValueError, match="Duplicate normalized participant-month"):
        _build(pd.concat([frame, duplicate], ignore_index=True))


@pytest.mark.parametrize("score", [-1, 28, 2.5, np.inf, "bad"])
@pytest.mark.parametrize("name", ["phq9_score_start", "phq9_score_end"])
def test_invalid_observed_scores_rejected(score, name):
    frame = _frame().astype({name: object})
    frame.loc[2, name] = score
    with pytest.raises(ValueError, match=name):
        _build(frame)


def test_measured_endpoint_at_intermediate_month_rejects_mapping():
    frame = _frame()
    frame.loc[0, "phq9_score_end"] = 5
    with pytest.raises(ValueError, match="non-quarter-end"):
        _build(frame)


def test_missing_measured_scores_excluded_without_imputation():
    frame = _frame()
    frame.loc[2, "phq9_score_end"] = np.nan
    frame.loc[5, "phq9_score_start"] = np.nan
    result = _build(frame)
    assert len(result) == 2
    assert np.isfinite(result.targets).all()
    assert result.audit["missing_measured_endpoint_pair"] == 2
    assert result.audit["missing_sm0_score"] == 1
    assert result.audit["missing_sm3_score"] == 1
    assert result.audit["measured_endpoint_pool"] == 2


def test_individual_missing_predictors_are_retained_not_complete_case_filtered():
    frame = _frame()
    frame.loc[0, WEARABLE_STATISTICS_FEATURES[0]] = np.nan
    frame.loc[1, WEARABLE_STATISTICS_FEATURES[1]] = np.nan
    frame.loc[0, BASELINE_DEMOGRAPHIC_FEATURES[0]] = np.nan
    result = _build(frame)
    assert len(result) == 4
    assert pd.isna(result.predictors.wearable.iloc[0, 0])
    assert pd.isna(result.predictors.wearable.iloc[1, 1])
    assert pd.isna(result.predictors.demographics.iloc[0, 0])
    assert result.audit["individual_missing_predictor_values_retained"] == {"wearable": 2, "demographics": 1}


def test_missing_history_pattern_partition_and_empty_cohort_are_auditable():
    frame = _frame()
    frame = frame.loc[~frame["__index_level_0__"].isin(["10_1", "10_5", "20_1", "20_2"])].copy()
    result = _build(frame)
    assert result.audit["history_patterns_samples"] == {"both": 1, "sm1_only": 1, "sm2_only": 1, "neither": 1}
    assert result.audit["excluded_missing_required_history"] == 3
    assert len(result) == 1
    only_endpoints = _frame().loc[lambda f: f["phq9_score_start"].notna()].copy()
    empty = _build(only_endpoints)
    assert len(empty) == 0
    assert empty.predictors.wearable.shape == (0, 27)
    assert empty.predictors.demographics.shape == (0, 29)
    assert empty.history_row_positions.shape == (0, 2)
    assert empty.predictors.sm0_phq.shape == (0, 1)
    assert empty.audit["excluded_missing_required_history"] == 4


def test_sample_payload_is_canonical_under_input_row_shuffle():
    first = _build()
    shuffled = _build(_frame().sample(frac=1, random_state=27))
    for name in ("participants", "quarters", "endpoint_months", "targets"):
        np.testing.assert_array_equal(getattr(first, name), getattr(shuffled, name))
    for name in ("wearable", "demographics"):
        pd.testing.assert_frame_equal(getattr(first.predictors, name), getattr(shuffled.predictors, name))
    np.testing.assert_array_equal(first.predictors.sm0_phq, shuffled.predictors.sm0_phq)
    np.testing.assert_array_equal(first.predictors.study_months, shuffled.predictors.study_months)


@pytest.mark.parametrize("change", ["missing", "wrong_category", "wrong_subcategory", "blank_description", "duplicate"])
def test_dictionary_must_match_reviewed_allowlist(change):
    dictionary = _dictionary()
    if change == "missing":
        dictionary = dictionary.iloc[1:]
    elif change == "wrong_category":
        dictionary.loc[0, "Category"] = "Steps"
    elif change == "wrong_subcategory":
        dictionary.loc[0, "Subcategory"] = "Dynamic"
    elif change == "blank_description":
        dictionary.loc[0, "Description"] = "  "
    else:
        dictionary = pd.concat([dictionary, dictionary.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="dictionary|description"):
        _build(dictionary=dictionary)


def test_adding_documented_features_never_expands_fixed_allowlist():
    frame = _frame()
    frame["new_wearable"] = 12345
    dictionary = pd.concat([
        _dictionary(), pd.DataFrame([{
            "Feature name": "new_wearable", "Category": "Steps", "Subcategory": "Statistics",
            "Description": "New feature that was not reviewed", "Notes": "",
        }]),
    ], ignore_index=True)
    result = _build(frame, dictionary)
    assert "new_wearable" not in result.predictors.wearable
    assert result.predictors.wearable.shape[1] == 27
    assert "fixed reviewed" in result.predictors.exclusion_metadata["new_wearable"]


def test_builder_does_not_modify_source_or_dictionary():
    frame, dictionary = _frame(), _dictionary()
    before_frame, before_dictionary = frame.copy(deep=True), dictionary.copy(deep=True)
    result = _build(frame, dictionary)
    result.predictors.wearable.iloc[0, 0] = -999
    result.predictors.demographics.iloc[0, 0] = -999
    pd.testing.assert_frame_equal(frame, before_frame)
    pd.testing.assert_frame_equal(dictionary, before_dictionary)
