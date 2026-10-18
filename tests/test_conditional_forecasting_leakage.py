"""Counterfactual leakage checks for conditional relative-month forecasting.

All identities and values below are synthetic. Tests change information that
would be unavailable at the cutoff and check the resulting sample contents,
rather than relying on column names or the builder's own audit assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kan_phenotyping.data import (
    ForecastConfig,
    load_feature_schema,
    prepare_forecasting_examples,
)
from kan_phenotyping.data.conditional_forecasting import (
    BASELINE_DEMOGRAPHIC_FEATURES,
    WEARABLE_STATISTICS_FEATURES,
    ConditionalForecastAssumptions,
    build_conditional_forecasting_dataset,
)


@pytest.fixture
def conditional_fixture():
    # Shared prefixes make substring/prefix matching fail the content checks.
    participants = ("1", "11", "111", "2", "22", "3", "33", "333")
    rows = []
    for person_index, person in enumerate(participants):
        for month in range(1, 7):
            quarter = (month - 1) // 3
            endpoint = month % 3 == 0
            row = {
                "__index_level_0__": f"{person}_{month}",
                "phq9_score_start": 2 + person_index + quarter * 3 if endpoint else np.nan,
                "phq9_score_end": 18 + person_index - quarter if endpoint else np.nan,
                "phq9_cat_start": 1 if endpoint else np.nan,
                "phq9_cat_end": 2 if endpoint else np.nan,
                "med_start": month % 2,
                "steps__awake__sum__score": person_index * 1000 + month,
                "generated_phq_category": month % 5,
                "generated_phq_probability": month / 10,
                "undocumented_future_proxy": person_index * 1000 + month,
            }
            row.update({
                name: person_index * 10000 + month * 100 + feature_index
                for feature_index, name in enumerate(WEARABLE_STATISTICS_FEATURES)
            })
            row.update({
                name: (person_index + feature_index) % 3
                for feature_index, name in enumerate(BASELINE_DEMOGRAPHIC_FEATURES)
            })
            rows.append(row)
    frame = pd.DataFrame(rows).sample(frac=1, random_state=91).reset_index(drop=True)
    specs = [
        (name, "Sleep" if name.startswith("sleep") else "Steps", "Statistics", f"Observed {name}", "")
        for name in WEARABLE_STATISTICS_FEATURES
    ] + [
        (name, "Demographic", "Static", f"Enrollment {name}", "")
        for name in BASELINE_DEMOGRAPHIC_FEATURES
    ] + [
        ("med_start", "Demographic", "Dynamic", "Started medication, past month", ""),
        ("steps__awake__sum__score", "Steps", "Linear regression", "", ""),
        ("phq9_score_start", "PHQ-9", "PHQ-9 score", "PHQ-9 score at the start of the 3 month period", "0-27"),
        ("phq9_score_end", "PHQ-9", "PHQ-9 score", "PHQ-9 score at the end of the 3 month period", "0-27"),
        ("phq9_cat_start", "PHQ-9", "PHQ-9 category", "Start category", "0-4"),
        ("phq9_cat_end", "PHQ-9", "PHQ-9 category", "End category", "0-4"),
    ]
    dictionary = pd.DataFrame(
        specs, columns=["Feature name", "Category", "Subcategory", "Description", "Notes"]
    )
    assumptions = ConditionalForecastAssumptions(
        assume_release_index_mapping=True,
        assume_historical_features_available_by_sm2=True,
        assume_sm0_phq_available_by_sm2=True,
        assume_enrollment_demographics=True,
        evidence_reference="Synthetic generator: known participant/quarter/month and enrollment values.",
    )
    return frame, dictionary, assumptions


def _build(fixture, frame=None):
    original, dictionary, assumptions = fixture
    return build_conditional_forecasting_dataset(
        original if frame is None else frame, dictionary, assumptions
    )


def _assert_predictors_identical(left, right):
    pd.testing.assert_frame_equal(
        left.predictors.wearable, right.predictors.wearable, check_dtype=False
    )
    pd.testing.assert_frame_equal(
        left.predictors.demographics, right.predictors.demographics, check_dtype=False
    )
    for name in ("sm0_phq", "relative_months", "study_months"):
        np.testing.assert_array_equal(getattr(left.predictors, name), getattr(right.predictors, name))


def _assert_mapping_identical(left, right):
    for name in (
        "participants", "quarters", "endpoint_months", "history_row_positions",
        "baseline_row_positions", "target_row_positions",
    ):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


@pytest.mark.parametrize("future_value", [1e100, "future packet unavailable at cutoff"])
def test_sm3_wearables_and_excluded_features_cannot_change_any_predictor(conditional_fixture, future_value):
    frame, _, _ = conditional_fixture
    original = _build(conditional_fixture)
    changed = frame.copy(deep=True)
    endpoints = changed["__index_level_0__"].str.rsplit("_", n=1).str[-1].astype(int) % 3 == 0
    # Known future observations become arbitrarily different, including missing.
    changed[list(WEARABLE_STATISTICS_FEATURES)] = changed[list(WEARABLE_STATISTICS_FEATURES)].astype(object)
    changed.loc[endpoints, list(WEARABLE_STATISTICS_FEATURES)] = future_value
    changed.loc[endpoints, WEARABLE_STATISTICS_FEATURES[0]] = np.nan
    # These exclusions apply at every month, not merely the endpoint.
    excluded = [
        "med_start", "steps__awake__sum__score", "generated_phq_category",
        "generated_phq_probability", "undocumented_future_proxy",
    ]
    changed[excluded] = -1e100
    changed.loc[endpoints, ["phq9_cat_start", "phq9_cat_end"]] = 4
    rebuilt = _build(conditional_fixture, changed)
    _assert_predictors_identical(original, rebuilt)
    _assert_mapping_identical(original, rebuilt)
    np.testing.assert_array_equal(original.targets, rebuilt.targets)
    assert set(rebuilt.predictors.wearable.columns) == set(WEARABLE_STATISTICS_FEATURES)
    assert set(rebuilt.predictors.demographics.columns) == set(BASELINE_DEMOGRAPHIC_FEATURES)


def test_changing_sm3_score_changes_only_its_target_not_the_next_baseline(conditional_fixture):
    frame, _, _ = conditional_fixture
    original = _build(conditional_fixture)
    changed = frame.copy(deep=True)
    changed.loc[changed["__index_level_0__"] == "1_3", "phq9_score_end"] = 0
    rebuilt = _build(conditional_fixture, changed)
    _assert_predictors_identical(original, rebuilt)
    _assert_mapping_identical(original, rebuilt)
    selected = (original.participants == "1") & (original.quarters == 0)
    assert selected.sum() == 1
    np.testing.assert_array_equal(original.targets[~selected], rebuilt.targets[~selected])
    assert np.asarray(rebuilt.targets[selected]).item() == 0
    assert np.asarray(original.targets[selected]).item() != 0


def test_each_quarter_uses_its_own_measured_sm0_not_one_enrollment_score(conditional_fixture):
    frame, _, _ = conditional_fixture
    examples = _build(conditional_fixture)
    source = frame.set_index("__index_level_0__")
    assert len(examples.targets) == 16
    assert examples.predictors.sm0_phq.shape == (16, 1)
    for index, (person, quarter) in enumerate(zip(examples.participants, examples.quarters)):
        endpoint_month = (int(quarter) + 1) * 3
        endpoint = source.loc[f"{person}_{endpoint_month}"]
        assert examples.predictors.sm0_phq[index, 0] == endpoint["phq9_score_start"]
        assert examples.targets[index] == endpoint["phq9_score_end"]
        assert examples.baseline_row_positions[index] == examples.target_row_positions[index]
    person_rows = examples.participants == "1"
    assert len(np.unique(examples.predictors.sm0_phq[person_rows])) == 2
    # The deliberately unequal consecutive scores make accidental derivation of
    # quarter two's baseline from quarter one's target observable in this test.
    assert source.loc["1_3", "phq9_score_end"] != source.loc["1_6", "phq9_score_start"]


def test_history_uses_exact_person_quarter_and_relative_positions_after_shuffle(conditional_fixture):
    frame, _, _ = conditional_fixture
    examples = _build(conditional_fixture)
    assert examples.predictors.wearable.shape == (32, 27)
    assert examples.predictors.demographics.shape == (16, 29)
    assert examples.history_row_positions.shape == (16, 2)
    np.testing.assert_array_equal(examples.predictors.relative_months, np.tile([1, 2], (16, 1)))
    for index, (person, quarter) in enumerate(zip(examples.participants, examples.quarters)):
        expected_months = np.array([quarter * 3 + 1, quarter * 3 + 2])
        expected_keys = [f"{person}_{month}" for month in expected_months]
        np.testing.assert_array_equal(examples.predictors.study_months[index], expected_months)
        rows = frame.iloc[examples.history_row_positions[index]]
        assert rows["__index_level_0__"].tolist() == expected_keys
        actual = examples.predictors.wearable.iloc[index * 2:index * 2 + 2].reset_index(drop=True)
        expected = rows[list(WEARABLE_STATISTICS_FEATURES)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


@pytest.mark.parametrize("missing", [("1_1",), ("1_2",), ("1_1", "1_2")])
def test_missing_history_cannot_be_borrowed_from_another_person_or_quarter(conditional_fixture, missing):
    frame, _, _ = conditional_fixture
    reduced = frame.loc[~frame["__index_level_0__"].isin(missing)].reset_index(drop=True)
    examples = _build(conditional_fixture, reduced)
    keys = set(zip(examples.participants.tolist(), examples.quarters.tolist()))
    assert len(examples.targets) == 15
    assert ("1", 0) not in keys
    assert ("1", 1) in keys
    assert ("11", 0) in keys and ("111", 0) in keys


@pytest.mark.parametrize("missing_score", ["phq9_score_start", "phq9_score_end"])
def test_missing_measured_endpoint_excludes_sample_instead_of_imputing(conditional_fixture, missing_score):
    frame, _, _ = conditional_fixture
    changed = frame.copy(deep=True)
    changed.loc[changed["__index_level_0__"] == "1_3", missing_score] = np.nan
    # An intermediate generated category remains present but cannot substitute.
    assert changed["generated_phq_category"].notna().all()
    examples = _build(conditional_fixture, changed)
    assert len(examples.targets) == 15
    assert ("1", 0) not in set(zip(examples.participants.tolist(), examples.quarters.tolist()))
    assert np.isfinite(examples.targets).all()


def test_predictor_missingness_is_retained_without_removing_or_padding_history(conditional_fixture):
    frame, _, _ = conditional_fixture
    changed = frame.copy(deep=True)
    changed.loc[changed["__index_level_0__"] == "1_1", list(WEARABLE_STATISTICS_FEATURES)] = np.nan
    changed.loc[changed["__index_level_0__"] == "1_2", WEARABLE_STATISTICS_FEATURES[0]] = np.nan
    examples = _build(conditional_fixture, changed)
    assert len(examples.targets) == 16
    selected = np.flatnonzero((examples.participants == "1") & (examples.quarters == 0)).item()
    history = examples.predictors.wearable.iloc[selected * 2:selected * 2 + 2]
    assert history.iloc[0].isna().all()
    assert pd.isna(history.iloc[1, 0])
    assert history.iloc[1, 1:].notna().all()
    assert examples.history_row_positions.shape == (16, 2)
    np.testing.assert_array_equal(examples.predictors.relative_months[selected], [1, 2])


def test_missing_demographic_is_not_backfilled_from_sm2_or_sm3(conditional_fixture):
    frame, _, _ = conditional_fixture
    changed = frame.copy(deep=True)
    feature = BASELINE_DEMOGRAPHIC_FEATURES[0]
    changed[feature] = changed[feature].astype(float)
    changed.loc[changed["__index_level_0__"] == "1_1", feature] = np.nan
    original = _build(conditional_fixture, changed)
    selected = np.flatnonzero((original.participants == "1") & (original.quarters == 0)).item()
    assert pd.isna(original.predictors.demographics.iloc[selected][feature])
    # A later nonmissing value must not fill the earlier source's missing value.
    changed.loc[changed["__index_level_0__"].isin(["1_2", "1_3"]), feature] = 999
    rebuilt = _build(conditional_fixture, changed)
    _assert_predictors_identical(original, rebuilt)
    np.testing.assert_array_equal(original.targets, rebuilt.targets)


def test_new_conditional_api_does_not_enable_the_existing_verified_forecast_guard(conditional_fixture, tmp_path):
    frame, dictionary, _ = conditional_fixture
    # Successfully building the conditional candidate must have no global side
    # effect on the older explicitly verified, seven-month forecast workflow.
    _build(conditional_fixture)
    path = tmp_path / "dictionary.tsv"
    dictionary.to_csv(path, sep="\t", index=False)
    schema = load_feature_schema(path, frame.columns.tolist())
    assert ForecastConfig().verified_index_semantics is False
    assert ForecastConfig().verified_outcome_alignment is False
    assert ForecastConfig().verified_feature_windows is False
    with pytest.raises(ValueError, match="Forecasting is blocked"):
        prepare_forecasting_examples(frame, schema)
