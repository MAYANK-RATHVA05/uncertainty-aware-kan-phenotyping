"""Synthetic checks for leakage exclusions, grouping, and fitted statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kan_phenotyping.data import (
    ForecastConfig, ParticipantSplit, TabularPreprocessor, group_split,
    load_dataset, load_feature_schema, prepare_forecasting_examples,
    prepare_tabular_examples,
)


@pytest.fixture
def synthetic_data(tmp_path):
    specs = [
        ("steps", "Steps", "Statistics", "Mean steps"),
        ("sleep", "Sleep", "Statistics", "Sleep duration"),
        ("all_missing", "Sleep", "Statistics", "Sometimes unavailable measurement"),
        ("sex", "Demographic", "Static", "Sex"),
        ("birthyear", "Demographic", "Static", "Birth year"),
        ("med_start", "Demographic", "Dynamic", "Started medication past month"),
        ("future_lr", "Steps", "Linear regression", ""),
        ("undescribed", "Steps", "Statistics", ""),
        ("phq9_score_start", "PHQ-9", "PHQ-9 score", "Start score"),
        ("phq9_score_end", "PHQ-9", "PHQ-9 score", "End score"),
        ("phq9_cat_start", "PHQ-9", "PHQ-9 category", "Start category"),
        ("phq9_cat_end", "PHQ-9", "PHQ-9 category", "End category"),
    ]
    dictionary = pd.DataFrame(specs, columns=["Feature name", "Category", "Subcategory", "Description"])
    dictionary["Notes"] = ""
    dictionary_path = tmp_path / "dictionary.tsv"
    dictionary.to_csv(dictionary_path, sep="\t", index=False)
    rows = []
    for participant in range(10):
        for month in range(1, 13):
            labeled = month in (3, 6, 9, 12)
            start = 5 + participant % 5 if labeled else np.nan
            rows.append({
                "__index_level_0__": f"{participant}_{month}",
                "steps": participant * 100 + month,
                "sleep": 420 + participant + month,
                "all_missing": np.nan,
                "sex": participant % 2,
                "birthyear": 1970 + participant,
                "med_start": month % 2,
                "future_lr": month * 1e10,
                "undescribed": 999,
                "unknown": 1000,
                "phq9_score_start": start,
                "phq9_score_end": start + participant % 3 if labeled else np.nan,
                "phq9_cat_start": 1 if labeled else np.nan,
                "phq9_cat_end": 1 if labeled else np.nan,
            })
    frame = pd.DataFrame(rows)
    return frame, load_feature_schema(dictionary_path, frame.columns.tolist()), dictionary_path


def test_loading_and_primary_tabular_excludes_outcomes_and_unverified_features(synthetic_data, tmp_path):
    frame, _, dictionary = synthetic_data
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    loaded, schema = load_dataset(path, dictionary)
    examples = prepare_tabular_examples(loaded, schema)
    assert len(examples) == 40
    assert set(examples.tabular.columns) == {"steps", "sleep", "all_missing", "sex", "birthyear"}
    for name in ["unknown", "future_lr", "med_start", "undescribed", "__index_level_0__", "phq9_score_start", "phq9_score_end"]:
        assert name in examples.schema.exclusion_reasons
        assert name not in examples.tabular.columns
    np.testing.assert_array_equal(examples.y, np.repeat(np.arange(10) % 3, 4))
    assert examples.y.dtype == np.float32
    metadata = examples.export_metadata()
    assert metadata["index_grouping"] == "composite_prefix_provisional"
    assert metadata["study_type"] == "retrospective_association"
    assert metadata["group_identity_verified"] is False
    assert metadata["temporal_semantics_verified"] is False
    assert "participant" not in metadata.get("index_grouping", "")


def test_group_split_is_deterministic_disjoint_and_covers_repeated_groups():
    groups = np.repeat(np.arange(100).astype(str), 4)
    split = group_split(groups, seed=13)
    repeated = group_split(groups, seed=13)
    for name, expected in [("train", 70), ("validation", 10), ("test", 20)]:
        values = getattr(split, name)
        np.testing.assert_array_equal(values, getattr(repeated, name))
        assert len(np.unique(groups[values])) == expected
    split.validate(groups)
    assert split.export_metadata(groups)["group_overlap"] is False
    changed = group_split(groups, seed=14)
    assert not np.array_equal(split.train, changed.train)
    assert set(groups[split.train]).isdisjoint(groups[split.validation])
    assert set(groups[split.train]).isdisjoint(groups[split.test])


def test_split_detects_overlap_and_rejects_insufficient_groups():
    with pytest.raises(ValueError, match="At least three"):
        group_split(np.array(["a", "a", "b"]))
    with pytest.raises(ValueError, match="Participant leakage"):
        ParticipantSplit(np.array([0]), np.array([1]), np.array([2])).validate(np.array(["a", "a", "b"]))
    for n in (3, 4, 5):
        group_split(np.arange(n)).validate(np.arange(n))


def test_preprocessing_statistics_are_training_only_and_empty_columns_survive(synthetic_data):
    frame, schema, _ = synthetic_data
    # The first participant supplies four labeled rows; other participants have
    # extreme held-out measurements and an unseen categorical code.
    frame.loc[frame.index >= 12, "steps"] = 1e9
    frame.loc[frame.index >= 12, "sex"] = 2
    frame.loc[2, "steps"] = np.nan
    examples = prepare_tabular_examples(frame, schema)
    train = np.arange(4)
    held_out = np.arange(4, len(examples))
    preprocessor = TabularPreprocessor(examples.schema).fit(examples, train)
    imputer = preprocessor.temporal_transformer.named_transformers_["numeric"].named_steps["imputer"]
    scaler = preprocessor.temporal_transformer.named_transformers_["numeric"].named_steps["scaler"]
    steps_index = list(examples.schema.temporal_columns).index("steps")
    assert imputer.statistics_[steps_index] == 9  # median of observed 6, 9, 12
    assert scaler.mean_[steps_index] == 9
    transformed = preprocessor.transform(examples, held_out)
    assert np.isfinite(transformed.tabular).all()
    assert transformed.temporal.shape[1] == 1
    names = preprocessor.feature_names["tabular"]
    assert "numeric__all_missing" in names
    assert "numeric__missingindicator_all_missing" in names
    # Unknown categorical code has no training one-hot category and maps to 0.
    sex_indices = [i for i, name in enumerate(names) if name.startswith("categorical__sex_")]
    assert sex_indices
    assert (transformed.tabular[:, sex_indices] == 0).all()
    assert not any("sex_2" in name for name in names)
    assert preprocessor.export_metadata()["fitted_partition"] == "training_only"
    np.testing.assert_allclose(
        transformed.tabular,
        np.concatenate([transformed.temporal[:, -1], transformed.static], axis=1),
    )


def test_preprocessor_rejects_transform_before_fit_and_bad_numeric_values(synthetic_data):
    frame, schema, _ = synthetic_data
    examples = prepare_tabular_examples(frame, schema)
    with pytest.raises(RuntimeError, match="training partition"):
        TabularPreprocessor(examples.schema).transform(examples, np.array([0]))
    examples.temporal["steps"] = examples.temporal["steps"].astype(object)
    examples.temporal.loc[0, "steps"] = "not a number"
    with pytest.raises(ValueError, match="nonnumeric"):
        TabularPreprocessor(examples.schema).fit(examples, np.array([0, 1, 2]))


def test_permutation_groups_keep_numeric_indicators_and_onehot_blocks_together(synthetic_data, tmp_path):
    import joblib

    frame, schema, _ = synthetic_data
    frame.loc[2, "steps"] = np.nan
    examples = prepare_tabular_examples(frame, schema)
    preprocessor = TabularPreprocessor(examples.schema).fit(examples, np.arange(len(examples)))
    names = preprocessor.feature_names["tabular"]
    groups = preprocessor.permutation_groups
    assert set(groups) == set(examples.tabular.columns)
    assert [names[index] for index in groups["steps"]] == ["numeric__steps", "numeric__missingindicator_steps"]
    assert [names[index] for index in groups["all_missing"]] == ["numeric__all_missing", "numeric__missingindicator_all_missing"]
    assert [names[index] for index in groups["sex"]] == ["categorical__sex_0", "categorical__sex_1"]
    assert sorted(index for indices in groups.values() for index in indices) == list(range(len(names)))
    assert preprocessor.export_metadata()["permutation_groups"] == groups
    # The mapping is derived from fitted state, including after serialization;
    # adding the method does not require refitting old experiment preprocessors.
    path = tmp_path / "preprocessor.joblib"
    joblib.dump(preprocessor, path)
    restored = joblib.load(path)
    assert restored.permutation_groups == groups


def test_forecasting_refuses_unverified_index_even_if_pattern_matches(synthetic_data):
    frame, schema, _ = synthetic_data
    with pytest.raises(ValueError, match="Forecasting is blocked"):
        prepare_forecasting_examples(frame, schema)
    with pytest.raises(ValueError, match="Forecasting is blocked"):
        prepare_forecasting_examples(frame, schema, ForecastConfig(verified_index_semantics=True))


def test_verified_synthetic_forecast_uses_exact_prior_months_and_cutoff_static_only(synthetic_data):
    frame, schema, _ = synthetic_data
    # Only the synthetic generator supplies verified timing here. No assertion
    # about the uploaded real-data index is made in this test or the implementation.
    config = ForecastConfig(True, "Synthetic fixture generator: explicit participant/month loop and known windows", verified_outcome_alignment=True, verified_feature_windows=True)
    after_cutoff = frame["__index_level_0__"].str.split("_").str[-1].astype(int) > 9
    frame.loc[after_cutoff, ["steps", "birthyear"]] = 1e9
    frame = frame.sample(frac=1, random_state=7).reset_index(drop=True)
    examples = prepare_forecasting_examples(frame, schema, config)
    assert examples.history_length == 7
    np.testing.assert_array_equal(examples.history_months, np.tile(np.arange(3, 10), (10, 1)))
    assert examples.temporal["steps"].max() < 1e9
    assert examples.static["birthyear"].max() < 1e9
    assert (examples.history_months.max(axis=1) == examples.outcome_months - 3).all()
    assert "phq9_score_start" not in examples.temporal
    assert "future_lr" not in examples.temporal
    transformed = TabularPreprocessor(examples.schema).fit(examples, np.arange(7)).transform(examples, np.arange(7, 10))
    assert transformed.temporal.shape[:2] == (3, 7)


def test_forecast_never_pads_missing_months_and_rejects_duplicate_pairs(synthetic_data):
    frame, schema, _ = synthetic_data
    config = ForecastConfig(True, "Synthetic fixture generator", verified_outcome_alignment=True, verified_feature_windows=True)
    incomplete = frame[frame["__index_level_0__"] != "0_4"].reset_index(drop=True)
    examples = prepare_forecasting_examples(incomplete, schema, config)
    assert len(examples) == 9
    assert "0" not in examples.groups
    assert examples.eligibility_counts["incomplete_history"] == 1
    duplicates = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate participant-month"):
        prepare_forecasting_examples(duplicates, schema, config)


def test_target_range_and_duplicate_group_keys_are_checked(synthetic_data):
    frame, schema, _ = synthetic_data
    frame.loc[2, "phq9_score_end"] = 28
    with pytest.raises(ValueError, match="documented range"):
        prepare_tabular_examples(frame, schema)
    frame.loc[2, "phq9_score_end"] = 5
    with pytest.raises(ValueError, match="Duplicate composite indices"):
        prepare_tabular_examples(pd.concat([frame, frame.iloc[[0]]], ignore_index=True), schema)
