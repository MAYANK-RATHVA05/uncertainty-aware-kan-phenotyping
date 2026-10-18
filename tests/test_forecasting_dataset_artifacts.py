"""End-to-end checks of conditional data artifacts; no model is trained."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from kan_phenotyping.data.conditional_forecasting import (
    BASELINE_DEMOGRAPHIC_FEATURES,
    WEARABLE_STATISTICS_FEATURES,
    ConditionalForecastAssumptions,
    build_conditional_forecasting_dataset,
)
from kan_phenotyping.data.forecasting_splits import PARTITION_NAMES
from kan_phenotyping.experiments import build_forecasting_dataset as artifact_module


@pytest.fixture
def artifact_sources(tmp_path):
    rows = []
    for person in range(10):
        for month in range(1, 7):
            quarter = (month - 1) // 3
            endpoint = month % 3 == 0
            row = {
                "__index_level_0__": f"{1000 + person}_{month}",
                "phq9_score_start": 2 + person + quarter * 2 if endpoint else np.nan,
                "phq9_score_end": 15 + person - quarter if endpoint else np.nan,
            }
            row.update({
                name: float(person * 10000 + month * 100 + feature)
                for feature, name in enumerate(WEARABLE_STATISTICS_FEATURES)
            })
            row.update({
                name: float((person + feature) % 3)
                for feature, name in enumerate(BASELINE_DEMOGRAPHIC_FEATURES)
            })
            rows.append(row)
    frame = pd.DataFrame(rows).sample(frac=1, random_state=19).reset_index(drop=True)
    frame.loc[frame["__index_level_0__"] == "1000_1", WEARABLE_STATISTICS_FEATURES[0]] = np.nan
    specs = [
        (name, "Steps" if name.startswith("steps_") else "Sleep", "Statistics", f"Observed {name}", "")
        for name in WEARABLE_STATISTICS_FEATURES
    ] + [
        (name, "Demographic", "Static", f"Enrollment {name}", "")
        for name in BASELINE_DEMOGRAPHIC_FEATURES
    ] + [
        ("phq9_score_start", "PHQ-9", "PHQ-9 score", "PHQ-9 at start of three-month period", "0-27"),
        ("phq9_score_end", "PHQ-9", "PHQ-9 score", "PHQ-9 at end of three-month period", "0-27"),
    ]
    dictionary = pd.DataFrame(specs, columns=["Feature name", "Category", "Subcategory", "Description", "Notes"])
    data_path = tmp_path / "input.csv"
    dictionary_path = tmp_path / "dictionary.tsv"
    frame.to_csv(data_path, index=False)
    dictionary.to_csv(dictionary_path, sep="\t", index=False)
    assumptions = ConditionalForecastAssumptions(
        True, True, True, True, "Known synthetic study-month generator; no available-at timestamp claim."
    )
    return frame, dictionary, data_path, dictionary_path, assumptions


@pytest.fixture
def saved_artifacts(artifact_sources, tmp_path):
    frame, dictionary, data, dictionary_path, assumptions = artifact_sources
    output = tmp_path / "prepared"
    manifest = artifact_module.prepare_forecasting_dataset(data, dictionary_path, output, assumptions, seed=17)
    dataset = build_conditional_forecasting_dataset(frame, dictionary, assumptions)
    return output, manifest, dataset


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_artifacts_separate_targets_and_reconstruct_consistent_samples(saved_artifacts, artifact_sources):
    output, manifest, dataset = saved_artifacts
    _, _, data, dictionary, _ = artifact_sources
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert manifest["model_training_performed"] is False
    assert manifest["conformal_analysis_performed"] is False
    assert manifest["dataset"]["available_at_timestamps_verified"] is False
    assert manifest["dataset"]["study_type"] == "conditional_relative_month_forecasting"
    mappings = _npz(output / "sample_index.npz")
    assert set(mappings) == {
        "participants", "quarters", "endpoint_months", "history_row_positions",
        "baseline_row_positions", "target_row_positions",
    }
    for name, values in mappings.items():
        np.testing.assert_array_equal(values, getattr(dataset, name))
    splits = _npz(output / "splits.npz")
    assert set(splits) == set(PARTITION_NAMES)
    np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(dataset)))
    people = {name: set(mappings["participants"][indices]) for name, indices in splits.items()}
    assert sum(map(len, people.values())) == len(set(mappings["participants"]))
    assert all(people[left].isdisjoint(people[right])
               for index, left in enumerate(PARTITION_NAMES) for right in PARTITION_NAMES[index + 1:])
    preprocessor = joblib.load(output / "preprocessor.joblib")
    assert preprocessor.n_training_samples_ == len(splits["train"])
    assert preprocessor.n_training_wearable_rows_ == 2 * len(splits["train"])
    assert preprocessor.baseline_scaler.n_samples_seen_ == len(splits["train"])
    np.testing.assert_allclose(
        preprocessor.baseline_scaler.mean_, dataset.predictors.sm0_phq[splits["train"]].mean(axis=0)
    )
    for name, indices in splits.items():
        predictors = _npz(output / f"{name}_inputs.npz")
        targets = _npz(output / f"{name}_targets.npz")
        assert set(predictors) == {"wearable", "demographics", "sm0_phq", "relative_months", "study_months", "tabular"}
        assert set(targets) == {"sm3_phq"}
        assert targets["sm3_phq"].dtype == np.float32
        np.testing.assert_array_equal(targets["sm3_phq"], dataset.targets[indices])
        restored = preprocessor.transform(dataset.predictors, indices)
        for key, values in predictors.items():
            assert values.dtype != object
            np.testing.assert_array_equal(values, getattr(restored, key))
            assert list(values.shape) == manifest["tensor_shapes"][name][key]
        assert manifest["split"][name]["samples"] == len(indices)
        assert manifest["split"][name]["participants"] == len(people[name])
        np.testing.assert_array_equal(predictors["relative_months"], np.tile([1, 2], (len(indices), 1)))
    assert manifest["inputs"]["data"]["sha256"] == _digest(data)
    assert manifest["inputs"]["dictionary"]["sha256"] == _digest(dictionary)
    assert set(manifest["artifact_sha256"]) == {path.name for path in output.iterdir()} - {"manifest.json"}
    for filename, digest in manifest["artifact_sha256"].items():
        assert digest == _digest(output / filename)
    source_root = Path(artifact_module.__file__).resolve().parents[1]
    assert set(manifest["source_sha256"]) == {
        "build_forecasting_dataset.py", "conditional_forecasting.py", "forecasting_splits.py",
        "forecasting_preprocessing.py", "preprocessing.py", "dataset.py",
    }
    for filename, digest in manifest["source_sha256"].items():
        path = Path(artifact_module.__file__) if filename == "build_forecasting_dataset.py" else source_root / "data" / filename
        assert digest == _digest(path)


def test_held_out_mutations_do_not_change_saved_training_inputs_or_fitted_statistics(
    artifact_sources, saved_artifacts, tmp_path
):
    frame, _, _, dictionary_path, assumptions = artifact_sources
    output, _, _ = saved_artifacts
    mappings = _npz(output / "sample_index.npz")
    splits = _npz(output / "splits.npz")
    held_out_people = set(mappings["participants"][np.concatenate([splits[name] for name in PARTITION_NAMES[1:]])])
    changed = frame.copy(deep=True)
    held_out = changed["__index_level_0__"].str.rsplit("_", n=1).str[0].isin(held_out_people)
    changed.loc[held_out, list(WEARABLE_STATISTICS_FEATURES)] = 1e9
    changed.loc[held_out, list(BASELINE_DEMOGRAPHIC_FEATURES)] = 999
    endpoints = changed["phq9_score_start"].notna()
    changed.loc[held_out & endpoints, "phq9_score_start"] = 27
    changed_path = tmp_path / "changed.csv"
    changed.to_csv(changed_path, index=False)
    changed_output = tmp_path / "changed_artifacts"
    artifact_module.prepare_forecasting_dataset(changed_path, dictionary_path, changed_output, assumptions, seed=17)
    before = joblib.load(output / "preprocessor.joblib")
    after = joblib.load(changed_output / "preprocessor.joblib")
    assert before.feature_names == after.feature_names
    for transformer_name in ("wearable_transformer", "demographic_transformer"):
        original = getattr(before, transformer_name).named_transformers_["numeric"]
        rebuilt = getattr(after, transformer_name).named_transformers_["numeric"]
        np.testing.assert_array_equal(original.named_steps["imputer"].statistics_, rebuilt.named_steps["imputer"].statistics_)
        np.testing.assert_array_equal(original.named_steps["scaler"].mean_, rebuilt.named_steps["scaler"].mean_)
        np.testing.assert_array_equal(original.named_steps["scaler"].scale_, rebuilt.named_steps["scaler"].scale_)
    np.testing.assert_array_equal(before.baseline_scaler.mean_, after.baseline_scaler.mean_)
    np.testing.assert_array_equal(before.baseline_scaler.scale_, after.baseline_scaler.scale_)
    for name, values in _npz(output / "train_inputs.npz").items():
        np.testing.assert_array_equal(values, _npz(changed_output / "train_inputs.npz")[name])
    for name in PARTITION_NAMES:
        np.testing.assert_array_equal(splits[name], _npz(changed_output / "splits.npz")[name])
        np.testing.assert_array_equal(
            _npz(output / f"{name}_targets.npz")["sm3_phq"],
            _npz(changed_output / f"{name}_targets.npz")["sm3_phq"],
        )


def test_same_input_and_seed_repeat_all_saved_arrays(saved_artifacts, artifact_sources, tmp_path):
    output, manifest, _ = saved_artifacts
    _, _, data, dictionary, assumptions = artifact_sources
    second = tmp_path / "repeat"
    repeated = artifact_module.prepare_forecasting_dataset(data, dictionary, second, assumptions, seed=17)
    assert manifest["split"] == repeated["split"]
    assert manifest["preprocessing"] == repeated["preprocessing"]
    assert manifest["tensor_shapes"] == repeated["tensor_shapes"]
    assert manifest["inputs"] == repeated["inputs"]
    for source_path in output.glob("*.npz"):
        first = _npz(source_path)
        again = _npz(second / source_path.name)
        assert first.keys() == again.keys()
        for name, values in first.items():
            np.testing.assert_array_equal(values, again[name])


@pytest.mark.parametrize("partial", [False, True])
def test_artifact_creation_requires_explicit_conditional_assumptions(artifact_sources, tmp_path, partial):
    _, _, data, dictionary, valid = artifact_sources
    assumptions = replace(valid, assume_sm0_phq_available_by_sm2=False) if partial else ConditionalForecastAssumptions()
    destination = tmp_path / "not_created"
    with pytest.raises(ValueError, match="explicit opt-in"):
        artifact_module.prepare_forecasting_dataset(data, dictionary, destination, assumptions)
    assert not destination.exists()


@pytest.mark.parametrize("existing_file", [False, True])
def test_existing_artifacts_cannot_be_overwritten(artifact_sources, tmp_path, existing_file):
    _, _, data, dictionary, assumptions = artifact_sources
    destination = tmp_path / "keep"
    if existing_file:
        destination.write_text("existing user file")
    else:
        destination.mkdir()
        (destination / "sentinel").write_text("existing user artifact")
    before = destination.read_bytes() if existing_file else (destination / "sentinel").read_bytes()
    with pytest.raises(ValueError, match="new or empty"):
        artifact_module.prepare_forecasting_dataset(data, dictionary, destination, assumptions)
    assert (destination.read_bytes() if existing_file else (destination / "sentinel").read_bytes()) == before
    if not existing_file:
        assert [path.name for path in destination.iterdir()] == ["sentinel"]


def test_artifact_writer_refuses_a_source_directory(artifact_sources, tmp_path):
    _, _, data, dictionary, assumptions = artifact_sources
    checkout = Path(artifact_module.__file__).resolve().parents[3]
    destination = checkout / "docs" / f"_rejected_forecast_{tmp_path.name}"
    assert not destination.exists()
    with pytest.raises(ValueError, match="outside tracked source"):
        artifact_module.prepare_forecasting_dataset(data, dictionary, destination, assumptions)
    assert not destination.exists()


def test_cli_requires_acknowledgment_before_reading_or_writing(tmp_path, capsys):
    destination = tmp_path / "not_created"
    with pytest.raises(SystemExit) as error:
        artifact_module.main([
            "--data", str(tmp_path / "absent.csv"), "--dictionary", str(tmp_path / "absent.tsv"),
            "--output", str(destination),
        ])
    assert error.value.code == 2
    assert "--acknowledge-conditional-assumptions" in capsys.readouterr().err
    assert not destination.exists()
