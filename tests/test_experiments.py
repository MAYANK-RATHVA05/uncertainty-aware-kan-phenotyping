import json
import shutil

import numpy as np
import pandas as pd
import pytest

from kan_phenotyping.evaluation.evaluate import evaluate_saved_run
from kan_phenotyping.experiments.common import load_run
from kan_phenotyping.experiments.summarize_runs import canonical_model_spec, summarize_runs
from kan_phenotyping.experiments.train_depression import main as train


def test_synthetic_end_to_end_cli_and_fingerprint_rejection(tmp_path):
    dictionary = pd.DataFrame([
        ("steps", "Steps", "Statistics", "Mean steps"),
        ("sex", "Demographic", "Static", "Sex"),
        ("phq9_score_start", "PHQ-9", "PHQ-9 score", "Start score"),
        ("phq9_score_end", "PHQ-9", "PHQ-9 score", "End score"),
        ("phq9_cat_start", "PHQ-9", "PHQ-9 category", "Start category"),
        ("phq9_cat_end", "PHQ-9", "PHQ-9 category", "End category"),
    ], columns=["Feature name", "Category", "Subcategory", "Description"])
    dictionary["Notes"] = ""
    dictionary_path = tmp_path / "dictionary.tsv"
    dictionary.to_csv(dictionary_path, index=False, sep="\t")
    frame = pd.DataFrame([
        {"__index_level_0__": f"{group}_{row}", "steps": group + row / 10,
         "sex": group % 2, "phq9_score_start": 5, "phq9_score_end": 4 + group % 3,
         "phq9_cat_start": 0, "phq9_cat_end": 0}
        for group in range(16) for row in range(3)
    ])
    data = tmp_path / "cohort.csv"
    frame.to_csv(data, index=False)
    configuration = tmp_path / "config.yaml"
    configuration.write_text("hidden_width: 4\ntraining:\n  max_epochs: 2\n  num_threads: 1\n")
    run = tmp_path / "run"
    trained = train(["--data", str(data), "--dictionary", str(dictionary_path),
                     "--config", str(configuration), "--output", str(run)])
    restored = evaluate_saved_run(run, data, dictionary_path)
    assert restored == trained
    metadata = json.loads((run / "run.json").read_text())
    assert metadata["model"]["kwargs"]["spline_degree"] == 3
    assert metadata["model"]["kwargs"]["grid_intervals"] == 8
    assert metadata["dataset"]["temporal_semantics_verified"] is False
    prepared = load_run(run, data, dictionary_path)
    assert set(prepared.examples.groups[prepared.split.train]).isdisjoint(
        prepared.examples.groups[prepared.split.test])
    frame.loc[0, "steps"] = 1e10
    frame.to_csv(data, index=False)
    with pytest.raises(ValueError, match="fingerprint"):
        load_run(run, data, dictionary_path)


def test_seed_summary_checks_frozen_protocol_and_sample_std(tmp_path):
    paths = []
    for seed, value in [(42, 1.0), (43, 2.0), (44, 3.0)]:
        root = tmp_path / str(seed)
        root.mkdir()
        paths.append(root)
        info = {"inputs": {"data_sha256": "data", "dictionary_sha256": "dictionary"},
                "training": {"seed": seed, "max_epochs": 2}, "target": "continuous_change",
                "model": {"name": "fixed"}}
        (root / "run.json").write_text(json.dumps(info))
        (root / "metrics.json").write_text(json.dumps({
            part: {name: value for name in ("mae", "rmse", "r2")}
            for part in ("validation", "test")}))
        if seed == 42:
            np.savez(root / "splits.npz", train=[1, 2], validation=[3], test=[4])
        else:
            shutil.copyfile(paths[0] / "splits.npz", root / "splits.npz")
    result = summarize_runs(paths)
    assert result["test"]["rmse"] == {"mean": 2.0, "std": 1.0}
    altered = json.loads((paths[-1] / "run.json").read_text())
    altered["model"] = {"name": "different_architecture"}
    (paths[-1] / "run.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="model"):
        summarize_runs(paths)
    altered["model"] = {"name": "fixed"}
    (paths[-1] / "run.json").write_text(json.dumps(altered))
    np.savez(paths[-1] / "splits.npz", train=[2, 3], validation=[1], test=[4])
    with pytest.raises(ValueError, match="frozen splits"):
        summarize_runs(paths)


def test_canonical_architecture_normalizes_only_equal_defaults():
    implicit = {"name": "tabular_kan", "input_kind": "tabular",
                "kwargs": {"n_features": 111, "hidden_width": 64, "task": "regression"}}
    explicit = {**implicit, "kwargs": {**implicit["kwargs"], "grid_intervals": 8,
                "spline_degree": 3, "grid_range": [-3.0, 3.0], "mixing_type": "kan"}}
    assert canonical_model_spec(implicit) == canonical_model_spec(explicit)
    changed = {**explicit, "kwargs": {**explicit["kwargs"], "grid_intervals": 4}}
    assert canonical_model_spec(implicit) != canonical_model_spec(changed)
    changed = {**explicit, "kwargs": {**explicit["kwargs"], "hidden_width": 32}}
    assert canonical_model_spec(implicit) != canonical_model_spec(changed)
