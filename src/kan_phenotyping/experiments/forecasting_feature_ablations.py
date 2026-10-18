"""Controlled Ridge input comparisons on frozen conditional forecasting data.

Select complete original-feature groups from fitted preprocessing metadata.
No cohort rebuilding, preprocessing fitting, or reserved-payload access occurs.
The compatible original full Ridge result is reused after reproduction checks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import shutil
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import Ridge

from ..data.conditional_forecasting import BASELINE_DEMOGRAPHIC_FEATURES, WEARABLE_STATISTICS_FEATURES
from .build_forecasting_dataset import _output_directory, sha256_file
from .forecasting_baselines import ALPHAS, SavedBaselineData, load_baseline_data, regression_metrics


SETUPS = {
    "A": ("Measured SM0 PHQ-9 only", {"SM0"}),
    "B": ("SM0 plus enrollment demographics", {"SM0", "demographics"}),
    "C": ("SM0 plus demographics plus SM2 wearables", {"SM0", "demographics", "SM2"}),
    "D": ("SM0 plus demographics plus SM1 and SM2 wearables", {"SM0", "demographics", "SM1", "SM2"}),
}
REPRODUCTION_ATOL = 1e-10


def _expanded_groups(transformer, original_names: tuple[str, ...]) -> list[tuple[str, list[int]]]:
    """Map fitted output positions to original fields, never parse name strings."""
    if list(transformer.feature_names_in_) != list(original_names):
        raise ValueError("Fitted raw feature names/order differ from the reviewed preprocessing metadata.")
    result = {name: [] for name in original_names}
    for kind, pipeline, columns in transformer.transformers_:
        if kind == "remainder":
            if len(columns):
                raise ValueError("Unexpected remainder features cannot enter an ablation.")
            continue
        span = transformer.output_indices_[kind]
        if not isinstance(span, slice) or span.step not in (None, 1):
            raise ValueError("Fitted output metadata must contain contiguous pipeline spans.")
        if any(name not in result for name in columns):
            raise ValueError("A fitted pipeline includes an unreviewed original feature.")
        if kind == "numeric":
            imputer = pipeline.named_steps["imputer"]
            if not imputer.keep_empty_features:
                raise ValueError("Original numeric features must remain represented even if training values are all missing.")
            indicators = list(getattr(imputer.indicator_, "features_", ()))
            if span.stop - span.start != len(columns) + len(indicators):
                raise ValueError("Numeric value/indicator widths do not match fitted metadata.")
            for position, original in enumerate(columns):
                result[original].append(span.start + position)
            for position, owner in enumerate(indicators):
                if not 0 <= owner < len(columns):
                    raise ValueError("Missingness indicator has no valid original feature owner.")
                result[columns[owner]].append(span.start + len(columns) + position)
        elif kind == "categorical":
            encoder = pipeline.named_steps["encoder"]
            if encoder.min_frequency is not None or encoder.max_categories is not None:
                raise ValueError("Unexpected infrequent-category grouping requires a separate metadata mapping.")
            if len(encoder.categories_) != len(columns):
                raise ValueError("Categorical vocabulary does not match original features.")
            cursor = span.start
            for position, (original, categories) in enumerate(zip(columns, encoder.categories_)):
                dropped = None if encoder.drop_idx_ is None else encoder.drop_idx_[position]
                width = len(categories) - int(dropped is not None)
                result[original].extend(range(cursor, cursor + width))
                cursor += width
            if cursor != span.stop:
                raise ValueError("One-hot spans do not match the fitted output metadata.")
        else:
            raise ValueError("Only the original numeric/categorical preprocessing pipelines are allowed.")
    positions = [index for indices in result.values() for index in indices]
    if (any(not indices for indices in result.values())
            or sorted(positions) != list(range(len(transformer.get_feature_names_out())))):
        raise ValueError("Each expanded column must belong to exactly one complete original-feature group.")
    return list(result.items())


def original_feature_groups(preprocessor) -> list[dict[str, Any]]:
    """Describe all original fields, their month identities, and derived columns."""
    if (tuple(preprocessor.wearable_columns_) != WEARABLE_STATISTICS_FEATURES
            or tuple(preprocessor.demographic_columns_) != BASELINE_DEMOGRAPHIC_FEATURES):
        raise ValueError("Require the exact saved forecasting feature allowlist.")
    names = preprocessor.feature_names
    wearable = preprocessor.wearable_transformer.get_feature_names_out().tolist()
    demographics = preprocessor.demographic_transformer.get_feature_names_out().tolist()
    if names["wearable"] != wearable or names["demographics"] != demographics:
        raise ValueError("Transformed feature-name metadata disagrees with fitted preprocessing.")
    if names["sm0_phq"] != ["measured_sm0_phq9_standardized"]:
        raise ValueError("The final scalar must be the documented measured SM0 input.")
    expected = ([f"SM{month}__{name}" for month in (1, 2) for name in wearable]
                + [f"baseline_demographic__{name}" for name in demographics] + names["sm0_phq"])
    if names["tabular"] != expected or len(set(expected)) != len(expected):
        raise ValueError("Feature metadata must preserve exact saved month identity and order.")
    local_wearable = _expanded_groups(preprocessor.wearable_transformer, WEARABLE_STATISTICS_FEATURES)
    local_demographics = _expanded_groups(preprocessor.demographic_transformer, BASELINE_DEMOGRAPHIC_FEATURES)
    groups = []
    for block, offset, local in (
        ("SM1", 0, local_wearable), ("SM2", len(wearable), local_wearable),
        ("demographics", 2 * len(wearable), local_demographics),
        ("SM0", len(expected) - 1, [("phq9_score_start", [0])]),
    ):
        for original, indices in local:
            absolute = [offset + index for index in indices]
            groups.append({"block": block, "original_feature": original, "column_indices": absolute,
                           "feature_names": [expected[index] for index in absolute]})
    covered = [index for group in groups for index in group["column_indices"]]
    if len(groups) != 84 or sorted(covered) != list(range(len(expected))):
        raise ValueError("The complete reviewed 84 original-feature/month groups must partition all transformed columns.")
    return groups


def feature_set_definitions(preprocessor) -> dict[str, dict[str, Any]]:
    groups = original_feature_groups(preprocessor)
    names = preprocessor.feature_names["tabular"]
    definitions = {}
    for key, (label, blocks) in SETUPS.items():
        selected = [group for group in groups if group["block"] in blocks]
        indices = sorted(index for group in selected for index in group["column_indices"])
        definitions[key] = {"name": label, "column_indices": indices,
                            "feature_names": [names[index] for index in indices],
                            "original_feature_groups": selected}
    for smaller, larger in zip("ABC", "BCD"):
        if not set(definitions[smaller]["column_indices"]) < set(definitions[larger]["column_indices"]):
            raise ValueError("A, B, C, D must be strictly nested complete feature sets.")
    return definitions


@dataclass(frozen=True)
class RidgeAblationResult:
    model: Ridge
    models: list[Ridge]
    candidates: list[dict[str, Any]]
    selected_alpha: float
    metrics: dict[str, float]
    predictions: np.ndarray
    definition: dict[str, Any]


def fit_ridge_ablation(data: SavedBaselineData, definition: dict[str, Any]) -> RidgeAblationResult:
    """Subset columns only, fit training outcomes, and select using validation."""
    valid = feature_set_definitions(data.preprocessor)
    if definition not in valid.values():
        raise ValueError("Select a reviewed complete feature set; partial indicators/categories are not allowed.")
    indices = definition["column_indices"]
    width = len(data.preprocessor.feature_names["tabular"])
    for partition in (data.train, data.validation):
        n = len(partition.targets)
        if (partition.tabular.shape != (n, width) or partition.sample_indices.shape != (n,)
                or partition.participants.shape != (n,) or not np.isfinite(partition.tabular).all()):
            raise ValueError("Frozen predictor rows, targets, and sample indices must remain aligned.")
    # Keep the original full Ridge array layout for numerical reproduction.
    x_train = np.ascontiguousarray(data.train.tabular[:, indices], dtype=np.float64)
    x_validation = np.ascontiguousarray(data.validation.tabular[:, indices], dtype=np.float64)
    models, candidates, predictions = [], [], []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver="svd", fit_intercept=True, positive=False)
        model.fit(x_train, np.asarray(data.train.targets, dtype=np.float64))
        prediction = model.predict(x_validation)
        models.append(model)
        predictions.append(prediction)
        candidates.append({"alpha": alpha, "validation_metrics": regression_metrics(data.validation.targets, prediction)})
    selected = min(range(len(candidates)), key=lambda index: candidates[index]["validation_metrics"]["rmse"])
    return RidgeAblationResult(models[selected], models, candidates, ALPHAS[selected],
                               candidates[selected]["validation_metrics"], predictions[selected], definition)


def reuse_full_ridge(data: SavedBaselineData, baseline_dir: str | Path,
                    reproduced: RidgeAblationResult) -> tuple[RidgeAblationResult, dict[str, Any]]:
    """Retain the original D result after compatibility and all-five checks."""
    root = Path(baseline_dir).resolve()
    run = json.loads((root / "run.json").read_text())
    if (run["dataset_manifest_sha256"] != data.manifest_sha256
            or run["original_input_hashes"] != data.manifest["inputs"]):
        raise ValueError("Existing full Ridge must use the identical saved dataset and original input hashes.")
    consumed = ("config.json", "validation_metrics.json", "validation_predictions.npz",
                "models.joblib", "ridge_candidates.joblib")
    for name in consumed:
        if sha256_file(root / name) != run["artifact_sha256"][name]:
            raise ValueError("Existing full Ridge artifact hash mismatch.")
    config = json.loads((root / "config.json").read_text())
    ridge_config = config["ridge"]
    if (config["feature_names"] != data.preprocessor.feature_names
            or config["target"] != data.manifest["dataset"]["target"]
            or config["assumptions"] != data.manifest["dataset"]["assumptions"]
            or config["prediction_clipping"] is not False or config["target_transformation"] != "none"
            or any(ridge_config[key] != value for key, value in {
                "alphas": list(ALPHAS), "solver": "svd", "fit_intercept": True, "positive": False,
                "dtype": "float64", "final_refit": False, "fitted_partition": "train",
            }.items())):
        raise ValueError("Existing full Ridge configuration is incompatible with the controlled comparisons.")
    report = json.loads((root / "validation_metrics.json").read_text())
    with np.load(root / "validation_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["sample_indices"], data.validation.sample_indices)
        np.testing.assert_array_equal(saved["measured_sm3"], data.validation.targets)
        original_prediction = saved["ridge"].copy()
    models = joblib.load(root / "ridge_candidates.joblib")
    selected_model = joblib.load(root / "models.joblib")["ridge"]
    if (len(models) != len(ALPHAS) or report["selected_alpha"] != reproduced.selected_alpha
            or ridge_config["selected_alpha"] != reproduced.selected_alpha):
        raise ValueError("Full Ridge alpha selection does not reproduce.")
    differences = []
    for alpha, cached, fresh, record in zip(ALPHAS, models, reproduced.models, report["ridge_candidates"], strict=True):
        if cached.alpha != alpha or record["alpha"] != alpha:
            raise ValueError("Full Ridge candidate grid/order differs.")
        np.testing.assert_allclose(cached.coef_, fresh.coef_, rtol=0, atol=REPRODUCTION_ATOL)
        np.testing.assert_allclose(cached.intercept_, fresh.intercept_, rtol=0, atol=REPRODUCTION_ATOL)
        prediction = cached.predict(data.validation.tabular)
        for metric, value in regression_metrics(data.validation.targets, prediction).items():
            np.testing.assert_allclose(value, record["validation_metrics"][metric], rtol=0, atol=REPRODUCTION_ATOL)
        np.testing.assert_allclose(prediction, fresh.predict(data.validation.tabular), rtol=0, atol=REPRODUCTION_ATOL)
        differences.append(float(np.max(np.abs(prediction - fresh.predict(data.validation.tabular)))))
    np.testing.assert_array_equal(selected_model.predict(data.validation.tabular), original_prediction)
    np.testing.assert_allclose(original_prediction, reproduced.predictions, rtol=0, atol=REPRODUCTION_ATOL)
    for metric, value in report["metrics"]["ridge"].items():
        np.testing.assert_allclose(value, reproduced.metrics[metric], rtol=0, atol=REPRODUCTION_ATOL)
    result = RidgeAblationResult(selected_model, models, report["ridge_candidates"], report["selected_alpha"],
                                 report["metrics"]["ridge"], original_prediction, reproduced.definition)
    verification = {
        "result_origin": "reused compatible existing full Ridge after independent reproduction",
        "baseline_dir": str(root), "original_run_sha256": sha256_file(root / "run.json"),
        "all_five_candidates_reproduced": True, "selected_alpha_reproduced": True,
        "sample_indices_and_targets_identical": True, "absolute_tolerance": REPRODUCTION_ATOL,
        "max_candidate_prediction_difference": max(differences),
        "selected_prediction_max_difference": float(np.max(np.abs(original_prediction - reproduced.predictions))),
        "consumed_sha256": {name: sha256_file(root / name) for name in ("run.json", *consumed)},
        "original_artifacts_replaced": False,
    }
    return result, verification


def run_feature_ablations(dataset_dir: str | Path, baseline_dir: str | Path,
                         output_dir: str | Path) -> dict[str, Any]:
    output = _output_directory(output_dir)
    data = load_baseline_data(dataset_dir)
    definitions = feature_set_definitions(data.preprocessor)
    preprocessor_path = data.dataset_dir / "preprocessor.joblib"
    preprocessing_hash = sha256_file(preprocessor_path)
    results = {key: fit_ridge_ablation(data, definition) for key, definition in definitions.items()}
    results["D"], reproduction = reuse_full_ridge(data, baseline_dir, results["D"])
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for key, result in results.items():
        directory = output / key
        directory.mkdir()
        report = {"name": result.definition["name"], "selected_alpha": result.selected_alpha,
                  "metrics": result.metrics, "candidates": result.candidates,
                  "feature_count": len(result.definition["column_indices"]),
                  "original_feature_groups": len(result.definition["original_feature_groups"]),
                  "result_origin": "reused_existing_full_ridge" if key == "D" else "new_training_only_fit"}
        (directory / "validation_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (directory / "features.json").write_text(json.dumps(result.definition, indent=2) + "\n")
        joblib.dump(result.model, directory / "selected_ridge.joblib")
        joblib.dump(result.models, directory / "ridge_candidates.joblib")
        np.savez_compressed(directory / "validation_predictions.npz", sample_indices=data.validation.sample_indices,
                            measured_sm3=data.validation.targets, prediction=result.predictions)
        reports[key] = report
    (output / "D" / "reproduction.json").write_text(json.dumps(reproduction, indent=2) + "\n")
    increments = {
        f"{later}_minus_{earlier}": {metric: reports[later]["metrics"][metric] - reports[earlier]["metrics"][metric]
                                    for metric in ("mae", "rmse", "r2")}
        for earlier, later in zip("ABC", "BCD")
    }
    summary = {
        "study_type": "conditional_relative_month_forecasting", "setups": reports, "increments": increments,
        "delta_definition": "later setup minus earlier; negative MAE/RMSE and positive R2 favor added inputs",
        "counts": {name: {"samples": len(part.targets), "participants": len(np.unique(part.participants))}
                   for name, part in (("train", data.train), ("validation", data.validation))},
        "comparison_type": "exploratory predictive comparison; not a causal effect or significance test",
    }
    config = {
        "workflow": "conditional_relative_month_ridge_feature_ablations",
        "alpha_grid": list(ALPHAS), "selection": "minimum validation RMSE; first ascending alpha on exact ties",
        "ridge": {"solver": "svd", "fit_intercept": True, "positive": False, "dtype": "float64"},
        "fit_partition": "train_only", "refit_on_validation": False,
        "preprocessing": "unchanged original training-fitted artifact; subset existing transformed columns only",
        "target": data.manifest["dataset"]["target"], "target_transformation": "none", "prediction_clipping": False,
        "assumptions": data.manifest["dataset"]["assumptions"], "horizon": data.manifest["dataset"]["horizon"],
        "eligibility": data.manifest["dataset"]["eligibility"], "split_configuration": data.manifest["split"],
        "original_full_feature_names": data.preprocessor.feature_names,
        "calibration_or_test_payloads_consumed": False, "available_at_timestamps_verified": False,
        "selection_unit": "complete original-feature/month groups including missingness indicators and all one-hot columns",
    }
    for name, value in (("summary.json", summary), ("config.json", config),
                        ("feature_groups.json", original_feature_groups(data.preprocessor))):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "used_split_indices.npz", train=data.train.sample_indices,
                        validation=data.validation.sample_indices)
    if sha256_file(preprocessor_path) != preprocessing_hash:
        raise RuntimeError("Saved training-only preprocessing changed.")
    shutil.copyfile(preprocessor_path, output / "preprocessor.joblib")
    provenance = {
        "dataset_dir": str(data.dataset_dir), "dataset_manifest_sha256": data.manifest_sha256,
        "dataset_files_consumed": list(data.read_files),
        "consumed_sha256": {name: sha256_file(data.dataset_dir / name) for name in data.read_files},
        "original_input_hashes": data.manifest["inputs"], "builder_source_hashes": data.manifest["source_sha256"],
        "preprocessor_sha256": preprocessing_hash, "full_ridge_reproduction": reproduction,
        "source_sha256": {Path(__file__).name: sha256_file(__file__),
                          "forecasting_baselines.py": sha256_file(Path(__file__).with_name("forecasting_baselines.py"))},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "thread_environment": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "artifact_sha256": {str(path.relative_to(output)): sha256_file(path) for path in sorted(output.rglob("*")) if path.is_file()},
    }
    (output / "run.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_feature_ablations(args.dataset_dir, args.baseline_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
