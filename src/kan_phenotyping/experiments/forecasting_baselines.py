"""Validation-only baselines for conditional relative-month PHQ-9 severity.

Consume an existing dataset bundle, never recreate its splits or eligibility.
Calibration/test observations and outcomes are neither opened nor evaluated.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..data.conditional_forecasting import (
    BASELINE_DEMOGRAPHIC_FEATURES, WEARABLE_STATISTICS_FEATURES,
    ConditionalForecastAssumptions,
)
from ..data.forecasting_preprocessing import ConditionalForecastPreprocessor
from ..data.forecasting_splits import ForecastPartitions, PARTITION_NAMES
from .build_forecasting_dataset import _output_directory, sha256_file


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
_INPUT_KEYS = {"wearable", "demographics", "sm0_phq", "relative_months", "study_months", "tabular"}
_CONSUMED_FILES = (
    "splits.npz", "sample_index.npz", "preprocessor.joblib",
    "train_inputs.npz", "train_targets.npz", "validation_inputs.npz", "validation_targets.npz",
)


def _measured_scores(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if (array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all()
            or not ((array >= 0) & (array <= 27)).all()
            or not np.equal(array, np.rint(array)).all()):
        raise ValueError(f"{name} must be observed integer PHQ-9 severities in 0..27.")
    return array.copy()


def recover_measured_sm0(standardized: np.ndarray, baseline_scaler: Any) -> np.ndarray:
    """Undo saved scaling and reconstruct the original integer input exactly.

    The builder stores standardized float32 values, so inverse scaling alone
    has tiny roundoff. Require proximity to a measured integer AND exact
    restandardization to the saved values before accepting that integer.
    This reconstructs an input; it never rounds or clips model predictions.
    """
    saved = np.asarray(standardized)
    if saved.ndim != 2 or saved.shape[1] != 1 or not np.isfinite(saved).all():
        raise ValueError("Saved SM0 must be a finite [N, 1] standardized input.")
    inverse = baseline_scaler.inverse_transform(saved.astype(np.float64))
    measured = np.rint(inverse)
    if not np.allclose(inverse, measured, rtol=0, atol=1e-4):
        raise ValueError("Saved SM0 does not reconstruct a measured integer score.")
    result = _measured_scores(measured, "Measured SM0")
    roundtrip = baseline_scaler.transform(result[:, None]).astype(saved.dtype)
    if not np.array_equal(roundtrip, saved):
        raise ValueError("Measured SM0 reconstruction does not reproduce the saved standardized input.")
    return result


class PersistenceBaseline:
    """Predict exactly the measured SM0 severity; no fitted parameters."""

    def predict(self, sm0_phq: np.ndarray) -> np.ndarray:
        return _measured_scores(sm0_phq, "Persistence input SM0")


class TrainingMeanBaseline:
    """One constant learned only from the training SM3 outcomes."""

    def fit(self, y_train: np.ndarray) -> "TrainingMeanBaseline":
        measured = _measured_scores(y_train, "Training SM3")
        self.mean_ = float(measured.mean(dtype=np.float64))
        self.n_training_samples_ = len(measured)
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("Fit the training mean before predicting.")
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 0:
            raise ValueError("Prediction count must be a nonnegative integer.")
        return np.full(n_samples, self.mean_, dtype=np.float64)


def regression_metrics(y: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if (truth.ndim != 1 or len(truth) < 2 or truth.shape != predicted.shape
            or not np.isfinite(truth).all() or not np.isfinite(predicted).all()):
        raise ValueError("Metrics require aligned finite outcome/prediction vectors with at least two samples.")
    return {"mae": float(mean_absolute_error(truth, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
            "r2": float(r2_score(truth, predicted))}


@dataclass(frozen=True)
class BaselinePartition:
    sample_indices: np.ndarray
    participants: np.ndarray
    sm0_phq: np.ndarray
    tabular: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class SavedBaselineData:
    train: BaselinePartition
    validation: BaselinePartition
    preprocessor: ConditionalForecastPreprocessor
    manifest: dict[str, Any]
    dataset_dir: Path
    read_files: tuple[str, ...]
    manifest_sha256: str


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def load_baseline_data(dataset_dir: str | Path) -> SavedBaselineData:
    """Verify and load only train/validation observations plus shared metadata.

    Raw CSV input hashes are inherited, not re-read: loading the complete CSV
    would unnecessarily reopen reserved outcomes. Reserved partition IDs are
    read only for checking assignment/group separation, never as predictors.
    """
    root = Path(dataset_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest.get("dataset", {})
    if (manifest.get("format_version") != 1
            or protocol.get("study_type") != "conditional_relative_month_forecasting"
            or protocol.get("target") != "measured SM3 PHQ-9 severity (phq9_score_end), not change"
            or protocol.get("available_at_timestamps_verified") is not False
            or protocol.get("prospective_clinical_validation") is not False):
        raise ValueError("Require the existing conditional relative-month measured-severity dataset bundle.")
    ConditionalForecastAssumptions(**protocol["assumptions"]).validate()
    allowlist = protocol["feature_allowlist"]
    if (allowlist["wearable_sm1_sm2"] != list(WEARABLE_STATISTICS_FEATURES)
            or allowlist["enrollment_demographics_from_sm1"] != list(BASELINE_DEMOGRAPHIC_FEATURES)
            or allowlist["measured_sm0_scalar"] != "phq9_score_start"):
        raise ValueError("Dataset feature allowlist differs from the reviewed forecasting builder.")
    # Check only consumed file contents. Never hash or open reserved X/Y files.
    for name in _CONSUMED_FILES:
        if sha256_file(root / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Saved artifact hash mismatch: {name}.")
    source_root = Path(__file__).resolve().parents[1]
    expected_sources = {"build_forecasting_dataset.py": Path(__file__).with_name("build_forecasting_dataset.py")}
    expected_sources.update({name: source_root / "data" / name for name in (
        "conditional_forecasting.py", "forecasting_splits.py", "forecasting_preprocessing.py",
        "preprocessing.py", "dataset.py",
    )})
    if set(manifest["source_sha256"]) != set(expected_sources):
        raise ValueError("Saved builder provenance does not match the expected source modules.")
    if any(sha256_file(path) != manifest["source_sha256"][name] for name, path in expected_sources.items()):
        raise ValueError("Builder or preprocessing source differs from the saved dataset provenance.")
    splits = _npz(root / "splits.npz")
    mapping = _npz(root / "sample_index.npz")
    if set(splits) != set(PARTITION_NAMES):
        raise ValueError("Reuse all four saved partition assignments without creating new splits.")
    partitions = ForecastPartitions(**splits)
    partitions.validate(mapping["participants"])
    for name in PARTITION_NAMES:
        indices = splits[name]
        if manifest["split"][name] != {
            "samples": len(indices), "participants": len(np.unique(mapping["participants"][indices])),
        }:
            raise ValueError("Saved partition counts disagree with the original manifest.")
    preprocessor = joblib.load(root / "preprocessor.joblib")
    if (not isinstance(preprocessor, ConditionalForecastPreprocessor)
            or preprocessor.export_metadata() != manifest["preprocessing"]
            or preprocessor.n_training_samples_ != len(splits["train"])
            or preprocessor.n_training_wearable_rows_ != 2 * len(splits["train"])
            or preprocessor.baseline_scaler.n_samples_seen_ != len(splits["train"])):
        raise ValueError("Saved preprocessing does not match the original training-only fit.")
    meta = preprocessor.export_metadata()
    if (meta["fitted_partition"] != "training_only"
            or any(meta[key] is not False for key in (
                "validation_used_for_fit", "calibration_used_for_fit", "test_used_for_fit", "targets_used_for_fit",
            ))):
        raise ValueError("Preprocessing must have been fitted on training predictors only.")
    names = preprocessor.feature_names
    expected_names = ([f"SM{month}__{name}" for month in (1, 2) for name in names["wearable"]]
                      + [f"baseline_demographic__{name}" for name in names["demographics"]] + names["sm0_phq"])
    if names["tabular"] != expected_names or len(set(expected_names)) != len(expected_names):
        raise ValueError("Flattened feature names must retain distinct SM1 and SM2 identities.")
    loaded = {}
    for name in ("train", "validation"):
        arrays = _npz(root / f"{name}_inputs.npz")
        targets = _npz(root / f"{name}_targets.npz")
        indices = splits[name]
        n = len(indices)
        if set(arrays) != _INPUT_KEYS or set(targets) != {"sm3_phq"}:
            raise ValueError("Unexpected predictor/target storage schema.")
        expected_shapes = {
            "wearable": (n, 2, len(names["wearable"])),
            "demographics": (n, len(names["demographics"])), "sm0_phq": (n, 1),
            "relative_months": (n, 2), "study_months": (n, 2),
            "tabular": (n, len(expected_names)),
        }
        if any(arrays[key].shape != shape for key, shape in expected_shapes.items()):
            raise ValueError("Saved predictors are not aligned with partition indices/feature names.")
        if (not np.array_equal(arrays["relative_months"], np.tile([1, 2], (n, 1)))
                or not np.array_equal(arrays["study_months"], mapping["quarters"][indices, None] * 3 + [1, 2])
                or not np.array_equal(mapping["endpoint_months"][indices], mapping["quarters"][indices] * 3 + 3)):
            raise ValueError("History positions must retain same-quarter SM1 and SM2.")
        flattened = np.concatenate([
            arrays["wearable"].reshape(n, -1), arrays["demographics"], arrays["sm0_phq"],
        ], axis=1)
        if not np.array_equal(flattened, arrays["tabular"]) or not np.isfinite(flattened).all():
            raise ValueError("Saved tabular inputs must preserve both historical month blocks exactly.")
        y = _measured_scores(targets["sm3_phq"], f"{name} measured SM3")
        if y.shape != (n,):
            raise ValueError("Targets and saved sample indices are not aligned.")
        loaded[name] = BaselinePartition(
            indices.copy(), mapping["participants"][indices].copy(),
            recover_measured_sm0(arrays["sm0_phq"], preprocessor.baseline_scaler),
            flattened.astype(np.float64), y,
        )
    return SavedBaselineData(loaded["train"], loaded["validation"], preprocessor, manifest, root,
                             ("manifest.json", *_CONSUMED_FILES), sha256_file(manifest_path))


@dataclass(frozen=True)
class BaselineResults:
    models: dict[str, Any]
    validation_predictions: dict[str, np.ndarray]
    metrics: dict[str, dict[str, float]]
    ridge_candidates: list[dict[str, Any]]
    selected_alpha: float
    fitted_candidates: list[Ridge]


def fit_baselines(data: SavedBaselineData) -> BaselineResults:
    """Fit every candidate on training only; use validation only for selection."""
    train, validation = data.train, data.validation
    persistence = PersistenceBaseline()
    mean = TrainingMeanBaseline().fit(train.targets)
    candidates, fitted = [], []
    for alpha in ALPHAS:
        ridge = Ridge(alpha=alpha, fit_intercept=True, solver="svd", positive=False)
        ridge.fit(np.asarray(train.tabular, dtype=np.float64), train.targets)
        prediction = ridge.predict(np.asarray(validation.tabular, dtype=np.float64))
        candidates.append({"alpha": alpha, "validation_metrics": regression_metrics(validation.targets, prediction)})
        fitted.append(ridge)
    # Python min retains the first (ascending-grid) candidate on exact ties.
    selected = min(range(len(candidates)), key=lambda i: candidates[i]["validation_metrics"]["rmse"])
    models = {"persistence": persistence, "training_mean": mean, "ridge": fitted[selected]}
    predictions = {
        "persistence": persistence.predict(validation.sm0_phq),
        "training_mean": mean.predict(len(validation.targets)),
        "ridge": models["ridge"].predict(validation.tabular),
    }
    metrics = {name: regression_metrics(validation.targets, values) for name, values in predictions.items()}
    return BaselineResults(models, predictions, metrics, candidates, ALPHAS[selected], fitted)


def run_forecasting_baselines(dataset_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Persist validation findings and fitted artifacts; reserved data stay shut."""
    output = _output_directory(output_dir)
    data = load_baseline_data(dataset_dir)
    result = fit_baselines(data)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.models, output / "models.joblib")
    joblib.dump(result.fitted_candidates, output / "ridge_candidates.joblib")
    shutil.copyfile(data.dataset_dir / "preprocessor.joblib", output / "preprocessor.joblib")
    np.savez_compressed(output / "validation_predictions.npz",
                        sample_indices=data.validation.sample_indices,
                        measured_sm3=data.validation.targets, measured_sm0=data.validation.sm0_phq,
                        **result.validation_predictions)
    np.savez_compressed(output / "used_split_indices.npz", train=data.train.sample_indices,
                        validation=data.validation.sample_indices)
    configuration = {
        "workflow": "conditional_relative_month_forecasting_baselines",
        "target": data.manifest["dataset"]["target"],
        "cutoff": "SM2", "horizon": data.manifest["dataset"]["horizon"],
        "assumptions": data.manifest["dataset"]["assumptions"],
        "eligibility": data.manifest["dataset"]["eligibility"],
        "feature_allowlist": data.manifest["dataset"]["feature_allowlist"],
        "feature_names": data.preprocessor.feature_names,
        "preprocessing": "reuse original training-fitted artifact without refitting",
        "persistence": "measured SM0 severity exactly; no fitted parameters",
        "training_mean": {"mean_sm3": result.models["training_mean"].mean_, "fitted_partition": "train"},
        "ridge": {"alphas": list(ALPHAS), "selected_alpha": result.selected_alpha,
                  "solver": "svd", "fit_intercept": True, "positive": False, "dtype": "float64",
                  "selection": "minimum validation RMSE; first ascending alpha on exact ties",
                  "final_refit": False, "fitted_partition": "train"},
        "prediction_clipping": False, "target_transformation": "none",
        "metric_weighting": "equal weight per quarter sample; repeated participants not independent",
        "evaluation_partition": "validation_only",
        "calibration_observations_opened": False, "test_observations_opened": False,
    }
    (output / "config.json").write_text(json.dumps(configuration, indent=2, allow_nan=False) + "\n")
    report = {
        "metrics": result.metrics, "selected_alpha": result.selected_alpha,
        "ridge_candidates": result.ridge_candidates,
        "counts": {name: {"samples": len(part.targets), "participants": len(np.unique(part.participants))}
                   for name, part in (("train", data.train), ("validation", data.validation))},
        "training_mean_sm3": result.models["training_mean"].mean_,
        "feature_count": len(data.preprocessor.feature_names["tabular"]),
        "metrics_scale": "MAE/RMSE in original PHQ-9 points; R2 dimensionless",
    }
    (output / "validation_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    provenance = {
        "format_version": 1, "workflow": configuration["workflow"],
        "dataset_dir": str(data.dataset_dir), "dataset_manifest_sha256": data.manifest_sha256,
        "original_input_hashes": data.manifest["inputs"],
        "original_input_hash_verification": "inherited from dataset manifest; raw CSV not reopened",
        "dataset_files_consumed": list(data.read_files),
        "consumed_sha256": {name: sha256_file(data.dataset_dir / name) for name in data.read_files},
        "dataset_source_sha256": data.manifest["source_sha256"],
        "baseline_source_sha256": sha256_file(__file__),
        "split_configuration": data.manifest["split"],
        "available_at_timestamps_verified": False, "prospective_clinical_validation": False,
        "kan_ode_conformal_used": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__},
        "thread_environment": {name: os.environ.get(name) for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        )},
        "artifact_sha256": {path.name: sha256_file(path) for path in sorted(output.iterdir())},
    }
    (output / "run.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_forecasting_baselines(args.dataset_dir, args.output), indent=2))


if __name__ == "__main__":
    # Serialize custom baseline objects under their importable module path,
    # rather than __main__, so a fresh process can load the fitted artifacts.
    from kan_phenotyping.experiments.forecasting_baselines import main as canonical_main
    canonical_main()
