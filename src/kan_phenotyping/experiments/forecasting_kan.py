"""Fixed-setting tabular KAN for conditional SM2 -> measured SM3 severity.

Reuse the frozen forecasting bundle and existing spline model/Trainer. Only
training/validation observations are consumed. No later research stage runs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ..evaluation.metrics import regression_metrics
from ..models import TabularDepressionKAN
from ..training.trainer import Trainer, TrainingConfig, make_loader, seed_everything, write_json
from .build_forecasting_dataset import _output_directory, sha256_file
from .forecasting_baselines import BaselinePartition, SavedBaselineData, load_baseline_data


SEEDS = (42, 43, 44)
SHUFFLE_SEED = 42
SANITY_SEED = 2026
N_FEATURES = 159


def model_spec() -> dict[str, Any]:
    return {
        "name": "TabularDepressionKAN", "input_kind": "tabular",
        "kwargs": {"n_features": N_FEATURES, "hidden_width": 64, "mixing_type": "kan",
                   "task": "regression", "grid_intervals": 8, "spline_degree": 3,
                   "grid_range": [-3.0, 3.0]},
        "target": "measured SM3 PHQ-9 severity, not change",
        "workflow": "conditional_relative_month_flattened_history_forecasting",
        "output": "unbounded scalar; no sigmoid, clipping, or rounding",
    }


def build_forecasting_model() -> TabularDepressionKAN:
    return TabularDepressionKAN(**model_spec()["kwargs"])


def experiment_config(seed: int) -> TrainingConfig:
    return TrainingConfig(
        task="regression", max_epochs=50, batch_size=128, learning_rate=0.001,
        weight_decay=0.0001, patience=10, seed=seed, device="cpu", num_threads=2,
        min_delta=0.0, gradient_clip=None, regression_loss="mse",
    )


class ForecastTensorDataset(Dataset):
    """The existing Trainer's tabular interface, with no artificial time axis."""

    def __init__(self, partition: BaselinePartition):
        x = np.asarray(partition.tabular, dtype=np.float32)
        y = np.asarray(partition.targets, dtype=np.float32)
        n = len(y) if y.ndim else 0
        if (x.shape != (n, N_FEATURES) or y.shape != (n,) or n == 0
                or np.asarray(partition.sample_indices).shape != (n,)
                or np.asarray(partition.participants).shape != (n,)):
            raise ValueError("Require aligned inputs [N,159], targets [N], sample indices, and participants.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Forecast inputs and measured targets must be finite.")
        if not ((y >= 0) & (y <= 27) & (y == np.rint(y))).all():
            raise ValueError("Target must remain measured SM3 PHQ-9 severity in 0..27.")
        self.x = torch.from_numpy(x.copy())
        self.y = torch.from_numpy(y.copy())

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"tabular": self.x[index], "target": self.y[index]}


def make_forecast_loader(partition: BaselinePartition, config: TrainingConfig, *, shuffle: bool):
    # Isolate initialization variability: every experiment sees identical
    # minibatch orders, independently of the model's initialization seed.
    return make_loader(ForecastTensorDataset(partition), replace(config, seed=SHUFFLE_SEED), shuffle=shuffle)


class FiniteGradientChecks:
    """Reject nonfinite backward gradients without changing the old Trainer."""

    def __init__(self, model: nn.Module):
        self.calls = {name: 0 for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.nonzero = set()
        self.handles = [parameter.register_hook(self._hook(name))
                        for name, parameter in model.named_parameters() if parameter.requires_grad]

    def _hook(self, name: str):
        def check(gradient: torch.Tensor):
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite KAN gradient: {name}")
            self.calls[name] += 1
            if name not in self.nonzero and torch.count_nonzero(gradient).item():
                self.nonzero.add(name)
            return gradient
        return check

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def metadata(self) -> dict[str, Any]:
        if not self.calls or any(count == 0 for count in self.calls.values()):
            raise RuntimeError("Every trainable tensor must receive backward gradients.")
        return {"all_observed_gradients_finite": True, "backward_calls_by_parameter": self.calls.copy(),
                "nonzero_gradient_seen_by_parameter": {name: name in self.nonzero for name in self.calls}}


def _slice_partition(partition: BaselinePartition, indices: np.ndarray) -> BaselinePartition:
    return BaselinePartition(**{name: getattr(partition, name)[indices].copy() for name in (
        "sample_indices", "participants", "sm0_phq", "tabular", "targets",
    )})


def run_overfit_sanity(train: BaselinePartition, output_dir: str | Path, *, max_steps: int = 500) -> dict[str, Any]:
    """Train-only 32-example diagnostic; no validation payload is accepted.

    Fixed pass rule: RMSE <=0.25 and MSE <=1% of initial MSE within 500 steps.
    Stop on success, save separately, and never transfer its weights to a run.
    """
    output = _output_directory(output_dir)
    if max_steps < 1 or max_steps > 500:
        raise ValueError("The sanity check permits 1..500 fixed-cap steps.")
    seed_everything(SANITY_SEED, 2)
    indices = np.sort(np.random.default_rng(SANITY_SEED).choice(len(train.targets), min(32, len(train.targets)), replace=False))
    subset = _slice_partition(train, indices)
    dataset = ForecastTensorDataset(subset)
    model = build_forecasting_model()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.MSELoss()
    checks = FiniteGradientChecks(model)
    history = []
    started = time.perf_counter()
    with torch.no_grad():
        initial_mse = float(criterion(model(dataset.x), dataset.y))
    if not np.isfinite(initial_mse) or initial_mse <= 0:
        raise RuntimeError("Sanity diagnostic needs a finite positive initial loss.")
    history.append({"step": 0, "training_mse": initial_mse})
    passed = False
    try:
        for step in range(1, max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(dataset.x)
            if prediction.shape != dataset.y.shape:
                raise ValueError("KAN must return exactly one severity per sample.")
            loss = criterion(prediction, dataset.y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite overfit loss.")
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                mse = float(criterion(model(dataset.x), dataset.y))
            if not np.isfinite(mse):
                raise FloatingPointError("Nonfinite overfit evaluation.")
            history.append({"step": step, "training_mse": mse})
            passed = mse <= 0.25 ** 2 and mse <= initial_mse * 0.01
            if passed:
                break
    finally:
        checks.close()
    gradient_metadata = checks.metadata()
    changed = any(not torch.equal(initial[name], parameter.detach()) for name, parameter in model.named_parameters())
    result = {"training_only": True, "seed": SANITY_SEED, "samples": len(dataset),
              "max_steps": max_steps, "steps_run": step, "initial_mse": initial_mse,
              "final_mse": mse, "final_rmse": float(np.sqrt(mse)),
              "pass_rule": "RMSE <=0.25 and MSE <=1% of initial; finite gradients; changed parameters",
              "passed": bool(passed and changed and all(gradient_metadata["nonzero_gradient_seen_by_parameter"].values())),
              "parameters_changed": changed, "gradient_checks": gradient_metadata,
              "seconds": time.perf_counter() - started,
              "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001},
              "warm_start_experiment": False}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "sanity.json", result)
    write_json(output / "history.json", history)
    np.savez_compressed(output / "training_subset_indices.npz", sample_indices=subset.sample_indices)
    torch.save({"model_state": model.state_dict(), "model_metadata": model_spec(),
                "diagnostic_only": True}, output / "sanity.pt")
    if not result["passed"]:
        raise RuntimeError("Training-only overfit sanity check failed; do not launch experiments.")
    return result


def restore_validation(checkpoint_path: str | Path, data: SavedBaselineData) -> tuple[dict, np.ndarray]:
    """Reload fresh model weights and evaluate the saved validation order only."""
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["model_metadata"] != model_spec() or saved["input_kind"] != "tabular":
        raise ValueError("Checkpoint must describe the fixed conditional forecasting KAN.")
    config = TrainingConfig(**saved["training_config"])
    if config.task != "regression":
        raise ValueError("Measured severity requires unbounded regression.")
    seed_everything(config.seed, config.num_threads)
    model = build_forecasting_model()
    model.load_state_dict(saved["model_state"])
    trainer = Trainer(model, config, input_kind="tabular")
    truth, prediction = trainer.predict(make_forecast_loader(data.validation, config, shuffle=False))
    np.testing.assert_array_equal(truth, data.validation.targets)
    return regression_metrics(truth, prediction), prediction


def train_seed(data: SavedBaselineData, output_dir: str | Path, seed: int,
               config: TrainingConfig | None = None) -> dict[str, Any]:
    """One initialization run; config override supports small synthetic tests.

    The command-line experiment always uses experiment_config unmodified.
    """
    output = _output_directory(output_dir)
    config = experiment_config(seed) if config is None else config
    if config.seed != seed or config.task != "regression" or config.regression_loss != "mse":
        raise ValueError("Seed and measured-severity MSE regression configuration must agree.")
    before = data.preprocessor.export_metadata()
    if before != data.manifest["preprocessing"] or len(data.preprocessor.feature_names["tabular"]) != N_FEATURES:
        raise ValueError("Reuse exactly the original 159-feature training-only preprocessing.")
    seed_everything(seed, config.num_threads)
    model = build_forecasting_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    train_loader = make_forecast_loader(data.train, config, shuffle=True)
    validation_loader = make_forecast_loader(data.validation, config, shuffle=False)
    trainer = Trainer(model, config, input_kind="tabular")
    checks = FiniteGradientChecks(model)
    started = time.perf_counter()
    try:
        fitted = trainer.fit(train_loader, validation_loader, output, model_metadata=model_spec())
    finally:
        checks.close()
    training_seconds = time.perf_counter() - started
    truth, prediction = trainer.predict(make_forecast_loader(data.validation, config, shuffle=False))
    np.testing.assert_array_equal(truth, data.validation.targets)
    metrics = regression_metrics(truth, prediction)
    restored_metrics, restored_prediction = restore_validation(fitted.checkpoint, data)
    np.testing.assert_array_equal(prediction, restored_prediction)
    if metrics != restored_metrics or metrics["rmse"] != fitted.best_score:
        raise RuntimeError("Restored checkpoint does not reproduce selected validation metrics.")
    if data.preprocessor.export_metadata() != before:
        raise RuntimeError("Saved preprocessing changed during training.")
    np.savez_compressed(output / "validation_predictions.npz", sample_indices=data.validation.sample_indices,
                        measured_sm3=data.validation.targets, prediction=prediction)
    result = {"seed": seed, "validation": metrics, "parameter_count": parameter_count,
              "best_epoch": fitted.best_epoch, "epochs_run": len(fitted.history),
              "training_seconds": training_seconds, "selection_metric": "validation_rmse",
              "gradient_checks": checks.metadata(), "restored_predictions_match": True,
              "preprocessing_unchanged": True, "shuffle_seed": SHUFFLE_SEED,
              "warm_start_from_sanity": False}
    write_json(output / "validation_metrics.json", result)
    return result


def aggregate_seed_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if len(records) < 2:
        raise ValueError("Sample SD requires at least two initialization runs.")
    return {metric: {"mean": float(np.mean([row["validation"][metric] for row in records])),
                     "sample_sd": float(np.std([row["validation"][metric] for row in records], ddof=1))}
            for metric in ("mae", "rmse", "r2")}


def _baseline_comparison(data: SavedBaselineData, root: Path) -> dict[str, Any]:
    """Read existing validation findings; never refit or reevaluate test data."""
    run = json.loads((root / "run.json").read_text())
    if run["dataset_manifest_sha256"] != data.manifest_sha256:
        raise ValueError("Baselines and KAN must share the identical dataset manifest.")
    for name in ("validation_metrics.json", "validation_predictions.npz", "config.json"):
        if sha256_file(root / name) != run["artifact_sha256"][name]:
            raise ValueError("Existing baseline artifact hash mismatch.")
    with np.load(root / "validation_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["sample_indices"], data.validation.sample_indices)
        np.testing.assert_array_equal(saved["measured_sm3"], data.validation.targets)
    return {"directory": str(root), "run_sha256": sha256_file(root / "run.json"),
            "validation_metrics_sha256": sha256_file(root / "validation_metrics.json"),
            "results": json.loads((root / "validation_metrics.json").read_text())}


def _plot_curves(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for seed, ax in zip(SEEDS, axes):
        rows = json.loads((output / f"seed_{seed}" / "history.json").read_text())
        best = json.loads((output / f"seed_{seed}" / "validation_metrics.json").read_text())["best_epoch"]
        ax.plot([r["epoch"] for r in rows], [np.sqrt(r["training_loss"]) for r in rows], label="Training (online)")
        ax.plot([r["epoch"] for r in rows], [r["validation"]["rmse"] for r in rows], label="Validation")
        ax.axvline(best, color="gray", linestyle=":", label="Selected checkpoint")
        ax.set(title=f"Initialization {seed}", xlabel="Epoch")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("RMSE (PHQ-9 points)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Conditional relative-month forecasting: flattened-history KAN")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.svg")
    plt.close(fig)


def run_forecasting_kan(dataset_dir: str | Path, baseline_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = _output_directory(output_dir)
    data = load_baseline_data(dataset_dir)
    comparison = _baseline_comparison(data, Path(baseline_dir).resolve())
    source_preprocessor = data.dataset_dir / "preprocessor.joblib"
    preprocessor_hash = sha256_file(source_preprocessor)
    output.mkdir(parents=True, exist_ok=True)
    configuration = {
        "workflow": model_spec()["workflow"], "model": model_spec(), "seeds": list(SEEDS),
        "training_by_seed": {str(seed): asdict(experiment_config(seed)) for seed in SEEDS},
        "optimizer": "AdamW", "loss": "MSE", "shuffle_seed": SHUFFLE_SEED,
        "hyperparameter_search": False, "prediction_clipping_or_rounding": False,
        "preprocessing_refit": False, "feature_names": data.preprocessor.feature_names,
        "dataset_protocol": data.manifest["dataset"], "saved_split": data.manifest["split"],
        "available_at_timestamps_verified": False, "prospective_clinical_validation": False,
        "calibration_or_test_observations_consumed": False,
    }
    write_json(output / "config.json", configuration)
    print("Starting separate training-only overfit sanity check.", flush=True)
    sanity = run_overfit_sanity(data.train, output / "sanity")
    print(f"Sanity passed: {sanity['steps_run']} steps, training RMSE {sanity['final_rmse']:.6f}.", flush=True)
    records = []
    for seed in SEEDS:
        print(f"Training fixed initialization seed {seed}.", flush=True)
        record = train_seed(data, output / f"seed_{seed}", seed)
        records.append(record)
        print(f"Seed {seed}: best epoch {record['best_epoch']}, validation RMSE {record['validation']['rmse']:.6f}.", flush=True)
    if sha256_file(source_preprocessor) != preprocessor_hash:
        raise RuntimeError("Original training-only preprocessor file changed.")
    shutil.copyfile(source_preprocessor, output / "preprocessor.joblib")
    np.savez_compressed(output / "used_split_indices.npz", train=data.train.sample_indices,
                        validation=data.validation.sample_indices)
    _plot_curves(output)
    summary = {"workflow": configuration["workflow"], "seeds": records,
               "mean_and_sample_sd": aggregate_seed_metrics(records), "baselines": comparison["results"],
               "counts": {name: {"samples": len(part.targets), "participants": len(np.unique(part.participants))}
                          for name, part in (("train", data.train), ("validation", data.validation))},
               "sanity": sanity, "parameter_count": records[0]["parameter_count"],
               "total_experiment_fit_seconds": sum(record["training_seconds"] for record in records),
               "sd_interpretation": "initialization variation on one split; not a population confidence interval"}
    write_json(output / "summary.json", summary)
    source_root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__), Path(__file__).with_name("forecasting_baselines.py"),
                    source_root / "models" / "depression_kan.py", source_root / "models" / "kan_layer.py",
                    source_root / "training" / "trainer.py", source_root / "evaluation" / "metrics.py"]
    provenance = {
        "dataset_dir": str(data.dataset_dir), "dataset_manifest_sha256": data.manifest_sha256,
        "dataset_files_consumed": list(data.read_files),
        "consumed_sha256": {name: sha256_file(data.dataset_dir / name) for name in data.read_files},
        "original_input_hashes": data.manifest["inputs"], "builder_source_hashes": data.manifest["source_sha256"],
        "baseline_comparison": comparison, "preprocessor_sha256": preprocessor_hash,
        "source_sha256": {str(path.relative_to(source_root)): sha256_file(path) for path in source_paths},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "device": "cpu", "deterministic_algorithms": True,
        "thread_environment": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "artifact_sha256": {str(path.relative_to(output)): sha256_file(path)
                            for path in sorted(output.rglob("*")) if path.is_file()},
    }
    write_json(output / "run.json", provenance)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_forecasting_kan(args.dataset_dir, args.baseline_dir, args.output)
    print(json.dumps({"mean_and_sample_sd": summary["mean_and_sample_sd"],
                      "total_experiment_fit_seconds": summary["total_experiment_fit_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
