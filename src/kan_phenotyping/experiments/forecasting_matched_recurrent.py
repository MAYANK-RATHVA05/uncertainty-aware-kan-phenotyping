"""Optimization-matched clip-5 recurrent control, with no ODE training.

Reuse frozen training/validation arrays and the unchanged recurrent model.
Fresh shared initial tensors must exactly match saved ODE initial artifacts.
Only initial ODE weights, aggregate results, and validation predictions enter.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import numpy as np
import torch

from ..evaluation.metrics import regression_metrics
from ..training.gradient_diagnostics import GradientDiagnostics
from ..training.trainer import Trainer, TrainingConfig, seed_everything, write_json
from .build_forecasting_dataset import _output_directory, sha256_file
from .forecasting_baselines import load_baseline_data
from .forecasting_kan import aggregate_seed_metrics
from .forecasting_kan_ode_full import model_spec as ode_spec
from .forecasting_sequential_kan import (
    SEEDS, SequentialData, build_sequential_model,
    make_sequential_loader, model_spec as recurrent_spec,
    prefix_causality_check, prepare_sequential_data,
)


COMPONENTS = ("static_encoder", "wearable_encoder", "gru", "head")


def model_spec() -> dict[str, Any]:
    spec = recurrent_spec()
    spec["workflow"] = "conditional_relative_month_optimization_matched_clip5_recurrent_control"
    return spec


def experiment_config(seed: int) -> TrainingConfig:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("The fixed experiment permits initialization seeds 42, 43, 44 only.")
    return TrainingConfig(task="regression", max_epochs=50, batch_size=128,
                          learning_rate=.001, weight_decay=.0001, patience=10,
                          seed=seed, device="cpu", num_threads=2, min_delta=0.,
                          gradient_clip=5., regression_loss="mse")


def initialization_check(model, ode_initial_path: str | Path, seed: int) -> dict:
    """Compare a fresh recurrent model; never copy or load trained ODE weights."""
    path = Path(ode_initial_path)
    if path.name != "initial.pt" or path.parent.name != f"seed_{seed}":
        raise ValueError("Only the matching seed's initial.pt artifact is permitted.")
    initial = torch.load(path, map_location="cpu", weights_only=True)
    if (set(initial) != {"model_state", "initialization_seed", "model_metadata"}
            or initial["initialization_seed"] != seed or initial["model_metadata"] != ode_spec()):
        raise ValueError("Require a documented fresh full-experiment ODE initialization artifact.")
    expected, actual = initial["model_state"], model.state_dict()
    extra = {"dynamics.base_weight", "dynamics.spline_weight", "dynamics.bias", "dynamics.knots"}
    if set(expected) != set(actual) | extra:
        raise ValueError("ODE initialization must contain exactly the recurrent components plus dynamics.")
    matches = {}
    for name, value in actual.items():
        reference = expected[name]
        if (value.dtype != torch.float32 or reference.dtype != torch.float32
                or not torch.isfinite(value).all() or not torch.isfinite(reference).all()
                or not torch.equal(value, reference)):
            raise ValueError(f"Initial shared component does not match saved ODE initialization: {name}")
        matches[name] = True
    if sum(p.numel() for p in model.parameters()) != 23169:
        raise ValueError("Require the unchanged 23,169-parameter recurrent control.")
    return {"seed": seed, "all_shared_state_tensors_exact": True,
            "state_tensor_matches": matches,
            "component_matches": {component: all(value for name, value in matches.items()
                                                 if name.startswith(component + ".")) for component in COMPONENTS},
            "shared_parameter_tensors": len(list(model.parameters())), "shared_state_tensors": len(matches),
            "precision": "float32", "weights_copied_from_ode": False,
            "trained_ode_weights_consumed": False, "ode_initial_sha256": sha256_file(path)}


def load_ode_reference(data: SequentialData, ode_dir: str | Path) -> dict:
    """Read hash-verified initial artifacts/results, never ODE best.pt."""
    root = Path(ode_dir).resolve()
    run = json.loads((root / "run.json").read_text())
    if (run["dataset_manifest_sha256"] != data.source.manifest_sha256
            or run["device"] != "cpu" or run["num_threads"] != 2
            or run["deterministic_algorithms"] is not True
            or run["versions"]["torch"] != str(torch.__version__)
            or run["versions"]["numpy"] != np.__version__):
        raise ValueError("Require the exact saved ODE cohort and matching runtime.")
    package = Path(__file__).resolve().parents[1]
    if any(sha256_file(package / name) != value for name, value in run["source_sha256"].items()):
        raise ValueError("Existing ODE/training/loading source differs from saved provenance.")
    for name in data.source.read_files:
        if sha256_file(data.source.dataset_dir / name) != run["consumed_sha256"][name]:
            raise ValueError("ODE and matched control must consume identical saved train/validation files.")
    files = ["config.json", "summary.json"] + [f"seed_{seed}/{name}" for seed in SEEDS for name in (
        "training_config.json", "initial.pt", "validation_metrics.json", "validation_predictions.npz")]
    for name in files:
        if sha256_file(root / name) != run["artifact_sha256"][name]:
            raise ValueError(f"ODE reference artifact hash mismatch: {name}")
    configuration = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if configuration["minibatch_order_seed"] != 42 or configuration["model"] != ode_spec():
        raise ValueError("Require the existing ODE architecture and minibatch protocol.")
    if [row["seed"] for row in summary["seeds"]] != list(SEEDS):
        raise ValueError("Require all three existing ODE initializations in canonical order.")
    metrics, configs = {}, {}
    for seed, row in zip(SEEDS, summary["seeds"]):
        config = json.loads((root / f"seed_{seed}/training_config.json").read_text())
        if config != asdict(experiment_config(seed)):
            raise ValueError("Matched recurrent optimization must equal the actual saved ODE configuration.")
        saved_result = json.loads((root / f"seed_{seed}/validation_metrics.json").read_text())
        with np.load(root / f"seed_{seed}/validation_predictions.npz", allow_pickle=False) as predictions:
            np.testing.assert_array_equal(predictions["sample_indices"], data.validation.sample_indices)
            np.testing.assert_array_equal(predictions["measured_sm3"], data.validation.targets)
            recomputed = regression_metrics(predictions["measured_sm3"], predictions["prediction"])
        if recomputed != row["validation"] or recomputed != saved_result["validation"]:
            raise ValueError("Existing ODE metrics must reproduce from aligned saved validation predictions.")
        metrics[str(seed)], configs[str(seed)] = recomputed, config
    if aggregate_seed_metrics(summary["seeds"]) != summary["mean_and_sample_sd"]:
        raise ValueError("Existing ODE initialization summary is inconsistent.")
    return {"directory": str(root), "seed_metrics": metrics, "training_by_seed": configs,
            "mean_and_sample_sd": summary["mean_and_sample_sd"],
            "parameter_count": summary["parameter_count"],
            "prior_unclipped_recurrent": summary["comparisons"]["recurrent_KAN"],
            "minibatch_order_seed": 42, "device": "cpu", "precision": "float32", "num_threads": 2,
            "ode_source_sha256": run["source_sha256"],
            "trained_ode_weights_consumed": False,
            "consumed_sha256": {name: sha256_file(root / name) for name in ("run.json", *files)}}


class MatchedTrainer(Trainer):
    """Observe clipping and numerical behavior, preserving the original updates."""

    def __init__(self, model, config, output: Path):
        super().__init__(model, config, input_kind="temporal")
        self.diagnostics = GradientDiagnostics(model, clip_norm=config.gradient_clip)
        self.output, self.epoch = output, 0
        self.epoch_diagnostics: list[dict] = []

    def _epoch(self, loader, optimizer=None):
        if optimizer is None:
            return super()._epoch(loader, optimizer)
        self.epoch += 1
        first_update = len(self.diagnostics.records)
        started = time.perf_counter()

        def before_step(opt, args, kwargs):
            self.diagnostics.finish_backward()["epoch"] = self.epoch

        def after_step(opt, args, kwargs):
            for name, parameter in self.model.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(f"Nonfinite recurrent parameter after update: {name}")

        handles = [optimizer.register_step_pre_hook(before_step), optimizer.register_step_post_hook(after_step)]
        try:
            result = super()._epoch(loader, optimizer)
        finally:
            for handle in handles:
                handle.remove()
            write_json(self.output / "gradient_norms.json", self.diagnostics.records)
            write_json(self.output / "gradient_checks.json", self.diagnostics.metadata())
        rows = self.diagnostics.records[first_update:]
        norms = [row["preclip_global_norm"] for row in rows]
        self.epoch_diagnostics.append({"epoch": self.epoch, "training_seconds": time.perf_counter() - started,
            "optimizer_updates": len(rows), "updates_above_clip5": sum(value > 5 for value in norms),
            "preclip_norm_min": min(norms), "preclip_norm_max": max(norms),
            "preclip_norm_mean": float(np.mean(norms))})
        write_json(self.output / "epoch_diagnostics.json", self.epoch_diagnostics)
        if self.epoch == 1 or self.epoch % 5 == 0:
            print(f"Matched recurrent seed {self.config.seed}, epoch {self.epoch}: training MSE {result[0]:.6f}.", flush=True)
        return result


def restore_validation(checkpoint: str | Path, data: SequentialData) -> tuple[dict, np.ndarray]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = TrainingConfig(**saved["training_config"])
    if (saved["model_metadata"] != model_spec() or saved["input_kind"] != "temporal"
            or config != experiment_config(config.seed)):
        raise ValueError("Require the optimization-matched clip-5 recurrent checkpoint.")
    seed_everything(config.seed, config.num_threads)
    model = build_sequential_model()
    model.load_state_dict(saved["model_state"])
    trainer = Trainer(model, config, input_kind="temporal")
    truth, prediction = trainer.predict(make_sequential_loader(data.validation, config, shuffle=False))
    np.testing.assert_array_equal(truth, data.validation.targets)
    return regression_metrics(truth, prediction), prediction


def train_seed(data: SequentialData, output_dir: str | Path, seed: int,
               ode_initial_path: str | Path) -> dict:
    output = _output_directory(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = experiment_config(seed)
    preprocessing = data.source.preprocessor.export_metadata()
    if preprocessing != data.source.manifest["preprocessing"]:
        raise ValueError("Require unchanged saved training-only preprocessing.")
    seed_everything(seed, config.num_threads)
    model = build_sequential_model()
    trainer = MatchedTrainer(model, config, output)
    phase = "initial component matching"
    started = time.perf_counter()
    try:
        matching = initialization_check(model, ode_initial_path, seed)
        write_json(output / "initialization_match.json", matching)
        torch.save({"model_state": model.state_dict(), "initialization_seed": seed,
                    "model_metadata": model_spec()}, output / "initial.pt")
        phase = "training and checkpoint selection"
        fit_started = time.perf_counter()
        fit = trainer.fit(make_sequential_loader(data.train, config, shuffle=True),
                          make_sequential_loader(data.validation, config, shuffle=False),
                          output, model_metadata=model_spec())
        elapsed = time.perf_counter() - fit_started
        gradient_checks = trainer.diagnostics.metadata()
        trainer.diagnostics.close()
        if not gradient_checks["all_parameters_received_nonzero_gradients"]:
            raise RuntimeError("Every recurrent parameter tensor must receive finite nonzero training gradients.")
        phase = "restored validation reproduction"
        truth, prediction = trainer.predict(make_sequential_loader(data.validation, config, shuffle=False))
        np.testing.assert_array_equal(truth, data.validation.targets)
        metrics = regression_metrics(truth, prediction)
        restored_metrics, restored = restore_validation(fit.checkpoint, data)
        np.testing.assert_array_equal(prediction, restored)
        if metrics != restored_metrics or metrics["rmse"] != fit.best_score:
            raise RuntimeError("Restored predictions and metrics must match the selected checkpoint exactly.")
        prefix = prefix_causality_check(model, data.train)
        if data.source.preprocessor.export_metadata() != preprocessing:
            raise RuntimeError("Saved preprocessing changed.")
        np.savez_compressed(output / "validation_predictions.npz", sample_indices=data.validation.sample_indices,
                            measured_sm3=data.validation.targets, prediction=prediction)
        norms = gradient_checks["preclip_norm_summary"]
        result = {"seed": seed, "validation": metrics, "parameter_count": sum(p.numel() for p in model.parameters()),
            "best_epoch": fit.best_epoch, "stopped_epoch": len(fit.history), "epochs_run": len(fit.history),
            "stop_reason": "epoch_budget" if len(fit.history) == config.max_epochs else "early_stopping",
            "training_seconds": elapsed, "total_seed_seconds": time.perf_counter() - started,
            "gradient_checks": gradient_checks,
            "clipping_frequency": {"updates_above_threshold": norms["above_clip_norm_count"], "updates": norms["n"],
                                   "fraction": norms["above_clip_norm_count"] / norms["n"], "threshold": 5.},
            "initialization_match": matching, "prefix_checks": prefix,
            "restored_predictions_and_metrics_exact": True, "preprocessing_unchanged": True,
            "fresh_initialization": True, "trained_ode_weights_consumed": False, "numerical_failures": []}
        write_json(output / "validation_metrics.json", result)
        return result
    except Exception as error:
        torch.save({"model_state": model.state_dict(), "diagnostic_only_failed_run": True}, output / "failed_state.pt")
        write_json(output / "failure.json", {"seed": seed, "phase": phase, "type": type(error).__name__,
            "message": str(error), "gradient_checks": trainer.diagnostics.metadata(),
            "performance_interpretation_permitted": False})
        raise
    finally:
        trainer.diagnostics.close()


def paired_differences(records: list[dict], ode_reference: dict) -> list[dict]:
    if [row["seed"] for row in records] != list(SEEDS):
        raise ValueError("Compare exactly seeds 42, 43, 44 in canonical order.")
    return [{"seed": row["seed"], "ode_minus_matched": {
        metric: ode_reference["seed_metrics"][str(row["seed"])][metric] - row["validation"][metric]
        for metric in ("mae", "rmse", "r2")}} for row in records]


def _plot(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for seed, ax in zip(SEEDS, axes):
        rows = json.loads((output / f"seed_{seed}/history.json").read_text())
        best = json.loads((output / f"seed_{seed}/validation_metrics.json").read_text())["best_epoch"]
        ax.plot([r["epoch"] for r in rows], [np.sqrt(r["training_loss"]) for r in rows], label="Training (online)")
        ax.plot([r["epoch"] for r in rows], [r["validation"]["rmse"] for r in rows], label="Validation")
        ax.axvline(best, color="gray", linestyle=":", label="Selected checkpoint")
        ax.set(title=f"Initialization {seed}", xlabel="Epoch")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("RMSE (PHQ-9 points)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Conditional forecasting: optimization-matched recurrent control, norm clip 5")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.svg")
    plt.close(fig)


def run_matched(dataset_dir: str | Path, ode_dir: str | Path, output_dir: str | Path) -> dict:
    output = _output_directory(output_dir)
    source = load_baseline_data(dataset_dir)
    data = prepare_sequential_data(source)
    if (len(data.train.targets), len(data.validation.targets), len(np.unique(data.train.participants)),
            len(np.unique(data.validation.participants))) != (5111, 851, 2025, 338):
        raise ValueError("Require the exact frozen 5111/851-sample participant-disjoint cohort.")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        reference = load_ode_reference(data, ode_dir)
        preprocessing_hash = sha256_file(source.dataset_dir / "preprocessor.joblib")
        write_json(output / "ode_reference.json", reference)
        write_json(output / "config.json", {"model": model_spec(), "seeds": list(SEEDS),
            "training_by_seed": {str(seed): asdict(experiment_config(seed)) for seed in SEEDS},
            "dataset_protocol": source.manifest["dataset"], "saved_split": source.manifest["split"],
            "feature_names": source.preprocessor.feature_names, "minibatch_order_seed": 42,
            "precision": "float32", "deterministic_algorithms": True, "hyperparameter_search": False,
            "preprocessing_refit": False, "trained_ode_weights_consumed": False,
            "calibration_or_test_payloads_consumed": False, "ode_retrained_or_tuned": False})
        # Establish matching for EVERY seed before any optimizer is constructed.
        preflight = []
        for seed in SEEDS:
            seed_everything(seed, 2)
            check = initialization_check(build_sequential_model(), Path(reference["directory"]) / f"seed_{seed}/initial.pt", seed)
            preflight.append(check)
        write_json(output / "initialization_preflight.json", preflight)
        print("All three fresh recurrent initializations exactly match saved initial ODE components.", flush=True)
        np.savez_compressed(output / "used_split_indices.npz", train=data.train.sample_indices,
                            validation=data.validation.sample_indices)
        for seed in SEEDS:
            print(f"Training fresh optimization-matched recurrent seed {seed}, clip norm 5.", flush=True)
            record = train_seed(data, output / f"seed_{seed}", seed,
                                Path(reference["directory"]) / f"seed_{seed}/initial.pt")
            records.append(record)
            write_json(output / "completed_seeds.json", records)
            print(f"Seed {seed}: best {record['best_epoch']}, stopped {record['stopped_epoch']}, validation RMSE {record['validation']['rmse']:.6f}.", flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
            "completed_seeds": [row["seed"] for row in records], "performance_interpretation_permitted": False})
        raise
    if sha256_file(source.dataset_dir / "preprocessor.joblib") != preprocessing_hash:
        raise RuntimeError("Original preprocessing changed.")
    differences = paired_differences(records, reference)
    summary = {"workflow": model_spec()["workflow"], "seeds": records,
        "mean_and_sample_sd": aggregate_seed_metrics(records), "ode_reference": reference,
        "per_seed_ode_minus_matched": differences,
        "difference_mean_and_sample_sd": aggregate_seed_metrics([
            {"seed": row["seed"], "validation": row["ode_minus_matched"]} for row in differences]),
        "counts": {name: {"samples": len(part.targets), "participants": len(np.unique(part.participants))}
                   for name, part in (("train", data.train), ("validation", data.validation))},
        "parameter_counts": {"matched_recurrent": 23169, "latent_kan_ode": reference["parameter_count"]},
        "total_fit_seconds": sum(row["training_seconds"] for row in records), "numerical_failures": [],
        "interpretation": "descriptive initialization comparison; optimization matched, capacity differs by 3088 parameters"}
    write_json(output / "summary.json", summary)
    shutil.copyfile(source.dataset_dir / "preprocessor.joblib", output / "preprocessor.joblib")
    _plot(output)
    package = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), Path(__file__).with_name("forecasting_sequential_kan.py"),
        Path(__file__).with_name("forecasting_baselines.py"), Path(__file__).with_name("forecasting_kan.py"),
        Path(__file__).with_name("forecasting_kan_ode_full.py"),
        package / "models/sequential_kan.py", package / "models/kan_layer.py",
        package / "training/trainer.py", package / "training/gradient_diagnostics.py",
        package / "evaluation/metrics.py"]
    write_json(output / "run.json", {"dataset_dir": str(source.dataset_dir),
        "dataset_manifest_sha256": source.manifest_sha256, "dataset_files_consumed": list(source.read_files),
        "consumed_sha256": {name: sha256_file(source.dataset_dir / name) for name in source.read_files},
        "original_input_hashes": source.manifest["inputs"], "builder_source_hashes": source.manifest["source_sha256"],
        "source_sha256": {str(path.relative_to(package)): sha256_file(path) for path in paths},
        "preprocessor_sha256": preprocessing_hash, "ode_reference_consumed_sha256": reference["consumed_sha256"],
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__},
        "device": "cpu", "precision": "float32", "num_threads": 2, "deterministic_algorithms": True,
        "artifact_sha256": {str(path.relative_to(output)): sha256_file(path)
                            for path in sorted(output.rglob("*")) if path.is_file()}})
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-dir", "ode-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_matched(args.dataset_dir, args.ode_dir, args.output)
    print(json.dumps({"mean_and_sample_sd": result["mean_and_sample_sd"],
                      "per_seed_ode_minus_matched": result["per_seed_ode_minus_matched"]}, indent=2))


if __name__ == "__main__":
    main()
