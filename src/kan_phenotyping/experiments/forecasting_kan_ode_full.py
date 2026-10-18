"""Fixed three-seed latent KAN-ODE training on frozen conditional forecasting.

No smoke/sanity weights, cohort construction, preprocessing fitting, reserved
payloads, architecture tuning or solver selection enter this experiment.
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
from .forecasting_kan_ode import (
    build_model, model_spec as implementation_spec, participant_subset, prefix_check,
    step_size_diagnostic, subset_rows,
)
from .forecasting_sequential_kan import (
    SEEDS, SequentialData, SequentialPartition, _comparisons,
    make_sequential_loader, prepare_sequential_data,
)


def model_spec() -> dict[str, Any]:
    spec = implementation_spec()
    spec["workflow"] = "conditional_relative_month_latent_KAN_ODE_full_training"
    return spec


def experiment_config(seed: int) -> TrainingConfig:
    if seed not in SEEDS:
        raise ValueError("This fixed experiment permits initialization seeds 42, 43, 44 only.")
    return TrainingConfig(task="regression", max_epochs=50, batch_size=128,
                          learning_rate=.001, weight_decay=.0001, patience=10,
                          seed=seed, device="cpu", num_threads=2, min_delta=0.,
                          gradient_clip=5., regression_loss="mse")


def diagnostic_partition(train: SequentialPartition) -> SequentialPartition:
    # Preserve the smoke step diagnostic's outcome-independent subset exactly,
    # while the optimizer receives ALL saved training samples.
    subset = participant_subset(train)
    return subset_rows(subset, np.arange(min(8, len(subset.targets))))


def _difference(after: dict, before: dict) -> dict:
    return {key: after[key] - before[key] for key in after}


def _parameter_delta(model, initial: dict[str, torch.Tensor]) -> dict[str, Any]:
    differences = {name: float(torch.linalg.vector_norm(
        parameter.detach().double() - initial[name].double()))
        for name, parameter in model.dynamics.named_parameters()}
    return {"l2_change_from_initial_by_parameter": differences,
            "every_dynamics_parameter_tensor_changed": all(value > 0 for value in differences.values())}


class DiagnosticTrainer(Trainer):
    """Observe the existing trainer without changing its loss or update order.

    Parameter hooks capture pre-clipping norms. An optimizer pre-hook completes
    that record after the existing trainer clips; its immutable captured norms
    still describe the gradients BEFORE clipping. A post-hook checks the update.
    """

    def __init__(self, model, config, output: Path):
        super().__init__(model, config, input_kind="temporal")
        self.diagnostics = GradientDiagnostics(model, clip_norm=config.gradient_clip)
        self.output = output
        self.training_epoch = 0
        self.epoch_diagnostics: list[dict] = []

    def _epoch(self, loader, optimizer=None):
        before_calls = self.model.integrator.stats()
        started = time.perf_counter()
        if optimizer is None:
            result = super()._epoch(loader, optimizer)
            if self.epoch_diagnostics:
                self.epoch_diagnostics[-1]["validation_solver_calls"] = _difference(
                    self.model.integrator.stats(), before_calls)
            return result
        self.training_epoch += 1
        first_update = len(self.diagnostics.records)
        previous = {name: value.detach().clone() for name, value in self.model.dynamics.named_parameters()}

        def before_step(opt, args, kwargs):
            record = self.diagnostics.finish_backward()
            record["epoch"] = self.training_epoch

        def after_step(opt, args, kwargs):
            record = self.diagnostics.records[-1]
            changes = {}
            for name, parameter in self.model.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(f"Nonfinite model parameter after update: {name}")
            for name, parameter in self.model.dynamics.named_parameters():
                changes[name] = float(torch.linalg.vector_norm(parameter.detach().double() - previous[name].double()))
                previous[name] = parameter.detach().clone()
            record["dynamics_l2_update_by_parameter"] = changes
            if not any(value > 0 for value in changes.values()):
                raise RuntimeError("Dynamics parameters did not change at an optimizer update.")

        handles = [optimizer.register_step_pre_hook(before_step), optimizer.register_step_post_hook(after_step)]
        try:
            result = super()._epoch(loader, optimizer)
        finally:
            for handle in handles:
                handle.remove()
            # Incremental evidence survives a failed later epoch or seed.
            write_json(self.output / "gradient_norms.json", self.diagnostics.records)
            write_json(self.output / "gradient_checks.json", self.diagnostics.metadata())
        rows = self.diagnostics.records[first_update:]
        self.epoch_diagnostics.append({
            "epoch": self.training_epoch, "training_seconds": time.perf_counter() - started,
            "optimizer_updates": len(rows),
            "preclip_norm_min": min(row["preclip_global_norm"] for row in rows),
            "preclip_norm_max": max(row["preclip_global_norm"] for row in rows),
            "training_solver_calls": _difference(self.model.integrator.stats(), before_calls),
        })
        write_json(self.output / "epoch_diagnostics.json", self.epoch_diagnostics)
        if self.training_epoch == 1 or self.training_epoch % 5 == 0:
            print(f"Seed {self.config.seed}, epoch {self.training_epoch}: training MSE {result[0]:.6f}.", flush=True)
        return result


def assert_stable(diagnostic: dict) -> None:
    if not diagnostic["all_states_and_predictions_finite"] or diagnostic["instability_warning"]:
        raise FloatingPointError("Material RK4 step-halving discrepancy on fixed training diagnostics; stop and diagnose.")


def restore_validation(checkpoint: str | Path, data: SequentialData) -> tuple[dict, np.ndarray]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = TrainingConfig(**saved["training_config"])
    if (saved["model_metadata"] != model_spec() or saved["input_kind"] != "temporal"
            or config != experiment_config(config.seed)):
        raise ValueError("Require the fixed full-training latent KAN-ODE checkpoint.")
    seed_everything(config.seed, config.num_threads)
    model = build_model()
    model.load_state_dict(saved["model_state"])
    trainer = Trainer(model, config, input_kind="temporal")
    truth, prediction = trainer.predict(make_sequential_loader(data.validation, config, shuffle=False))
    np.testing.assert_array_equal(truth, data.validation.targets)
    return regression_metrics(truth, prediction), prediction


def train_seed(data: SequentialData, output_dir: str | Path, seed: int) -> dict[str, Any]:
    output = _output_directory(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = experiment_config(seed)
    preprocessing = data.source.preprocessor.export_metadata()
    if preprocessing != data.source.manifest["preprocessing"]:
        raise ValueError("Require unchanged saved training-only preprocessing.")
    seed_everything(seed, config.num_threads)
    model = build_model()  # never opens or resumes a prior checkpoint
    initial_dynamics = {name: value.detach().clone() for name, value in model.dynamics.named_parameters()}
    torch.save({"model_state": model.state_dict(), "initialization_seed": seed,
                "model_metadata": model_spec()}, output / "initial.pt")
    trainer = DiagnosticTrainer(model, config, output)
    diagnostic = diagnostic_partition(data.train)
    phase = "initial numerical diagnostic"
    started = time.perf_counter()
    try:
        initial_step = step_size_diagnostic(model, diagnostic)
        write_json(output / "step_size_initial.json", initial_step)
        assert_stable(initial_step)
        model.integrator.reset_stats()
        phase = "training and validation checkpoint selection"
        fit_started = time.perf_counter()
        fit = trainer.fit(make_sequential_loader(data.train, config, shuffle=True),
                          make_sequential_loader(data.validation, config, shuffle=False),
                          output, model_metadata=model_spec())
        fit_seconds = time.perf_counter() - fit_started
        fit_calls = model.integrator.stats()
        gradient_checks = trainer.diagnostics.metadata()
        if not all(gradient_checks["nonzero_gradient_seen_by_parameter"].values()):
            raise RuntimeError("Every model parameter tensor must receive nonzero training gradients.")
        dynamics_change = _parameter_delta(model, initial_dynamics)
        if not dynamics_change["every_dynamics_parameter_tensor_changed"]:
            raise RuntimeError("Restored dynamics must differ from fresh initialization.")
        phase = "restored numerical diagnostic"
        selected_step = step_size_diagnostic(model, diagnostic)
        write_json(output / "step_size_restored.json", selected_step)
        assert_stable(selected_step)  # BEFORE final performance interpretation
        phase = "restored validation reproduction"
        model.integrator.reset_stats()
        truth, prediction = trainer.predict(make_sequential_loader(data.validation, config, shuffle=False))
        evaluation_calls = model.integrator.stats()
        np.testing.assert_array_equal(truth, data.validation.targets)
        metrics = regression_metrics(truth, prediction)
        reloaded_metrics, reloaded = restore_validation(fit.checkpoint, data)
        np.testing.assert_array_equal(prediction, reloaded)
        if metrics != reloaded_metrics or metrics["rmse"] != fit.best_score:
            raise RuntimeError("Restored evaluations must exactly reproduce selected metrics/predictions.")
        prefix = prefix_check(model, diagnostic)
        if data.source.preprocessor.export_metadata() != preprocessing:
            raise RuntimeError("Saved preprocessing changed.")
        np.savez_compressed(output / "validation_predictions.npz", sample_indices=data.validation.sample_indices,
                            measured_sm3=data.validation.targets, prediction=prediction)
        result = {
            "seed": seed, "validation": metrics, "best_epoch": fit.best_epoch,
            "stopped_epoch": len(fit.history), "epochs_run": len(fit.history),
            "stop_reason": "epoch_budget" if len(fit.history) == config.max_epochs else "early_stopping",
            "training_seconds": fit_seconds, "total_seed_seconds": time.perf_counter() - started,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "solver_calls_during_fit_including_epoch_validation": fit_calls,
            "solver_calls_final_validation": evaluation_calls,
            "gradient_checks": gradient_checks, "dynamics_change": dynamics_change,
            "step_size_initial": initial_step, "step_size_restored": selected_step,
            "prefix_checks": prefix, "restored_predictions_and_metrics_exact": True,
            "preprocessing_unchanged": True, "fresh_initialization": True,
            "warm_start_from_smoke_or_sanity": False, "numerical_failures": [],
        }
        write_json(output / "validation_metrics.json", result)
        write_json(output / "epoch_diagnostics.json", trainer.epoch_diagnostics)
        return result
    except Exception as error:
        torch.save({"model_state": model.state_dict(), "model_metadata": model_spec(),
                    "diagnostic_only_failed_run": True}, output / "failed_state.pt")
        write_json(output / "failure.json", {
            "seed": seed, "phase": phase, "type": type(error).__name__, "message": str(error),
            "elapsed_seconds": time.perf_counter() - started,
            "solver_calls": model.integrator.stats(), "gradient_checks": trainer.diagnostics.metadata(),
            "silent_solver_changes": False, "performance_interpretation_permitted": False,
        })
        raise
    finally:
        trainer.diagnostics.close()


def load_comparisons(source, baseline_dir, ablation_dir, tabular_dir, recurrent_dir) -> dict:
    comparisons = _comparisons(source, {"baselines": Path(baseline_dir).resolve(),
                                       "ablations": Path(ablation_dir).resolve(),
                                       "tabular": Path(tabular_dir).resolve()})
    root = Path(recurrent_dir).resolve()
    run = json.loads((root / "run.json").read_text())
    if run["dataset_manifest_sha256"] != source.manifest_sha256:
        raise ValueError("Recurrent control must share the exact saved cohort.")
    files = ["summary.json"] + [f"seed_{seed}/{name}" for seed in SEEDS
                               for name in ("validation_predictions.npz", "training_config.json")]
    for relative in files:
        if sha256_file(root / relative) != run["artifact_sha256"][relative]:
            raise ValueError("Recurrent comparison artifact hash mismatch.")
        if relative.endswith(".npz"):
            with np.load(root / relative, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["sample_indices"], source.validation.sample_indices)
                np.testing.assert_array_equal(saved["measured_sm3"], source.validation.targets)
    summary = json.loads((root / "summary.json").read_text())
    controls, differences = {}, {}
    for seed in SEEDS:
        control = json.loads((root / f"seed_{seed}/training_config.json").read_text())
        expected = asdict(experiment_config(seed))
        diff = {name: {"recurrent": control.get(name), "latent_kan_ode": expected.get(name)}
                for name in set(control) | set(expected) if control.get(name) != expected.get(name)}
        if diff != {"gradient_clip": {"recurrent": None, "latent_kan_ode": 5.}}:
            raise ValueError("Unexpected recurrent-control training configuration; review before running.")
        controls[str(seed)], differences[str(seed)] = control, diff
    comparisons["recurrent_KAN"] = {"mean_and_sample_sd": summary["mean_and_sample_sd"],
                                    "seeds": [{"seed": row["seed"], "validation": row["validation"]}
                                              for row in summary["seeds"]]}
    comparisons["recurrent_actual_training_configs"] = controls
    comparisons["training_config_differences"] = differences
    comparisons["comparison_limitations"] = {
        "gradient_clipping": "recurrent none versus latent KAN-ODE norm5",
        "parameters": {"recurrent": 23169, "latent_kan_ode": 26257},
        "not_a_perfectly_matched_dynamics_control": True,
    }
    comparisons["provenance"]["recurrent"] = {"directory": str(root),
        "sha256": {name: sha256_file(root / name) for name in ("run.json", *files)}}
    return comparisons


def _plot(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for seed, ax in zip(SEEDS, axes):
        rows = json.loads((output / f"seed_{seed}/history.json").read_text())
        result = json.loads((output / f"seed_{seed}/validation_metrics.json").read_text())
        ax.plot([r["epoch"] for r in rows], [np.sqrt(r["training_loss"]) for r in rows], label="Training (online)")
        ax.plot([r["epoch"] for r in rows], [r["validation"]["rmse"] for r in rows], label="Validation")
        ax.axvline(result["best_epoch"], color="gray", linestyle=":", label="Selected checkpoint")
        ax.set(title=f"Initialization {seed}", xlabel="Epoch")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("RMSE (PHQ-9 points)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Conditional relative-month latent KAN-ODE; fixed RK4 step 0.25")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.svg")
    plt.close(fig)


def run_full(dataset_dir, baseline_dir, ablation_dir, tabular_dir, recurrent_dir, output_dir) -> dict:
    output = _output_directory(output_dir)
    source = load_baseline_data(dataset_dir)
    data = prepare_sequential_data(source)
    if (len(data.train.targets) != 5111 or len(data.validation.targets) != 851
            or len(np.unique(data.train.participants)) != 2025
            or len(np.unique(data.validation.participants)) != 338):
        raise ValueError("Require the exact saved 5111/851 training/validation cohort.")
    comparisons = load_comparisons(source, baseline_dir, ablation_dir, tabular_dir, recurrent_dir)
    output.mkdir(parents=True, exist_ok=True)
    preprocessing_hash = sha256_file(source.dataset_dir / "preprocessor.joblib")
    write_json(output / "config.json", {
        "model": model_spec(), "seeds": list(SEEDS),
        "training_by_seed": {str(seed): asdict(experiment_config(seed)) for seed in SEEDS},
        "dataset_protocol": source.manifest["dataset"], "saved_split": source.manifest["split"],
        "feature_names": source.preprocessor.feature_names,
        "minibatch_order_seed": 42, "preprocessing_refit": False,
        "calibration_or_test_payloads_consumed": False, "hyperparameter_search": False,
        "fresh_initialization": True, "resume_smoke_or_sanity": False,
        "diagnostic_subset": "same eight training samples as implementation smoke, reused across seeds",
        "material_step_discrepancy_rule": "stop if max prediction difference >0.01 PHQ point; never select step by validation accuracy",
        "recurrent_control_differences": comparisons["comparison_limitations"],
    })
    write_json(output / "comparisons.json", comparisons)
    np.savez_compressed(output / "used_split_indices.npz", train=data.train.sample_indices,
                        validation=data.validation.sample_indices,
                        diagnostic_train=diagnostic_partition(data.train).sample_indices)
    records = []
    try:
        for seed in SEEDS:
            print(f"Starting fresh latent KAN-ODE initialization {seed} on all 5111 training samples.", flush=True)
            record = train_seed(data, output / f"seed_{seed}", seed)
            records.append(record)
            write_json(output / "completed_seeds.json", records)
            print(f"Seed {seed}: best epoch {record['best_epoch']}, stopped {record['stopped_epoch']}, validation RMSE {record['validation']['rmse']:.6f}.", flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
            "completed_seeds": [row["seed"] for row in records],
            "performance_interpretation_permitted": False, "silent_solver_changes": False})
        raise
    if sha256_file(source.dataset_dir / "preprocessor.joblib") != preprocessing_hash:
        raise RuntimeError("Original saved preprocessing changed.")
    shutil.copyfile(source.dataset_dir / "preprocessor.joblib", output / "preprocessor.joblib")
    _plot(output)
    summary = {"workflow": model_spec()["workflow"], "seeds": records,
               "mean_and_sample_sd": aggregate_seed_metrics(records), "comparisons": comparisons,
               "counts": {name: {"samples": len(part.targets), "participants": len(np.unique(part.participants))}
                          for name, part in (("train", data.train), ("validation", data.validation))},
               "parameter_count": records[0]["parameter_count"],
               "total_fit_seconds": sum(row["training_seconds"] for row in records),
               "sd_interpretation": "initialization spread on one fixed split, not population uncertainty",
               "numerical_failures": []}
    write_json(output / "summary.json", summary)
    package = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), Path(__file__).with_name("forecasting_kan_ode.py"),
             Path(__file__).with_name("forecasting_sequential_kan.py"),
             Path(__file__).with_name("forecasting_baselines.py"), Path(__file__).with_name("forecasting_kan.py"),
             package / "models/latent_kan_ode.py", package / "models/sequential_kan.py",
             package / "models/kan_layer.py", package / "numerics/rk4.py",
             package / "training/gradient_diagnostics.py", package / "training/trainer.py",
             package / "evaluation/metrics.py"]
    write_json(output / "run.json", {
        "dataset_manifest_sha256": source.manifest_sha256, "dataset_dir": str(source.dataset_dir),
        "dataset_files_consumed": list(source.read_files),
        "consumed_sha256": {name: sha256_file(source.dataset_dir / name) for name in source.read_files},
        "original_input_hashes": source.manifest["inputs"],
        "original_input_hash_verification": "inherited; raw CSV never reopened",
        "builder_source_hashes": source.manifest["source_sha256"],
        "source_sha256": {str(path.relative_to(package)): sha256_file(path) for path in paths},
        "preprocessor_sha256": preprocessing_hash, "comparisons": comparisons["provenance"],
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "rk4_local_implementation": model_spec()["solver"]["implementation_version"]},
        "deterministic_algorithms": True, "device": "cpu", "num_threads": 2,
        "artifact_sha256": {str(path.relative_to(output)): sha256_file(path)
                            for path in sorted(output.rglob("*")) if path.is_file()},
    })
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-dir", "baseline-dir", "ablation-dir", "tabular-dir", "recurrent-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_full(args.dataset_dir, args.baseline_dir, args.ablation_dir,
                       args.tabular_dir, args.recurrent_dir, args.output)
    print(json.dumps({"mean_and_sample_sd": summary["mean_and_sample_sd"],
                      "total_fit_seconds": summary["total_fit_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
