"""Limited latent KAN-ODE implementation check, not a full forecasting study.

Frozen conditional relative-month train/validation arrays only. A separate
training-only overfit diagnostic precedes one fresh seed-42, two-epoch smoke
run on a deterministic subset of training participants. No tuning is exposed.
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
from torch import nn

from ..evaluation.metrics import regression_metrics
from ..models.latent_kan_ode import LatentKANODE
from ..numerics.rk4 import RK4_IMPLEMENTATION_VERSION
from ..training.trainer import Trainer, TrainingConfig, seed_everything, write_json
from .build_forecasting_dataset import _output_directory, sha256_file
from .forecasting_baselines import load_baseline_data
from .forecasting_kan import FiniteGradientChecks
from .forecasting_sequential_kan import (
    SequentialData, SequentialPartition, SequentialTensorDataset,
    make_sequential_loader, prepare_sequential_data,
)


SMOKE_SEED = 42
PARTICIPANT_BUDGET = 128
SANITY_SEED = 2026
DIAGNOSTIC_SAMPLES = 8
STEP_DIFFERENCE_WARNING = 0.01  # PHQ points; diagnostic flag, never tuning.


def model_spec() -> dict[str, Any]:
    return {
        "name": "LatentKANODE", "input_kind": "temporal",
        "kwargs": {"n_wearable_features": 48, "n_static_features": 63,
                   "hidden_width": 16, "grid_intervals": 8, "spline_degree": 3,
                   "grid_range": [-3.0, 3.0], "step_size": 0.25},
        "workflow": "conditional_relative_month_latent_KAN_ODE_smoke",
        "state_equation": "dh/dt = KANLinear(16,16)(h)",
        "state_order": ["h0", "h1_before", "h1_after", "h2_before", "h2_after", "h3"],
        "observation_updates": "shared KAN(48,16) and shared conventional GRUCell at SM1 and SM2",
        "static_context": "KAN(63,16) initializes h0 once, not a time step",
        "times": [0, 1, 2, 3], "time_units": "relative study-month coordinates, not dates",
        "target": "measured SM3 PHQ-9 severity; endpoint-only MSE",
        "output": "unbounded scalar, no clipping or rounding",
        "dynamics_initialization": {"base_and_spline_scale": 0.01, "bias": 0.0},
        "solver": {"algorithm": "local classical fixed-step RK4", "implementation_version": RK4_IMPLEMENTATION_VERSION,
                   "step_size": 0.25, "differentiation": "direct PyTorch autograd, not adjoint",
                   "final_partial_step": True, "zero_duration_identity": True,
                   "maintained_dependencies_checked_not_installed": ["torchdiffeq", "torchode", "torchdyn"]},
        "attribution": {"authors": ["Benjamin C. Koenig", "Suyong Kim", "Sili Deng"],
                        "paper": "https://arxiv.org/abs/2407.04192v3",
                        "equations": "Section II, equations 4-9, physical pages 4-5",
                        "adaptation_not_replication": True},
    }


def build_model() -> LatentKANODE:
    return LatentKANODE(**model_spec()["kwargs"])


def smoke_config() -> TrainingConfig:
    return TrainingConfig(task="regression", max_epochs=2, batch_size=128,
                          learning_rate=0.001, weight_decay=0.0001, patience=10,
                          seed=SMOKE_SEED, device="cpu", num_threads=2,
                          min_delta=0.0, gradient_clip=5.0, regression_loss="mse")


def subset_rows(partition: SequentialPartition, indices: np.ndarray) -> SequentialPartition:
    return SequentialPartition(**{name: getattr(partition, name)[indices].copy() for name in (
        "sample_indices", "participants", "wearable", "static", "targets",
    )})


def participant_subset(train: SequentialPartition, count: int = PARTICIPANT_BUDGET,
                       seed: int = SMOKE_SEED) -> SequentialPartition:
    """Select IDs without outcomes, retain every saved row for each selected ID."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Participant budget must be a positive integer.")
    participants = np.unique(train.participants)
    selected = np.random.default_rng(seed).permutation(participants)[:min(count, len(participants))]
    indices = np.flatnonzero(np.isin(train.participants, selected))
    result = subset_rows(train, indices)
    if set(result.participants) != set(selected):
        raise RuntimeError("Participant subset did not retain the selected groups.")
    return result


def prefix_check(model: LatentKANODE, train: SequentialPartition) -> dict[str, Any]:
    """Verify the first observed update without supplying the second packet."""
    model.eval()
    w = torch.tensor(train.wearable[:DIAGNOSTIC_SAMPLES], requires_grad=True)
    c = torch.tensor(train.static[:DIAGNOSTIC_SAMPLES], requires_grad=True)
    states = model.trace_states(w, c)
    gw, gc = torch.autograd.grad(states[:, 2].sum(), (w, c))
    changed = w.detach().clone()
    changed[:, 1] = -changed[:, 1] + 0.375
    with torch.no_grad():
        other = model.trace_states(changed, c.detach())
        h0 = model.initial_state(c.detach())
        prefix_only = model.step(w.detach()[:, 0], model.integrator.integrate(model.dynamics, h0, 0, 1))
    if (torch.count_nonzero(gw[:, 1]).item() or not torch.equal(states.detach()[:, :3], other[:, :3])
            or not torch.equal(prefix_only, states.detach()[:, 2])):
        raise RuntimeError("SM1 update obtained computational access to SM2.")
    if not torch.count_nonzero(gw[:, 0]).item() or not torch.count_nonzero(gc).item():
        raise RuntimeError("SM1 update should receive historical/context gradients.")
    return {"sm1_gradient_wrt_sm2_exactly_zero": True,
            "sm1_state_unchanged_when_sm2_changes": True,
            "prefix_without_sm2_matches": True,
            "sm1_and_context_gradients_nonzero": True}


def step_size_diagnostic(model: LatentKANODE, train: SequentialPartition) -> dict[str, Any]:
    """Same fixed training inputs and weights; no validation accuracy selection."""
    w = torch.tensor(train.wearable[:DIAGNOSTIC_SAMPLES])
    c = torch.tensor(train.static[:DIAGNOSTIC_SAMPLES])
    original_step = model.step_size
    original_weights = {key: value.clone() for key, value in model.state_dict().items()}
    predictions, states, calls = {}, {}, {}
    try:
        model.eval()
        with torch.no_grad():
            for step in (0.25, 0.125):
                model.step_size = step
                model.integrator.reset_stats()
                trace = model.trace_states(w, c)
                prediction = model.head(trace[:, -1]).squeeze(-1)
                if not torch.isfinite(trace).all() or not torch.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite step-size diagnostic.")
                states[str(step)] = trace[:, -1].clone()
                predictions[str(step)] = prediction.clone()
                calls[str(step)] = model.integrator.stats()
    finally:
        model.step_size = original_step
    for key, value in model.state_dict().items():
        if not torch.equal(value, original_weights[key]):
            raise RuntimeError("Numerical diagnostic must not change fitted parameters.")
    difference = predictions["0.25"] - predictions["0.125"]
    maximum = float(difference.abs().max())
    return {"training_only": True, "samples": len(w), "steps_compared": [0.25, 0.125],
            "prediction_max_absolute_difference": maximum,
            "prediction_rms_difference": float(torch.sqrt(torch.mean(difference.square()))),
            "final_hidden_max_absolute_difference": float((states["0.25"] - states["0.125"]).abs().max()),
            "maximum_absolute_final_hidden": {key: float(value.abs().max()) for key, value in states.items()},
            "warning_threshold_phq_points": STEP_DIFFERENCE_WARNING,
            "instability_warning": maximum > STEP_DIFFERENCE_WARNING,
            "all_states_and_predictions_finite": True,
            "solver_calls": calls, "training_step_size_unchanged": model.step_size == 0.25,
            "validation_used_for_step_selection": False, "parameters_unchanged": True}


def run_overfit_sanity(train: SequentialPartition, output_dir: str | Path,
                       *, max_steps: int = 1000) -> dict[str, Any]:
    """Separate train-only optimization check; fitted weights never reused."""
    output = _output_directory(output_dir)
    if not 1 <= max_steps <= 1000:
        raise ValueError("Sanity budget must be within its predeclared 1000-step cap.")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(SANITY_SEED, 2)
    rows = np.sort(np.random.default_rng(SANITY_SEED).choice(len(train.targets), min(32, len(train.targets)), replace=False))
    part = subset_rows(train, rows)
    batch = SequentialTensorDataset(part).arrays
    model = build_model()
    initial = {key: parameter.detach().clone() for key, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.MSELoss()
    gradients = FiniteGradientChecks(model)
    started = time.perf_counter()
    with torch.no_grad():
        initial_mse = float(loss_fn(model(batch["temporal"], batch["static"]), batch["target"]))
    if not np.isfinite(initial_mse) or initial_mse <= 0:
        raise RuntimeError("Sanity diagnostic requires positive finite initial MSE.")
    history = [{"step": 0, "training_mse": initial_mse}]
    passed = False
    try:
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch["temporal"], batch["static"]), batch["target"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite overfit loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                mse = float(loss_fn(model(batch["temporal"], batch["static"]), batch["target"]))
            if not np.isfinite(mse):
                raise FloatingPointError("Nonfinite overfit evaluation.")
            history.append({"step": step, "training_mse": mse})
            if step % 100 == 0:
                print(f"Overfit diagnostic: step {step}, training RMSE {np.sqrt(mse):.6f}.", flush=True)
            passed = mse <= 0.25 ** 2 and mse <= initial_mse * 0.01
            if passed:
                break
    finally:
        gradients.close()
    gradient_checks = gradients.metadata()
    changed = {key: not torch.equal(initial[key], value.detach()) for key, value in model.named_parameters()}
    result = {
        "training_only": True, "seed": SANITY_SEED, "samples": len(part.targets),
        "max_steps": max_steps, "steps_run": step, "initial_mse": initial_mse,
        "final_mse": mse, "final_rmse": float(np.sqrt(mse)),
        "seconds": time.perf_counter() - started,
        "pass_rule": "RMSE <=0.25, MSE <=1% initial, all parameter gradients finite/nonzero and changed weights",
        "passed": bool(passed and all(changed.values()) and all(gradient_checks["nonzero_gradient_seen_by_parameter"].values())),
        "gradient_checks": gradient_checks, "parameters_changed": changed,
        "solver_calls": model.integrator.stats(), "numerical_failures": [],
        "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
        "gradient_norm_clip": 5.0, "warm_start_smoke": False,
    }
    write_json(output / "sanity.json", result)
    write_json(output / "history.json", history)
    np.savez_compressed(output / "training_subset_indices.npz", sample_indices=part.sample_indices)
    torch.save({"model_state": model.state_dict(), "model_metadata": model_spec(),
                "diagnostic_only": True}, output / "sanity.pt")
    if not result["passed"]:
        raise RuntimeError("Training-only overfit diagnostic failed; do not start smoke training.")
    return result


def restore_validation(checkpoint: str | Path, data: SequentialData) -> tuple[dict, np.ndarray]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved["model_metadata"] != model_spec() or saved["input_kind"] != "temporal"
            or saved["training_config"] != asdict(smoke_config())):
        raise ValueError("Require the fixed two-epoch latent KAN-ODE smoke checkpoint.")
    seed_everything(SMOKE_SEED, 2)
    model = build_model()
    model.load_state_dict(saved["model_state"])
    trainer = Trainer(model, smoke_config(), input_kind="temporal")
    truth, prediction = trainer.predict(make_sequential_loader(data.validation, smoke_config(), shuffle=False))
    np.testing.assert_array_equal(truth, data.validation.targets)
    return regression_metrics(truth, prediction), prediction


def train_smoke(data: SequentialData, train: SequentialPartition,
                output_dir: str | Path) -> dict[str, Any]:
    output = _output_directory(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    before = data.source.preprocessor.export_metadata()
    if before != data.source.manifest["preprocessing"]:
        raise ValueError("Require unchanged training-fitted preprocessing.")
    # Validate subset membership and exact content; callers cannot substitute a
    # different cohort, resampled target, or participant's partial history.
    canonical = participant_subset(data.train)
    for key in ("sample_indices", "participants", "wearable", "static", "targets"):
        np.testing.assert_array_equal(getattr(train, key), getattr(canonical, key))
    seed_everything(SMOKE_SEED, 2)  # fresh after sanity, no weight transfer
    model = build_model()
    torch.save({"model_state": model.state_dict(), "model_metadata": model_spec(),
                "initialization_seed": SMOKE_SEED}, output / "initial.pt")
    before_step = step_size_diagnostic(model, train)
    write_json(output / "step_size_before.json", before_step)
    prefix_before = prefix_check(model, train)
    trainer = Trainer(model, smoke_config(), input_kind="temporal")
    gradients = FiniteGradientChecks(model)
    model.integrator.reset_stats()
    started = time.perf_counter()
    try:
        fit = trainer.fit(make_sequential_loader(train, smoke_config(), shuffle=True),
                          make_sequential_loader(data.validation, smoke_config(), shuffle=False),
                          output, model_metadata=model_spec())
    finally:
        gradients.close()
    elapsed = time.perf_counter() - started
    fit_calls = model.integrator.stats()
    gradient_checks = gradients.metadata()
    if not all(gradient_checks["nonzero_gradient_seen_by_parameter"].values()):
        raise RuntimeError("Every model parameter tensor must receive finite nonzero training gradients.")
    truth, prediction = trainer.predict(make_sequential_loader(data.validation, smoke_config(), shuffle=False))
    np.testing.assert_array_equal(truth, data.validation.targets)
    metrics = regression_metrics(truth, prediction)
    restored_metrics, restored = restore_validation(fit.checkpoint, data)
    np.testing.assert_array_equal(prediction, restored)
    if metrics != restored_metrics or metrics["rmse"] != fit.best_score:
        raise RuntimeError("Restored checkpoint must reproduce selected metrics exactly.")
    after_step = step_size_diagnostic(model, train)
    write_json(output / "step_size_after.json", after_step)
    prefix_after = prefix_check(model, train)
    if data.source.preprocessor.export_metadata() != before:
        raise RuntimeError("Saved preprocessing changed during the smoke run.")
    np.savez_compressed(output / "validation_predictions.npz", sample_indices=data.validation.sample_indices,
                        measured_sm3=data.validation.targets, prediction=prediction)
    result = {
        "preliminary_smoke_only": True, "seed": SMOKE_SEED, "epochs_run": len(fit.history),
        "best_epoch": fit.best_epoch, "validation": metrics, "training_seconds": elapsed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "solver_calls_during_fit_including_epoch_validation": fit_calls,
        "gradient_checks": gradient_checks, "step_size_before": before_step, "step_size_after": after_step,
        "prefix_before": prefix_before, "prefix_after": prefix_after,
        "restored_predictions_match": True, "preprocessing_unchanged": True,
        "fresh_initialization_not_sanity_weights": True, "numerical_failures": [],
        "selection_metric": "validation_rmse within the two smoke epochs",
    }
    write_json(output / "validation_metrics.json", result)
    return result


def _plot(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    rows = json.loads((output / "smoke/history.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["epoch"] for r in rows], [np.sqrt(r["training_loss"]) for r in rows], "o-", label="Training (online)")
    ax.plot([r["epoch"] for r in rows], [r["validation"]["rmse"] for r in rows], "o-", label="Validation")
    ax.set(xlabel="Smoke epoch", ylabel="RMSE (PHQ-9 points)",
           title="Latent KAN-ODE: preliminary two-epoch smoke run", xticks=[1, 2])
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.svg")
    plt.close(fig)


def run_smoke(dataset_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = _output_directory(output_dir)
    source = load_baseline_data(dataset_dir)
    data = prepare_sequential_data(source)
    train = participant_subset(data.train)
    original_preprocessor = source.dataset_dir / "preprocessor.joblib"
    preprocessing_hash = sha256_file(original_preprocessor)
    counts = {name: {"samples": len(part.targets), "participants": len(np.unique(part.participants))}
              for name, part in (("saved_training", data.train), ("smoke_training", train),
                                 ("validation", data.validation))}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {
        "model": model_spec(), "training": asdict(smoke_config()),
        "participant_selection": {"budget": PARTICIPANT_BUDGET, "seed": SMOKE_SEED,
                                  "method": "permutation of sorted unique saved training IDs; retain all their rows",
                                  "outcomes_used": False},
        "counts": counts, "dataset_protocol": source.manifest["dataset"],
        "saved_split": source.manifest["split"],
        "feature_names": {"wearable": source.preprocessor.feature_names["wearable"],
                          "static": source.preprocessor.feature_names["demographics"] + source.preprocessor.feature_names["sm0_phq"]},
        "preprocessing_refit": False, "calibration_or_test_payloads_consumed": False,
        "raw_dataset_reopened": False, "available_at_timestamps_verified": False,
        "prospective_clinical_validation": False, "hyperparameter_search": False,
        "full_three_seed_training": False, "conformal_implemented": False,
    })
    np.savez_compressed(output / "used_split_indices.npz", saved_train=data.train.sample_indices,
                        smoke_train=train.sample_indices, validation=data.validation.sample_indices,
                        diagnostic_train=train.sample_indices[:DIAGNOSTIC_SAMPLES])
    phase = "training-only overfit diagnostic"
    try:
        print("Starting separate training-only latent KAN-ODE overfit diagnostic.", flush=True)
        sanity = run_overfit_sanity(data.train, output / "sanity")
        print(f"Overfit passed: {sanity['steps_run']} steps, RMSE {sanity['final_rmse']:.6f}.", flush=True)
        phase = "two-epoch seed-42 smoke"
        print(f"Smoke training: {counts['smoke_training']['participants']} participants, {len(train.targets)} samples, two epochs.", flush=True)
        smoke = train_smoke(data, train, output / "smoke")
    except Exception as error:
        write_json(output / "failure.json", {"phase": phase, "type": type(error).__name__,
                                             "message": str(error), "silent_solver_changes": False})
        raise
    if sha256_file(original_preprocessor) != preprocessing_hash:
        raise RuntimeError("Original fitted preprocessing file changed.")
    shutil.copyfile(original_preprocessor, output / "preprocessor.joblib")
    _plot(output)
    summary = {"workflow": model_spec()["workflow"], "counts": counts,
               "sanity": sanity, "smoke": smoke, "numerical_failures": [],
               "interpretation": "implementation smoke only; not evidence of predictive improvement or clinically meaningful dynamics"}
    write_json(output / "summary.json", summary)
    package = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), Path(__file__).with_name("forecasting_sequential_kan.py"),
             Path(__file__).with_name("forecasting_baselines.py"), Path(__file__).with_name("forecasting_kan.py"),
             package / "models/latent_kan_ode.py", package / "models/sequential_kan.py",
             package / "models/kan_layer.py", package / "numerics/rk4.py", package / "training/trainer.py",
             package / "evaluation/metrics.py"]
    write_json(output / "run.json", {
        "dataset_dir": str(source.dataset_dir), "dataset_manifest_sha256": source.manifest_sha256,
        "dataset_files_consumed": list(source.read_files),
        "consumed_sha256": {name: sha256_file(source.dataset_dir / name) for name in source.read_files},
        "original_input_hashes": source.manifest["inputs"],
        "builder_source_hashes": source.manifest["source_sha256"],
        "source_sha256": {str(path.relative_to(package)): sha256_file(path) for path in paths},
        "preprocessor_sha256": preprocessing_hash,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "rk4_local_implementation": RK4_IMPLEMENTATION_VERSION},
        "device": "cpu", "num_threads": 2, "deterministic_algorithms": True,
        "artifact_sha256": {str(path.relative_to(output)): sha256_file(path)
                            for path in sorted(output.rglob("*")) if path.is_file()},
    })
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_smoke(args.dataset_dir, args.output)
    print(json.dumps({"counts": summary["counts"], "smoke": summary["smoke"]["validation"],
                      "training_seconds": summary["smoke"]["training_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
