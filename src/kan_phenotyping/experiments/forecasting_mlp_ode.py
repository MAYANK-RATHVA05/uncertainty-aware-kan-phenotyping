"""Fixed MLP-dynamics-only ODE control on frozen conditional forecasting data."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from torch import nn

from ..evaluation.metrics import regression_metrics
from ..models.kan_layer import KANLinear
from ..models.latent_mlp_ode import LatentMLPODE
from ..training.gradient_diagnostics import GradientDiagnostics
from ..training.trainer import Trainer, TrainingConfig, seed_everything, write_json
from .build_forecasting_dataset import _output_directory, sha256_file
from .forecasting_baselines import load_baseline_data
from .forecasting_kan import aggregate_seed_metrics
from .forecasting_kan_ode import prefix_check, step_size_diagnostic, subset_rows
from .forecasting_kan_ode_full import DiagnosticTrainer, assert_stable, diagnostic_partition, model_spec as kan_spec
from .forecasting_matched_recurrent import experiment_config, load_ode_reference
from .forecasting_sequential_kan import (
    SEEDS, SequentialTensorDataset, build_sequential_model,
    make_sequential_loader, prepare_sequential_data,
)


def model_spec():
    return {"name": "LatentMLPODE", "kwargs": {"step_size": .25}, "input_kind": "temporal",
        "workflow": "conditional_relative_month_MLP_dynamics_only_ODE_control",
        "shared_architecture": "static KAN63->16; shared packet KAN48->16; GRUCell16; KANhead16->1",
        "dynamics": "Linear(16,93)->Tanh->Linear(93,16)",
        "dynamics_parameters": 3085, "shared_parameters": 23169, "total_parameters": 26254,
        "initialization": "standard Linear; final weights x0.01, final bias zero; shared components first",
        "function_space_initialization_matched": False,
        "integration_schedule": [[0, 1], [1, 2], [2, 3]], "observations": [1, 2],
        "solver": "local classical RK4 1.0; step0.25; direct autograd",
        "target": "measured SM3 PHQ-9 severity, unbounded scalar output, endpoint MSE"}


def build_model():
    return LatentMLPODE(**model_spec()["kwargs"])


def initial_artifact(path, seed):
    path = Path(path)
    if path.name != "initial.pt" or path.parent.name != f"seed_{seed}":
        raise ValueError("Only the corresponding saved initial.pt artifact is allowed.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("initialization_seed") != seed or saved.get("model_metadata") != kan_spec():
        raise ValueError("Require the documented fresh KAN-ODE initialization.")
    return saved["model_state"]


def initialization_check(model, path, seed):
    expected = {k: v for k, v in initial_artifact(path, seed).items() if not k.startswith("dynamics.")}
    actual = {k: v for k, v in model.state_dict().items() if not k.startswith("dynamics.")}
    if set(expected) != set(actual) or len(actual) != 16:
        raise ValueError("Require exactly the unchanged 16 shared state tensors.")
    for name, value in actual.items():
        if (value.dtype != torch.float32 or expected[name].dtype != torch.float32
                or not torch.isfinite(value).all() or not torch.equal(value, expected[name])):
            raise ValueError(f"Shared initialization mismatch: {name}")
    counts = {"dynamics": sum(p.numel() for p in model.dynamics.parameters()),
              "total": sum(p.numel() for p in model.parameters())}
    if counts != {"dynamics": 3085, "total": 26254}:
        raise ValueError("The prescribed parameter counts must remain fixed.")
    return {"seed": seed, "shared_state_tensors_exact": 16, "shared_parameter_tensors": 13,
            "component_matches": {k: True for k in ("static_encoder", "wearable_encoder", "gru", "head")},
            "parameter_counts": counts, "weights_copied": False, "trained_checkpoint_used": False,
            "kan_initial_sha256": sha256_file(path)}


def derivative_diagnostic(model, train, kan_initial_path, seed):
    """Compare the two INITIAL fields at exactly the same static-derived h0."""
    state = initial_artifact(kan_initial_path, seed)
    with torch.random.fork_rng(devices=[]):
        kan = KANLinear(16, 16)
    kan.load_state_dict({k.removeprefix("dynamics."): v for k, v in state.items() if k.startswith("dynamics.")})
    with torch.no_grad():
        h0 = model.initial_state(torch.tensor(train.static))
        fields = {"mlp": model.dynamics(h0), "kan": kan(h0)}
        if not all(torch.isfinite(v).all() for v in fields.values()):
            raise FloatingPointError("Nonfinite initial derivative.")
        stats = {name: {"coordinate_rms": float(value.square().mean().sqrt()),
                        "maximum_absolute_coordinate": float(value.abs().max()),
                        "mean_state_l2": float(value.norm(dim=1).mean()),
                        "maximum_state_l2": float(value.norm(dim=1).max())}
                 for name, value in fields.items()}
        stats["difference_coordinate_rms"] = float((fields["mlp"] - fields["kan"]).square().mean().sqrt())
    return {"training_only": True, "samples": len(train.targets), "evaluation_state": "same h0=static_KAN(context)",
            "statistics": stats, "function_space_initialization_matched": False,
            "initialization_adjusted_using_validation": False}


def architecture_checks(model, train):
    w, c = torch.tensor(train.wearable), torch.tensor(train.static)
    zero = deepcopy(model)
    with torch.random.fork_rng(devices=[]):
        recurrent = build_sequential_model()
    recurrent.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("dynamics.")})
    with torch.no_grad():
        for parameter in zero.dynamics.parameters():
            parameter.zero_()
        a, b = zero(w, c), recurrent(w, c)
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    prefix = prefix_check(model, train)
    model.zero_grad(set_to_none=True)
    loss = nn.MSELoss()(model(w, c), torch.tensor(train.targets, dtype=torch.float32))
    loss.backward()
    gradients = {name: bool(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad))
                 for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    if not all(gradients.values()):
        raise RuntimeError("Finite nonzero gradients must reach all shared and dynamics parameters.")
    with torch.no_grad():
        first = model(w, c)
        model(w.flip(0), c.flip(0))
        repeated = model(w, c)
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        singles = torch.cat([model(w[i:i+1], c[i:i+1]) for i in range(len(w))])
        torch.testing.assert_close(first, singles, rtol=1e-5, atol=1e-5)
    return {"zero_dynamics_matches_recurrent": True, "zero_dynamics_max_difference": float((a-b).abs().max()),
            "prefix": prefix, "finite_nonzero_gradients": gradients, "no_state_carry": True}


def load_comparisons(data, kan_dir, matched_dir, ridge_dir):
    kan = load_ode_reference(data, kan_dir)
    references = {"KAN_ODE": {"seed_metrics": kan["seed_metrics"], "mean_and_sample_sd": kan["mean_and_sample_sd"]}}
    provenance = {"KAN_ODE": kan["consumed_sha256"]}
    for label, directory, is_ridge in (("matched_recurrent", matched_dir, False), ("ridge", ridge_dir, True)):
        root = Path(directory).resolve()
        run = json.loads((root / "run.json").read_text())
        if run["dataset_manifest_sha256"] != data.source.manifest_sha256:
            raise ValueError("Every reference must share the same frozen forecasting cohort.")
        files = ([f"{name}/{file}" for name in ("B", "D") for file in ("validation_metrics.json", "validation_predictions.npz")]
                 if is_ridge else ["summary.json"] + [f"seed_{s}/{f}" for s in SEEDS for f in (
                     "training_config.json", "validation_metrics.json", "validation_predictions.npz")])
        for name in files:
            if sha256_file(root / name) != run["artifact_sha256"][name]:
                raise ValueError("Reference artifact hash mismatch.")
        if not is_ridge:
            for source, digest in run["source_sha256"].items():
                if sha256_file(Path(__file__).resolve().parents[1] / source) != digest:
                    raise ValueError("Matched-control source provenance changed.")
        metrics = {}
        for key in (("B", "D") if is_ridge else SEEDS):
            folder = str(key) if is_ridge else f"seed_{key}"
            result = json.loads((root / folder / "validation_metrics.json").read_text())
            recorded = result["metrics"] if is_ridge else result["validation"]
            with np.load(root / folder / "validation_predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["sample_indices"], data.validation.sample_indices)
                np.testing.assert_array_equal(saved["measured_sm3"], data.validation.targets)
                computed = regression_metrics(saved["measured_sm3"], saved["prediction"])
            if any(not np.isclose(computed[m], recorded[m], rtol=0, atol=1e-12) for m in ("mae", "rmse", "r2")):
                raise ValueError("Reference metrics do not reproduce from aligned predictions.")
            metrics[str(key)] = recorded
            if not is_ridge and json.loads((root / folder / "training_config.json").read_text()) != asdict(experiment_config(key)):
                raise ValueError("Matched-control optimization settings differ.")
        references[label] = ({"metrics": metrics} if is_ridge else {"seed_metrics": metrics,
            "mean_and_sample_sd": aggregate_seed_metrics([{"validation": metrics[str(s)]} for s in SEEDS])})
        provenance[label] = {name: sha256_file(root / name) for name in ("run.json", *files)}
    return {"results": references, "consumed_sha256": provenance, "kan_directory": str(Path(kan_dir).resolve())}


def run_overfit_sanity(train, output_dir, *, max_steps=1000):
    output = _output_directory(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not 1 <= max_steps <= 1000:
        raise ValueError("Predeclared overfit cap is 1000 steps.")
    seed_everything(2026, 2)
    rows = np.sort(np.random.default_rng(2026).choice(len(train.targets), min(32, len(train.targets)), replace=False))
    part = subset_rows(train, rows)
    batch = SequentialTensorDataset(part).arrays
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    gradients = GradientDiagnostics(model)
    loss_fn = nn.MSELoss()
    started = time.perf_counter()
    with torch.no_grad():
        initial = float(loss_fn(model(batch["temporal"], batch["static"]), batch["target"]))
    if not np.isfinite(initial) or initial <= 0:
        raise FloatingPointError("Need positive finite initial sanity MSE.")
    history = [{"step": 0, "training_mse": initial}]
    passed = False
    try:
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch["temporal"], batch["static"]), batch["target"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite overfit loss.")
            loss.backward()
            gradients.finish_backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                mse = float(loss_fn(model(batch["temporal"], batch["static"]), batch["target"]))
            if not np.isfinite(mse):
                raise FloatingPointError("Nonfinite overfit evaluation.")
            history.append({"step": step, "training_mse": mse})
            if step % 100 == 0:
                print(f"Training-only MLP-dynamics overfit: step {step}, RMSE {np.sqrt(mse):.6f}.", flush=True)
            passed = mse <= .25**2 and mse <= .01*initial
            if passed:
                break
    finally:
        gradients.close()
        write_json(output / "history.json", history)
        write_json(output / "gradient_checks.json", gradients.metadata())
    result = {"training_only": True, "seed": 2026, "samples": len(part.targets), "steps": step,
        "max_steps": max_steps, "initial_mse": initial, "final_mse": mse, "final_rmse": float(np.sqrt(mse)),
        "seconds": time.perf_counter()-started, "passed": bool(passed and gradients.metadata()["all_parameters_received_nonzero_gradients"]),
        "pass_rule": "RMSE<=0.25 and MSE<=1% initial; all parameter tensors finite/nonzero gradients",
        "solver_calls": model.integrator.stats(), "weights_reused_for_experiment": False}
    write_json(output / "sanity.json", result)
    np.savez_compressed(output / "training_indices.npz", sample_indices=part.sample_indices)
    torch.save({"model_state": model.state_dict(), "diagnostic_only": True}, output / "sanity.pt")
    if not result["passed"]:
        raise RuntimeError("Training-only overfit failed; do not start full training.")
    return result


def restore_validation(checkpoint, data):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = TrainingConfig(**saved["training_config"])
    if saved["model_metadata"] != model_spec() or config != experiment_config(config.seed) or saved["input_kind"] != "temporal":
        raise ValueError("Require the fixed MLP-dynamics-only checkpoint.")
    seed_everything(config.seed, config.num_threads)
    model = build_model()
    model.load_state_dict(saved["model_state"])
    trainer = Trainer(model, config, input_kind="temporal")
    y, p = trainer.predict(make_sequential_loader(data.validation, config, shuffle=False))
    np.testing.assert_array_equal(y, data.validation.targets)
    return regression_metrics(y, p), p


def train_seed(data, output_dir, seed, kan_initial_path):
    output = _output_directory(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = experiment_config(seed)
    if data.source.preprocessor.export_metadata() != data.source.manifest["preprocessing"]:
        raise ValueError("Require unchanged saved preprocessing.")
    seed_everything(seed, config.num_threads)
    model = build_model()
    trainer = DiagnosticTrainer(model, config, output)
    diagnostic = diagnostic_partition(data.train)
    initial_weights = {k: v.detach().clone() for k, v in model.dynamics.named_parameters()}
    phase = "initialization and numerical checks"
    try:
        matching = initialization_check(model, kan_initial_path, seed)
        derivative = derivative_diagnostic(model, diagnostic, kan_initial_path, seed)
        write_json(output / "initialization_match.json", matching)
        write_json(output / "initial_derivatives.json", derivative)
        torch.save({"model_state": model.state_dict(), "seed": seed, "model_metadata": model_spec()}, output / "initial.pt")
        before = step_size_diagnostic(model, diagnostic)
        write_json(output / "step_size_initial.json", before)
        assert_stable(before)
        model.integrator.reset_stats()
        phase = "training"
        started = time.perf_counter()
        fit = trainer.fit(make_sequential_loader(data.train, config, shuffle=True),
            make_sequential_loader(data.validation, config, shuffle=False), output, model_metadata=model_spec())
        elapsed = time.perf_counter()-started
        calls = model.integrator.stats()
        gradients = trainer.diagnostics.metadata()
        trainer.diagnostics.close()
        if not gradients["all_parameters_received_nonzero_gradients"]:
            raise RuntimeError("All shared and dynamics tensors must receive nonzero gradients.")
        delta = {k: float((v.detach().double()-initial_weights[k].double()).norm()) for k, v in model.dynamics.named_parameters()}
        if not all(v > 0 for v in delta.values()):
            raise RuntimeError("All selected MLP dynamics tensors must change from initialization.")
        phase = "restored solver diagnostics"
        after = step_size_diagnostic(model, diagnostic)
        write_json(output / "step_size_restored.json", after)
        assert_stable(after)
        phase = "restoration reproduction"
        model.integrator.reset_stats()
        y, p = trainer.predict(make_sequential_loader(data.validation, config, shuffle=False))
        evaluation_calls = model.integrator.stats()
        np.testing.assert_array_equal(y, data.validation.targets)
        metrics = regression_metrics(y, p)
        fresh_metrics, fresh_p = restore_validation(fit.checkpoint, data)
        np.testing.assert_array_equal(p, fresh_p)
        if metrics != fresh_metrics or metrics["rmse"] != fit.best_score:
            raise RuntimeError("Fresh restoration must reproduce selected metrics exactly.")
        prefix = prefix_check(model, diagnostic)
        np.savez_compressed(output / "validation_predictions.npz", sample_indices=data.validation.sample_indices,
                            measured_sm3=data.validation.targets, prediction=p)
        result = {"seed": seed, "validation": metrics, "best_epoch": fit.best_epoch, "stopped_epoch": len(fit.history),
            "training_seconds": elapsed, "parameter_count": 26254, "dynamics_parameter_count": 3085,
            "solver_calls_during_fit_including_epoch_validation": calls, "solver_calls_final_validation": evaluation_calls,
            "initialization_match": matching, "initial_derivatives": derivative, "gradient_checks": gradients,
            "dynamics_l2_change_from_initial": delta, "step_size_initial": before, "step_size_restored": after,
            "prefix_checks": prefix, "restored_predictions_and_metrics_exact": True, "numerical_failures": []}
        write_json(output / "validation_metrics.json", result)
        write_json(output / "epoch_diagnostics.json", trainer.epoch_diagnostics)
        return result
    except Exception as error:
        torch.save({"model_state": model.state_dict(), "diagnostic_only_failed_run": True}, output / "failed_state.pt")
        write_json(output / "failure.json", {"seed": seed, "phase": phase, "type": type(error).__name__,
            "message": str(error), "gradient_checks": trainer.diagnostics.metadata(), "performance_interpretation_permitted": False})
        raise
    finally:
        trainer.diagnostics.close()


def metric_differences(records, references):
    if [r["seed"] for r in records] != list(SEEDS):
        raise ValueError("Require all three seeds in order.")
    result = []
    for row in records:
        seed = row["seed"]
        differences = {name: {m: row["validation"][m] - ref["seed_metrics"][str(seed)][m]
                             for m in ("mae", "rmse", "r2")}
                       for name, ref in references["results"].items() if "seed_metrics" in ref}
        differences.update({f"Ridge_{k}": {m: row["validation"][m] - value[m] for m in ("mae", "rmse", "r2")}
                            for k, value in references["results"]["ridge"]["metrics"].items()})
        result.append({"seed": seed, "MLP_ODE_minus_reference": differences})
    return result


def _plot(output):
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
        ax.set(title=f"Seed {seed}", xlabel="Epoch")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("RMSE (PHQ-9 points)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Conditional forecasting: fixed MLP dynamics, KAN encoders/head + GRU")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.svg")
    plt.close(fig)


def run_experiment(dataset_dir, kan_dir, matched_dir, ridge_dir, output_dir):
    output = _output_directory(output_dir)
    source = load_baseline_data(dataset_dir)
    data = prepare_sequential_data(source)
    if (len(data.train.targets), len(data.validation.targets), len(np.unique(data.train.participants)),
            len(np.unique(data.validation.participants))) != (5111, 851, 2025, 338):
        raise ValueError("Require exactly the saved 5111/851-sample forecasting cohort.")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    phase = "reference preflight"
    try:
        references = load_comparisons(data, kan_dir, matched_dir, ridge_dir)
        write_json(output / "comparisons.json", references)
        preprocessing_hash = sha256_file(source.dataset_dir / "preprocessor.joblib")
        write_json(output / "config.json", {"model": model_spec(), "seeds": list(SEEDS),
            "training_by_seed": {str(s): asdict(experiment_config(s)) for s in SEEDS},
            "dataset_protocol": source.manifest["dataset"], "saved_split": source.manifest["split"],
            "feature_names": source.preprocessor.feature_names, "minibatch_order_seed": 42,
            "device": "cpu", "precision": "float32", "num_threads": 2, "preprocessing_refit": False,
            "hyperparameter_search": False, "calibration_or_test_payloads_consumed": False,
            "material_solver_discrepancy_threshold": .01, "function_space_initialization_matched": False})
        diagnostic = diagnostic_partition(data.train)
        np.savez_compressed(output / "used_split_indices.npz", train=data.train.sample_indices,
                            validation=data.validation.sample_indices, diagnostic_train=diagnostic.sample_indices)
        phase = "all-seed pretraining checks"
        checks = []
        for seed in SEEDS:
            seed_everything(seed, 2)
            model = build_model()
            path = Path(kan_dir) / f"seed_{seed}/initial.pt"
            checks.append({"initialization": initialization_check(model, path, seed),
                "architecture": architecture_checks(model, diagnostic),
                "derivatives": derivative_diagnostic(model, diagnostic, path, seed)})
        write_json(output / "pretraining_checks.json", checks)
        phase = "separate training-only overfit"
        print("All shared initializations and architecture checks passed; starting separate overfit.", flush=True)
        sanity = run_overfit_sanity(data.train, output / "sanity")
        print(f"Overfit passed in {sanity['steps']} steps, RMSE {sanity['final_rmse']:.6f}.", flush=True)
        phase = "full training"
        for seed in SEEDS:
            print(f"Starting fresh MLP-dynamics ODE seed {seed}.", flush=True)
            row = train_seed(data, output / f"seed_{seed}", seed, Path(kan_dir) / f"seed_{seed}/initial.pt")
            records.append(row)
            write_json(output / "completed_seeds.json", records)
            print(f"Seed {seed}: best {row['best_epoch']}, stopped {row['stopped_epoch']}, RMSE {row['validation']['rmse']:.6f}.", flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"phase": phase, "type": type(error).__name__, "message": str(error),
            "completed_seeds": [r["seed"] for r in records], "performance_interpretation_permitted": False})
        raise
    if sha256_file(source.dataset_dir / "preprocessor.joblib") != preprocessing_hash:
        raise RuntimeError("Original preprocessing changed.")
    summary = {"model": model_spec(), "seeds": records, "mean_and_sample_sd": aggregate_seed_metrics(records),
        "comparisons": references, "per_seed_differences": metric_differences(records, references),
        "sanity": sanity, "total_fit_seconds": sum(r["training_seconds"] for r in records),
        "counts": {name: {"samples": len(p.targets), "participants": len(np.unique(p.participants))}
                   for name, p in (("train", data.train), ("validation", data.validation))},
        "numerical_failures": [], "sd_interpretation": "initialization spread, not population uncertainty or equivalence"}
    write_json(output / "summary.json", summary)
    shutil.copyfile(source.dataset_dir / "preprocessor.joblib", output / "preprocessor.joblib")
    _plot(output)
    package = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), package / "models/latent_mlp_ode.py", package / "models/latent_kan_ode.py",
        package / "models/sequential_kan.py", package / "models/kan_layer.py", package / "numerics/rk4.py",
        package / "training/trainer.py", package / "training/gradient_diagnostics.py"]
    paths.extend(Path(__file__).with_name(n) for n in ("forecasting_matched_recurrent.py", "forecasting_kan_ode_full.py",
        "forecasting_kan_ode.py", "forecasting_sequential_kan.py", "forecasting_baselines.py"))
    write_json(output / "run.json", {"dataset_manifest_sha256": source.manifest_sha256,
        "dataset_dir": str(source.dataset_dir), "dataset_files_consumed": list(source.read_files),
        "consumed_sha256": {n: sha256_file(source.dataset_dir / n) for n in source.read_files},
        "original_input_hashes": source.manifest["inputs"], "preprocessor_sha256": preprocessing_hash,
        "source_sha256": {str(p.relative_to(package)): sha256_file(p) for p in paths},
        "reference_consumed_sha256": references["consumed_sha256"],
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "rk4_local": "1.0"},
        "artifact_sha256": {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-dir", "kan-dir", "matched-dir", "ridge-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_experiment(args.dataset_dir, args.kan_dir, args.matched_dir, args.ridge_dir, args.output)
    print(json.dumps({"mean_and_sample_sd": result["mean_and_sample_sd"], "total_fit_seconds": result["total_fit_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
