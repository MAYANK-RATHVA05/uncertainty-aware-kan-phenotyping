"""Fixed KAN-encoded GRU baseline on frozen conditional forecasting samples.

The conventional GRU performs two observation updates, SM1 then SM2. Static
context initializes its state once. No ODE, interpolation, or new split occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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
from ..models.sequential_kan import KANEncodedRecurrentBaseline
from ..training.trainer import Trainer, TrainingConfig, make_loader, seed_everything, write_json
from .build_forecasting_dataset import _output_directory, sha256_file
from .forecasting_baselines import SavedBaselineData, load_baseline_data
from .forecasting_kan import FiniteGradientChecks, aggregate_seed_metrics, experiment_config


SEEDS = (42, 43, 44)
SHUFFLE_SEED = 42
SANITY_SEED = 2026


@dataclass(frozen=True)
class SequentialPartition:
    sample_indices: np.ndarray
    participants: np.ndarray
    wearable: np.ndarray
    static: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class SequentialData:
    train: SequentialPartition
    validation: SequentialPartition
    source: SavedBaselineData


def model_spec() -> dict[str, Any]:
    return {
        "name": "KANEncodedRecurrentBaseline", "input_kind": "temporal",
        "kwargs": {"n_wearable_features": 48, "n_static_features": 63, "hidden_width": 16,
                   "grid_intervals": 8, "spline_degree": 3, "grid_range": [-3.0, 3.0]},
        "workflow": "conditional_relative_month_KAN_encoded_recurrent_forecasting",
        "target": "measured SM3 PHQ-9 severity, not change",
        "updates": "static KAN -> h0; shared packet KAN and conventional GRUCell at SM1 then SM2; KAN head",
        "output": "unbounded scalar; no sigmoid, clipping, or rounding",
        "trainer_dispatch": "sequence and static arguments; no temporal convolution or ODE",
    }


def build_sequential_model() -> KANEncodedRecurrentBaseline:
    return KANEncodedRecurrentBaseline(**model_spec()["kwargs"])


def prepare_sequential_data(saved: SavedBaselineData) -> SequentialData:
    """Restore the two saved packet axes; do not transform or refit anything."""
    names = saved.preprocessor.feature_names
    if (len(names["wearable"]) != 48 or len(names["demographics"]) != 62
            or names["sm0_phq"] != ["measured_sm0_phq9_standardized"]):
        raise ValueError("Require frozen 48-column packets and 63-column demographic/SM0 context.")
    expected = ([f"SM{month}__{name}" for month in (1, 2) for name in names["wearable"]]
                + [f"baseline_demographic__{name}" for name in names["demographics"]] + names["sm0_phq"])
    if names["tabular"] != expected or len(set(expected)) != 159:
        raise ValueError("Saved metadata must preserve SM1, SM2, enrollment demographics, and measured SM0 order.")
    blocks = {}
    for key in ("train", "validation"):
        original = getattr(saved, key)
        n = len(original.targets)
        if original.tabular.shape != (n, 159):
            raise ValueError("Saved flat inputs must have shape [N,159].")
        x = np.asarray(original.tabular, dtype=np.float32)
        blocks[key] = SequentialPartition(
            original.sample_indices.copy(), original.participants.copy(),
            x[:, :96].reshape(n, 2, 48).copy(), x[:, 96:].copy(), original.targets.copy(),
        )
        SequentialTensorDataset(blocks[key])
        np.testing.assert_array_equal(np.concatenate([
            blocks[key].wearable.reshape(n, 96), blocks[key].static,
        ], axis=1), np.asarray(original.tabular, dtype=np.float32))
    return SequentialData(blocks["train"], blocks["validation"], saved)


class SequentialTensorDataset(Dataset):
    def __init__(self, partition: SequentialPartition):
        y = np.asarray(partition.targets, dtype=np.float32)
        n = len(y) if y.ndim else 0
        if (n == 0 or y.shape != (n,) or np.asarray(partition.wearable).shape != (n, 2, 48)
                or np.asarray(partition.static).shape != (n, 63)
                or np.asarray(partition.sample_indices).shape != (n,)
                or np.asarray(partition.participants).shape != (n,)):
            raise ValueError("Require aligned wearable [N,2,48], static [N,63], and measured targets [N].")
        arrays = {"temporal": partition.wearable, "static": partition.static, "target": y}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Saved predictors and measured targets must be finite.")
        if not ((y >= 0) & (y <= 27) & (y == np.rint(y))).all():
            raise ValueError("Keep measured SM3 severity in 0..27; do not change or impute targets.")
        self.arrays = {key: torch.tensor(value, dtype=torch.float32) for key, value in arrays.items()}

    def __len__(self) -> int:
        return len(self.arrays["target"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.arrays.items()}


def make_sequential_loader(partition: SequentialPartition, config: TrainingConfig, *, shuffle: bool):
    return make_loader(SequentialTensorDataset(partition), replace(config, seed=SHUFFLE_SEED), shuffle=shuffle)


def prefix_causality_check(model: KANEncodedRecurrentBaseline, train: SequentialPartition) -> dict[str, Any]:
    """Check actual prefix dependencies, independent of forecast performance."""
    model.eval()
    wearable = torch.tensor(train.wearable[:8], dtype=torch.float32, requires_grad=True)
    static = torch.tensor(train.static[:8], dtype=torch.float32, requires_grad=True)
    events, encoder_ids, gru_ids = [], [], []
    def context_hook(module, args):
        events.append("initialize_static")
    def encoder_hook(module, args):
        position = len(encoder_ids)
        np.testing.assert_array_equal(args[0].detach().numpy(), wearable[:, position, :].detach().numpy())
        encoder_ids.append(id(module))
        events.append(f"encode_SM{position + 1}")
    def gru_hook(module, args):
        position = len(gru_ids)
        gru_ids.append(id(module))
        events.append(f"update_SM{position + 1}")
    handles = [model.static_encoder.register_forward_pre_hook(context_hook),
               model.wearable_encoder.register_forward_pre_hook(encoder_hook),
               model.gru.register_forward_pre_hook(gru_hook)]
    try:
        states = model.trace_states(wearable, static)
    finally:
        for handle in handles:
            handle.remove()
    expected = ["initialize_static", "encode_SM1", "update_SM1", "encode_SM2", "update_SM2"]
    if events != expected or len(set(encoder_ids)) != 1 or len(set(gru_ids)) != 1:
        raise RuntimeError("Encoder/update sharing or sequential ordering was violated.")
    gradient_wearable, gradient_static = torch.autograd.grad(states[:, 1].sum(), (wearable, static))
    if torch.count_nonzero(gradient_wearable[:, 1]).item():
        raise RuntimeError("SM1 state depends on the future SM2 packet.")
    changed = wearable.detach().clone()
    changed[:, 1] = -changed[:, 1] + 0.375
    with torch.no_grad():
        other = model.trace_states(changed, static.detach())
        prefix_only = model.step(wearable.detach()[:, 0], model.initial_state(static.detach()))
    if not torch.equal(states.detach()[:, :2], other[:, :2]) or not torch.equal(prefix_only, states.detach()[:, 1]):
        raise RuntimeError("SM1 prefix must be identical when SM2 changes or is unavailable to the prefix API.")
    return {
        "events": events, "static_initializations": 1,
        "wearable_encoder_shared": True, "gru_shared": True,
        "sm1_state_unchanged_when_sm2_changes": True,
        "prefix_only_api_matches_first_update": True,
        "sm1_gradient_wrt_sm2_exactly_zero": True,
        "sm1_gradient_wrt_sm1_nonzero": bool(torch.count_nonzero(gradient_wearable[:, 0]).item()),
        "sm1_gradient_wrt_static_nonzero": bool(torch.count_nonzero(gradient_static).item()),
        "reversal_performance_requirement": False,
    }


def _subset(partition: SequentialPartition, indices: np.ndarray) -> SequentialPartition:
    return SequentialPartition(**{key: getattr(partition, key)[indices].copy() for key in (
        "sample_indices", "participants", "wearable", "static", "targets",
    )})


def run_overfit_sanity(train: SequentialPartition, output_dir: str | Path,
                       *, max_steps: int = 1000) -> dict[str, Any]:
    output = _output_directory(output_dir)
    if not 1 <= max_steps <= 1000:
        raise ValueError("The predeclared sanity budget is 1..1000 steps.")
    seed_everything(SANITY_SEED, 2)
    indices = np.sort(np.random.default_rng(SANITY_SEED).choice(len(train.targets), min(32, len(train.targets)), replace=False))
    subset = _subset(train, indices)
    batch = SequentialTensorDataset(subset).arrays
    model = build_sequential_model()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.MSELoss()
    checks = FiniteGradientChecks(model)
    started = time.perf_counter()
    with torch.no_grad():
        initial_mse = float(criterion(model(batch["temporal"], batch["static"]), batch["target"]))
    if not np.isfinite(initial_mse) or initial_mse <= 0:
        raise RuntimeError("Sanity check needs finite positive initial MSE.")
    history = [{"step": 0, "training_mse": initial_mse}]
    passed = False
    try:
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["temporal"], batch["static"])
            loss = criterion(prediction, batch["target"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite sanity loss.")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                mse = float(criterion(model(batch["temporal"], batch["static"]), batch["target"]))
            if not np.isfinite(mse):
                raise FloatingPointError("Nonfinite sanity evaluation.")
            history.append({"step": step, "training_mse": mse})
            passed = mse <= 0.25 ** 2 and mse <= initial_mse * 0.01
            if passed:
                break
    finally:
        checks.close()
    gradient_checks = checks.metadata()
    changed = any(not torch.equal(initial[name], parameter.detach()) for name, parameter in model.named_parameters())
    result = {"training_only": True, "seed": SANITY_SEED, "samples": len(subset.targets),
              "max_steps": max_steps, "steps_run": step, "initial_mse": initial_mse, "final_mse": mse,
              "final_rmse": float(np.sqrt(mse)), "seconds": time.perf_counter() - started,
              "pass_rule": "RMSE <=0.25, MSE <=1% initial, finite/nonzero gradients, changed parameters",
              "passed": bool(passed and changed and all(gradient_checks["nonzero_gradient_seen_by_parameter"].values())),
              "gradient_checks": gradient_checks, "parameters_changed": changed,
              "warm_start_experiment": False,
              "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "sanity.json", result)
    write_json(output / "history.json", history)
    np.savez_compressed(output / "training_subset_indices.npz", sample_indices=subset.sample_indices)
    torch.save({"model_state": model.state_dict(), "model_metadata": model_spec(), "diagnostic_only": True}, output / "sanity.pt")
    if not result["passed"]:
        raise RuntimeError("Training-only overfit sanity failed; do not launch the experiments.")
    return result


def restore_validation(checkpoint_path: str | Path, data: SequentialData) -> tuple[dict, np.ndarray]:
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["model_metadata"] != model_spec() or saved["input_kind"] != "temporal":
        raise ValueError("Require the fixed KAN-encoded recurrent checkpoint.")
    config = TrainingConfig(**saved["training_config"])
    if config.task != "regression" or config.regression_loss != "mse":
        raise ValueError("Measured severity requires MSE regression.")
    seed_everything(config.seed, config.num_threads)
    model = build_sequential_model()
    model.load_state_dict(saved["model_state"])
    trainer = Trainer(model, config, input_kind="temporal")
    truth, predictions = trainer.predict(make_sequential_loader(data.validation, config, shuffle=False))
    np.testing.assert_array_equal(truth, data.validation.targets)
    return regression_metrics(truth, predictions), predictions


def train_seed(data: SequentialData, output_dir: str | Path, seed: int,
               config: TrainingConfig | None = None) -> dict[str, Any]:
    output = _output_directory(output_dir)
    config = experiment_config(seed) if config is None else config
    if config.seed != seed or config.task != "regression" or config.regression_loss != "mse":
        raise ValueError("Initialization seed and measured-severity MSE configuration must agree.")
    before = data.source.preprocessor.export_metadata()
    if before != data.source.manifest["preprocessing"]:
        raise ValueError("Require unchanged saved training-only preprocessing.")
    seed_everything(seed, config.num_threads)
    model = build_sequential_model()
    trainer = Trainer(model, config, input_kind="temporal")
    checks = FiniteGradientChecks(model)
    started = time.perf_counter()
    try:
        fitted = trainer.fit(make_sequential_loader(data.train, config, shuffle=True),
                             make_sequential_loader(data.validation, config, shuffle=False),
                             output, model_metadata=model_spec())
    finally:
        checks.close()
    elapsed = time.perf_counter() - started
    truth, predictions = trainer.predict(make_sequential_loader(data.validation, config, shuffle=False))
    np.testing.assert_array_equal(truth, data.validation.targets)
    metrics = regression_metrics(truth, predictions)
    restored_metrics, restored = restore_validation(fitted.checkpoint, data)
    np.testing.assert_array_equal(predictions, restored)
    if metrics != restored_metrics or metrics["rmse"] != fitted.best_score:
        raise RuntimeError("Restored checkpoint must reproduce selected predictions and metrics.")
    if data.source.preprocessor.export_metadata() != before:
        raise RuntimeError("Saved preprocessing changed during training.")
    prefix_checks = prefix_causality_check(model, data.train)
    np.savez_compressed(output / "validation_predictions.npz", sample_indices=data.validation.sample_indices,
                        measured_sm3=data.validation.targets, prediction=predictions)
    result = {"seed": seed, "validation": metrics, "parameter_count": sum(p.numel() for p in model.parameters()),
              "best_epoch": fitted.best_epoch, "epochs_run": len(fitted.history), "training_seconds": elapsed,
              "gradient_checks": checks.metadata(), "prefix_checks": prefix_checks,
              "restored_predictions_match": True, "preprocessing_unchanged": True,
              "shuffle_seed": SHUFFLE_SEED, "warm_start_from_sanity": False,
              "selection_metric": "validation_rmse"}
    write_json(output / "validation_metrics.json", result)
    return result


def _comparisons(source: SavedBaselineData, roots: dict[str, Path]) -> dict[str, Any]:
    provenance, loaded = {}, {}
    for name, root in roots.items():
        run = json.loads((root / "run.json").read_text())
        if run["dataset_manifest_sha256"] != source.manifest_sha256:
            raise ValueError("Every comparison must share the frozen dataset manifest.")
        report_name = "validation_metrics.json" if name == "baselines" else "summary.json"
        files = [report_name]
        if name == "baselines":
            files.append("validation_predictions.npz")
        elif name == "ablations":
            files += [f"{key}/validation_predictions.npz" for key in ("B", "D")]
        else:
            files += [f"seed_{seed}/validation_predictions.npz" for seed in SEEDS]
        for relative in files:
            if sha256_file(root / relative) != run["artifact_sha256"][relative]:
                raise ValueError("Existing comparison artifact hash mismatch.")
            if relative.endswith(".npz"):
                with np.load(root / relative, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["sample_indices"], source.validation.sample_indices)
                    np.testing.assert_array_equal(saved["measured_sm3"], source.validation.targets)
        loaded[name] = json.loads((root / report_name).read_text())
        provenance[name] = {"directory": str(root), "sha256": {
            relative: sha256_file(root / relative) for relative in ("run.json", *files)}}
    if loaded["baselines"]["metrics"]["ridge"] != loaded["ablations"]["setups"]["D"]["metrics"]:
        raise ValueError("Original full Ridge and ablation D must agree.")
    return {"persistence": loaded["baselines"]["metrics"]["persistence"],
            "ridge_B": loaded["ablations"]["setups"]["B"]["metrics"],
            "ridge_D": loaded["baselines"]["metrics"]["ridge"],
            "tabular_KAN": {"mean_and_sample_sd": loaded["tabular"]["mean_and_sample_sd"],
                            "seeds": [{"seed": row["seed"], "validation": row["validation"]}
                                      for row in loaded["tabular"]["seeds"]]},
            "provenance": provenance}


def _plot_curves(output: Path) -> None:
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
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("RMSE (PHQ-9 points)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Conditional relative-month forecasting: KAN encoders + conventional GRU")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.svg")
    plt.close(fig)


def run_forecasting_sequential_kan(dataset_dir: str | Path, baseline_dir: str | Path,
                                  ablation_dir: str | Path, tabular_kan_dir: str | Path,
                                  output_dir: str | Path) -> dict[str, Any]:
    output = _output_directory(output_dir)
    source = load_baseline_data(dataset_dir)
    data = prepare_sequential_data(source)
    comparisons = _comparisons(source, {name: Path(path).resolve() for name, path in (
        ("baselines", baseline_dir), ("ablations", ablation_dir), ("tabular", tabular_kan_dir))})
    preprocessor = source.dataset_dir / "preprocessor.joblib"
    preprocessing_hash = sha256_file(preprocessor)
    output.mkdir(parents=True, exist_ok=True)
    configuration = {
        "workflow": model_spec()["workflow"], "model": model_spec(), "seeds": list(SEEDS),
        "training_by_seed": {str(seed): asdict(experiment_config(seed)) for seed in SEEDS},
        "shuffle_seed": SHUFFLE_SEED, "optimizer": "AdamW", "loss": "MSE", "hyperparameter_search": False,
        "feature_names": {"wearable": source.preprocessor.feature_names["wearable"],
                          "static": source.preprocessor.feature_names["demographics"] + source.preprocessor.feature_names["sm0_phq"]},
        "sequence_positions": [1, 2], "static_is_time_step": False,
        "input_shapes": {"wearable": ["N", 2, 48], "static": ["N", 63], "target": ["N"]},
        "dataset_protocol": source.manifest["dataset"], "saved_split": source.manifest["split"],
        "preprocessing_refit": False, "prediction_clipping_or_rounding": False,
        "calibration_or_test_payloads_consumed": False, "available_at_timestamps_verified": False,
        "prospective_clinical_validation": False, "ode_or_conformal_implemented": False,
    }
    write_json(output / "config.json", configuration)
    write_json(output / "comparisons.json", comparisons)
    print("Starting separate training-only recurrent overfit sanity check.", flush=True)
    sanity = run_overfit_sanity(data.train, output / "sanity")
    print(f"Sanity passed: {sanity['steps_run']} steps, RMSE {sanity['final_rmse']:.6f}.", flush=True)
    records = []
    for seed in SEEDS:
        print(f"Training fixed recurrent initialization {seed}.", flush=True)
        record = train_seed(data, output / f"seed_{seed}", seed)
        records.append(record)
        print(f"Seed {seed}: best epoch {record['best_epoch']}, validation RMSE {record['validation']['rmse']:.6f}.", flush=True)
    if sha256_file(preprocessor) != preprocessing_hash:
        raise RuntimeError("Original preprocessing file changed.")
    shutil.copyfile(preprocessor, output / "preprocessor.joblib")
    np.savez_compressed(output / "used_split_indices.npz", train=data.train.sample_indices,
                        validation=data.validation.sample_indices)
    _plot_curves(output)
    summary = {"workflow": model_spec()["workflow"], "seeds": records,
               "mean_and_sample_sd": aggregate_seed_metrics(records), "comparisons": comparisons,
               "counts": {name: {"samples": len(part.targets), "participants": len(np.unique(part.participants))}
                          for name, part in (("train", data.train), ("validation", data.validation))},
               "sanity": sanity, "parameter_count": records[0]["parameter_count"],
               "total_experiment_fit_seconds": sum(record["training_seconds"] for record in records),
               "sd_interpretation": "initialization variation on one split, not a population confidence interval"}
    write_json(output / "summary.json", summary)
    package_root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), Path(__file__).with_name("forecasting_kan.py"), Path(__file__).with_name("forecasting_baselines.py"),
             package_root / "models/sequential_kan.py", package_root / "models/kan_layer.py",
             package_root / "training/trainer.py", package_root / "evaluation/metrics.py"]
    write_json(output / "run.json", {
        "dataset_dir": str(source.dataset_dir), "dataset_manifest_sha256": source.manifest_sha256,
        "dataset_files_consumed": list(source.read_files),
        "consumed_sha256": {name: sha256_file(source.dataset_dir / name) for name in source.read_files},
        "original_input_hashes": source.manifest["inputs"], "builder_source_hashes": source.manifest["source_sha256"],
        "source_sha256": {str(path.relative_to(package_root)): sha256_file(path) for path in paths},
        "preprocessor_sha256": preprocessing_hash, "comparisons": comparisons["provenance"],
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "deterministic_algorithms": True, "device": "cpu", "num_threads": 2,
        "thread_environment": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "artifact_sha256": {str(path.relative_to(output)): sha256_file(path) for path in sorted(output.rglob("*")) if path.is_file()},
    })
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--tabular-kan-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_forecasting_sequential_kan(args.dataset_dir, args.baseline_dir, args.ablation_dir,
                                            args.tabular_kan_dir, args.output)
    print(json.dumps({"mean_and_sample_sd": summary["mean_and_sample_sd"],
                      "total_experiment_fit_seconds": summary["total_experiment_fit_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
