"""Meaningful tabular ablations; temporal and focal-loss claims are excluded."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .common import build_model, load_run, model_spec, sha256_file
from ..training.trainer import Trainer, TrainingConfig, seed_everything, write_json


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variants", nargs="+", choices=("mlp_mixing", "huber"),
                        default=["mlp_mixing", "huber"])
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args(argv)
    prepared = load_run(args.run_dir, args.data, args.dictionary)
    if prepared.metadata["model"]["name"] != "tabular_kan":
        raise ValueError("Tabular KAN ablations need a tabular_kan reference run")
    config = TrainingConfig(**prepared.metadata["training"])
    if config.regression_loss != "mse":
        raise ValueError("The reference must use MSE before comparing the Huber ablation")
    if args.max_epochs is not None:
        if args.max_epochs != config.max_epochs:
            raise ValueError("Ablations must use the reference maximum-epoch training budget")
    output = args.output or args.run_dir / "ablations"
    output.mkdir(parents=True, exist_ok=True)
    comparison = output / "comparison.json"
    results = json.loads(comparison.read_text()) if comparison.exists() else {}
    results["reference_tabular_kan"] = json.loads((args.run_dir / "metrics.json").read_text())
    base_options = {key: value for key, value in prepared.metadata["model"]["kwargs"].items()
                    if key not in {"n_features", "hidden_width", "task"}}
    if base_options.get("mixing_type", "kan") != "kan":
        raise ValueError("The reference must use KAN mixing")
    for variant in dict.fromkeys(args.variants):
        directory = output / variant
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite an ablation: {directory}")
        settings = replace(config, regression_loss="huber") if variant == "huber" else config
        options = {**base_options, **({"mixing_type": "mlp"} if variant == "mlp_mixing" else {})}
        spec = model_spec(prepared.features["train"].tabular.shape[1], name="tabular_kan",
                          hidden_width=prepared.metadata["model"]["kwargs"]["hidden_width"], options=options)
        seed_everything(settings.seed, settings.num_threads)
        trainer = Trainer(build_model(spec), settings, input_kind="tabular")
        selected = trainer.fit(prepared.loader("train", settings, shuffle=True),
                               prepared.loader("validation", settings), directory, model_metadata=spec)
        metrics = {"analysis_type": "exploratory_not_confirmatory", "task": "regression",
                   "best_epoch": selected.best_epoch, "selection_metric": "validation_rmse",
                   "validation": trainer.evaluate(prepared.loader("validation", settings)),
                   "test": trainer.evaluate(prepared.loader("test", settings))}
        write_json(directory / "ablation.json", {"variant": variant, "model": spec,
                   "training": asdict(settings), "shared_inputs": prepared.metadata["inputs"],
                   "shared_splits_sha256": sha256_file(args.run_dir / "splits.npz"),
                   "limitations": prepared.metadata["limitations"]})
        write_json(directory / "metrics.json", metrics)
        results[variant] = metrics
        print(f"{variant}: validation RMSE={metrics['validation']['rmse']:.6f}; test RMSE={metrics['test']['rmse']:.6f}")
    write_json(comparison, results)
    return results


if __name__ == "__main__":
    main()
