"""Fair regression baselines on a saved run's exact split and preprocessing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from .common import build_model, load_run, model_spec, sha256_file
from ..evaluation.metrics import regression_metrics
from ..training.trainer import Trainer, TrainingConfig, seed_everything, write_json


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Trusted local reference run")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/baselines")
    parser.add_argument("--models", nargs="+", choices=("mean", "ridge", "random_forest", "mlp"),
                        default=["mean", "ridge", "random_forest", "mlp"])
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trees", type=int, default=100)
    args = parser.parse_args(argv)
    if args.trees < 1:
        parser.error("--trees must be positive")
    prepared = load_run(args.run_dir, args.data, args.dictionary)
    config = TrainingConfig(**prepared.metadata["training"])
    if args.max_epochs is not None:
        config = replace(config, max_epochs=args.max_epochs)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    output = args.output or args.run_dir / "baselines"
    output.mkdir(parents=True, exist_ok=True)
    comparison = output / "comparison.json"
    results = json.loads(comparison.read_text()) if comparison.exists() else {}
    for name in dict.fromkeys(args.models):
        directory = output / name
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite a baseline: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        seed_everything(config.seed, config.num_threads)
        metadata = {"model": name, "task": "regression", "analysis_type": "exploratory_not_confirmatory",
                    "shared_inputs": prepared.metadata["inputs"],
                    "shared_splits_sha256": sha256_file(args.run_dir / "splits.npz"),
                    "training": asdict(config), "limitations": prepared.metadata["limitations"]}
        if name == "mlp":
            spec = model_spec(prepared.features["train"].tabular.shape[1], name="mlp",
                              hidden_width=prepared.metadata["model"]["kwargs"]["hidden_width"])
            metadata["architecture"] = spec
            trainer = Trainer(build_model(spec), config, input_kind="tabular")
            selected = trainer.fit(prepared.loader("train", config, shuffle=True),
                                   prepared.loader("validation", config), directory, model_metadata=spec)
            metrics = {"best_epoch": selected.best_epoch, "selection_metric": "validation_rmse",
                       "validation": trainer.evaluate(prepared.loader("validation", config)),
                       "test": trainer.evaluate(prepared.loader("test", config))}
        else:
            estimator = ({"mean": lambda: DummyRegressor(strategy="mean"),
                          "ridge": lambda: Ridge(alpha=1.0),
                          "random_forest": lambda: RandomForestRegressor(
                              n_estimators=args.trees, random_state=config.seed,
                              n_jobs=config.num_threads)}[name])()
            estimator.fit(prepared.features["train"].tabular, prepared.examples.y[prepared.split.train])
            joblib.dump(estimator, directory / "model.joblib")
            metadata["estimator_parameters"] = estimator.get_params()
            metrics = {partition: regression_metrics(
                prepared.examples.y[getattr(prepared.split, partition)],
                estimator.predict(prepared.features[partition].tabular))
                for partition in ("validation", "test")}
            metrics["selection_metric"] = "none: fixed baseline hyperparameters"
        metrics.update({"analysis_type": "exploratory_not_confirmatory", "task": "regression"})
        write_json(directory / "baseline.json", metadata)
        write_json(directory / "metrics.json", metrics)
        results[name] = metrics
        print(f"{name}: validation RMSE={metrics['validation']['rmse']:.6f}; test RMSE={metrics['test']['rmse']:.6f}")
    write_json(comparison, results)
    return results


if __name__ == "__main__":
    main()
