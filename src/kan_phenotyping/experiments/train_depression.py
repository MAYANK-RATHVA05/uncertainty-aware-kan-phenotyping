"""Train a tabular depression-severity-change regressor; no diagnosis claim."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .common import build_model, model_spec, prepare_run, read_config, training_config
from ..training.trainer import Trainer, seed_everything, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", required=True, type=Path)
    result.add_argument("--dictionary", required=True, type=Path)
    result.add_argument("--config", type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--seed", type=int)
    result.add_argument("--split-seed", type=int, help="Frozen split seed; independent of initialization seed")
    result.add_argument("--max-epochs", type=int)
    result.add_argument("--max-samples", type=int, help="Approximate group-preserving smoke budget across all splits")
    result.add_argument("--model", choices=("tabular_kan", "mlp"))
    result.add_argument("--device", default=None)
    return result


def main(argv: list[str] | None = None) -> dict:
    args = parser().parse_args(argv)
    config = read_config(args.config)
    training = training_config(config, seed=args.seed, max_epochs=args.max_epochs, device=args.device)
    seed_everything(training.seed, training.num_threads)
    prepared = prepare_run(args.data, args.dictionary, args.output,
                           split_seed=args.split_seed if args.split_seed is not None else config.get("split_seed", 42),
                           max_samples=args.max_samples if args.max_samples is not None else config.get("max_samples"))
    settings = config.get("model", "tabular_kan")
    if isinstance(settings, str):
        settings = {"name": settings}
    settings = dict(settings)
    name = args.model or settings.pop("name", "tabular_kan")
    if "name" in settings:
        settings.pop("name")
    hidden_width = settings.pop("hidden_width", config.get("hidden_width", 64))
    spec = model_spec(prepared.features["train"].tabular.shape[1], name=name,
                      hidden_width=hidden_width, options=settings)
    prepared.metadata.update({"model": spec, "training": asdict(training)})
    write_json(args.output / "run.json", prepared.metadata)
    trainer = Trainer(build_model(spec), training, input_kind="tabular")
    result = trainer.fit(prepared.loader("train", training, shuffle=True),
                         prepared.loader("validation", training), args.output, model_metadata=spec)
    metrics = {"analysis_type": "exploratory_not_confirmatory", "task": "regression",
               "best_epoch": result.best_epoch, "selection_metric": "validation_rmse",
               "validation": trainer.evaluate(prepared.loader("validation", training)),
               "test": trainer.evaluate(prepared.loader("test", training)),
               "limitations": prepared.metadata["limitations"]}
    write_json(args.output / "metrics.json", metrics)
    print(f"Saved restored-best checkpoint and exploratory regression metrics to {args.output}")
    print(f"Validation RMSE={metrics['validation']['rmse']:.6f}; test RMSE={metrics['test']['rmse']:.6f}")
    return metrics


if __name__ == "__main__":
    main()
