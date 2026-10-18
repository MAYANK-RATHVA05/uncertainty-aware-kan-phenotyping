"""Restore a trusted local checkpoint and evaluate the frozen partitions."""

from __future__ import annotations

from pathlib import Path

import torch

from ..experiments.common import build_model, load_run
from ..training.trainer import Trainer, TrainingConfig, seed_everything, write_json


def evaluate_saved_run(run_dir: str | Path, data: str | Path, dictionary: str | Path,
                       *, device: str = "cpu") -> dict:
    root = Path(run_dir)
    prepared = load_run(root, data, dictionary)
    saved = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    if saved["model_metadata"] != prepared.metadata["model"]:
        raise ValueError("Checkpoint architecture differs from run metadata")
    if saved["training_config"] != prepared.metadata["training"]:
        raise ValueError("Checkpoint training configuration differs from run metadata")
    settings = dict(saved["training_config"])
    settings["device"] = device
    config = TrainingConfig(**settings)
    seed_everything(config.seed, config.num_threads)
    model = build_model(saved["model_metadata"])
    model.load_state_dict(saved["model_state"])
    trainer = Trainer(model, config, input_kind=saved["input_kind"])
    result = {"analysis_type": "exploratory_not_confirmatory", "task": "regression",
              "best_epoch": saved["epoch"], "selection_metric": "validation_rmse",
              "validation": trainer.evaluate(prepared.loader("validation", config)),
              "test": trainer.evaluate(prepared.loader("test", config)),
              "limitations": prepared.metadata["limitations"]}
    write_json(root / "evaluation_restored.json", result)
    return result
