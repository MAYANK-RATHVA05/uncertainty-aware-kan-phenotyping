"""Deterministic, validation-selected training for a single continuous outcome.

Binary training is available only for callers that supply an explicitly defined
binary target. The PSYCHE-D forecasting workflow always uses regression.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..evaluation.metrics import binary_classification_metrics, regression_metrics
from .focal_loss import BinaryFocalLoss


@dataclass(frozen=True)
class TrainingConfig:
    task: Literal["regression", "binary"] = "regression"
    max_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    seed: int = 42
    device: str = "auto"
    num_threads: int = 2
    min_delta: float = 0.0
    gradient_clip: float | None = 5.0
    binary_loss: Literal["focal", "bce"] = "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    regression_loss: Literal["mse", "huber"] = "mse"

    def __post_init__(self) -> None:
        if self.task not in {"regression", "binary"}:
            raise ValueError("task must be regression or an explicitly defined binary task")
        if self.max_epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("max_epochs, batch_size, and patience must be positive")
        if self.num_threads < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer/thread configuration")
        if self.min_delta < 0 or (self.gradient_clip is not None and self.gradient_clip <= 0):
            raise ValueError("Invalid early-stopping/gradient-clip configuration")
        if self.binary_loss not in {"focal", "bce"}:
            raise ValueError("binary_loss must be focal or bce")
        if self.regression_loss not in {"mse", "huber"}:
            raise ValueError("regression_loss must be mse or huber")


def seed_everything(seed: int, num_threads: int = 2) -> None:
    """Seed Python, NumPy and PyTorch; deterministic kernels are required."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class FeatureDataset(Dataset):
    """A fixed set of transformed examples, with no fitting or split logic."""

    def __init__(self, features: Any, targets: np.ndarray):
        self.arrays = {
            name: torch.as_tensor(np.asarray(getattr(features, name)), dtype=torch.float32)
            for name in ("temporal", "static", "tabular")
        }
        self.targets = torch.as_tensor(np.asarray(targets), dtype=torch.float32).reshape(-1)
        n = len(self.targets)
        if any(len(value) != n for value in self.arrays.values()):
            raise ValueError("Feature and target lengths differ")
        if n == 0 or not torch.isfinite(self.targets).all():
            raise ValueError("Targets must be nonempty and finite")
        if any(not torch.isfinite(value).all() for value in self.arrays.values()):
            raise ValueError("Preprocessed features must be finite")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {**{key: value[index] for key, value in self.arrays.items()}, "target": self.targets[index]}


def make_loader(dataset: Dataset, config: TrainingConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator, drop_last=False)


def forward_batch(model: nn.Module, batch: dict[str, torch.Tensor], input_kind: str,
                  device: torch.device) -> torch.Tensor:
    if input_kind == "temporal":
        output = model(batch["temporal"].to(device), batch["static"].to(device))
    elif input_kind == "tabular":
        output = model(batch["tabular"].to(device))
    else:
        raise ValueError(f"Unknown input_kind: {input_kind}")
    output = output.reshape(-1)
    if output.shape != batch["target"].shape:
        raise ValueError("Model must return exactly one output per example")
    if not torch.isfinite(output).all():
        raise FloatingPointError("Model returned nonfinite predictions")
    return output


def write_json(path: str | Path, value: Any) -> None:
    """Write strict JSON: undefined metrics are null, never nonstandard NaN."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@dataclass
class TrainingResult:
    best_epoch: int
    best_score: float
    checkpoint: Path
    history: list[dict[str, Any]]


class Trainer:
    """Select checkpoints using validation RMSE (or AUROC for explicit binary).

    No test data is accepted by ``fit``. The best validation checkpoint is
    restored before returning so callers cannot accidentally evaluate the last
    epoch. Checkpoints contain model weights and non-secret metadata only.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig, *, input_kind: str = "temporal"):
        self.config = config
        self.input_kind = input_kind
        requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested if config.device == "auto" else config.device)
        self.model = model.to(self.device)
        self.loss_fn = ((nn.MSELoss() if config.regression_loss == "mse" else nn.HuberLoss())
                        if config.task == "regression" else
                        BinaryFocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
                        if config.binary_loss == "focal" else nn.BCEWithLogitsLoss())

    def _epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer | None = None
               ) -> tuple[float, np.ndarray, np.ndarray]:
        training = optimizer is not None
        self.model.train(training)
        loss_sum, count = 0.0, 0
        actual, predictions = [], []
        with torch.set_grad_enabled(training):
            for batch in loader:
                target = batch["target"].to(self.device)
                if self.config.task == "binary" and not torch.all((target == 0) | (target == 1)):
                    raise ValueError("Explicit binary targets must contain only 0 and 1")
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                output = forward_batch(self.model, batch, self.input_kind, self.device)
                loss = self.loss_fn(output, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training/evaluation loss")
                if optimizer is not None:
                    loss.backward()
                    if self.config.gradient_clip is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip,
                                                 error_if_nonfinite=True)
                    optimizer.step()
                prediction = torch.sigmoid(output) if self.config.task == "binary" else output
                predictions.append(prediction.detach().cpu().numpy())
                actual.append(target.detach().cpu().numpy())
                loss_sum += float(loss.detach()) * len(target)
                count += len(target)
        if not count:
            raise ValueError("Cannot train or evaluate an empty loader")
        return loss_sum / count, np.concatenate(actual), np.concatenate(predictions)

    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        loss, target, prediction = self._epoch(loader)
        metrics = (regression_metrics(target, prediction) if self.config.task == "regression"
                   else binary_classification_metrics(target, prediction))
        return {"loss": loss, **metrics}

    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        _, targets, predictions = self._epoch(loader)
        return targets, predictions

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader,
            output_dir: str | Path, *, model_metadata: dict[str, Any] | None = None) -> TrainingResult:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "best.pt"
        if checkpoint.exists() or (output / "history.json").exists():
            raise FileExistsError(f"Refusing to overwrite an existing training run: {output}")
        write_json(output / "training_config.json", asdict(self.config))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)
        metric = "rmse" if self.config.task == "regression" else "auroc"
        best = math.inf if metric == "rmse" else -math.inf
        stale, best_epoch = 0, -1
        history: list[dict[str, Any]] = []
        for epoch in range(1, self.config.max_epochs + 1):
            training_loss, _, _ = self._epoch(train_loader, optimizer)
            validation = self.evaluate(validation_loader)
            score = validation[metric]
            if score is None or not math.isfinite(score):
                raise ValueError(f"Validation {metric} is undefined; revise the split before training")
            history.append({"epoch": epoch, "training_loss": training_loss,
                            "validation_loss": validation["loss"], "validation": validation})
            improved = (score < best - self.config.min_delta if metric == "rmse"
                        else score > best + self.config.min_delta)
            if improved:
                best, best_epoch, stale = float(score), epoch, 0
                temporary = output / "best.pt.tmp"
                torch.save({"model_state": self.model.state_dict(), "epoch": epoch,
                            "selection_metric": metric, "selection_score": best,
                            "training_config": asdict(self.config), "input_kind": self.input_kind,
                            "model_metadata": model_metadata or {}}, temporary)
                temporary.replace(checkpoint)
            else:
                stale += 1
            write_json(output / "history.json", history)
            if stale >= self.config.patience:
                break
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(saved["model_state"])
        return TrainingResult(best_epoch, best, checkpoint, history)
