"""Shared, fingerprinted retrospective data protocol for all comparisons."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import torch
import yaml

from ..data.dataset import load_dataset, prepare_tabular_examples
from ..data.preprocessing import ForecastPreprocessor
from ..data.splitting import ParticipantSplit, group_split
from ..models import MLPBaseline, TabularDepressionKAN
from ..training.trainer import FeatureDataset, TrainingConfig, make_loader, write_json


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_fingerprints(data: str | Path, dictionary: str | Path) -> dict[str, str]:
    return {"data_sha256": sha256_file(data), "dictionary_sha256": sha256_file(dictionary)}


def read_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if value.get("task", "regression") != "regression":
        raise ValueError("PSYCHE-D CLI supports continuous PHQ change regression only")
    if value.get("data", {}).get("workflow", "tabular_retrospective") != "tabular_retrospective":
        raise ValueError("This CLI supports retrospective tabular analysis; forecasting prerequisites remain unverified")
    return value


def model_spec(n_features: int, *, name: str = "tabular_kan", hidden_width: int = 64,
               options: dict[str, Any] | None = None) -> dict[str, Any]:
    if name not in {"tabular_kan", "mlp"}:
        raise ValueError("Retrospective rows support tabular_kan or mlp; no temporal order is inferred")
    supplied = dict(options or {})
    allowed = {"grid_intervals", "spline_degree", "grid_range", "mixing_type"}
    if set(supplied) - allowed:
        raise ValueError(f"Unknown model options: {sorted(set(supplied) - allowed)}")
    if name == "mlp" and supplied:
        raise ValueError("Spline/mixing options do not apply to MLP baseline")
    if name == "tabular_kan":
        supplied = {"grid_intervals": 8, "spline_degree": 3, "grid_range": [-3.0, 3.0],
                    "mixing_type": "kan", **supplied}
        supplied["grid_range"] = list(supplied["grid_range"])
    return {"name": name, "kwargs": {"n_features": n_features, "hidden_width": hidden_width,
                                      "task": "regression", **supplied}, "input_kind": "tabular"}


def build_model(spec: dict[str, Any]) -> torch.nn.Module:
    constructors = {"tabular_kan": TabularDepressionKAN, "mlp": MLPBaseline}
    if spec.get("name") not in constructors or spec.get("input_kind") != "tabular":
        raise ValueError("Checkpoint does not describe a supported tabular model")
    if spec["kwargs"].get("task") != "regression":
        raise ValueError("Retrospective PHQ workflow requires a regression checkpoint")
    return constructors[spec["name"]](**spec["kwargs"])


def _cap_groups(indices: np.ndarray, groups: np.ndarray, budget: int, seed: int) -> np.ndarray:
    unique = np.unique(groups[indices])
    order = np.random.default_rng(seed).permutation(unique)
    chosen: list[np.ndarray] = []
    remaining = budget
    for group in order:
        members = indices[groups[indices] == group]
        if len(members) <= remaining:
            chosen.append(members)
            remaining -= len(members)
    if not chosen:
        # A group is indivisible. A budget smaller than one group is a soft cap.
        chosen = [min((indices[groups[indices] == group] for group in unique), key=len)]
    return np.sort(np.concatenate(chosen))


def limit_split(split: ParticipantSplit, groups: np.ndarray, max_samples: int | None,
                seed: int) -> ParticipantSplit:
    if max_samples is None:
        return split
    if max_samples < 3:
        raise ValueError("max_samples must allow at least one group in each partition")
    budgets = [max(1, int(max_samples * fraction)) for fraction in (0.7, 0.1, 0.2)]
    return ParticipantSplit(*(_cap_groups(indices, groups, budget, seed + offset)
                              for offset, (indices, budget) in enumerate(zip(
                                  (split.train, split.validation, split.test), budgets))))


def validate_saved_split(split: ParticipantSplit, groups: np.ndarray) -> None:
    indices = [np.asarray(part) for part in (split.train, split.validation, split.test)]
    if any(part.ndim != 1 or not np.issubdtype(part.dtype, np.integer) or len(part) == 0 for part in indices):
        raise ValueError("Saved splits must contain nonempty integer index vectors")
    joined = np.concatenate(indices)
    if (joined < 0).any() or (joined >= len(groups)).any() or len(np.unique(joined)) != len(joined):
        raise ValueError("Saved split indices overlap or are out of range")
    sets = [set(groups[part]) for part in indices]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Saved splits cross provisional group boundaries")


def split_metadata(split: ParticipantSplit, groups: np.ndarray) -> dict[str, Any]:
    validate_saved_split(split, groups)
    return {name: {"examples": len(indices), "provisional_groups": int(len(np.unique(groups[indices])))}
            for name, indices in (("train", split.train), ("validation", split.validation), ("test", split.test))} | {
                "group_overlap": False, "group_identity_verified": False,
                "stratification": "none: continuous regression",
                "rule": "fixed 70/10/20 split of provisional composite-index prefixes"}


@dataclass
class PreparedRun:
    examples: Any
    preprocessor: Any
    split: ParticipantSplit
    features: dict[str, Any]
    metadata: dict[str, Any]

    def loader(self, partition: str, config: TrainingConfig, *, shuffle: bool = False):
        indices = getattr(self.split, partition)
        return make_loader(FeatureDataset(self.features[partition], self.examples.y[indices]), config, shuffle=shuffle)


def prepare_run(data: str | Path, dictionary: str | Path, output_dir: str | Path, *,
                split_seed: int = 42, max_samples: int | None = None) -> PreparedRun:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new, empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = input_fingerprints(data, dictionary)
    frame, schema = load_dataset(data, dictionary)
    examples = prepare_tabular_examples(frame, schema)
    split = limit_split(group_split(examples.groups, seed=split_seed), examples.groups, max_samples, split_seed)
    validate_saved_split(split, examples.groups)
    preprocessor = ForecastPreprocessor(examples.schema).fit(examples, split.train)
    features = {name: preprocessor.transform(examples, getattr(split, name))
                for name in ("train", "validation", "test")}
    metadata = {
        "format_version": 1, "workflow": "tabular_retrospective", "task": "regression",
        "target": "phq9_score_end - phq9_score_start", "selection_metric": "validation_rmse",
        "analysis_type": "exploratory_not_confirmatory", "split_seed": split_seed,
        "max_samples": max_samples, "inputs": fingerprint, "dataset": examples.export_metadata(),
        "splits": split_metadata(split, examples.groups),
        "preprocessing": preprocessor.export_metadata(),
        "feature_names": features["train"].feature_names,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "sklearn": sklearn.__version__},
        "limitations": ["Composite-index prefixes are provisional groups; participant identity is unverified.",
                        "Covariate timing is unverified; this is retrospective association, not forecasting.",
                        "Metrics do not establish clinical validity or diagnosis."],
    }
    np.savez_compressed(output / "splits.npz", train=split.train, validation=split.validation, test=split.test)
    np.savez_compressed(output / "source_row_positions.npz", **{
        name: examples.row_positions[getattr(split, name)] for name in ("train", "validation", "test")})
    joblib.dump(preprocessor, output / "preprocessor.joblib")
    write_json(output / "run.json", metadata)
    return PreparedRun(examples, preprocessor, split, features, metadata)


def load_run(run_dir: str | Path, data: str | Path, dictionary: str | Path) -> PreparedRun:
    """Rebuild exact transformed partitions from a trusted, local run directory.

    Only load artifacts you trust: joblib is a Python serialization format.
    Inputs must match their saved content hashes before any artifact is loaded.
    """
    root = Path(run_dir)
    metadata = json.loads((root / "run.json").read_text())
    if metadata.get("format_version") != 1 or metadata.get("workflow") != "tabular_retrospective":
        raise ValueError("Unsupported saved run format/workflow")
    if input_fingerprints(data, dictionary) != metadata["inputs"]:
        raise ValueError("Dataset or feature dictionary fingerprint differs from the original run")
    frame, schema = load_dataset(data, dictionary)
    examples = prepare_tabular_examples(frame, schema)
    with np.load(root / "splits.npz", allow_pickle=False) as saved:
        split = ParticipantSplit(*(saved[key] for key in ("train", "validation", "test")))
    validate_saved_split(split, examples.groups)
    if split_metadata(split, examples.groups) != metadata["splits"]:
        raise ValueError("Saved split sizes/grouping differ from run metadata")
    preprocessor = joblib.load(root / "preprocessor.joblib")
    features = {name: preprocessor.transform(examples, getattr(split, name))
                for name in ("train", "validation", "test")}
    if features["train"].feature_names != metadata["feature_names"]:
        raise ValueError("Preprocessed feature names differ from the saved run")
    return PreparedRun(examples, preprocessor, split, features, metadata)


def training_config(config: dict[str, Any], *, seed: int | None = None,
                    max_epochs: int | None = None, device: str | None = None) -> TrainingConfig:
    settings = {"max_epochs": 50, **config.get("training", {})}
    settings["task"] = "regression"
    for name, value in (("seed", seed), ("max_epochs", max_epochs), ("device", device)):
        if value is not None:
            settings[name] = value
    return TrainingConfig(**settings)
