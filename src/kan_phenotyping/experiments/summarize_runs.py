"""Aggregate initialization seeds while requiring identical frozen splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import sha256_file
from ..training.trainer import write_json


def canonical_model_spec(spec: dict) -> dict:
    """Normalize recorded versus implicit constructor defaults, without edits."""
    if spec.get("name") != "tabular_kan":
        return spec
    kwargs = {"hidden_width": 64, "task": "regression", "grid_intervals": 8,
              "spline_degree": 3, "grid_range": [-3.0, 3.0], "mixing_type": "kan",
              **spec.get("kwargs", {})}
    kwargs["grid_range"] = list(kwargs["grid_range"])
    return {**spec, "kwargs": kwargs}


def summarize_runs(run_dirs: list[str | Path]) -> dict:
    if not run_dirs:
        raise ValueError("At least one run is required")
    runs = [Path(path) for path in run_dirs]
    metadata = [json.loads((path / "run.json").read_text()) for path in runs]
    metrics = [json.loads((path / "metrics.json").read_text()) for path in runs]
    protocol = []
    for root, info in zip(runs, metadata):
        training = {key: value for key, value in info["training"].items() if key != "seed"}
        protocol.append({"inputs": info["inputs"], "split": sha256_file(root / "splits.npz"),
                         "model": canonical_model_spec(info["model"]), "training": training, "task": info["target"]})
    if any(value != protocol[0] for value in protocol[1:]):
        raise ValueError("Runs must share inputs, target, model, training budget and exact frozen splits")
    seeds = [info["training"]["seed"] for info in metadata]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Initialization seeds must be distinct")
    result = {"analysis_type": "exploratory_not_confirmatory", "seeds": seeds, "n_runs": len(runs),
              "spread": "sample standard deviation across initialization seeds (ddof=1); not a confidence interval",
              "splits_sha256": protocol[0]["split"], "inputs": protocol[0]["inputs"]}
    for partition in ("validation", "test"):
        result[partition] = {}
        for name in ("mae", "rmse", "r2"):
            values = [run[partition][name] for run in metrics]
            result[partition][name] = {"mean": float(np.mean(values)) if all(v is not None for v in values) else None,
                                       "std": float(np.std(values, ddof=1)) if len(values) > 1 and all(v is not None for v in values) else None}
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = summarize_runs(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"Saved mean and sample standard deviation over {result['n_runs']} runs to {args.output}")
    return result


if __name__ == "__main__":
    main()
