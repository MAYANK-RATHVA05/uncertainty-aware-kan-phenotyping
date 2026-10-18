"""Prepare conditional relative-month data artifacts; never train a model.

Run with ``python -m kan_phenotyping.experiments.build_forecasting_dataset``.
Raw identities, split definitions, fitted transforms, and hashes belong in an
ignored artifact directory, never in tracked source or public reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from kan_phenotyping.data.conditional_forecasting import (
    ConditionalForecastAssumptions, build_conditional_forecasting_dataset,
)
from kan_phenotyping.data.forecasting_preprocessing import ConditionalForecastPreprocessor
from kan_phenotyping.data.forecasting_splits import PARTITION_NAMES, participant_forecast_split


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_directory(output: str | Path) -> Path:
    destination = Path(output).resolve()
    # Discover the enclosing source checkout, including editable installations.
    checkout = next((parent for parent in Path(__file__).resolve().parents
                     if (parent / "pyproject.toml").exists() and (parent / ".git").exists()), None)
    if checkout is not None and destination.is_relative_to(checkout):
        relative = destination.relative_to(checkout)
        if not relative.parts or relative.parts[0] not in {"artifacts", "outputs", "runs"}:
            raise ValueError("Save generated data outside tracked source, under artifacts/, outputs/, runs/, or outside the checkout.")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output directory must be new or empty; existing artifacts are never overwritten.")
    return destination


def prepare_forecasting_dataset(
    data_path: str | Path,
    dictionary_path: str | Path,
    output_dir: str | Path,
    assumptions: ConditionalForecastAssumptions,
    seed: int = 42,
    fractions: tuple[float, float, float, float] = (0.6, 0.1, 0.1, 0.2),
) -> dict[str, Any]:
    """Build, split, fit training-only transforms, and save separate X/Y files.

    Calibration predictors are transformed with the training transform only.
    No outcomes are passed to preprocessing, split selection, or predictor
    construction; no learning algorithm or conformal calibration is run.
    The returned manifest contains aggregate counts, not participant records.
    """
    assumptions.validate()
    output = _output_directory(output_dir)
    inputs = {"data": Path(data_path).resolve(), "dictionary": Path(dictionary_path).resolve()}
    input_metadata = {name: {"path": str(path), "sha256": sha256_file(path)}
                      for name, path in inputs.items()}
    frame = pd.read_csv(inputs["data"], dtype={"__index_level_0__": str})
    dataset = build_conditional_forecasting_dataset(frame, inputs["dictionary"], assumptions)
    for name, path in inputs.items():
        if sha256_file(path) != input_metadata[name]["sha256"]:
            raise ValueError("An input changed during construction; retry with stable input files.")
    partitions = participant_forecast_split(dataset.participants, seed, fractions)
    preprocessing = ConditionalForecastPreprocessor().fit(dataset.predictors, partitions.train)
    transformed = {name: preprocessing.transform(dataset.predictors, getattr(partitions, name))
                   for name in PARTITION_NAMES}
    # Finish validation before creating artifacts; do not leave a successful
    # manifest for failed input/schema/partition/preprocessing checks.
    output.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for name, values in transformed.items():
        indices = getattr(partitions, name)
        arrays = {key: getattr(values, key) for key in (
            "wearable", "demographics", "sm0_phq", "relative_months", "study_months", "tabular",
        )}
        np.savez_compressed(output / f"{name}_inputs.npz", **arrays)
        np.savez_compressed(output / f"{name}_targets.npz", sm3_phq=dataset.targets[indices])
        shapes[name] = {key: list(value.shape) for key, value in arrays.items()}
        shapes[name]["separate_sm3_target"] = [len(indices)]
    np.savez_compressed(output / "splits.npz", **{
        name: getattr(partitions, name) for name in PARTITION_NAMES
    })
    np.savez_compressed(output / "sample_index.npz", **{
        name: getattr(dataset, name) for name in (
            "participants", "quarters", "endpoint_months", "history_row_positions",
            "baseline_row_positions", "target_row_positions",
        )
    })
    joblib.dump(preprocessing, output / "preprocessor.joblib")
    source_files = [Path(__file__), *[
        Path(__file__).parents[1] / "data" / name for name in (
            "conditional_forecasting.py", "forecasting_splits.py", "forecasting_preprocessing.py",
            "preprocessing.py", "dataset.py",
        )
    ]]
    manifest = {
        "format_version": 1,
        "dataset": dataset.export_metadata(),
        "inputs": input_metadata,
        "source_sha256": {path.name: sha256_file(path) for path in source_files},
        "split": {"seed": seed, "fractions": dict(zip(PARTITION_NAMES, fractions)),
                  **partitions.metadata(dataset.participants)},
        "preprocessing": preprocessing.export_metadata(),
        "tensor_shapes": shapes,
        "storage": {
            "predictors": "<partition>_inputs.npz (no targets or identities)",
            "targets": "<partition>_targets.npz: measured sm3_phq, unscaled and never imputed",
            "private_mapping": "sample_index.npz; splits.npz indexes its canonical sample order",
            "quarter_numbering": "zero-based; 0, 1, 2, 3",
        },
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "model_training_performed": False,
        "conformal_analysis_performed": False,
        "artifact_sha256": {path.name: sha256_file(path) for path in sorted(output.iterdir())},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", type=float, nargs=4, default=(0.6, 0.1, 0.1, 0.2),
                        metavar=("TRAIN", "VALIDATION", "CALIBRATION", "TEST"))
    parser.add_argument("--acknowledge-conditional-assumptions", action="store_true", required=True,
                        help="Explicitly accept the four working assumptions; does not verify timing.")
    parser.add_argument("--assumption-reference", default="docs/temporal_feasibility.md",
                        help="Evidence and limitations reference retained in the manifest.")
    args = parser.parse_args(argv)
    assumptions = ConditionalForecastAssumptions(
        assume_release_index_mapping=True,
        assume_historical_features_available_by_sm2=True,
        assume_sm0_phq_available_by_sm2=True,
        assume_enrollment_demographics=True,
        evidence_reference=args.assumption_reference,
    )
    manifest = prepare_forecasting_dataset(args.data, args.dictionary, args.output, assumptions,
                                          args.seed, tuple(args.fractions))
    print(json.dumps({"study_type": manifest["dataset"]["study_type"],
                      "eligible_samples": manifest["dataset"]["n_samples"],
                      "eligible_participants": manifest["dataset"]["n_participants_under_assumed_mapping"],
                      "split": manifest["split"], "tensor_shapes": manifest["tensor_shapes"]}, indent=2))


if __name__ == "__main__":
    main()
