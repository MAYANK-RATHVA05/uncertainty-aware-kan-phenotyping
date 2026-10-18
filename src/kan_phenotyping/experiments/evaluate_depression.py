"""Re-evaluate a restored checkpoint against the exact fingerprinted inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..evaluation.evaluate import evaluate_saved_run


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = evaluate_saved_run(args.run_dir, args.data, args.dictionary, device=args.device)
    print(f"Restored epoch {result['best_epoch']}: test RMSE={result['test']['rmse']:.6f}")
    return result


if __name__ == "__main__":
    main()
