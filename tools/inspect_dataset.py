"""Inspect an entire dataset without printing rows, identifiers, or selecting a target.

CSV has no stored types. Its reported dtypes are parser inferences observed across
all chunks; literal missing tokens are retained because their meaning is unknown.
Only empty CSV fields count as missing. Parquet nulls and NaNs count as missing.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


class InspectionError(ValueError):
    """A deliberately record-free inspection error suitable for CLI output."""


def _format(path: Path) -> tuple[str, str | None]:
    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic == b"PAR1":
            handle.seek(-4, 2)
            if handle.read(4) != b"PAR1":
                raise InspectionError("Parquet footer is missing or invalid.")
            return "parquet", None
    compressed = magic[:2] == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(65536)
    if not sample or "\x00" in sample or any(
        ord(character) < 32 and character not in "\t\r\n" for character in sample
    ):
        raise InspectionError("Expected a nonempty UTF-8 CSV or Parquet dataset.")
    if sample.lstrip().startswith(("{", "[", "<")):
        raise InspectionError("Unsupported file content; expected CSV or Parquet.")
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        # A legitimate single-column CSV has no detectable delimiter.
        delimiter = ","
    return ("gzip_csv" if compressed else "csv"), delimiter


def _validate_columns(names: list[str]) -> None:
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise InspectionError("Column names must be nonempty and unique.")


def _candidates(names: list[str]) -> dict:
    patterns = {
        "depression": r"phq|depress",
        "participant_identifier": r"(^|[_\s-])(id|pid|participant|subject|person|user)([_\s-]|$)",
        "time": r"time|date|month|week|day|visit|wave|index",
    }
    return {
        "status": "Unverified name matches only; no target, identifier, or time semantics established.",
        **{key: [name for name in names if re.search(pattern, name, re.I)]
           for key, pattern in patterns.items()},
    }


def inspect_dataset(path: str | Path, *, chunksize: int = 100000) -> dict:
    """Return aggregate structure; never return records or unique field values."""
    path = Path(path)
    if chunksize < 1:
        raise InspectionError("Chunk size must be positive.")
    try:
        before = path.stat()
        detected, delimiter = _format(path)
        sha256 = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                sha256.update(block)

        if detected == "parquet":
            parquet = pq.ParquetFile(path)
            names = parquet.schema_arrow.names
            _validate_columns(names)
            stored_types = [str(field.type) for field in parquet.schema_arrow]
            batches = (batch.to_pandas() for batch in parquet.iter_batches(batch_size=chunksize))
            expected_rows = parquet.metadata.num_rows
        else:
            opener = gzip.open if detected == "gzip_csv" else open
            with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
                names = next(csv.reader(handle, delimiter=delimiter))
            _validate_columns(names)
            stored_types = None
            batches = pd.read_csv(
                path, sep=delimiter, encoding="utf-8-sig",
                compression="gzip" if detected == "gzip_csv" else None,
                chunksize=chunksize, keep_default_na=False, na_values=[""],
                on_bad_lines="error", low_memory=False, index_col=False,
            )
            expected_rows = None

        missing = [0] * len(names)
        seen_types = [set() for _ in names]
        rows = 0
        with warnings.catch_warnings():
            # Prevent pandas from silently dropping excess fields or quoting
            # potentially sensitive parser diagnostics in warning output.
            warnings.simplefilter("error", pd.errors.ParserWarning)
            for batch in batches:
                if list(batch.columns) != names:
                    raise InspectionError("Column structure changed during parsing.")
                rows += len(batch)
                for index, name in enumerate(names):
                    missing[index] += int(batch[name].isna().sum())
                    if len(batch):
                        seen_types[index].add(str(batch[name].dtype))
        if expected_rows is not None and rows != expected_rows:
            raise InspectionError("Parquet scan row count disagrees with its metadata.")
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise InspectionError("Dataset changed while it was being inspected.")

        return {
            "file_name": path.name,
            "format": detected,
            "sha256": sha256.hexdigest(),
            "file_size_bytes": before.st_size,
            "row_count": rows,
            "column_count": len(names),
            "columns": [
                {"name": name,
                 "dtype_source": "stored Parquet schema" if stored_types else "inferred CSV parser types",
                 "dtype": stored_types[index] if stored_types else sorted(seen_types[index]),
                 "missing_count": missing[index]}
                for index, name in enumerate(names)
            ],
            "missing_value_policy": (
                "Parquet nulls and IEEE NaNs are missing."
                if stored_types else
                "Only empty CSV fields are missing; literal NA/NULL and whitespace are retained. "
                "CSV has no stored dtypes; reported types are observed parser inferences across all chunks."
            ),
            "csv_delimiter": delimiter,
            "candidate_columns": _candidates(names),
            "scientific_status": (
                "Structure inspection only. Target, observation unit, participant/time semantics, "
                "prediction cutoff, leakage exclusions, and valid temporal sequences remain unverified."
            ),
        }
    except InspectionError:
        raise
    except Exception:
        # Library parse errors can quote participant records. Do not print them.
        raise InspectionError(
            "Dataset could not be read safely; check file access, integrity, encoding, and format."
        ) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, help="Write aggregate JSON here; otherwise use stdout.")
    parser.add_argument("--chunksize", type=int, default=100000)
    args = parser.parse_args(argv)
    try:
        report = inspect_dataset(args.dataset, chunksize=args.chunksize)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            if args.output.resolve() == args.dataset.resolve():
                raise InspectionError("Output must not overwrite the input dataset.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            sys.stdout.write(encoded)
        return 0
    except InspectionError as error:
        print(f"Inspection failed: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("Inspection failed: the aggregate report could not be saved.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
