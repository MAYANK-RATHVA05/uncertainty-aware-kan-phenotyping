"""Synthetic fixtures only: these tests do not validate real PSYCHE-D semantics."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "inspection_tool", Path(__file__).parents[1] / "tools" / "inspect_dataset.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
inspect_dataset = _MODULE.inspect_dataset
InspectionError = _MODULE.InspectionError


def test_csv_complete_missing_counts_and_privacy(tmp_path):
    path = tmp_path / "data.parquet"
    content = "participant_id,phq_value,visit_time,note\nPRIVATE_A,1,1,NA\nPRIVATE_B,,2,\nPRIVATE_A,3,,NULL\n"
    path.write_text(content)
    report = inspect_dataset(path, chunksize=1)
    assert report["format"] == "csv"
    assert report["row_count"] == 3
    assert report["column_count"] == 4
    assert [column["missing_count"] for column in report["columns"]] == [0, 1, 1, 1]
    assert report["sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert report["candidate_columns"]["depression"] == ["phq_value"]
    assert "Unverified" in report["candidate_columns"]["status"]
    assert all(column["dtype_source"] == "inferred CSV parser types" for column in report["columns"])
    assert "PRIVATE_A" not in json.dumps(report)
    assert "PRIVATE_B" not in json.dumps(report)
    assert inspect_dataset(path, chunksize=1) == report


def test_parquet_detected_independently_of_suffix(tmp_path):
    path = tmp_path / "disguised.csv"
    pd.DataFrame({"participant_id": ["PRIVATE_A", "PRIVATE_B"], "value": [1.0, None]}).to_parquet(path, index=False)
    report = inspect_dataset(path, chunksize=1)
    assert report["format"] == "parquet"
    assert report["row_count"] == 2
    assert report["columns"][1]["missing_count"] == 1
    assert report["columns"][1]["dtype_source"] == "stored Parquet schema"
    assert "PRIVATE_A" not in json.dumps(report)


def test_gzip_csv_detected_by_contents(tmp_path):
    path = tmp_path / "compressed.data"
    with gzip.open(path, "wt") as handle:
        handle.write("value,other\n1,\n,4\n")
    report = inspect_dataset(path, chunksize=1)
    assert report["format"] == "gzip_csv"
    assert report["row_count"] == 2
    assert [column["missing_count"] for column in report["columns"]] == [1, 1]


@pytest.mark.parametrize("content", [b"", b"\x00\x01\x02\x03", b'{"secret": "PRIVATE_A"}', b"a,a\n1,2\n", b"a,\n1,2\n", b"a,b\nPRIVATE_A,2,3\n"])
def test_invalid_input_never_echoes_records(tmp_path, capsys, content):
    path = tmp_path / "invalid.csv"
    path.write_bytes(content)
    assert _MODULE.main([str(path)]) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "PRIVATE_A" not in output.err


def test_header_only_and_invalid_chunksize(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("x,y\n")
    report = inspect_dataset(path)
    assert report["row_count"] == 0
    assert all(column["missing_count"] == 0 for column in report["columns"])
    with pytest.raises(InspectionError, match="positive"):
        inspect_dataset(path, chunksize=0)


def test_cli_writes_report_and_protects_input(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x,y\n1,2\n")
    output = tmp_path / "reports" / "structure.json"
    assert _MODULE.main([str(path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["row_count"] == 1
    original = path.read_bytes()
    assert _MODULE.main([str(path), "--output", str(path)]) == 2
    assert path.read_bytes() == original
