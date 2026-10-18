#!/usr/bin/env bash
set -euo pipefail
KAN_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
KAN_VENV_DIR="${KAN_VENV_DIR:-/workspace/.venvs/uncertainty-aware-kan-phenotyping}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/workspace/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export MPLBACKEND=Agg
mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
if [[ ! -x "$KAN_VENV_DIR/bin/python" ]]; then
  uv venv --python python3.12 "$KAN_VENV_DIR"
fi
"$KAN_VENV_DIR/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'
uv pip install --python "$KAN_VENV_DIR/bin/python" -r "$KAN_REPO_DIR/requirements-torch-cpu.txt"
uv pip install --python "$KAN_VENV_DIR/bin/python" --index-url https://pypi.org/simple -r "$KAN_REPO_DIR/requirements.lock.txt"
uv pip install --python "$KAN_VENV_DIR/bin/python" --no-deps --no-build-isolation -e "$KAN_REPO_DIR"
uv pip check --python "$KAN_VENV_DIR/bin/python"
"$KAN_VENV_DIR/bin/python" -c 'import torch, pandas, pyarrow, sklearn, kan_phenotyping; print("Dependencies verified; torch", torch.__version__, "CUDA", torch.cuda.is_available())'
