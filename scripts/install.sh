#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# xgboost needs the OpenMP runtime, which isn't bundled on macOS.
if [[ "$(uname)" == "Darwin" ]] && ! [[ -f /opt/homebrew/opt/libomp/lib/libomp.dylib || -f /usr/local/opt/libomp/lib/libomp.dylib ]]; then
  brew install libomp
fi

pip install -r requirements.txt

# src-layout: tests and the `python -m edge_engine.*` entry points import
# `edge_engine`, which isn't importable from a bare checkout. Without this
# the documented first step after installing (`python -m pytest -q`) fails
# with ModuleNotFoundError.
pip install -e .
