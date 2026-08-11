#!/usr/bin/env bash
# nfl_data_py's published pin (pandas<2.0, numpy<2.0) has no prebuilt wheels
# for modern Python, so it must be installed with --no-deps after the real
# (modern) dependency stack is in place. In practice nfl_data_py's actual
# code works fine against pandas 2.x/3.x for the functions this project uses.
set -euo pipefail
cd "$(dirname "$0")/.."

# xgboost needs the OpenMP runtime, which isn't bundled on macOS.
if [[ "$(uname)" == "Darwin" ]] && ! [[ -f /opt/homebrew/opt/libomp/lib/libomp.dylib || -f /usr/local/opt/libomp/lib/libomp.dylib ]]; then
  brew install libomp
fi

pip install -r requirements.txt
pip install --no-deps nfl_data_py==0.3.3

# src-layout: tests and the `python -m edge_engine.*` entry points import
# `edge_engine`, which isn't importable from a bare checkout. Without this
# the documented first step after installing (`python -m pytest -q`) fails
# with ModuleNotFoundError.
pip install -e .
