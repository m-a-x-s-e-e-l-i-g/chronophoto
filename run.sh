#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "Creating Python environment..."
    "${PYTHON_BIN:-python3}" -m venv .venv
fi

echo "Preparing Chronophoto..."
.venv/bin/python -m pip install -e .
exec .venv/bin/python -m chronophoto "$@"
