#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "Creating Python environment..."
    "${PYTHON_BIN:-python3}" -m venv .venv
fi

echo "Installing build tools..."
.venv/bin/python -m pip install -e ".[build]"

echo "Building Chronophoto..."
.venv/bin/python -m PyInstaller --noconfirm --clean chronophoto.spec

if [[ "$(uname -s)" == "Darwin" ]]; then
    dist/Chronophoto.app/Contents/MacOS/Chronophoto --version
    echo "Build ready: dist/Chronophoto.app"
else
    dist/Chronophoto/Chronophoto --version
    echo "Build ready: dist/Chronophoto/Chronophoto"
fi
