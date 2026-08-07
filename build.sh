#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${CHRONOPHOTO_VENV:-.venv}"
if [[ "$VENV_DIR" == ".venv" && -x .venv/Scripts/python.exe && ! -x .venv/bin/python ]]; then
    VENV_DIR=".venv-posix"
fi
VENV_PYTHON="$VENV_DIR/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Creating Python environment..."
    "${PYTHON_BIN:-python3}" -m venv "$VENV_DIR"
fi

echo "Installing build tools..."
"$VENV_PYTHON" -m pip install -e ".[build]"

echo "Building Chronophoto..."
"$VENV_PYTHON" -m PyInstaller --noconfirm --clean chronophoto.spec

if [[ "$(uname -s)" == "Darwin" ]]; then
    dist/Chronophoto.app/Contents/MacOS/Chronophoto --version
    echo "Build ready: dist/Chronophoto.app"
else
    dist/Chronophoto/Chronophoto --version
    echo "Build ready: dist/Chronophoto/Chronophoto"
fi
