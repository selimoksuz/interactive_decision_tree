#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=${PYTHON_BIN:-python3}
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=${PYTHON_BIN:-python}
else
    echo "Python 3.10+ is required but was not found on PATH." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_release.py" --platform linux --archive both "$@"
