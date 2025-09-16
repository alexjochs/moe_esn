#!/bin/bash
# Bootstrap a Python virtual environment for the Dask smoke test.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR=${VENV_DIR:-"$SCRIPT_DIR/.venv"}

module purge
module load python/3.12 || true

if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install --upgrade "dask[distributed]" dask-jobqueue

echo "Virtualenv ready at $VENV_DIR"
