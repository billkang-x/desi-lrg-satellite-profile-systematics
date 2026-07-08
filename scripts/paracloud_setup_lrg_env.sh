#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/public3/home/scg7816/desi_hod_mocks}"
VENV="${PROJECT}/.venv-pycorr"

source /public1/soft/modules/module.sh
module load gcc/12.2 || true
module load python/3.9.6-cjj || module load python/3.9.6-lmy || module load python/3.7.12 || module load python/3.7.6 || true

mkdir -p "${PROJECT}/logs" "${PROJECT}/results" "${PROJECT}/data/dr1_lss_v1.5"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m venv "${VENV}"
source "${VENV}/bin/activate"

"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV}/bin/python" -m pip install numpy scipy astropy matplotlib pyyaml mpi4py
"${VENV}/bin/python" -m pip install Corrfunc
"${VENV}/bin/python" -m pip install "git+https://github.com/cosmodesi/pycorr.git"

"${VENV}/bin/python" - <<'PY'
import numpy
import astropy
from pycorr import TwoPointCorrelationFunction
print("numpy", numpy.__version__)
print("astropy", astropy.__version__)
print("pycorr import OK", TwoPointCorrelationFunction)
PY

echo "Environment ready: ${VENV}"
