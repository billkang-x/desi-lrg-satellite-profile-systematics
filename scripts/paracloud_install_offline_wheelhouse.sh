#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/public3/home/scg7816/desi_hod_mocks}"
VENV="${PROJECT}/.venv-pycorr"
WHEELHOUSE="${PROJECT}/wheelhouse"

source /public1/soft/modules/module.sh
module purge >/dev/null 2>&1 || true
module load gcc/12.2 || true
module load gsl/2.5-cjj || module load gsl/2.8-intel21-gcc9-avx2 || true
module load python/3.9.6-cjj || module load python/3.9.6-lmy || module load python/3.7.12 || module load python/3.7.6 || true

cd "${PROJECT}"
rm -rf "${VENV}"
python3 -m venv "${VENV}"
source "${VENV}/bin/activate"

"${VENV}/bin/python" -m pip install --no-index --find-links "${WHEELHOUSE}" setuptools wheel packaging Cython
"${VENV}/bin/python" -m pip install --no-index --find-links "${WHEELHOUSE}" numpy scipy fitsio astropy pyyaml mpi4py future wurlitzer
"${VENV}/bin/python" -m pip install --no-index --find-links "${WHEELHOUSE}" --no-build-isolation "${WHEELHOUSE}/Corrfunc-desi.zip"
"${VENV}/bin/python" -m pip install --no-index --find-links "${WHEELHOUSE}" --no-build-isolation "${WHEELHOUSE}/pycorr-1.0.0.zip"

"${VENV}/bin/python" - <<'PY'
import numpy
import scipy
import fitsio
import astropy
import mpi4py
import Corrfunc
from pycorr import TwoPointCorrelationFunction
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("fitsio", fitsio.__version__)
print("astropy", astropy.__version__)
print("mpi4py", mpi4py.__version__)
print("Corrfunc", getattr(Corrfunc, "__version__", "unknown"))
print("pycorr import OK", TwoPointCorrelationFunction)
PY

echo "Offline environment ready: ${VENV}"
