#!/usr/bin/env bash
# Prepare an isolated AbacusHOD environment on ParaCloud.

set -euo pipefail

PROJECT="${PROJECT:-/public3/home/scg7816/desi_hod_mocks}"
VENV="${VENV:-${PROJECT}/.venv-abacushod}"

source /public1/soft/modules/module.sh
module load gcc/12.2 || true
module load python/3.9.6-cjj || module load python/3.9.6-lmy || module load python/3.7.12 || true

cd "${PROJECT}"
python -m venv "${VENV}"
source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy scipy pyyaml h5py asdf numba astropy
python -m pip install git+https://github.com/abacusorg/abacusutils.git

python - <<'PY'
import importlib.util
mods = ["abacusnbody", "asdf", "h5py", "yaml", "numpy", "numba"]
for mod in mods:
    print(mod, bool(importlib.util.find_spec(mod)))
PY
