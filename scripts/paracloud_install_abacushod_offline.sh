#!/usr/bin/env bash
# Install AbacusHOD on ParaCloud from an uploaded Linux cp39 wheelhouse.

set -euo pipefail

PROJECT="${PROJECT:-/public3/home/scg7816/desi_hod_mocks}"
VENV="${VENV:-${PROJECT}/.venv-abacushod}"
WHEELHOUSE="${WHEELHOUSE:-${PROJECT}/wheelhouse_abacushod_linux_cp39}"

source /public1/soft/modules/module.sh
module load gcc/12.2 || true
module load gsl/2.5-cjj || module load gsl/2.8-intel21-gcc9-avx2 || true
module load python/3.9.6-cjj || module load python/3.9.6-lmy || true

cd "${PROJECT}"
python3 -m venv "${VENV}"
source "${VENV}/bin/activate"

python -m pip install --no-index --find-links "${WHEELHOUSE}" --upgrade pip setuptools wheel || true
python -m pip install --no-index --find-links "${WHEELHOUSE}" \
  numpy scipy pyyaml h5py asdf astropy numba \
  "importlib-metadata<8" "zipp<4" \
  blosc msgpack parallel_numpy_rng asdf-astropy abacusutils \
  future wurlitzer

if [[ -f "${WHEELHOUSE}/corrfunc-2.5.3.tar.gz" ]]; then
  python -m pip install --no-index --no-build-isolation --find-links "${WHEELHOUSE}" "${WHEELHOUSE}/corrfunc-2.5.3.tar.gz"
else
  echo "Missing ${WHEELHOUSE}/corrfunc-2.5.3.tar.gz" >&2
  exit 2
fi

python - <<'PY'
import importlib.util
mods = ["abacusnbody", "asdf", "h5py", "yaml", "numpy", "numba"]
for mod in mods:
    print(mod, bool(importlib.util.find_spec(mod)))
PY
