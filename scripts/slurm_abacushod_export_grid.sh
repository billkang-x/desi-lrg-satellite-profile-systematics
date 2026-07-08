#!/usr/bin/env bash
# Export a small AbacusHOD LRG grid once the minimal z0.800 data are staged.

#SBATCH -J abacus_hod_grid
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 08:00:00
#SBATCH -o logs/abacus_hod_grid_%j.out
#SBATCH -e logs/abacus_hod_grid_%j.err

set -euo pipefail

PROJECT="${PROJECT:-/public3/home/scg7816/desi_hod_mocks}"
VENV="${VENV:-${PROJECT}/.venv-abacushod}"
CONFIG="${CONFIG:-configs/abacus_hod_lrg_z0p8_minimal.yaml}"
OUTDIR="${OUTDIR:-results/abacushod_lrg_z0p8_grid}"
LOGM1_GRID="${LOGM1_GRID:-14.2 14.4 14.6}"
ALPHA_S_GRID="${ALPHA_S_GRID:-0.8 1.0 1.2}"
RESEED="${RESEED:-12345}"
NTHREAD="${NTHREAD:-32}"

source /public1/soft/modules/module.sh
module load gcc/12.2 || true
module load python/3.9.6-cjj || module load python/3.9.6-lmy || module load python/3.7.12 || true

cd "${PROJECT}"
source "${VENV}/bin/activate"
mkdir -p "${OUTDIR}" logs results

python scripts/check_abacus_minimal_tree.py \
  --sim-root data/abacus_summit_minimal \
  --sim-name AbacusSummit_base_c000_ph000 \
  --zdir z0.800 \
  --min-gb 80 \
  --output results/abacus_minimal_tree_check.json

for logm1 in ${LOGM1_GRID}; do
  for alpha_s in ${ALPHA_S_GRID}; do
    tag="logM1${logm1}_alphaS${alpha_s}"
    tag="${tag//./p}"
    output="${OUTDIR}/LRG_z0p8_${tag}.hdf5"
    echo "Running ${tag}"
    python scripts/export_abacushod_mock.py \
      --config "${CONFIG}" \
      --output "${output}" \
      --tracer LRG \
      --alpha-s "${alpha_s}" \
      --set "logM1=${logm1}" \
      --reseed "${RESEED}" \
      --nthread "${NTHREAD}"
  done
done
