#!/usr/bin/env bash
#SBATCH -J abacus_nfw_prof
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH -o /public3/home/scg7816/desi_hod_mocks/logs/abacus_nfw_profile_%j.out
#SBATCH -e /public3/home/scg7816/desi_hod_mocks/logs/abacus_nfw_profile_%j.err

set -euo pipefail

cd /public3/home/scg7816/desi_hod_mocks

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export LD_LIBRARY_PATH=/public1/soft/python/3.9.6/lib:${LD_LIBRARY_PATH:-}
source .venv-pycorr/bin/activate

CATALOG=data/abacus_summit/processed/abacus_z0p8_fullbox_lrg_candidates_fsat0p15.npz
FSAT=0.04,0.06,0.08,0.10,0.12
ALPHA=0.7,1.1,1.3
CRATIO=0.35,0.50,0.70,1.00,1.30
THREADS="${SLURM_CPUS_PER_TASK:-32}"

MODEL_DIR=results/abacus_fullbox_minimal_hod_nfw_profile_v1
SPLIT_DIR=results/abacus_fullbox_nfw_profile_splits_v1

python scripts/fit_abacus_nfw_profile_pycorr.py \
  --catalog-npz "${CATALOG}" \
  --fsat-grid "${FSAT}" \
  --alpha-s-grid "${ALPHA}" \
  --concentration-ratio-grid "${CRATIO}" \
  --c-ref 5.0 \
  --sigma-logm 0.25 \
  --satellite-power 1.0 \
  --output-dir "${MODEL_DIR}" \
  --nthreads "${THREADS}"

python scripts/score_nfw_profile_observable_splits.py \
  --label minimal_hod_nfw_profile \
  --model-dir "${MODEL_DIR}" \
  --grid-csv "${MODEL_DIR}/abacus_fullbox_minimal_hod_nfw_profile_grid.csv" \
  --output-dir "${SPLIT_DIR}"
