#!/usr/bin/env bash
#SBATCH -J abacus_ext_sham_hod
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH -o /public3/home/scg7816/desi_hod_mocks/logs/abacus_ext_sham_hod_%j.out
#SBATCH -e /public3/home/scg7816/desi_hod_mocks/logs/abacus_ext_sham_hod_%j.err

set -euo pipefail

cd /public3/home/scg7816/desi_hod_mocks

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export LD_LIBRARY_PATH=/public1/soft/python/3.9.6/lib:${LD_LIBRARY_PATH:-}
source .venv-pycorr/bin/activate

CATALOG=data/abacus_summit/processed/abacus_z0p8_fullbox_lrg_candidates_fsat0p15.npz
GRID_FSAT=0.02,0.04,0.06,0.08,0.10,0.12,0.15
GRID_ALPHA=0.5,0.7,0.9,1.1,1.3
THREADS="${SLURM_CPUS_PER_TASK:-32}"

SHAM_DIR=results/abacus_fullbox_sham_fsat0p15_alpha_v1
HOD_DIR=results/abacus_fullbox_minimal_hod_fsat0p15_alpha_v1
SPLIT_DIR=results/abacus_fullbox_observable_splits_v1

python scripts/fit_abacus_fullbox_pycorr.py \
  --catalog-npz "${CATALOG}" \
  --fsat-grid "${GRID_FSAT}" \
  --alpha-s-grid "${GRID_ALPHA}" \
  --output-dir "${SHAM_DIR}" \
  --nthreads "${THREADS}"

python scripts/fit_abacus_minimal_hod_pycorr.py \
  --catalog-npz "${CATALOG}" \
  --fsat-grid "${GRID_FSAT}" \
  --alpha-s-grid "${GRID_ALPHA}" \
  --sigma-logm 0.25 \
  --satellite-power 1.0 \
  --output-dir "${HOD_DIR}" \
  --nthreads "${THREADS}"

python scripts/score_model_vector_observable_splits.py \
  --label sham_extended \
  --model-dir "${SHAM_DIR}" \
  --grid-csv "${SHAM_DIR}/abacus_fullbox_fsat_alpha_grid.csv" \
  --output-dir "${SPLIT_DIR}"

python scripts/score_model_vector_observable_splits.py \
  --label minimal_hod \
  --model-dir "${HOD_DIR}" \
  --grid-csv "${HOD_DIR}/abacus_fullbox_minimal_hod_fsat_alpha_grid.csv" \
  --output-dir "${SPLIT_DIR}"
