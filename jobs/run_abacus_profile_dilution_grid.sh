#!/usr/bin/env bash
#SBATCH -J abacus_profile_dil
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH -o /public3/home/scg7816/desi_hod_mocks/logs/abacus_profile_dilution_%j.out
#SBATCH -e /public3/home/scg7816/desi_hod_mocks/logs/abacus_profile_dilution_%j.err

set -euo pipefail

cd /public3/home/scg7816/desi_hod_mocks

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export LD_LIBRARY_PATH=/public1/soft/python/3.9.6/lib:${LD_LIBRARY_PATH:-}
source .venv-pycorr/bin/activate

CATALOG=data/abacus_summit/processed/abacus_z0p8_fullbox_lrg_candidates_fsat0p15.npz
FSAT=0.04,0.06,0.08,0.10,0.12
ALPHA=0.7,1.1,1.3
RSCALE=1.0,1.5,2.0,3.0,4.0
THREADS="${SLURM_CPUS_PER_TASK:-32}"

HOD_DIR=results/abacus_fullbox_minimal_hod_profile_dilution_v1
SHAM_DIR=results/abacus_fullbox_sham_profile_dilution_v1
SPLIT_DIR=results/abacus_fullbox_profile_dilution_splits_v1

python scripts/fit_abacus_profile_dilution_pycorr.py \
  --catalog-npz "${CATALOG}" \
  --mode minimal_hod \
  --fsat-grid "${FSAT}" \
  --alpha-s-grid "${ALPHA}" \
  --radial-scale-grid "${RSCALE}" \
  --sigma-logm 0.25 \
  --satellite-power 1.0 \
  --output-dir "${HOD_DIR}" \
  --nthreads "${THREADS}"

python scripts/fit_abacus_profile_dilution_pycorr.py \
  --catalog-npz "${CATALOG}" \
  --mode sham \
  --fsat-grid "0.04,0.08,0.12" \
  --alpha-s-grid "0.7,1.1" \
  --radial-scale-grid "${RSCALE}" \
  --output-dir "${SHAM_DIR}" \
  --nthreads "${THREADS}"

python scripts/score_profile_dilution_observable_splits.py \
  --label minimal_hod_profile \
  --model-dir "${HOD_DIR}" \
  --grid-csv "${HOD_DIR}/abacus_fullbox_minimal_hod_profile_dilution_grid.csv" \
  --output-dir "${SPLIT_DIR}"

python scripts/score_profile_dilution_observable_splits.py \
  --label sham_profile \
  --model-dir "${SHAM_DIR}" \
  --grid-csv "${SHAM_DIR}/abacus_fullbox_sham_profile_dilution_grid.csv" \
  --output-dir "${SPLIT_DIR}"
