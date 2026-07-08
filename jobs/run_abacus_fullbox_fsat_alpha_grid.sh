#!/usr/bin/env bash
#SBATCH -J abacus_fullbox_fsat_alpha
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH -o /public3/home/scg7816/desi_hod_mocks/logs/abacus_fullbox_fsat_alpha_%j.out
#SBATCH -e /public3/home/scg7816/desi_hod_mocks/logs/abacus_fullbox_fsat_alpha_%j.err

set -euo pipefail

cd /public3/home/scg7816/desi_hod_mocks

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export LD_LIBRARY_PATH=/public1/soft/python/3.9.6/lib:${LD_LIBRARY_PATH:-}
source .venv-pycorr/bin/activate

CATALOG=data/abacus_summit/processed/abacus_z0p8_fullbox_lrg_candidates.npz
COV=results/matched_covariance/LRG_DR1_kmeans96_r0-1-2-3_full_wp_xi_covariance.npy
GRID_FSAT=0.015,0.020,0.025,0.030,0.035
GRID_ALPHA=0.7,0.8,0.9,1.0,1.1
THREADS="${SLURM_CPUS_PER_TASK:-32}"

python scripts/fit_abacus_fullbox_pycorr.py \
  --catalog-npz "${CATALOG}" \
  --fsat-grid "${GRID_FSAT}" \
  --alpha-s-grid "${GRID_ALPHA}" \
  --output-dir results/abacus_fullbox_fsat_alpha_heuristic_v2 \
  --nthreads "${THREADS}"

python scripts/fit_abacus_fullbox_pycorr.py \
  --catalog-npz "${CATALOG}" \
  --fsat-grid "${GRID_FSAT}" \
  --alpha-s-grid "${GRID_ALPHA}" \
  --covariance "${COV}" \
  --covariance-mode diagonal \
  --precision-scale 0.23157894736842105 \
  --output-dir results/abacus_fullbox_fsat_alpha_covjk96_diag_v2 \
  --nthreads "${THREADS}"

python scripts/fit_abacus_fullbox_pycorr.py \
  --catalog-npz "${CATALOG}" \
  --fsat-grid "${GRID_FSAT}" \
  --alpha-s-grid "${GRID_ALPHA}" \
  --covariance "${COV}" \
  --covariance-mode full \
  --covariance-rcond 1e-6 \
  --precision-scale 0.23157894736842105 \
  --output-dir results/abacus_fullbox_fsat_alpha_covjk96_full_v2 \
  --nthreads "${THREADS}"
