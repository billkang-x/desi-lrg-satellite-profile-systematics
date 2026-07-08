#!/usr/bin/env bash
#SBATCH -J abacus_fullbox_smoke
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o /public3/home/scg7816/desi_hod_mocks/logs/abacus_fullbox_smoke_%j.out
#SBATCH -e /public3/home/scg7816/desi_hod_mocks/logs/abacus_fullbox_smoke_%j.err

set -euo pipefail

cd /public3/home/scg7816/desi_hod_mocks

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH=/public1/soft/python/3.9.6/lib:${LD_LIBRARY_PATH:-}
source .venv-pycorr/bin/activate

python scripts/fit_abacus_fullbox_pycorr.py \
  --catalog-npz data/abacus_summit/processed/abacus_z0p8_fullbox_lrg_candidates.npz \
  --fsat-grid 0.025 \
  --alpha-s-grid 0.9 \
  --output-dir results/abacus_fullbox_fsat_alpha_smoke \
  --nthreads "${SLURM_CPUS_PER_TASK:-8}"
