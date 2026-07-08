#!/usr/bin/env bash
#SBATCH -J desi_lrg_pycorr
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -o logs/desi_lrg_pycorr_%j.out
#SBATCH -e logs/desi_lrg_pycorr_%j.err

set -euo pipefail

PROJECT="${PROJECT:-/public3/home/scg7816/desi_hod_mocks}"
VENV="${PROJECT}/.venv-pycorr"
DATA_DIR="${DATA_DIR:-data/dr1_lss_v1.5}"
TRACER="${TRACER:-LRG}"
ZMIN="${ZMIN:-0.6}"
ZMAX="${ZMAX:-0.8}"
RANDOM_INDICES="${RANDOM_INDICES:-0}"
CAPS="${CAPS:-NGC,SGC}"
ENGINE="${ENGINE:-corrfunc}"
OUTPUT_DIR="${OUTPUT_DIR:-results/pycorr_lrg}"
MAX_DATA_ROWS="${MAX_DATA_ROWS:-0}"
MAX_RANDOM_ROWS="${MAX_RANDOM_ROWS:-0}"
JACKKNIFE_NRA="${JACKKNIFE_NRA:-0}"
JACKKNIFE_NDEC="${JACKKNIFE_NDEC:-0}"
JACKKNIFE_EXCLUDE_SAMPLE="${JACKKNIFE_EXCLUDE_SAMPLE:--1}"
JACKKNIFE_RA_MIN="${JACKKNIFE_RA_MIN:-0}"
JACKKNIFE_RA_MAX="${JACKKNIFE_RA_MAX:-360}"
JACKKNIFE_DEC_MIN="${JACKKNIFE_DEC_MIN:--30}"
JACKKNIFE_DEC_MAX="${JACKKNIFE_DEC_MAX:-90}"

source /public1/soft/modules/module.sh
module load gcc/12.2 || true
module load gsl/2.5-cjj || module load gsl/2.8-intel21-gcc9-avx2 || true
module load python/3.9.6-cjj || module load python/3.9.6-lmy || module load python/3.7.12 || module load python/3.7.6 || true
cd "${PROJECT}"
source "${VENV}/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${OUTPUT_DIR}" logs

"${VENV}/bin/python" scripts/measure_lrg_pycorr.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --tracer "${TRACER}" \
  --caps "${CAPS}" \
  --random-indices "${RANDOM_INDICES}" \
  --zmin "${ZMIN}" \
  --zmax "${ZMAX}" \
  --engine "${ENGINE}" \
  --nthreads "${OMP_NUM_THREADS}" \
  --max-data-rows "${MAX_DATA_ROWS}" \
  --max-random-rows "${MAX_RANDOM_ROWS}" \
  --jackknife-nra "${JACKKNIFE_NRA}" \
  --jackknife-ndec "${JACKKNIFE_NDEC}" \
  --jackknife-exclude-sample "${JACKKNIFE_EXCLUDE_SAMPLE}" \
  --jackknife-ra-min "${JACKKNIFE_RA_MIN}" \
  --jackknife-ra-max "${JACKKNIFE_RA_MAX}" \
  --jackknife-dec-min "${JACKKNIFE_DEC_MIN}" \
  --jackknife-dec-max "${JACKKNIFE_DEC_MAX}" \
  --s-min 2 --s-max 60 --s-bins 29 --mu-bins 120 --ells 0,2,4 \
  --rp-min 0.3 --rp-max 30 --rp-bins 18 --rp-spacing log --pi-max 80 --pi-bins 80
