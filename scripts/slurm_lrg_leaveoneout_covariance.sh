#!/usr/bin/env bash
#SBATCH -J lrg_loo_cov
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -o logs/lrg_loo_cov_%A_%a.out
#SBATCH -e logs/lrg_loo_cov_%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-/public3/home/scg7816/desi_hod_mocks}"
VENV="${PROJECT}/.venv-pycorr"
DATA_DIR="${DATA_DIR:-data/dr1_lss_v1.5}"
TRACER="${TRACER:-LRG}"
CAPS="${CAPS:-NGC,SGC}"
RANDOM_INDICES="${RANDOM_INDICES:-0-3}"
ZMIN="${ZMIN:-0.6}"
ZMAX="${ZMAX:-0.8}"
JACKKNIFE_NRA="${JACKKNIFE_NRA:-6}"
JACKKNIFE_NDEC="${JACKKNIFE_NDEC:-4}"
JACKKNIFE_CENTERS="${JACKKNIFE_CENTERS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/pycorr_lrg_leaveoneout}"
DATAVECTOR_DIR="${DATAVECTOR_DIR:-results/datavectors_leaveoneout}"
ENGINE="${ENGINE:-corrfunc}"

source /public1/soft/modules/module.sh
module load gcc/12.2 || true
module load gsl/2.5-cjj || module load gsl/2.8-intel21-gcc9-avx2 || true
module load python/3.9.6-cjj || module load python/3.9.6-lmy || module load python/3.7.12 || module load python/3.7.6 || true
cd "${PROJECT}"
source "${VENV}/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${OUTPUT_ROOT}" "${DATAVECTOR_DIR}" logs

REGION="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR="${OUTPUT_ROOT}/exclude_${REGION}"

JACKKNIFE_ARGS=(--jackknife-exclude-sample "${REGION}")
if [[ -n "${JACKKNIFE_CENTERS}" ]]; then
  JACKKNIFE_ARGS+=(--jackknife-centers "${JACKKNIFE_CENTERS}")
else
  JACKKNIFE_ARGS+=(--jackknife-nra "${JACKKNIFE_NRA}" --jackknife-ndec "${JACKKNIFE_NDEC}")
fi

"${VENV}/bin/python" scripts/measure_lrg_pycorr.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${RUN_DIR}" \
  --tracer "${TRACER}" \
  --caps "${CAPS}" \
  --random-indices "${RANDOM_INDICES}" \
  --zmin "${ZMIN}" \
  --zmax "${ZMAX}" \
  --engine "${ENGINE}" \
  --nthreads "${OMP_NUM_THREADS}" \
  "${JACKKNIFE_ARGS[@]}" \
  --s-min 2 --s-max 60 --s-bins 29 --mu-bins 120 --ells 0,2,4 \
  --rp-min 0.3 --rp-max 30 --rp-bins 18 --rp-spacing log --pi-max 80 --pi-bins 80

RANDOM_LABEL="${RANDOM_INDICES//,/-}"
if [[ "${RANDOM_INDICES}" =~ ^[0-9]+-[0-9]+$ ]]; then
  START="${RANDOM_INDICES%-*}"
  STOP="${RANDOM_INDICES#*-}"
  RANDOM_LABEL="$(seq -s- "${START}" "${STOP}")"
fi
LABEL="${TRACER}_DR1_${CAPS/,/-}_z$(printf '%.3f' "${ZMIN}")-$(printf '%.3f' "${ZMAX}")_r${RANDOM_LABEL}"

"${VENV}/bin/python" scripts/build_lrg_datavector.py \
  --input-dir "${RUN_DIR}" \
  --output-dir "${DATAVECTOR_DIR}/exclude_${REGION}" \
  --label "${LABEL}" \
  --ells 0,2 \
  --wp-min 0.5 --wp-max 30 \
  --xi-min 5 --xi-max 60
