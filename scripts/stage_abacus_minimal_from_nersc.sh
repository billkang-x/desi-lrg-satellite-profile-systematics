#!/usr/bin/env bash
# Extract or copy the minimal AbacusSummit z0.800 subset needed by AbacusHOD.
#
# Intended use:
#   1. Run on NERSC if CFS/HPSS is available.
#   2. Stage into a scratch directory.
#   3. Transfer staged directory to ParaCloud project data/abacus_summit_minimal.
#
# This script does not download the full 6.6 TB halo tarball.

set -euo pipefail

SIM_NAME="${SIM_NAME:-AbacusSummit_base_c000_ph000}"
Z_DIR="${Z_DIR:-z0.800}"
STAGE_ROOT="${STAGE_ROOT:-${PWD}/abacus_summit_minimal}"
CFS_ROOT="${CFS_ROOT:-/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit}"
HPSS_TAR="${HPSS_TAR:-/nersc/projects/desi/cosmosim/Abacus/${SIM_NAME}/Abacus_${SIM_NAME}_halos.tar}"
MODE="${MODE:-auto}"  # auto, cfs, htar-list, htar-extract

TARGET="${STAGE_ROOT}/${SIM_NAME}/halos/${Z_DIR}"
mkdir -p "${TARGET}"

echo "SIM_NAME=${SIM_NAME}"
echo "Z_DIR=${Z_DIR}"
echo "STAGE_ROOT=${STAGE_ROOT}"
echo "MODE=${MODE}"

copy_from_cfs() {
  local src="${CFS_ROOT}/${SIM_NAME}/halos/${Z_DIR}"
  if [[ ! -d "${src}" ]]; then
    echo "CFS source not found: ${src}" >&2
    return 1
  fi
  rsync -ah --info=progress2 "${src}/halo_info" "${TARGET}/"
  rsync -ah --info=progress2 "${src}/halo_rv_A" "${TARGET}/"
}

list_hpss() {
  htar -t -f "${HPSS_TAR}" "./halos/${Z_DIR}/halo_info" "./halos/${Z_DIR}/halo_rv_A"
}

extract_hpss() {
  mkdir -p "${STAGE_ROOT}/${SIM_NAME}"
  cd "${STAGE_ROOT}/${SIM_NAME}"
  htar -x -f "${HPSS_TAR}" "./halos/${Z_DIR}/halo_info" "./halos/${Z_DIR}/halo_rv_A"
}

case "${MODE}" in
  cfs)
    copy_from_cfs
    ;;
  htar-list)
    list_hpss
    ;;
  htar-extract)
    extract_hpss
    ;;
  auto)
    if [[ -d "${CFS_ROOT}/${SIM_NAME}/halos/${Z_DIR}" ]]; then
      copy_from_cfs
    elif command -v htar >/dev/null 2>&1; then
      extract_hpss
    else
      echo "Neither CFS source nor htar is available on this machine." >&2
      exit 2
    fi
    ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

echo "Staged tree:"
du -sh "${STAGE_ROOT}/${SIM_NAME}/halos/${Z_DIR}"/*
