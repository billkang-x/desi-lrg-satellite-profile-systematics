#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/public3/home/scg7816/desi_hod_mocks}"
DATA_DIR="${PROJECT}/data/dr1_lss_v1.5"
BASE_URL="https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5"
RANDOM_INDICES="${RANDOM_INDICES:-0}"

mkdir -p "${DATA_DIR}"

download_one() {
  local name="$1"
  local url="${BASE_URL}/${name}"
  local dest="${DATA_DIR}/${name}"
  echo "Downloading ${name}"
  curl -L -C - --retry 8 --retry-delay 10 -o "${dest}" "${url}"
}

for cap in NGC SGC; do
  download_one "LRG_${cap}_clustering.dat.fits"
  download_one "LRG_${cap}_nz.txt"
  IFS=',' read -ra indices <<< "${RANDOM_INDICES}"
  for idx in "${indices[@]}"; do
    download_one "LRG_${cap}_${idx}_clustering.ran.fits"
  done
done

ls -lh "${DATA_DIR}"

