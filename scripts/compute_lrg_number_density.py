"""Compute DESI LRG number density from LSScats n(z) files.

The DESI LSS n(z) files provide n(z), weighted counts, and comoving shell
volumes per cap. For an HOD/SHAM density term we want the volume-weighted
number density over the same redshift interval used by the clustering vector.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_nz(path: Path) -> tuple[dict[str, float], np.ndarray]:
    metadata: dict[str, float] = {}
    comments = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            comments.append(line.strip())
    for line in comments:
        if line.startswith("#area is"):
            metadata["area_deg2"] = float(line.split("is", 1)[1].split("square", 1)[0])
        elif line.startswith("#effective area is"):
            metadata["effective_area_deg2"] = float(line.split("is", 1)[1].split("square", 1)[0])
    table = np.loadtxt(path)
    if table.ndim == 1:
        table = table[None, :]
    return metadata, table


def cap_summary(path: Path, zmin: float, zmax: float) -> dict[str, float | str]:
    metadata, table = read_nz(path)
    zmid, zlow, zhigh, nbar, nbin, vol = [table[:, i] for i in range(6)]
    tolerance = 1e-8
    mask = (zlow >= zmin - tolerance) & (zhigh <= zmax + tolerance)
    if not np.any(mask):
        raise ValueError(f"No full n(z) bins within {zmin} <= z < {zmax} for {path}")
    total_n = float(np.sum(nbin[mask]))
    total_vol = float(np.sum(vol[mask]))
    return {
        "cap": path.name.split("_")[1],
        "path": str(path),
        "area_deg2": float(metadata.get("area_deg2", np.nan)),
        "effective_area_deg2": float(metadata.get("effective_area_deg2", np.nan)),
        "n_weighted": total_n,
        "volume_mpc_h3": total_vol,
        "nbar_h3_mpc3": total_n / total_vol,
        "zeff_volume_weighted": float(np.sum(zmid[mask] * vol[mask]) / total_vol),
        "zmin": zmin,
        "zmax": zmax,
        "n_bins": int(np.count_nonzero(mask)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dr1_lss_v1.5"))
    parser.add_argument("--tracer", default="LRG")
    parser.add_argument("--caps", default="NGC,SGC")
    parser.add_argument("--zmin", type=float, default=0.6)
    parser.add_argument("--zmax", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=Path("results/number_density"))
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    caps = [item.strip() for item in args.caps.split(",") if item.strip()]
    summaries = [
        cap_summary(args.data_dir / f"{args.tracer}_{cap}_nz.txt", args.zmin, args.zmax)
        for cap in caps
    ]
    total_n = float(sum(item["n_weighted"] for item in summaries))
    total_vol = float(sum(item["volume_mpc_h3"] for item in summaries))
    combined = {
        "tracer": args.tracer,
        "caps": caps,
        "zmin": args.zmin,
        "zmax": args.zmax,
        "n_weighted": total_n,
        "volume_mpc_h3": total_vol,
        "nbar_h3_mpc3": total_n / total_vol,
        "zeff_volume_weighted": float(
            sum(item["zeff_volume_weighted"] * item["volume_mpc_h3"] for item in summaries)
            / total_vol
        ),
        "cap_summaries": summaries,
        "notes": [
            "nbar_h3_mpc3 is volume-weighted over full n(z) bins inside the requested redshift interval.",
            "Volumes are the DESI LSScats Vol_bin values, in (Mpc/h)^3.",
        ],
    }

    tag = f"{args.tracer}_GCcomb_z{args.zmin:.1f}-{args.zmax:.1f}".replace(".", "p")
    json_path = args.output_dir / f"{tag}_number_density.json"
    csv_path = args.output_dir / f"{tag}_number_density_by_cap.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(combined, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(json_path)
    print(csv_path)
    print(f"nbar_h3_mpc3={combined['nbar_h3_mpc3']:.8e}")
    print(f"zeff={combined['zeff_volume_weighted']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
