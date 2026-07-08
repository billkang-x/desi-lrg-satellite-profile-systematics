"""Export DESI DR1 full-shape/BAO correlation data and covariance.

This converts the official DESI VAC HDF5 files into a simple CSV/NPY contract
for HOD/SHAM comparisons. It supports two common inputs:

- a standalone observable file with groups "0", "2", "4"
- a likelihood file with groups "observable" and "covariance/value"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode(value[()])
    return value


def get_observable_root(handle: h5py.File | h5py.Group) -> h5py.Group:
    if "observable" in handle:
        return handle["observable"]
    return handle


def available_ells(root: h5py.Group) -> list[int]:
    ells = []
    for key in root.keys():
        if key.lstrip("-").isdigit() and "value" in root[key]:
            ells.append(int(key))
    return sorted(ells)


def read_observable(path: Path, ells: list[int] | None = None) -> tuple[list[dict[str, float | int]], dict[str, object]]:
    rows: list[dict[str, float | int]] = []
    with h5py.File(path, "r") as handle:
        root = get_observable_root(handle)
        if ells is None:
            ells = available_ells(root)
        for ell in ells:
            group = root[str(ell)]
            s = np.asarray(group["s"], dtype="f8")
            value = np.asarray(group["value"], dtype="f8")
            s_edges = np.asarray(group["s_edges"], dtype="f8")
            for i, (radius, val) in enumerate(zip(s, value)):
                rows.append(
                    {
                        "index": len(rows),
                        "statistic": "xi",
                        "component": f"xi{ell}",
                        "ell": int(ell),
                        "r": float(radius),
                        "s_low": float(s_edges[i, 0]),
                        "s_high": float(s_edges[i, 1]),
                        "value": float(val),
                    }
                )
        metadata = {
            "source": str(path),
            "ells": ells,
            "n_data": len(rows),
            "zeff": float(handle.attrs["zeff"]) if "zeff" in handle.attrs else None,
            "name": decode(handle["name"][()]) if "name" in handle else None,
        }
    return rows, metadata


def read_covariance(path: Path, ells: list[int] | None = None) -> tuple[np.ndarray, list[dict[str, float | int]], dict[str, object]]:
    with h5py.File(path, "r") as handle:
        if "covariance" in handle:
            cov_group = handle["covariance"]
            cov = np.asarray(cov_group["value"], dtype="f8")
            observable_root = cov_group["observable"]
        else:
            cov = np.asarray(handle["value"], dtype="f8")
            observable_root = handle["observable"]
        full_rows: list[dict[str, float | int]] = []
        full_ells = available_ells(observable_root)
        selected_ells = full_ells if ells is None else ells
        selected_indices: list[int] = []
        for ell in full_ells:
            group = observable_root[str(ell)]
            s = np.asarray(group["s"], dtype="f8")
            s_edges = np.asarray(group["s_edges"], dtype="f8")
            value = np.asarray(group["value"], dtype="f8")
            for i, (radius, val) in enumerate(zip(s, value)):
                row_index = len(full_rows)
                full_rows.append(
                    {
                        "index": row_index,
                        "statistic": "xi",
                        "component": f"xi{ell}",
                        "ell": int(ell),
                        "r": float(radius),
                        "s_low": float(s_edges[i, 0]),
                        "s_high": float(s_edges[i, 1]),
                        "value": float(val),
                    }
                )
                if ell in selected_ells:
                    selected_indices.append(row_index)
        if cov.shape[0] != len(full_rows) or cov.shape[1] != len(full_rows):
            raise ValueError(f"Covariance shape {cov.shape} does not match {len(full_rows)} observable bins")
        rows = [dict(full_rows[index]) for index in selected_indices]
        for new_index, row in enumerate(rows):
            row["index"] = new_index
        cov = cov[np.ix_(selected_indices, selected_indices)]
        metadata = {
            "source": str(path),
            "ells": selected_ells,
            "available_ells": full_ells,
            "n_data": len(rows),
            "covariance_shape": list(cov.shape),
            "name": decode(handle["name"][()]) if "name" in handle else None,
        }
    return cov, rows, metadata


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observable", type=Path, default=Path("data/desi_dr1_full_shape_bao_v1.0/correlation-recsym-poles_LRG_GCcomb_z0.6-0.8.h5"))
    parser.add_argument("--covariance", type=Path, default=Path("data/desi_dr1_full_shape_bao_v1.0/covariance_correlation-recsym-poles_LRG_GCcomb_z0.6-0.8.h5"))
    parser.add_argument("--likelihood", type=Path, default=Path("data/desi_dr1_full_shape_bao_v1.0/likelihood_correlation-recon-poles_LRG_GCcomb_z0.6-0.8.h5"))
    parser.add_argument("--ells", default="0,2")
    parser.add_argument("--output-dir", type=Path, default=Path("results/desi_vac_covariance"))
    args = parser.parse_args(argv)

    ells = [int(item.strip()) for item in args.ells.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    obs_rows, obs_meta = read_observable(args.observable, ells=ells)
    cov, cov_rows, cov_meta = read_covariance(args.covariance, ells=ells)
    likelihood_cov, likelihood_rows, likelihood_meta = read_covariance(args.likelihood, ells=ells)

    write_rows(args.output_dir / "desi_dr1_lrg_z0p6-0p8_recsym_observable.csv", obs_rows)
    write_rows(args.output_dir / "desi_dr1_lrg_z0p6-0p8_rascalc_covariance_bins.csv", cov_rows)
    write_rows(args.output_dir / "desi_dr1_lrg_z0p6-0p8_likelihood_bins.csv", likelihood_rows)
    np.save(args.output_dir / "desi_dr1_lrg_z0p6-0p8_rascalc_covariance.npy", cov)
    np.save(args.output_dir / "desi_dr1_lrg_z0p6-0p8_likelihood_covariance.npy", likelihood_cov)

    summary = {
        "observable": obs_meta,
        "rascalc_covariance": cov_meta,
        "likelihood_covariance": likelihood_meta,
        "recommended_first_likelihood": "Use likelihood_covariance for xi0+xi2 tests matching the DESI VAC likelihood binning; keep wp covariance separate until unreconstructed wp/mock covariance is built.",
        "caveats": [
            "The VAC likelihood downloaded here is reconstructed correlation-poles; our current pycorr data vector is unreconstructed.",
            "RascalC recsym covariance includes correlation-poles only, not wp(rp).",
        ],
    }
    with (args.output_dir / "desi_dr1_lrg_z0p6-0p8_covariance_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(args.output_dir / "desi_dr1_lrg_z0p6-0p8_covariance_summary.json")
    print("rascalc covariance", cov.shape)
    print("likelihood covariance", likelihood_cov.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
