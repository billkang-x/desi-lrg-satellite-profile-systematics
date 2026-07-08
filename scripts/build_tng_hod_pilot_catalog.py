"""Build a compact HOD/SHAM pilot catalog from minimal TNG groupcat fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def read_dataset(path: Path, group: str, field: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        for name in [f"{group}/{field}", field, f"{group}s/{field}"]:
            if name in handle:
                return np.asarray(handle[name])
        matches: list[str] = []

        def visitor(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset) and name.endswith(field):
                matches.append(name)

        handle.visititems(visitor)
        if len(matches) == 1:
            return np.asarray(handle[matches[0]])
        raise KeyError(f"Could not find {group}/{field} in {path}; matches={matches}")


def field_path(raw_dir: Path, sim: str, snapshot: int, group: str, field: str) -> Path:
    path = raw_dir / sim / f"snap{snapshot:03d}" / f"{sim}_snap{snapshot:03d}_{group}_{field}.hdf5"
    if path.exists():
        return path
    matches = sorted(raw_dir.rglob(f"*_{group}_{field}.hdf5"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(path)


def load_metadata(raw_dir: Path, sim: str, snapshot: int) -> dict:
    path = raw_dir / sim / f"snap{snapshot:03d}" / f"{sim}_snap{snapshot:03d}_minimal_groupcat_metadata.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    return {}


def build_catalog(args: argparse.Namespace) -> dict[str, object]:
    raw_dir = Path(args.raw_dir)
    metadata = load_metadata(raw_dir, args.sim, args.snapshot)
    boxsize = float(args.boxsize if args.boxsize is not None else metadata.get("boxsize_ckpc_h", 205000.0) / 1000.0)
    redshift = float(args.redshift if args.redshift is not None else metadata.get("redshift", 0.700106353718523))

    pos = read_dataset(field_path(raw_dir, args.sim, args.snapshot, "Subhalo", "SubhaloPos"), "Subhalo", "SubhaloPos")
    vel = read_dataset(field_path(raw_dir, args.sim, args.snapshot, "Subhalo", "SubhaloVel"), "Subhalo", "SubhaloVel")
    mass_type = read_dataset(field_path(raw_dir, args.sim, args.snapshot, "Subhalo", "SubhaloMassType"), "Subhalo", "SubhaloMassType")
    grnr = read_dataset(field_path(raw_dir, args.sim, args.snapshot, "Subhalo", "SubhaloGrNr"), "Subhalo", "SubhaloGrNr")
    flag = read_dataset(field_path(raw_dir, args.sim, args.snapshot, "Subhalo", "SubhaloFlag"), "Subhalo", "SubhaloFlag")
    first_sub = read_dataset(field_path(raw_dir, args.sim, args.snapshot, "Group", "GroupFirstSub"), "Group", "GroupFirstSub")

    subhalo_id = np.arange(len(grnr), dtype=np.int64)
    valid_host = (grnr >= 0) & (grnr < len(first_sub))
    central_id = np.full(len(grnr), -1, dtype=np.int64)
    central_id[valid_host] = first_sub[grnr[valid_host]]
    is_sat = subhalo_id != central_id
    stellar_mass = np.asarray(mass_type[:, 4], dtype="f8")
    mask = (
        (np.asarray(flag) > 0)
        & valid_host
        & (central_id >= 0)
        & np.isfinite(stellar_mass)
        & (stellar_mass >= args.min_stellar_mass_1e10msun_h)
        & np.all(np.isfinite(pos), axis=1)
        & np.all(np.isfinite(vel), axis=1)
    )
    candidate_indices = np.flatnonzero(mask)
    if args.target_number_density > 0:
        target_count = int(round(args.target_number_density * boxsize**3 * args.candidate_multiplier))
    elif args.target_count > 0:
        target_count = int(round(args.target_count * args.candidate_multiplier))
    else:
        target_count = len(candidate_indices)
    target_count = min(max(target_count, 1), len(candidate_indices))
    order = candidate_indices[np.argsort(stellar_mass[candidate_indices])[::-1]]
    selected = np.sort(order[:target_count])
    actual_nbar = len(selected) / boxsize**3

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        for name, values in {
            "x": pos[selected, 0] / 1000.0,
            "y": pos[selected, 1] / 1000.0,
            "z": pos[selected, 2] / 1000.0,
            "vx": vel[selected, 0],
            "vy": vel[selected, 1],
            "vz": vel[selected, 2],
            "stellar_mass_1e10Msun_h": stellar_mass[selected],
            "rank_value": stellar_mass[selected],
        }.items():
            handle.create_dataset(name, data=np.asarray(values, dtype="f4"), compression="gzip", shuffle=True)
        handle.create_dataset("is_sat", data=np.asarray(is_sat[selected], dtype="i1"), compression="gzip", shuffle=True)
        handle.create_dataset("host_halo_id", data=np.asarray(grnr[selected], dtype="i8"), compression="gzip", shuffle=True)
        handle.create_dataset("subhalo_id", data=np.asarray(subhalo_id[selected], dtype="i8"), compression="gzip", shuffle=True)
        handle.attrs["sim"] = args.sim
        handle.attrs["snapshot"] = args.snapshot
        handle.attrs["redshift"] = redshift
        handle.attrs["boxsize"] = boxsize
        handle.attrs["boxsize_units"] = "Mpc/h"
        handle.attrs["target_number_density_h3_mpc3"] = args.target_number_density
        handle.attrs["actual_number_density_h3_mpc3"] = actual_nbar
        handle.attrs["candidate_multiplier"] = args.candidate_multiplier

    summary = {
        "output": str(output_path),
        "sim": args.sim,
        "snapshot": args.snapshot,
        "redshift": redshift,
        "boxsize_mpc_h": boxsize,
        "n_candidates": int(len(candidate_indices)),
        "n_selected": int(len(selected)),
        "n_sat": int(np.count_nonzero(is_sat[selected])),
        "f_sat": float(np.count_nonzero(is_sat[selected]) / len(selected)),
        "target_number_density_h3_mpc3": args.target_number_density,
        "actual_number_density_h3_mpc3": actual_nbar,
        "stellar_mass_min_selected_1e10Msun_h": float(np.min(stellar_mass[selected])),
        "stellar_mass_max_selected_1e10Msun_h": float(np.max(stellar_mass[selected])),
        "candidate_multiplier": args.candidate_multiplier,
    }
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/tng300_pilot/raw")
    parser.add_argument("--sim", default="TNG300-3")
    parser.add_argument("--snapshot", type=int, default=59)
    parser.add_argument("--redshift", type=float, default=None)
    parser.add_argument("--boxsize", type=float, default=None)
    parser.add_argument("--target-number-density", type=float, default=5.31589034e-4)
    parser.add_argument("--target-count", type=int, default=0)
    parser.add_argument("--candidate-multiplier", type=float, default=1.0)
    parser.add_argument("--min-stellar-mass-1e10msun-h", type=float, default=0.0)
    parser.add_argument("--output", default="data/tng300_pilot/catalogs/TNG300-3_snap059_lrg_pilot.hdf5")
    args = parser.parse_args(argv)
    print(json.dumps(build_catalog(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
