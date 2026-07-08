"""Build a compact AbacusSummit HOD/SHAM pilot catalog from halo slabs.

This converts a local subset of AbacusSummit CompaSO ``halo_info`` plus
``halo_rv_A`` slabs into the compact HDF5 contract used by
``fit_hod_catalog_fsat_alpha.py``. Centrals are halo centers; satellite
candidates are drawn from each host halo's subsample-A particles, preserving
host halo id and host velocity so velocity-bias grids can scale internal
satellite velocities rather than the full bulk velocity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


def patch_windows_affinity() -> None:
    if not hasattr(os, "sched_getaffinity"):
        os.sched_getaffinity = lambda pid: set(range(os.cpu_count() or 1))  # type: ignore[attr-defined]


def parse_slabs(text: str) -> list[int]:
    slabs: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = [int(item) for item in chunk.split("-", 1)]
            slabs.extend(range(lo, hi + 1))
        else:
            slabs.append(int(chunk))
    return sorted(set(slabs))


def halo_files(zroot: Path, slabs: list[int]) -> list[Path]:
    files = [zroot / "halo_info" / f"halo_info_{slab:03d}.asdf" for slab in slabs]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing halo_info slabs: " + ", ".join(missing))
    missing_rv = [
        str(zroot / "halo_rv_A" / f"halo_rv_A_{slab:03d}.asdf")
        for slab in slabs
        if not (zroot / "halo_rv_A" / f"halo_rv_A_{slab:03d}.asdf").exists()
    ]
    if missing_rv:
        raise FileNotFoundError("Missing halo_rv_A slabs: " + ", ".join(missing_rv))
    return files


def inside_subbox(pos: np.ndarray, origin: np.ndarray, side: float) -> np.ndarray:
    return np.all((pos >= origin) & (pos < origin + side), axis=1)


def choose_particle_offsets(npart: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    if npart <= 0 or limit <= 0:
        return np.asarray([], dtype=np.int64)
    if npart <= limit:
        return np.arange(npart, dtype=np.int64)
    return np.sort(rng.choice(npart, size=limit, replace=False).astype(np.int64))


def append_rows(rows: dict[str, list[np.ndarray]], **columns: np.ndarray) -> None:
    for key, value in columns.items():
        rows.setdefault(key, []).append(np.asarray(value))


def concat_rows(rows: dict[str, list[np.ndarray]], key: str, dtype: str) -> np.ndarray:
    if key not in rows or not rows[key]:
        return np.asarray([], dtype=dtype)
    return np.concatenate(rows[key]).astype(dtype, copy=False)


def build_catalog(args: argparse.Namespace) -> dict[str, object]:
    patch_windows_affinity()
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    slabs = parse_slabs(args.slabs)
    zroot = Path(args.zroot)
    files = halo_files(zroot, slabs)
    origin = np.asarray(args.origin, dtype="f4")
    side = float(args.side)
    target_count = int(round(args.target_number_density * side**3))
    central_limit = int(round(target_count * args.candidate_multiplier))
    satellite_limit = int(round(target_count * args.candidate_multiplier))
    rng = np.random.default_rng(args.seed)

    def filter_func(halos):
        pos = np.asarray(halos["x_L2com"])
        n = np.asarray(halos["N"])
        return (n >= args.min_particles) & inside_subbox(pos, origin, side)

    cat = CompaSOHaloCatalog(
        [str(path) for path in files],
        fields=["id", "N", "x_L2com", "v_L2com", "npstartA", "npoutA"],
        cleaned=False,
        subsamples=dict(A=True, rv=True),
        filter_func=filter_func,
        verbose=args.verbose,
    )
    halos = cat.halos
    subsamples = cat.subsamples
    n_halo_loaded = len(halos)
    if n_halo_loaded == 0:
        raise ValueError("No halos passed the subbox/min-particles filter")

    hid = np.asarray(halos["id"], dtype="i8")
    n_particles = np.asarray(halos["N"], dtype="f8")
    hpos = np.asarray(halos["x_L2com"], dtype="f8")
    hvel = np.asarray(halos["v_L2com"], dtype="f8")
    npstart = np.asarray(halos["npstartA"], dtype=np.int64)
    npout = np.asarray(halos["npoutA"], dtype=np.int64)
    order = np.argsort(n_particles)[::-1]

    central_order = order[: min(central_limit, len(order))]
    rows: dict[str, list[np.ndarray]] = {}
    cen_pos = np.mod(hpos[central_order] - origin, side)
    cen_vel = hvel[central_order]
    append_rows(
        rows,
        x=cen_pos[:, 0],
        y=cen_pos[:, 1],
        z=cen_pos[:, 2],
        vx=cen_vel[:, 0],
        vy=cen_vel[:, 1],
        vz=cen_vel[:, 2],
        host_vx=cen_vel[:, 0],
        host_vy=cen_vel[:, 1],
        host_vz=cen_vel[:, 2],
        host_halo_id=hid[central_order],
        is_sat=np.zeros(len(central_order), dtype="i1"),
        rank_value=n_particles[central_order],
        halo_n=n_particles[central_order],
    )

    sat_pos_all = np.asarray(subsamples["pos"], dtype="f8")
    sat_vel_all = np.asarray(subsamples["vel"], dtype="f8")
    sat_chunks = 0
    for halo_index in order:
        if sat_chunks >= satellite_limit:
            break
        count = int(npout[halo_index])
        if count <= 0:
            continue
        take = min(args.max_sat_per_halo, count, satellite_limit - sat_chunks)
        offsets = choose_particle_offsets(count, take, rng)
        if len(offsets) == 0:
            continue
        idx = npstart[halo_index] + offsets
        spos = np.mod(sat_pos_all[idx] - origin, side)
        svel = sat_vel_all[idx]
        host_velocity = hvel[halo_index]
        append_rows(
            rows,
            x=spos[:, 0],
            y=spos[:, 1],
            z=spos[:, 2],
            vx=svel[:, 0],
            vy=svel[:, 1],
            vz=svel[:, 2],
            host_vx=np.full(len(idx), host_velocity[0]),
            host_vy=np.full(len(idx), host_velocity[1]),
            host_vz=np.full(len(idx), host_velocity[2]),
            host_halo_id=np.full(len(idx), hid[halo_index]),
            is_sat=np.ones(len(idx), dtype="i1"),
            rank_value=np.full(len(idx), n_particles[halo_index]),
            halo_n=np.full(len(idx), n_particles[halo_index]),
        )
        sat_chunks += len(idx)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x": concat_rows(rows, "x", "f4"),
        "y": concat_rows(rows, "y", "f4"),
        "z": concat_rows(rows, "z", "f4"),
        "vx": concat_rows(rows, "vx", "f4"),
        "vy": concat_rows(rows, "vy", "f4"),
        "vz": concat_rows(rows, "vz", "f4"),
        "host_vx": concat_rows(rows, "host_vx", "f4"),
        "host_vy": concat_rows(rows, "host_vy", "f4"),
        "host_vz": concat_rows(rows, "host_vz", "f4"),
        "host_halo_id": concat_rows(rows, "host_halo_id", "i8"),
        "is_sat": concat_rows(rows, "is_sat", "i1"),
        "rank_value": concat_rows(rows, "rank_value", "f4"),
        "halo_n": concat_rows(rows, "halo_n", "f4"),
    }
    with h5py.File(output, "w") as handle:
        for key, values in arrays.items():
            handle.create_dataset(key, data=values, compression="gzip", shuffle=True)
        handle.attrs["sim"] = args.sim_name
        handle.attrs["redshift"] = args.redshift
        handle.attrs["boxsize"] = side
        handle.attrs["boxsize_units"] = "Mpc/h"
        handle.attrs["subbox_origin"] = origin
        handle.attrs["slabs"] = ",".join(f"{slab:03d}" for slab in slabs)
        handle.attrs["target_number_density_h3_mpc3"] = args.target_number_density
        handle.attrs["candidate_multiplier"] = args.candidate_multiplier
        handle.attrs["min_particles"] = args.min_particles
        handle.attrs["max_sat_per_halo"] = args.max_sat_per_halo

    n_total = len(arrays["x"])
    n_sat = int(np.count_nonzero(arrays["is_sat"]))
    summary = {
        "output": str(output),
        "sim": args.sim_name,
        "redshift": args.redshift,
        "slabs": slabs,
        "subbox_origin": origin.astype(float).tolist(),
        "boxsize_mpc_h": side,
        "target_number_density_h3_mpc3": args.target_number_density,
        "target_count": target_count,
        "candidate_multiplier": args.candidate_multiplier,
        "n_halo_loaded": int(n_halo_loaded),
        "n_candidates": int(n_total),
        "n_central_candidates": int(n_total - n_sat),
        "n_satellite_candidates": int(n_sat),
        "candidate_f_sat": float(n_sat / n_total) if n_total else float("nan"),
        "min_particles": args.min_particles,
        "max_sat_per_halo": args.max_sat_per_halo,
    }
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zroot", default="data/abacus_summit/AbacusSummit_base_c000_ph000/halos/z0.800")
    parser.add_argument("--sim-name", default="AbacusSummit_base_c000_ph000")
    parser.add_argument("--redshift", type=float, default=0.8)
    parser.add_argument("--slabs", default="0")
    parser.add_argument("--origin", type=float, nargs=3, default=[-1000.0, -1000.0, -1000.0])
    parser.add_argument("--side", type=float, default=60.0)
    parser.add_argument("--target-number-density", type=float, default=5.31589034e-4)
    parser.add_argument("--candidate-multiplier", type=float, default=5.0)
    parser.add_argument("--min-particles", type=int, default=300)
    parser.add_argument("--max-sat-per-halo", type=int, default=5)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--output", default="data/abacus_summit/processed/abacus_z0p8_hod_pilot.hdf5")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build_catalog(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
