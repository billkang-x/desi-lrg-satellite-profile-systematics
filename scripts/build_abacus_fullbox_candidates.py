"""Build compact full-box AbacusSummit HOD/SHAM candidates.

The full AbacusSummit z=0.8 halo catalog is too large to load with all
subsamples at once. This script streams the 34 halo slabs twice:

1. read halo_info only and keep top-ranked central candidates plus top host
   halos for satellites;
2. read halo_rv_A only for those top satellite hosts and draw a compact
   satellite reservoir.

The output is both HDF5 for local inspection and NPZ for ParaCloud pycorr runs,
where h5py may be unavailable in the pycorr virtualenv.
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


def halo_info_path(zroot: Path, slab: int) -> Path:
    return zroot / "halo_info" / f"halo_info_{slab:03d}.asdf"


def halo_rv_path(zroot: Path, slab: int) -> Path:
    return zroot / "halo_rv_A" / f"halo_rv_A_{slab:03d}.asdf"


def require_slabs(zroot: Path, slabs: list[int]) -> None:
    missing = []
    for slab in slabs:
        for path in [halo_info_path(zroot, slab), halo_rv_path(zroot, slab)]:
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing Abacus slab files:\n" + "\n".join(missing))


def wrap_positions(pos: np.ndarray, boxsize: float) -> np.ndarray:
    return np.mod(pos + 0.5 * boxsize, boxsize).astype("f4")


def empty_store() -> dict[str, np.ndarray]:
    return {}


def update_top(store: dict[str, np.ndarray], new: dict[str, np.ndarray], limit: int) -> dict[str, np.ndarray]:
    if limit <= 0 or len(new["rank_value"]) == 0:
        return store
    if not store:
        combined = new
    else:
        combined = {key: np.concatenate([store[key], new[key]]) for key in new}
    n = len(combined["rank_value"])
    if n > limit:
        keep = np.argpartition(combined["rank_value"], n - limit)[n - limit :]
        order = keep[np.argsort(combined["rank_value"][keep])[::-1]]
    else:
        order = np.argsort(combined["rank_value"])[::-1]
    return {key: value[order] for key, value in combined.items()}


def store_from_halos(halos, boxsize: float) -> dict[str, np.ndarray]:
    pos = wrap_positions(np.asarray(halos["x_L2com"], dtype="f4"), boxsize)
    vel = np.asarray(halos["v_L2com"], dtype="f4")
    rank = np.asarray(halos["N"], dtype="f4")
    return {
        "x": pos[:, 0],
        "y": pos[:, 1],
        "z": pos[:, 2],
        "vx": vel[:, 0],
        "vy": vel[:, 1],
        "vz": vel[:, 2],
        "host_vx": vel[:, 0],
        "host_vy": vel[:, 1],
        "host_vz": vel[:, 2],
        "host_halo_id": np.asarray(halos["id"], dtype="i8"),
        "rank_value": rank,
        "halo_n": rank,
    }


def draw_satellites(halos, subsamples, boxsize: float, max_sat_per_halo: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if len(halos) == 0:
        return empty_store()
    starts = np.asarray(halos["npstartA"], dtype=np.int64)
    counts = np.asarray(halos["npoutA"], dtype=np.int64)
    host_ids = np.asarray(halos["id"], dtype="i8")
    host_rank = np.asarray(halos["N"], dtype="f4")
    host_vel = np.asarray(halos["v_L2com"], dtype="f4")
    pos_all = np.asarray(subsamples["pos"], dtype="f4")
    vel_all = np.asarray(subsamples["vel"], dtype="f4")
    chunks: dict[str, list[np.ndarray]] = {
        "x": [],
        "y": [],
        "z": [],
        "vx": [],
        "vy": [],
        "vz": [],
        "host_vx": [],
        "host_vy": [],
        "host_vz": [],
        "host_halo_id": [],
        "rank_value": [],
        "halo_n": [],
    }
    order = np.argsort(host_rank)[::-1]
    for ih in order:
        count = int(counts[ih])
        if count <= 0:
            continue
        take = min(int(max_sat_per_halo), count)
        if count <= take:
            offsets = np.arange(count, dtype=np.int64)
        else:
            offsets = np.sort(rng.choice(count, size=take, replace=False).astype(np.int64))
        idx = starts[ih] + offsets
        pos = wrap_positions(pos_all[idx], boxsize)
        vel = vel_all[idx]
        n = len(idx)
        chunks["x"].append(pos[:, 0])
        chunks["y"].append(pos[:, 1])
        chunks["z"].append(pos[:, 2])
        chunks["vx"].append(vel[:, 0])
        chunks["vy"].append(vel[:, 1])
        chunks["vz"].append(vel[:, 2])
        chunks["host_vx"].append(np.full(n, host_vel[ih, 0], dtype="f4"))
        chunks["host_vy"].append(np.full(n, host_vel[ih, 1], dtype="f4"))
        chunks["host_vz"].append(np.full(n, host_vel[ih, 2], dtype="f4"))
        chunks["host_halo_id"].append(np.full(n, host_ids[ih], dtype="i8"))
        chunks["rank_value"].append(np.full(n, host_rank[ih], dtype="f4"))
        chunks["halo_n"].append(np.full(n, host_rank[ih], dtype="f4"))
    return {
        key: np.concatenate(value) if value else np.asarray([], dtype=("i8" if key == "host_halo_id" else "f4"))
        for key, value in chunks.items()
    }


def load_halo_catalog():
    patch_windows_affinity()
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    return CompaSOHaloCatalog


def build(args: argparse.Namespace) -> dict[str, object]:
    CompaSOHaloCatalog = load_halo_catalog()
    zroot = Path(args.zroot)
    slabs = parse_slabs(args.slabs)
    require_slabs(zroot, slabs)
    boxsize = float(args.boxsize)
    target_count = int(round(args.target_number_density * boxsize**3))
    central_limit = int(args.central_candidate_count or target_count)
    max_sat_grid = max(float(item) for item in args.fsat_grid.split(",") if item.strip())
    min_satellite_limit = int(np.ceil(max_sat_grid * target_count))
    satellite_limit = int(args.satellite_candidate_count or max(min_satellite_limit * 2, 200000))
    satellite_host_limit = int(np.ceil(satellite_limit / max(1, args.max_sat_per_halo)))
    rng = np.random.default_rng(args.seed)

    central_store: dict[str, np.ndarray] = {}
    sat_host_store: dict[str, np.ndarray] = {}
    slab_summaries = []

    for slab in slabs:
        path = halo_info_path(zroot, slab)
        cat = CompaSOHaloCatalog(
            str(path),
            fields=["id", "N", "x_L2com", "v_L2com", "npstartA", "npoutA"],
            cleaned=False,
            subsamples=False,
            filter_func=lambda h: np.asarray(h["N"]) >= args.min_particles,
            verbose=args.verbose,
        )
        halos = cat.halos
        store = store_from_halos(halos, boxsize)
        central_store = update_top(central_store, store, central_limit)
        sat_host_store = update_top(sat_host_store, store, satellite_host_limit)
        slab_summaries.append({"slab": slab, "n_halos_min_particles": int(len(halos))})
        if args.verbose:
            print(f"pass1 slab={slab:03d} halos={len(halos)} central_top={len(central_store['rank_value'])} sat_hosts={len(sat_host_store['rank_value'])}")

    selected_sat_ids = set(int(item) for item in sat_host_store["host_halo_id"])
    satellite_store: dict[str, np.ndarray] = {}
    for slab in slabs:
        path = halo_info_path(zroot, slab)

        def sat_filter(h):
            ids = np.asarray(h["id"], dtype="i8")
            n = np.asarray(h["N"])
            mask_id = np.fromiter((int(item) in selected_sat_ids for item in ids), dtype=bool, count=len(ids))
            return mask_id & (n >= args.min_particles)

        cat = CompaSOHaloCatalog(
            str(path),
            fields=["id", "N", "x_L2com", "v_L2com", "npstartA", "npoutA"],
            cleaned=False,
            subsamples=dict(A=True, rv=True),
            filter_func=sat_filter,
            verbose=args.verbose,
        )
        new_sat = draw_satellites(cat.halos, cat.subsamples, boxsize, args.max_sat_per_halo, rng)
        satellite_store = update_top(satellite_store, new_sat, satellite_limit)
        if args.verbose:
            print(f"pass2 slab={slab:03d} sat_hosts={len(cat.halos)} sat_top={len(satellite_store.get('rank_value', []))}")

    arrays: dict[str, np.ndarray] = {}
    for key in ["x", "y", "z", "vx", "vy", "vz", "host_vx", "host_vy", "host_vz", "host_halo_id", "rank_value", "halo_n"]:
        arrays[key] = np.concatenate([central_store[key], satellite_store[key]])
    is_sat = np.concatenate(
        [
            np.zeros(len(central_store["rank_value"]), dtype="i1"),
            np.ones(len(satellite_store["rank_value"]), dtype="i1"),
        ]
    )
    arrays["is_sat"] = is_sat
    order = np.lexsort((-arrays["rank_value"], arrays["is_sat"]))
    for key in arrays:
        arrays[key] = arrays[key][order]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        for key, values in arrays.items():
            handle.create_dataset(key, data=values, compression="gzip", shuffle=True)
        handle.attrs["sim"] = args.sim_name
        handle.attrs["redshift"] = args.redshift
        handle.attrs["boxsize"] = boxsize
        handle.attrs["boxsize_units"] = "Mpc/h"
        handle.attrs["target_number_density_h3_mpc3"] = args.target_number_density
        handle.attrs["slabs"] = ",".join(f"{slab:03d}" for slab in slabs)
        handle.attrs["min_particles"] = args.min_particles
        handle.attrs["max_sat_per_halo"] = args.max_sat_per_halo

    npz_output = Path(args.npz_output) if args.npz_output else output.with_suffix(".npz")
    npz_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_output,
        **arrays,
        boxsize=np.asarray(boxsize, dtype="f8"),
        redshift=np.asarray(args.redshift, dtype="f8"),
        target_number_density=np.asarray(args.target_number_density, dtype="f8"),
    )

    n_sat = int(np.count_nonzero(arrays["is_sat"]))
    summary = {
        "output": str(output),
        "npz_output": str(npz_output),
        "sim": args.sim_name,
        "redshift": args.redshift,
        "boxsize_mpc_h": boxsize,
        "target_number_density_h3_mpc3": args.target_number_density,
        "target_count": target_count,
        "central_candidate_count": int(np.count_nonzero(arrays["is_sat"] == 0)),
        "satellite_candidate_count": n_sat,
        "n_candidates": int(len(arrays["is_sat"])),
        "candidate_f_sat": float(n_sat / len(arrays["is_sat"])),
        "min_required_satellites_for_grid": min_satellite_limit,
        "slabs": slabs,
        "slab_summaries": slab_summaries,
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
    parser.add_argument("--boxsize", type=float, default=2000.0)
    parser.add_argument("--slabs", default="0-33")
    parser.add_argument("--target-number-density", type=float, default=5.31589034e-4)
    parser.add_argument("--fsat-grid", default="0.015,0.020,0.025,0.030,0.035")
    parser.add_argument("--central-candidate-count", type=int, default=0)
    parser.add_argument("--satellite-candidate-count", type=int, default=250000)
    parser.add_argument("--min-particles", type=int, default=300)
    parser.add_argument("--max-sat-per-halo", type=int, default=3)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--output", default="data/abacus_summit/processed/abacus_z0p8_fullbox_lrg_candidates.hdf5")
    parser.add_argument("--npz-output", default="data/abacus_summit/processed/abacus_z0p8_fullbox_lrg_candidates.npz")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
