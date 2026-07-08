"""Production DESI DR1 LRG two-point measurement with cosmodesi pycorr.

This script is intended for ParaCloud/Linux/HPC production runs. It reads DESI
DR1 LRG clustering-ready catalogs, converts redshifts to comoving distances in a
flat LCDM fiducial cosmology, and measures:

- redshift-space multipoles from pycorr ``mode='smu'``
- projected clustering ``wp(rp)`` from pycorr ``mode='rppi'``

The default settings are a conservative first production target for the HOD/mock
paper: DESI DR1 LRG, 0.6 < z < 0.8, one random catalog per cap, WEIGHT only.
Increase random indices and remove sampling limits for publication runs.
"""

from __future__ import print_function

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


DESI_LSS_BASE_URL = (
    "https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/"
    "LSS/iron/LSScats/v1.5"
)
SPEED_OF_LIGHT_KM_S = 299792.458


def parse_csv(value):
    if value is None or value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value):
    items = []
    for item in parse_csv(value):
        if ".." in item:
            start, stop = item.split("..", 1)
            items.extend(range(int(start), int(stop) + 1))
        elif "-" in item and item.count("-") == 1 and not item.startswith("-"):
            start, stop = item.split("-", 1)
            items.extend(range(int(start), int(stop) + 1))
        else:
            items.append(int(item))
    return sorted(set(items))


def parse_float_list(value):
    return [float(item) for item in parse_csv(value)]


def parse_ell_list(value):
    return tuple(int(item) for item in parse_csv(value))


def make_linear_edges(vmin, vmax, nbins):
    return np.linspace(float(vmin), float(vmax), int(nbins) + 1)


def make_log_edges(vmin, vmax, nbins):
    return np.geomspace(float(vmin), float(vmax), int(nbins) + 1)


def flat_lcdm_comoving_distance_mpc_h(z, omega_m=0.315192, h=0.6736, ngrid=200000):
    """Comoving radial distance in Mpc/h using a dense trapezoidal grid.

    The calculation avoids scipy so the measurement script has only numpy plus a
    FITS reader as hard pre-pycorr dependencies. The output distance unit is
    Mpc/h, matching the bin definitions below.
    """
    z = np.asarray(z, dtype="f8")
    if z.size == 0:
        return z.copy()
    zmax = max(1.2, float(np.nanmax(z)) + 0.02)
    grid = np.linspace(0.0, zmax, int(ngrid))
    inv_e = 1.0 / np.sqrt(omega_m * (1.0 + grid) ** 3 + (1.0 - omega_m))
    dz = np.diff(grid)
    integral = np.empty_like(grid)
    integral[0] = 0.0
    integral[1:] = np.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * dz)
    distance_mpc = SPEED_OF_LIGHT_KM_S / (100.0 * h) * np.interp(z, grid, integral)
    return distance_mpc * h


def get_rank_info():
    try:
        from mpi4py import MPI
    except Exception:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def import_fits_backend():
    try:
        import fitsio

        return "fitsio", fitsio
    except Exception:
        pass
    try:
        from astropy.io import fits

        return "astropy", fits
    except Exception as exc:
        raise RuntimeError("Need either fitsio or astropy to read FITS catalogs") from exc


def read_columns(path, names):
    backend_name, backend = import_fits_backend()
    if backend_name == "fitsio":
        table = backend.read(str(path), columns=list(names), ext=1)
        return {name: np.asarray(table[name]) for name in names if name in table.dtype.names}

    with backend.open(str(path), memmap=True) as hdul:
        data = hdul[1].data
        available = set(data.names)
        return {name: np.asarray(data[name]) for name in names if name in available}


def existing_columns(path):
    backend_name, backend = import_fits_backend()
    if backend_name == "fitsio":
        with backend.FITS(str(path)) as hdul:
            return list(hdul[1].get_colnames())

    with backend.open(str(path), memmap=True) as hdul:
        return list(hdul[1].data.names)


def choose_indices(mask, max_rows, seed):
    indices = np.flatnonzero(mask)
    if max_rows and len(indices) > max_rows:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=int(max_rows), replace=False)
        indices.sort()
    return indices


def sky_region_samples(ra, dec, nra, ndec, ra_min=None, ra_max=None, dec_min=None, dec_max=None):
    """Assign simple RA/DEC grid jackknife samples.

    This intentionally uses a transparent grid rather than an adaptive scheme so
    the resulting covariance can be reproduced without extra dependencies.
    """
    ra = np.asarray(ra, dtype="f8")
    dec = np.asarray(dec, dtype="f8")
    if ra_min is None:
        ra_min = float(np.nanmin(ra))
    if ra_max is None:
        ra_max = float(np.nanmax(ra))
    if dec_min is None:
        dec_min = float(np.nanmin(dec))
    if dec_max is None:
        dec_max = float(np.nanmax(dec))
    ra_scale = max(float(ra_max) - float(ra_min), 1e-12)
    dec_scale = max(float(dec_max) - float(dec_min), 1e-12)
    ira = np.floor((ra - float(ra_min)) / ra_scale * int(nra)).astype("i8")
    idec = np.floor((dec - float(dec_min)) / dec_scale * int(ndec)).astype("i8")
    ira = np.clip(ira, 0, int(nra) - 1)
    idec = np.clip(idec, 0, int(ndec) - 1)
    return (idec * int(nra) + ira).astype("i4")


def sky_to_unit_vectors(ra, dec):
    ra_rad = np.radians(np.asarray(ra, dtype="f8"))
    dec_rad = np.radians(np.asarray(dec, dtype="f8"))
    cos_dec = np.cos(dec_rad)
    return np.column_stack([cos_dec * np.cos(ra_rad), cos_dec * np.sin(ra_rad), np.sin(dec_rad)])


def load_jackknife_centers(path):
    table = np.genfromtxt(str(path), delimiter=",", names=True, dtype=None, encoding=None)
    table = np.atleast_1d(table)
    if table.dtype.names is None:
        raise ValueError("Jackknife center file must be a CSV with named columns: {0}".format(path))
    names = set(table.dtype.names)
    if {"x", "y", "z"}.issubset(names):
        centers = np.column_stack(
            [
                np.asarray(table["x"], dtype="f8"),
                np.asarray(table["y"], dtype="f8"),
                np.asarray(table["z"], dtype="f8"),
            ]
        )
    elif {"ra_deg", "dec_deg"}.issubset(names):
        centers = sky_to_unit_vectors(table["ra_deg"], table["dec_deg"])
    elif {"RA", "DEC"}.issubset(names):
        centers = sky_to_unit_vectors(table["RA"], table["DEC"])
    else:
        raise ValueError(
            "Jackknife center file needs x/y/z or ra_deg/dec_deg columns: {0}".format(path)
        )
    norm = np.linalg.norm(centers, axis=1)
    if np.any(norm <= 0.0):
        raise ValueError("Jackknife center file contains a zero-length center: {0}".format(path))
    return centers / norm[:, None]


def center_region_samples(ra, dec, centers, chunk_size=200000):
    """Assign each sky position to the nearest precomputed unit-vector center."""
    centers = np.asarray(centers, dtype="f8")
    samples = np.empty(len(ra), dtype="i4")
    for start in range(0, len(samples), int(chunk_size)):
        stop = min(start + int(chunk_size), len(samples))
        points = sky_to_unit_vectors(ra[start:stop], dec[start:stop])
        samples[start:stop] = np.argmax(points.dot(centers.T), axis=1).astype("i4")
    return samples


def load_catalog(path, args, seed, max_rows=0):
    weight_columns = [] if args.no_weights else parse_csv(args.weight_columns)
    needed = ["RA", "DEC", "Z"] + weight_columns
    columns = existing_columns(path)
    missing = [name for name in needed if name not in columns]
    if missing:
        raise ValueError("Missing required columns in {0}: {1}".format(path, ", ".join(missing)))

    data = read_columns(path, needed)
    z = np.asarray(data["Z"], dtype="f8")
    weights = np.ones_like(z, dtype="f8")
    for name in weight_columns:
        weights *= np.asarray(data[name], dtype="f8")

    mask = (
        (z >= args.zmin)
        & (z < args.zmax)
        & np.isfinite(z)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    indices = choose_indices(mask, max_rows=max_rows, seed=seed)

    ra = np.asarray(data["RA"][indices], dtype="f8")
    dec = np.asarray(data["DEC"][indices], dtype="f8")
    zsel = z[indices]
    wsel = weights[indices]
    distance = flat_lcdm_comoving_distance_mpc_h(
        zsel,
        omega_m=args.omega_m,
        h=args.hubble_h,
        ngrid=args.distance_grid_size,
    )
    samples = None
    if args.jackknife_centers:
        samples = center_region_samples(ra, dec, load_jackknife_centers(args.jackknife_centers))
    elif args.jackknife_nra > 0 and args.jackknife_ndec > 0:
        samples = sky_region_samples(
            ra,
            dec,
            nra=args.jackknife_nra,
            ndec=args.jackknife_ndec,
            ra_min=args.jackknife_ra_min,
            ra_max=args.jackknife_ra_max,
            dec_min=args.jackknife_dec_min,
            dec_max=args.jackknife_dec_max,
        )
    if samples is not None:
        if args.jackknife_exclude_sample >= 0:
            keep = samples != int(args.jackknife_exclude_sample)
            ra = ra[keep]
            dec = dec[keep]
            zsel = zsel[keep]
            wsel = wsel[keep]
            distance = distance[keep]
            samples = None
    positions = np.vstack([ra, dec, distance])
    summary = {
        "path": str(path),
        "rows_after_cuts": int(len(zsel)),
        "weighted_sum": float(np.sum(wsel)),
        "z_min": float(np.min(zsel)) if len(zsel) else None,
        "z_max": float(np.max(zsel)) if len(zsel) else None,
        "z_median": float(np.median(zsel)) if len(zsel) else None,
    }
    return positions, wsel, samples, summary


def concatenate_catalogs(parts):
    positions = [item[0] for item in parts]
    weights = [item[1] for item in parts]
    samples = [item[2] for item in parts if item[2] is not None]
    summaries = [item[3] for item in parts]
    if not positions:
        raise ValueError("No catalogs were loaded")
    concatenated_samples = np.concatenate(samples) if samples else None
    return np.concatenate(positions, axis=1), np.concatenate(weights), concatenated_samples, summaries


def data_file(data_dir, tracer, cap):
    return Path(data_dir) / "{0}_{1}_clustering.dat.fits".format(tracer, cap)


def random_file(data_dir, tracer, cap, index):
    return Path(data_dir) / "{0}_{1}_{2}_clustering.ran.fits".format(tracer, cap, index)


def load_data_and_randoms(args, rank):
    tracer = args.tracer
    caps = parse_csv(args.caps)
    random_indices = parse_int_list(args.random_indices)
    if not random_indices:
        raise ValueError("--random-indices must specify at least one random catalog")

    data_parts = []
    random_parts = []
    for cap_index, cap in enumerate(caps):
        dpath = data_file(args.data_dir, tracer, cap)
        if not dpath.exists():
            raise FileNotFoundError(str(dpath))
        data_parts.append(
            load_catalog(
                dpath,
                args,
                seed=args.seed + 1000 * cap_index,
                max_rows=args.max_data_rows,
            )
        )

        for ridx in random_indices:
            rpath = random_file(args.data_dir, tracer, cap, ridx)
            if not rpath.exists():
                raise FileNotFoundError(str(rpath))
            random_parts.append(
                load_catalog(
                    rpath,
                    args,
                    seed=args.seed + 100000 + 1000 * cap_index + ridx,
                    max_rows=args.max_random_rows,
                )
            )

    data_positions, data_weights, data_samples, data_summary = concatenate_catalogs(data_parts)
    random_positions, random_weights, random_samples, random_summary = concatenate_catalogs(random_parts)
    if rank == 0:
        print("Loaded data objects: {0:,}".format(data_positions.shape[1]))
        print("Loaded random objects: {0:,}".format(random_positions.shape[1]))
        if data_samples is not None:
            print("Jackknife samples: {0}".format(len(np.unique(data_samples))))
    return data_positions, data_weights, data_samples, random_positions, random_weights, random_samples, data_summary, random_summary


def build_label(args):
    cap_label = "-".join(parse_csv(args.caps))
    random_label = "r" + "-".join(str(i) for i in parse_int_list(args.random_indices))
    return "{0}_DR1_{1}_z{2:.3f}-{3:.3f}_{4}".format(
        args.tracer,
        cap_label,
        args.zmin,
        args.zmax,
        random_label,
    )


def as_columns_for_poles(sep, corr, ells):
    sep = np.asarray(sep)
    corr = np.asarray(corr)
    if corr.ndim == 1:
        corr = corr.reshape(1, -1)
    if corr.shape[0] == len(ells):
        pole_rows = corr
    elif corr.shape[-1] == len(ells):
        pole_rows = corr.T
    else:
        raise ValueError("Cannot interpret pole array shape {0}".format(corr.shape))
    return np.column_stack([sep] + [pole_rows[i] for i in range(len(ells))])


def save_csv(path, array, header):
    np.savetxt(str(path), np.asarray(array), delimiter=",", header=header, comments="")


def split_corr_return(value):
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return value[0], value[1]
    raise ValueError("Unexpected pycorr return value with type {0}".format(type(value)))


def try_write_covariance(result, output_dir, stem, mode, rank, **kwargs):
    if rank != 0:
        return
    try:
        cov = result.cov(mode=mode, **kwargs)
    except Exception as exc:
        print("Could not compute jackknife covariance for {0}: {1}".format(stem, exc))
        return
    np.save(str(output_dir / (stem + "_cov.npy")), np.asarray(cov, dtype="f8"))
    print("Saved jackknife covariance: {0}".format(output_dir / (stem + "_cov.npy")))


def measure_smu(args, label, output_dir, data_pos, data_w, data_samples, random_pos, random_w, random_samples, mpicomm, rank):
    from pycorr import TwoPointCorrelationFunction

    s_edges = make_linear_edges(args.s_min, args.s_max, args.s_bins)
    mu_edges = make_linear_edges(-1.0, 1.0, args.mu_bins)
    result = TwoPointCorrelationFunction(
        "smu",
        (s_edges, mu_edges),
        data_positions1=data_pos,
        randoms_positions1=random_pos,
        data_weights1=data_w,
        randoms_weights1=random_w,
        data_samples1=data_samples,
        randoms_samples1=random_samples,
        position_type="rdd",
        estimator="landyszalay",
        los=args.los,
        engine=args.engine,
        nthreads=args.nthreads,
        dtype="f8",
        D1D2_weight_type="product_individual",
        D1R2_weight_type="product_individual",
        R1R2_weight_type="product_individual",
        mpicomm=mpicomm,
        mpiroot=0 if mpicomm is not None else None,
    )
    if rank == 0:
        result_path = output_dir / (label + "_smu.npy")
        result.save(str(result_path))
        ells = parse_ell_list(args.ells)
        sep, poles = split_corr_return(result.get_corr(return_sep=True, mode="poles", ells=ells))
        pole_table = as_columns_for_poles(sep, poles, ells)
        pole_header = "s_Mpch," + ",".join("xi{0}".format(ell) for ell in ells)
        save_csv(output_dir / (label + "_poles.csv"), pole_table, pole_header)
        result.save_txt(
            str(output_dir / (label + "_poles_pycorr.txt")),
            mode="poles",
            ells=ells,
            header=["DESI DR1 LRG pycorr multipoles", pole_header],
        )
        print("Saved smu result: {0}".format(result_path))
        if data_samples is not None:
            try_write_covariance(
                result,
                output_dir,
                label + "_poles",
                mode="poles",
                rank=rank,
                ells=parse_ell_list(args.ells),
            )


def measure_rppi(args, label, output_dir, data_pos, data_w, data_samples, random_pos, random_w, random_samples, mpicomm, rank):
    from pycorr import TwoPointCorrelationFunction

    if args.rp_spacing == "log":
        rp_edges = make_log_edges(args.rp_min, args.rp_max, args.rp_bins)
    else:
        rp_edges = make_linear_edges(args.rp_min, args.rp_max, args.rp_bins)
    pi_edges = make_linear_edges(-args.pi_max, args.pi_max, 2 * args.pi_bins)
    result = TwoPointCorrelationFunction(
        "rppi",
        (rp_edges, pi_edges),
        data_positions1=data_pos,
        randoms_positions1=random_pos,
        data_weights1=data_w,
        randoms_weights1=random_w,
        data_samples1=data_samples,
        randoms_samples1=random_samples,
        position_type="rdd",
        estimator="landyszalay",
        los=args.los,
        engine=args.engine,
        nthreads=args.nthreads,
        dtype="f8",
        D1D2_weight_type="product_individual",
        D1R2_weight_type="product_individual",
        R1R2_weight_type="product_individual",
        mpicomm=mpicomm,
        mpiroot=0 if mpicomm is not None else None,
    )
    if rank == 0:
        result_path = output_dir / (label + "_rppi.npy")
        result.save(str(result_path))
        rp, wp = split_corr_return(result.get_corr(return_sep=True, mode="wp", pimax=args.pi_max))
        save_csv(output_dir / (label + "_wp.csv"), np.column_stack([rp, wp]), "rp_Mpch,wp")
        result.save_txt(
            str(output_dir / (label + "_wp_pycorr.txt")),
            mode="wp",
            pimax=args.pi_max,
            header=["DESI DR1 LRG pycorr projected correlation", "rp_Mpch,wp"],
        )
        print("Saved rppi result: {0}".format(result_path))
        if data_samples is not None:
            try_write_covariance(
                result,
                output_dir,
                label + "_wp",
                mode="wp",
                rank=rank,
                pimax=args.pi_max,
            )


def write_metadata(args, label, output_dir, data_summary, random_summary, runtime_s, nranks):
    metadata = {
        "label": label,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": runtime_s,
        "nranks": nranks,
        "args": vars(args),
        "data_summary": data_summary,
        "random_summary": random_summary,
        "desi_lss_base_url": DESI_LSS_BASE_URL,
        "notes": [
            "Distances are flat-LCDM comoving distances in Mpc/h.",
            "Default weights use DESI catalog WEIGHT only.",
            "Use multiple random catalogs and no sampling limits for publication measurements.",
        ],
    }
    with (output_dir / (label + "_metadata.json")).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/dr1_lss_v1.5")
    parser.add_argument("--output-dir", default="results/pycorr_lrg")
    parser.add_argument("--tracer", default="LRG")
    parser.add_argument("--caps", default="NGC,SGC")
    parser.add_argument("--random-indices", default="0")
    parser.add_argument("--zmin", type=float, default=0.6)
    parser.add_argument("--zmax", type=float, default=0.8)
    parser.add_argument("--weight-columns", default="WEIGHT")
    parser.add_argument("--no-weights", action="store_true")
    parser.add_argument("--omega-m", type=float, default=0.315192)
    parser.add_argument("--hubble-h", type=float, default=0.6736)
    parser.add_argument("--distance-grid-size", type=int, default=200000)
    parser.add_argument("--engine", default="corrfunc")
    parser.add_argument("--nthreads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "1")))
    parser.add_argument("--los", default="midpoint")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-data-rows", type=int, default=0)
    parser.add_argument("--max-random-rows", type=int, default=0)
    parser.add_argument("--jackknife-nra", type=int, default=0)
    parser.add_argument("--jackknife-ndec", type=int, default=0)
    parser.add_argument("--jackknife-ra-min", type=float, default=0.0)
    parser.add_argument("--jackknife-ra-max", type=float, default=360.0)
    parser.add_argument("--jackknife-dec-min", type=float, default=-30.0)
    parser.add_argument("--jackknife-dec-max", type=float, default=90.0)
    parser.add_argument("--jackknife-centers", default="")
    parser.add_argument("--jackknife-exclude-sample", type=int, default=-1)
    parser.add_argument("--skip-smu", action="store_true")
    parser.add_argument("--skip-rppi", action="store_true")
    parser.add_argument("--s-min", type=float, default=2.0)
    parser.add_argument("--s-max", type=float, default=60.0)
    parser.add_argument("--s-bins", type=int, default=29)
    parser.add_argument("--mu-bins", type=int, default=120)
    parser.add_argument("--ells", default="0,2,4")
    parser.add_argument("--rp-min", type=float, default=0.3)
    parser.add_argument("--rp-max", type=float, default=30.0)
    parser.add_argument("--rp-bins", type=int, default=18)
    parser.add_argument("--rp-spacing", choices=["log", "linear"], default="log")
    parser.add_argument("--pi-max", type=float, default=80.0)
    parser.add_argument("--pi-bins", type=int, default=80)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.time()
    mpicomm, rank, nranks = get_rank_info()

    output_dir = Path(args.output_dir)
    label = build_label(args)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Measurement label: {0}".format(label))
        print("MPI ranks: {0}".format(nranks))
        print("Output directory: {0}".format(output_dir))

    if mpicomm is None or rank == 0:
        data_pos, data_w, data_samples, random_pos, random_w, random_samples, data_summary, random_summary = load_data_and_randoms(args, rank)
    else:
        data_pos = data_w = data_samples = random_pos = random_w = random_samples = None
        data_summary = random_summary = None

    if mpicomm is not None:
        mpicomm.Barrier()

    if not args.skip_smu:
        measure_smu(args, label, output_dir, data_pos, data_w, data_samples, random_pos, random_w, random_samples, mpicomm, rank)
    if not args.skip_rppi:
        measure_rppi(args, label, output_dir, data_pos, data_w, data_samples, random_pos, random_w, random_samples, mpicomm, rank)

    if rank == 0:
        write_metadata(
            args,
            label,
            output_dir,
            data_summary,
            random_summary,
            runtime_s=time.time() - started,
            nranks=nranks,
        )
        print("Done in {0:.1f} s".format(time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
