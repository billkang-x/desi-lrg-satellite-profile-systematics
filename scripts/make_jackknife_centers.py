"""Create reproducible sky jackknife centers for DESI LRG measurements.

The output centers are unit vectors on the sky.  They are intentionally simple
and dependency-light so the same center file can be reused on ParaCloud by
``measure_lrg_pycorr.py`` for data and random catalogs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def sky_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, dtype="f8"))
    dec = np.radians(np.asarray(dec_deg, dtype="f8"))
    cos_dec = np.cos(dec)
    return np.column_stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)])


def unit_to_sky(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = np.clip(vectors[:, 2], -1.0, 1.0)
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arcsin(z))
    return ra, dec


def read_positions(data_dir: Path, tracer: str, caps: list[str], zmin: float, zmax: float) -> tuple[np.ndarray, np.ndarray]:
    ra_parts = []
    dec_parts = []
    for cap in caps:
        path = data_dir / f"{tracer}_{cap}_clustering.dat.fits"
        with fits.open(path, memmap=True) as hdul:
            data = hdul[1].data
            z = np.asarray(data["Z"], dtype="f8")
            mask = (z >= zmin) & (z < zmax) & np.isfinite(z)
            ra_parts.append(np.asarray(data["RA"], dtype="f8")[mask])
            dec_parts.append(np.asarray(data["DEC"], dtype="f8")[mask])
    return np.concatenate(ra_parts), np.concatenate(dec_parts)


def assign(points: np.ndarray, centers: np.ndarray, chunk_size: int) -> np.ndarray:
    labels = np.empty(points.shape[0], dtype="i4")
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        labels[start:stop] = np.argmax(points[start:stop].dot(centers.T), axis=1)
    return labels


def recompute_centers(points: np.ndarray, labels: np.ndarray, nregions: int, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((nregions, 3), dtype="f8")
    counts = np.bincount(labels, minlength=nregions).astype("i8")
    np.add.at(sums, labels, points)
    empty = np.flatnonzero(counts == 0)
    if len(empty):
        nearest_score = np.max(points.dot(centers.T), axis=1)
        replacement_order = np.argsort(nearest_score)
        used = set()
        for region, point_index in zip(empty, replacement_order):
            while int(point_index) in used:
                replacement_order = replacement_order[1:]
                point_index = replacement_order[0]
            used.add(int(point_index))
            sums[region] = points[point_index]
            counts[region] = 1
    norm = np.linalg.norm(sums, axis=1)
    if np.any(norm == 0.0):
        raise RuntimeError("Failed to initialize all jackknife centers")
    return sums / norm[:, None], counts


def spherical_kmeans(points: np.ndarray, nregions: int, seed: int, max_iterations: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if points.shape[0] < nregions:
        raise ValueError("Need at least as many points as requested regions")
    centers = points[rng.choice(points.shape[0], size=nregions, replace=False)].copy()
    counts = np.zeros(nregions, dtype="i8")
    previous_labels = None
    for iteration in range(max_iterations):
        labels = assign(points, centers, chunk_size)
        centers, counts = recompute_centers(points, labels, nregions, centers)
        changed = points.shape[0] if previous_labels is None else int(np.count_nonzero(labels != previous_labels))
        print(f"iteration={iteration + 1} changed={changed} min_count={counts.min()} median_count={np.median(counts):.1f}")
        if previous_labels is not None and changed == 0:
            break
        previous_labels = labels
    labels = assign(points, centers, chunk_size)
    centers, counts = recompute_centers(points, labels, nregions, centers)
    return centers, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dr1_lss_v1.5"))
    parser.add_argument("--tracer", default="LRG")
    parser.add_argument("--caps", default="NGC,SGC")
    parser.add_argument("--zmin", type=float, default=0.6)
    parser.add_argument("--zmax", type=float, default=0.8)
    parser.add_argument("--nregions", type=int, default=96)
    parser.add_argument("--sample-size", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--output", type=Path, default=Path("data/jackknife/LRG_DR1_z0p6-0p8_kmeans96_centers.csv"))
    args = parser.parse_args(argv)

    ra, dec = read_positions(args.data_dir, args.tracer, parse_csv(args.caps), args.zmin, args.zmax)
    points_all = sky_to_unit(ra, dec)
    rng = np.random.default_rng(args.seed)
    if args.sample_size and points_all.shape[0] > args.sample_size:
        indices = rng.choice(points_all.shape[0], size=args.sample_size, replace=False)
        points_fit = points_all[indices]
    else:
        points_fit = points_all

    centers, _ = spherical_kmeans(
        points_fit,
        nregions=args.nregions,
        seed=args.seed,
        max_iterations=args.max_iterations,
        chunk_size=args.chunk_size,
    )
    labels_all = assign(points_all, centers, args.chunk_size)
    counts = np.bincount(labels_all, minlength=args.nregions).astype("i8")
    ra_centers, dec_centers = unit_to_sky(centers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack(
        [
            np.arange(args.nregions, dtype="i8"),
            ra_centers,
            dec_centers,
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            counts,
        ]
    )
    np.savetxt(
        args.output,
        rows,
        delimiter=",",
        header="region,ra_deg,dec_deg,x,y,z,n_data",
        comments="",
        fmt=["%d", "%.10f", "%.10f", "%.12e", "%.12e", "%.12e", "%d"],
    )
    summary = {
        "data_dir": str(args.data_dir),
        "tracer": args.tracer,
        "caps": parse_csv(args.caps),
        "zmin": args.zmin,
        "zmax": args.zmax,
        "nregions": args.nregions,
        "n_data_total": int(points_all.shape[0]),
        "sample_size": int(points_fit.shape[0]),
        "seed": args.seed,
        "counts": {
            "min": int(counts.min()),
            "median": float(np.median(counts)),
            "max": int(counts.max()),
            "nonempty": int(np.count_nonzero(counts)),
        },
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(args.output)
    print(summary_path)
    print(json.dumps(summary["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
