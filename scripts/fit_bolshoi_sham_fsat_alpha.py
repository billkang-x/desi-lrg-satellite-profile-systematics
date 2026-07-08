"""Shared periodic-box clustering utilities and Bolshoi SHAM pilot hooks.

This file provides the low-level readers and pair-count approximations used by
the HOD/TNG pilot scripts. It is intentionally lightweight: the paper-level
measurement still uses pycorr for survey catalogs, while these utilities are
for compact periodic-box pilot catalogs.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import h5py
import numpy as np


def read_observed_poles(path: Path, s_min: float, s_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True)
    mask = (table["s_Mpch"] >= s_min) & (table["s_Mpch"] <= s_max)
    return table["s_Mpch"][mask], table["xi0"][mask], table["xi2"][mask]


def read_observed_wp(path: Path, rp_min: float, rp_max: float) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True)
    mask = (table["rp_Mpch"] >= rp_min) & (table["rp_Mpch"] <= rp_max)
    return table["rp_Mpch"][mask], table["wp"][mask]


def read_observed_number_density(path: Path) -> float:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if "nbar_h3_mpc3" in payload:
        return float(payload["nbar_h3_mpc3"])
    if "nbar" in payload:
        return float(payload["nbar"])
    raise KeyError(f"No physical number density found in {path}")


def pair_separations_periodic(positions: np.ndarray, max_sep: float, boxsize: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(positions, boxsize=boxsize)
    pairs = tree.query_pairs(max_sep, output_type="ndarray")
    if pairs.size == 0:
        empty = np.asarray([], dtype="f8")
        return empty, empty, empty
    delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    delta -= boxsize * np.rint(delta / boxsize)
    rp = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
    pi = np.abs(delta[:, 2])
    sep = np.sqrt(rp**2 + pi**2)
    return sep, rp, pi


def xi_from_pairs(sep: np.ndarray, pi: np.ndarray, edges: np.ndarray, n: int, boxsize: float) -> tuple[np.ndarray, np.ndarray]:
    mu = pi / np.maximum(sep, 1e-30)
    bin_index = np.searchsorted(edges, sep, side="right") - 1
    valid = (bin_index >= 0) & (bin_index < len(edges) - 1)
    bin_index = bin_index[valid]
    mu = mu[valid]
    dd0 = np.bincount(bin_index, minlength=len(edges) - 1).astype("f8")
    l2 = 0.5 * (3.0 * mu**2 - 1.0)
    dd2 = np.bincount(bin_index, weights=l2, minlength=len(edges) - 1).astype("f8")
    shell_volume = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rr = 0.5 * n * (n - 1) * shell_volume / boxsize**3
    xi0 = dd0 / np.maximum(rr, 1e-30) - 1.0
    xi2 = 5.0 * dd2 / np.maximum(rr, 1e-30)
    return xi0, xi2


def wp_from_pairs(rp: np.ndarray, pi: np.ndarray, rp_edges: np.ndarray, pimax: float, n: int, boxsize: float) -> np.ndarray:
    valid = pi <= pimax
    rp = rp[valid]
    bin_index = np.searchsorted(rp_edges, rp, side="right") - 1
    valid = (bin_index >= 0) & (bin_index < len(rp_edges) - 1)
    bin_index = bin_index[valid]
    dd = np.bincount(bin_index, minlength=len(rp_edges) - 1).astype("f8")
    annulus = np.pi * (rp_edges[1:] ** 2 - rp_edges[:-1] ** 2)
    cylinder_volume = annulus * (2.0 * pimax)
    rr = 0.5 * n * (n - 1) * cylinder_volume / boxsize**3
    return 2.0 * pimax * (dd / np.maximum(rr, 1e-30) - 1.0)


def interpolate_model(s_centers: np.ndarray, xi0: np.ndarray, xi2: np.ndarray, obs_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.interp(obs_s, s_centers, xi0), np.interp(obs_s, s_centers, xi2)


def interpolate_wp_model(rp_centers: np.ndarray, wp: np.ndarray, obs_rp: np.ndarray) -> np.ndarray:
    return np.interp(obs_rp, rp_centers, wp)


def parse_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/halotools_bolshoi/hlist_0.67035.list.halotools_v0p4.hdf5")
    parser.add_argument("--output-dir", default="results/bolshoi_sham_constraints")
    args = parser.parse_args(argv)
    if not Path(args.catalog).exists():
        raise FileNotFoundError(args.catalog)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with h5py.File(args.catalog, "r") as handle:
        print(handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
