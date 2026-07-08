"""Fit f_sat and alpha_s using a large HOD/SHAM galaxy catalog.

The catalog must contain positions, velocities, and central/satellite
classification. For resource-limited pilots, ``--selection-mode
fixed-nbar-ranked`` chooses top-ranked centrals and satellites separately to
match the DESI number density at each requested satellite fraction. This mode
requires a ``rank_value`` dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import h5py
import numpy as np

from fit_bolshoi_sham_fsat_alpha import (
    interpolate_model,
    interpolate_wp_model,
    pair_separations_periodic,
    read_observed_number_density,
    read_observed_poles,
    read_observed_wp,
    wp_from_pairs,
    xi_from_pairs,
)


def dataset_to_array(handle: h5py.File, name: str):
    if name in handle:
        return np.asarray(handle[name])
    for group_name in ["data", "galaxies", "catalog"]:
        if group_name in handle and name in handle[group_name]:
            return np.asarray(handle[group_name][name])
    raise KeyError(name)


def optional_dataset_to_array(handle: h5py.File, name: str):
    try:
        return dataset_to_array(handle, name)
    except KeyError:
        return None


def read_observed_datavector(path: Path) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return np.asarray(table["value"], dtype="f8")


def load_covariance_precision(path: Path, rcond: float, precision_scale: float, mode: str) -> tuple[np.ndarray, dict[str, object]]:
    cov = np.load(path)
    if mode == "full":
        precision = np.linalg.pinv(cov, rcond=rcond) * precision_scale
    elif mode == "diagonal":
        precision = np.diag(1.0 / np.maximum(np.diag(cov), 1e-30)) * precision_scale
    else:
        raise ValueError(f"Unknown covariance mode: {mode}")
    eigvals = np.linalg.eigvalsh(cov)
    return precision, {
        "covariance": str(path),
        "covariance_shape": list(cov.shape),
        "covariance_mode": mode,
        "covariance_rcond": rcond,
        "precision_scale": precision_scale,
        "eig_min": float(np.min(eigvals)),
        "eig_max": float(np.max(eigvals)),
    }


def load_hod_catalog(path: Path, boxsize: float | None = None, redshift: float | None = None) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as handle:
        x = dataset_to_array(handle, "x")
        y = dataset_to_array(handle, "y")
        z = dataset_to_array(handle, "z")
        vx = dataset_to_array(handle, "vx")
        vy = dataset_to_array(handle, "vy")
        vz = dataset_to_array(handle, "vz")
        host_vx = optional_dataset_to_array(handle, "host_vx")
        host_vy = optional_dataset_to_array(handle, "host_vy")
        host_vz = optional_dataset_to_array(handle, "host_vz")
        if "is_sat" in handle or ("data" in handle and "is_sat" in handle["data"]):
            is_sat = dataset_to_array(handle, "is_sat").astype(bool)
        elif "Ncent" in handle.attrs:
            ncent = int(handle.attrs["Ncent"])
            is_sat = np.zeros(len(x), dtype=bool)
            is_sat[ncent:] = True
        elif "Ncent" in handle:
            ncent = int(np.asarray(handle["Ncent"])[()])
            is_sat = np.zeros(len(x), dtype=bool)
            is_sat[ncent:] = True
        else:
            raise KeyError("Need is_sat dataset or Ncent attribute/dataset")
        rank_value = optional_dataset_to_array(handle, "rank_value")
        attrs = dict(handle.attrs)
    output = {
        "x": np.asarray(x, dtype="f8"),
        "y": np.asarray(y, dtype="f8"),
        "z": np.asarray(z, dtype="f8"),
        "vx": np.asarray(vx, dtype="f8"),
        "vy": np.asarray(vy, dtype="f8"),
        "vz": np.asarray(vz, dtype="f8"),
        "is_sat": is_sat,
        "boxsize": float(boxsize if boxsize is not None else attrs.get("boxsize", attrs.get("Lbox", 2000.0))),
        "redshift": float(redshift if redshift is not None else attrs.get("redshift", 0.7)),
    }
    if rank_value is not None:
        output["rank_value"] = np.asarray(rank_value, dtype="f8")
    if host_vx is not None and host_vy is not None and host_vz is not None:
        output["host_vx"] = np.asarray(host_vx, dtype="f8")
        output["host_vy"] = np.asarray(host_vy, dtype="f8")
        output["host_vz"] = np.asarray(host_vz, dtype="f8")
    return output


def make_sample_indices(is_sat: np.ndarray, target_fsat: float, rng: np.random.Generator) -> np.ndarray:
    sat_idx = np.flatnonzero(is_sat)
    cen_idx = np.flatnonzero(~is_sat)
    target_nsat = int(round(target_fsat / max(1e-6, 1.0 - target_fsat) * len(cen_idx)))
    target_nsat = max(0, min(len(sat_idx), target_nsat))
    if target_nsat < len(sat_idx):
        sat_idx = rng.choice(sat_idx, size=target_nsat, replace=False)
    return np.sort(np.concatenate([cen_idx, sat_idx]))


def make_fixed_nbar_ranked_indices(is_sat: np.ndarray, rank_value: np.ndarray, target_fsat: float, target_count: int) -> np.ndarray:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    sat_idx = np.flatnonzero(is_sat)
    cen_idx = np.flatnonzero(~is_sat)
    n_sat = int(round(target_fsat * target_count))
    n_cen = target_count - n_sat
    if n_sat > len(sat_idx) or n_cen > len(cen_idx):
        raise ValueError(
            f"Not enough objects for f_sat={target_fsat:.3f}: need {n_cen} centrals/{n_sat} satellites, have {len(cen_idx)}/{len(sat_idx)}"
        )
    sat_order = sat_idx[np.argsort(rank_value[sat_idx])[::-1]][:n_sat]
    cen_order = cen_idx[np.argsort(rank_value[cen_idx])[::-1]][:n_cen]
    return np.sort(np.concatenate([cen_order, sat_order]))


def redshift_space_positions(cat: dict[str, np.ndarray | float], indices: np.ndarray, alpha_s: float, los_axis: int) -> np.ndarray:
    pos = np.column_stack([cat["x"][indices], cat["y"][indices], cat["z"][indices]]).astype("f8")
    vel = np.column_stack([cat["vx"][indices], cat["vy"][indices], cat["vz"][indices]]).astype("f8")
    is_sat = np.asarray(cat["is_sat"])[indices]
    if all(key in cat for key in ["host_vx", "host_vy", "host_vz"]):
        host_vel = np.column_stack([cat["host_vx"][indices], cat["host_vy"][indices], cat["host_vz"][indices]]).astype("f8")
        vel_los = vel[:, los_axis].copy()
        vel_los[is_sat] = host_vel[is_sat, los_axis] + alpha_s * (vel[is_sat, los_axis] - host_vel[is_sat, los_axis])
    else:
        vel_los = vel[:, los_axis].copy()
        vel_los[is_sat] *= alpha_s
    zred = float(cat["redshift"])
    omega_m = 0.315192
    ez = np.sqrt(omega_m * (1.0 + zred) ** 3 + (1.0 - omega_m))
    displacement = vel_los * (1.0 + zred) / (100.0 * ez)
    boxsize = float(cat["boxsize"])
    pos[:, los_axis] = np.mod(pos[:, los_axis] + displacement, boxsize)
    return pos


def evaluate_grid(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    obs_s, obs_xi0, obs_xi2 = read_observed_poles(Path(args.observed_poles), args.s_min, args.s_max)
    obs_rp, obs_wp = read_observed_wp(Path(args.observed_wp), args.rp_min, args.rp_max)
    obs_number_density = read_observed_number_density(Path(args.observed_number_density))
    covariance_info = None
    obs_vector = None
    precision = None
    if args.covariance:
        obs_vector = read_observed_datavector(Path(args.observed_datavector))
        precision, covariance_info = load_covariance_precision(Path(args.covariance), args.covariance_rcond, args.precision_scale, args.covariance_mode)
        if precision.shape[0] != len(obs_vector):
            raise ValueError(f"Covariance shape {precision.shape} does not match observed vector length {len(obs_vector)}")

    cat = load_hod_catalog(Path(args.catalog), boxsize=args.boxsize, redshift=args.redshift)
    is_sat = np.asarray(cat["is_sat"])
    if args.target_number_density is not None:
        obs_number_density = args.target_number_density
    target_count = int(round(obs_number_density * float(cat["boxsize"]) ** 3))
    rng = np.random.default_rng(args.seed)
    edges = np.linspace(args.s_min, args.s_max, args.s_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rp_edges = np.logspace(np.log10(args.rp_min), np.log10(args.rp_max), args.rp_bins + 1)
    rp_centers = np.sqrt(rp_edges[:-1] * rp_edges[1:])
    max_sep = max(args.s_max, np.sqrt(args.rp_max**2 + args.pimax**2))
    rows = []
    for f_sat, alpha_s in product(args.fsat_grid, args.alpha_s_grid):
        try:
            if args.selection_mode == "fixed-nbar-ranked":
                if "rank_value" not in cat:
                    raise KeyError("fixed-nbar-ranked selection requires rank_value")
                idx = make_fixed_nbar_ranked_indices(is_sat, np.asarray(cat["rank_value"]), f_sat, target_count)
            else:
                idx = make_sample_indices(is_sat, f_sat, np.random.default_rng(rng.integers(0, 2**32 - 1)))
        except Exception as exc:
            rows.append({"f_sat_target": f_sat, "alpha_s": alpha_s, "chi2_total": np.inf, "chi2_total_cov": np.inf, "status": str(exc)})
            continue
        pos = redshift_space_positions(cat, idx, alpha_s=alpha_s, los_axis=args.los_axis)
        sep, rp, pi = pair_separations_periodic(pos, max_sep, float(cat["boxsize"]))
        xi0, xi2 = xi_from_pairs(sep, pi, edges, len(pos), float(cat["boxsize"]))
        wp = wp_from_pairs(rp, pi, rp_edges, args.pimax, len(pos), float(cat["boxsize"]))
        model_xi0, model_xi2 = interpolate_model(centers, xi0, xi2, obs_s)
        model_wp = interpolate_wp_model(rp_centers, wp, obs_rp)
        sigma0 = np.maximum(np.abs(obs_xi0) * args.frac_err_xi0, args.floor_err_xi0)
        sigma2 = np.maximum(np.abs(obs_xi2) * args.frac_err_xi2, args.floor_err_xi2)
        sigmawp = np.maximum(np.abs(obs_wp) * args.frac_err_wp, args.floor_err_wp)
        chi2_xi0 = float(np.sum(((model_xi0 - obs_xi0) / sigma0) ** 2))
        chi2_xi2 = float(np.sum(((model_xi2 - obs_xi2) / sigma2) ** 2))
        chi2_wp = float(np.sum(((model_wp - obs_wp) / sigmawp) ** 2))
        model_number_density = float(len(idx) / float(cat["boxsize"]) ** 3)
        chi2_density = float((np.log(model_number_density / obs_number_density) / args.sigma_ln_density) ** 2)
        chi2_cov = np.nan
        chi2_total_cov = np.nan
        if precision is not None and obs_vector is not None:
            model_vector = np.concatenate([model_wp, model_xi0, model_xi2])
            residual = model_vector - obs_vector
            chi2_cov = float(residual @ precision @ residual)
            chi2_total_cov = chi2_cov + chi2_density
        rows.append(
            {
                "f_sat_target": f_sat,
                "alpha_s": alpha_s,
                "n_gal": int(len(idx)),
                "n_sat": int(np.count_nonzero(is_sat[idx])),
                "f_sat_actual": float(np.count_nonzero(is_sat[idx]) / len(idx)),
                "model_number_density_h3_mpc3": model_number_density,
                "observed_number_density_h3_mpc3": obs_number_density,
                "chi2_xi0": chi2_xi0,
                "chi2_xi2": chi2_xi2,
                "chi2_wp": chi2_wp,
                "chi2_density": chi2_density,
                "chi2_cov": chi2_cov,
                "chi2_total_cov": chi2_total_cov,
                "chi2_total": chi2_xi0 + chi2_xi2 + chi2_wp + chi2_density,
                "status": "ok",
            }
        )
    score_column = "chi2_total_cov" if precision is not None else "chi2_total"
    rows.sort(key=lambda row: row[score_column])
    metadata = {
        "catalog": str(args.catalog),
        "boxsize": float(cat["boxsize"]),
        "redshift": float(cat["redshift"]),
        "n_input": int(len(is_sat)),
        "input_f_sat": float(np.count_nonzero(is_sat) / len(is_sat)),
        "observed_number_density": obs_number_density,
        "selection_mode": args.selection_mode,
        "target_count": target_count,
        "score_column": score_column,
    }
    if covariance_info is not None:
        metadata["covariance_info"] = covariance_info
    return rows, metadata


def parse_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--observed-poles", default="results/pycorr_lrg_r0-3/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_poles.csv")
    parser.add_argument("--observed-wp", default="results/pycorr_lrg_r0-3/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp.csv")
    parser.add_argument("--observed-number-density", default="results/number_density/LRG_GCcomb_z0p6-0p8_number_density.json")
    parser.add_argument("--observed-datavector", default="results/datavectors/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp0.5-30_xi5-60_ells02_datavector.csv")
    parser.add_argument("--covariance", default=None)
    parser.add_argument("--covariance-mode", choices=["full", "diagonal"], default="full")
    parser.add_argument("--covariance-rcond", type=float, default=1e-6)
    parser.add_argument("--precision-scale", type=float, default=1.0)
    parser.add_argument("--target-number-density", type=float, default=None)
    parser.add_argument("--output-dir", default="results/hod_catalog_fsat_alpha")
    parser.add_argument("--boxsize", type=float, default=None)
    parser.add_argument("--redshift", type=float, default=None)
    parser.add_argument("--fsat-grid", type=parse_grid, default=parse_grid("0.08,0.12,0.16,0.20,0.24"))
    parser.add_argument("--alpha-s-grid", type=parse_grid, default=parse_grid("0.6,0.8,1.0,1.2,1.4"))
    parser.add_argument("--selection-mode", choices=["keep-centrals", "fixed-nbar-ranked"], default="keep-centrals")
    parser.add_argument("--s-min", type=float, default=5.0)
    parser.add_argument("--s-max", type=float, default=60.0)
    parser.add_argument("--s-bins", type=int, default=28)
    parser.add_argument("--rp-min", type=float, default=0.5)
    parser.add_argument("--rp-max", type=float, default=30.0)
    parser.add_argument("--rp-bins", type=int, default=16)
    parser.add_argument("--pimax", type=float, default=80.0)
    parser.add_argument("--los-axis", type=int, default=2)
    parser.add_argument("--frac-err-xi0", type=float, default=0.08)
    parser.add_argument("--frac-err-xi2", type=float, default=0.20)
    parser.add_argument("--frac-err-wp", type=float, default=0.10)
    parser.add_argument("--floor-err-xi0", type=float, default=0.006)
    parser.add_argument("--floor-err-xi2", type=float, default=0.025)
    parser.add_argument("--floor-err-wp", type=float, default=1.0)
    parser.add_argument("--sigma-ln-density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=24680)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata = evaluate_grid(args)
    csv_path = output_dir / "hod_catalog_fsat_alpha_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metadata["best"] = rows[:10]
    with (output_dir / "hod_catalog_fsat_alpha_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(csv_path)
    print("Best:", rows[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
