"""Fit f_sat and alpha_s for a periodic Abacus full-box catalog with pycorr."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from itertools import product
from pathlib import Path

import numpy as np


def parse_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def read_number_density(path: Path) -> float:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return float(payload.get("nbar_h3_mpc3", payload.get("nbar")))


def read_observed_poles(path: Path, s_min: float, s_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True)
    mask = (table["s_Mpch"] >= s_min) & (table["s_Mpch"] <= s_max)
    return table["s_Mpch"][mask], table["xi0"][mask], table["xi2"][mask]


def read_observed_wp(path: Path, rp_min: float, rp_max: float) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True)
    mask = (table["rp_Mpch"] >= rp_min) & (table["rp_Mpch"] <= rp_max)
    return table["rp_Mpch"][mask], table["wp"][mask]


def read_datavector(path: Path) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return np.asarray(table["value"], dtype="f8")


def load_precision(path: Path, mode: str, rcond: float, scale: float) -> tuple[np.ndarray, dict[str, object]]:
    cov = np.load(path)
    if mode == "full":
        precision = np.linalg.pinv(cov, rcond=rcond) * scale
    elif mode == "diagonal":
        precision = np.diag(1.0 / np.maximum(np.diag(cov), 1e-30)) * scale
    else:
        raise ValueError(mode)
    eigvals = np.linalg.eigvalsh(cov)
    return precision, {
        "covariance": str(path),
        "covariance_mode": mode,
        "covariance_shape": list(cov.shape),
        "covariance_rcond": rcond,
        "precision_scale": scale,
        "eig_min": float(np.min(eigvals)),
        "eig_max": float(np.max(eigvals)),
    }


def load_catalog(path: Path) -> dict[str, np.ndarray | float]:
    data = np.load(path)
    cat = {key: data[key] for key in data.files if key not in {"boxsize", "redshift", "target_number_density"}}
    cat["boxsize"] = float(np.asarray(data["boxsize"]))
    cat["redshift"] = float(np.asarray(data["redshift"]))
    return cat


def make_indices(cat: dict[str, np.ndarray | float], f_sat: float, target_count: int) -> np.ndarray:
    is_sat = np.asarray(cat["is_sat"], dtype=bool)
    rank = np.asarray(cat["rank_value"], dtype="f8")
    sat_idx = np.flatnonzero(is_sat)
    cen_idx = np.flatnonzero(~is_sat)
    n_sat = int(round(f_sat * target_count))
    n_cen = target_count - n_sat
    if n_cen > len(cen_idx) or n_sat > len(sat_idx):
        raise ValueError(f"Need {n_cen} centrals and {n_sat} satellites, have {len(cen_idx)} and {len(sat_idx)}")
    cen = cen_idx[np.argpartition(rank[cen_idx], len(cen_idx) - n_cen)[len(cen_idx) - n_cen :]]
    sat = sat_idx[np.argpartition(rank[sat_idx], len(sat_idx) - n_sat)[len(sat_idx) - n_sat :]] if n_sat else np.asarray([], dtype=cen.dtype)
    return np.sort(np.concatenate([cen, sat]))


def redshift_space_positions(cat: dict[str, np.ndarray | float], idx: np.ndarray, alpha_s: float, los_axis: int) -> np.ndarray:
    pos = np.column_stack([cat["x"][idx], cat["y"][idx], cat["z"][idx]]).astype("f8")
    vel = np.column_stack([cat["vx"][idx], cat["vy"][idx], cat["vz"][idx]]).astype("f8")
    host_vel = np.column_stack([cat["host_vx"][idx], cat["host_vy"][idx], cat["host_vz"][idx]]).astype("f8")
    is_sat = np.asarray(cat["is_sat"], dtype=bool)[idx]
    vel_los = vel[:, los_axis].copy()
    vel_los[is_sat] = host_vel[is_sat, los_axis] + alpha_s * (vel[is_sat, los_axis] - host_vel[is_sat, los_axis])
    zred = float(cat["redshift"])
    omega_m = 0.315192
    ez = np.sqrt(omega_m * (1.0 + zred) ** 3 + (1.0 - omega_m))
    displacement = vel_los * (1.0 + zred) / (100.0 * ez)
    boxsize = float(cat["boxsize"])
    pos[:, los_axis] = np.mod(pos[:, los_axis] + displacement, boxsize)
    return pos


def measure_pycorr(pos: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from pycorr import TwoPointCorrelationFunction

    boxsize = float(args.boxsize)
    s_edges = np.linspace(args.s_min, args.s_max, args.s_bins + 1)
    mu_edges = np.linspace(-1.0, 1.0, args.mu_bins + 1)
    rppi_edges = (
        np.geomspace(args.rp_min, args.rp_max, args.rp_bins + 1),
        np.linspace(-args.pimax, args.pimax, args.pi_bins + 1),
    )
    data_positions = [pos[:, 0], pos[:, 1], pos[:, 2]]
    result_smu = TwoPointCorrelationFunction(
        "smu",
        edges=(s_edges, mu_edges),
        data_positions1=data_positions,
        boxsize=boxsize,
        los="z",
        nthreads=args.nthreads,
        engine=args.engine,
    )
    s, poles = result_smu(ells=(0, 2), return_sep=True)
    xi0 = np.asarray(poles[0], dtype="f8")
    xi2 = np.asarray(poles[1], dtype="f8")
    result_rppi = TwoPointCorrelationFunction(
        "rppi",
        edges=rppi_edges,
        data_positions1=data_positions,
        boxsize=boxsize,
        los="z",
        nthreads=args.nthreads,
        engine=args.engine,
    )
    rppi_result = result_rppi(return_sep=True)
    if len(rppi_result) != 3:
        raise ValueError(f"Expected pycorr rppi to return (rp, pi, xi), got {len(rppi_result)} items")
    rp, pi, xi_rppi = rppi_result
    xi_rppi = np.nan_to_num(np.asarray(xi_rppi, dtype="f8"), nan=0.0, posinf=0.0, neginf=0.0)
    pi_widths = np.diff(rppi_edges[1])
    if xi_rppi.shape == (len(rp), len(pi_widths)):
        wp = np.sum(xi_rppi * pi_widths[None, :], axis=1)
    elif xi_rppi.shape == (len(pi_widths), len(rp)):
        wp = np.sum(xi_rppi * pi_widths[:, None], axis=0)
    else:
        raise ValueError(f"Unexpected xi_rppi shape {xi_rppi.shape} for rp={len(rp)}, pi={len(pi_widths)}")
    return np.asarray(s, dtype="f8"), xi0, xi2, np.asarray(rp, dtype="f8"), wp


def interp_model(x: np.ndarray, y: np.ndarray, xobs: np.ndarray) -> np.ndarray:
    return np.interp(xobs, x, y)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-npz", required=True)
    parser.add_argument("--output-dir", default="results/abacus_fullbox_fsat_alpha")
    parser.add_argument("--observed-poles", default="results/pycorr_lrg_r0-3/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_poles.csv")
    parser.add_argument("--observed-wp", default="results/pycorr_lrg_r0-3/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp.csv")
    parser.add_argument("--observed-number-density", default="results/number_density/LRG_GCcomb_z0p6-0p8_number_density.json")
    parser.add_argument("--observed-datavector", default="results/datavectors/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp0.5-30_xi5-60_ells02_datavector.csv")
    parser.add_argument("--covariance", default=None)
    parser.add_argument("--covariance-mode", choices=["full", "diagonal"], default="full")
    parser.add_argument("--covariance-rcond", type=float, default=1e-6)
    parser.add_argument("--precision-scale", type=float, default=1.0)
    parser.add_argument("--fsat-grid", default="0.015,0.020,0.025,0.030,0.035")
    parser.add_argument("--alpha-s-grid", default="0.7,0.8,0.9,1.0,1.1")
    parser.add_argument("--boxsize", type=float, default=2000.0)
    parser.add_argument("--s-min", type=float, default=5.0)
    parser.add_argument("--s-max", type=float, default=60.0)
    parser.add_argument("--s-bins", type=int, default=28)
    parser.add_argument("--mu-bins", type=int, default=120)
    parser.add_argument("--rp-min", type=float, default=0.5)
    parser.add_argument("--rp-max", type=float, default=30.0)
    parser.add_argument("--rp-bins", type=int, default=16)
    parser.add_argument("--pimax", type=float, default=80.0)
    parser.add_argument("--pi-bins", type=int, default=160)
    parser.add_argument("--los-axis", type=int, default=2)
    parser.add_argument("--engine", default="corrfunc")
    parser.add_argument("--nthreads", type=int, default=32)
    parser.add_argument("--frac-err-xi0", type=float, default=0.08)
    parser.add_argument("--frac-err-xi2", type=float, default=0.20)
    parser.add_argument("--frac-err-wp", type=float, default=0.10)
    parser.add_argument("--floor-err-xi0", type=float, default=0.006)
    parser.add_argument("--floor-err-xi2", type=float, default=0.025)
    parser.add_argument("--floor-err-wp", type=float, default=1.0)
    parser.add_argument("--sigma-ln-density", type=float, default=0.15)
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    obs_s, obs_xi0, obs_xi2 = read_observed_poles(Path(args.observed_poles), args.s_min, args.s_max)
    obs_rp, obs_wp = read_observed_wp(Path(args.observed_wp), args.rp_min, args.rp_max)
    obs_nbar = read_number_density(Path(args.observed_number_density))
    precision = None
    obs_vector = None
    covariance_info = None
    if args.covariance:
        obs_vector = read_datavector(Path(args.observed_datavector))
        precision, covariance_info = load_precision(Path(args.covariance), args.covariance_mode, args.covariance_rcond, args.precision_scale)
    cat = load_catalog(Path(args.catalog_npz))
    target_count = int(round(obs_nbar * float(cat["boxsize"]) ** 3))
    rows = []
    for f_sat, alpha_s in product(parse_grid(args.fsat_grid), parse_grid(args.alpha_s_grid)):
        try:
            idx = make_indices(cat, f_sat, target_count)
            pos = redshift_space_positions(cat, idx, alpha_s, args.los_axis)
            s, xi0, xi2, rp, wp = measure_pycorr(pos, args)
            model_xi0 = interp_model(s, xi0, obs_s)
            model_xi2 = interp_model(s, xi2, obs_s)
            model_wp = interp_model(rp, wp, obs_rp)
            sigma0 = np.maximum(np.abs(obs_xi0) * args.frac_err_xi0, args.floor_err_xi0)
            sigma2 = np.maximum(np.abs(obs_xi2) * args.frac_err_xi2, args.floor_err_xi2)
            sigmawp = np.maximum(np.abs(obs_wp) * args.frac_err_wp, args.floor_err_wp)
            chi2_xi0 = float(np.sum(((model_xi0 - obs_xi0) / sigma0) ** 2))
            chi2_xi2 = float(np.sum(((model_xi2 - obs_xi2) / sigma2) ** 2))
            chi2_wp = float(np.sum(((model_wp - obs_wp) / sigmawp) ** 2))
            model_nbar = float(len(idx) / float(cat["boxsize"]) ** 3)
            chi2_density = float((np.log(model_nbar / obs_nbar) / args.sigma_ln_density) ** 2)
            chi2_cov = np.nan
            chi2_total_cov = np.nan
            if precision is not None and obs_vector is not None:
                model_vector = np.concatenate([model_wp, model_xi0, model_xi2])
                residual = model_vector - obs_vector
                chi2_cov = float(residual @ precision @ residual)
                chi2_total_cov = chi2_cov + chi2_density
            row = {
                "f_sat_target": f_sat,
                "alpha_s": alpha_s,
                "n_gal": int(len(idx)),
                "n_sat": int(np.count_nonzero(np.asarray(cat["is_sat"], dtype=bool)[idx])),
                "f_sat_actual": float(np.count_nonzero(np.asarray(cat["is_sat"], dtype=bool)[idx]) / len(idx)),
                "model_number_density_h3_mpc3": model_nbar,
                "observed_number_density_h3_mpc3": obs_nbar,
                "chi2_xi0": chi2_xi0,
                "chi2_xi2": chi2_xi2,
                "chi2_wp": chi2_wp,
                "chi2_density": chi2_density,
                "chi2_cov": chi2_cov,
                "chi2_total_cov": chi2_total_cov,
                "chi2_total": chi2_xi0 + chi2_xi2 + chi2_wp + chi2_density,
                "status": "ok",
                "traceback": "",
            }
            rows.append(row)
            np.savez_compressed(
                outdir / f"model_fsat{f_sat:.3f}_alpha{alpha_s:.3f}.npz".replace(".", "p"),
                s=s,
                xi0=xi0,
                xi2=xi2,
                rp=rp,
                wp=wp,
                obs_s=obs_s,
                model_xi0=model_xi0,
                model_xi2=model_xi2,
                obs_rp=obs_rp,
                model_wp=model_wp,
            )
            print("OK", row, flush=True)
        except Exception as exc:
            row = {
                "f_sat_target": f_sat,
                "alpha_s": alpha_s,
                "chi2_total": np.inf,
                "chi2_total_cov": np.inf,
                "status": repr(exc),
                "traceback": traceback.format_exc(limit=5),
            }
            rows.append(row)
            print("FAIL", row, flush=True)
    score_column = "chi2_total_cov" if precision is not None else "chi2_total"
    rows.sort(key=lambda row: row[score_column])
    csv_path = outdir / "abacus_fullbox_fsat_alpha_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "catalog_npz": args.catalog_npz,
        "target_count": target_count,
        "boxsize": float(cat["boxsize"]),
        "redshift": float(cat["redshift"]),
        "score_column": score_column,
        "best": rows[:10],
        "covariance_info": covariance_info,
    }
    with (outdir / "abacus_fullbox_fsat_alpha_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(csv_path)
    print("Best:", rows[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
