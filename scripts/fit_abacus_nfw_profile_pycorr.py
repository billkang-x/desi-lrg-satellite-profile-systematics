"""Run an NFW-like satellite radial-profile concentration test.

The previous profile-dilution test used a direct radial scaling,
x_sat -> x_host + q (x_sat - x_host).  This script keeps the same compact
Abacus catalog and minimal-HOD occupation layer, but remaps satellite radii
through the cumulative NFW profile.  For a selected satellite with normalized
radius y = r / R_200m, we compute its reference NFW quantile at c_ref and
place it at the same quantile in a target profile c_sat = c_ref * c_ratio.

This is still a controlled diagnostic rather than the official AbacusHOD
satellite profile model, but it is closer to a physical concentration
parameter than uniform radial dilution.
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from itertools import product
from pathlib import Path

import numpy as np

from fit_abacus_fullbox_pycorr import (
    interp_model,
    load_catalog,
    measure_pycorr,
    parse_grid,
    read_number_density,
    read_observed_poles,
    read_observed_wp,
)
from fit_abacus_minimal_hod_pycorr import (
    gumbel_noise,
    make_host_count_per_sat,
    make_minimal_hod_indices,
)
from fit_abacus_profile_dilution_pycorr import host_position_lookup, host_positions_for


def token(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def model_filename(f_sat: float, alpha_s: float, c_ratio: float) -> str:
    return f"model_fsat{token(f_sat)}_alpha{token(alpha_s)}_cratio{token(c_ratio)}.npz"


def nfw_enclosed_fraction(y: np.ndarray, concentration: float) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype="f8"), 0.0, 1.0)
    concentration = float(concentration)
    x = concentration * y
    enclosed = np.log1p(x) - x / (1.0 + x)
    norm = np.log1p(concentration) - concentration / (1.0 + concentration)
    return enclosed / norm


def nfw_inverse_enclosed_fraction(u: np.ndarray, concentration: float, n_grid: int = 16384) -> np.ndarray:
    grid_y = np.linspace(0.0, 1.0, n_grid + 1)
    grid_u = nfw_enclosed_fraction(grid_y, concentration)
    return np.interp(np.clip(u, 0.0, 1.0), grid_u, grid_y)


def r200m_from_halo_n(
    halo_n: np.ndarray,
    particle_mass_hinv_msun: float,
    omega_m: float,
    overdensity: float,
) -> np.ndarray:
    rho_crit = 2.77536627e11  # h^2 Msun Mpc^-3
    rho_m = omega_m * rho_crit
    mass = np.maximum(np.asarray(halo_n, dtype="f8"), 1.0) * particle_mass_hinv_msun
    return (3.0 * mass / (4.0 * np.pi * overdensity * rho_m)) ** (1.0 / 3.0)


def redshift_space_positions_with_nfw_profile(
    cat: dict[str, np.ndarray | float],
    idx: np.ndarray,
    alpha_s: float,
    c_ratio: float,
    args: argparse.Namespace,
    lookup: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    boxsize = float(cat["boxsize"])
    pos = np.column_stack([cat["x"][idx], cat["y"][idx], cat["z"][idx]]).astype("f8")
    vel = np.column_stack([cat["vx"][idx], cat["vy"][idx], cat["vz"][idx]]).astype("f8")
    host_vel = np.column_stack([cat["host_vx"][idx], cat["host_vy"][idx], cat["host_vz"][idx]]).astype("f8")
    is_sat = np.asarray(cat["is_sat"], dtype=bool)[idx]

    profile_info = {
        "c_ref": float(args.c_ref),
        "c_ratio": float(c_ratio),
        "c_sat": float(args.c_ref * c_ratio),
        "sat_radius_scale_median": float("nan"),
        "sat_radius_scale_p16": float("nan"),
        "sat_radius_scale_p84": float("nan"),
        "sat_r_over_r200m_median": float("nan"),
    }
    if np.any(is_sat):
        host_ids = np.asarray(cat["host_halo_id"], dtype="i8")[idx][is_sat]
        host_pos = host_positions_for(lookup, host_ids)
        delta = pos[is_sat] - host_pos
        delta -= boxsize * np.rint(delta / boxsize)
        radius = np.linalg.norm(delta, axis=1)
        host_halo_n = np.asarray(cat["halo_n"], dtype="f8")[idx][is_sat]
        r200m = r200m_from_halo_n(host_halo_n, args.particle_mass, args.omega_m, args.overdensity)
        # A small fraction of proxy particles can lie outside the crude R200m
        # estimate.  Expanding only those radii prevents artificial truncation
        # while keeping the NFW mapping well-defined.
        effective_r200m = np.maximum(r200m, radius / 0.995)
        y_ref = np.clip(radius / effective_r200m, 1e-6, 0.995)
        quantile = nfw_enclosed_fraction(y_ref, args.c_ref)
        y_new = nfw_inverse_enclosed_fraction(quantile, args.c_ref * c_ratio)
        new_radius = y_new * effective_r200m
        scale = np.where(radius > 0.0, new_radius / radius, 1.0)
        pos[is_sat] = np.mod(host_pos + delta * scale[:, None], boxsize)
        profile_info.update(
            {
                "sat_radius_scale_median": float(np.median(scale)),
                "sat_radius_scale_p16": float(np.percentile(scale, 16.0)),
                "sat_radius_scale_p84": float(np.percentile(scale, 84.0)),
                "sat_r_over_r200m_median": float(np.median(radius / effective_r200m)),
            }
        )

    vel_los = vel[:, args.los_axis].copy()
    vel_los[is_sat] = host_vel[is_sat, args.los_axis] + alpha_s * (vel[is_sat, args.los_axis] - host_vel[is_sat, args.los_axis])
    zred = float(cat["redshift"])
    ez = np.sqrt(args.omega_m * (1.0 + zred) ** 3 + (1.0 - args.omega_m))
    displacement = vel_los * (1.0 + zred) / (100.0 * ez)
    pos[:, args.los_axis] = np.mod(pos[:, args.los_axis] + displacement, boxsize)
    return pos, profile_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-npz", required=True)
    parser.add_argument("--output-dir", default="results/abacus_fullbox_minimal_hod_nfw_profile")
    parser.add_argument("--observed-poles", default="results/pycorr_lrg_r0-3/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_poles.csv")
    parser.add_argument("--observed-wp", default="results/pycorr_lrg_r0-3/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp.csv")
    parser.add_argument("--observed-number-density", default="results/number_density/LRG_GCcomb_z0p6-0p8_number_density.json")
    parser.add_argument("--fsat-grid", default="0.04,0.06,0.08,0.10,0.12")
    parser.add_argument("--alpha-s-grid", default="0.7,1.1,1.3")
    parser.add_argument("--concentration-ratio-grid", default="0.35,0.50,0.70,1.00,1.30")
    parser.add_argument("--c-ref", type=float, default=5.0)
    parser.add_argument("--particle-mass", type=float, default=2.1e9)
    parser.add_argument("--omega-m", type=float, default=0.315192)
    parser.add_argument("--overdensity", type=float, default=200.0)
    parser.add_argument("--sigma-logm", type=float, default=0.25)
    parser.add_argument("--satellite-power", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13579)
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
    cat = load_catalog(Path(args.catalog_npz))
    target_count = int(round(obs_nbar * float(cat["boxsize"]) ** 3))
    lookup = host_position_lookup(cat)

    is_sat = np.asarray(cat["is_sat"], dtype=bool)
    cen_idx = np.flatnonzero(~is_sat)
    sat_idx = np.flatnonzero(is_sat)
    rng = np.random.default_rng(args.seed)
    central_gumbel = gumbel_noise(rng, len(cen_idx))
    satellite_gumbel = gumbel_noise(rng, len(sat_idx))
    sat_count_per_host = make_host_count_per_sat(np.asarray(cat["host_halo_id"], dtype="i8")[sat_idx])

    rows = []
    grids = product(parse_grid(args.fsat_grid), parse_grid(args.alpha_s_grid), parse_grid(args.concentration_ratio_grid))
    for f_sat, alpha_s, c_ratio in grids:
        try:
            idx, hod_info = make_minimal_hod_indices(
                cat,
                f_sat,
                target_count,
                args.sigma_logm,
                args.satellite_power,
                central_gumbel,
                satellite_gumbel,
                sat_count_per_host,
            )
            pos, profile_info = redshift_space_positions_with_nfw_profile(cat, idx, alpha_s, c_ratio, args, lookup)
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
            n_sat = int(np.count_nonzero(is_sat[idx]))
            row = {
                "model": "minimal_hod_nfw_profile",
                "f_sat_target": f_sat,
                "alpha_s": alpha_s,
                "concentration_ratio": c_ratio,
                "sigma_logm": args.sigma_logm,
                "satellite_power": args.satellite_power,
                "particle_mass_hinv_msun": args.particle_mass,
                "n_gal": int(len(idx)),
                "n_sat": n_sat,
                "f_sat_actual": float(n_sat / len(idx)),
                "model_number_density_h3_mpc3": model_nbar,
                "observed_number_density_h3_mpc3": obs_nbar,
                "chi2_xi0": chi2_xi0,
                "chi2_xi2": chi2_xi2,
                "chi2_wp": chi2_wp,
                "chi2_density": chi2_density,
                "chi2_total": chi2_xi0 + chi2_xi2 + chi2_wp + chi2_density,
                "status": "ok",
                "traceback": "",
                **hod_info,
                **profile_info,
            }
            rows.append(row)
            np.savez_compressed(
                outdir / model_filename(f_sat, alpha_s, c_ratio),
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
                "model": "minimal_hod_nfw_profile",
                "f_sat_target": f_sat,
                "alpha_s": alpha_s,
                "concentration_ratio": c_ratio,
                "chi2_total": np.inf,
                "status": repr(exc),
                "traceback": traceback.format_exc(limit=5),
            }
            rows.append(row)
            print("FAIL", row, flush=True)

    rows.sort(key=lambda row: row["chi2_total"])
    csv_path = outdir / "abacus_fullbox_minimal_hod_nfw_profile_grid.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "catalog_npz": args.catalog_npz,
        "target_count": target_count,
        "boxsize": float(cat["boxsize"]),
        "redshift": float(cat["redshift"]),
        "c_ref": args.c_ref,
        "particle_mass_hinv_msun": args.particle_mass,
        "best": rows[:10],
    }
    with (outdir / "abacus_fullbox_minimal_hod_nfw_profile_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(csv_path)
    print("Best:", rows[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
