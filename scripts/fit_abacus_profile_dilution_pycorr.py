"""Run satellite radial-profile dilution tests for Abacus compact catalogs."""

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
    make_indices,
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


def token(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def model_filename(f_sat: float, alpha_s: float, radial_scale: float) -> str:
    return f"model_fsat{token(f_sat)}_alpha{token(alpha_s)}_rscale{token(radial_scale)}.npz"


def host_position_lookup(cat: dict[str, np.ndarray | float]) -> dict[str, np.ndarray]:
    is_sat = np.asarray(cat["is_sat"], dtype=bool)
    cen = np.flatnonzero(~is_sat)
    host_ids = np.asarray(cat["host_halo_id"], dtype="i8")[cen]
    order = np.argsort(host_ids)
    return {
        "host_id": host_ids[order],
        "x": np.asarray(cat["x"], dtype="f4")[cen][order],
        "y": np.asarray(cat["y"], dtype="f4")[cen][order],
        "z": np.asarray(cat["z"], dtype="f4")[cen][order],
    }


def host_positions_for(
    lookup: dict[str, np.ndarray],
    host_ids: np.ndarray,
) -> np.ndarray:
    sorted_ids = lookup["host_id"]
    loc = np.searchsorted(sorted_ids, host_ids)
    ok = (loc < len(sorted_ids)) & (sorted_ids[loc] == host_ids)
    if not np.all(ok):
        missing = int(np.count_nonzero(~ok))
        raise ValueError(f"{missing} satellite host IDs are absent from central candidate lookup")
    return np.column_stack([lookup["x"][loc], lookup["y"][loc], lookup["z"][loc]]).astype("f8")


def redshift_space_positions_with_profile(
    cat: dict[str, np.ndarray | float],
    idx: np.ndarray,
    alpha_s: float,
    radial_scale: float,
    los_axis: int,
    lookup: dict[str, np.ndarray],
) -> np.ndarray:
    boxsize = float(cat["boxsize"])
    pos = np.column_stack([cat["x"][idx], cat["y"][idx], cat["z"][idx]]).astype("f8")
    vel = np.column_stack([cat["vx"][idx], cat["vy"][idx], cat["vz"][idx]]).astype("f8")
    host_vel = np.column_stack([cat["host_vx"][idx], cat["host_vy"][idx], cat["host_vz"][idx]]).astype("f8")
    is_sat = np.asarray(cat["is_sat"], dtype=bool)[idx]

    if radial_scale != 1.0 and np.any(is_sat):
        host_ids = np.asarray(cat["host_halo_id"], dtype="i8")[idx][is_sat]
        host_pos = host_positions_for(lookup, host_ids)
        delta = pos[is_sat] - host_pos
        delta -= boxsize * np.rint(delta / boxsize)
        pos[is_sat] = np.mod(host_pos + radial_scale * delta, boxsize)

    vel_los = vel[:, los_axis].copy()
    vel_los[is_sat] = host_vel[is_sat, los_axis] + alpha_s * (vel[is_sat, los_axis] - host_vel[is_sat, los_axis])
    zred = float(cat["redshift"])
    omega_m = 0.315192
    ez = np.sqrt(omega_m * (1.0 + zred) ** 3 + (1.0 - omega_m))
    displacement = vel_los * (1.0 + zred) / (100.0 * ez)
    pos[:, los_axis] = np.mod(pos[:, los_axis] + displacement, boxsize)
    return pos


def make_selection(
    mode: str,
    cat: dict[str, np.ndarray | float],
    f_sat: float,
    target_count: int,
    args: argparse.Namespace,
    hod_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    if mode == "sham":
        return make_indices(cat, f_sat, target_count), {}
    if mode == "minimal_hod":
        idx, info = make_minimal_hod_indices(
            cat,
            f_sat,
            target_count,
            args.sigma_logm,
            args.satellite_power,
            hod_cache["central_gumbel"],
            hod_cache["satellite_gumbel"],
            hod_cache["sat_count_per_host"],
        )
        return idx, info
    raise ValueError(mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-npz", required=True)
    parser.add_argument("--output-dir", default="results/abacus_fullbox_profile_dilution")
    parser.add_argument("--mode", choices=["minimal_hod", "sham"], default="minimal_hod")
    parser.add_argument("--observed-poles", default="results/pycorr_lrg_r0-3/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_poles.csv")
    parser.add_argument("--observed-wp", default="results/pycorr_lrg_r0-3/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp.csv")
    parser.add_argument("--observed-number-density", default="results/number_density/LRG_GCcomb_z0p6-0p8_number_density.json")
    parser.add_argument("--fsat-grid", default="0.04,0.06,0.08,0.10,0.12")
    parser.add_argument("--alpha-s-grid", default="0.7,1.1,1.3")
    parser.add_argument("--radial-scale-grid", default="1.0,1.5,2.0,3.0,4.0")
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

    hod_cache: dict[str, np.ndarray] = {}
    if args.mode == "minimal_hod":
        is_sat = np.asarray(cat["is_sat"], dtype=bool)
        cen_idx = np.flatnonzero(~is_sat)
        sat_idx = np.flatnonzero(is_sat)
        rng = np.random.default_rng(args.seed)
        hod_cache["central_gumbel"] = gumbel_noise(rng, len(cen_idx))
        hod_cache["satellite_gumbel"] = gumbel_noise(rng, len(sat_idx))
        hod_cache["sat_count_per_host"] = make_host_count_per_sat(np.asarray(cat["host_halo_id"], dtype="i8")[sat_idx])

    rows = []
    is_sat_all = np.asarray(cat["is_sat"], dtype=bool)
    grids = product(parse_grid(args.fsat_grid), parse_grid(args.alpha_s_grid), parse_grid(args.radial_scale_grid))
    for f_sat, alpha_s, radial_scale in grids:
        try:
            idx, extra = make_selection(args.mode, cat, f_sat, target_count, args, hod_cache)
            pos = redshift_space_positions_with_profile(cat, idx, alpha_s, radial_scale, args.los_axis, lookup)
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
            n_sat = int(np.count_nonzero(is_sat_all[idx]))
            row = {
                "model": args.mode,
                "f_sat_target": f_sat,
                "alpha_s": alpha_s,
                "radial_scale": radial_scale,
                "sigma_logm": args.sigma_logm if args.mode == "minimal_hod" else np.nan,
                "satellite_power": args.satellite_power if args.mode == "minimal_hod" else np.nan,
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
                **extra,
            }
            rows.append(row)
            np.savez_compressed(
                outdir / model_filename(f_sat, alpha_s, radial_scale),
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
                "model": args.mode,
                "f_sat_target": f_sat,
                "alpha_s": alpha_s,
                "radial_scale": radial_scale,
                "chi2_total": np.inf,
                "status": repr(exc),
                "traceback": traceback.format_exc(limit=5),
            }
            rows.append(row)
            print("FAIL", row, flush=True)

    rows.sort(key=lambda row: row["chi2_total"])
    csv_path = outdir / f"abacus_fullbox_{args.mode}_profile_dilution_grid.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "catalog_npz": args.catalog_npz,
        "mode": args.mode,
        "target_count": target_count,
        "boxsize": float(cat["boxsize"]),
        "redshift": float(cat["redshift"]),
        "best": rows[:10],
    }
    with (outdir / f"abacus_fullbox_{args.mode}_profile_dilution_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(csv_path)
    print("Best:", rows[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
