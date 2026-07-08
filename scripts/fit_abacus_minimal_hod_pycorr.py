"""Fit f_sat and alpha_s with a minimal HOD-like occupation model.

This is intentionally a lightweight model for resource-limited tests. It uses
the compact Abacus catalog produced from halo_info + halo_rv_A, then constructs
an exact-number-density sample with:

- central occupation: noisy mass-rank selection, controlled by sigma_logm;
- satellite occupation: host-mass-weighted selection from the satellite
  particle reservoir, restricted to halos with occupied centrals;
- satellite velocity bias: v_sat,los = v_host,los + alpha_s * (v_particle-v_host).

It is not the official AbacusHOD implementation, but it is closer to an HOD
occupation layer than the baseline abundance-ranked SHAM-like reservoir.
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
    redshift_space_positions,
)


def gumbel_noise(rng: np.random.Generator, size: int) -> np.ndarray:
    u = np.clip(rng.random(size, dtype="f8"), 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(u))


def top_n(values: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=np.int64)
    if n > len(values):
        raise ValueError(f"Need {n} objects, only {len(values)} available")
    keep = np.argpartition(values, len(values) - n)[len(values) - n :]
    return keep[np.argsort(values[keep])[::-1]]


def make_host_count_per_sat(host_ids: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(host_ids, return_inverse=True, return_counts=True)
    return counts[inverse].astype("f8")


def make_minimal_hod_indices(
    cat: dict[str, np.ndarray | float],
    f_sat: float,
    target_count: int,
    sigma_logm: float,
    satellite_power: float,
    central_gumbel: np.ndarray,
    satellite_gumbel: np.ndarray,
    sat_count_per_host: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    is_sat = np.asarray(cat["is_sat"], dtype=bool)
    halo_n = np.asarray(cat["halo_n"], dtype="f8")
    host_ids = np.asarray(cat["host_halo_id"], dtype="i8")
    cen_idx = np.flatnonzero(~is_sat)
    sat_idx = np.flatnonzero(is_sat)

    n_sat = int(round(f_sat * target_count))
    n_cen = target_count - n_sat
    if n_sat > len(sat_idx):
        raise ValueError(f"Need {n_sat} satellites, have {len(sat_idx)} in the reservoir")
    if n_cen > len(cen_idx):
        raise ValueError(f"Need {n_cen} centrals, have {len(cen_idx)} in the reservoir")

    cen_logn = np.log10(np.maximum(halo_n[cen_idx], 1.0))
    scatter = max(float(sigma_logm), 1e-6)
    cen_score = cen_logn / scatter + central_gumbel
    chosen_cen_local = top_n(cen_score, n_cen)
    chosen_cen = cen_idx[chosen_cen_local]
    chosen_hosts = np.sort(host_ids[chosen_cen])

    sat_hosts = host_ids[sat_idx]
    allowed = np.isin(sat_hosts, chosen_hosts, assume_unique=False)
    allowed_sat_local = np.flatnonzero(allowed)
    if n_sat > len(allowed_sat_local):
        raise ValueError(f"Need {n_sat} satellites, but only {len(allowed_sat_local)} candidates live in occupied-central hosts")

    sat_logn = np.log10(np.maximum(halo_n[sat_idx], 1.0))
    pivot = np.median(sat_logn[allowed_sat_local]) if len(allowed_sat_local) else np.median(sat_logn)
    # Divide by available candidates per host so the host, not the particle
    # reservoir depth, carries the occupation weight.
    log_weight = satellite_power * (sat_logn - pivot) - np.log(np.maximum(sat_count_per_host, 1.0))
    sat_score = log_weight + satellite_gumbel
    chosen_allowed_local = allowed_sat_local[top_n(sat_score[allowed_sat_local], n_sat)]
    chosen_sat = sat_idx[chosen_allowed_local]
    idx = np.sort(np.concatenate([chosen_cen, chosen_sat]))
    info = {
        "n_cen_target": float(n_cen),
        "n_sat_target": float(n_sat),
        "n_allowed_sat_candidates": float(len(allowed_sat_local)),
        "central_log10N_min_selected": float(np.min(cen_logn[chosen_cen_local])),
        "central_log10N_median_selected": float(np.median(cen_logn[chosen_cen_local])),
        "satellite_log10N_median_selected": float(np.median(sat_logn[chosen_allowed_local])) if n_sat else float("nan"),
    }
    return idx, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-npz", required=True)
    parser.add_argument("--output-dir", default="results/abacus_fullbox_minimal_hod_fsat_alpha")
    parser.add_argument("--observed-poles", default="results/pycorr_lrg_r0-3/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_poles.csv")
    parser.add_argument("--observed-wp", default="results/pycorr_lrg_r0-3/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp.csv")
    parser.add_argument("--observed-number-density", default="results/number_density/LRG_GCcomb_z0p6-0p8_number_density.json")
    parser.add_argument("--fsat-grid", default="0.02,0.04,0.06,0.08,0.10,0.12,0.15")
    parser.add_argument("--alpha-s-grid", default="0.5,0.7,0.9,1.1,1.3")
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

    is_sat = np.asarray(cat["is_sat"], dtype=bool)
    cen_idx = np.flatnonzero(~is_sat)
    sat_idx = np.flatnonzero(is_sat)
    rng = np.random.default_rng(args.seed)
    central_gumbel = gumbel_noise(rng, len(cen_idx))
    satellite_gumbel = gumbel_noise(rng, len(sat_idx))
    sat_count_per_host = make_host_count_per_sat(np.asarray(cat["host_halo_id"], dtype="i8")[sat_idx])

    rows = []
    for f_sat, alpha_s in product(parse_grid(args.fsat_grid), parse_grid(args.alpha_s_grid)):
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
            n_sat_actual = int(np.count_nonzero(is_sat[idx]))
            row = {
                "model": "minimal_hod",
                "f_sat_target": f_sat,
                "alpha_s": alpha_s,
                "sigma_logm": args.sigma_logm,
                "satellite_power": args.satellite_power,
                "n_gal": int(len(idx)),
                "n_sat": n_sat_actual,
                "f_sat_actual": float(n_sat_actual / len(idx)),
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
                "model": "minimal_hod",
                "f_sat_target": f_sat,
                "alpha_s": alpha_s,
                "sigma_logm": args.sigma_logm,
                "satellite_power": args.satellite_power,
                "chi2_total": np.inf,
                "status": repr(exc),
                "traceback": traceback.format_exc(limit=5),
            }
            rows.append(row)
            print("FAIL", row, flush=True)

    rows.sort(key=lambda row: row["chi2_total"])
    csv_path = outdir / "abacus_fullbox_minimal_hod_fsat_alpha_grid.csv"
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
        "sigma_logm": args.sigma_logm,
        "satellite_power": args.satellite_power,
        "best": rows[:10],
    }
    with (outdir / "abacus_fullbox_minimal_hod_fsat_alpha_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(csv_path)
    print("Best:", rows[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
