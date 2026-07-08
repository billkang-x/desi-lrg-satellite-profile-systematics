"""Export HOD/SHAM model vectors in paper-plot-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

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
from fit_hod_catalog_fsat_alpha import load_hod_catalog, make_fixed_nbar_ranked_indices, make_sample_indices, redshift_space_positions


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_tag(value: float) -> str:
    return f"{value:.3f}".replace(".", "p").replace("-", "m")


def model_id(prefix: str, f_sat: float, alpha_s: float) -> str:
    return f"{prefix}_fsat{safe_tag(f_sat)}_alphaS{safe_tag(alpha_s)}"


def table_stem(prefix: str) -> str:
    return f"{prefix}_paper_plot"


def key_for(f_sat: float, alpha_s: float) -> tuple[float, float]:
    return (round(float(f_sat), 10), round(float(alpha_s), 10))


def read_grid(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in read_rows(path):
        if row.get("status", "ok") != "ok":
            continue
        parsed: dict[str, object] = dict(row)
        for column in [
            "f_sat_target",
            "alpha_s",
            "n_gal",
            "n_sat",
            "f_sat_actual",
            "model_number_density_h3_mpc3",
            "chi2_total",
            "chi2_total_cov",
            "chi2_cov",
            "chi2_xi0",
            "chi2_xi2",
            "chi2_wp",
        ]:
            if column in parsed and parsed[column] not in ("", "nan", "inf"):
                parsed[column] = float(parsed[column])
        rows.append(parsed)
    return rows


def load_score_tables(score_args: list[str]) -> tuple[dict[tuple[float, float], dict[str, object]], list[str]]:
    score_by_key: dict[tuple[float, float], dict[str, object]] = {}
    score_columns: list[str] = []
    for item in score_args:
        label, path_text = item.split("=", 1)
        rows = read_grid(Path(path_text))
        score_column = "chi2_total_cov" if any("chi2_total_cov" in row for row in rows) else "chi2_total"
        finite_rows = [row for row in rows if math.isfinite(float(row.get(score_column, math.inf)))]
        finite_rows.sort(key=lambda row: float(row[score_column]))
        for rank, row in enumerate(finite_rows, start=1):
            bucket = score_by_key.setdefault(key_for(float(row["f_sat_target"]), float(row["alpha_s"])), {})
            for column in ["chi2_total", "chi2_cov", "chi2_total_cov", "chi2_xi0", "chi2_xi2", "chi2_wp"]:
                if column in row:
                    output_column = f"{label}_{column}"
                    bucket[output_column] = row[column]
                    if output_column not in score_columns:
                        score_columns.append(output_column)
            rank_column = f"{label}_rank"
            bucket[rank_column] = rank
            if rank_column not in score_columns:
                score_columns.append(rank_column)
    return score_by_key, score_columns


def selected_reason(scores: dict[str, object], top_n: int) -> tuple[bool, str]:
    reasons = []
    for column, value in scores.items():
        if column.endswith("_rank") and int(value) <= top_n:
            reasons.append(column.replace("_rank", f"_top{top_n}"))
    return bool(reasons), ";".join(reasons)


def read_datavector(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in read_rows(path):
        parsed: dict[str, object] = dict(row)
        parsed["index"] = int(row["index"])
        parsed["ell"] = int(row["ell"])
        parsed["r"] = float(row["r"])
        parsed["value"] = float(row["value"])
        rows.append(parsed)
    return rows


def model_vector_for_row(args: argparse.Namespace, cat: dict[str, object], grid_row: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
    is_sat = np.asarray(cat["is_sat"])
    target_count = int(round(args.target_number_density * float(cat["boxsize"]) ** 3))
    f_sat = float(grid_row["f_sat_target"])
    alpha_s = float(grid_row["alpha_s"])
    if args.selection_mode == "fixed-nbar-ranked":
        idx = make_fixed_nbar_ranked_indices(is_sat, np.asarray(cat["rank_value"]), f_sat, target_count)
    else:
        idx = make_sample_indices(is_sat, f_sat, np.random.default_rng(args.seed))
    obs_s, _, _ = read_observed_poles(Path(args.observed_poles), args.s_min, args.s_max)
    obs_rp, _ = read_observed_wp(Path(args.observed_wp), args.rp_min, args.rp_max)
    s_edges = np.linspace(args.s_min, args.s_max, args.s_bins + 1)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    rp_edges = np.logspace(np.log10(args.rp_min), np.log10(args.rp_max), args.rp_bins + 1)
    rp_centers = np.sqrt(rp_edges[:-1] * rp_edges[1:])
    max_sep = max(args.s_max, np.sqrt(args.rp_max**2 + args.pimax**2))
    pos = redshift_space_positions(cat, idx, alpha_s=alpha_s, los_axis=args.los_axis)
    sep, rp, pi = pair_separations_periodic(pos, max_sep, float(cat["boxsize"]))
    xi0, xi2 = xi_from_pairs(sep, pi, s_edges, len(pos), float(cat["boxsize"]))
    wp = wp_from_pairs(rp, pi, rp_edges, args.pimax, len(pos), float(cat["boxsize"]))
    model_xi0, model_xi2 = interpolate_model(s_centers, xi0, xi2, obs_s)
    model_wp = interpolate_wp_model(rp_centers, wp, obs_rp)
    return np.concatenate([model_wp, model_xi0, model_xi2]), {
        "n_gal": int(len(idx)),
        "n_sat": int(np.count_nonzero(is_sat[idx])),
        "f_sat_actual": float(np.count_nonzero(is_sat[idx]) / len(idx)),
        "model_number_density_h3_mpc3": float(len(idx) / float(cat["boxsize"]) ** 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--score-grid", action="append", default=[])
    parser.add_argument("--output-dir", default="results/hod_model_vectors")
    parser.add_argument("--model-prefix", default="tng3002")
    parser.add_argument("--observed-datavector", default="results/datavectors/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp0.5-30_xi5-60_ells02_datavector.csv")
    parser.add_argument("--observed-poles", default="results/pycorr_lrg_r0-3/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_poles.csv")
    parser.add_argument("--observed-wp", default="results/pycorr_lrg_r0-3/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp.csv")
    parser.add_argument("--observed-number-density", default="results/number_density/LRG_GCcomb_z0p6-0p8_number_density.json")
    parser.add_argument("--covariance", default="results/matched_covariance/LRG_DR1_kmeans96_r0-1-2-3_full_wp_xi_covariance.npy")
    parser.add_argument("--boxsize", type=float, default=None)
    parser.add_argument("--redshift", type=float, default=None)
    parser.add_argument("--selection-mode", choices=["fixed-nbar-ranked", "keep-centrals"], default="fixed-nbar-ranked")
    parser.add_argument("--selected-top-n", type=int, default=5)
    parser.add_argument("--s-min", type=float, default=5.0)
    parser.add_argument("--s-max", type=float, default=60.0)
    parser.add_argument("--s-bins", type=int, default=28)
    parser.add_argument("--rp-min", type=float, default=0.5)
    parser.add_argument("--rp-max", type=float, default=30.0)
    parser.add_argument("--rp-bins", type=int, default=16)
    parser.add_argument("--pimax", type=float, default=80.0)
    parser.add_argument("--los-axis", type=int, default=2)
    parser.add_argument("--seed", type=int, default=24680)
    args = parser.parse_args(argv)

    args.target_number_density = read_observed_number_density(Path(args.observed_number_density))
    output_dir = Path(args.output_dir)
    vector_dir = output_dir / "vectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    cat = load_hod_catalog(Path(args.catalog), boxsize=args.boxsize, redshift=args.redshift)
    data_rows = read_datavector(Path(args.observed_datavector))
    sigma_jk = np.sqrt(np.diag(np.load(args.covariance))) if args.covariance else np.full(len(data_rows), np.nan)
    grid_rows = read_grid(Path(args.grid))
    score_by_key, score_columns = load_score_tables(args.score_grid)
    long_rows = []
    vector_paths = []
    for grid_row in grid_rows:
        f_sat = float(grid_row["f_sat_target"])
        alpha_s = float(grid_row["alpha_s"])
        scores = score_by_key.get(key_for(f_sat, alpha_s), {})
        selected, reason = selected_reason(scores, args.selected_top_n)
        mid = model_id(args.model_prefix, f_sat, alpha_s)
        vector, meta = model_vector_for_row(args, cat, grid_row)
        vector_path = vector_dir / f"{mid}_model_vector.csv"
        write_csv(
            vector_path,
            [{"index": r["index"], "component": r["component"], "statistic": r["statistic"], "ell": r["ell"], "r": r["r"], "model_value": float(v)} for r, v in zip(data_rows, vector)],
            ["index", "component", "statistic", "ell", "r", "model_value"],
        )
        vector_paths.append(str(vector_path))
        for data_row, model_value, err in zip(data_rows, vector, sigma_jk):
            data_value = float(data_row["value"])
            residual = float(model_value - data_value)
            row = {
                "model_id": mid,
                "model_label": f"f_sat={f_sat:.2f}, alpha_s={alpha_s:.1f}",
                "selected_for_paper": int(selected),
                "selection_reason": reason,
                "f_sat_target": f_sat,
                "alpha_s": alpha_s,
                "n_gal": meta["n_gal"],
                "n_sat": meta["n_sat"],
                "f_sat_actual": meta["f_sat_actual"],
                "model_number_density_h3_mpc3": meta["model_number_density_h3_mpc3"],
                "index": data_row["index"],
                "component": data_row["component"],
                "statistic": data_row["statistic"],
                "ell": data_row["ell"],
                "r": data_row["r"],
                "data_value": data_value,
                "model_value": float(model_value),
                "residual": residual,
                "fractional_residual": residual / max(abs(data_value), 1e-30),
                "sigma_jk": float(err),
                "pull_jk": residual / err if np.isfinite(err) and err > 0 else np.nan,
            }
            row.update(scores)
            long_rows.append(row)
    base_cols = [
        "model_id",
        "model_label",
        "selected_for_paper",
        "selection_reason",
        "f_sat_target",
        "alpha_s",
        "n_gal",
        "n_sat",
        "f_sat_actual",
        "model_number_density_h3_mpc3",
        "index",
        "component",
        "statistic",
        "ell",
        "r",
        "data_value",
        "model_value",
        "residual",
        "fractional_residual",
        "sigma_jk",
        "pull_jk",
    ]
    stem = table_stem(args.model_prefix)
    write_csv(output_dir / f"{stem}_long.csv", long_rows, base_cols + score_columns)
    write_csv(output_dir / f"{stem}_selected_long.csv", [r for r in long_rows if int(r["selected_for_paper"]) == 1], base_cols + score_columns)
    summary_rows = []
    for mid in sorted({r["model_id"] for r in long_rows}):
        for comp in ["all", "wp", "xi0", "xi2"]:
            rows = [r for r in long_rows if r["model_id"] == mid and (comp == "all" or r["component"] == comp)]
            residual = np.asarray([float(r["residual"]) for r in rows])
            data = np.asarray([float(r["data_value"]) for r in rows])
            sigma = np.asarray([float(r["sigma_jk"]) for r in rows])
            valid = np.isfinite(sigma) & (sigma > 0)
            first = rows[0]
            row = {
                "model_id": mid,
                "component": comp,
                "n_bins": len(rows),
                "f_sat_target": first["f_sat_target"],
                "alpha_s": first["alpha_s"],
                "f_sat_actual": first["f_sat_actual"],
                "rms_fractional_residual": float(np.sqrt(np.mean((residual / np.maximum(np.abs(data), 1e-30)) ** 2))),
                "mean_abs_fractional_residual": float(np.mean(np.abs(residual / np.maximum(np.abs(data), 1e-30)))),
                "chi2_diag_jk_raw": float(np.sum((residual[valid] / sigma[valid]) ** 2)),
                "rms_pull_jk": float(np.sqrt(np.mean((residual[valid] / sigma[valid]) ** 2))),
                "selected_for_paper": first["selected_for_paper"],
                "selection_reason": first["selection_reason"],
            }
            row.update({c: first.get(c, "") for c in score_columns})
            summary_rows.append(row)
    summary_cols = [
        "model_id",
        "component",
        "n_bins",
        "f_sat_target",
        "alpha_s",
        "f_sat_actual",
        "rms_fractional_residual",
        "mean_abs_fractional_residual",
        "chi2_diag_jk_raw",
        "rms_pull_jk",
        "selected_for_paper",
        "selection_reason",
    ] + score_columns
    write_csv(output_dir / f"{stem}_component_summary.csv", summary_rows, summary_cols)
    manifest = {
        "catalog": args.catalog,
        "grid": args.grid,
        "score_grids": args.score_grid,
        "observed_datavector": args.observed_datavector,
        "covariance": args.covariance,
        "n_models": len(grid_rows),
        "n_bins": len(data_rows),
        "long_table": str(output_dir / f"{stem}_long.csv"),
        "selected_long_table": str(output_dir / f"{stem}_selected_long.csv"),
        "component_summary": str(output_dir / f"{stem}_component_summary.csv"),
        "vector_dir": str(vector_dir),
        "vector_files": vector_paths,
    }
    with (output_dir / f"{stem}_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output_dir / f"{stem}_long.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
