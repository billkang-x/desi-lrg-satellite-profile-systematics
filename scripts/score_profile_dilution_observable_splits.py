"""Score saved satellite-profile dilution model vectors by observable split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fit_abacus_profile_dilution_pycorr import model_filename


SLICES = {
    "wp": slice(0, 16),
    "xi0": slice(16, 44),
    "xi2": slice(44, 72),
    "xi02": slice(16, 72),
    "joint": slice(0, 72),
}


def read_grid(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_datavector(path: Path) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return np.asarray(table["value"], dtype="f8")


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def vector_path(model_dir: Path, row: dict[str, str]) -> Path:
    path = model_dir / model_filename(as_float(row, "f_sat_target"), as_float(row, "alpha_s"), as_float(row, "radial_scale"))
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def model_vector(path: Path) -> np.ndarray:
    payload = np.load(path)
    return np.concatenate([payload["model_wp"], payload["model_xi0"], payload["model_xi2"]]).astype("f8")


def precision_matrix(cov: np.ndarray, mode: str, n_realizations: int, rcond: float, hartlap: bool) -> tuple[np.ndarray, float]:
    if mode == "diagonal":
        precision = np.diag(1.0 / np.maximum(np.diag(cov), 1e-30))
    elif mode == "full":
        precision = np.linalg.pinv(cov, rcond=rcond)
    else:
        raise ValueError(mode)
    scale = 1.0
    if hartlap:
        p = cov.shape[0]
        scale = max((n_realizations - p - 2) / (n_realizations - 1), 0.0)
        precision *= scale
    return precision, scale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--grid-csv", required=True)
    parser.add_argument("--observed-datavector", default="results/datavectors/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp0.5-30_xi5-60_ells02_datavector.csv")
    parser.add_argument("--covariance", default="results/matched_covariance/LRG_DR1_kmeans96_r0-1-2-3_full_wp_xi_covariance.npy")
    parser.add_argument("--covariance-modes", default="diagonal,full")
    parser.add_argument("--splits", default="wp,xi02,joint")
    parser.add_argument("--n-realizations", type=int, default=96)
    parser.add_argument("--covariance-rcond", type=float, default=1e-6)
    parser.add_argument("--no-hartlap", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_grid(Path(args.grid_csv)) if row.get("status", "ok") == "ok"]
    obs = read_datavector(Path(args.observed_datavector))
    cov = np.load(args.covariance)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    modes = [item.strip() for item in args.covariance_modes.split(",") if item.strip()]

    vector_cache: dict[tuple[float, float, float], np.ndarray] = {}
    out_rows = []
    for split in splits:
        sl = SLICES[split]
        obs_sub = obs[sl]
        cov_sub = cov[sl, sl]
        for mode in modes:
            precision, scale = precision_matrix(cov_sub, mode, args.n_realizations, args.covariance_rcond, not args.no_hartlap)
            for row in rows:
                key = (as_float(row, "f_sat_target"), as_float(row, "alpha_s"), as_float(row, "radial_scale"))
                if key not in vector_cache:
                    vector_cache[key] = model_vector(vector_path(model_dir, row))
                residual = vector_cache[key][sl] - obs_sub
                chi2 = float(residual @ precision @ residual)
                out_rows.append(
                    {
                        "label": args.label,
                        "split": split,
                        "covariance_mode": mode,
                        "hartlap_scale": scale,
                        "f_sat_target": key[0],
                        "alpha_s": key[1],
                        "radial_scale": key[2],
                        "f_sat_actual": as_float(row, "f_sat_actual"),
                        "chi2": chi2,
                        "source_chi2_total": as_float(row, "chi2_total"),
                        "source_chi2_wp": as_float(row, "chi2_wp"),
                        "source_chi2_xi0": as_float(row, "chi2_xi0"),
                        "source_chi2_xi2": as_float(row, "chi2_xi2"),
                    }
                )

    out_rows.sort(key=lambda item: (str(item["split"]), str(item["covariance_mode"]), float(item["chi2"])))
    score_path = outdir / f"{args.label}_profile_dilution_split_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    summary_rows = []
    for split in splits:
        for mode in modes:
            subset = [row for row in out_rows if row["split"] == split and row["covariance_mode"] == mode]
            subset.sort(key=lambda row: float(row["chi2"]))
            best = subset[0]
            second = subset[1] if len(subset) > 1 else best
            summary_rows.append(
                {
                    "label": args.label,
                    "split": split,
                    "covariance_mode": mode,
                    "best_f_sat": best["f_sat_target"],
                    "best_alpha_s": best["alpha_s"],
                    "best_radial_scale": best["radial_scale"],
                    "best_f_sat_actual": best["f_sat_actual"],
                    "best_chi2": best["chi2"],
                    "delta_chi2_to_second": float(second["chi2"]) - float(best["chi2"]),
                    "hartlap_scale": best["hartlap_scale"],
                }
            )
    summary_path = outdir / f"{args.label}_profile_dilution_split_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (outdir / f"{args.label}_profile_dilution_split_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "label": args.label,
                "model_dir": args.model_dir,
                "grid_csv": args.grid_csv,
                "observed_datavector": args.observed_datavector,
                "covariance": args.covariance,
                "splits": splits,
                "covariance_modes": modes,
                "summary": summary_rows,
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(summary_path)
    for row in summary_rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
