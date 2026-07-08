"""Score saved NFW-profile model vectors by observable split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fit_abacus_nfw_profile_pycorr import model_filename


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
    path = model_dir / model_filename(
        as_float(row, "f_sat_target"),
        as_float(row, "alpha_s"),
        as_float(row, "concentration_ratio"),
    )
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
    data = read_datavector(Path(args.observed_datavector))
    cov_full = np.load(args.covariance)
    covariance_modes = [item.strip() for item in args.covariance_modes.split(",") if item.strip()]
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    scores: list[dict[str, object]] = []
    for split in splits:
        slc = SLICES[split]
        data_split = data[slc]
        cov_split = cov_full[slc, slc]
        for mode in covariance_modes:
            precision, hartlap_scale = precision_matrix(
                cov_split,
                mode,
                args.n_realizations,
                args.covariance_rcond,
                not args.no_hartlap,
            )
            for row in rows:
                residual = model_vector(vector_path(model_dir, row))[slc] - data_split
                chi2 = float(residual @ precision @ residual)
                scores.append(
                    {
                        "label": args.label,
                        "split": split,
                        "covariance_mode": mode,
                        "f_sat_target": as_float(row, "f_sat_target"),
                        "alpha_s": as_float(row, "alpha_s"),
                        "concentration_ratio": as_float(row, "concentration_ratio"),
                        "c_sat": as_float(row, "c_sat"),
                        "f_sat_actual": as_float(row, "f_sat_actual"),
                        "chi2": chi2,
                        "hartlap_scale": hartlap_scale,
                    }
                )

    score_path = outdir / f"{args.label}_nfw_profile_split_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores[0].keys()))
        writer.writeheader()
        writer.writerows(scores)

    summary: list[dict[str, object]] = []
    for split in splits:
        for mode in covariance_modes:
            subset = [row for row in scores if row["split"] == split and row["covariance_mode"] == mode]
            subset.sort(key=lambda row: float(row["chi2"]))
            best = subset[0]
            second = subset[1] if len(subset) > 1 else best
            summary.append(
                {
                    "label": args.label,
                    "split": split,
                    "covariance_mode": mode,
                    "best_f_sat": best["f_sat_target"],
                    "best_alpha_s": best["alpha_s"],
                    "best_concentration_ratio": best["concentration_ratio"],
                    "best_c_sat": best["c_sat"],
                    "best_f_sat_actual": best["f_sat_actual"],
                    "best_chi2": best["chi2"],
                    "delta_chi2_to_second": float(second["chi2"]) - float(best["chi2"]),
                    "hartlap_scale": best["hartlap_scale"],
                }
            )
    summary_path = outdir / f"{args.label}_nfw_profile_split_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    metadata = {
        "label": args.label,
        "grid_csv": args.grid_csv,
        "model_dir": args.model_dir,
        "observed_datavector": args.observed_datavector,
        "covariance": args.covariance,
        "n_realizations": args.n_realizations,
        "hartlap": not args.no_hartlap,
        "summary": summary,
    }
    with (outdir / f"{args.label}_nfw_profile_split_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(score_path)
    print(summary_path)
    for item in summary:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
