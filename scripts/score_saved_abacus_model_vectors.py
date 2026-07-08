"""Score saved Abacus model vectors against an observed vector and covariance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def model_path(model_dir: Path, f_sat: float, alpha_s: float) -> Path:
    stem = f"model_fsat{f_sat:.3f}_alpha{alpha_s:.3f}.npz".replace(".", "p")
    path = model_dir / f"{stem}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


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


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_float(row: dict[str, object], key: str) -> float:
    return float(str(row[key]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--grid-csv", required=True)
    parser.add_argument("--observed-datavector", default="results/datavectors/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp0.5-30_xi5-60_ells02_datavector.csv")
    parser.add_argument("--covariance", required=True)
    parser.add_argument("--covariance-mode", choices=["full", "diagonal"], required=True)
    parser.add_argument("--covariance-rcond", type=float, default=1e-6)
    parser.add_argument("--precision-scale", type=float, default=0.23157894736842105)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    obs_vector = read_datavector(Path(args.observed_datavector))
    precision, covariance_info = load_precision(Path(args.covariance), args.covariance_mode, args.covariance_rcond, args.precision_scale)

    rows = []
    for row in read_rows(Path(args.grid_csv)):
        f_sat = as_float(row, "f_sat_target")
        alpha_s = as_float(row, "alpha_s")
        payload = np.load(model_path(model_dir, f_sat, alpha_s))
        model_vector = np.concatenate([payload["model_wp"], payload["model_xi0"], payload["model_xi2"]])
        residual = model_vector - obs_vector
        out = dict(row)
        out["chi2_cov"] = float(residual @ precision @ residual)
        out["chi2_total_cov"] = float(out["chi2_cov"]) + as_float(row, "chi2_density")
        out["status"] = "ok"
        out["traceback"] = ""
        rows.append(out)

    rows.sort(key=lambda item: float(item["chi2_total_cov"]))
    csv_path = outdir / "abacus_fullbox_fsat_alpha_grid.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "model_dir": str(model_dir),
        "grid_csv": args.grid_csv,
        "observed_datavector": args.observed_datavector,
        "score_column": "chi2_total_cov",
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
