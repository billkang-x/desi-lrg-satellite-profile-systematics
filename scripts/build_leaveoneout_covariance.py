"""Build a full covariance from leave-one-out data vectors.

This is the strict route for a fully matched covariance: each input CSV must
have exactly the same rows as the final wp + xi0/xi2 data vector. The covariance
contains all cross-terms, including wp-xi cross covariance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_vector(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    with path.open("r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    values = np.asarray([float(row["value"]) for row in rows], dtype="f8")
    return rows, values


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    # pycorr writes pair-weighted effective separations, so the reported ``r``
    # can move slightly between leave-one-out realizations.  The bin identity is
    # the stable index plus statistic label.
    return (row["index"], row["component"], row["statistic"], row["ell"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-vector", type=Path, required=True)
    parser.add_argument("--jackknife-glob", required=True, help="Glob of leave-one-out datavector CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("results/matched_covariance"))
    parser.add_argument("--tag", default="leaveoneout")
    args = parser.parse_args(argv)

    full_rows, full_values = read_vector(args.full_vector)
    paths = sorted(Path().glob(args.jackknife_glob))
    if len(paths) < 3:
        raise ValueError("Need at least three leave-one-out vectors")
    vectors = []
    for path in paths:
        rows, values = read_vector(path)
        if [row_key(row) for row in rows] != [row_key(row) for row in full_rows]:
            raise ValueError(f"Row mismatch in {path}")
        vectors.append(values)
    matrix = np.vstack(vectors)
    mean_loo = np.mean(matrix, axis=0)
    n_jack = matrix.shape[0]
    delta = matrix - mean_loo
    covariance = (n_jack - 1.0) / n_jack * delta.T.dot(delta)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cov_path = args.output_dir / f"{args.tag}_full_wp_xi_covariance.npy"
    vec_path = args.output_dir / f"{args.tag}_leaveoneout_vectors.npy"
    bins_path = args.output_dir / f"{args.tag}_full_wp_xi_covariance_bins.csv"
    summary_path = args.output_dir / f"{args.tag}_full_wp_xi_covariance_summary.json"
    np.save(cov_path, covariance)
    np.save(vec_path, matrix)
    with bins_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(full_rows[0].keys()))
        writer.writeheader()
        writer.writerows(full_rows)
    summary = {
        "full_vector": str(args.full_vector),
        "n_jackknife": int(n_jack),
        "n_data": int(matrix.shape[1]),
        "covariance_shape": list(covariance.shape),
        "jackknife_vectors": [str(path) for path in paths],
        "outputs": {
            "covariance": str(cov_path),
            "leaveoneout_vectors": str(vec_path),
            "bins": str(bins_path),
        },
        "notes": [
            "This covariance includes wp-xi cross covariance because it is built from full leave-one-out vectors.",
            "Rows must match build_lrg_datavector.py exactly.",
        ],
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(cov_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
