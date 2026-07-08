"""Build covariance matched to the wp + xi multipole data vector.

Inputs are the pycorr measurement CSV files and optional covariance matrices
written by measure_lrg_pycorr.py when jackknife samples are enabled. The output
row order matches build_lrg_datavector.py:

1. wp(rp)
2. xi0(s)
3. xi2(s)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_ells(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def read_csv(path: Path) -> np.ndarray:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if table.ndim == 0:
        table = np.asarray([table], dtype=table.dtype)
    return table


def make_rows(label: str, input_dir: Path, wp_min: float, wp_max: float, xi_min: float, xi_max: float, ells: list[int]):
    rows = []
    wp = read_csv(input_dir / f"{label}_wp.csv")
    wp_mask = (wp["rp_Mpch"] >= wp_min) & (wp["rp_Mpch"] <= wp_max)
    wp_indices = np.flatnonzero(wp_mask)
    for index in wp_indices:
        rows.append(
            {
                "index": len(rows),
                "component": "wp",
                "statistic": "wp",
                "ell": -1,
                "r": float(wp["rp_Mpch"][index]),
                "source_cov": "wp",
                "source_index": int(index),
            }
        )

    poles = read_csv(input_dir / f"{label}_poles.csv")
    xi_mask = (poles["s_Mpch"] >= xi_min) & (poles["s_Mpch"] <= xi_max)
    xi_indices = np.flatnonzero(xi_mask)
    n_s = len(poles)
    for ell in ells:
        component = f"xi{ell}"
        for index in xi_indices:
            rows.append(
                {
                    "index": len(rows),
                    "component": component,
                    "statistic": "xi",
                    "ell": int(ell),
                    "r": float(poles["s_Mpch"][index]),
                    "source_cov": "poles",
                    "source_index": int(ells.index(ell) * n_s + index),
                }
            )
    return rows, wp_indices, xi_indices, n_s


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/pycorr_lrg_jackknife")
    parser.add_argument("--output-dir", default="results/matched_covariance")
    parser.add_argument("--label", required=True)
    parser.add_argument("--ells", default="0,2")
    parser.add_argument("--wp-min", type=float, default=0.5)
    parser.add_argument("--wp-max", type=float, default=30.0)
    parser.add_argument("--xi-min", type=float, default=5.0)
    parser.add_argument("--xi-max", type=float, default=60.0)
    parser.add_argument("--allow-block-diagonal", action="store_true")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ells = parse_ells(args.ells)
    rows, wp_indices, xi_indices, n_s = make_rows(
        args.label, input_dir, args.wp_min, args.wp_max, args.xi_min, args.xi_max, ells
    )

    wp_cov_path = input_dir / f"{args.label}_wp_cov.npy"
    poles_cov_path = input_dir / f"{args.label}_poles_cov.npy"
    if not wp_cov_path.exists() or not poles_cov_path.exists():
        missing = [str(path) for path in [wp_cov_path, poles_cov_path] if not path.exists()]
        raise FileNotFoundError("Missing covariance files: " + ", ".join(missing))

    wp_cov = np.load(wp_cov_path)
    poles_cov = np.load(poles_cov_path)
    wp_sub = wp_cov[np.ix_(wp_indices, wp_indices)]
    pole_source_indices = np.asarray(
        [row["source_index"] for row in rows if row["source_cov"] == "poles"], dtype="i8"
    )
    poles_sub = poles_cov[np.ix_(pole_source_indices, pole_source_indices)]
    n_wp = len(wp_indices)
    n_xi = len(pole_source_indices)
    cov = np.zeros((n_wp + n_xi, n_wp + n_xi), dtype="f8")
    cov[:n_wp, :n_wp] = wp_sub
    cov[n_wp:, n_wp:] = poles_sub

    tag = f"{args.label}_wp{args.wp_min:g}-{args.wp_max:g}_xi{args.xi_min:g}-{args.xi_max:g}_ells{args.ells.replace(',', '')}"
    cov_path = output_dir / f"{tag}_covariance.npy"
    bins_path = output_dir / f"{tag}_covariance_bins.csv"
    summary_path = output_dir / f"{tag}_covariance_summary.json"
    np.save(cov_path, cov)
    write_rows(bins_path, rows)
    summary = {
        "label": args.label,
        "input_dir": str(input_dir),
        "output_covariance": str(cov_path),
        "output_bins": str(bins_path),
        "shape": list(cov.shape),
        "n_wp": int(n_wp),
        "n_xi": int(n_xi),
        "ells": ells,
        "scale_cuts": {
            "wp_min": args.wp_min,
            "wp_max": args.wp_max,
            "xi_min": args.xi_min,
            "xi_max": args.xi_max,
        },
        "warning": "This combines pycorr jackknife wp and poles covariances as block diagonal; cross-covariance between wp and xi is not included unless a joint jackknife-vector builder is used.",
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(cov_path)
    print(bins_path)
    print(summary_path)
    print("covariance shape", cov.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
