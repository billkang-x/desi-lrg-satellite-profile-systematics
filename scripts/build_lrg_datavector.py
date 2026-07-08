"""Build a fitting-ready DESI LRG data vector from pycorr outputs.

The produced vector is intentionally simple: one row per observable bin, with
explicit component labels and scale cuts. It is meant to be the stable contract
between the observed DESI measurement and HOD/SHAM mock predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_LABEL = "LRG_DR1_NGC-SGC_z0.600-0.800_r0"


def parse_ells(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("At least one ell must be requested")
    return values


def read_named_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if table.ndim == 0:
        table = np.asarray([table], dtype=table.dtype)
    return table


def finite_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    kept = []
    rejected = []
    for row in rows:
        value = float(row["value"])
        radius = float(row["r"])
        if np.isfinite(value) and np.isfinite(radius):
            kept.append(row)
        else:
            rejected.append(row)
    return kept, rejected


def collect_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    label = args.label
    input_dir = Path(args.input_dir)
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "label": label,
        "input_dir": str(input_dir),
        "scale_cuts": {
            "wp_min": args.wp_min,
            "wp_max": args.wp_max,
            "xi_min": args.xi_min,
            "xi_max": args.xi_max,
        },
        "ells": parse_ells(args.ells),
    }

    if not args.no_wp:
        wp_path = input_dir / f"{label}_wp.csv"
        wp = read_named_csv(wp_path)
        mask = (wp["rp_Mpch"] >= args.wp_min) & (wp["rp_Mpch"] <= args.wp_max)
        for radius, value in zip(wp["rp_Mpch"][mask], wp["wp"][mask]):
            rows.append(
                {
                    "component": "wp",
                    "statistic": "wp",
                    "ell": -1,
                    "r": float(radius),
                    "value": float(value),
                    "source_column": "wp",
                    "source_file": wp_path.name,
                }
            )
        summary["wp_bins_available"] = int(len(wp))
        summary["wp_bins_kept"] = int(np.count_nonzero(mask))

    poles_path = input_dir / f"{label}_poles.csv"
    poles = read_named_csv(poles_path)
    xi_mask = (poles["s_Mpch"] >= args.xi_min) & (poles["s_Mpch"] <= args.xi_max)
    for ell in parse_ells(args.ells):
        column = f"xi{ell}"
        if column not in poles.dtype.names:
            raise ValueError(f"Missing column {column} in {poles_path}")
        for radius, value in zip(poles["s_Mpch"][xi_mask], poles[column][xi_mask]):
            rows.append(
                {
                    "component": column,
                    "statistic": "xi",
                    "ell": int(ell),
                    "r": float(radius),
                    "value": float(value),
                    "source_column": column,
                    "source_file": poles_path.name,
                }
            )
    summary["xi_bins_available"] = int(len(poles))
    summary["xi_bins_kept_per_ell"] = int(np.count_nonzero(xi_mask))

    rows, rejected = finite_rows(rows)
    summary["n_rejected_nonfinite"] = len(rejected)
    summary["n_data"] = len(rows)
    if rejected and not args.allow_nonfinite:
        raise ValueError(f"Rejected {len(rejected)} non-finite rows; use --allow-nonfinite to continue")
    if args.allow_nonfinite:
        rows.extend(rejected)
    return rows, summary


def write_vector_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "index",
        "component",
        "statistic",
        "ell",
        "r",
        "value",
        "source_column",
        "source_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = dict(row)
            output["index"] = index
            writer.writerow(output)


def write_model_template(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["index", "component", "statistic", "ell", "r", "model_value"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "index": index,
                    "component": row["component"],
                    "statistic": row["statistic"],
                    "ell": row["ell"],
                    "r": row["r"],
                    "model_value": "",
                }
            )


def write_npz(path: Path, rows: list[dict[str, object]]) -> None:
    np.savez(
        path,
        value=np.asarray([row["value"] for row in rows], dtype="f8"),
        r=np.asarray([row["r"] for row in rows], dtype="f8"),
        ell=np.asarray([row["ell"] for row in rows], dtype="i4"),
        component=np.asarray([row["component"] for row in rows]),
        statistic=np.asarray([row["statistic"] for row in rows]),
    )


def load_observation_metadata(input_dir: Path, label: str) -> dict[str, object]:
    path = input_dir / f"{label}_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/pycorr_lrg")
    parser.add_argument("--output-dir", default="results/datavectors")
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--ells", default="0,2")
    parser.add_argument("--wp-min", type=float, default=0.5)
    parser.add_argument("--wp-max", type=float, default=30.0)
    parser.add_argument("--xi-min", type=float, default=5.0)
    parser.add_argument("--xi-max", type=float, default=60.0)
    parser.add_argument("--no-wp", action="store_true")
    parser.add_argument("--allow-nonfinite", action="store_true")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = collect_rows(args)
    observation_metadata = load_observation_metadata(input_dir, args.label)
    summary["observation_metadata"] = {
        "created_utc": observation_metadata.get("created_utc"),
        "data_summary": observation_metadata.get("data_summary", []),
        "random_summary": observation_metadata.get("random_summary", []),
        "measurement_args": observation_metadata.get("args", {}),
    }

    suffix = f"{args.label}_wp{args.wp_min:g}-{args.wp_max:g}_xi{args.xi_min:g}-{args.xi_max:g}_ells{args.ells.replace(',', '')}"
    csv_path = output_dir / f"{suffix}_datavector.csv"
    npz_path = output_dir / f"{suffix}_datavector.npz"
    json_path = output_dir / f"{suffix}_summary.json"
    template_path = output_dir / f"{suffix}_model_template.csv"

    write_vector_csv(csv_path, rows)
    write_npz(npz_path, rows)
    write_model_template(template_path, rows)
    summary["outputs"] = {
        "datavector_csv": str(csv_path),
        "datavector_npz": str(npz_path),
        "summary_json": str(json_path),
        "model_template_csv": str(template_path),
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(f"Wrote {len(rows)} rows")
    print(csv_path)
    print(npz_path)
    print(template_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
