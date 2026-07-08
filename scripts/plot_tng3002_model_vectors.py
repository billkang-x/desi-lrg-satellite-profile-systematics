"""Make a quick paper-style preview plot from exported TNG300-2 model vectors."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("D:/研究方向/tmp/mplconfig")).resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def model_sort_key(rows: list[dict[str, str]]) -> tuple[int, int, int, str]:
    row = rows[0]
    ranks = []
    for column in ["heuristic_rank", "diagcov_rank", "fullcov_rank"]:
        value = row.get(column, "")
        ranks.append(int(float(value)) if value else 999)
    return (min(ranks), ranks[0], ranks[1], row["model_id"])


def choose_models(rows: list[dict[str, str]], max_models: int) -> list[str]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model_id"]].append(row)
    return [group[0]["model_id"] for group in sorted(grouped.values(), key=model_sort_key)[:max_models]]


def as_float(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in rows], dtype="f8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/tng300_pilot_model_vectors_TNG300-2_x5/tng3002_paper_plot_selected_long.csv")
    parser.add_argument("--output-prefix", default="results/tng300_pilot_model_vectors_TNG300-2_x5/figures/tng3002_wp_xi_model_vectors_preview")
    parser.add_argument("--max-models", type=int, default=6)
    args = parser.parse_args(argv)
    rows = read_rows(Path(args.input))
    models = choose_models(rows, args.max_models)
    components = ["wp", "xi0", "xi2"]
    titles = {"wp": r"$w_p(r_p)$", "xi0": r"$\xi_0(s)$", "xi2": r"$\xi_2(s)$"}
    xlabels = {"wp": r"$r_p\,[h^{-1}{\rm Mpc}]$", "xi0": r"$s\,[h^{-1}{\rm Mpc}]$", "xi2": r"$s\,[h^{-1}{\rm Mpc}]$"}
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), sharex="col", gridspec_kw={"height_ratios": [2.2, 1.0]})
    for col, component in enumerate(components):
        component_rows = [row for row in rows if row["component"] == component]
        data_rows = [row for row in component_rows if row["model_id"] == models[0]]
        x = as_float(data_rows, "r")
        y = as_float(data_rows, "data_value")
        err = as_float(data_rows, "sigma_jk")
        ax = axes[0, col]
        rax = axes[1, col]
        ax.errorbar(x, y, yerr=err, fmt="o", ms=3.0, color="black", ecolor="0.65", elinewidth=0.8, label="DESI DR1 LRG")
        rax.axhline(0.0, color="0.25", lw=0.8)
        for color, model in zip(colors, models):
            model_rows = [row for row in component_rows if row["model_id"] == model]
            model_rows.sort(key=lambda row: int(row["index"]))
            mx = as_float(model_rows, "r")
            my = as_float(model_rows, "model_value")
            frac = as_float(model_rows, "fractional_residual")
            ax.plot(mx, my, lw=1.5, color=color, label=model_rows[0]["model_label"])
            rax.plot(mx, frac, lw=1.3, color=color)
        ax.set_xscale("log")
        rax.set_xscale("log")
        if component == "wp":
            ax.set_yscale("log")
            ticks = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        else:
            ticks = [5.0, 10.0, 20.0, 40.0, 60.0]
        for target_ax in [ax, rax]:
            target_ax.xaxis.set_major_locator(FixedLocator(ticks))
            target_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            target_ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(titles[component])
        ax.set_ylabel("observable")
        rax.set_ylabel("(model-data)/data")
        rax.set_xlabel(xlabels[component])
        ax.grid(alpha=0.2)
        rax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7.5, loc="best", frameon=False)
    fig.tight_layout()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
