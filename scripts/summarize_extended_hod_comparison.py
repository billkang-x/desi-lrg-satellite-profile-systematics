"""Summarize the extended SHAM vs minimal-HOD comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\研究方向\desi_hod_mocks")
OUT = Path(r"D:\研究方向\outputs")


RUNS = [
    (
        "SHAM-like reservoir",
        ROOT / "results/abacus_fullbox_sham_fsat0p15_alpha_v1/abacus_fullbox_fsat_alpha_grid.csv",
        "chi2_total",
    ),
    (
        "minimal HOD occupation",
        ROOT / "results/abacus_fullbox_minimal_hod_fsat0p15_alpha_v1/abacus_fullbox_minimal_hod_fsat_alpha_grid.csv",
        "chi2_total",
    ),
]

SPLIT_SUMMARIES = [
    ROOT / "results/abacus_fullbox_observable_splits_v1/sham_extended_observable_split_summary.csv",
    ROOT / "results/abacus_fullbox_observable_splits_v1/minimal_hod_observable_split_summary.csv",
]

LITERATURE = [
    {
        "label": "DESI One-Percent LRG RSD AbacusHOD",
        "f_sat": "0.136 +/- 0.011",
        "alpha_s": "0.95 +0.07/-0.06",
        "source": "https://academic.oup.com/mnras/article/530/1/947/7643636",
    },
    {
        "label": "DESI One-Percent LRG wp-only AbacusHOD",
        "f_sat": "0.104 +0.023/-0.019",
        "alpha_s": "not constrained",
        "source": "https://academic.oup.com/mnras/article/530/1/947/7643636",
    },
    {
        "label": "BOSS CMASS AbacusHOD baseline",
        "f_sat": "0.11",
        "alpha_s": "0.98",
        "source": "https://academic.oup.com/mnras/article/510/3/3301/6446006",
    },
    {
        "label": "BOSS/eBOSS LRG HOD",
        "f_sat": "0.13 +/- 0.03",
        "alpha_s": "not quoted",
        "source": "https://arxiv.org/abs/1607.05383",
    },
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def best_rows() -> list[dict[str, object]]:
    out = []
    for label, path, score_key in RUNS:
        rows = [row for row in read_rows(path) if row.get("status", "ok") == "ok"]
        rows.sort(key=lambda row: as_float(row, score_key))
        best = rows[0]
        second = rows[1]
        out.append(
            {
                "comparison_group": label,
                "score": score_key,
                "best_f_sat": best["f_sat_target"],
                "best_alpha_s": best["alpha_s"],
                "best_f_sat_actual": best["f_sat_actual"],
                "best_chi2": f"{as_float(best, score_key):.6g}",
                "delta_chi2_to_second": f"{as_float(second, score_key) - as_float(best, score_key):.6g}",
                "chi2_wp": best["chi2_wp"],
                "chi2_xi0": best["chi2_xi0"],
                "chi2_xi2": best["chi2_xi2"],
            }
        )
    for row in LITERATURE:
        out.append(
            {
                "comparison_group": row["label"],
                "score": "literature",
                "best_f_sat": row["f_sat"],
                "best_alpha_s": row["alpha_s"],
                "best_f_sat_actual": "",
                "best_chi2": "",
                "delta_chi2_to_second": "",
                "chi2_wp": "",
                "chi2_xi0": "",
                "chi2_xi2": "",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmaps() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
    for ax, (label, path, score_key) in zip(axes, RUNS):
        rows = [row for row in read_rows(path) if row.get("status", "ok") == "ok"]
        fsats = sorted({as_float(row, "f_sat_target") for row in rows})
        alphas = sorted({as_float(row, "alpha_s") for row in rows})
        grid = np.full((len(fsats), len(alphas)), np.nan)
        for row in rows:
            i = fsats.index(as_float(row, "f_sat_target"))
            j = alphas.index(as_float(row, "alpha_s"))
            grid[i, j] = as_float(row, score_key)
        delta = grid - np.nanmin(grid)
        image = ax.imshow(delta, origin="lower", aspect="auto", cmap="magma_r")
        ax.set_title(label)
        ax.set_xlabel(r"$\alpha_s$")
        ax.set_ylabel(r"$f_{\rm sat}$")
        ax.set_xticks(range(len(alphas)), [f"{value:.1f}" for value in alphas])
        ax.set_yticks(range(len(fsats)), [f"{value:.2f}" for value in fsats])
        for i in range(len(fsats)):
            for j in range(len(alphas)):
                color = "white" if delta[i, j] > 0.55 * np.nanmax(delta) else "black"
                ax.text(j, i, f"{delta[i, j]:.0f}", ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(image, ax=ax, label=r"$\Delta\chi^2_{\rm heuristic}$")
    fig.savefig(OUT / "abacus_extended_sham_hod_heatmaps.png", dpi=220)
    plt.close(fig)


def plot_wp_chi2() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    for label, path, _ in RUNS:
        rows = [row for row in read_rows(path) if row.get("status", "ok") == "ok"]
        fsats = sorted({as_float(row, "f_sat_target") for row in rows})
        best_wp = []
        for f_sat in fsats:
            subset = [row for row in rows if as_float(row, "f_sat_target") == f_sat]
            best_wp.append(min(as_float(row, "chi2_wp") for row in subset))
        ax.plot(fsats, best_wp, marker="o", label=label)
    ax.set_xlabel(r"$f_{\rm sat}$")
    ax.set_ylabel(r"best heuristic $\chi^2(w_p)$")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.savefig(OUT / "abacus_extended_wp_chi2_vs_fsat.png", dpi=220)
    plt.close(fig)


def combined_split_summary() -> list[dict[str, str]]:
    rows = []
    for path in SPLIT_SUMMARIES:
        rows.extend(read_rows(path))
    return rows


def write_report(best: list[dict[str, object]], split_rows: list[dict[str, str]]) -> None:
    best_table = "\n".join(
        f"| {row['comparison_group']} | {row['best_f_sat']} | {row['best_alpha_s']} | {row['best_chi2']} | {row['chi2_wp']} | {row['chi2_xi0']} | {row['chi2_xi2']} |"
        for row in best
    )
    split_table = "\n".join(
        f"| {row['label']} | {row['split']} | {row['covariance_mode']} | {row['best_f_sat']} | {row['best_alpha_s']} | {float(row['best_chi2']):.3g} | {float(row['delta_chi2_to_second']):.3g} |"
        for row in split_rows
    )
    text = f"""# Extended SHAM vs minimal-HOD comparison

Date: 2026-07-08

## What was run

- Simulation: AbacusSummit_base_c000_ph000, z=0.8, full 2000 Mpc/h periodic box.
- New compact reservoir: 4,252,712 central candidates plus 800,000 satellite candidates, supporting f_sat up to 0.15.
- Grid: f_sat = 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15; alpha_s = 0.5, 0.7, 0.9, 1.1, 1.3.
- Comparison groups:
  - SHAM-like reservoir: fixed satellite fraction chosen by rank from the satellite reservoir.
  - minimal HOD occupation: noisy central mass-rank selection plus host-mass-weighted satellite occupation.
  - literature: DESI One-Percent / BOSS / eBOSS HOD values.

## Joint heuristic best fits

| group | best f_sat | best alpha_s | best chi2 | chi2_wp | chi2_xi0 | chi2_xi2 |
|---|---:|---:|---:|---:|---:|---:|
{best_table}

## Observable-split best fits using kmeans96 covariance

| model | split | covariance | best f_sat | best alpha_s | best chi2 | delta chi2 to second |
|---|---|---|---:|---:|---:|---:|
{split_table}

## Interpretation

1. Extending f_sat to the literature range does not move the current best fit toward 0.10-0.15. Both SHAM-like and minimal-HOD variants still prefer low f_sat in the joint fits.
2. The reason is now clear: wp strongly penalizes high f_sat. In the SHAM-like model, the best heuristic chi2_wp rises from about 214 at f_sat=0.02 to about 5885 at f_sat=0.15. In the minimal-HOD model, wp improves substantially but still rises from about 52 at f_sat=0.02 to about 4209 at f_sat=0.15.
3. Minimal HOD is valuable: it lowers the joint heuristic best chi2 from 372 to 283 and shifts the heuristic best from f_sat=0.02 to f_sat=0.04. This demonstrates that the satellite prescription matters strongly.
4. The observable split shows the tension structure. wp-only prefers f_sat=0.02 for both models, while xi0+xi2 can prefer higher f_sat, especially minimal HOD with full covariance preferring f_sat=0.06. The joint result is therefore dominated by the one-halo/small-scale wp constraint.
5. Compared with literature f_sat around 0.10-0.15, the remaining gap is likely not solved by simply adding more satellites. It points to missing ingredients in the satellite profile/selection/covariance treatment: real AbacusHOD satellite profile freedoms, DESI selection/cutsky effects, and mock covariance.

## Most useful next move

Do not keep expanding f_sat blindly. The next highest-value test is a satellite-profile dilution test: vary the satellite radial/velocity proxy so that high-f_sat models do not overproduce small-scale wp. With our data, this can be done by drawing satellites from lower-rank particle radii, adding radial dilution, or downweighting central halo cores before rerunning only a compact subset around f_sat=0.04-0.12.

## Output files

- abacus_extended_sham_hod_literature_summary.csv
- abacus_extended_observable_split_summary.csv
- abacus_extended_sham_hod_heatmaps.png
- abacus_extended_wp_chi2_vs_fsat.png
"""
    (OUT / "abacus_extended_sham_hod_comparison.md").write_text(text, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    best = best_rows()
    split_rows = combined_split_summary()
    write_csv(OUT / "abacus_extended_sham_hod_literature_summary.csv", best)
    write_csv(OUT / "abacus_extended_observable_split_summary.csv", split_rows)
    plot_heatmaps()
    plot_wp_chi2()
    write_report(best, split_rows)
    for row in best:
        print(row)
    print(OUT / "abacus_extended_sham_hod_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
