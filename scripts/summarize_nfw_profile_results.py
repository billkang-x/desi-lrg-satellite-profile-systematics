"""Summarize the NFW satellite-profile concentration test."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit_abacus_nfw_profile_pycorr import model_filename


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "outputs"
MODEL_DIR = ROOT / "results/abacus_fullbox_minimal_hod_nfw_profile_v1"
GRID_CSV = MODEL_DIR / "abacus_fullbox_minimal_hod_nfw_profile_grid.csv"
SPLIT_CSV = (
    ROOT
    / "results/abacus_fullbox_nfw_profile_splits_v1/minimal_hod_nfw_profile_nfw_profile_split_summary.csv"
)
DATAVECTOR = (
    ROOT
    / "results/datavectors/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp0.5-30_xi5-60_ells02_datavector.csv"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def ok_rows() -> list[dict[str, str]]:
    return [row for row in read_rows(GRID_CSV) if row.get("status", "ok") == "ok"]


def best_row(rows: list[dict[str, str]], score_key: str = "chi2_total") -> dict[str, str]:
    return min(rows, key=lambda row: as_float(row, score_key))


def read_observed() -> dict[str, np.ndarray]:
    table = np.genfromtxt(DATAVECTOR, delimiter=",", names=True, dtype=None, encoding="utf-8")
    component = np.asarray(table["component"])
    values = np.asarray(table["value"], dtype="f8")
    radii = np.asarray(table["r"], dtype="f8")
    return {
        "rp": radii[component == "wp"],
        "wp": values[component == "wp"],
        "s": radii[component == "xi0"],
        "xi0": values[component == "xi0"],
        "xi2": values[component == "xi2"],
    }


def summary_row(criterion: str, row: dict[str, str]) -> dict[str, object]:
    return {
        "model": "minimal HOD NFW profile",
        "criterion": criterion,
        "f_sat_target": f"{as_float(row, 'f_sat_target'):.3f}",
        "f_sat_actual": f"{as_float(row, 'f_sat_actual'):.6f}",
        "alpha_s": f"{as_float(row, 'alpha_s'):.3f}",
        "concentration_ratio": f"{as_float(row, 'concentration_ratio'):.3f}",
        "c_sat": f"{as_float(row, 'c_sat'):.3f}",
        "radius_scale_median": f"{as_float(row, 'sat_radius_scale_median'):.3f}",
        "chi2_total": f"{as_float(row, 'chi2_total'):.6g}",
        "chi2_wp": f"{as_float(row, 'chi2_wp'):.6g}",
        "chi2_xi0": f"{as_float(row, 'chi2_xi0'):.6g}",
        "chi2_xi2": f"{as_float(row, 'chi2_xi2'):.6g}",
    }


def build_best_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = [
        summary_row("best_total_all_c", best_row(rows)),
        summary_row(
            "best_total_c_ratio_1",
            best_row([row for row in rows if np.isclose(as_float(row, "concentration_ratio"), 1.0)]),
        ),
        summary_row("best_wp_all_c", best_row(rows, "chi2_wp")),
    ]
    for f_sat in (0.08, 0.10, 0.12):
        subset = [row for row in rows if np.isclose(as_float(row, "f_sat_target"), f_sat)]
        out.append(summary_row(f"best_total_fsat{f_sat:.2f}", best_row(subset)))
    return out


def min_grid(rows: list[dict[str, str]], score_key: str) -> tuple[list[float], list[float], np.ndarray]:
    fsats = sorted({as_float(row, "f_sat_target") for row in rows})
    cratios = sorted({as_float(row, "concentration_ratio") for row in rows})
    grid = np.full((len(fsats), len(cratios)), np.nan)
    for i, f_sat in enumerate(fsats):
        for j, c_ratio in enumerate(cratios):
            subset = [
                row
                for row in rows
                if np.isclose(as_float(row, "f_sat_target"), f_sat)
                and np.isclose(as_float(row, "concentration_ratio"), c_ratio)
            ]
            if subset:
                grid[i, j] = min(as_float(row, score_key) for row in subset)
    return fsats, cratios, grid


def plot_heatmap(rows: list[dict[str, str]]) -> None:
    fsats, cratios, grid = min_grid(rows, "chi2_total")
    delta = grid - np.nanmin(grid)
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    image = ax.imshow(delta, origin="lower", aspect="auto", cmap="magma_r")
    ax.set_xlabel(r"$c_{\rm sat}/c_{\rm ref}$")
    ax.set_ylabel(r"$f_{\rm sat}$")
    ax.set_xticks(range(len(cratios)), [f"{value:.2f}" for value in cratios])
    ax.set_yticks(range(len(fsats)), [f"{value:.2f}" for value in fsats])
    ax.set_title("minimal HOD: NFW concentration remapping")
    vmax = np.nanmax(delta)
    for i in range(len(fsats)):
        for j in range(len(cratios)):
            color = "white" if vmax > 0 and delta[i, j] > 0.55 * vmax else "black"
            ax.text(j, i, f"{delta[i, j]:.0f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=ax, label=r"min over $\alpha_s$: $\Delta\chi^2_{\rm heuristic}$")
    fig.savefig(OUT / "abacus_nfw_profile_heatmap.png", dpi=220)
    fig.savefig(ROOT / "paper/figures/fig_nfw_profile_heatmap.png", dpi=220)
    plt.close(fig)


def plot_wp_curves(rows: list[dict[str, str]]) -> None:
    observed = read_observed()
    selected = [
        ("baseline c=1", best_row([row for row in rows if np.isclose(as_float(row, "concentration_ratio"), 1.0)])),
        ("best c free", best_row(rows)),
        ("high-fsat example", best_row([row for row in rows if np.isclose(as_float(row, "f_sat_target"), 0.10)])),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    ax.plot(observed["rp"], observed["wp"], color="black", marker="o", lw=1.5, label="DESI DR1 LRG")
    for label, row in selected:
        payload = np.load(MODEL_DIR / model_filename(as_float(row, "f_sat_target"), as_float(row, "alpha_s"), as_float(row, "concentration_ratio")))
        full_label = (
            f"{label}: f={as_float(row, 'f_sat_target'):.2f}, "
            f"a={as_float(row, 'alpha_s'):.1f}, c/c0={as_float(row, 'concentration_ratio'):.2f}"
        )
        ax.plot(payload["obs_rp"], payload["model_wp"], marker="s", ms=3, lw=1.2, label=full_label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r_p\ [h^{-1}{\rm Mpc}]$")
    ax.set_ylabel(r"$w_p(r_p)$")
    ax.legend(frameon=False, fontsize=7.0)
    fig.savefig(OUT / "abacus_nfw_profile_wp_curves.png", dpi=220)
    fig.savefig(ROOT / "paper/figures/fig_nfw_profile_wp_curves.png", dpi=220)
    plt.close(fig)


def combined_split_summary() -> list[dict[str, object]]:
    rows = read_rows(SPLIT_CSV)
    out = []
    for row in rows:
        out.append(
            {
                "model": row["label"],
                "split": row["split"],
                "covariance_mode": row["covariance_mode"],
                "best_f_sat": f"{as_float(row, 'best_f_sat'):.3f}",
                "best_alpha_s": f"{as_float(row, 'best_alpha_s'):.3f}",
                "best_concentration_ratio": f"{as_float(row, 'best_concentration_ratio'):.3f}",
                "best_c_sat": f"{as_float(row, 'best_c_sat'):.3f}",
                "best_f_sat_actual": f"{as_float(row, 'best_f_sat_actual'):.6f}",
                "best_chi2": f"{as_float(row, 'best_chi2'):.6g}",
                "delta_chi2_to_second": f"{as_float(row, 'delta_chi2_to_second'):.6g}",
                "hartlap_scale": f"{as_float(row, 'hartlap_scale'):.6g}",
            }
        )
    return out


def write_report(best: list[dict[str, object]], splits: list[dict[str, object]]) -> None:
    best_lines = "\n".join(
        f"| {row['criterion']} | {row['f_sat_target']} | {row['alpha_s']} | {row['concentration_ratio']} | "
        f"{row['c_sat']} | {row['radius_scale_median']} | {row['chi2_total']} | {row['chi2_wp']} | "
        f"{row['chi2_xi0']} | {row['chi2_xi2']} |"
        for row in best
    )
    split_lines = "\n".join(
        f"| {row['split']} | {row['covariance_mode']} | {row['best_f_sat']} | {row['best_alpha_s']} | "
        f"{row['best_concentration_ratio']} | {row['best_chi2']} | {row['delta_chi2_to_second']} |"
        for row in splits
    )
    text = f"""# NFW Satellite Profile Control

Date: 2026-07-08

## Heuristic best fits

| criterion | f_sat | alpha_s | c_ratio | c_sat | median radius scale | chi2_total | chi2_wp | chi2_xi0 | chi2_xi2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{best_lines}

## Covariance split best fits

| split | covariance | f_sat | alpha_s | c_ratio | chi2 | delta chi2 to second |
|---|---|---:|---:|---:|---:|---:|
{split_lines}

## Interpretation

The NFW concentration remapping is more physical than the uniform q-scaling, but its radial changes are also milder.  It tests whether ordinary changes in satellite concentration can explain the low-f_sat preference.  In this grid, high f_sat remains strongly disfavoured.  Therefore the q-scaling result should be interpreted as a sensitivity diagnostic, not as evidence that a standard NFW concentration shift alone solves the tension.
"""
    (OUT / "abacus_nfw_profile_report.md").write_text(text, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = ok_rows()
    best = build_best_summary(rows)
    splits = combined_split_summary()
    write_rows(OUT / "abacus_nfw_profile_best_summary.csv", best)
    write_rows(OUT / "abacus_nfw_profile_split_summary.csv", splits)
    plot_heatmap(rows)
    plot_wp_curves(rows)
    write_report(best, splits)
    print(f"Wrote NFW-profile summary products to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
