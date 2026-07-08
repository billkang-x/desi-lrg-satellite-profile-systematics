"""Summarize full-box Abacus f_sat/alpha_s grid results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\研究方向\desi_hod_mocks")
OUT = Path(r"D:\研究方向\outputs")


RESULTS = [
    ("heuristic", ROOT / "results/abacus_fullbox_fsat_alpha_heuristic_v2/abacus_fullbox_fsat_alpha_grid.csv", "chi2_total"),
    (
        "jk96_diagonal",
        ROOT / "results/abacus_fullbox_fsat_alpha_covjk96_diag_rescore_v2/abacus_fullbox_fsat_alpha_grid.csv",
        "chi2_total_cov",
    ),
    (
        "jk96_full",
        ROOT / "results/abacus_fullbox_fsat_alpha_covjk96_full_rescore_v2/abacus_fullbox_fsat_alpha_grid.csv",
        "chi2_total_cov",
    ),
]


LITERATURE = [
    {
        "sample": "DESI One-Percent LRG 0.6<z<0.8, AbacusHOD RSD fit",
        "f_sat": "0.136 +/- 0.011",
        "alpha_s": "0.95 +0.07/-0.06",
        "notes": "Same redshift range is the closest external comparison; full-shape HOD fit to DESI One-Percent clustering.",
        "source": "https://academic.oup.com/mnras/article/530/1/947/7643636",
    },
    {
        "sample": "DESI One-Percent LRG 0.6<z<0.8, AbacusHOD wp-only",
        "f_sat": "0.104 +0.023/-0.019",
        "alpha_s": "not constrained by wp-only",
        "notes": "Useful because wp heavily constrains one-halo satellite contribution.",
        "source": "https://academic.oup.com/mnras/article/530/1/947/7643636",
    },
    {
        "sample": "BOSS CMASS, AbacusHOD baseline",
        "f_sat": "0.11",
        "alpha_s": "0.98",
        "notes": "CMASS HOD benchmark with full-shape clustering; extensions give f_sat about 0.13-0.15.",
        "source": "https://academic.oup.com/mnras/article/510/3/3301/6446006",
    },
    {
        "sample": "BOSS CMASS, AbacusHOD decorated/profile variants",
        "f_sat": "0.13-0.15",
        "alpha_s": "0.84-1.00",
        "notes": "Shows model extensions can move satellite fraction upward.",
        "source": "https://academic.oup.com/mnras/article/510/3/3301/6446006",
    },
    {
        "sample": "BOSS+eBOSS LRG 0.6<z<1.0, HOD clustering",
        "f_sat": "0.13 +/- 0.03",
        "alpha_s": "not quoted in that HOD summary",
        "notes": "Broader redshift sample; useful external LRG HOD scale.",
        "source": "https://arxiv.org/abs/1607.05383",
    },
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def summarize_grid(label: str, path: Path, score_key: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    rows = read_rows(path)
    rows.sort(key=lambda row: as_float(row, score_key))
    best = rows[0]
    best_score = as_float(best, score_key)
    summary = {
        "score": label,
        "score_column": score_key,
        "best_f_sat": best["f_sat_target"],
        "best_alpha_s": best["alpha_s"],
        "best_f_sat_actual": best["f_sat_actual"],
        "best_chi2": f"{best_score:.6g}",
        "delta_chi2_to_second": f"{as_float(rows[1], score_key) - best_score:.6g}",
        "chi2_xi0": best["chi2_xi0"],
        "chi2_xi2": best["chi2_xi2"],
        "chi2_wp": best["chi2_wp"],
        "n_gal": best["n_gal"],
        "n_sat": best["n_sat"],
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmaps(all_rows: dict[str, tuple[str, list[dict[str, str]]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    for ax, (label, (score_key, rows)) in zip(axes, all_rows.items()):
        fsats = sorted({as_float(row, "f_sat_target") for row in rows})
        alphas = sorted({as_float(row, "alpha_s") for row in rows})
        grid = np.full((len(fsats), len(alphas)), np.nan)
        for row in rows:
            i = fsats.index(as_float(row, "f_sat_target"))
            j = alphas.index(as_float(row, "alpha_s"))
            grid[i, j] = as_float(row, score_key)
        delta = grid - np.nanmin(grid)
        image = ax.imshow(delta, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(label)
        ax.set_xlabel(r"$\alpha_s$")
        ax.set_ylabel(r"$f_{\rm sat}$")
        ax.set_xticks(range(len(alphas)), [f"{value:.1f}" for value in alphas])
        ax.set_yticks(range(len(fsats)), [f"{value:.3f}" for value in fsats])
        for i in range(len(fsats)):
            for j in range(len(alphas)):
                ax.text(j, i, f"{delta[i, j]:.1f}", ha="center", va="center", color="white", fontsize=7)
        fig.colorbar(image, ax=ax, label=r"$\Delta\chi^2$")
    fig.savefig(OUT / "abacus_fullbox_fsat_alpha_heatmaps.png", dpi=220)
    plt.close(fig)


def write_report(summaries: list[dict[str, str]]) -> None:
    summary_lines = "\n".join(
        f"| {row['score']} | {row['best_f_sat']} | {row['best_alpha_s']} | {row['best_chi2']} | {row['delta_chi2_to_second']} |"
        for row in summaries
    )
    lit_lines = "\n".join(
        f"| {row['sample']} | {row['f_sat']} | {row['alpha_s']} | {row['notes']} | {row['source']} |"
        for row in LITERATURE
    )
    text = f"""# Abacus full-box HOD/SHAM grid status

Date: 2026-07-08

## Inputs

- Simulation input: AbacusSummit_base_c000_ph000, z=0.8, full periodic 2000 Mpc/h box, 34 halo slabs.
- Compact catalog: 4,252,712 target galaxies at nbar=5.31589e-4 h^3 Mpc^-3, plus a 250,000-object satellite reservoir.
- Grid: f_sat = 0.015, 0.020, 0.025, 0.030, 0.035; alpha_s = 0.7, 0.8, 0.9, 1.0, 1.1.
- Data vector: DESI DR1 LRG 0.6<z<0.8, unreconstructed wp(0.5-30) + xi0/xi2(5-60), matched to a 72-element vector.
- Covariance: local kmeans96 jackknife covariance for exactly the same wp + xi0 + xi2 vector, with Hartlap-like precision scaling 22/95 = 0.231578947. This is a useful robustness test, not yet a final mock covariance.

## Grid results

| score | best f_sat | best alpha_s | best chi2 | delta chi2 to second |
|---|---:|---:|---:|---:|
{summary_lines}

## External HOD comparison

| sample | literature f_sat | literature alpha_s | notes | source |
|---|---:|---:|---|---|
{lit_lines}

## Interpretation

1. The full-box result is no longer the very low f_sat suggested by the early L500 pilot. Once the full 2000 Mpc/h box is used, the preferred range across reasonable scoring choices is f_sat about 0.025-0.035.
2. The upper edge of the tested f_sat range is still favored by the heuristic and full-covariance scores, so the next grid should extend to at least f_sat=0.06, and preferably to 0.10 for a direct DESI/BOSS comparison.
3. Even after moving upward, the current f_sat remains lower than DESI One-Percent and BOSS/eBOSS HOD values, which are typically about 0.10-0.15 for LRG-like samples.
4. The most likely reason is model construction, not necessarily galaxy physics: this pilot uses an abundance-ranked central+satellite reservoir built from halo_info plus subsample-A particles, not the full AbacusHOD occupation model with satellite profile, assembly bias, cleaning, and full DESI selection treatment.
5. The covariance dependence is scientifically useful. Diagonal covariance prefers f_sat=0.025 and high alpha_s, while full covariance prefers higher f_sat and lower alpha_s, indicating a degeneracy between satellite fraction, FoG/RSD shape, and covariance weighting.

## Immediate next steps

- Extend the full-box grid upward: f_sat = 0.02-0.10, alpha_s = 0.5-1.2.
- Add a genuine HOD occupation layer rather than treating the satellite reservoir as a fixed abundance-ranked SHAM-like component.
- Replace or cross-check the kmeans96 jackknife covariance with EZmock/Abacus mock covariance if available.
- Separate fits for wp-only, xi0+xi2-only, and joint wp+xi0+xi2 to diagnose why the covariance weights pull f_sat differently.
"""
    (OUT / "abacus_fullbox_current_conclusions.md").write_text(text, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_rows = {}
    for label, path, score_key in RESULTS:
        summary, rows = summarize_grid(label, path, score_key)
        summaries.append(summary)
        all_rows[label] = (score_key, rows)
    write_csv(OUT / "abacus_fullbox_fsat_alpha_summary.csv", summaries)
    write_csv(OUT / "abacus_fullbox_literature_comparison.csv", LITERATURE)
    plot_heatmaps(all_rows)
    write_report(summaries)
    for row in summaries:
        print(row)
    print(OUT / "abacus_fullbox_fsat_alpha_heatmaps.png")
    print(OUT / "abacus_fullbox_current_conclusions.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
