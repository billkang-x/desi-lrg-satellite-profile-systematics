"""Summarize the Abacus satellite-profile dilution test."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "outputs"
PAPER_FIGURES = ROOT / "paper/figures"

RUNS = {
    "minimal_hod_profile": {
        "label": "minimal HOD",
        "dir": ROOT / "results/abacus_fullbox_minimal_hod_profile_dilution_v1",
        "grid": ROOT
        / "results/abacus_fullbox_minimal_hod_profile_dilution_v1/abacus_fullbox_minimal_hod_profile_dilution_grid.csv",
    },
    "sham_profile": {
        "label": "SHAM-like",
        "dir": ROOT / "results/abacus_fullbox_sham_profile_dilution_v1",
        "grid": ROOT
        / "results/abacus_fullbox_sham_profile_dilution_v1/abacus_fullbox_sham_profile_dilution_grid.csv",
    },
}

SPLIT_SUMMARIES = [
    ROOT
    / "results/abacus_fullbox_profile_dilution_splits_v1/minimal_hod_profile_profile_dilution_split_summary.csv",
    ROOT / "results/abacus_fullbox_profile_dilution_splits_v1/sham_profile_profile_dilution_split_summary.csv",
]

DATAVECTOR = (
    ROOT
    / "results/datavectors/LRG_DR1_NGC-SGC_z0.600-0.800_r0-1-2-3_wp0.5-30_xi5-60_ells02_datavector.csv"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def ok_rows(path: Path) -> list[dict[str, str]]:
    return [row for row in read_rows(path) if row.get("status", "ok") == "ok"]


def best_row(rows: list[dict[str, str]], score_key: str = "chi2_total") -> dict[str, str]:
    return min(rows, key=lambda row: as_float(row, score_key))


def token(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def model_path(model_dir: Path, row: dict[str, str]) -> Path:
    f_sat = as_float(row, "f_sat_target")
    alpha = as_float(row, "alpha_s")
    radial_scale = as_float(row, "radial_scale")
    return model_dir / f"model_fsat{token(f_sat)}_alpha{token(alpha)}_rscale{token(radial_scale)}.npz"


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


def row_to_summary(model: str, criterion: str, row: dict[str, str]) -> dict[str, object]:
    return {
        "model": model,
        "criterion": criterion,
        "f_sat_target": f"{as_float(row, 'f_sat_target'):.3f}",
        "f_sat_actual": f"{as_float(row, 'f_sat_actual'):.6f}",
        "alpha_s": f"{as_float(row, 'alpha_s'):.3f}",
        "radial_scale": f"{as_float(row, 'radial_scale'):.3f}",
        "chi2_total": f"{as_float(row, 'chi2_total'):.6g}",
        "chi2_wp": f"{as_float(row, 'chi2_wp'):.6g}",
        "chi2_xi0": f"{as_float(row, 'chi2_xi0'):.6g}",
        "chi2_xi2": f"{as_float(row, 'chi2_xi2'):.6g}",
    }


def build_best_summary() -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for key, spec in RUNS.items():
        rows = ok_rows(spec["grid"])
        all_best = best_row(rows)
        q1_best = best_row([row for row in rows if np.isclose(as_float(row, "radial_scale"), 1.0)])
        q4_best = best_row([row for row in rows if np.isclose(as_float(row, "radial_scale"), 4.0)])
        wp_best = best_row(rows, "chi2_wp")
        summary.append(row_to_summary(spec["label"], "best_total_all_q", all_best))
        summary.append(row_to_summary(spec["label"], "best_total_q1_baseline", q1_best))
        summary.append(row_to_summary(spec["label"], "best_total_q4", q4_best))
        summary.append(row_to_summary(spec["label"], "best_wp_all_q", wp_best))
        if key == "minimal_hod_profile":
            for f_sat in (0.08, 0.10, 0.12):
                subset = [row for row in rows if np.isclose(as_float(row, "f_sat_target"), f_sat)]
                summary.append(row_to_summary(spec["label"], f"best_total_fsat{f_sat:.2f}", best_row(subset)))
    return summary


def combined_split_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in SPLIT_SUMMARIES:
        rows.extend(read_rows(path))
    out = []
    for row in rows:
        out.append(
            {
                "model": row["label"],
                "split": row["split"],
                "covariance_mode": row["covariance_mode"],
                "best_f_sat": f"{as_float(row, 'best_f_sat'):.3f}",
                "best_alpha_s": f"{as_float(row, 'best_alpha_s'):.3f}",
                "best_radial_scale": f"{as_float(row, 'best_radial_scale'):.3f}",
                "best_f_sat_actual": f"{as_float(row, 'best_f_sat_actual'):.6f}",
                "best_chi2": f"{as_float(row, 'best_chi2'):.6g}",
                "delta_chi2_to_second": f"{as_float(row, 'delta_chi2_to_second'):.6g}",
                "hartlap_scale": f"{as_float(row, 'hartlap_scale'):.6g}",
            }
        )
    return out


def min_grid(rows: list[dict[str, str]], score_key: str) -> tuple[list[float], list[float], np.ndarray]:
    fsats = sorted({as_float(row, "f_sat_target") for row in rows})
    qvals = sorted({as_float(row, "radial_scale") for row in rows})
    grid = np.full((len(fsats), len(qvals)), np.nan)
    for i, f_sat in enumerate(fsats):
        for j, q in enumerate(qvals):
            subset = [
                row
                for row in rows
                if np.isclose(as_float(row, "f_sat_target"), f_sat)
                and np.isclose(as_float(row, "radial_scale"), q)
            ]
            if subset:
                grid[i, j] = min(as_float(row, score_key) for row in subset)
    return fsats, qvals, grid


def plot_heatmaps() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for ax, spec in zip(axes, RUNS.values()):
        rows = ok_rows(spec["grid"])
        fsats, qvals, grid = min_grid(rows, "chi2_total")
        delta = grid - np.nanmin(grid)
        image = ax.imshow(delta, origin="lower", aspect="auto", cmap="magma_r")
        ax.set_title(spec["label"])
        ax.set_xlabel("radial scale q")
        ax.set_ylabel(r"$f_{\rm sat}$")
        ax.set_xticks(range(len(qvals)), [f"{value:.1f}" for value in qvals])
        ax.set_yticks(range(len(fsats)), [f"{value:.2f}" for value in fsats])
        vmax = np.nanmax(delta)
        for i in range(len(fsats)):
            for j in range(len(qvals)):
                text_color = "white" if vmax > 0 and delta[i, j] > 0.55 * vmax else "black"
                ax.text(j, i, f"{delta[i, j]:.0f}", ha="center", va="center", fontsize=7, color=text_color)
        fig.colorbar(image, ax=ax, label=r"min over $\alpha_s$: $\Delta\chi^2_{\rm heuristic}$")
    fig.savefig(OUT / "abacus_profile_dilution_heatmaps.png", dpi=220)
    plt.close(fig)


def plot_wp_tradeoff() -> None:
    rows = ok_rows(RUNS["minimal_hod_profile"]["grid"])
    fsats, qvals, wp_grid = min_grid(rows, "chi2_wp")
    _, _, xi2_grid = min_grid(rows, "chi2_xi2")
    _, _, total_grid = min_grid(rows, "chi2_total")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.76, wspace=0.30)
    grids = [(wp_grid, r"$\chi^2(w_p)$"), (xi2_grid, r"$\chi^2(\xi_2)$"), (total_grid, r"$\chi^2_{\rm total}$")]
    for ax, (grid, ylabel) in zip(axes, grids):
        for i, f_sat in enumerate(fsats):
            ax.plot(qvals, grid[i], marker="o", label=f"{f_sat:.2f}")
        ax.set_xlabel("radial scale q")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title=r"$f_{\rm sat}$",
        frameon=False,
        ncol=len(labels),
        fontsize=8,
        title_fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.savefig(OUT / "abacus_profile_dilution_observable_tradeoff.png", dpi=220, bbox_inches="tight")
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES / "fig_profile_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def find_row(rows: list[dict[str, str]], f_sat: float, alpha_s: float, radial_scale: float) -> dict[str, str]:
    for row in rows:
        if (
            np.isclose(as_float(row, "f_sat_target"), f_sat)
            and np.isclose(as_float(row, "alpha_s"), alpha_s)
            and np.isclose(as_float(row, "radial_scale"), radial_scale)
        ):
            return row
    raise ValueError((f_sat, alpha_s, radial_scale))


def plot_wp_curves() -> None:
    observed = read_observed()
    rows = ok_rows(RUNS["minimal_hod_profile"]["grid"])
    selected = [
        ("same f,a: q=1", find_row(rows, 0.04, 1.1, 1.0)),
        ("same f,a: q=4", find_row(rows, 0.04, 1.1, 4.0)),
        ("best q free", best_row(rows)),
        (
            "high-fsat example",
            best_row([row for row in rows if np.isclose(as_float(row, "f_sat_target"), 0.10)]),
        ),
    ]

    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    ax.plot(observed["rp"], observed["wp"], color="black", marker="o", lw=1.5, label="DESI DR1 LRG")
    for label, row in selected:
        payload = np.load(model_path(RUNS["minimal_hod_profile"]["dir"], row))
        full_label = (
            f"{label}: f={as_float(row, 'f_sat_target'):.2f}, "
            f"a={as_float(row, 'alpha_s'):.1f}, q={as_float(row, 'radial_scale'):.1f}"
        )
        ax.plot(payload["obs_rp"], payload["model_wp"], marker="s", ms=3, lw=1.2, label=full_label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r_p\ [h^{-1}{\rm Mpc}]$")
    ax.set_ylabel(r"$w_p(r_p)$")
    ax.legend(frameon=False, fontsize=7.0)
    fig.savefig(OUT / "abacus_profile_dilution_wp_curves.png", dpi=220)
    plt.close(fig)


def format_best_line(row: dict[str, object]) -> str:
    return (
        f"| {row['model']} | {row['criterion']} | {row['f_sat_target']} | {row['alpha_s']} | "
        f"{row['radial_scale']} | {row['chi2_total']} | {row['chi2_wp']} | {row['chi2_xi0']} | {row['chi2_xi2']} |"
    )


def write_report(best_rows: list[dict[str, object]], split_rows: list[dict[str, object]]) -> None:
    lookup = {(row["model"], row["criterion"]): row for row in best_rows}
    minimal_all = lookup[("minimal HOD", "best_total_all_q")]
    minimal_q1 = lookup[("minimal HOD", "best_total_q1_baseline")]
    sham_all = lookup[("SHAM-like", "best_total_all_q")]
    split_lines = "\n".join(
        f"| {row['model']} | {row['split']} | {row['covariance_mode']} | {row['best_f_sat']} | "
        f"{row['best_alpha_s']} | {row['best_radial_scale']} | {row['best_chi2']} | {row['delta_chi2_to_second']} |"
        for row in split_rows
    )
    best_lines = "\n".join(format_best_line(row) for row in best_rows)

    improvement = float(minimal_q1["chi2_total"]) - float(minimal_all["chi2_total"])
    text = f"""# Satellite Profile Dilution Test

Date: 2026-07-08

## 本轮测试做了什么

- Simulation: AbacusSummit_base_c000_ph000, z=0.8, full 2000 Mpc/h periodic box.
- Data vector: DESI DR1 LRG, z=0.600-0.800, unreconstructed wp + xi0 + xi2.
- minimal HOD occupation grid: f_sat = 0.04, 0.06, 0.08, 0.10, 0.12; alpha_s = 0.7, 1.1, 1.3; radial scale q = 1.0, 1.5, 2.0, 3.0, 4.0.
- SHAM-like diagnostic grid: f_sat = 0.04, 0.08, 0.12; alpha_s = 0.7, 1.1; same q grid.
- profile 操作：对 satellite 做 x_sat -> x_host + q * (x_sat - x_host)，再做 periodic wrapping。q > 1 是一个有意简化的 radial dilution 诊断参数，还不是最终物理 satellite profile model。

## heuristic best fits

| model | criterion | f_sat | alpha_s | q | chi2_total | chi2_wp | chi2_xi0 | chi2_xi2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{best_lines}

## covariance split best fits

| model | split | covariance | f_sat | alpha_s | q | chi2 | delta chi2 to second |
|---|---|---|---:|---:|---:|---:|---:|
{split_lines}

## 主要结论

1. satellite radial profile 是造成低 f_sat 偏好的重要因素。minimal-HOD 网格中，允许 q 自由变化后，heuristic total chi2 从 q=1 baseline 的 {minimal_q1['chi2_total']} 降到 q={minimal_all['radial_scale']} 的 {minimal_all['chi2_total']}，Delta chi2 = {improvement:.2f}。
2. minimal-HOD 的 heuristic 最优点移动到 f_sat={minimal_all['f_sat_target']}, alpha_s={minimal_all['alpha_s']}, q={minimal_all['radial_scale']}。这说明之前的 satellite prescription 对 wp 来说过于中心集中。
3. covariance-weighted joint split 更保守：diagonal/full joint covariance 都仍偏好 f_sat=0.04, q=4。因此 profile dilution 会削弱低 f_sat 偏好，但还不能完全消除。
4. 文献常见的 f_sat~0.10-0.15 不能靠 uniform radial scaling 单独救回来。即使 q=4，minimal-HOD 在 f_sat=0.10-0.12 的最优行仍明显差于 f_sat=0.04-0.06，主要因为 wp 和 RSD multipoles 不能同时匹配。
5. SHAM-like profile dilution 的趋势类似，但 paper-level 解释价值较弱：其 heuristic 最优仍是 f_sat={sham_all['f_sat_target']}, alpha_s={sham_all['alpha_s']}, q={sham_all['radial_scale']}，covariance split 也强烈停在 f_sat=0.04。

## 对论文主线的解释

目前最合适的表述不是简单说 "DESI prefers very low satellite fraction"，而是：

> 在当前 Abacus full-box compact HOD/SHAM 构造下，small-scale wp 会把模型推向低 f_sat，除非 satellites 被设置得比原始 selected subhalo-like reservoir 明显更低 concentration。简单 radial dilution 能改善拟合，并把 heuristic optimum 推到 f_sat about 0.06；但它单独不足以在同时匹配 xi0/xi2 的情况下恢复 f_sat about 0.10-0.15。

这把 low-f_sat 结果从一个可能的 artifact，转化成一个有价值的诊断：satellite radial profile、satellite selection 和 velocity bias 必须联合拟合。下一步最值得做的不是继续盲目扩 f_sat 网格，而是加入更物理的 radial-profile 参数，例如 satellite concentration/core radius，或 rank-dependent radial selection。

## 输出文件

- abacus_profile_dilution_best_summary.csv
- abacus_profile_dilution_split_summary.csv
- abacus_profile_dilution_heatmaps.png
- abacus_profile_dilution_observable_tradeoff.png
- abacus_profile_dilution_wp_curves.png
"""
    (OUT / "abacus_profile_dilution_report.md").write_text(text, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    best = build_best_summary()
    splits = combined_split_summary()
    write_rows(OUT / "abacus_profile_dilution_best_summary.csv", best)
    write_rows(OUT / "abacus_profile_dilution_split_summary.csv", splits)
    plot_heatmaps()
    plot_wp_tradeoff()
    plot_wp_curves()
    write_report(best, splits)
    print(f"Wrote profile-dilution summary products to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
