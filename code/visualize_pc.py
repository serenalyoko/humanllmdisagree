"""
Political Compass - Multi-model visualization
Usage:
    python code/visualize_pc.py                      # auto-loads all JSONs in data/pc_results/
    python code/visualize_pc.py --results path1 path2 ...
"""

import json
import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

RATING_TO_NUM = {
    "strongly agree": 2,
    "agree": 1,
    "neutral": 0,
    "disagree": -1,
    "strongly disagree": -2,
}

MODEL_COLORS = plt.cm.tab10.colors


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def label_from_path(path: str) -> str:
    """Use the filename stem as the model label (e.g. 'claude_pc_result_3_passes' → 'claude')."""
    stem = pathlib.Path(path).stem
    return stem.split("_")[0].capitalize()


# ── Plot 1: Political Compass ─────────────────────────────────────────────────

def plot_political_compass(datasets: list[tuple[str, dict]], ax=None):
    """
    Scatter plot of model positions on the political compass with error crosshairs.
    datasets: list of (label, result_dict)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Background quadrant shading
    ax.fill_between([-10, 0], 0, 10, color="#e8c8c8", alpha=0.35)   # Auth Left
    ax.fill_between([0, 10],  0, 10, color="#c8c8e8", alpha=0.35)   # Auth Right
    ax.fill_between([-10, 0], -10, 0, color="#c8e8c8", alpha=0.35)  # Lib Left
    ax.fill_between([0, 10],  -10, 0, color="#e8e8c8", alpha=0.35)  # Lib Right

    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(0, color="black", linewidth=1.2)

    quad_kw = dict(fontsize=10, alpha=0.45, ha="center", va="center")
    ax.text(-6,  7, "Authoritarian\nLeft",   **quad_kw)
    ax.text( 6,  7, "Authoritarian\nRight",  **quad_kw)
    ax.text(-6, -7, "Libertarian\nLeft",     **quad_kw)
    ax.text( 6, -7, "Libertarian\nRight",    **quad_kw)

    for i, (label, data) in enumerate(datasets):
        pc = data["political_compass_averaged"]
        econ   = pc["economic_score"]
        social = pc["social_score"]
        e_std  = pc.get("economic_score_std", 0)
        s_std  = pc.get("social_score_std", 0)
        color  = MODEL_COLORS[i % len(MODEL_COLORS)]

        ax.errorbar(econ, social, xerr=e_std, yerr=s_std,
                    fmt="none", color=color, alpha=0.5, capsize=5, linewidth=1.5)
        ax.scatter(econ, social, color=color, s=120, zorder=5,
                   label=f"{label}  ({econ:+.2f}, {social:+.2f})")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("Economic axis  ←  Left  |  Right  →", fontsize=11)
    ax.set_ylabel("Social axis  ←  Libertarian  |  Authoritarian  →", fontsize=11)
    ax.set_title("Political Compass — LLM Positions", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.3)

    return fig, ax


# ── Plot 2: Per-pass trajectories ─────────────────────────────────────────────

def plot_pass_trajectories(datasets: list[tuple[str, dict]], axes=None):
    """
    Line charts showing economic and social scores across passes.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig = axes[0].figure

    ax_econ, ax_soc = axes

    for i, (label, data) in enumerate(datasets):
        passes       = data["per_pass_scores"]
        pass_nums    = [p["pass"] for p in passes]
        econ_scores  = [p["economic_score"] for p in passes]
        social_scores = [p["social_score"] for p in passes]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        ax_econ.plot(pass_nums, econ_scores,  marker="o", color=color, label=label, linewidth=2)
        ax_soc.plot(pass_nums, social_scores, marker="o", color=color, label=label, linewidth=2)

    for ax, title, ylabel in [
        (ax_econ, "Economic Score per Pass",  "Economic score (Left ← 0 → Right)"),
        (ax_soc,  "Social Score per Pass",    "Social score (Libertarian ← 0 → Auth)"),
    ]:
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Pass", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(pass_nums)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Score Consistency Across Passes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ── Plot 3: Rating heatmap ────────────────────────────────────────────────────

def plot_rating_heatmap(datasets: list[tuple[str, dict]], ax=None):
    """
    Heatmap: questions (rows) × models (cols), colored by dominant rating.
    """
    labels   = [label for label, _ in datasets]
    questions = [r["question"][:55] + "…" if len(r["question"]) > 55 else r["question"]
                 for r in datasets[0][1]["results"]]
    n_q = len(questions)
    n_m = len(labels)

    matrix = np.full((n_q, n_m), np.nan)
    for j, (_, data) in enumerate(datasets):
        for i, q in enumerate(data["results"]):
            matrix[i, j] = RATING_TO_NUM.get(q["dominant_rating"], np.nan)

    fig_h = max(10, n_q * 0.28)
    fig_w = max(5,  n_m * 1.8 + 3)

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.figure

    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(n_m))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_q))
    ax.set_yticklabels(questions, fontsize=7)
    ax.set_title("Dominant Rating per Question × Model", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.set_ticklabels(["Str. Disagree", "Disagree", "Neutral", "Agree", "Str. Agree"])

    fig.tight_layout()
    return fig, ax


# ── Plot 4: Rating distribution bars ─────────────────────────────────────────

def plot_rating_distribution(datasets: list[tuple[str, dict]], ax=None):
    """
    Stacked/grouped bar chart of rating category counts per model.
    """
    categories = ["strongly agree", "agree", "neutral", "disagree", "strongly disagree"]
    cat_colors = ["#2d6a4f", "#74c69d", "#adb5bd", "#e07a5f", "#c1121f"]

    model_counts = {}
    for label, data in datasets:
        counts = {c: 0 for c in categories}
        for q in data["results"]:
            r = q["dominant_rating"]
            if r in counts:
                counts[r] += 1
        model_counts[label] = counts

    n_m = len(datasets)
    x = np.arange(n_m)
    width = 0.55

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_m * 2), 5))
    else:
        fig = ax.figure

    bottoms = np.zeros(n_m)
    for cat, color in zip(categories, cat_colors):
        vals = np.array([model_counts[lbl][cat] for lbl, _ in datasets])
        ax.bar(x, vals, width, bottom=bottoms, color=color, label=cat, edgecolor="white")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in datasets], fontsize=11)
    ax.set_ylabel("Number of questions", fontsize=10)
    ax.set_title("Rating Distribution per Model (dominant rating)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig, ax


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="*", help="Paths to result JSON files")
    parser.add_argument("--outdir", default="data/figures", help="Output directory for saved figures")
    args = parser.parse_args()

    results_dir = pathlib.Path("data/pc_results")
    out_dir = pathlib.Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.results:
        paths = args.results
    else:
        paths = sorted(results_dir.glob("*_3_passes.json"))

    if not paths:
        print("No result files found.")
        return

    datasets = [(label_from_path(str(p)), load_results(str(p))) for p in paths]
    print(f"Loaded {len(datasets)} model(s): {[l for l, _ in datasets]}")

    fig1, _ = plot_political_compass(datasets)
    fig1.savefig(out_dir / "01_political_compass.png", dpi=150, bbox_inches="tight")

    fig2, _ = plot_pass_trajectories(datasets)
    fig2.savefig(out_dir / "02_pass_trajectories.png", dpi=150, bbox_inches="tight")

    fig3, _ = plot_rating_heatmap(datasets)
    fig3.savefig(out_dir / "03_rating_heatmap.png", dpi=150, bbox_inches="tight")

    fig4, _ = plot_rating_distribution(datasets)
    fig4.savefig(out_dir / "04_rating_distribution.png", dpi=150, bbox_inches="tight")

    print(f"Saved 4 figures to {out_dir}/")
    plt.show()


if __name__ == "__main__":
    main()
