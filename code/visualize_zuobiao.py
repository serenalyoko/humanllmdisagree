"""
Zuobiao Ideology — Multi-model visualization
Parallel to visualize_pc.py but for Pan & Xu (2018) lt1/lt2/lt3 scores.

Plots produced:
  01_zuobiao_quadrant.png   — lt1 × lt2 quadrant (Pan & Xu Figure 1 style)
  02_pass_trajectories.png  — lt1, lt2, lt3 scores across passes per model
  03_rating_heatmap.png     — dominant rating per question × model (Red-Green)
  04_rating_distribution.png— stacked bar of rating category counts per model
  05_lt3_comparison.png     — nationalism score (lt3) bar chart across models

Usage:
    python visualize_zuobiao.py                         # auto-loads all JSONs in data/pc_results/
    python visualize_zuobiao.py --results path1 path2   # explicit paths
    python visualize_zuobiao.py --outdir my/figures     # custom output directory
"""

import json
import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

RATING_TO_NUM = {
    "strongly agree":    2,
    "agree":             1,
    "neutral":           0,
    "disagree":         -1,
    "strongly disagree": -2,
}

MODEL_COLORS = plt.cm.tab10.colors


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def label_from_path(path: str) -> str:
    """Derive a short model label from the filename stem."""
    stem = pathlib.Path(path).stem          # e.g. "claude_zuobiao_result_3_passes"
    return stem.split(".")[0].capitalize()  # → "Claude"


def _get_zuobiao(data: dict, key: str = "zuobiao_averaged") -> dict:
    """
    Safely retrieve the zuobiao score block, falling back gracefully.
    Supports both 'zuobiao_averaged'/'zuobiao_dominant' and legacy keys.
    """
    if key in data:
        return data[key]
    # fallback: try dominant
    fallback = "zuobiao_dominant" if key == "zuobiao_averaged" else "zuobiao_averaged"
    if fallback in data:
        return data[fallback]
    raise KeyError(f"Neither '{key}' nor '{fallback}' found in result JSON.")


def _quadrant_label(lt1: float, lt2: float) -> str:
    """Return the Pan & Xu quadrant name for a point."""
    row = "liberal" if lt1 > 0 else "authoritarian"
    col = "promarket" if lt2 > 0 else "interventionist"
    return f"{row} / {col}"


# ── Plot 1: Zuobiao quadrant (Pan & Xu Figure 1 style) ───────────────────────

def plot_zuobiao_quadrant(datasets: list[tuple[str, dict]], ax=None):
    """
    Scatter plot on the Pan & Xu ideological plane:
      x-axis : lt2  (left = interventionist / collectivist,
                      right = promarket / individualist)
      y-axis : lt1  (bottom = authoritarian,
                      top    = liberal)
    Error crosshairs show ± std dev across passes (averaged block).
    Individual pass dots are also plotted semi-transparently.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Quadrant shading (matches R plot_base colours loosely)
    ax.fill_between([-10, 0],  0, 10, color="#e8c8c8", alpha=0.30)   # auth-interventionist
    ax.fill_between([ 0, 10],  0, 10, color="#c8c8e8", alpha=0.30)   # auth-promarket
    ax.fill_between([-10, 0], -10, 0, color="#c8e8c8", alpha=0.30)   # lib-interventionist
    ax.fill_between([ 0, 10], -10, 0, color="#e8e8c8", alpha=0.30)   # lib-promarket

    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(0, color="black", linewidth=1.2)

    quad_kw = dict(fontsize=10, alpha=0.40, ha="center", va="center")
    ax.text(-6,  7, "Authoritarian\nInterventionist",  **quad_kw)
    ax.text( 6,  7, "Authoritarian\nPromarket",        **quad_kw)
    ax.text(-6, -7, "Liberal\nInterventionist",        **quad_kw)
    ax.text( 6, -7, "Liberal\nPromarket",              **quad_kw)

    for i, (label, data) in enumerate(datasets):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        # ── Per-pass dots ─────────────────────────────────────────────────────
        for pp in data.get("per_pass_scores", []):
            ax.scatter(pp["lt2"], pp["lt1"],
                       color=color, s=40, alpha=0.30, zorder=3,
                       marker="o", linewidths=0)

        # ── Averaged point with error bars ────────────────────────────────────
        avg = _get_zuobiao(data, "zuobiao_averaged")
        lt1_avg = avg["lt1"]
        lt2_avg = avg["lt2"]
        lt1_std = avg.get("lt1_std", 0)
        lt2_std = avg.get("lt2_std", 0)

        ax.errorbar(lt2_avg, lt1_avg,
                    xerr=lt2_std, yerr=lt1_std,
                    fmt="none", color=color, alpha=0.55,
                    capsize=5, linewidth=1.5, zorder=4)

        # ── Dominant point (larger, solid) ────────────────────────────────────
        dom = _get_zuobiao(data, "zuobiao_dominant")
        lt1_dom = dom["lt1"]
        lt2_dom = dom["lt2"]

        ax.scatter(lt2_dom, lt1_dom,
                   color=color, s=140, zorder=5, marker="o",
                   edgecolors="white", linewidths=1.2,
                   label=(f"{label}  "
                          f"lt1={lt1_dom:+.2f}  "
                          f"lt2={lt2_dom:+.2f}  "
                          f"[{_quadrant_label(lt1_dom, lt2_dom)}]"))

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel(
        "lt2 — Economic axis\n"
        "←  Interventionist / Collectivist  |  Promarket / Individualist  →",
        fontsize=11)
    ax.set_ylabel(
        "lt1 — Political Liberalism axis\n"
        "↓  Authoritarian  |  Liberal  ↑",
        fontsize=11)
    ax.set_title("Zuobiao Ideological Quadrant — LLM Positions\n(Pan & Xu 2018)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.25)

    # Custom legend entries for dot sizes
    legend_extra = [
        Line2D([0], [0], marker="o", color="gray", markersize=10,
               label="dominant (3-pass)", linestyle="None",
               markeredgecolor="white", markeredgewidth=1),
        Line2D([0], [0], marker="o", color="gray", markersize=6,
               alpha=0.4, label="per-pass", linestyle="None"),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_extra,
              labels=ax.get_legend_handles_labels()[1] + ["dominant (3-pass)", "per-pass"],
              loc="lower right", fontsize=8, framealpha=0.85)

    return fig, ax


# ── Plot 2: Per-pass trajectories ─────────────────────────────────────────────

def plot_pass_trajectories(datasets: list[tuple[str, dict]], axes=None):
    """
    Three line charts — lt1, lt2, lt3 scores across passes.
    Mirrors visualize_pc.py's two-panel layout but adds a third panel for lt3.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    else:
        fig = axes[0].figure

    ax_lt1, ax_lt2, ax_lt3 = axes

    for i, (label, data) in enumerate(datasets):
        passes = data.get("per_pass_scores", [])
        if not passes:
            continue
        pass_nums = [p["pass"] for p in passes]
        lt1s      = [p["lt1"]  for p in passes]
        lt2s      = [p["lt2"]  for p in passes]
        lt3s      = [p["lt3"]  for p in passes]
        color     = MODEL_COLORS[i % len(MODEL_COLORS)]

        ax_lt1.plot(pass_nums, lt1s, marker="o", color=color, label=label, linewidth=2)
        ax_lt2.plot(pass_nums, lt2s, marker="o", color=color, label=label, linewidth=2)
        ax_lt3.plot(pass_nums, lt3s, marker="o", color=color, label=label, linewidth=2)

    panel_cfg = [
        (ax_lt1, "lt1 — Political Liberalism",
         "score  (←  Authoritarian  |  Liberal  →)"),
        (ax_lt2, "lt2 — Promarket / Non-traditional",
         "score  (←  Interventionist  |  Promarket  →)"),
        (ax_lt3, "lt3 — Nationalism × (−1)",
         "score  (←  Nationalist  |  Anti-national  →)"),
    ]
    all_pass_nums = [p["pass"]
                     for _, data in datasets
                     for p in data.get("per_pass_scores", [])]
    xticks = sorted(set(all_pass_nums)) if all_pass_nums else [1]

    for ax, title, ylabel in panel_cfg:
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Pass", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(xticks)
        ax.set_ylim(-10, 10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle("Score Consistency Across Passes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ── Plot 3: Rating heatmap ────────────────────────────────────────────────────

def plot_rating_heatmap(datasets: list[tuple[str, dict]], ax=None):
    """
    Heatmap: questions (rows) × models (columns), colored by dominant rating.
    Identical logic to visualize_pc.py but reads from any JSON with 'results'.
    """
    labels    = [label for label, _ in datasets]
    questions = [
        (r["question"][:60] + "…" if len(r["question"]) > 60 else r["question"])
        for r in datasets[0][1]["results"]
    ]
    n_q = len(questions)
    n_m = len(labels)

    matrix = np.full((n_q, n_m), np.nan)
    for j, (_, data) in enumerate(datasets):
        for i, q in enumerate(data["results"]):
            matrix[i, j] = RATING_TO_NUM.get(q.get("dominant_rating"), np.nan)

    fig_h = max(10, n_q * 0.30)
    fig_w = max(5,  n_m * 2.0 + 3)

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.figure

    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    im   = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(n_m))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_q))
    ax.set_yticklabels(questions, fontsize=7)
    ax.set_title("Dominant Rating per Question × Model", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.35, pad=0.02)
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.set_ticklabels(["Str. Disagree", "Disagree", "Neutral", "Agree", "Str. Agree"])

    fig.tight_layout()
    return fig, ax


# ── Plot 4: Rating distribution bars ─────────────────────────────────────────

def plot_rating_distribution(datasets: list[tuple[str, dict]], ax=None):
    """
    Stacked bar chart: rating category counts per model (dominant ratings).
    Identical to visualize_pc.py but reads question count from any schema.
    """
    categories = ["strongly agree", "agree", "neutral", "disagree", "strongly disagree"]
    cat_colors = ["#2d6a4f", "#74c69d", "#adb5bd", "#e07a5f", "#c1121f"]

    model_counts = {}
    for label, data in datasets:
        counts = {c: 0 for c in categories}
        for q in data.get("results", []):
            r = q.get("dominant_rating")
            if r in counts:
                counts[r] += 1
        model_counts[label] = counts

    n_m = len(datasets)
    x   = np.arange(n_m)
    width = 0.55

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_m * 2), 5))
    else:
        fig = ax.figure

    bottoms = np.zeros(n_m)
    for cat, color in zip(categories, cat_colors):
        vals = np.array([model_counts[lbl][cat] for lbl, _ in datasets])
        ax.bar(x, vals, width, bottom=bottoms,
               color=color, label=cat, edgecolor="white")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in datasets], fontsize=11)
    ax.set_ylabel("Number of questions", fontsize=10)
    ax.set_title("Rating Distribution per Model (dominant rating)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    return fig, ax


# ── Plot 5: lt3 nationalism comparison ───────────────────────────────────────

def plot_lt3_comparison(datasets: list[tuple[str, dict]], ax=None):
    """
    Horizontal bar chart comparing lt3 (nationalism × −1) across models.
    Positive = less nationalist, negative = more nationalist.
    Includes dominant and averaged values side-by-side.
    """
    labels = [label for label, _ in datasets]
    dom_vals = []
    avg_vals = []
    avg_stds = []

    for _, data in datasets:
        dom = _get_zuobiao(data, "zuobiao_dominant")
        avg = _get_zuobiao(data, "zuobiao_averaged")
        dom_vals.append(dom["lt3"])
        avg_vals.append(avg["lt3"])
        avg_stds.append(avg.get("lt3_std", 0))

    n_m   = len(labels)
    y     = np.arange(n_m)
    height = 0.35

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(4, n_m * 0.9)))
    else:
        fig = ax.figure

    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(n_m)]

    ax.barh(y + height / 2, dom_vals, height,
            color=colors, alpha=0.85, label="dominant")
    ax.barh(y - height / 2, avg_vals, height,
            color=colors, alpha=0.45, xerr=avg_stds,
            error_kw=dict(elinewidth=1.2, capsize=4),
            label="averaged ± std")

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(
        "lt3 score  ←  Nationalist  |  Anti-nationalist  →",
        fontsize=10)
    ax.set_title("lt3 — Nationalism × (−1) Comparison",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-10, 10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    # Annotate bars with values
    for i, (dv, av) in enumerate(zip(dom_vals, avg_vals)):
        ax.text(dv + (0.2 if dv >= 0 else -0.2),
                i + height / 2,
                f"{dv:+.2f}",
                va="center",
                ha="left" if dv >= 0 else "right",
                fontsize=8)

    fig.tight_layout()
    return fig, ax


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Zuobiao (Pan & Xu 2018) ideology scores across LLMs."
    )
    parser.add_argument(
        "--results", nargs="*",
        help="Paths to Zuobiao result JSON files. "
             "If omitted, auto-loads all *zuobiao*.json in data/zuobiao_results/."
    )
    parser.add_argument(
        "--outdir", default="plots/",
        help="Directory to save PNG figures (default: plots/)."
    )
    args = parser.parse_args()

    results_dir = pathlib.Path("data/zuobiao_results")
    out_dir     = pathlib.Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load files ────────────────────────────────────────────────────────────
    if args.results:
        paths = [pathlib.Path(p) for p in args.results]
    else:
        paths = sorted(results_dir.glob("*zuobiao*.json"))
        if not paths:
            # fall back to any json with zuobiao_dominant key
            candidates = list(results_dir.glob("*.json"))
            paths = []
            for p in candidates:
                try:
                    d = load_results(str(p))
                    if "zuobiao_dominant" in d or "zuobiao_averaged" in d:
                        paths.append(p)
                except Exception:
                    pass

    if not paths:
        print("No Zuobiao result files found. "
              "Pass --results path1 path2 ... or place files in data/pc_results/.")
        return

    datasets = []
    for p in paths:
        try:
            data  = load_results(str(p))
            label = label_from_path(str(p))
            datasets.append((label, data))
            print(f"  Loaded: {p.name}  →  label={label!r}")
        except Exception as e:
            print(f"  Skipped {p.name}: {e}")

    if not datasets:
        print("No valid datasets loaded.")
        return

    print(f"\nGenerating figures for {len(datasets)} model(s): "
          f"{[l for l, _ in datasets]}\n")

    # ── Figure 1: Quadrant ────────────────────────────────────────────────────
    fig1, _ = plot_zuobiao_quadrant(datasets)
    out1 = out_dir / "01_zuobiao_quadrant.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out1}")

    # ── Figure 2: Pass trajectories ───────────────────────────────────────────
    fig2, _ = plot_pass_trajectories(datasets)
    out2 = out_dir / "02_pass_trajectories.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out2}")

    # ── Figure 3: Rating heatmap ──────────────────────────────────────────────
    fig3, _ = plot_rating_heatmap(datasets)
    out3 = out_dir / "03_rating_heatmap.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out3}")

    # ── Figure 4: Rating distribution ─────────────────────────────────────────
    fig4, _ = plot_rating_distribution(datasets)
    out4 = out_dir / "04_rating_distribution.png"
    fig4.savefig(out4, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out4}")

    # ── Figure 5: lt3 nationalism comparison ──────────────────────────────────
    fig5, _ = plot_lt3_comparison(datasets)
    out5 = out_dir / "05_lt3_comparison.png"
    fig5.savefig(out5, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out5}")

    print(f"\nAll figures saved to {out_dir}/")
    plt.show()


if __name__ == "__main__":
    main()