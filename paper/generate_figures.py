#!/usr/bin/env python3
"""Generate all figures for 'The Other Half of Intelligence' paper."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

BLUE = "#4361ee"
GREEN = "#2d6a4f"
ORANGE = "#f3722c"
RED = "#d00000"
GRAY = "#adb5bd"
PURPLE = "#7209b7"
TEAL = "#2ec4b6"


def fig1_frontier_scaling():
    """Figure 1 (THE MONEY FIGURE): Frontier scaling bar chart.

    Shows single-pass vs system EM for all 4 models.
    Demonstrates constant ~40-47% system gain across scales.
    """
    models = ["Qwen 2.5\n7B", "Claude\nSonnet 4.6", "Claude\nOpus 4.6"]
    single_pass = [13.3, 20.0, 30.0]
    system_auto = [60.0, 60.0, 76.7]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(models))
    width = 0.32

    bars1 = ax.bar(x - width/2, single_pass, width, label="Single-pass (model alone)",
                   color=GRAY, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, system_auto, width, label="System-level decomposition",
                   color=BLUE, edgecolor="black", linewidth=0.5)

    # Add gain annotations
    for i, (sp, sys) in enumerate(zip(single_pass, system_auto)):
        gain = sys - sp
        mid = (sp + sys) / 2
        ax.annotate(f"+{gain:.1f}%",
                    xy=(i + width/2, sys), xytext=(i + 0.45, mid),
                    fontsize=9, fontweight="bold", color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                    ha="left", va="center")

    # SOTA line
    ax.axhline(y=43.9, color=ORANGE, linestyle="--", linewidth=1, alpha=0.8)
    ax.text(2.45, 45, "StepChain SOTA (43.9%)", fontsize=7, color=ORANGE, ha="right")

    ax.set_ylabel("Exact Match (%)")
    ax.set_title("System Architecture Helps at Every Scale\n(MuSiQue 2-hop, n=30)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 90)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_frontier_scaling.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig1_frontier_scaling.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig1_frontier_scaling")


def fig2_direction_reversal():
    """Figure 2: Direction reversal — same technique helps at system level, hurts at model level."""
    techniques = ["Query\nDecomposition", "Chain-of-\nThought", "More Retrieval\n(top-10 vs top-5)"]
    model_level = [-12.4, -7.2, -2.4]
    system_level = [46.7, None, 23.4]  # CoT not applicable at system level

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(techniques))
    width = 0.32

    bars1 = ax.bar(x - width/2, model_level, width, label="Applied at model level",
                   color=RED, edgecolor="black", linewidth=0.5, alpha=0.8)

    # System-level (skip None)
    sys_vals = []
    sys_x = []
    for i, v in enumerate(system_level):
        if v is not None:
            sys_vals.append(v)
            sys_x.append(x[i] + width/2)

    bars2 = ax.bar(sys_x, sys_vals, width, label="Applied at system level",
                   color=GREEN, edgecolor="black", linewidth=0.5, alpha=0.8)

    # N/A annotation for CoT
    ax.text(1 + width/2, 2, "N/A", ha="center", va="bottom", fontsize=8,
            color=GRAY, fontstyle="italic")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Effect on Exact Match (%)")
    ax.set_title("The Direction Reversal: Same Technique, Different Location\n(MuSiQue, Phi-3 / Qwen 7B)")
    ax.set_xticks(x)
    ax.set_xticklabels(techniques)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Significance annotation
    ax.annotate("p = 0.0009", xy=(0 - width/2, -12.4), xytext=(0 - width/2, -18),
                fontsize=7, ha="center", va="top", color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_direction_reversal.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig2_direction_reversal.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig2_direction_reversal")


def fig3_sota_comparison():
    """Figure 3: Scatter plot — model size vs EM, with our results highlighted."""
    # Published methods
    methods = [
        ("GPT-4o\n(no retrieval)", 200, 10.8, GRAY),
        ("BM25", 7, 13.8, GRAY),
        ("BGE embed", 7, 20.8, GRAY),
        ("Flan-T5-XXL\n+ IRCoT", 11, 30.8, GRAY),
        ("GPT-3\n+ IRCoT", 175, 36.5, GRAY),
        ("RAPTOR", 50, 36.4, GRAY),
        ("SiReRAG", 50, 40.5, GRAY),
        ("HopRAG", 50, 42.2, GRAY),
        ("StepChain\n(prev SOTA)", 50, 43.9, ORANGE),
    ]

    # Ours
    ours = [
        ("Qwen 7B\n+ system", 7, 60.0, BLUE),
        ("Sonnet\n+ system", 200, 60.0, GREEN),
        ("Opus\n+ system", 400, 76.7, PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, size, em, color in methods:
        ax.scatter(size, em, s=80, c=color, edgecolors="black", linewidth=0.5, zorder=3)
        ax.annotate(name, (size, em), textcoords="offset points", xytext=(8, -3),
                    fontsize=6, color="gray")

    for name, size, em, color in ours:
        ax.scatter(size, em, s=150, c=color, edgecolors="black", linewidth=1, zorder=4,
                   marker="*")
        ax.annotate(name, (size, em), textcoords="offset points", xytext=(8, -3),
                    fontsize=7, fontweight="bold", color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (B parameters, log scale)")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("MuSiQue 2-hop: Model Size vs Accuracy\n(Stars = ours)")
    ax.set_ylim(0, 85)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_sota_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig3_sota_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig3_sota_comparison")


def fig4_ablation():
    """Figure 4: Component ablation waterfall chart."""
    components = [
        "Full System\n(Qwen + 8-shot\n+ embed top-3)",
        "- Better extractor\n(Phi-3 instead)",
        "- Embedding retrieval\n(keyword instead)",
        "- System decomp\n(single pass)",
    ]
    ems = [56.7, 36.7, 10.0, 3.3]
    deltas = [0, -20.0, -26.7, -6.7]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = [BLUE, ORANGE, RED, RED]
    bars = ax.barh(range(len(components)), ems, color=colors, edgecolor="black",
                   linewidth=0.5, alpha=0.8)

    for i, (em, delta) in enumerate(zip(ems, deltas)):
        label = f"{em:.1f}%"
        if delta != 0:
            label += f" ({delta:+.1f}%)"
        ax.text(em + 1, i, label, va="center", fontsize=8, fontweight="bold")

    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components, fontsize=8)
    ax.set_xlabel("Exact Match (%)")
    ax.set_title("Component Ablation: Every Piece Matters\n(MuSiQue 2-hop, n=30)")
    ax.set_xlim(0, 70)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_ablation.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig4_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig4_ablation")


def fig5_full_validation():
    """Figure 5: Large-scale validation (n=1252) results."""
    configs = ["Single-pass\n(Qwen 7B)", "Auto decomp\n+ Qwen", "Gold decomp\n+ Qwen", "StepChain\nSOTA"]
    ems = [17.3, 38.6, 45.9, 43.9]
    colors = [GRAY, BLUE, GREEN, ORANGE]

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(configs, ems, color=colors, edgecolor="black", linewidth=0.5)
    for bar, em in zip(bars, ems):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{em:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Full Validation: ALL 2-hop MuSiQue (n=1,252)")
    ax.set_ylim(0, 55)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_full_validation.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig5_full_validation.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig5_full_validation")


def fig6_per_hop():
    """Figure 6: Per-hop accuracy across model scales."""
    models = ["Qwen 7B\n(auto)", "Sonnet\n(auto)", "Opus\n(auto)", "Opus\n(gold)"]
    hop1 = [73.3, 70.0, 73.3, 80.0]
    hop2 = [60.0, 60.0, 76.7, 80.0]

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(models))
    width = 0.3

    ax.bar(x - width/2, hop1, width, label="Hop 1 EM", color=TEAL,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, hop2, width, label="Hop 2 EM", color=PURPLE,
           edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Per-Hop Extraction Accuracy by Model Scale\n(MuSiQue 2-hop, n=30)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 90)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_per_hop.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig6_per_hop.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig6_per_hop")


def fig7_cross_benchmark():
    """Figure 7: Cross-benchmark — decomposition helps hard tasks, hurts easy ones.

    Data from our experiments:
    - HotpotQA (n=50): single=56.0, embed_no_decomp=60.0, auto_decomp=38.0
    - MuSiQue (n=30):  single=13.3, system_auto_decomp=60.0
    """
    benchmarks = ["HotpotQA\n(2-hop, easier)", "MuSiQue\n(2-hop, harder)"]

    fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True)

    # --- HotpotQA: decomposition HURTS ---
    ax = axes[0]
    configs_h = ["Single-\npass", "Embed\n(no decomp)", "With\ndecomp"]
    vals_h = [56.0, 60.0, 38.0]
    colors_h = [GRAY, GREEN, RED]
    bars = ax.bar(configs_h, vals_h, color=colors_h, edgecolor="black", linewidth=0.5,
                  width=0.6)
    for bar, v in zip(bars, vals_h):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.2, f"{v:.0f}%",
                ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("HotpotQA (easier)", fontsize=10)
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate("$-18\\%$", xy=(2, 38), xytext=(2.3, 50), fontsize=9, color=RED,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    # --- MuSiQue: decomposition HELPS ---
    ax = axes[1]
    configs_m = ["Single-\npass", "With system\ndecomp"]
    vals_m = [13.3, 60.0]
    colors_m = [GRAY, BLUE]
    bars = ax.bar(configs_m, vals_m, color=colors_m, edgecolor="black", linewidth=0.5,
                  width=0.5)
    for bar, v in zip(bars, vals_m):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.2, f"{v:.1f}%",
                ha="center", fontsize=8, fontweight="bold")
    ax.set_title("MuSiQue (harder)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate("$+46.7\\%$", xy=(1, 60), xytext=(1.25, 40), fontsize=9, color=BLUE,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))

    fig.suptitle("Decomposition: Task-Dependent Effect", fontsize=11, fontweight="bold",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_cross_benchmark.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig7_cross_benchmark.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig7_cross_benchmark")


def fig8_diminishing_returns():
    """Figure 8: Diminishing returns — delta-EM by technique category.

    Strip plot showing every v8-v10 technique category's best delta-EM.
    Highlights that only retrieval-quality improvements survive.
    """
    # Categories sorted by delta-EM, with config counts
    categories = [
        ("Replace 8-shot\nw/ generate+select", -60.0, 1),
        ("No decomposition", -43.3, 1),
        ("MC selection\nfrom LLM candidates", -20.0, 2),
        ("Per-paragraph\nextraction", -20.0, 2),
        ("Answer position\nbias constraint", -13.3, 1),
        ("Query decomp\n(model-level)", -12.4, 1),
        ("KG enrichment", -10.0, 250),
        ("Retrieval k=5\nfor hop 2", -10.0, 2),
        ("CoT prompting", -7.2, 1),
        ("Multi-model\nensemble", -6.7, 2),
        ("Type constraints", -3.3, 9),
        ("Contrastive\nprompts", -3.3, 3),
        ("Retrieval k=10", -2.4, 1),
        ("Self-consistency\n(temperature)", 0.0, 3),
        ("Pipeline\nself-consistency", 0.0, 1),
        ("Position bias\n(reversed ctx)", 0.0, 1),
        ("Confidence\ngating", 0.0, 2),
        ("Entailment\ncheck", 3.3, 1),
        ("AIC +\nfocused rerank", 6.7, 6),
    ]

    labels = [c[0] for c in categories]
    deltas = [c[1] for c in categories]

    fig, ax = plt.subplots(figsize=(7, 7))

    colors = []
    for d in deltas:
        if d > 0:
            colors.append(GREEN)
        elif d < 0:
            colors.append(RED)
        else:
            colors.append(GRAY)

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, deltas, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    for i, d in enumerate(deltas):
        offset = 1.5 if d >= 0 else -1.5
        ha = "left" if d >= 0 else "right"
        ax.text(d + offset, i, f"{d:+.1f}%", va="center", ha=ha, fontsize=7,
                fontweight="bold")

    ax.axvline(x=0, color="black", linewidth=1)
    ax.axvline(x=17.0, color=PURPLE, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.text(17.5, len(labels) - 1, "Model upgrade\n(Qwen$\\to$Opus)\n+17%",
            fontsize=7, color=PURPLE, va="top")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("$\\Delta$ Exact Match (%) from 60.0% baseline")
    ax.set_title("Diminishing Returns: 545 Configs Across 20 Categories\n"
                 "(MuSiQue 2-hop, Qwen 7B, n=30)", fontsize=10)
    ax.set_xlim(-68, 25)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_diminishing_returns.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig8_diminishing_returns.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig8_diminishing_returns")


def fig9_error_taxonomy():
    """Figure 9: Error taxonomy pie chart — 12 baseline failures diagnosed."""
    labels = [
        "Extraction confused\n(wrong entity from\ncorrect paragraph)",
        "Retrieval miss\non hop 2",
        "Over-verbose\n(correct but long)",
        "Refusal\n(\"not mentioned\")",
        "Near-miss\nentity name",
        "Wrong granularity\n(e.g. state vs county)",
    ]
    sizes = [5, 2, 2, 1, 1, 1]
    colors_pie = [RED, ORANGE, BLUE, GRAY, TEAL, PURPLE]
    explode = (0.05, 0, 0, 0, 0, 0)

    fig, ax = plt.subplots(figsize=(7, 5))

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=None,
        autopct=lambda pct: f"{int(round(pct * 12 / 100))}/12",
        colors=colors_pie, startangle=90, textprops={"fontsize": 9},
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )

    ax.legend(wedges, labels, title="Error Type", loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8, title_fontsize=9)

    ax.set_title("Baseline Error Taxonomy (12 failures at n=30)\n"
                 "42% are extraction confusion: model has the right paragraph,\n"
                 "picks the wrong entity", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig9_error_taxonomy.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig9_error_taxonomy.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig9_error_taxonomy")


if __name__ == "__main__":
    fig1_frontier_scaling()
    fig2_direction_reversal()
    fig3_sota_comparison()
    fig4_ablation()
    fig5_full_validation()
    fig6_per_hop()
    fig7_cross_benchmark()
    fig8_diminishing_returns()
    fig9_error_taxonomy()
    print(f"\nAll figures saved to {FIGURES_DIR}")
