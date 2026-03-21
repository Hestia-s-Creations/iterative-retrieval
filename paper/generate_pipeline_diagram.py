#!/usr/bin/env python3
"""Generate the pipeline architecture diagram for the paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGURES_DIR = "figures"

# Colors
BLUE = "#4361ee"
GREEN = "#2d6a4f"
ORANGE = "#f3722c"
RED = "#d00000"
GRAY = "#6c757d"
LIGHT_GRAY = "#e9ecef"
PURPLE = "#7209b7"
TEAL = "#2ec4b6"
LIGHT_BLUE = "#d0e1ff"
LIGHT_GREEN = "#d0f0d0"
LIGHT_ORANGE = "#ffe0c0"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 300,
})


def draw_box(ax, x, y, w, h, text, color, textcolor="white", fontsize=9, bold=False):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="black", linewidth=0.8,
        zorder=3
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight=weight, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color="black", style="->", lw=1.2):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=2)


def pipeline_diagram():
    """Create the system-level decomposition pipeline diagram."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")

    # Title
    ax.text(3.5, 5.2, "System-Level Decomposition Pipeline",
            ha="center", va="center", fontsize=12, fontweight="bold")

    # === Top row: Input ===
    draw_box(ax, 1.0, 4.4, 2.8, 0.55,
             "Multi-hop Question\n+ 20 candidate paragraphs", GRAY, fontsize=8)

    # === SYSTEM LEVEL background ===
    system_bg = FancyBboxPatch(
        (0.1, 0.7), 6.8, 3.3,
        boxstyle="round,pad=0.1",
        facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.5, linestyle="--",
        alpha=0.3, zorder=0
    )
    ax.add_patch(system_bg)
    ax.text(0.35, 3.8, "SYSTEM\nLEVEL", ha="center", va="center",
            fontsize=7, color=BLUE, fontweight="bold", fontstyle="italic")

    # === MODEL LEVEL background ===
    model_bg = FancyBboxPatch(
        (3.7, 1.5), 2.5, 1.2,
        boxstyle="round,pad=0.08",
        facecolor=LIGHT_GREEN, edgecolor=GREEN, linewidth=1.2, linestyle="--",
        alpha=0.3, zorder=0
    )
    ax.add_patch(model_bg)
    ax.text(4.95, 2.55, "MODEL LEVEL", ha="center", va="center",
            fontsize=7, color=GREEN, fontweight="bold", fontstyle="italic")

    # === Step 1: Decompose ===
    draw_box(ax, 1.5, 3.3, 2.0, 0.5,
             "1. Decompose\n(Qwen 7B)", BLUE, fontsize=8, bold=True)
    draw_arrow(ax, 1.3, 4.12, 1.4, 3.57, GRAY)

    # Sub-questions output
    ax.text(1.5, 2.72, "Q1: \"Who directed X?\"\nQ2: \"Where was [ans] born?\"",
            ha="center", va="center", fontsize=6.5, color=BLUE,
            fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                     edgecolor=BLUE, alpha=0.8, linewidth=0.5))

    # === Step 2: Retrieve (for Q1) ===
    draw_box(ax, 4.0, 3.3, 1.6, 0.5,
             "2. Retrieve\n(BGE embed)", PURPLE, fontsize=8, bold=True)
    draw_arrow(ax, 2.5, 3.3, 3.2, 3.3, BLUE, lw=1.0)

    # Top-3 paragraphs
    ax.text(5.8, 3.3, "top-3\nparagraphs",
            ha="center", va="center", fontsize=7, color=PURPLE,
            fontstyle="italic")

    # === Step 3: Extract ===
    draw_box(ax, 4.95, 2.0, 1.8, 0.5,
             "3. Extract\n(8-shot LLM)", GREEN, fontsize=8, bold=True)
    draw_arrow(ax, 4.8, 3.05, 4.9, 2.27, PURPLE, lw=1.0)
    draw_arrow(ax, 5.5, 3.05, 5.1, 2.27, PURPLE, lw=1.0)

    # === Step 4: Chain ===
    draw_box(ax, 1.5, 1.5, 1.8, 0.5,
             "4. Chain\n(substitute answer)", ORANGE, fontsize=8, bold=True)
    # Arrow from Extract back to Chain
    draw_arrow(ax, 4.05, 2.0, 2.4, 1.5, GREEN, lw=1.0)
    # Arrow from Chain back to Retrieve
    draw_arrow(ax, 1.5, 1.77, 3.5, 3.05, ORANGE, lw=1.0)

    # Loop annotation
    ax.text(2.2, 2.2, "repeat\nper hop", ha="center", va="center",
            fontsize=6.5, color=ORANGE, fontstyle="italic", fontweight="bold")

    # === Step 5: Output ===
    draw_box(ax, 5.5, 0.9, 2.0, 0.45,
             "5. Final Answer", RED, fontsize=9, bold=True)
    draw_arrow(ax, 4.95, 1.73, 5.3, 1.14, GREEN, lw=1.0)

    # === Annotations ===
    # Key insight callout
    ax.text(6.5, 4.4,
            "Key: Each model call\nis single-hop extraction\n(within competence zone)",
            ha="center", va="center", fontsize=7, color=GREEN,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=LIGHT_GREEN,
                     edgecolor=GREEN, alpha=0.9, linewidth=0.8))

    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig0_pipeline.pdf", bbox_inches="tight")
    fig.savefig(f"{FIGURES_DIR}/fig0_pipeline.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated: fig0_pipeline")


if __name__ == "__main__":
    pipeline_diagram()
