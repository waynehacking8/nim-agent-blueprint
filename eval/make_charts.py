#!/usr/bin/env python3
"""Generate result charts from the committed eval numbers.

The source of truth for every value below is eval/report.md (which is itself
produced by eval/run_eval.py). The headline ratios live only as prose / small
integer tables in report.md, so they are encoded here as constants with a
pointer to the exact report.md location. Re-run this script after re-running
run_eval.py if any number changes.

Outputs (under eval/):
  * hallucination_ablation.png — guarded vs unguarded vs residual-after-gate
  * gate_confusion.png         — validate() gate 2x2 confusion matrix

No seaborn; matplotlib only.
"""
import os

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 150

# --- Numbers from eval/report.md -------------------------------------------
# Hallucination rate on the 10 unanswerable (out-of-corpus) questions.
#   report.md "Headline" table:        guarded 0%, unguarded 40%   (lines 13-14)
#   report.md gate section:            residual after gate 30%     (line 28)
UNGUARDED_HALLUC = 40.0  # 4/10  — unguarded generator
GUARDED_HALLUC = 0.0     # 0/10  — guarded generator (first line of defense)
RESIDUAL_AFTER_GATE = 30.0  # 3/10 — unguarded run after the validate() gate

# validate() gate scored as a hallucination detector on the UNGUARDED run.
#   report.md confusion matrix (lines 24-25) and metrics (line 27).
TP, FN = 1, 3
FP, TN = 1, 21
PRECISION = TP / (TP + FP)              # 0.50
RECALL = TP / (TP + FN)                # 0.25
F1 = 2 * PRECISION * RECALL / (PRECISION + RECALL)  # 0.333...

NVIDIA_GREEN = "#76b900"
RED = "#c0392b"
AMBER = "#e08e0b"


def chart_hallucination_ablation(path: str) -> None:
    """Grouped/sequential bar chart of the core ablation, read at a glance."""
    labels = [
        "Unguarded\n(ablation)",
        "Guarded\nprompt",
        "Unguarded + validate() gate\n(residual)",
    ]
    values = [UNGUARDED_HALLUC, GUARDED_HALLUC, RESIDUAL_AFTER_GATE]
    colors = [RED, NVIDIA_GREEN, AMBER]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Hallucination rate on unanswerable questions (%)")
    ax.set_ylim(0, 50)
    ax.set_title(
        "Hallucination on out-of-corpus questions (N=10 unanswerable)\n"
        "guarded prompt → 0% · weak gate alone → 30% residual",
        fontsize=11,
        pad=12,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    counts = ["4 / 10", "0 / 10", "3 / 10"]
    for bar, val, cnt in zip(bars, values, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.0f}%\n({cnt})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def chart_gate_confusion(path: str) -> None:
    """2x2 annotated heatmap of the validate() gate confusion matrix."""
    # rows = true class (hallucinated / grounded), cols = gate decision
    matrix = [[TP, FN], [FP, TN]]
    cell_labels = [
        [f"TP\n{TP}", f"FN\n{FN}"],
        [f"FP\n{FP}", f"TN\n{TN}"],
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = LinearSegmentedColormap.from_list("blues", ["#f5f9ff", "#2c6fbb"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(TN, 1))

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["gate blocks", "gate passes"])
    ax.set_yticklabels(["hallucinated\n(should block)", "grounded/abstained\n(should pass)"])
    ax.set_xlabel("validate() gate decision")
    ax.set_ylabel("true class")

    # value annotations with contrast-aware text color
    thresh = max(TN, 1) / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cell_labels[i][j],
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white" if matrix[i][j] > thresh else "black",
            )

    ax.set_title(
        "validate() gate as a hallucination detector (unguarded run)\n"
        f"precision {PRECISION:.0%} · recall {RECALL:.0%} · F1 {F1:.2f} — "
        "weak 2nd line of defense\n(shared blind spots: judge lacks the same knowledge)",
        fontsize=10,
        pad=16,
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    # Reserve headroom so the 3-line title is not cramped against the heatmap.
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main() -> None:
    abl = os.path.join(HERE, "hallucination_ablation.png")
    conf = os.path.join(HERE, "gate_confusion.png")
    chart_hallucination_ablation(abl)
    chart_gate_confusion(conf)
    print(f"wrote {abl}")
    print(f"wrote {conf}")


if __name__ == "__main__":
    main()
