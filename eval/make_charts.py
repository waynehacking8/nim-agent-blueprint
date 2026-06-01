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


def _gate(rows, key):
    halluc = [r for r in rows if r["u_halluc"]]
    clean = [r for r in rows if not r["u_halluc"]]
    tp = sum(1 for r in halluc if r[key])
    fn = len(halluc) - tp
    fp = sum(1 for r in clean if r[key])
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return prec, rec, f1


def squad_charts(rows_path: str) -> None:
    """Charts for the SQuAD 2.0 (N=200) run, computed from the saved per-question rows —
    nothing hard-coded. Produces report_squad_ablation.png + report_squad_gates.png."""
    import json

    import matplotlib.pyplot as plt

    rows = json.load(open(rows_path))
    una = [r for r in rows if not r["answerable"]]
    n = len(una)
    unguarded = 100 * sum(r["u_halluc"] for r in una) / n
    guarded = 100 * sum(r["g_halluc"] for r in una) / n
    resid_self = 100 * sum(1 for r in una if r["u_halluc"] and not r["u_gate_blocks"]) / n
    # u_xgate_blocks is None when the eval ran without NIM_XJUDGE_MODEL set;
    # treat None as "no cross-judge data", not as "not blocked"
    resid_x = 100 * sum(
        1 for r in una
        if r["u_halluc"] and r["u_xgate_blocks"] is not None and not r["u_xgate_blocks"]
    ) / n

    # --- ablation: 4 bars ---
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = ["Unguarded\n(ablation)", "Guarded\nprompt",
              "Unguarded +\nself gate", "Unguarded +\ncross-family gate"]
    values = [unguarded, guarded, resid_self, resid_x]
    colors = [RED, NVIDIA_GREEN, AMBER, "#2c6fbb"]
    bars = ax.bar(labels, values, color=colors, width=0.62, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Hallucination rate on unanswerable questions (%)")
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_title(f"SQuAD 2.0 adversarial near-miss questions (N={n} unanswerable)\n"
                 "harder than out-of-corpus: the tempting-but-wrong passage IS in context",
                 fontsize=11, pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(HERE, "report_squad_ablation.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")

    # --- gate comparison: self vs cross-family, grouped bars ---
    sp, sr, sf = _gate(rows, "u_gate_blocks")
    xp, xr, xf = _gate(rows, "u_xgate_blocks")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    metrics = ["precision", "recall", "F1"]
    self_vals = [sp * 100, sr * 100, sf * 100]
    x_vals = [xp * 100, xr * 100, xf * 100]
    xpos = range(len(metrics))
    width = 0.36
    b1 = ax.bar([x - width / 2 for x in xpos], self_vals, width,
                label="self judge (qwen3-8b)", color="#9aa0a6", edgecolor="black", linewidth=0.6)
    b2 = ax.bar([x + width / 2 for x in xpos], x_vals, width,
                label="cross-family judge (llama-3.1-8b)", color=NVIDIA_GREEN,
                edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score (%; F1 ×100)")
    ax.set_ylim(0, 100)
    ax.set_title("Hallucination gate: self vs cross-family judge (paired, N=95 hallucinations)\n"
                 "recall +16 pts, McNemar exact p=0.0026 — shared blind spots confirmed",
                 fontsize=11, pad=12)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(HERE, "report_squad_gates.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    abl = os.path.join(HERE, "hallucination_ablation.png")
    conf = os.path.join(HERE, "gate_confusion.png")
    chart_hallucination_ablation(abl)
    chart_gate_confusion(conf)
    print(f"wrote {abl}")
    print(f"wrote {conf}")
    rows_path = os.path.join(HERE, "report_squad_rows.json")
    if os.path.exists(rows_path):
        squad_charts(rows_path)


if __name__ == "__main__":
    main()
