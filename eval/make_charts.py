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
    # title stats computed from the rows (never hard-coded: re-running the eval must
    # regenerate this chart with zero manual edits)
    from math import comb
    halluc = [r for r in rows if r["u_halluc"]]
    a_only = sum(1 for r in halluc if r["u_gate_blocks"] and not r["u_xgate_blocks"])
    b_only = sum(1 for r in halluc if not r["u_gate_blocks"] and r["u_xgate_blocks"])
    nd = a_only + b_only
    p_mcnemar = (min(1.0, sum(comb(nd, k) for k in range(min(a_only, b_only) + 1)) * 2 / 2 ** nd)
                 if nd else 1.0)
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
    ax.set_title(f"Hallucination gate: self vs cross-family judge (paired, N={len(halluc)} "
                 "hallucinations)\n"
                 f"recall {(xr - sr) * 100:+.0f} pts, McNemar exact p={p_mcnemar:.4f} — "
                 "shared blind spots confirmed",
                 fontsize=11, pad=12)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(HERE, "report_squad_gates.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def judge_variants_chart(rows_path: str) -> None:
    """All hallucination gates measured in this repo on one chart (roadmap Phase 5):
    plain judges, CoT judges, MiniCheck grounded NLI, and the deployment union."""
    import json

    rows = json.load(open(rows_path))
    if "u_cot_gate_blocks" not in rows[0]:
        return  # CoT pass not run yet
    gates = [("self\n(plain)", "u_gate_blocks", "#9aa0a6"),
             ("cross-family\n(plain)", "u_xgate_blocks", "#6b7280"),
             ("self\n+ CoT", "u_cot_gate_blocks", AMBER),
             ("cross-family\n+ CoT", "u_cot_xgate_blocks", "#b45309"),
             ("MiniCheck 770M\n(grounded NLI)", "u_minicheck_blocks", "#2c6fbb"),
             ("cross OR MiniCheck\n(union)", "u_union_blocks", NVIDIA_GREEN)]
    gates = [(n, k, c) for n, k, c in gates if k in rows[0]]
    halluc = [r for r in rows if r["u_halluc"]]
    # shared blind spots of the two PLAIN judges - the set Phase 5 attacks
    blind = [r for r in halluc if not r["u_gate_blocks"] and not r["u_xgate_blocks"]]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5), width_ratios=[3, 2])
    xpos = range(len(gates))
    width = 0.38
    recs = [_gate(rows, k)[1] * 100 for _, k, _ in gates]
    precs = [_gate(rows, k)[0] * 100 for _, k, _ in gates]
    b1 = ax.bar([x - width / 2 for x in xpos], recs, width, label="recall",
                color=[c for _, _, c in gates], edgecolor="black", linewidth=0.6)
    b2 = ax.bar([x + width / 2 for x in xpos], precs, width, label="precision",
                color=[c for _, _, c in gates], alpha=0.45, edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
    ax.set_xticks(list(xpos))
    ax.set_xticklabels([n for n, _, _ in gates], fontsize=9)
    ax.set_ylabel("score (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Every gate vs the same {len(halluc)} hallucinations\n"
                 "solid = recall · translucent = precision", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # right panel: who catches the shared blind spots?
    catch = [sum(1 for r in blind if r[k]) for _, k, _ in gates]
    bars = ax2.barh([n.replace("\n", " ") for n, _, _ in gates], catch,
                    color=[c for _, _, c in gates], edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, catch):
        ax2.text(v + 0.4, bar.get_y() + bar.get_height() / 2, str(v), va="center",
                 fontsize=10, fontweight="bold")
    ax2.set_xlabel(f"caught of the {len(blind)} hallucinations BOTH plain 8B judges miss")
    ax2.set_xlim(0, max(catch) * 1.25 if any(catch) else 1)
    ax2.set_title("Shared-blind-spot recovery\n(grounding > more parametric judging)", fontsize=11)
    ax2.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.invert_yaxis()
    fig.tight_layout()
    out = os.path.join(HERE, "judge_variants.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def rerank_chart(base_path: str, rerank_path: str) -> None:
    """Fusion-only vs fusion+reranker, paired on the same questions (roadmap Phase 5)."""
    import json

    base = json.load(open(base_path))
    rer = json.load(open(rerank_path))
    ans_b = [r for r in base if r["answerable"]]
    ans_r = [r for r in rer if r["answerable"]]
    una_b = [r for r in base if not r["answerable"]]
    una_r = [r for r in rer if not r["answerable"]]

    metrics = ["retrieval\nrecall@3", "answer accuracy\n(LLM judge)",
               "hallucination\nguarded", "hallucination\nunguarded"]
    base_vals = [100 * sum(r["recall"] for r in ans_b) / len(ans_b),
                 100 * sum(r["correct_llm"] for r in ans_b) / len(ans_b),
                 100 * sum(r["g_halluc"] for r in una_b) / len(una_b),
                 100 * sum(r["u_halluc"] for r in una_b) / len(una_b)]
    rer_vals = [100 * sum(r["recall"] for r in ans_r) / len(ans_r),
                100 * sum(r["correct_llm"] for r in ans_r) / len(ans_r),
                100 * sum(r["g_halluc"] for r in una_r) / len(una_r),
                100 * sum(r["u_halluc"] for r in una_r) / len(una_r)]

    fig, ax = plt.subplots(figsize=(9, 5))
    xpos = range(len(metrics))
    width = 0.36
    b1 = ax.bar([x - width / 2 for x in xpos], base_vals, width, label="fusion retrieval only",
                color="#9aa0a6", edgecolor="black", linewidth=0.6)
    b2 = ax.bar([x + width / 2 for x in xpos], rer_vals, width,
                label="fusion + bge-reranker-v2-m3", color=NVIDIA_GREEN,
                edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
    # divider: left half = "higher is better", right half = "lower is better"
    ax.axvline(1.5, color="black", linestyle=":", linewidth=1)
    ax.text(0.5, 105, "higher = better", ha="center", fontsize=10, style="italic")
    ax.text(2.5, 105, "lower = better", ha="center", fontsize=10, style="italic")
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("%")
    ax.set_ylim(0, 112)
    ax.set_title("What the reranker buys - and what it does not\n"
                 "retrieval and accuracy improve; hallucination on adversarial near-miss "
                 "questions does NOT",
                 fontsize=11, pad=12)
    ax.legend(loc="center left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(HERE, "rerank_compare.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def semantic_entropy_chart(scores_path: str) -> None:
    """Semantic entropy AUROC vs the published Nature 2024 ceiling (roadmap Phase 5).

    The headline is the SLICE difference: the signal works on answerable questions and
    collapses to chance on adversarial near-miss questions (systematic confident errors,
    which the paper explicitly excludes from semantic entropy's scope)."""
    import json

    scores = json.load(open(scores_path))

    def auroc(rows, key):
        labels = [r["u_halluc"] for r in rows]
        vals = [r[key] for r in rows]
        pos = [v for v, l in zip(vals, labels) if l]
        neg = [v for v, l in zip(vals, labels) if not l]
        if not pos or not neg:
            return float("nan")
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return wins / (len(pos) * len(neg))

    ans = [r for r in scores if r["answerable"]]
    una = [r for r in scores if not r["answerable"]]
    slices = [(f"full set\n(N={len(scores)})", scores),
              (f"answerable\n(N={len(ans)})", ans),
              (f"adversarial near-miss\n(N={len(una)})", una)]
    methods = [("semantic entropy", "semantic_entropy", NVIDIA_GREEN),
               ("discrete SE", "discrete_se", "#2c6fbb"),
               ("naive entropy", "naive_entropy", "#9aa0a6")]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    width = 0.26
    for mi, (mname, key, color) in enumerate(methods):
        xs = [i + (mi - 1) * width for i in range(len(slices))]
        vals = [auroc(rows, key) for _, rows in slices]
        bars = ax.bar(xs, vals, width, label=mname, color=color,
                      edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    # published reference lines (Farquhar et al., Nature 2024, avg across tasks/models)
    ax.axhline(0.790, color=NVIDIA_GREEN, linestyle="--", linewidth=1.2)
    ax.text(2.38, 0.795, "published SE 0.790", ha="right", fontsize=9, color="#3a7000")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.text(2.38, 0.505, "chance", ha="right", fontsize=9)
    ax.set_xticks(range(len(slices)))
    ax.set_xticklabels([s for s, _ in slices])
    ax.set_ylabel("AUROC (hallucination detection)")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Semantic entropy detects confabulations, not confident errors\n"
                 "works on answerable questions - collapses to chance exactly where the "
                 "questions are designed to fool the model", fontsize=11, pad=12)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(HERE, "semantic_entropy.png")
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
        judge_variants_chart(rows_path)
    rerank_path = os.path.join(HERE, "report_squad_rerank_rows.json")
    if os.path.exists(rows_path) and os.path.exists(rerank_path):
        rerank_chart(rows_path, rerank_path)
    se_path = os.path.join(HERE, "semantic_entropy_scores.json")
    if os.path.exists(se_path):
        semantic_entropy_chart(se_path)


if __name__ == "__main__":
    main()
