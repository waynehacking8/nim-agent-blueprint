#!/usr/bin/env python3
"""Judge-variant comparison on the saved eval rows (roadmap Phase 5).

Re-scores the SAME unguarded answers (temp=0 => identical generations) with:
  * self judge + CoT            (arXiv:2511.11087 - published: recall ~22% -> ~58%)
  * cross-family judge + CoT
  * MiniCheck-Flan-T5-Large     (arXiv:2404.10774 - grounded NLI verifier; scored on a GPU
                                 by eval/minicheck_score.py -> minicheck_rows.json, merged here)

Baselines (plain self / cross-family judge verdicts) come from the rows file itself, so
every arm is paired on the same answers and McNemar-testable against every other arm.

The key diagnostic is the shared-blind-spot set: hallucinations missed by BOTH plain 8B
judges. For each new arm we report how many of those it recovers - that is the question
Phase 5 exists to answer (grounding vs capacity).

Usage (CoT passes need the same env as run_eval.py: NIM_* + NIM_XJUDGE_*):
  python eval/judge_variants.py
Env:
  ROWS=report_squad_rows.json   rows produced by run_eval.py (with u_answer / u_ctx_idx)
  EVAL_CORPUS=corpus_squad.jsonl
  REPORT=report_judge_variants.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.agent import validate
from run_eval import ci, gate_metrics, load_jsonl, mcnemar

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.getenv("ROWS", "report_squad_rows.json")
CORPUS = os.getenv("EVAL_CORPUS", "corpus_squad.jsonl")
REPORT = os.getenv("REPORT", "report_judge_variants.md")
MINICHECK = os.getenv("MINICHECK_ROWS", "minicheck_rows.json")
XJUDGE_MODEL = os.getenv("NIM_XJUDGE_MODEL")
XJUDGE_URL = os.getenv("NIM_XJUDGE_URL")


def _path(name):
    return name if os.path.isabs(name) else os.path.join(HERE, name)


def run_cot_passes(rows, passages):
    """Add u_cot_gate_blocks / u_cot_xgate_blocks to every row (idempotent - skips rows
    already scored, so an interrupted run resumes instead of repaying the network cost)."""
    for i, r in enumerate(rows):
        ctx = [passages[j] for j in r["u_ctx_idx"]]
        if "u_cot_gate_blocks" not in r:
            r["u_cot_gate_blocks"] = not validate(r["u_answer"], ctx, cot=True)
        if "u_cot_xgate_blocks" not in r and (XJUDGE_MODEL or XJUDGE_URL):
            r["u_cot_xgate_blocks"] = not validate(r["u_answer"], ctx, judge_model=XJUDGE_MODEL,
                                                   judge_url=XJUDGE_URL, cot=True)
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            print(f"  .. CoT judge {i + 1}/{len(rows)}", flush=True)
            # checkpoint so progress survives interruption
            json.dump(rows, open(_path(ROWS), "w"), indent=1)
    return rows


def merge_minicheck(rows):
    """Merge MiniCheck predictions (eval/minicheck_score.py output) if present."""
    p = _path(MINICHECK)
    if not os.path.exists(p):
        print(f"  (no {MINICHECK} - MiniCheck column skipped; run eval/minicheck_score.py on a GPU host)")
        return False
    mc = json.load(open(p))
    if len(mc) != len(rows):
        raise SystemExit(f"minicheck rows ({len(mc)}) != eval rows ({len(rows)}) - stale file?")
    for r, m in zip(rows, mc):
        # MiniCheck label 1 = claim supported by document; gate blocks when unsupported.
        r["u_minicheck_blocks"] = (m["pred"] == 0)
        r["u_minicheck_prob"] = m["prob"]
    return True


def main():
    rows = json.load(open(_path(ROWS)))
    if "u_answer" not in rows[0]:
        raise SystemExit("rows file lacks u_answer/u_ctx_idx - re-run eval/run_eval.py first")
    corpus = load_jsonl(CORPUS)
    passages = [c["text"] for c in corpus]

    rows = run_cot_passes(rows, passages)
    has_mc = merge_minicheck(rows)

    gates = [("self (plain)", "u_gate_blocks"),
             ("cross-family (plain)", "u_xgate_blocks"),
             ("self + CoT", "u_cot_gate_blocks"),
             ("cross-family + CoT", "u_cot_xgate_blocks")]
    if has_mc:
        gates.append(("MiniCheck-FT5 770M (grounded NLI)", "u_minicheck_blocks"))
        # the actionable combination: generative judge (parametric knowledge) OR grounded
        # NLI (evidence check) - they catch different hallucinations, so the union is the
        # natural deployment configuration for validate()
        for r in rows:
            r["u_union_blocks"] = bool(r["u_xgate_blocks"]) or bool(r["u_minicheck_blocks"])
        gates.append(("cross-family OR MiniCheck (union)", "u_union_blocks"))
    # persist every derived column so make_charts.py can regenerate charts with zero diff
    json.dump(rows, open(_path(ROWS), "w"), indent=1)

    una = [r for r in rows if not r["answerable"]]
    n_halluc = sum(r["u_halluc"] for r in una)
    # the shared blind spots: hallucinations BOTH plain 8B judges missed
    blind = [r for r in rows if r["u_halluc"] and not r["u_gate_blocks"] and not r["u_xgate_blocks"]]

    L = []
    w = L.append
    w("# Judge variants - CoT judging and grounded NLI vs the plain 8B judges\n")
    w(f"Same {len(rows)} SQuAD 2.0 rows / same unguarded answers (temp=0) as "
      "`report_squad.md`; only the judge differs, so all comparisons are paired "
      "(exact McNemar).\n")
    w(f"Ground truth: {n_halluc} hallucinations among {len(una)} unanswerable questions; "
      f"**{len(blind)}** of them escape BOTH plain 8B judges (the shared-blind-spot set).\n")

    w("## Gate metrics (hallucination detection on the unguarded run)\n")
    w("| gate | precision | recall | F1 | catches of the "
      f"{len(blind)} shared blind spots |")
    w("|---|---|---|---|---|")
    stats = {}
    for name, key in gates:
        gm = gate_metrics(rows, key)
        stats[key] = gm
        catches = sum(1 for r in blind if r[key])
        w(f"| {name} | {ci(gm['tp'], gm['tp'] + gm['fp'])} | "
          f"{ci(gm['tp'], gm['tp'] + gm['fn'])} | {gm['f1']:.2f} | {catches} |")
    w("")

    w("## Paired tests (exact McNemar, two-sided, on hallucinated answers)\n")
    w("| comparison | A only | B only | both | neither | p |")
    w("|---|---|---|---|---|---|")
    pairs = [("self plain vs self+CoT", "u_gate_blocks", "u_cot_gate_blocks"),
             ("cross plain vs cross+CoT", "u_xgate_blocks", "u_cot_xgate_blocks"),
             ("self+CoT vs cross+CoT", "u_cot_gate_blocks", "u_cot_xgate_blocks")]
    if has_mc:
        pairs += [("cross plain vs MiniCheck", "u_xgate_blocks", "u_minicheck_blocks"),
                  ("cross+CoT vs MiniCheck", "u_cot_xgate_blocks", "u_minicheck_blocks")]
    for name, a, b in pairs:
        mn = mcnemar(rows, a, b)
        w(f"| {name} | {mn['a_only']} | {mn['b_only']} | {mn['both']} | {mn['neither']} | "
          f"{mn['p']:.4f} |")
    w("")

    # the honest closing number: how many hallucinations survive EVERY gate measured here
    all_keys = [k for _, k in gates]
    halluc_rows = [r for r in rows if r["u_halluc"]]
    residual = sum(1 for r in halluc_rows if not any(r.get(k) for k in all_keys))
    w(f"**Residual blind spots: {residual} of {len(halluc_rows)} hallucinations escape EVERY "
      "gate above** (both plain judges, both CoT judges, MiniCheck, and the union). This is "
      "the floor that the next method class (a 70B judge, retrieval-aware generation, or "
      "semantic-entropy filtering) has to attack.\n")

    w("## Literature comparison\n")
    w("| claim | published | measured here |")
    w("|---|---|---|")
    sp, sc = stats["u_gate_blocks"], stats["u_cot_gate_blocks"]
    xp, xc = stats["u_xgate_blocks"], stats["u_cot_xgate_blocks"]
    w(f"| CoT raises self-detection recall (arXiv:2511.11087) | ~22% -> ~58% | "
      f"{sp['rec']:.0%} -> {sc['rec']:.0%} |")
    w(f"| CoT on the cross-family judge | (not reported) | {xp['rec']:.0%} -> {xc['rec']:.0%} |")
    if has_mc:
        mc = stats["u_minicheck_blocks"]
        w(f"| MiniCheck 770M ~ GPT-4 level grounded verification (arXiv:2404.10774) | "
          f"77.4% bal. acc. on LLM-AGGREFACT | recall {mc['rec']:.0%} / precision {mc['prec']:.0%} "
          f"on this set |")
    w("")
    w("Charts: `python eval/make_charts.py` regenerates `judge_variants.png` from the rows JSON.\n")

    report_path = _path(REPORT)
    with open(report_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
