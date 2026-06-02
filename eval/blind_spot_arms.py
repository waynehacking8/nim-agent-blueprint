#!/usr/bin/env python3
"""The three-arm blind-spot experiment (roadmap Phase 4, final item).

48 of 95 hallucinations escape BOTH plain 8B judges; 23 escape every gate measured so far
(plain x2, CoT x2, MiniCheck, union). Is that a CAPACITY problem (8B judges too small) or a
STRUCTURAL one (parametric judging has a ceiling)? Three arms, all scoring the SAME saved
answers (paired -> exact McNemar):

  (a) larger judge     - Llama-3.3-70B-Instruct (AWQ) as a parametric cross-family judge
  (b) judge panel      - PoLL (arXiv:2404.18796): majority vote of qwen3-8b + llama-3.1-8b +
                         phi-4 (3 families; qwen3/llama votes reused from the rows file)
  (c) grounded 8B      - llama-3.1-8b with a QUESTION-aware grounded prompt
                         (validate(question=...): the plain judge never sees the question,
                         which is exactly what near-miss hallucinations exploit)

The decisive comparison is (c) vs (a): if a question-aware-grounded 8B beats a parametric 70B,
the bottleneck is grounding, not capacity.

Usage (needs the judge endpoints served; see env below):
  NIM_70B_MODEL=... NIM_70B_URL=... NIM_PHI4_MODEL=... NIM_PHI4_URL=... \
  NIM_XJUDGE_MODEL=... NIM_XJUDGE_URL=... python eval/blind_spot_arms.py
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
REPORT = os.getenv("REPORT", "report_blind_spot_arms.md")

# arm (a): the large parametric judge
B70_MODEL = os.getenv("NIM_70B_MODEL")
B70_URL = os.getenv("NIM_70B_URL")
# arm (b): the third panel member (third model family)
PHI_MODEL = os.getenv("NIM_PHI4_MODEL")
PHI_URL = os.getenv("NIM_PHI4_URL")
# arm (c): the grounded 8B judge = the existing cross-family judge endpoint
XJUDGE_MODEL = os.getenv("NIM_XJUDGE_MODEL")
XJUDGE_URL = os.getenv("NIM_XJUDGE_URL")


def _path(name):
    return name if os.path.isabs(name) else os.path.join(HERE, name)


def run_arm(rows, passages, col, fn, label):
    """Run one judge pass over all rows, with per-row checkpointing (resume on rerun)."""
    todo = [r for r in rows if col not in r]
    if not todo:
        print(f"  {label}: already scored, skip")
        return
    print(f"  {label}: scoring {len(todo)} rows...")
    for i, r in enumerate(rows):
        if col in r:
            continue
        ctx = [passages[j] for j in r["u_ctx_idx"]]
        r[col] = not fn(r, ctx)
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            json.dump(rows, open(_path(ROWS), "w"), indent=1)
            print(f"    .. {i + 1}/{len(rows)}", flush=True)
    json.dump(rows, open(_path(ROWS), "w"), indent=1)


def main():
    rows = json.load(open(_path(ROWS)))
    if "u_answer" not in rows[0]:
        raise SystemExit("rows file lacks u_answer/u_ctx_idx - run eval/run_eval.py first")
    corpus = load_jsonl(CORPUS)
    passages = [c["text"] for c in corpus]

    # ---- the three arms (all judge the same saved unguarded answers) ----
    if B70_MODEL or B70_URL:
        run_arm(rows, passages, "u_70b_blocks",
                lambda r, ctx: validate(r["u_answer"], ctx, judge_model=B70_MODEL, judge_url=B70_URL),
                f"arm a: 70B parametric judge ({B70_MODEL})")
    if PHI_MODEL or PHI_URL:
        run_arm(rows, passages, "u_phi4_blocks",
                lambda r, ctx: validate(r["u_answer"], ctx, judge_model=PHI_MODEL, judge_url=PHI_URL),
                f"arm b3: third panel member ({PHI_MODEL})")
    if XJUDGE_MODEL or XJUDGE_URL:
        run_arm(rows, passages, "u_qa_grounded_blocks",
                lambda r, ctx: validate(r["u_answer"], ctx, judge_model=XJUDGE_MODEL,
                                        judge_url=XJUDGE_URL, question=r["q"]),
                f"arm c: question-aware grounded 8B ({XJUDGE_MODEL})")

    # arm (b): PoLL majority vote (computed from the three families' votes)
    if all(k in rows[0] for k in ("u_gate_blocks", "u_xgate_blocks", "u_phi4_blocks")):
        for r in rows:
            votes = [bool(r["u_gate_blocks"]), bool(r["u_xgate_blocks"]), bool(r["u_phi4_blocks"])]
            r["u_poll_blocks"] = sum(votes) >= 2
        json.dump(rows, open(_path(ROWS), "w"), indent=1)

    # ---- analysis ----
    halluc = [r for r in rows if r["u_halluc"]]
    blind48 = [r for r in halluc if not r["u_gate_blocks"] and not r["u_xgate_blocks"]]
    prior_keys = ["u_gate_blocks", "u_xgate_blocks", "u_cot_gate_blocks", "u_cot_xgate_blocks",
                  "u_minicheck_blocks", "u_union_blocks"]
    escape_all = [r for r in halluc if not any(r.get(k) for k in prior_keys)]

    gates = [("self 8B (plain) [baseline]", "u_gate_blocks"),
             ("cross-family 8B (plain) [baseline]", "u_xgate_blocks"),
             ("MiniCheck 770M [baseline]", "u_minicheck_blocks"),
             ("cross OR MiniCheck union [baseline]", "u_union_blocks"),
             ("(a) Llama-3.3-70B parametric", "u_70b_blocks"),
             ("(b) PoLL 3-family majority", "u_poll_blocks"),
             ("(b3) phi-4 alone", "u_phi4_blocks"),
             ("(c) question-aware grounded 8B", "u_qa_grounded_blocks")]
    gates = [(n, k) for n, k in gates if k in rows[0]]

    L = []
    w = L.append
    w("# The three-arm blind-spot experiment: capacity vs grounding\n")
    w(f"Same {len(rows)} rows / same answers as every other gate (paired, exact McNemar). "
      f"Ground truth this run: {len(halluc)} hallucinations; **{len(blind48)}** escape both "
      f"plain 8B judges; **{len(escape_all)}** escape every previously-measured gate.\n")
    w("Judges: arm (a) `casperhansen/llama-3.3-70b-instruct-awq` (the open AWQ build of "
      "Llama-3.3-70B-Instruct; the official Meta repo is license-gated), arm (b) PoLL panel = "
      "qwen3-8b + llama-3.1-8b + microsoft/phi-4 (third family; gemma-2-9b is gated), arm (c) "
      "llama-3.1-8b with the question-aware grounded prompt (`validate(question=...)`).\n")

    w("## All gates on the same hallucinations\n")
    w(f"| gate | precision | recall | F1 | of the {len(blind48)} both-8B-missed | "
      f"of the {len(escape_all)} escape-everything |")
    w("|---|---|---|---|---|---|")
    stats = {}
    for name, key in gates:
        gm = gate_metrics(rows, key)
        stats[key] = gm
        c48 = sum(1 for r in blind48 if r.get(key))
        c_all = sum(1 for r in escape_all if r.get(key))
        w(f"| {name} | {ci(gm['tp'], gm['tp'] + gm['fp'])} | {ci(gm['tp'], gm['tp'] + gm['fn'])} | "
          f"{gm['f1']:.2f} | {c48} | {c_all} |")
    w("")

    w("## Paired tests (exact McNemar, on hallucinated answers)\n")
    w("| comparison | A only | B only | both | neither | p |")
    w("|---|---|---|---|---|---|")
    pairs = [("cross 8B vs (a) 70B", "u_xgate_blocks", "u_70b_blocks"),
             ("cross 8B vs (b) PoLL", "u_xgate_blocks", "u_poll_blocks"),
             ("cross 8B vs (c) grounded", "u_xgate_blocks", "u_qa_grounded_blocks"),
             ("(a) 70B vs (c) grounded  <- the decisive one", "u_70b_blocks", "u_qa_grounded_blocks"),
             ("(a) 70B vs (b) PoLL", "u_70b_blocks", "u_poll_blocks")]
    for name, a, b in pairs:
        if a not in rows[0] or b not in rows[0]:
            continue
        mn = mcnemar(rows, a, b)
        w(f"| {name} | {mn['a_only']} | {mn['b_only']} | {mn['both']} | {mn['neither']} | {mn['p']:.4f} |")
    w("")

    # the new best combination
    if "u_70b_blocks" in rows[0] and "u_qa_grounded_blocks" in rows[0]:
        for r in rows:
            r["u_best_union_blocks"] = (bool(r.get("u_qa_grounded_blocks")) or
                                        bool(r.get("u_minicheck_blocks")) or
                                        bool(r.get("u_70b_blocks")))
        json.dump(rows, open(_path(ROWS), "w"), indent=1)
        gm = gate_metrics(rows, "u_best_union_blocks")
        resid = sum(1 for r in halluc if not r["u_best_union_blocks"])
        w(f"**Best achievable union (grounded-8B OR MiniCheck OR 70B): recall "
          f"{ci(gm['tp'], gm['tp'] + gm['fn'])}, precision {ci(gm['tp'], gm['tp'] + gm['fp'])}, "
          f"F1 {gm['f1']:.2f} — residual {resid}/{len(halluc)} hallucinations escape even this.**\n")

    w("Chart: `python eval/make_charts.py` regenerates `blind_spot_arms.png` from the rows JSON.\n")

    report_path = _path(REPORT)
    with open(report_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
