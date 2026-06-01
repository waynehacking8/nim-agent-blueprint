#!/usr/bin/env python3
"""Evaluate the agentic-RAG blueprint on the signals an SA would actually show a partner.

Two conditions per question (ablation):
  * guarded   — the generator is told to answer only from context and abstain otherwise.
  * unguarded — that instruction is removed, so the generator will happily make things up.

Metrics (each with a 95% Wilson confidence interval):
  1. Retrieval recall@k.
  2. Answer accuracy (guarded, answerable) — substring AND LLM-judge (semantic).
  3. Hallucination rate, guarded vs unguarded, on UNANSWERABLE questions.
  4. The validate() groundedness gate as a hallucination detector on the *unguarded* run:
     precision / recall / F1 — for the SELF judge (same model as the generator) and,
     when NIM_XJUDGE_MODEL/NIM_XJUDGE_URL are set, an independent CROSS-FAMILY judge.
  5. Per-hop latency budget.

Eval profiles (env-selected so the original demo set keeps working):
  EVAL_CORPUS / EVAL_DATASET / EVAL_REPORT — default corpus.jsonl / dataset.jsonl / report.md;
  point at corpus_squad.jsonl / dataset_squad.jsonl (built by build_squad_eval.py) for the
  200-question SQuAD 2.0 run.

Runs against any OpenAI-compatible LLM + embedding endpoint via app/provider.py.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.retriever import HybridRetriever
from app.agent import answer_traced, validate
from app import provider

HERE = os.path.dirname(os.path.abspath(__file__))
K = int(os.getenv("EVAL_K", "3"))
CORPUS = os.getenv("EVAL_CORPUS", "corpus.jsonl")
DATASET = os.getenv("EVAL_DATASET", "dataset.jsonl")
REPORT = os.getenv("EVAL_REPORT", "report.md")
# Optional independent judge from a different model family (roadmap Phase 4): when set,
# the gate is scored twice — self judge vs cross-family judge — on the same answers.
XJUDGE_MODEL = os.getenv("NIM_XJUDGE_MODEL")
XJUDGE_URL = os.getenv("NIM_XJUDGE_URL")
# Accuracy judge (semantic match). Defaults to the cross-family judge so the model is not
# grading its own answers; falls back to the generator endpoint.
ACC_JUDGE_MODEL = os.getenv("NIM_ACC_JUDGE_MODEL", XJUDGE_MODEL)
ACC_JUDGE_URL = os.getenv("NIM_ACC_JUDGE_URL", XJUDGE_URL)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def ci(k, n):
    lo, hi = wilson(k, n)
    return f"{k / n:.0%} ({k}/{n}, 95% CI {lo:.0%}–{hi:.0%})" if n else "n/a"


def load_jsonl(name):
    path = name if os.path.isabs(name) else os.path.join(HERE, name)
    return [json.loads(l) for l in open(path) if l.strip()]


def golds(ex):
    return [g.lower() for g in ex.get("gold_alternatives") or [ex["gold"]] if g]


def substring_correct(ex, out):
    return (ex["answerable"] and not out["abstained"]
            and any(g in out["answer"].lower() for g in golds(ex)))


def llm_accuracy_judge(ex, out):
    """Semantic accuracy via LLM judge — robust to phrasing, unlike the substring check."""
    if not ex["answerable"] or out["abstained"]:
        return False
    verdict = provider.chat(
        [{"role": "system", "content":
          "You grade question answering. Reply with exactly one word: correct if the "
          "RESPONSE conveys the same answer as the REFERENCE, otherwise wrong."},
         {"role": "user", "content":
          f"QUESTION: {ex['q']}\nREFERENCE: {ex['gold']}\nRESPONSE: {out['answer']}"}],
        max_tokens=8, model=ACC_JUDGE_MODEL or None, base_url=ACC_JUDGE_URL or None)
    v = verdict.strip().lower()
    return "correct" in v and "incorrect" not in v and "wrong" not in v


def is_hallucination(ex, out):
    """An answer that asserts something (didn't abstain) yet isn't a correct grounded answer."""
    if out["abstained"]:
        return False
    if ex["answerable"]:
        return not any(g in out["answer"].lower() for g in golds(ex))
    return True  # answered an unanswerable question


def gate_metrics(rows, key):
    """Precision/recall/F1 of a gate column `key` against u_halluc ground truth."""
    tp = sum(1 for r in rows if r["u_halluc"] and r[key])
    fn = sum(1 for r in rows if r["u_halluc"] and not r[key])
    fp = sum(1 for r in rows if not r["u_halluc"] and r[key])
    tn = sum(1 for r in rows if not r["u_halluc"] and not r[key])
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "prec": prec, "rec": rec, "f1": f1}


def mcnemar(rows, key_a, key_b):
    """Exact McNemar test (paired, two-sided) comparing two gates on hallucinated answers.

    The two judges score the SAME answers, so the paired test is the right one — it only
    looks at discordant pairs (caught by exactly one judge).
    """
    from math import comb
    halluc = [r for r in rows if r["u_halluc"]]
    a_only = sum(1 for r in halluc if r[key_a] and not r[key_b])
    b_only = sum(1 for r in halluc if not r[key_a] and r[key_b])
    both = sum(1 for r in halluc if r[key_a] and r[key_b])
    neither = sum(1 for r in halluc if not r[key_a] and not r[key_b])
    n = a_only + b_only
    p = min(1.0, sum(comb(n, k) for k in range(min(a_only, b_only) + 1)) * 2 / 2 ** n) if n else 1.0
    return {"a_only": a_only, "b_only": b_only, "both": both, "neither": neither, "p": p}


def main():
    corpus = load_jsonl(CORPUS)
    passages = [c["text"] for c in corpus]
    id_to_idx = {c["id"]: i for i, c in enumerate(corpus)}
    retriever = HybridRetriever(passages)
    data = load_jsonl(DATASET)
    dual_judge = bool(XJUDGE_MODEL or XJUDGE_URL)

    rows, hops = [], {"plan": [], "retrieve": [], "generate": [], "validate": []}
    for i, ex in enumerate(data):
        g = answer_traced(ex["q"], retriever, k=K, guarded=True)
        u = answer_traced(ex["q"], retriever, k=K, guarded=False)
        for h in hops:
            hops[h].append(g["latency"][h])
        # cross-family judge re-scores the SAME unguarded answer (temp=0 ⇒ identical
        # generations), isolating the judge as the only variable.
        u_ctx = [retriever.passages[j] for j in u["retrieved_idx"]]
        x_blocks = (not validate(u["answer"], u_ctx, judge_model=XJUDGE_MODEL,
                                 judge_url=XJUDGE_URL)) if dual_judge else None
        recall = (id_to_idx.get(ex["gold_passage"]) in g["retrieved_idx"]) if ex["answerable"] else None
        rows.append({
            "q": ex["q"], "answerable": ex["answerable"], "recall": recall,
            "correct": substring_correct(ex, g),
            "correct_llm": llm_accuracy_judge(ex, g),
            "g_abstain": g["abstained"], "u_abstain": u["abstained"],
            "g_halluc": is_hallucination(ex, g), "u_halluc": is_hallucination(ex, u),
            "u_gate_blocks": (not u["grounded"]), "u_xgate_blocks": x_blocks,
            "lat": round(g["latency_total"], 2)})
        if (i + 1) % 20 == 0 or i + 1 == len(data):
            print(f"  .. {i + 1}/{len(data)} questions", flush=True)

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    n_ans, n_una = len(ans), len(una)

    n_recall = sum(r["recall"] for r in ans)
    n_correct = sum(r["correct"] for r in ans)
    n_correct_llm = sum(r["correct_llm"] for r in ans)
    n_g_halluc = sum(r["g_halluc"] for r in una)
    n_u_halluc = sum(r["u_halluc"] for r in una)

    self_gate = gate_metrics(rows, "u_gate_blocks")
    x_gate = gate_metrics(rows, "u_xgate_blocks") if dual_judge else None
    n_resid = sum(1 for r in una if r["u_halluc"] and not r["u_gate_blocks"])
    n_xresid = sum(1 for r in una if r["u_halluc"] and not r["u_xgate_blocks"]) if dual_judge else None

    is_squad = "squad" in os.path.basename(CORPUS).lower()
    L = []
    w = L.append
    w("# Agentic-RAG eval — retrieval, hallucination gating, judge calibration, latency\n")
    if is_squad:
        w(f"Corpus: {len(corpus)} SQuAD 2.0 dev passages · {n_ans} answerable + {n_una} "
          "unanswerable questions (SQuAD 2.0's unanswerable questions are crowdworker-written "
          "adversarial near-misses: the on-topic paragraph IS in the corpus but does not "
          f"contain the answer). Self-hosted on H100: vLLM generator + Ollama embeddings. "
          f"k={K}, temp=0. All percentages carry 95% Wilson confidence intervals.\n")
    else:
        w(f"Corpus {len(corpus)} passages · {n_ans} answerable + {n_una} unanswerable "
          "(incl. adversarial near-miss). Self-hosted on a single H100: vLLM generator + "
          f"Ollama embeddings. k={K}, temp=0.\n")
        w(f"> **Illustrative, not a statistical benchmark (N={len(rows)}).** "
          "See the SQuAD 2.0 profile (`make eval-squad`) for the statistically powered run.\n")

    w("## Headline\n")
    w("| metric | value |")
    w("|---|---|")
    w(f"| retrieval recall@{K} (answerable) | {ci(n_recall, n_ans)} |")
    w(f"| answer accuracy — substring (guarded, answerable) | {ci(n_correct, n_ans)} |")
    w(f"| answer accuracy — LLM judge (guarded, answerable) | {ci(n_correct_llm, n_ans)} |")
    w(f"| hallucination on unanswerable — **guarded** generator | **{ci(n_g_halluc, n_una)}** |")
    w(f"| hallucination on unanswerable — **unguarded** generator | **{ci(n_u_halluc, n_una)}** |")
    w("")
    w("The guarded prompt (\"answer only from context, else say you don't know\") is the "
      "first line of defense. The ablation shows what happens without it.\n")
    w(f"![Hallucination ablation]({os.path.basename(REPORT).replace('.md', '')}_ablation.png)\n")

    w("## The validate() groundedness gate as a safety net (unguarded run)\n")
    w("Second line of defense: an LLM-as-judge checks each answer against the retrieved "
      "context and blocks unsupported ones. Scored on the unguarded run, where "
      "hallucinations exist to catch.\n")

    def gate_block(name, gm, resid, n_unans):
        w(f"### {name}\n")
        w("| | gate blocks | gate passes |")
        w("|---|---|---|")
        w(f"| hallucinated (should block) | TP={gm['tp']} | FN={gm['fn']} |")
        w(f"| grounded/abstained (should pass) | FP={gm['fp']} | TN={gm['tn']} |")
        w("")
        w(f"- precision **{ci(gm['tp'], gm['tp'] + gm['fp'])}** · "
          f"recall **{ci(gm['tp'], gm['tp'] + gm['fn'])}** · F1 **{gm['f1']:.2f}**")
        w(f"- residual hallucination on unanswerable after gating: "
          f"**{ci(resid, n_unans)}** (down from {ci(n_u_halluc, n_unans)})\n")

    gate_block("Self judge (same model/endpoint as the generator)", self_gate, n_resid, n_una)
    if dual_judge:
        gate_block(f"Cross-family judge ({XJUDGE_MODEL})", x_gate, n_xresid, n_una)
        delta = (x_gate["rec"] - self_gate["rec"]) * 100
        mn = mcnemar(rows, "u_gate_blocks", "u_xgate_blocks")
        w("### Self vs cross-family judge — the shared-blind-spots test\n")
        w("Hypothesis: a judge that is the same model that hallucinated misses the same "
          "facts it got wrong (shared blind spots), so an independent judge from a "
          "different model family should catch more.\n")
        w("| judge | precision | recall | F1 |")
        w("|---|---|---|---|")
        w(f"| self ({os.getenv('NIM_LLM_MODEL', 'generator')}) | {self_gate['prec']:.0%} | "
          f"{self_gate['rec']:.0%} | {self_gate['f1']:.2f} |")
        w(f"| cross-family ({XJUDGE_MODEL}) | {x_gate['prec']:.0%} | {x_gate['rec']:.0%} | "
          f"{x_gate['f1']:.2f} |")
        w("")
        w(f"Recall delta: **{delta:+.0f} points**, and the comparison is *paired* (both judges "
          "score the same answers), so an exact McNemar test applies: of the hallucinations "
          f"caught by exactly one judge, the cross-family judge caught {mn['b_only']} vs the "
          f"self judge's {mn['a_only']} — **p = {mn['p']:.4f}**. Cross-model detection "
          "literature (e.g. FINCH-ZK, arXiv:2508.14314: detection F1 improved by 6–39% on "
          "FELM from cross-model "
          "consistency) predicts this gain when blind spots, not judging ability, are the "
          "bottleneck.\n")
        w(f"The equally important honest number: **{mn['neither']}** of the "
          f"{mn['both'] + mn['neither'] + mn['a_only'] + mn['b_only']} hallucinations were "
          "caught by *neither* 8B judge — two small same-size judges still share most blind "
          "spots with each other. The next rungs are a larger judge, a judge panel (PoLL), or "
          "retrieval-grounded verification.\n")
    w(f"![Gate comparison]({os.path.basename(REPORT).replace('.md', '')}_gates.png)\n")

    w("## Per-hop latency budget (mean seconds, guarded)\n")
    m = {h: statistics.mean(v) for h, v in hops.items()}
    w("| plan | retrieve | generate | validate | total |")
    w("|---|---|---|---|---|")
    w(f"| {m['plan']:.2f} | {m['retrieve']:.3f} | {m['generate']:.2f} | {m['validate']:.2f} | "
      f"{sum(m.values()):.2f} |")
    w("")

    w("## Per-question detail\n")
    cols = "| question | answerable | recall@k | correct (LLM) | unguarded halluc | self gate | x-gate |"
    w(cols)
    w("|" + "---|" * (cols.count("|") - 1))
    for r in rows:
        w(f"| {r['q'][:46]} | {r['answerable']} | {r['recall']} | {r['correct_llm']} | "
          f"{r['u_halluc']} | {r['u_gate_blocks']} | {r['u_xgate_blocks']} |")
    w("")

    report_path = REPORT if os.path.isabs(REPORT) else os.path.join(HERE, REPORT)
    with open(report_path, "w") as f:
        f.write("\n".join(L) + "\n")
    # raw rows for re-analysis / charting without re-running the GPU eval
    with open(report_path.replace(".md", "_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\n".join(L[:40]))
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
