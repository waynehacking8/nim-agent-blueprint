#!/usr/bin/env python3
"""Evaluate the agentic-RAG blueprint on the signals an SA would actually show a partner.

Two conditions per question (ablation):
  * guarded   — the generator is told to answer only from context and abstain otherwise.
  * unguarded — that instruction is removed, so the generator will happily make things up.

Metrics:
  1. Retrieval recall@k.
  2. Answer accuracy (guarded, answerable).
  3. Hallucination rate, guarded vs unguarded, on UNANSWERABLE (out-of-corpus) questions.
  4. The validate() groundedness gate as a hallucination detector on the *unguarded* run
     (that is where hallucinations exist to catch): precision / recall / F1 + residual
     hallucination rate after gating.
  5. Per-hop latency budget.

Runs against any OpenAI-compatible LLM + embedding endpoint via app/provider.py, so the
same harness works on hosted build.nvidia.com NIMs or self-hosted NIMs/vLLM on H100s.
"""
import json, os, sys, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.retriever import HybridRetriever
from app.agent import answer_traced

HERE = os.path.dirname(__file__)
K = int(os.getenv("EVAL_K", "3"))


def load_jsonl(name):
    return [json.loads(l) for l in open(os.path.join(HERE, name)) if l.strip()]


def is_hallucination(ex, out):
    """An answer that asserts something (didn't abstain) yet isn't a correct grounded answer."""
    if out["abstained"]:
        return False
    if ex["answerable"]:
        return ex["gold"].lower() not in out["answer"].lower()  # answered, but wrong
    return True  # answered an unanswerable (out-of-corpus) question


def main():
    corpus = load_jsonl("corpus.jsonl")
    passages = [c["text"] for c in corpus]
    id_to_idx = {c["id"]: i for i, c in enumerate(corpus)}
    retriever = HybridRetriever(passages)
    data = load_jsonl("dataset.jsonl")

    rows, hops = [], {"plan": [], "retrieve": [], "generate": [], "validate": []}
    for ex in data:
        g = answer_traced(ex["q"], retriever, k=K, guarded=True)
        u = answer_traced(ex["q"], retriever, k=K, guarded=False)
        for h in hops:
            hops[h].append(g["latency"][h])
        recall = (id_to_idx.get(ex["gold_passage"]) in g["retrieved_idx"]) if ex["answerable"] else None
        correct = (ex["answerable"] and ex["gold"].lower() in g["answer"].lower()
                   and not g["abstained"])
        rows.append({
            "q": ex["q"], "answerable": ex["answerable"], "recall": recall, "correct": correct,
            "g_abstain": g["abstained"], "u_abstain": u["abstained"],
            "g_halluc": is_hallucination(ex, g), "u_halluc": is_hallucination(ex, u),
            "u_gate_blocks": (not u["grounded"]), "lat": round(g["latency_total"], 2)})

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    n_una = len(una)

    recall_at_k = sum(r["recall"] for r in ans) / len(ans)
    accuracy = sum(r["correct"] for r in ans) / len(ans)
    g_halluc = sum(r["g_halluc"] for r in una) / n_una
    u_halluc = sum(r["u_halluc"] for r in una) / n_una

    # validate() gate as a hallucination detector, on the UNGUARDED run.
    tp = sum(1 for r in rows if r["u_halluc"] and r["u_gate_blocks"])
    fn = sum(1 for r in rows if r["u_halluc"] and not r["u_gate_blocks"])
    fp = sum(1 for r in rows if not r["u_halluc"] and r["u_gate_blocks"])
    tn = sum(1 for r in rows if not r["u_halluc"] and not r["u_gate_blocks"])
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    residual_una = sum(1 for r in una if r["u_halluc"] and not r["u_gate_blocks"]) / n_una

    L = []; w = L.append
    w("# Agentic-RAG eval — retrieval, hallucination gating, judge calibration, latency\n")
    w(f"Corpus {len(corpus)} passages · {len(ans)} answerable + {n_una} unanswerable "
      "(incl. adversarial near-miss: B200/H200/FP4 questions a model is tempted to answer "
      "from the H100 facts in context). Self-hosted on H100: vLLM (Qwen3-8B, "
      f"OpenAI-compatible NIM stand-in) + Ollama embeddings. k={K}, temp=0.\n")

    w("## Headline\n")
    w("| metric | value |")
    w("|---|---|")
    w(f"| retrieval recall@{K} (answerable) | {recall_at_k:.0%} |")
    w(f"| answer accuracy (answerable) | {accuracy:.0%} |")
    w(f"| hallucination on unanswerable — **guarded** generator | **{g_halluc:.0%}** |")
    w(f"| hallucination on unanswerable — **unguarded** generator | **{u_halluc:.0%}** |")
    w("")
    w("The guarded prompt (\"answer only from context, else say you don't know\") is the "
      "first line of defense. The ablation shows what happens without it: the same model "
      f"hallucinates on {u_halluc:.0%} of out-of-corpus questions.\n")

    w("## The validate() groundedness gate as a safety net (unguarded run)\n")
    w("Second line of defense: an LLM-as-judge checks each answer against the context and "
      "blocks unsupported ones. Scored on the unguarded run, where hallucinations exist:\n")
    w("| | gate blocks | gate passes |")
    w("|---|---|---|")
    w(f"| hallucinated (should block) | TP={tp} | FN={fn} |")
    w(f"| grounded/abstained (should pass) | FP={fp} | TN={tn} |")
    w("")
    w(f"- precision **{prec:.0%}** · recall **{rec:.0%}** · F1 **{f1:.2f}**")
    w(f"- residual hallucination on unanswerable *after* gating: "
      f"**{residual_una:.0%}** (down from {u_halluc:.0%})\n")
    w("Recall is the safety number — of answers that *should* be blocked, the share the "
      "gate caught. FN are the dangerous misses. Two cheap defenses stacked (guarded prompt "
      "+ judge gate) is how you get a low residual without a fine-tune.\n")

    w("## Per-hop latency budget (mean seconds, guarded)\n")
    m = {h: statistics.mean(v) for h, v in hops.items()}
    w("| plan | retrieve | generate | validate | total |")
    w("|---|---|---|---|---|")
    w(f"| {m['plan']:.2f} | {m['retrieve']:.3f} | {m['generate']:.2f} | {m['validate']:.2f} | "
      f"{sum(m.values()):.2f} |")
    w("")
    w("plan + validate are the agentic tax (2 extra LLM calls wrapping each answer); "
      "retrieve is cheap (embed + in-memory fuse). The validate hop is what buys the "
      "safety-net recall above.\n")

    w("## Per-question detail\n")
    w("| question | answerable | recall@k | correct | guarded halluc | unguarded halluc | gate blocks |")
    w("|---|---|---|---|---|---|---|")
    for r in rows:
        w(f"| {r['q'][:46]} | {r['answerable']} | {r['recall']} | {r['correct']} | "
          f"{r['g_halluc']} | {r['u_halluc']} | {r['u_gate_blocks']} |")
    w("")

    open(os.path.join(HERE, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:26]))
    print(f"\nwrote {os.path.join(HERE, 'report.md')}")


if __name__ == "__main__":
    main()
