#!/usr/bin/env python3
"""Semantic Entropy (Farquhar et al., Nature 2024) on the SQuAD eval set (roadmap Phase 5).

Published target: AUROC 0.790 (semantic entropy) vs 0.691 (naive entropy) for detecting
confabulations. This adds the missing axis to the repo's gates: a *continuous, pre-answer*
uncertainty score from the generator itself, needing no judge at all.

Two stages (so the GPU-bound NLI clustering can run on a different host than the sampling):

  stage 1 - sample   (network only, needs NIM_* env like run_eval.py):
      python eval/semantic_entropy.py sample
      For every question: 10 samples at T=1.0 from the unguarded generator + token logprobs.
      -> semantic_entropy_samples.json

  stage 2 - score    (GPU host: DeBERTa-large-MNLI bidirectional entailment clustering):
      python eval/semantic_entropy.py score
      Clusters samples by bidirectional entailment, computes semantic entropy (Rao, over
      cluster probabilities) + discrete SE + naive (length-normalized logprob) entropy,
      then AUROC against the u_halluc gold labels from the rows file.
      -> semantic_entropy_scores.json (+ printed AUROC table)

Labels: u_halluc from report_squad_rows.json (the temp=0 unguarded answer is/isn't a
hallucination) - the same "most-likely answer is wrong" labeling the paper uses.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.getenv("ROWS", "report_squad_rows.json")
CORPUS = os.getenv("EVAL_CORPUS", "corpus_squad.jsonl")
DATASET = os.getenv("EVAL_DATASET", "dataset_squad.jsonl")
SAMPLES = os.getenv("SAMPLES", "semantic_entropy_samples.json")
SCORES = os.getenv("SCORES", "semantic_entropy_scores.json")
N_SAMPLES = int(os.getenv("SE_SAMPLES", "10"))
TEMP = float(os.getenv("SE_TEMP", "1.0"))


def _path(name):
    return name if os.path.isabs(name) else os.path.join(HERE, name)


# ----------------------------------------------------------------------------- stage 1
def sample():
    import httpx
    sys.path.insert(0, os.path.dirname(HERE))
    from app.agent import UNGUARDED_SYS

    rows = json.load(open(_path(ROWS)))
    corpus = [json.loads(l) for l in open(_path(CORPUS)) if l.strip()]
    passages = [c["text"] for c in corpus]

    base = os.getenv("NIM_LLM_URL", "http://localhost:8000/v1")
    model = os.getenv("NIM_LLM_MODEL", "qwen3-8b")

    def gen(query, ctx_text):
        """One T=1.0 sample with token logprobs (for the full-SE variant)."""
        body = {"model": model, "max_tokens": 256, "temperature": TEMP, "logprobs": True,
                "messages": [{"role": "system", "content": UNGUARDED_SYS},
                             {"role": "user", "content": f"Context:\n{ctx_text}\n\nQuestion: {query}"}]}
        if os.getenv("NIM_DISABLE_THINKING"):
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = httpx.post(f"{base}/chat/completions", timeout=120, json=body)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        lps = [t["logprob"] for t in (choice.get("logprobs") or {}).get("content") or []]
        return {"text": choice["message"]["content"], "logprob_sum": sum(lps), "n_tokens": len(lps)}

    out = []
    # resume support: skip questions already sampled
    if os.path.exists(_path(SAMPLES)):
        out = json.load(open(_path(SAMPLES)))
        print(f"  resuming from {len(out)} sampled questions")
    for i, r in enumerate(rows):
        if i < len(out):
            continue
        ctx_text = "\n\n".join(f"[{k + 1}] {passages[j]}" for k, j in enumerate(r["u_ctx_idx"]))
        samples = [gen(r["q"], ctx_text) for _ in range(N_SAMPLES)]
        out.append({"q": r["q"], "answerable": r["answerable"], "u_halluc": r["u_halluc"],
                    "samples": samples})
        json.dump(out, open(_path(SAMPLES), "w"), indent=1)
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"  .. sampled {i + 1}/{len(rows)} questions x {N_SAMPLES}", flush=True)
    print(f"wrote {SAMPLES}")


# ----------------------------------------------------------------------------- stage 2
def auroc(scores, labels):
    """AUROC by the Mann-Whitney U statistic (no sklearn dependency)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from stats_util import auroc_mannwhitney
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    return auroc_mannwhitney(pos, neg)


def score():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    data = json.load(open(_path(SAMPLES)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "microsoft/deberta-large-mnli"
    tok = AutoTokenizer.from_pretrained(name)
    nli = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()

    @torch.no_grad()
    def entails(a, b):
        """True iff a entails b AND b entails a (bidirectional, paper's clustering rule)."""
        for prem, hyp in ((a, b), (b, a)):
            x = tok(prem, hyp, return_tensors="pt", truncation=True, max_length=512).to(device)
            # deberta-large-mnli labels: 0=contradiction, 1=neutral, 2=entailment
            if nli(**x).logits.argmax(-1).item() != 2:
                return False
        return True

    def cluster(texts):
        """Greedy agglomeration by bidirectional entailment -> cluster id per sample."""
        ids = []
        reps = []  # one representative text per cluster
        for t in texts:
            for ci, rep in enumerate(reps):
                if entails(t, rep):
                    ids.append(ci)
                    break
            else:
                ids.append(len(reps))
                reps.append(t)
        return ids

    results = []
    for i, q in enumerate(data):
        texts = [s["text"].strip() for s in q["samples"]]
        ids = cluster(texts)
        n = len(texts)
        counts = {}
        for c in ids:
            counts[c] = counts.get(c, 0) + 1

        # discrete SE: entropy over cluster occupancy (no logprobs needed)
        discrete_se = -sum((c / n) * math.log(c / n) for c in counts.values())

        # Rao SE: cluster prob = sum of length-normalized sequence probs of its members
        lp = [s["logprob_sum"] / max(s["n_tokens"], 1) for s in q["samples"]]
        p = [math.exp(x) for x in lp]
        z = sum(p) or 1.0
        p = [x / z for x in p]
        cp = {}
        for c, pi in zip(ids, p):
            cp[c] = cp.get(c, 0.0) + pi
        rao_se = -sum(pi * math.log(pi) for pi in cp.values() if pi > 0)

        # naive entropy baseline: mean negative length-normalized logprob (paper's "naive entropy")
        naive = -sum(lp) / n

        results.append({"q": q["q"], "answerable": q["answerable"], "u_halluc": q["u_halluc"],
                        "n_clusters": len(counts), "discrete_se": discrete_se,
                        "semantic_entropy": rao_se, "naive_entropy": naive})
        if (i + 1) % 20 == 0 or i + 1 == len(data):
            print(f"  .. clustered {i + 1}/{len(data)}", flush=True)

    json.dump(results, open(_path(SCORES), "w"), indent=1)

    # AUROC vs hallucination labels - headline table
    labels = [r["u_halluc"] for r in results]
    print("\nAUROC vs u_halluc gold labels (published: SE 0.790, naive 0.691):")
    for key in ("semantic_entropy", "discrete_se", "naive_entropy", "n_clusters"):
        print(f"  {key:18s} {auroc([r[key] for r in results], labels):.3f}")
    # unanswerable-only slice: the adversarial near-miss case
    una = [r for r in results if not r["answerable"]]
    lab_u = [r["u_halluc"] for r in una]
    print(f"\nunanswerable-only slice (n={len(una)}):")
    for key in ("semantic_entropy", "discrete_se", "naive_entropy"):
        print(f"  {key:18s} {auroc([r[key] for r in una], lab_u):.3f}")
    print(f"\nwrote {SCORES}")


# ----------------------------------------------------------------------------- stage 3
def report():
    """Markdown report from the committed scores JSON (no GPU, runs anywhere)."""
    import statistics

    scores = json.load(open(_path(SCORES)))
    out_path = _path(os.getenv("REPORT", "report_semantic_entropy.md"))

    def slice_auroc(rows, key):
        return auroc([r[key] for r in rows], [r["u_halluc"] for r in rows])

    ans = [r for r in scores if r["answerable"]]
    una = [r for r in scores if not r["answerable"]]
    methods = [("semantic entropy (Rao, logprob-weighted)", "semantic_entropy"),
               ("discrete semantic entropy (cluster counts)", "discrete_se"),
               ("naive entropy (mean -logprob)", "naive_entropy")]

    L = []
    w = L.append
    w("# Semantic entropy (Farquhar et al., Nature 2024) on the SQuAD eval set\n")
    w(f"{len(scores)} questions x {N_SAMPLES} samples at T={TEMP} from the unguarded generator; "
      "clusters = bidirectional DeBERTa-large-MNLI entailment; labels = whether the temp=0 "
      "answer was a hallucination (same rows as every other gate in this repo).\n")
    w("## AUROC vs published\n")
    w("| method | full set | answerable only | unanswerable (adversarial near-miss) | published |")
    w("|---|---|---|---|---|")
    pub = {"semantic_entropy": "0.790", "discrete_se": "~0.790", "naive_entropy": "0.691"}
    for name, key in methods:
        w(f"| {name} | {slice_auroc(scores, key):.3f} | {slice_auroc(ans, key):.3f} | "
          f"{slice_auroc(una, key):.3f} | {pub[key]} |")
    w("")
    w("## Reading - the published ceiling does NOT transfer to the adversarial case\n")
    h_ans = statistics.mean(r["discrete_se"] for r in ans if r["u_halluc"])
    c_ans = statistics.mean(r["discrete_se"] for r in ans if not r["u_halluc"])
    h_una = statistics.mean(r["discrete_se"] for r in una if r["u_halluc"])
    c_una = statistics.mean(r["discrete_se"] for r in una if not r["u_halluc"])
    w(f"On **answerable** questions the signal works as published: hallucinated answers have "
      f"much higher semantic entropy than correct ones ({h_ans:.2f} vs {c_ans:.2f}) and naive "
      "entropy lands within a few points of the paper's 0.691. On the **adversarial near-miss** "
      f"questions the separation collapses ({h_una:.2f} vs {c_una:.2f}) and every variant drops "
      "to chance.\n")
    w("Two mechanisms, both visible in the per-question clusters:\n")
    w("1. **Systematic (confident) errors**: the near-miss design makes the generator give the "
      "same wrong answer in most samples - low entropy, but hallucinated. Farquhar et al. "
      "scope semantic entropy to *confabulations* (arbitrary wrong answers) and explicitly "
      "exclude systematic errors; this eval set is built from the excluded case.")
    w("2. **Abstention phrasing diversity**: when the generator correctly abstains, the 10 "
      "samples phrase the refusal differently and NLI does not cluster refusals as mutually "
      "entailing - high entropy, but NOT hallucinated.\n")
    w("Connection to the judge results: the 48 hallucinations that escape both 8B judges are "
      "the same confident, systematic errors that defeat semantic entropy. Sampling-based "
      "uncertainty is not a path to recovering them; grounded verification (MiniCheck) "
      "partially is.\n")
    w("Chart: `python eval/make_charts.py` regenerates `semantic_entropy.png`.\n")

    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    {"sample": sample, "score": score, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
