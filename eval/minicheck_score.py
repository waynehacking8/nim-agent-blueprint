#!/usr/bin/env python3
"""MiniCheck grounded-NLI scoring of the eval rows (roadmap Phase 5, arXiv:2404.10774).

Scores every (retrieved context, unguarded answer) pair with MiniCheck-Flan-T5-Large -
a 770M *discriminative* claim-vs-document verifier. Published claim: GPT-4-level balanced
accuracy (77.4% on LLM-AGGREFACT) at 400x lower cost.

This is the structural test for the 46/95 shared-blind-spot finding: the generative 8B
judges judge from parametric knowledge; MiniCheck checks the answer against the retrieved
evidence. If a 770M grounded model catches what two 8B generative judges miss, the
bottleneck is grounding, not capacity.

Runs on a GPU host (the eval rows + corpus must be alongside this script):
  pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
  python minicheck_score.py

Output: minicheck_rows.json - [{"pred": 0|1, "prob": float}, ...] aligned 1:1 with the
eval rows; merged into the judge-variant report by eval/judge_variants.py.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.getenv("ROWS", "report_squad_rows.json")
CORPUS = os.getenv("EVAL_CORPUS", "corpus_squad.jsonl")
OUT = os.getenv("OUT", "minicheck_rows.json")


def _path(name):
    return name if os.path.isabs(name) else os.path.join(HERE, name)


def main():
    rows = json.load(open(_path(ROWS)))
    corpus = [json.loads(l) for l in open(_path(CORPUS)) if l.strip()]
    passages = [c["text"] for c in corpus]

    # doc = the same retrieved context the generator (and every other judge) saw;
    # claim = the unguarded answer. Alignment with the rows file is positional.
    docs = ["\n\n".join(passages[j] for j in r["u_ctx_idx"]) for r in rows]
    claims = [r["u_answer"] for r in rows]

    from minicheck.minicheck import MiniCheck
    scorer = MiniCheck(model_name="flan-t5-large", cache_dir=os.path.join(HERE, "ckpts"))
    pred_label, raw_prob, _, _ = scorer.score(docs=docs, claims=claims)

    out = [{"pred": int(p), "prob": float(pr)} for p, pr in zip(pred_label, raw_prob)]
    json.dump(out, open(_path(OUT), "w"), indent=1)
    n_support = sum(o["pred"] for o in out)
    print(f"wrote {OUT}: {len(out)} pairs, {n_support} supported / {len(out) - n_support} unsupported")


if __name__ == "__main__":
    main()
