#!/usr/bin/env python3
"""Build a statistically useful eval set from SQuAD 2.0 dev (corpus_squad.jsonl + dataset_squad.jsonl).

Why SQuAD 2.0: its unanswerable questions were written by crowdworkers to *look* answerable
from the paragraph (adversarial near-miss by construction) — the same design as this repo's
hand-written demo set, but at a scale where percentages carry confidence intervals, and with
published baselines to compare against.

Sampling (deterministic, seed=7):
  * PASSAGES paragraphs sampled across all 35 dev articles -> corpus passages.
  * For those paragraphs: up to QA_PER_PARA answerable + unanswerable questions each,
    until N_ANSWERABLE + N_UNANSWERABLE total questions.
  * Unanswerable questions keep their source paragraph as `near_miss_passage` — the
    paragraph IS in the corpus, it just does not contain the answer (the adversarial part).

Usage:
  python3 eval/build_squad_eval.py [path/to/dev-v2.0.json]
  (downloads the public dev set to /tmp if no path is given)
"""
import json
import os
import random
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"

PASSAGES = 100
N_ANSWERABLE = 100
N_UNANSWERABLE = 100
QA_PER_PARA = 2
SEED = 7


def load_squad(path):
    if not path:
        path = "/tmp/squad_dev_v2.json"
        if not os.path.exists(path):
            print(f">> downloading SQuAD 2.0 dev to {path}")
            urllib.request.urlretrieve(SQUAD_URL, path)
    with open(path) as f:
        return json.load(f)["data"]


def main():
    articles = load_squad(sys.argv[1] if len(sys.argv) > 1 else None)
    rng = random.Random(SEED)

    # flatten to (article_title, paragraph) keeping article diversity: round-robin sample
    paras_by_article = [(a["title"], p) for a in articles for p in a["paragraphs"]]
    rng.shuffle(paras_by_article)

    # keep paragraphs that have BOTH answerable and unanswerable questions where possible
    def n_qas(p, impossible):
        return sum(1 for q in p["qas"] if q["is_impossible"] == impossible)

    ranked = sorted(paras_by_article,
                    key=lambda tp: -(min(n_qas(tp[1], False), 1) + min(n_qas(tp[1], True), 1)))
    chosen = ranked[:PASSAGES]

    corpus, dataset = [], []
    n_ans, n_una = 0, 0
    for i, (title, para) in enumerate(chosen):
        pid = f"squad-{i:03d}"
        corpus.append({"id": pid, "title": title, "text": para["context"]})

        answerable = [q for q in para["qas"] if not q["is_impossible"]][:QA_PER_PARA]
        unanswerable = [q for q in para["qas"] if q["is_impossible"]][:QA_PER_PARA]
        for q in answerable:
            if n_ans >= N_ANSWERABLE:
                break
            golds = sorted({a["text"] for a in q["answers"]}, key=len)
            dataset.append({"q": q["question"], "answerable": True,
                            "gold_passage": pid, "gold": golds[0],
                            "gold_alternatives": golds})
            n_ans += 1
        for q in unanswerable:
            if n_una >= N_UNANSWERABLE:
                break
            dataset.append({"q": q["question"], "answerable": False,
                            "gold_passage": None, "gold": None,
                            "near_miss_passage": pid})
            n_una += 1

    rng.shuffle(dataset)
    with open(os.path.join(HERE, "corpus_squad.jsonl"), "w") as f:
        f.writelines(json.dumps(c) + "\n" for c in corpus)
    with open(os.path.join(HERE, "dataset_squad.jsonl"), "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in dataset)
    print(f">> corpus_squad.jsonl: {len(corpus)} passages")
    print(f">> dataset_squad.jsonl: {n_ans} answerable + {n_una} unanswerable "
          f"(near-miss adversarial) = {len(dataset)} questions")


if __name__ == "__main__":
    main()
