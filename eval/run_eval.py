#!/usr/bin/env python3
"""Evaluate the blueprint: retrieval hit-rate, groundedness, latency -> eval/report.md."""
import json, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.retriever import HybridRetriever
from app.serve import CORPUS
from app.agent import answer

def main():
    retriever = HybridRetriever(CORPUS)
    rows, hits, grounded, lat = [], 0, 0, []
    for line in open(os.path.join(os.path.dirname(__file__), "dataset.jsonl")):
        ex = json.loads(line)
        t0 = time.perf_counter()
        out = answer(ex["q"], retriever)
        dt = time.perf_counter() - t0; lat.append(dt)
        hit = ex["gold"].lower() in out["answer"].lower()
        hits += hit; grounded += out["grounded"]
        rows.append((ex["q"], hit, out["grounded"], round(dt, 2)))
    n = len(rows)
    with open(os.path.join(os.path.dirname(__file__), "report.md"), "w") as w:
        w.write(f"# Eval report\n\n- hit-rate: {hits}/{n}\n- grounded: {grounded}/{n}\n")
        w.write(f"- mean latency: {sum(lat)/n:.2f}s\n\n| question | hit | grounded | s |\n|---|---|---|---|\n")
        for q, h, g, d in rows:
            w.write(f"| {q} | {h} | {g} | {d} |\n")
    print(open(os.path.join(os.path.dirname(__file__), "report.md")).read())

if __name__ == "__main__":
    main()
