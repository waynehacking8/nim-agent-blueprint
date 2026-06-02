# Reranker gain - fusion retrieval vs fusion + cross-encoder rerank

Same 200 SQuAD 2.0 questions, same generator (temp=0), same judges; the ONLY difference is retrieval: dense+keyword fusion vs fusion -> `BAAI/bge-reranker-v2-m3` rerank (top-20 candidates -> rerank -> top-3). All comparisons are paired per question (exact McNemar).

> The roadmap names NVIDIA's NV-RerankQA (nvidia/llama-3.2-nv-rerankqa-1b-v2). That checkpoint is license-gated on HuggingFace, so this run substitutes the strongest open cross-encoder reranker (bge-reranker-v2-m3, 568M). The claim under test - what a reranking stage buys over embedding-only retrieval - is the same; the published +14% figure is from the NV-RerankQA paper (arXiv:2409.07691).

## Results

| metric | fusion only | + reranker | delta | McNemar p |
|---|---|---|---|---|
| retrieval recall@3 (answerable) | 85% (85/100, 95% CI 77%–91%) | 96% (96/100, 95% CI 90%–98%) | +11 pts | 0.0010 |
| answer accuracy, LLM judge (answerable) | 72% (72/100, 95% CI 63%–80%) | 80% (80/100, 95% CI 71%–87%) | +8 pts | 0.0386 |
| answer accuracy, substring (answerable) | 70% (70/100, 95% CI 60%–78%) | 81% (81/100, 95% CI 72%–87%) | +11 pts | 0.0034 |
| hallucination, guarded (unanswerable) | 48% (48/100, 95% CI 38%–58%) | 52% (52/100, 95% CI 42%–62%) | +4 pts | 0.5235 |
| hallucination, unguarded (unanswerable) | 78% (78/100, 95% CI 69%–85%) | 86% (86/100, 95% CI 78%–91%) | +8 pts | 0.1153 |

## Reading

The reranker delivers the published-scale retrieval gain (+11 pts recall@3 vs the paper's +14%), and it carries through to answer accuracy (both significant). The second-order question - does better retrieval reduce hallucination? - gets a clear NO: on *unanswerable* (adversarial near-miss) questions hallucination moves +8 pts (not statistically significant, so the honest claim is 'no reduction', not 'an increase'). Mechanism: better retrieval surfaces more on-topic context for questions that have no answer in the corpus, and on-topic-but-answerless context does not make the generator more willing to abstain. Retrieval quality and hallucination safety are different axes; improving the first does not buy the second on the hard case.

Charts: `python eval/make_charts.py` regenerates `rerank_compare.png`.

