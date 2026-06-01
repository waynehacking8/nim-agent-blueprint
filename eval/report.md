# Agentic-RAG eval — retrieval, hallucination gating, judge calibration, latency

Corpus 20 passages · 16 answerable + 10 unanswerable (incl. adversarial near-miss: B200/H200/FP4 questions a model is tempted to answer from the H100 facts in context). Self-hosted on a single H100: vLLM (Qwen3-8B, OpenAI-compatible NIM stand-in) + Ollama embeddings. k=3, temp=0.

> **Illustrative, not a statistical benchmark (N=26).** Percentages are small integer ratios (e.g. 16/16 recall, 4/10 unguarded hallucinations); they show the harness works and the directional effect, not a precise score. Scale the corpus and eval set for a real number.

## Headline

| metric | value |
|---|---|
| retrieval recall@3 (answerable) | 100% (16/16 — small illustrative set; near-trivial retrieval at 20-passage corpus, k=3) |
| answer accuracy (answerable) | 100% (16/16 — same small illustrative set, N=26 total; not a statistical benchmark) |
| hallucination on unanswerable — **guarded** generator | **0%** |
| hallucination on unanswerable — **unguarded** generator | **40%** |

The guarded prompt ("answer only from context, else say you don't know") is the first line of defense. The ablation shows what happens without it: the same model hallucinates on 40% of out-of-corpus questions.

**Guarded prompting -> 0%; ablation -> 40%; the weak `validate()` gate alone only claws it back to 30% (illustrative N=10 unanswerable set):**

![Hallucination rate on unanswerable questions: unguarded 40%, guarded 0%, residual 30% after the gate alone](hallucination_ablation.png)

## The validate() groundedness gate as a safety net (unguarded run)

Second line of defense: an LLM-as-judge checks each answer against the context and blocks unsupported ones. The judge (`validate()`) uses the same model and endpoint as the generator. The low 25% gate recall below is consistent with published self-detection findings (arXiv:2511.11087 reports ~22% recall without chain-of-thought, 58% with it); the primary mechanism is shared blind spots — the judge lacks the same knowledge whose absence caused the hallucination, so prompt-level fixes cannot close the gap. Scored on the unguarded run, where hallucinations exist:

| | gate blocks | gate passes |
|---|---|---|
| hallucinated (should block) | TP=1 | FN=3 |
| grounded/abstained (should pass) | FP=1 | TN=21 |

- precision **50%** · recall **25%** · F1 **0.33**
- residual hallucination on unanswerable *after* gating: **30%** (down from 40%)

**The `validate()` gate as a hallucination detector — precision 50%, recall 25%, F1 0.33 (caught 1 of 4). This ~25% recall matches published self-detection findings (arXiv:2511.11087: ~22% without CoT); the primary mechanism is shared blind spots — the judge lacks the same knowledge whose absence caused the hallucination (small illustrative N=26):**

![Confusion matrix of the validate() gate on the unguarded run: TP=1, FN=3, FP=1, TN=21](gate_confusion.png)

Recall is the safety number — of answers that *should* be blocked, the share the gate caught. FN are the dangerous misses. Because the dominant cause is missing knowledge (shared blind spots), prompt-level fixes cannot close the gap; the mitigations are an independent judge model (different model family), a judge panel (PoLL), or retrieval-grounded verification (give the judge the retrieved context to check against — natural for RAG), plus CoT judging as a cheap partial gain. Two cheap defenses stacked (guarded prompt + judge gate) is how you get a low residual without a fine-tune.

## Per-hop latency budget (mean seconds, guarded)

| plan | retrieve | generate | validate | total |
|---|---|---|---|---|
| 0.24 | 0.165 | 0.30 | 0.16 | 0.87 |

plan + validate are the agentic tax (2 extra LLM calls wrapping each answer); retrieve is cheap (embed + in-memory fuse). The validate hop is what buys the safety-net recall above.

## Per-question detail

| question | answerable | recall@k | correct | guarded halluc | unguarded halluc | gate blocks |
|---|---|---|---|---|---|---|
| What API style does NVIDIA NIM expose? | True | True | True | False | False | False |
| What operation syncs activations across GPUs i | True | True | True | False | False | False |
| Which server provides in-flight batching for T | True | True | True | False | False | False |
| Why is tensor-parallel decode communication-bo | True | True | True | False | False | False |
| What does NVLS (NVLink SHARP) do during an all | True | True | True | False | False | False |
| Roughly what is the per-GPU unidirectional NVL | True | True | True | False | False | False |
| In which regime does FP8 quantization help inf | True | True | True | False | False | False |
| How does paged KV-cache manage attention keys  | True | True | True | False | False | False |
| How do CUDA Graphs reduce overhead during deco | True | True | True | False | False | False |
| What does a reranker model do to retrieved pas | True | True | True | False | False | False |
| What two signals does hybrid retrieval combine | True | True | True | False | False | False |
| What is groundedness checking for? | True | True | True | False | False | False |
| What does speculative decoding use a small dra | True | True | True | False | False | False |
| What does MIG do to an H100? | True | True | True | False | False | False |
| What is the pipeline bubble? | True | True | True | False | False | False |
| How is a NIM container distributed and what do | True | True | True | False | False | False |
| What is the exact list price of an NVIDIA H100 | False | None | False | False | False | False |
| What time zone is the NVIDIA headquarters in? | False | None | False | False | False | True |
| How many parameters does GPT-5 have? | False | None | False | False | False | False |
| What is the capital of Australia? | False | None | False | False | True | True |
| What was NVIDIA's quarterly revenue last quart | False | None | False | False | False | False |
| Which sport did Jensen Huang play professional | False | None | False | False | False | False |
| What is the per-GPU unidirectional NVLink band | False | None | False | False | True | False |
| What is the FP4 tensor-core throughput of an H | False | None | False | False | False | False |
| How many NVLink links does an H200 SXM GPU hav | False | None | False | False | True | False |
| What is the default maximum batch size Triton  | False | None | False | False | True | False |

