# Agentic-RAG eval — retrieval, hallucination gating, judge calibration, latency

Corpus 20 passages · 16 answerable + 10 unanswerable (incl. adversarial near-miss: B200/H200/FP4 questions a model is tempted to answer from the H100 facts in context). Self-hosted on H100: vLLM (Qwen3-8B, OpenAI-compatible NIM stand-in) + Ollama embeddings. k=3, temp=0.

## Headline

| metric | value |
|---|---|
| retrieval recall@3 (answerable) | 94% |
| answer accuracy (answerable) | 94% |
| hallucination on unanswerable — **guarded** generator | **0%** |
| hallucination on unanswerable — **unguarded** generator | **50%** |

The guarded prompt ("answer only from context, else say you don't know") is the first line of defense. The ablation shows what happens without it: the same model hallucinates on 50% of out-of-corpus questions.

## The validate() groundedness gate as a safety net (unguarded run)

Second line of defense: an LLM-as-judge checks each answer against the context and blocks unsupported ones. Scored on the unguarded run, where hallucinations exist:

| | gate blocks | gate passes |
|---|---|---|
| hallucinated (should block) | TP=2 | FN=3 |
| grounded/abstained (should pass) | FP=2 | TN=19 |

- precision **50%** · recall **40%** · F1 **0.44**
- residual hallucination on unanswerable *after* gating: **30%** (down from 50%)

Recall is the safety number — of answers that *should* be blocked, the share the gate caught. FN are the dangerous misses. Two cheap defenses stacked (guarded prompt + judge gate) is how you get a low residual without a fine-tune.

## Per-hop latency budget (mean seconds, guarded)

| plan | retrieve | generate | validate | total |
|---|---|---|---|---|
| 0.21 | 0.155 | 0.29 | 0.16 | 0.81 |

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
| What does MIG do to an H100? | True | False | False | False | False | True |
| What is the pipeline bubble? | True | True | True | False | False | False |
| How is a NIM container distributed and what do | True | True | True | False | False | False |
| What is the exact list price of an NVIDIA H100 | False | None | False | False | False | False |
| What time zone is the NVIDIA headquarters in? | False | None | False | False | False | True |
| How many parameters does GPT-5 have? | False | None | False | False | False | False |
| What is the capital of Australia? | False | None | False | False | True | True |
| What was NVIDIA's quarterly revenue last quart | False | None | False | False | False | False |
| Which sport did Jensen Huang play professional | False | None | False | False | True | True |
| What is the per-GPU unidirectional NVLink band | False | None | False | False | True | False |
| What is the FP4 tensor-core throughput of an H | False | None | False | False | False | False |
| How many NVLink links does an H200 SXM GPU hav | False | None | False | False | True | False |
| What is the default maximum batch size Triton  | False | None | False | False | True | False |

