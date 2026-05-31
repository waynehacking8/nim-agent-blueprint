"""Plan -> retrieve -> generate -> validate agent loop, OpenTelemetry-traced.

The validate step is the differentiator: every answer is checked for groundedness
(supported by retrieved context) and basic safety before it is returned.
"""
from app import provider
try:
    from opentelemetry import trace
    tracer = trace.get_tracer("nim-agent-blueprint")
except ImportError:  # tracing is optional — the loop runs without the OTel SDK installed
    import contextlib
    class _NoTracer:
        @contextlib.contextmanager
        def start_as_current_span(self, _name):
            yield None
    tracer = _NoTracer()

def plan(query):
    with tracer.start_as_current_span("plan"):
        out = provider.chat([{"role": "system", "content": "Rewrite the user question into a focused search query. Reply with only the query."},
                             {"role": "user", "content": query}], max_tokens=64)
        return out.strip()

GUARDED_SYS = "Answer using ONLY the context. Cite [n]. If unsupported, say you don't know."
# Unguarded prompt: no abstention instruction. Used only for the ablation that measures
# how much the validate() gate recovers when the generator is *not* told to refuse.
UNGUARDED_SYS = "You are a helpful assistant. Answer the question."


def generate(query, context, guarded=True):
    with tracer.start_as_current_span("generate"):
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
        sys = GUARDED_SYS if guarded else UNGUARDED_SYS
        return provider.chat([{"role": "system", "content": sys},
                              {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}])

def validate(answer, context):
    with tracer.start_as_current_span("validate"):
        ctx = "\n\n".join(context)
        verdict = provider.chat([{"role": "system", "content": "Is the ANSWER fully supported by the CONTEXT? Reply 'grounded' or 'unsupported'."},
                                 {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}"}], max_tokens=16)
        return "grounded" in verdict.lower()

def answer(query, retriever):
    with tracer.start_as_current_span("agent"):
        q = plan(query)
        context = retriever.retrieve(q)
        ans = generate(query, context)
        grounded = validate(ans, context)
        return {"answer": ans, "grounded": grounded, "context_used": len(context)}


_ABSTAIN = ("don't know", "do not know", "cannot answer", "can't answer", "no information",
            "not supported", "unable to", "not in the context", "isn't in the context",
            "does not contain", "doesn't contain", "no relevant", "not mentioned",
            "not provide", "doesn't mention", "does not mention")


def is_abstention(text):
    t = text.lower()
    return any(p in t for p in _ABSTAIN)


def answer_traced(query, retriever, k=8, guarded=True):
    """Same loop as answer(), but returns per-hop latency, retrieved indices and an
    abstention flag — the signals the eval harness needs for recall@k, groundedness
    calibration, hallucination gating and a per-hop latency budget. `guarded=False`
    drops the abstention instruction (ablation) so the validate() gate can be stress-tested."""
    import time
    t = {}
    with tracer.start_as_current_span("agent"):
        s = time.perf_counter(); q = plan(query);                        t["plan"] = time.perf_counter() - s
        s = time.perf_counter(); idx = retriever.retrieve_idx(q, k);     t["retrieve"] = time.perf_counter() - s
        context = [retriever.passages[i] for i in idx]
        s = time.perf_counter(); ans = generate(query, context, guarded); t["generate"] = time.perf_counter() - s
        s = time.perf_counter(); grounded = validate(ans, context);      t["validate"] = time.perf_counter() - s
    return {"answer": ans, "grounded": grounded, "retrieved_idx": idx,
            "abstained": is_abstention(ans), "rewritten_query": q,
            "latency": t, "latency_total": sum(t.values())}
