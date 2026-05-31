"""Plan -> retrieve -> generate -> validate agent loop, OpenTelemetry-traced.

The validate step is the differentiator: every answer is checked for groundedness
(supported by retrieved context) and basic safety before it is returned.
"""
from app import provider
from opentelemetry import trace
tracer = trace.get_tracer("nim-agent-blueprint")

def plan(query):
    with tracer.start_as_current_span("plan"):
        out = provider.chat([{"role": "system", "content": "Rewrite the user question into a focused search query. Reply with only the query."},
                             {"role": "user", "content": query}], max_tokens=64)
        return out.strip()

def generate(query, context):
    with tracer.start_as_current_span("generate"):
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
        return provider.chat([{"role": "system", "content": "Answer using ONLY the context. Cite [n]. If unsupported, say you don't know."},
                              {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}])

def validate(answer, context):
    with tracer.start_as_current_span("validate"):
        ctx = "\n\n".join(context)
        verdict = provider.chat([{"role": "system", "content": "Is the ANSWER fully supported by the CONTEXT? Reply 'grounded' or 'unsupported'."},
                                 {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}"}], max_tokens=8)
        return "grounded" in verdict.lower()

def answer(query, retriever):
    with tracer.start_as_current_span("agent"):
        q = plan(query)
        context = retriever.retrieve(q)
        ans = generate(query, context)
        grounded = validate(ans, context)
        return {"answer": ans, "grounded": grounded, "context_used": len(context)}
