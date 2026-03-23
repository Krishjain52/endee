"""
agent.py — Agentic RAG loop for ResearchMind

Flow:
  1. DECOMPOSE   → Break user question into sub-questions (Groq)
  2. SEARCH      → Query Endee for each sub-question
  3. REFLECT     → Decide if results are sufficient or need a refined search (Groq)
  4. ITERATE     → If needed, generate a new search query and repeat
  5. SYNTHESIZE  → Generate a structured research brief from all evidence (Groq)
"""

import os
from groq import Groq
from endee import Endee
from sentence_transformers import SentenceTransformer
from ingest import INDEX_NAME, DIMENSION, EMBEDDING_MODEL

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"   # fast + capable
TOP_K        = 8   # chunks per search

# ── Singletons ────────────────────────────────────────────────────────────────
_groq   = None
_model  = None
_client = None

def get_groq():
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_endee():
    global _client
    if _client is None:
        _client = Endee()
    return _client


# ── LLM helpers ───────────────────────────────────────────────────────────────
def llm(system: str, user: str, max_tokens: int = 800) -> str:
    resp = get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ── Endee search ──────────────────────────────────────────────────────────────
def vector_search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed query and search Endee, return list of result dicts."""
    model  = get_model()
    endee  = get_endee()
    vec    = model.encode(query).tolist()
    try:
        index   = endee.get_index(INDEX_NAME)
        results = index.query(vector=vec, top_k=top_k)
        return [
            {
                "text":  r.meta.get("text", ""),
                "paper": r.meta.get("paper", "unknown"),
                "chunk": r.meta.get("chunk", 0),
                "score": r.similarity,
            }
            for r in results
        ]
    except Exception:
        return []


# ── Agent steps ───────────────────────────────────────────────────────────────
def decompose(question: str) -> list[str]:
    """Break the research question into 2–4 focused sub-questions."""
    raw = llm(
        system=(
            "You are a research strategist. Given a broad research question, "
            "decompose it into 2 to 4 precise, distinct sub-questions that together "
            "fully answer the original. Return ONLY a numbered list, one per line. "
            "No preamble, no explanation."
        ),
        user=f"Research question: {question}",
        max_tokens=300,
    )
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    # Strip leading number/dot
    sub_qs = []
    for line in lines:
        clean = line.lstrip("0123456789.-) ").strip()
        if clean:
            sub_qs.append(clean)
    return sub_qs[:4] if sub_qs else [question]


def reflect(question: str, sub_question: str, retrieved_chunks: list[dict]) -> dict:
    """
    Reflect on whether the retrieved chunks adequately answer the sub-question.
    Returns {"sufficient": bool, "thought": str, "new_query": str|None}
    """
    snippets = "\n".join(
        f"[{c['paper']}]: {c['text'][:200]}" for c in retrieved_chunks[:5]
    )
    raw = llm(
        system=(
            "You are a critical research analyst. You are given a sub-question and "
            "retrieved document chunks. Decide:\n"
            "1. Are the chunks SUFFICIENT to answer the sub-question? (yes/no)\n"
            "2. One sentence explaining why.\n"
            "3. If NOT sufficient, provide a BETTER search query (1 line, no quotes).\n\n"
            "Respond in EXACTLY this format:\n"
            "SUFFICIENT: yes|no\n"
            "THOUGHT: <one sentence>\n"
            "NEW_QUERY: <refined query or NONE>"
        ),
        user=(
            f"Sub-question: {sub_question}\n\n"
            f"Retrieved chunks:\n{snippets}"
        ),
        max_tokens=200,
    )

    lines_dict = {}
    for line in raw.split('\n'):
        if ':' in line:
            k, _, v = line.partition(':')
            lines_dict[k.strip().upper()] = v.strip()

    sufficient = lines_dict.get("SUFFICIENT", "no").lower().startswith("y")
    thought    = lines_dict.get("THOUGHT", "Proceeding with available context.")
    new_query  = lines_dict.get("NEW_QUERY", "NONE")
    if new_query.upper() == "NONE" or not new_query:
        new_query = None

    return {"sufficient": sufficient, "thought": thought, "new_query": new_query}


def synthesize(original_question: str, all_chunks: list[dict]) -> str:
    """Generate a structured markdown research brief from all retrieved evidence."""
    # Build context block (deduplicate, keep top scoring)
    seen   = set()
    unique = []
    for c in sorted(all_chunks, key=lambda x: x["score"], reverse=True):
        key = c["text"][:80]
        if key not in seen:
            seen.add(key)
            unique.append(c)
        if len(unique) >= 20:
            break

    context = "\n\n".join(
        f"[{c['paper']}] (score {c['score']:.3f})\n{c['text']}"
        for c in unique
    )

    brief = llm(
        system=(
            "You are a senior AI research scientist writing a structured research brief "
            "for a technical audience. Use ONLY information from the provided context. "
            "Do not hallucinate. Cite paper names inline like (arxiv_xxxx.pdf). "
            "Format your response as Markdown with these exact sections:\n\n"
            "## Summary\n"
            "## Key Findings\n"
            "## Methodologies & Approaches\n"
            "## Open Challenges\n"
            "## Suggested Next Steps\n\n"
            "Be precise, technical, and concise."
        ),
        user=(
            f"Research question: {original_question}\n\n"
            f"Evidence from papers:\n{context}"
        ),
        max_tokens=1500,
    )
    return brief


# ── Main Agent ────────────────────────────────────────────────────────────────
class ResearchAgent:
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations

    def run_stream(self, question: str):
        """
        Generator that yields agent events for real-time UI rendering.

        Event types:
          {"type": "decompose", "subquestions": [...]}
          {"type": "search",    "query": str, "iteration": int}
          {"type": "found",     "count": int, "papers": int, "top_score": float}
          {"type": "reflect",   "thought": str}
          {"type": "synthesize"}
          {"type": "done",      "brief": str, "sources": [...]}
        """
        all_chunks: list[dict] = []

        # ── Step 1: Decompose ──────────────────────────────────────────────
        sub_questions = decompose(question)
        yield {"type": "decompose", "subquestions": sub_questions}

        # ── Step 2-4: Search + Reflect loop per sub-question ──────────────
        iteration = 0
        for sq in sub_questions:
            current_query = sq
            sq_iterations = 0

            while sq_iterations < self.max_iterations:
                iteration    += 1
                sq_iterations += 1

                # Search Endee
                yield {"type": "search", "query": current_query, "iteration": iteration}
                chunks = vector_search(current_query)

                if chunks:
                    papers    = len(set(c["paper"] for c in chunks))
                    top_score = chunks[0]["score"]
                    yield {"type": "found", "count": len(chunks), "papers": papers, "top_score": top_score}
                    all_chunks.extend(chunks)
                else:
                    yield {"type": "found", "count": 0, "papers": 0, "top_score": 0.0}

                # Reflect
                if sq_iterations < self.max_iterations:
                    reflection = reflect(question, current_query, chunks)
                    yield {"type": "reflect", "thought": reflection["thought"]}

                    if reflection["sufficient"] or not reflection["new_query"]:
                        break   # move to next sub-question
                    else:
                        current_query = reflection["new_query"]
                else:
                    break

        # ── Step 5: Synthesize ─────────────────────────────────────────────
        yield {"type": "synthesize"}
        brief = synthesize(question, all_chunks)

        yield {
            "type":    "done",
            "brief":   brief,
            "sources": all_chunks,
        }
