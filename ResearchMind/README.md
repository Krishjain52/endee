# 🔬 ResearchMind — Agentic Research Intelligence System

> An AI agent that reads scientific papers, reasons across them, and synthesizes structured research briefs — powered by **Endee Vector Database** and **Groq (Llama 3.3 70B)**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Endee](https://img.shields.io/badge/Vector%20DB-Endee-7c6af5?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)

---

## 📌 Problem Statement

Researchers and engineers often need to synthesize information across **many papers** to answer a single research question. Traditional keyword search fails here — it can't understand semantic meaning or reason about relationships between ideas.

ResearchMind solves this by combining:
- **Semantic vector search** (Endee) for meaning-aware retrieval
- **Agentic reasoning** (Groq LLM) that plans, reflects, and iterates — not just "ask once, answer once"

---

## 🧠 System Design

```
User Question
      │
      ▼
┌─────────────────────┐
│   DECOMPOSE (LLM)   │  Break into 2-4 focused sub-questions
└──────────┬──────────┘
           │
     ┌─────▼──────────────────────────────┐
     │         AGENT LOOP (per sub-q)      │
     │                                     │
     │  ┌──────────────────────────────┐   │
     │  │  SEARCH → Endee Vector DB    │   │
     │  │  (cosine similarity, INT8)   │   │
     │  └──────────────┬───────────────┘   │
     │                 │                   │
     │  ┌──────────────▼───────────────┐   │
     │  │  REFLECT (LLM)               │   │
     │  │  Sufficient? → Done          │   │
     │  │  Not sufficient? → Refine    │◄──┤
     │  │  query and search again      │   │
     │  └──────────────────────────────┘   │
     └─────────────────────────────────────┘
           │
      ▼ (all retrieved chunks)
┌─────────────────────┐
│   SYNTHESIZE (LLM)  │  Generate structured research brief
└──────────┬──────────┘
           │
      ▼
  📋 Research Brief (Markdown)
  📎 Source Evidence with scores
  ⬇️  Downloadable report
```

### Why this is agentic (not just RAG):
| Feature | Standard RAG | ResearchMind |
|---|---|---|
| Query strategy | Single fixed query | Decomposes into sub-questions |
| Search iterations | 1 | Up to N (configurable) |
| Self-evaluation | ❌ | ✅ Reflects on result quality |
| Query refinement | ❌ | ✅ Generates improved queries |
| Output | Raw answer | Structured research brief |

---

## 🏗️ How Endee Is Used

Endee serves as the **core knowledge store** for the entire pipeline.

### Index Creation
```python
client.create_index(
    name="research_papers",
    dimension=384,          # all-MiniLM-L6-v2 output size
    space_type="cosine",    # cosine similarity for semantic search
    precision=Precision.INT8  # 4x memory savings vs float32
)
```

### Ingestion (PDF → Chunks → Vectors → Endee)
```python
# Each paper is chunked into 250-word overlapping segments
# Each chunk is embedded with sentence-transformers and stored:
index.upsert([{
    "id": "arxiv_2310.06825.pdf__chunk__42",
    "vector": embedding,   # 384-dim float list
    "meta": {
        "text":  chunk_text,    # stored for retrieval
        "paper": "arxiv_2310.06825.pdf",
        "chunk": 42
    }
}])
```

### Semantic Search (inside the agent loop)
```python
query_vector = model.encode(sub_question).tolist()
results = index.query(vector=query_vector, top_k=8)

# Each result has:
# result.similarity   → cosine score (higher = more relevant)
# result.meta["text"] → the actual text chunk
# result.meta["paper"] → source paper name
```

### Why Endee over alternatives?
- **Single-node scalability**: Can handle up to 1B vectors — future-proof as the paper corpus grows
- **INT8 precision**: Reduces memory footprint by ~4x with minimal accuracy loss, critical for large corpora
- **Low latency**: Sub-millisecond query times even at scale — the agent loop makes multiple sequential searches, so latency matters

---

## 🚀 Setup & Running

### Prerequisites
- Docker (for Endee)
- Python 3.10+
- [Groq API Key](https://console.groq.com) (free tier available)

### 1. Start Endee
```bash
docker compose up -d
# Endee dashboard: http://localhost:8080
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
export GROQ_API_KEY=your_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Ingest papers & ask questions
1. Upload PDFs via the sidebar **or** paste an ArXiv URL (e.g. `https://arxiv.org/abs/1706.03762`)
2. Type a research question in the main panel
3. Watch the agent decompose, search, reflect, and synthesize in real-time

---

## 💡 Example Usage

**Question:** *"What techniques are used to reduce hallucinations in large language models?"*

**Agent trace:**
```
🧩 DECOMPOSE → 3 sub-questions:
   "What causes hallucinations in LLMs?"
   "What training methods reduce hallucination?"
   "What inference-time techniques improve factuality?"

🔍 ENDEE SEARCH [1/3] → "What causes hallucinations in LLMs?"
✅ RETRIEVED → 8 chunks from 3 papers — top score: 0.847

💭 REFLECT → Results cover causes well. Moving to training methods.

🔍 ENDEE SEARCH [2/3] → "What training methods reduce hallucination?"
✅ RETRIEVED → 8 chunks from 4 papers — top score: 0.831

💭 REFLECT → Limited detail on RLHF. Refining query.

🔍 ENDEE SEARCH [3/3] → "RLHF reinforcement learning human feedback factuality"
✅ RETRIEVED → 8 chunks from 2 papers — top score: 0.792

⚡ SYNTHESIZE → Generating research brief...
```

---

## 📁 Project Structure

```
researchmind/
├── app.py              # Streamlit UI — renders agent trace + results
├── agent.py            # Agentic loop: decompose → search → reflect → synthesize
├── ingest.py           # PDF/ArXiv ingestion pipeline into Endee
├── requirements.txt
├── docker-compose.yml  # Endee server
├── .env.example
└── README.md
```

---

## 🔧 Configuration

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 250 words | Chunk size for document splitting |
| `OVERLAP` | 40 words | Overlap between chunks |
| `TOP_K` | 8 | Chunks retrieved per search |
| `GROQ_MODEL` | llama-3.3-70b-versatile | LLM for decompose/reflect/synthesize |
| `max_iterations` | 3 | Max search-reflect iterations per sub-question |

---

## 📜 License
MIT
