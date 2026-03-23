# ResearchMind — Agentic Research Intelligence System

> An AI agent that reads scientific papers, reasons across them, and synthesizes structured research briefs — powered by **Endee Vector Database** and **Groq (Llama 3.3 70B)**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Endee](https://img.shields.io/badge/Vector%20DB-Endee-7c6af5?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)

---

## Problem Statement

Researchers and engineers often need to synthesize information across many papers to answer a single research question. Traditional keyword search fails here — it cannot understand semantic meaning or reason about relationships between ideas across multiple documents.

ResearchMind solves this by combining:
- **Semantic vector search** via Endee for meaning-aware retrieval
- **Agentic reasoning** via Groq LLM that plans, reflects, and iterates — not just "ask once, answer once"

---

## System Design

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
  Research Brief (Markdown)
  Source Evidence with similarity scores
  Downloadable report
```

### Why this is agentic, not just RAG

| Feature | Standard RAG | ResearchMind |
|---|---|---|
| Query strategy | Single fixed query | Decomposes into sub-questions |
| Search iterations | 1 | Up to N (configurable) |
| Self-evaluation | No | Reflects on result quality |
| Query refinement | No | Generates improved queries if needed |
| Output | Raw answer | Structured research brief with citations |

---

## Technical Approach

**Embedding model:** `all-MiniLM-L6-v2` (384 dimensions) from sentence-transformers — lightweight, fast, and runs locally with no API cost.

**Chunking strategy:** Each PDF is split into 250-word overlapping chunks (40-word overlap) to preserve context across chunk boundaries.

**Agent loop:** For each sub-question, the agent searches Endee, then asks the LLM to reflect on whether the retrieved chunks sufficiently answer the question. If not, the LLM generates a refined query and searches again — up to N iterations. This self-correcting loop is what separates this from a basic RAG pipeline.

**Synthesis:** All retrieved evidence across all sub-questions is passed to Groq's Llama 3.3 70B, which generates a structured brief with Summary, Key Findings, Methodologies, Open Challenges, and Suggested Next Steps.

---

## How Endee Is Used

Endee serves as the core knowledge store for the entire pipeline.

### Index Creation
```python
client.create_index(
    name="research_papers",
    dimension=384,             # matches all-MiniLM-L6-v2 output
    space_type="cosine",       # cosine similarity for semantic search
    precision=Precision.INT8   # 4x memory savings vs float32
)
```

### Ingestion — PDF to Vectors
```python
# Each paper is chunked into 250-word overlapping segments
# Each chunk is embedded and stored in Endee:
index.upsert([{
    "id": "paper.pdf__chunk__42",
    "vector": embedding,        # 384-dim float list
    "meta": {
        "text":  chunk_text,    # raw text, retrieved at query time
        "paper": "paper.pdf",
        "chunk": 42
    }
}])
```

### Semantic Search inside the Agent Loop
```python
query_vector = model.encode(sub_question).tolist()
results = index.query(vector=query_vector, top_k=8)

# Each result returns:
# result.similarity    → cosine score (higher = more relevant)
# result.meta["text"]  → the actual text chunk
# result.meta["paper"] → source paper filename
```

### Why Endee over alternatives
- **Single-node scalability:** Handles up to 1B vectors on a single node — the corpus can grow without infrastructure changes
- **INT8 precision:** Reduces memory footprint by ~4x with minimal accuracy loss, important when indexing many papers
- **Low latency:** The agent loop makes multiple sequential Endee queries per question, so fast retrieval directly affects response time

---

## Project Structure

```
ResearchMind/
├── app.py              # Streamlit UI
├── agent.py            # Agentic loop: decompose → search → reflect → synthesize
├── ingest.py           # PDF ingestion pipeline into Endee
├── requirements.txt    # Python dependencies
├── docker-compose.yml  # Endee server setup
├── .env.example        # Environment variable template
└── README.md
```

---

## Setup & Running

### Prerequisites
- Docker
- Python 3.10+
- Groq API Key (free at https://console.groq.com)

### 1. Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/endee
cd endee/ResearchMind
```

### 2. Start Endee
```bash
docker compose up -d
```
Endee dashboard will be available at http://localhost:8080

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key
```bash
export GROQ_API_KEY=your_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```
App opens at http://localhost:8501

### 6. Use it
1. Upload research PDFs via the sidebar
2. Type a research question in the main panel
3. Click Launch Agent — the agent will search, reflect, and synthesize a research brief

---

## Example

**Question:** *"What are the two RAG formulations proposed and how do they differ?"*

**Agent process:**
```
DECOMPOSE → 3 sub-questions:
  "What is RAG-Sequence?"
  "What is RAG-Token?"
  "How do RAG-Sequence and RAG-Token differ in generation?"

SEARCH → Endee query: "RAG-Sequence formulation"
RETRIEVED → 8 chunks from 1 paper, top score: 0.863

REFLECT → Good coverage of RAG-Sequence. Searching for RAG-Token.

SEARCH → Endee query: "RAG-Token generation per token retrieval"
RETRIEVED → 8 chunks from 1 paper, top score: 0.841

SYNTHESIZE → Generating research brief...
```

**Output:** A structured brief covering both formulations, their mathematical differences, and tradeoffs — with source citations.

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 250 words | Words per chunk |
| `OVERLAP` | 40 words | Overlap between consecutive chunks |
| `TOP_K` | 8 | Chunks retrieved per Endee search |
| `GROQ_MODEL` | llama-3.3-70b-versatile | LLM used for all reasoning steps |
| `max_iterations` | 3 | Max search-reflect cycles per sub-question |

---

## License
MIT