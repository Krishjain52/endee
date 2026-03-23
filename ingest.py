"""
ingest.py — Load research papers into Endee vector database
Supports: local PDFs, ArXiv URLs
"""
import re
import urllib.request
import tempfile
import os

import pypdf
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# ── Config ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, free
INDEX_NAME      = "research_papers"
DIMENSION       = 384
CHUNK_SIZE      = 250   # words per chunk
OVERLAP         = 40    # word overlap between chunks

# ── Singletons ───────────────────────────────────────────────────────────────
_model  = None
_client = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_client():
    global _client
    if _client is None:
        _client = Endee()   # connects to localhost:8080 by default
    return _client

def get_or_create_index():
    client = get_client()
    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            space_type="cosine",
            precision=Precision.INT8
        )
    except Exception:
        pass  # index already exists
    return client.get_index(INDEX_NAME)


# ── Chunking ─────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP) -> list[str]:
    """Split text into overlapping word-level chunks."""
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.split()) > 20:   # skip tiny tail chunks
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ── PDF Ingestion ─────────────────────────────────────────────────────────────
def ingest_pdf(pdf_path: str, display_name: str = None) -> int:
    """
    Ingest a PDF file into Endee.
    Returns number of chunks indexed.
    """
    if display_name is None:
        display_name = os.path.basename(pdf_path)

    # Extract text
    reader = pypdf.PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages_text.append(t)
    full_text = " ".join(pages_text)

    if not full_text.strip():
        raise ValueError(f"Could not extract text from {display_name}")

    # Chunk
    chunks = chunk_text(full_text)

    # Embed + upsert
    model = get_model()
    index = get_or_create_index()

    batch = []
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        batch.append({
            "id": f"{display_name}__chunk__{i}",
            "vector": embedding,
            "meta": {
                "text":   chunk,
                "paper":  display_name,
                "chunk":  i,
                "total":  len(chunks),
            }
        })
        if len(batch) >= 100:
            index.upsert(batch)
            batch = []

    if batch:
        index.upsert(batch)

    return len(chunks)


# ── ArXiv Ingestion ───────────────────────────────────────────────────────────
def ingest_arxiv_url(url: str) -> dict:
    """
    Fetch a paper from ArXiv and ingest it.
    Accepts:  https://arxiv.org/abs/2310.06825
              https://arxiv.org/pdf/2310.06825
              https://arxiv.org/pdf/2310.06825.pdf
    Returns:  {"success": bool, "title": str, "chunks": int, "error": str}
    """
    try:
        # Extract ArXiv ID
        arxiv_id = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]+)', url)
        if not arxiv_id:
            return {"success": False, "error": "Not a valid ArXiv URL. Expected format: https://arxiv.org/abs/XXXX.XXXXX"}

        paper_id = arxiv_id.group(1)
        title    = f"arxiv_{paper_id}.pdf"

        # ArXiv requires a real browser User-Agent + follow redirects
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,*/*",
        }

        # Try multiple URL patterns ArXiv uses
        candidate_urls = [
            f"https://arxiv.org/pdf/{paper_id}.pdf",
            f"https://arxiv.org/pdf/{paper_id}",
            f"https://export.arxiv.org/pdf/{paper_id}",
        ]

        tmp_path = None
        last_error = None

        for pdf_url in candidate_urls:
            try:
                req = urllib.request.Request(pdf_url, headers=headers)
                with urllib.request.urlopen(req, timeout=45) as resp:
                    content_type = resp.headers.get("Content-Type", "")
                    data = resp.read()

                # Verify it's actually a PDF
                if not data.startswith(b"%PDF"):
                    last_error = f"Response from {pdf_url} is not a PDF (got {content_type})"
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                break  # success

            except Exception as e:
                last_error = str(e)
                continue

        if tmp_path is None:
            return {
                "success": False,
                "error": (
                    f"Could not download PDF for arxiv:{paper_id}. "
                    f"Last error: {last_error}. "
                    "Try downloading the PDF manually and uploading it via the file uploader."
                )
            }

        chunks = ingest_pdf(tmp_path, title)
        os.unlink(tmp_path)
        return {"success": True, "title": title, "chunks": chunks}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Stats ─────────────────────────────────────────────────────────────────────
def get_index_stats() -> dict:
    """Return quick stats about the current index."""
    try:
        client = get_client()

        # Check if index exists
        try:
            indexes = client.list_indexes()
            index_names = [ix.name if hasattr(ix, 'name') else str(ix) for ix in indexes]
            if INDEX_NAME not in index_names:
                return {"vectors": 0, "papers": 0}
        except Exception:
            pass  # list_indexes may not exist, try get_index directly

        index = client.get_index(INDEX_NAME)

        # Sample with a random vector to get real results
        import random
        dummy_vec = [random.uniform(-0.1, 0.1) for _ in range(DIMENSION)]
        results   = index.query(vector=dummy_vec, top_k=100)
        papers    = set(r.meta.get("paper", "") for r in results if r.meta.get("paper"))

        return {
            "vectors": len(results),
            "papers":  len(papers),
        }
    except Exception as e:
        print(f"[get_index_stats error] {e}")
        return {"vectors": 0, "papers": 0}