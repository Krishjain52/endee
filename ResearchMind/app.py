import streamlit as st
import tempfile
import os
import time as _time
from ingest import ingest_pdf, ingest_arxiv_url, get_index_stats
from agent import ResearchAgent

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
* { font-family: 'Syne', sans-serif; }
code, pre { font-family: 'Space Mono', monospace; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
section[data-testid="stSidebar"] { background: #0f0f1a !important; border-right: 1px solid #1e1e3a; }

.main-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.8rem;
    background: linear-gradient(135deg, #7c6af5 0%, #4fc3f7 50%, #81c784 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -1px; margin-bottom: 0; }
.subtitle { color: #6060a0; font-size: 0.95rem; letter-spacing: 2px;
    text-transform: uppercase; margin-top: 4px; font-family: 'Space Mono', monospace; }

.toast {
    position: fixed; bottom: 32px; left: 50%; transform: translateX(-50%);
    background: #12122a; border: 1px solid #7c6af5;
    border-radius: 12px; padding: 16px 28px;
    display: flex; align-items: center; gap: 12px;
    box-shadow: 0 8px 40px rgba(124,106,245,0.25);
    z-index: 9999; animation: slideUp 0.3s ease, fadeOut 0.4s ease 2.6s forwards;
    font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #c0c0e8;
    min-width: 320px; max-width: 520px;
}
.toast-dot { width: 8px; height: 8px; background: #7c6af5; border-radius: 50%; flex-shrink: 0; }
.toast.error { border-color: #f56a6a; box-shadow: 0 8px 40px rgba(245,106,106,0.2); }
.toast.error .toast-dot { background: #f56a6a; }
@keyframes slideUp {
    from { opacity: 0; transform: translateX(-50%) translateY(20px); }
    to   { opacity: 1; transform: translateX(-50%) translateY(0); }
}
@keyframes fadeOut {
    from { opacity: 1; }
    to   { opacity: 0; pointer-events: none; }
}

.paper-card { background: #0f0f1a; border: 1px solid #1e1e3a; border-radius: 8px; padding: 16px; margin: 8px 0; }
.paper-chunk { color: #8080b0; font-size: 0.85rem; margin-top: 8px; line-height: 1.6; }
.score-badge { display: inline-block; background: #1a1a30; border: 1px solid #2e2e5a;
    border-radius: 3px; padding: 2px 8px; font-family: 'Space Mono', monospace;
    font-size: 0.75rem; color: #7c6af5; float: right; }

.brief-section { background: #0d0d1f; border: 1px solid #1e1e3a; border-radius: 8px; padding: 24px; margin: 12px 0; }
.brief-section h3 { color: #7c6af5; font-family: 'Space Mono', monospace; font-size: 0.85rem;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px; }
.brief-section p, .brief-section li { color: #b0b0d0; line-height: 1.8; font-size: 0.92rem; }

.stat-box { background: #0f0f1a; border: 1px solid #1e1e3a; border-radius: 6px; padding: 12px 16px; text-align: center; }
.stat-num { font-family: 'Space Mono', monospace; font-size: 1.8rem; color: #7c6af5; font-weight: 700; }
.stat-label { font-size: 0.75rem; color: #4040a0; letter-spacing: 1px; text-transform: uppercase; }

div[data-testid="stTextInput"] input { background: #0f0f1a !important; border: 1px solid #2e2e5a !important;
    border-radius: 6px !important; color: #e0e0f0 !important; }
div[data-testid="stTextInput"] input:focus { border-color: #7c6af5 !important; }
.stButton > button { background: linear-gradient(135deg, #7c6af5, #4fc3f7) !important;
    color: white !important; border: none !important; border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important; letter-spacing: 1px !important; }
hr { border-color: #1e1e3a !important; }

.thinking-bar { background: #0f0f1a; border: 1px solid #1e1e3a; border-radius: 8px;
    padding: 20px 24px; margin: 16px 0; text-align: center; }
.thinking-bar p { color: #6060a0; font-family: 'Space Mono', monospace; font-size: 0.85rem; margin: 0; }
.dot-pulse { display: inline-flex; gap: 6px; margin-top: 12px; }
.dot-pulse span { width: 8px; height: 8px; background: #7c6af5; border-radius: 50%;
    animation: pulse 1.2s infinite ease-in-out; }
.dot-pulse span:nth-child(2) { animation-delay: 0.2s; }
.dot-pulse span:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse { 0%,80%,100% { opacity: 0.2; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1); } }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("has_papers", False),
    ("toast", None),
    ("toast_timer", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def show_toast(msg: str, ok: bool = True):
    st.session_state["toast"] = {"msg": msg, "ok": ok}
    st.session_state["toast_timer"] = _time.time()

# Toast — show for 3 seconds then clear
if st.session_state["toast"]:
    elapsed = _time.time() - st.session_state["toast_timer"]
    if elapsed < 3:
        t   = st.session_state["toast"]
        cls = "toast" if t["ok"] else "toast error"
        st.markdown(
            f'<div class="{cls}"><span class="toast-dot"></span>{t["msg"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.session_state["toast"] = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">ResearchMind</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Agentic Research Intelligence · Powered by Endee + Groq</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Paper Corpus")
    st.markdown("---")

    stats = get_index_stats()
    if st.session_state["has_papers"] and stats["vectors"] == 0:
        stats = {"vectors": "✓", "papers": "✓"}

    st.markdown(f'<div class="stat-box"><div class="stat-num">{stats["papers"]}</div><div class="stat-label">Papers Uploaded</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Upload PDFs**")
    uploaded_files = st.file_uploader(
        "Drop research papers here",
        type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        if st.button("INGEST PAPERS", use_container_width=True):
            names = []
            total_chunks = 0
            with st.spinner("Ingesting papers..."):
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    count = ingest_pdf(tmp_path, f.name)
                    os.unlink(tmp_path)
                    names.append(f.name)
                    total_chunks += count
            st.session_state["has_papers"] = True
            label = names[0] if len(names) == 1 else f"{len(names)} papers"
            show_toast(f"{label} ingested — {total_chunks} chunks stored in Endee")
            st.rerun()

    st.markdown("---")

    st.markdown("**Or paste ArXiv URL**")
    arxiv_url = st.text_input(
        "ArXiv URL", placeholder="https://arxiv.org/abs/xxxx.xxxxx",
        label_visibility="collapsed", key="arxiv_input"
    )
    if st.button("FETCH & INGEST", use_container_width=True):
        if not arxiv_url:
            st.warning("Paste a URL first.")
        else:
            with st.spinner("Fetching & ingesting (~20s)..."):
                result = ingest_arxiv_url(arxiv_url)
            if result["success"]:
                st.session_state["has_papers"] = True
                show_toast(f"Fetched & ingested — {result['chunks']} chunks stored in Endee")
                st.rerun()
            else:
                show_toast(f"Failed: {result['error']}", ok=False)
                st.rerun()

    st.markdown("---")
    with st.expander("Try sample topics"):
        st.markdown("""
        Ingest papers from ArXiv on:
        - Transformers / Attention
        - Diffusion Models
        - LLM Fine-tuning
        - Reinforcement Learning
        """)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("### Ask the Research Agent")
st.markdown('<p style="color:#5050a0; font-size:0.85rem; font-family: Space Mono, monospace;">The agent decomposes your question, searches Endee iteratively, and synthesizes findings across papers.</p>', unsafe_allow_html=True)

query = st.text_input(
    "Research question",
    placeholder="e.g. What are the two RAG formulations and how do they differ?",
    label_visibility="collapsed"
)

col_btn, col_depth = st.columns([2, 1])
with col_btn:
    run = st.button("LAUNCH AGENT", use_container_width=True)
with col_depth:
    depth = st.selectbox(
        "Depth", ["Quick", "Standard", "Deep"],
        index=1, label_visibility="collapsed"
    )

depth_map = {"Quick": 2, "Standard": 3, "Deep": 5}
max_iter = depth_map[depth]

if run and query:
    if not st.session_state["has_papers"] and stats["vectors"] == 0:
        st.warning("No papers ingested yet. Upload PDFs or fetch from ArXiv in the sidebar first.")
    else:
        st.markdown("---")

        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
            <div class="thinking-bar">
                <p>Agent is researching your question...</p>
                <div class="dot-pulse"><span></span><span></span><span></span></div>
            </div>
        """, unsafe_allow_html=True)

        agent = ResearchAgent(max_iterations=max_iter)
        final_result = None

        for event in agent.run_stream(query):
            if event["type"] == "done":
                final_result = event

        thinking_placeholder.empty()

        if final_result:
            brief = final_result["brief"]

            st.markdown("### Research Brief")

            current_section = "Overview"
            current_lines   = []
            sections        = {}
            for line in brief.split('\n'):
                if line.startswith('## '):
                    if current_lines:
                        sections[current_section] = '\n'.join(current_lines).strip()
                    current_section = line[3:].strip()
                    current_lines   = []
                elif not line.startswith('# '):
                    current_lines.append(line)
            if current_lines:
                sections[current_section] = '\n'.join(current_lines).strip()

            for title, content in sections.items():
                if content:
                    st.markdown(
                        f'<div class="brief-section"><h3>{title}</h3><p>{content}</p></div>',
                        unsafe_allow_html=True
                    )

            st.markdown("### Retrieved Evidence")
            sources  = final_result["sources"]
            by_paper = {}
            for s in sources:
                p = s["paper"]
                if p not in by_paper:
                    by_paper[p] = []
                by_paper[p].append(s)

            for paper, chunks in by_paper.items():
                with st.expander(f"{paper} — {len(chunks)} relevant chunks"):
                    for c in chunks:
                        st.markdown(
                            f'<div class="paper-card">'
                            f'<span class="score-badge">{c["score"]:.3f}</span>'
                            f'<div class="paper-chunk">{c["text"][:400]}{"..." if len(c["text"]) > 400 else ""}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            st.download_button(
                "Download Research Brief",
                data=brief,
                file_name=f"brief_{query[:30].replace(' ','_')}.md",
                mime="text/markdown"
            )