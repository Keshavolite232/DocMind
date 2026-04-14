"""
Streamlit UI — RAG PDF Chatbot
Run with: streamlit run app/ui.py
"""

import os
import sys
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Streamlit adds the script's folder (app/) to sys.path, not the project root.
# This line ensures `from app.rag_engine import RAGEngine` resolves correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()   # picks up ANTHROPIC_API_KEY and OPENAI_API_KEY from your .env file

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="DocMind · PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #0d0f14;
    --surface:  #151821;
    --border:   #252a38;
    --accent:   #5b8dee;
    --accent2:  #e05b8d;
    --text:     #e8eaf0;
    --muted:    #7b8099;
    --success:  #4caf7d;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
}

/* Left panel (replaces native sidebar) */
div[data-testid="column"]:first-child {
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 1.5rem 1rem !important;
    min-height: 100vh;
}

/* Header */
.docmind-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 2rem;
}
.docmind-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.docmind-tag {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 2px 8px;
    border-radius: 4px;
}

/* Chat messages */
.chat-bubble {
    padding: 14px 18px;
    border-radius: 12px;
    margin-bottom: 12px;
    line-height: 1.65;
    font-size: 0.95rem;
    max-width: 85%;
}
.chat-user {
    background: linear-gradient(135deg, #1e2a4a, #1a2240);
    border: 1px solid #2d3d6e;
    margin-left: auto;
    text-align: right;
}
.chat-assistant {
    background: var(--surface);
    border: 1px solid var(--border);
}
.source-pill {
    display: inline-block;
    font-size: 0.72rem;
    background: #1a2240;
    border: 1px solid #2d3d6e;
    color: var(--accent);
    padding: 2px 8px;
    border-radius: 20px;
    margin: 4px 3px 0 0;
    font-family: 'DM Mono', monospace;
}

/* Doc card */
.doc-card {
    background: #1c2030;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}
.doc-card .doc-name { font-weight: 600; color: var(--text); }
.doc-card .doc-meta { color: var(--muted); font-size: 0.78rem; margin-top: 2px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #4a7de0);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Input */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border);
    border-radius: 10px;
    padding: 8px;
}

/* Selectbox */
.stSelectbox div[data-baseweb="select"] {
    background: var(--surface);
    border-color: var(--border);
}

/* Status/alert */
.stSuccess { background: #1a3028 !important; border-color: var(--success) !important; }
.stInfo    { background: #1a2a40 !important; border-color: var(--accent) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "engine": None,
        "messages": [],           # [{role, content, sources}]
        "ingested_docs": [],      # [{name, pages, chunks}]
        "api_key_set": False,
        "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_engine(anthropic_key: str):
    """Cached engine — uses fixed defaults, no user configuration."""
    from app.rag_engine import RAGEngine
    return RAGEngine(anthropic_api_key=anthropic_key)


def render_message(role: str, content: str, sources: list = None):
    css_class = "chat-user" if role == "user" else "chat-assistant"
    icon = "🧑" if role == "user" else "🤖"
    html = f'<div class="chat-bubble {css_class}"><strong>{icon}</strong> {content}'
    if sources:
        pills = "".join(
            f'<span class="source-pill">📄 {s["file"]} p.{s["page"]}</span>'
            for s in sources
        )
        html += f"<div style='margin-top:8px'>{pills}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── Two-column layout (replaces native sidebar which collapses on Render) ───────

col_panel, col_chat = st.columns([1, 3], gap="small")

# ── Left panel ─────────────────────────────────────────────────────────────────

with col_panel:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.4rem;
                background:linear-gradient(135deg,#5b8dee,#e05b8d);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;margin-bottom:1.5rem'>
    📄 DocMind
    </div>
    """, unsafe_allow_html=True)

    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not api_key:
        st.error("ANTHROPIC_API_KEY not found in environment.")

    st.markdown("#### 📥 Upload PDFs")

    uploaded = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded and st.button("Ingest Documents", use_container_width=True):
        if not st.session_state.engine:
            with st.spinner("Loading embedding model — first load takes ~30s..."):
                try:
                    st.session_state.engine = get_engine(api_key)
                    st.session_state.api_key_set = True
                except Exception as e:
                    st.error(f"Failed to start engine: {e}")
                    st.stop()

        if not st.session_state.engine:
            st.error("Engine failed to initialise. Check your ANTHROPIC_API_KEY.")
            st.stop()

        for file in uploaded:
            with st.status(f"Processing {file.name}...", expanded=True) as status:
                tmp_path = None
                try:
                    st.write("💾 Saving upload...")
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    st.write("📄 Loading PDF pages...")
                    engine = st.session_state.engine

                    st.write("✂️ Splitting into chunks...")
                    st.write("🔢 Generating embeddings & storing...")
                    st.write("🔗 Building retrieval chain...")

                    result = engine.ingest_pdf(tmp_path)
                    result["name"] = file.name

                    if result["status"] == "success":
                        st.session_state.ingested_docs.append(result)
                        status.update(
                            label=f"✓ {file.name} — {result['pages']} pages, {result['chunks']} chunks",
                            state="complete",
                        )
                    else:
                        status.update(label=f"↩ {file.name} already ingested", state="complete")

                except Exception as e:
                    status.update(label=f"✗ Failed: {file.name}", state="error")
                    st.error(f"Error at step above: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    if st.session_state.ingested_docs:
        st.markdown("---")
        st.markdown("#### 📚 Indexed Documents")
        for doc in st.session_state.ingested_docs:
            st.markdown(f"""
            <div class="doc-card">
              <div class="doc-name">📄 {doc['name']}</div>
              <div class="doc-meta">{doc.get('pages','?')} pages · {doc.get('chunks','?')} chunks</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.engine:
            st.session_state.engine.clear_memory()
        st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────────

with col_chat:
    st.markdown("""
    <div class="docmind-header">
      <p class="docmind-title">DocMind</p>
      <span class="docmind-tag">RAG · PDF Q&A</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.ingested_docs:
        st.markdown("""
        <div style='text-align:center;padding:4rem 2rem;color:#7b8099'>
          <div style='font-size:3rem;margin-bottom:1rem'>📄</div>
          <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:#e8eaf0;margin-bottom:.5rem'>
            No documents yet
          </div>
          <div style='font-size:.9rem'>
            Upload PDFs on the left to get started.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style='color:#7b8099;font-size:.9rem;text-align:center;padding:1.5rem 0'>
                  Documents loaded. Ask anything about them below ↓
                </div>
                """, unsafe_allow_html=True)
            for msg in st.session_state.messages:
                render_message(msg["role"], msg["content"], msg.get("sources"))

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        input_col, send_col = st.columns([5, 1])
        with input_col:
            user_input = st.text_input(
                "Ask a question",
                placeholder="What does this document say about...?",
                label_visibility="collapsed",
                key="user_input",
            )
        with send_col:
            send = st.button("Send →", use_container_width=True)

        if send and user_input.strip():
            question = user_input.strip()
            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.engine.query(question)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Error: {e}",
                        "sources": [],
                    })
            st.rerun()
