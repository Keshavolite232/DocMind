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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg:       #0f1117;
    --panel:    #161b27;
    --surface:  #1e2433;
    --border:   #2a3348;
    --accent:   #6366f1;
    --accent-h: #4f46e5;
    --teal:     #2dd4bf;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --success:  #34d399;
    --danger:   #f87171;
    --radius:   10px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
}

/* ── App shell ── */
[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stHorizontalBlock"]  { gap: 0 !important; align-items: stretch; }

/* ── Left panel ── */
[data-testid="stColumn"]:first-child > div:first-child {
    background: var(--panel);
    border-right: 1px solid var(--border);
    min-height: 100vh;
    padding: 1.8rem 1.2rem;
}

/* ── Panel logo ── */
.panel-logo {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 1.25rem;
    color: var(--text);
    letter-spacing: -0.02em;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-logo span {
    background: linear-gradient(135deg, var(--accent), var(--teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
}

/* ── Chat header ── */
.chat-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 1.2rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.chat-title {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, var(--accent), var(--teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.chat-badge {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 3px 8px;
    border-radius: 20px;
}

/* ── Chat bubbles ── */
.chat-bubble {
    padding: 12px 16px;
    border-radius: var(--radius);
    margin-bottom: 10px;
    line-height: 1.6;
    font-size: 0.92rem;
    max-width: 82%;
}
.chat-user {
    background: var(--accent);
    background: linear-gradient(135deg, var(--accent-h), var(--accent));
    color: #fff;
    margin-left: auto;
    border-bottom-right-radius: 3px;
}
.chat-assistant {
    background: var(--surface);
    border: 1px solid var(--border);
    border-bottom-left-radius: 3px;
}
.source-pill {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 500;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    padding: 2px 8px;
    border-radius: 20px;
    margin: 5px 3px 0 0;
}

/* ── Doc card ── */
.doc-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 9px 12px;
    margin-bottom: 6px;
    font-size: 0.82rem;
}
.doc-card .doc-name { font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.doc-card .doc-meta { color: var(--muted); font-size: 0.74rem; margin-top: 2px; }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 1rem !important;
    transition: background 0.2s, transform 0.1s !important;
    width: 100%;
}
.stButton > button:hover { background: var(--accent-h) !important; }
.stButton > button:active { transform: scale(0.98) !important; }

/* Clear button — subdued */
.stButton:last-child > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
}
.stButton:last-child > button:hover { border-color: var(--danger) !important; color: var(--danger) !important; }

/* ── Text input ── */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    padding: 4px 8px;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* ── Status/alerts ── */
[data-testid="stStatusWidget"] { background: var(--surface) !important; border-color: var(--border) !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: var(--muted) !important; font-size: 0.85rem !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header[data-testid="stHeader"] { display: none; }
[data-testid="stDecoration"] { display: none; }
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
    <div class="panel-logo">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
        <polyline points="10 9 9 9 8 9"/>
      </svg>
      <span>DocMind</span>
      <span class="chat-badge">RAG · PDF Q&A</span>
    </div>
    """, unsafe_allow_html=True)

    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not api_key:
        st.error("ANTHROPIC_API_KEY not set.")

    st.markdown('<p class="section-label">Upload Documents</p>', unsafe_allow_html=True)

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
        st.markdown('<p class="section-label">Indexed Documents</p>', unsafe_allow_html=True)
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
    if not st.session_state.ingested_docs:
        st.markdown("""
        <div style='text-align:center;padding:5rem 2rem;'>
          <div style='font-size:2.5rem;margin-bottom:1rem;opacity:0.3'>📄</div>
          <div style='font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:.4rem;letter-spacing:-0.01em'>
            No documents loaded
          </div>
          <div style='font-size:0.85rem;color:#64748b;'>
            Upload one or more PDFs on the left, then click Ingest.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style='color:#64748b;font-size:0.85rem;text-align:center;padding:2rem 0;'>
                  Documents ready — ask anything below
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
