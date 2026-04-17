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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

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

/* ── Chat input bar ── */
[data-testid="stForm"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 6px 6px 6px 14px !important;
}
[data-testid="stForm"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
/* Hide "Press Enter to submit form" hint */
[data-testid="InputInstructions"] { display: none !important; }

/* Input inside form — transparent, no border */
[data-testid="stForm"] .stTextInput > div > div > input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    padding-left: 0 !important;
}
[data-testid="stForm"] .stTextInput > div > div > input:focus {
    border: none !important;
    box-shadow: none !important;
}

/* Submit button — compact, right side */
[data-testid="stFormSubmitButton"] {
    display: flex !important;
    justify-content: flex-end !important;
}
[data-testid="stFormSubmitButton"] > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.4rem 1.2rem !important;
    transition: background 0.2s, transform 0.1s !important;
    width: auto !important;
    min-width: 90px;
    height: 2.3rem;
}
[data-testid="stFormSubmitButton"] > button:hover { background: var(--accent-h) !important; }
[data-testid="stFormSubmitButton"] > button:active { transform: scale(0.97) !important; }

/* ── Text input (outside form) ── */
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

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div { background: var(--surface) !important; border-radius: 99px; }
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, var(--accent), var(--teal)) !important; border-radius: 99px; }

/* ── Badge ── */
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
    /* override the gradient applied to all panel-logo span children */
    -webkit-text-fill-color: var(--muted) !important;
    background-clip: unset !important;
    -webkit-background-clip: unset !important;
}

/* ── Empty / loading state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    text-align: center;
    height: 100%;
}
.empty-state-icon { font-size: 2.2rem; margin-bottom: 0.8rem; opacity: 0.25; }
.empty-state-title { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 0.3rem; }
.empty-state-sub   { font-size: 0.82rem; color: var(--muted); }

/* ── Thinking / typing indicator ── */
.chat-thinking {
    display: inline-flex !important;
    align-items: center;
    gap: 5px;
    padding: 10px 16px !important;
    width: auto !important;
}
.chat-thinking .dot {
    width: 6px;
    height: 6px;
    background: var(--muted);
    border-radius: 50%;
    display: inline-block;
    animation: typing-dot 1.2s ease-in-out infinite;
}
.chat-thinking .dot:nth-child(2) { animation-delay: 0.2s; }
.chat-thinking .dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing-dot {
    0%, 60%, 100% { transform: translateY(0);    opacity: 0.35; }
    30%            { transform: translateY(-5px); opacity: 1; }
}

/* ── Scrollable chat container ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    border: none !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header[data-testid="stHeader"] { display: none; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "engine":        None,
        "messages":      [],
        "ingested_docs": [],
        "api_key_set":      False,
        "error":            None,
        "engine_ready":     False,
        "pending_question": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()



# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_engine(anthropic_key: str):
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


# ── Two-column layout ──────────────────────────────────────────────────────────

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
            with st.spinner("⏳ Loading embedding model — first load takes ~30s..."):
                try:
                    st.session_state.engine = get_engine(api_key)
                    st.session_state.api_key_set = True
                    st.session_state.engine_ready = True
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
                    progress_bar = st.progress(0, text="Starting...")

                    def make_progress_callback(pbar):
                        def on_progress(step, total, message):
                            pbar.progress(step / total, text=message)
                        return on_progress

                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    engine = st.session_state.engine
                    result = engine.ingest_pdf(
                        tmp_path,
                        progress_callback=make_progress_callback(progress_bar),
                        display_name=file.name,
                    )
                    result["name"] = file.name

                    if result["status"] == "success":
                        progress_bar.progress(1.0, text="✓ Complete")
                        st.session_state.ingested_docs.append(result)
                        status.update(
                            label=f"✓ {file.name} — {result['pages']} pages, {result['chunks']} chunks",
                            state="complete",
                        )
                    else:
                        progress_bar.empty()
                        status.update(label=f"↩ {file.name} already ingested", state="complete")

                except Exception as e:
                    status.update(label=f"✗ Failed: {file.name}", state="error")
                    st.error(f"Error: {e}")
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
        # Empty state — no documents ingested yet
        st.markdown("""
        <div class="empty-state">
          <div class="empty-state-icon">📄</div>
          <div class="empty-state-title">No documents loaded</div>
          <div class="empty-state-sub">Upload one or more PDFs on the left panel, then click <strong>Ingest Documents</strong>.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Fixed-height scrollable messages area ──────────────────────────────
        with st.container(height=540, border=False):
            if not st.session_state.messages:
                st.markdown("""
                <div class="empty-state" style="padding:3rem 2rem;">
                  <div class="empty-state-icon">💬</div>
                  <div class="empty-state-title">Documents ready</div>
                  <div class="empty-state-sub">Ask anything about the ingested documents below.</div>
                </div>
                """, unsafe_allow_html=True)
            for msg in st.session_state.messages:
                render_message(msg["role"], msg["content"], msg.get("sources"))
            # Thinking indicator — shows while query is in-flight
            if st.session_state.pending_question:
                st.markdown("""
                <div class="chat-bubble chat-assistant chat-thinking">
                  <span class="dot"></span>
                  <span class="dot"></span>
                  <span class="dot"></span>
                </div>
                """, unsafe_allow_html=True)

        # ── Phase 2: process pending question (no blocking spinner) ───────────
        if st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = None
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

        # ── Phase 1: capture new input (Enter key or Send button) ─────────────
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask a question",
                placeholder="What does this document say about...?",
                label_visibility="collapsed",
            )
            send = st.form_submit_button("Send →")

        if send and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            st.session_state.pending_question = user_input.strip()
            st.rerun()
