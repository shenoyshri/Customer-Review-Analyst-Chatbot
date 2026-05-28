import os
import pickle
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
INDEX_PATH   = "faiss_index.pkl"
DOCS_PATH    = "documents.pkl"

st.set_page_config(page_title="Customer Review Analyst", page_icon="💬", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        padding: 1.4rem 2rem; border-radius: 12px;
        margin-bottom: 1.5rem; color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.7; font-size: 0.88rem; }
    .bubble-user {
        background: #e8f4fd; border-left: 4px solid #2196F3;
        padding: 0.7rem 1rem; border-radius: 0 10px 10px 0; margin: 0.5rem 0;
        color: #0d1b2a !important;
    }
    .bubble-bot {
        background: #f0f9f0; border-left: 4px solid #4CAF50;
        padding: 0.7rem 1rem; border-radius: 0 10px 10px 0; margin: 0.5rem 0;
        color: #0d1b2a !important;
    }
    .bubble-user *, .bubble-bot * { color: #0d1b2a !important; }        
    .label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
             letter-spacing: 0.05em; margin-bottom: 0.25rem; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load resources (cached so they run only once) ─────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model …")
def load_embed_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Loading FAISS index …")
def load_index_and_docs():
    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    return index, docs


# ── RAG helpers ───────────────────────────────────────────────────────────────

def retrieve_context(query, model, index, documents, top_k=3):
    query_emb = model.encode([query]).astype("float32")
    _, indices = index.search(query_emb, k=top_k)
    return "\n\n".join(documents[i] for i in indices[0])


def build_prompt(question, context, chat_history):
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in chat_history[-6:]
    )
    return f"""You are a Customer Review Analyst Assistant.

Conversation History:
{history_text or "None yet."}

Context (retrieved customer reviews):
{context}

Question:
{question}

Instructions:
- Answer only from the reviews in the context above.
- Summarise customer opinions clearly and conversationally.
- Highlight common issues or praises when multiple reviews mention them.
- If the context lacks enough information, say so politely.
"""


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    api_key = st.text_input(
        "Gemini API Key", type="password", placeholder="AIza…",
        help="Get yours at https://aistudio.google.com/app/apikey"
    ) or os.getenv("GEMINI_API_KEY", "")

    st.markdown("---")
    top_k = st.slider("Reviews to retrieve (Top-K)", min_value=1, max_value=10, value=3)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>💬 Customer Review Analyst</h1>
    <p>RAG Chatbot · SentenceTransformer + FAISS + Gemini 2.5 Flash</p>
</div>
""", unsafe_allow_html=True)

# ── Guards ────────────────────────────────────────────────────────────────────

if not api_key:
    st.warning("👈 Enter your Gemini API key in the sidebar to start.", icon="🔑")
    st.stop()

try:
    embed_model = load_embed_model()
    faiss_index, documents = load_index_and_docs()
except FileNotFoundError as e:
    st.error(
        f"Pickle file not found: **{e.filename}**\n\n"
        "Run the last cell in your notebook to generate `faiss_index.pkl` and `documents.pkl`, "
        "then place them in the same folder as `app.py`."
    )
    st.stop()

gemini_client = genai.Client(api_key=api_key)

# Stats row
c1, c2, c3 = st.columns(3)
c1.metric("Reviews Indexed", f"{faiss_index.ntotal:,}")
c2.metric("Embedding Model", MODEL_NAME)
c3.metric("LLM", GEMINI_MODEL)
st.markdown("---")

# ── Chat ──────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='bubble-user'><div class='label' style='color:#1565C0;'>🧑 You</div>{msg['content']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='bubble-bot'><div class='label' style='color:#2E7D32;'>🤖 Assistant</div>{msg['content']}</div>",
            unsafe_allow_html=True,
        )

# Starter question buttons (shown only on fresh chat)
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking …")
    starters = [
        "Which product has the best reviews?",
        "What are the most common complaints?",
        "Which product is good for long battery life?",
        "What do customers say about build quality?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(starters):
        if cols[i % 2].button(q, use_container_width=True):
            st.session_state.pending = q
            st.rerun()

# Pick up starter-button click or typed input
user_input = st.session_state.pop("pending", None) or st.chat_input("Ask about the customer reviews …")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(
        f"<div class='bubble-user'><div class='label' style='color:#1565C0;'>🧑 You</div>{user_input}</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Thinking …"):
        try:
            context = retrieve_context(user_input, embed_model, faiss_index, documents, top_k=top_k)
            prompt  = build_prompt(user_input, context, st.session_state.messages[:-1])
            answer  = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text
        except Exception as e:
            answer = f"⚠️ Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.markdown(
        f"<div class='bubble-bot'><div class='label' style='color:#2E7D32;'>🤖 Assistant</div>{answer}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("📄 Retrieved reviews used for this answer"):
        st.text(context)
