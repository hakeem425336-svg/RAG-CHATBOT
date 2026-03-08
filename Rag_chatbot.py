import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import SQLChatMessageHistory  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ─── Setup ────────────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="CHATBOT · RAG", layout="wide", page_icon="📚")

DB_PATH = "chat_history.db"  

# ─── Global Styles ────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">

<style>
:root {
  --bg:        #09090C;
  --surface:   #111318;
  --card:      #16181F;
  --border:    #23262F;
  --border2:   #2E3140;
  --gold:      #C9921A;
  --gold-lt:   #E8B84B;
  --gold-dim:  rgba(201,146,26,0.15);
  --text:      #DDE1EE;
  --text-dim:  #5C6070;
  --text-mid:  #9099B0;
  --user-bg:   #1C1F2B;
  --ai-bg:     #111318;
  --radius:    12px;
}

* { box-sizing: border-box; }

.stApp {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--text);
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }

.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

.brand-block {
  padding: 32px 24px 20px;
  border-bottom: 1px solid var(--border);
}
.brand-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: 30px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: 0.04em;
  line-height: 1;
  margin: 0 0 4px;
}
.brand-subtitle {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--gold);
  margin: 0 0 12px;
}
.brand-divider {
  width: 32px; height: 1px;
  background: var(--gold);
  opacity: 0.5;
}

.sidebar-section {
  padding: 20px 24px;
  border-bottom: 1px solid var(--border);
}
.sidebar-label {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 10px;
}

section[data-testid="stSidebar"] .stTextInput > div > div {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 13px !important;
}
section[data-testid="stSidebar"] .stTextInput > div > div:focus-within {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px var(--gold-dim) !important;
}
section[data-testid="stSidebar"] .stTextInput label {
  font-family: 'DM Mono', monospace !important;
  font-size: 9px !important;
  letter-spacing: 0.16em !important;
  text-transform: uppercase !important;
  color: var(--text-dim) !important;
}

[data-testid="stFileUploaderDropzone"] {
  background: var(--card) !important;
  border: 1px dashed var(--border2) !important;
  border-radius: var(--radius) !important;
  transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--gold) !important;
  background: var(--gold-dim) !important;
}
[data-testid="stFileUploaderDropzone"] span { color: var(--text-dim) !important; }
[data-testid="stFileUploaderFile"] {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 6px !important;
}
[data-testid="stFileUploaderFileName"] {
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 12px !important;
}

.stat-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: var(--gold-dim);
  border: 1px solid rgba(201,146,26,0.3);
  border-radius: 20px;
  padding: 5px 12px;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--gold-lt);
}
.stat-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--gold);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:0.4; transform:scale(0.7); }
}

.doc-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 6px;
}
.doc-icon { font-size: 14px; }
.doc-name {
  font-size: 12px;
  color: var(--text-mid);
  font-family: 'DM Mono', monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}

.main-wrap { padding: 0 48px; }

.page-header {
  padding: 36px 0 22px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 28px;
}
.page-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: 44px;
  font-weight: 300;
  font-style: italic;
  color: var(--text);
  line-height: 1;
  margin: 0 0 8px;
}
.page-title span { color: var(--gold-lt); font-weight: 600; font-style: normal; }
.page-meta {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--text-dim);
  letter-spacing: 0.1em;
}

.stTextInput > div > div {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 13px !important;
}
.stTextInput > div > div:focus-within {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px var(--gold-dim) !important;
}
.stTextInput label {
  font-family: 'DM Mono', monospace !important;
  font-size: 9px !important;
  letter-spacing: 0.16em !important;
  text-transform: uppercase !important;
  color: var(--text-dim) !important;
}

[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 6px 0 !important;
  animation: fadeUp 0.22s ease forwards;
}
@keyframes fadeUp {
  from { opacity:0; transform:translateY(8px); }
  to   { opacity:1; transform:translateY(0); }
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {
  background: var(--user-bg) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 16px 4px 16px 16px !important;
  padding: 13px 17px !important;
  font-size: 14px !important;
  color: var(--text) !important;
  max-width: 68% !important;
  margin-left: auto !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] {
  background: var(--ai-bg) !important;
  border: 1px solid var(--border) !important;
  border-left: 2px solid var(--gold) !important;
  border-radius: 4px 16px 16px 16px !important;
  padding: 13px 17px !important;
  font-size: 14px !important;
  line-height: 1.75 !important;
  color: var(--text) !important;
  max-width: 84% !important;
}

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 50% !important;
  width: 32px !important; height: 32px !important;
  min-width: 32px !important;
  font-size: 14px !important;
}

[data-testid="stChatInput"] {
  background: var(--surface) !important;
  border-top: 1px solid var(--border) !important;
  padding: 16px 48px !important;
}
[data-testid="stChatInput"] textarea {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px var(--gold-dim) !important;
  outline: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-dim) !important; }

.streamlit-expanderHeader {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-dim) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.06em !important;
}
.streamlit-expanderContent {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 8px 8px !important;
  padding: 14px !important;
}

code, pre {
  background: #0D0F15 !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 12px !important;
  color: var(--gold-lt) !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold); }

.empty-state { text-align: center; padding: 88px 40px; }
.empty-icon { font-size: 52px; margin-bottom: 16px; opacity: 0.35; }
.empty-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: 26px; font-weight: 300; font-style: italic;
  color: var(--text-dim); margin-bottom: 8px;
}
.empty-sub {
  font-family: 'DM Mono', monospace;
  font-size: 11px; color: var(--text-dim);
  letter-spacing: 0.08em; opacity: 0.55;
}

.chunk-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 8px;
}
.chunk-meta {
  font-family: 'DM Mono', monospace;
  font-size: 10px; color: var(--gold);
  letter-spacing: 0.1em; text-transform: uppercase;
  margin-bottom: 6px;
}
.chunk-text { font-size: 12px; color: var(--text-mid); line-height: 1.6; }

div[data-testid="stNotification"] {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--radius) !important;
  color: var(--text-mid) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-block">
      <p class="brand-subtitle">Document Intelligence</p>
      <h1 class="brand-title">RAG CHATBOT</h1>
      <div class="brand-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Authentication</p>', unsafe_allow_html=True)
    api_key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_••••••••••••")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Upload Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here", type="pdf",
        accept_multiple_files=True, label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:20px 24px;">
      <p style="font-family:'DM Mono',monospace;font-size:10px;
                color:#3A3D4A;line-height:1.8;letter-spacing:0.05em;margin:0;">
        Upload PDFs from the sidebar<br>
        then ask questions below.<br>
        Answers are grounded in<br>
        your documents only.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Auth guard ───────────────────────────────────────────────────────────────
api_key = api_key_input or os.getenv("GROQ_API_KEY")
if not api_key:
    st.markdown("""
    <div class="main-wrap">
      <div class="empty-state">
        <div class="empty-icon">🔑</div>
        <p class="empty-title">Enter your Groq API key</p>
        <p class="empty-sub">Paste it in the sidebar to unlock the assistant</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Models ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

embeddings = load_embeddings()
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")


# ─── Upload guard ─────────────────────────────────────────────────────────────
if not uploaded_files:
    st.markdown("""
    <div class="main-wrap">
      <div class="empty-state">
        <div class="empty-icon">📂</div>
        <p class="empty-title">No documents loaded</p>
        <p class="empty-sub">Upload one or more PDFs from the sidebar to begin</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── upload PDFs 
all_docs, tmp_paths = [], []
for pdf in uploaded_files:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf.getvalue())
    tmp.close()
    tmp_paths.append(tmp.name)
    loader = PyPDFLoader(tmp.name)
    docs = loader.load()
    for d in docs:
        d.metadata["source_file"] = pdf.name
    all_docs.extend(docs)

for p in tmp_paths:
    try:
        os.unlink(p)
    except Exception:
        pass


#  loaded docs + stats 
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Loaded Files</p>', unsafe_allow_html=True)
    for pdf in uploaded_files:
        st.markdown(f"""
        <div class="doc-item">
          <span class="doc-icon">📄</span>
          <span class="doc-name">{pdf.name}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


#  Build index 
@st.cache_resource(show_spinner="Building index…")
def build_index(_docs, file_key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    splits = splitter.split_documents(_docs)
    vs = Chroma.from_documents(splits, embeddings, persist_directory="chroma_index")
    return vs, splits

file_key = tuple(f.name for f in uploaded_files)
vectorstore, splits = build_index(all_docs, file_key)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 24px 20px;">
      <div class="stat-badge">
        <div class="stat-dot"></div>
        {len(splits)} chunks · {len(all_docs)} pages
      </div>
    </div>
    """, unsafe_allow_html=True)


#  Prompts 
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest question into a standalone search query using the chat history for context. "
     "Return only the rewritten query, no extra text."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a STRICT RAG assistant. Answer using ONLY the provided context.\n"
     "If the context does NOT contain the answer, reply exactly:\n"
     "'Out of scope — not found in provided documents.'\n"
     "Do NOT use outside knowledge.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


# ─── Permanent memory via SQLite ──────────────────────────────────────────────
# SQLChatMessageHistory saves every message to chat_history.db on disk.
# The history survives app restarts — just use the same session_id to reload it.

def get_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{DB_PATH}"
    )


# ─── Main chat UI 
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

st.markdown(f"""
<div class="page-header">
  <h2 class="page-title">Ask your <span>documents</span></h2>
  <p class="page-meta">
    {len(uploaded_files)} PDF{"s" if len(uploaded_files) > 1 else ""} loaded
    &nbsp;·&nbsp; RAG &nbsp;·&nbsp; Llama 3.1 8B &nbsp;·&nbsp; Groq
    &nbsp;·&nbsp; 💾 Persistent Memory
  </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 5])
with col1:
    session_id = st.text_input("Session", value="default", placeholder="default")
with col2:
    # Button to wipe history for this session
    if st.button("🗑 Clear this session's history"):
        get_history(session_id).clear()
        st.success("History cleared!")
        st.rerun()

# Load persistent history and display it
history = get_history(session_id)
past_messages = history.messages

if past_messages:
    for m in past_messages:
        role = "user" if m.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(m.content)
else:
    st.markdown("""
    <div style="text-align:center;padding:56px 0 16px;opacity:0.45;">
      <p style="font-family:'Cormorant Garamond',serif;font-size:22px;
                font-weight:300;font-style:italic;color:#5C6070;margin:0;">
        Your conversation will appear here…
      </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)



def _join_docs(docs, max_chars=7000):
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)


#  Chat input + inference 
user_q = st.chat_input("Ask something about your documents…")

if user_q:
    history = get_history(session_id)

    with st.chat_message("user"):
        st.write(user_q)

    with st.spinner(""):
        rewrite_msgs = contextualize_q_prompt.format_messages(
            chat_history=history.messages, input=user_q
        )
        standalone_q = llm.invoke(rewrite_msgs).content.strip()

        docs = retriever.invoke(standalone_q)

        if not docs:
            answer = "Out of scope — not found in provided documents."
        else:
            context_str = _join_docs(docs)
            qa_msgs = qa_prompt.format_messages(
                chat_history=history.messages,
                input=user_q,
                context=context_str
            )
            answer = llm.invoke(qa_msgs).content

    with st.chat_message("assistant"):
        st.write(answer)

    #  Save permanently 
    history.add_user_message(user_q)
    history.add_ai_message(answer)

    if docs:
        with st.expander("✦  Rewritten query"):
            st.code(standalone_q or "(empty)", language="text")
            st.caption(f"Retrieved {len(docs)} chunk(s)")

        with st.expander("✦  Source chunks"):
            for i, doc in enumerate(docs, 1):
                src     = doc.metadata.get("source_file", "Unknown")
                page    = doc.metadata.get("page", "?")
                preview = doc.page_content[:420] + ("…" if len(doc.page_content) > 420 else "")
                st.markdown(f"""
                <div class="chunk-card">
                  <p class="chunk-meta">#{i} &nbsp;·&nbsp; {src} &nbsp;·&nbsp; page {page}</p>
                  <p class="chunk-text">{preview}</p>
                </div>

                """, unsafe_allow_html=True)
