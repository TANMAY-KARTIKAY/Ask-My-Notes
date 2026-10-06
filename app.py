"""
Ask My Notes - Interactive Streamlit Web UI.
Upload college notes, syllabus, or textbooks in PDF format and ask questions.
The assistant provides grounded answers citing specific page numbers.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pdf_loader import extract_pages_from_pdf
from text_splitter import chunk_document_pages
from embeddings import EmbeddingManager
from vector_store import FAISSVectorStore
from llm_client import LLMClient
from rag_chain import AskMyNotesRAG
from create_sample_pdf import generate_sample_pdf

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Ask My Notes | Grounded Study Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .badge-page {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-score {
        background-color: #DCFCE7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-doc {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.82rem;
    }
    .chip-button {
        display: inline-block;
        margin: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# CACHED RESOURCES
# =============================================================================
@st.cache_resource(show_spinner="🧠 Loading local embedding model (all-MiniLM-L6-v2)...")
def get_embedding_manager() -> EmbeddingManager:
    """Loads and caches the SentenceTransformer embedding model."""
    return EmbeddingManager(model_name="all-MiniLM-L6-v2")


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
embedder = get_embedding_manager()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = FAISSVectorStore(dimension=embedder.dimension)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.title("📚 Ask My Notes")
    st.markdown("*Upload notes and query with verified page citations.*")
    st.divider()

    # --- 1. LLM Model Selection ---
    st.subheader("⚙️ 1. LLM Provider")
    provider = st.selectbox(
        "Choose Model Backend:",
        options=["groq", "gemini", "ollama", "openai", "claude", "offline"],
        format_func=lambda x: {
            "groq": "⚡ Groq (Free & Fast - Llama 3.3 70B)",
            "gemini": "🌟 Google Gemini (Free Tier - Flash 2.5)",
            "ollama": "🦙 Local Ollama (100% Private, No Key)",
            "openai": "🤖 OpenAI (GPT-4o-mini)",
            "claude": "🧠 Anthropic Claude (Claude 3.5)",
            "offline": "🔒 Local Extractive (No API Key Required)",
        }[x],
        index=0,
    )

    api_key = ""
    model_name = ""
    ollama_url = "http://localhost:11434"

    if provider in ["groq", "gemini", "openai", "claude"]:
        key_labels = {
            "groq": ("Groq API Key", "gsk_...", "https://console.groq.com/keys"),
            "gemini": ("Gemini API Key", "AIzaSy...", "https://aistudio.google.com/app/apikey"),
            "openai": ("OpenAI API Key", "sk-...", "https://platform.openai.com/api-keys"),
            "claude": ("Claude API Key", "sk-ant-...", "https://console.anthropic.com/settings/keys"),
        }
        label, placeholder, link = key_labels[provider]
        api_key = st.text_input(
            label,
            type="password",
            placeholder=placeholder,
            help=f"Get your key at: {link}",
        )
        st.caption(f"[Get your free {provider.capitalize()} API key here]({link})")

        # Configurable model name
        default_model = LLMClient.DEFAULT_MODELS.get(provider, "")
        model_name = st.text_input("Model Name:", value=default_model)

    elif provider == "ollama":
        ollama_url = st.text_input("Ollama Base URL:", value="http://localhost:11434")
        model_name = st.text_input("Ollama Model Name:", value="llama3.2")
        st.caption("Ensure Ollama is running (`ollama serve`). Model can be pulled via `ollama pull llama3.2`")

    elif provider == "offline":
        st.info("💡 Offline mode directly synthesizes answers from the top matching chunks without calling an external API.")

    st.divider()

    # --- 2. Document Ingestion ---
    st.subheader("📄 2. Document Upload")
    uploaded_files = st.file_uploader(
        "Upload College PDFs (notes, syllabus, slides):",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("📥 Index PDFs", use_container_width=True, type="primary")
    with col2:
        demo_btn = st.button("🧪 Demo Notes", use_container_width=True)

    # Process Uploaded PDFs
    if process_btn and uploaded_files:
        with st.spinner("Extracting text and building vector embeddings..."):
            total_added_chunks = 0
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                if filename in st.session_state.indexed_files:
                    st.toast(f"'{filename}' is already indexed.", icon="ℹ️")
                    continue

                pages = extract_pages_from_pdf(uploaded_file.getvalue(), filename=filename)
                chunks = chunk_document_pages(pages, target_words=400, overlap_words=50)

                if chunks:
                    chunk_texts = [c["text"] for c in chunks]
                    embeddings = embedder.embed_texts(chunk_texts)
                    st.session_state.vector_store.add_chunks(chunks, embeddings)
                    st.session_state.indexed_files.add(filename)
                    total_added_chunks += len(chunks)

            if total_added_chunks > 0:
                st.success(f"Successfully indexed {total_added_chunks} chunks from {len(uploaded_files)} PDF(s)!")
            else:
                st.warning("No new content could be indexed.")

    # Load Pre-built College Demo Notes
    if demo_btn:
        with st.spinner("Generating and indexing CS101 College Notes..."):
            sample_pdf_path = BASE_DIR / "CS101_Data_Structures_Notes.pdf"
            if not sample_pdf_path.exists():
                generate_sample_pdf(str(sample_pdf_path))

            filename = "CS101_Data_Structures_Notes.pdf"
            pages = extract_pages_from_pdf(str(sample_pdf_path), filename=filename)
            chunks = chunk_document_pages(pages, target_words=400, overlap_words=50)
            chunk_texts = [c["text"] for c in chunks]
            embeddings = embedder.embed_texts(chunk_texts)

            st.session_state.vector_store.clear()
            st.session_state.indexed_files.clear()
            st.session_state.vector_store.add_chunks(chunks, embeddings)
            st.session_state.indexed_files.add(filename)

            # Add welcome suggestion to chat
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "🎓 **Demo College Notes Loaded!**\n\n"
                        "I've indexed **'CS101: Data Structures and Algorithms Notes'** (4 pages).\n"
                        "You can test me right now with questions like:\n"
                        "- *'What is the late assignment submission penalty?'*\n"
                        "- *'Compare Arrays vs Linked Lists in memory and time complexity.'*\n"
                        "- *'What are AVL trees and what is their worst-case search time?'*\n"
                        "- *'What is the grading breakdown for exams and projects?'*"
                    ),
                    "sources": [],
                }
            )
            st.success("Loaded 4-page CS101 demo course notes!")
            st.rerun()

    # --- 3. Retrieval Settings ---
    with st.expander("🛠️ Advanced RAG Settings"):
        top_k = st.slider("Passages to retrieve (Top-K):", min_value=1, max_value=8, value=4)
        min_similarity = st.slider("Similarity threshold:", min_value=0.0, max_value=0.8, value=0.1, step=0.05)

    # --- 4. Index Status & Reset ---
    st.divider()
    stats = st.session_state.vector_store.get_stats()
    st.markdown(f"**📊 Indexed Chunks:** `{stats['total_chunks']}`")
    st.markdown(f"**📑 Documents:** `{stats['total_documents']}`")

    if stats["documents"]:
        with st.expander("View Indexed Documents"):
            for d in stats["documents"]:
                st.markdown(f"• **{d['name']}** ({d['pages_indexed']} pages)")

    if st.button("🗑️ Reset All Notes & Chat", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.indexed_files.clear()
        st.session_state.messages.clear()
        st.rerun()


# =============================================================================
# MAIN INTERFACE
# =============================================================================
st.markdown('<div class="main-title">Ask My Notes 📚</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Chat with your college notes, syllabus, and textbooks. Answers are grounded in your files with <b>verified page citations</b>.</div>',
    unsafe_allow_html=True,
)

# Suggested question buttons if documents are indexed
stats = st.session_state.vector_store.get_stats()
if stats["total_chunks"] > 0 and len(st.session_state.messages) <= 2:
    st.markdown("**💡 Quick Questions to Try:**")
    quick_cols = st.columns(3)
    sample_queries = [
        "What is the penalty for late assignments?",
        "How do arrays differ from linked lists?",
        "What is the worst-case complexity of AVL trees?",
    ]
    for col, q in zip(quick_cols, sample_queries):
        if col.button(f"👉 {q}", key=f"quick_{q}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display source citations if available
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📑 Sources & Verified Citations ({len(sources)} passages)"):
                for idx, src in enumerate(sources, 1):
                    doc = src.get("document_name", "Document")
                    page = src.get("page_number", "?")
                    score = src.get("similarity_percentage", "N/A")
                    text = src.get("text", "")

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <b>Source {idx}:</b>
                            <span class="badge-doc">📄 {doc}</span>
                            <span class="badge-page">🔖 Page {page}</span>
                            <span class="badge-score">🎯 {score} Match</span>
                            <br/><br/>
                            <i>"{text[:350]}{'...' if len(text) > 350 else ''}"</i>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# =============================================================================
# QUERY EXECUTION
# =============================================================================
# Handle pending query from quick button or chat input
user_query = None
if "pending_query" in st.session_state and st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    user_query = st.chat_input("Ask a question about your uploaded notes...")

if user_query:
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Check if notes are loaded
    if st.session_state.vector_store.total_chunks == 0:
        warning_msg = (
            "⚠️ **No notes have been indexed yet.**\n\n"
            "Please upload your PDF notes in the sidebar and click **'Index PDFs'**, "
            "or click **'Demo Notes'** to test immediately with sample college notes."
        )
        st.session_state.messages.append({"role": "assistant", "content": warning_msg, "sources": []})
        with st.chat_message("assistant"):
            st.markdown(warning_msg)
        st.stop()

    # 3. Initialize LLM & RAG Chain
    llm = LLMClient(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        ollama_url=ollama_url,
    )
    rag = AskMyNotesRAG(
        vector_store=st.session_state.vector_store,
        embedding_manager=embedder,
        llm_client=llm,
    )

    # 4. Stream response
    with st.chat_message("assistant"):
        stream, sources = rag.query_stream(
            query=user_query,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        response_text = st.write_stream(stream)

        # Show source cards right under the response
        if sources:
            with st.expander(f"📑 Sources & Verified Citations ({len(sources)} passages)"):
                for idx, src in enumerate(sources, 1):
                    doc = src.get("document_name", "Document")
                    page = src.get("page_number", "?")
                    score = src.get("similarity_percentage", "N/A")
                    text = src.get("text", "")

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <b>Source {idx}:</b>
                            <span class="badge-doc">📄 {doc}</span>
                            <span class="badge-page">🔖 Page {page}</span>
                            <span class="badge-score">🎯 {score} Match</span>
                            <br/><br/>
                            <i>"{text[:350]}{'...' if len(text) > 350 else ''}"</i>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": sources,
        }
    )
