# 📚 Ask My Notes: Grounded Document Chatbot

An AI-powered study assistant that lets you upload your own PDFs (college lecture notes, syllabus, textbooks, research papers) and ask questions in plain language. Answers are strictly grounded in your notes, accompanied by exact **source document names and page numbers**.

---

## 🚀 How It Works (RAG Pipeline)

```
[ Upload PDF ]
      │
      ▼
[ PDF Loader & Page Parser (pypdf) ] ──> Extracts text & tracks exact page numbers (1..N)
      │
      ▼
[ Semantic Chunker (~400-500 words) ] ──> Preserves sentence boundaries & page attribution
      │
      ▼
[ Dense Vector Embedding (SentenceTransformers) ] ──> 'all-MiniLM-L6-v2' (Normalized 384-dim)
      │
      ▼
[ Vector Database (FAISS FlatIP Index) ] ──> Fast Cosine Similarity retrieval
      │
      ▼ (When user asks a question)
[ Top-K Semantic Similarity Search ] ──> Retrieves highest-scoring note chunks
      │
      ▼
[ Grounded System Prompt Assembly ] ──> Strictly forbids hallucination + injects citations
      │
      ▼
[ LLM Generation (Groq / Gemini / Ollama / OpenAI / Claude / Offline) ]
      │
      ▼
[ Interactive UI with Page Citations & Excerpt Cards ]
```

---

## ✨ Features

- **Exact Page Number Tracking**: Every chunk retains its originating PDF name, page number, and word count.
- **Strict Anti-Hallucination Grounding**: Configured with strict prompt instructions to only answer using provided excerpts and decline when information is absent.
- **Fast Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` locally on your CPU/GPU (no embedding API charges).
- **High-Performance Vector Store**: Uses FAISS inner-product index on L2-normalized vectors for exact cosine similarity search.
- **Multiple LLM Backends**:
  - **⚡ Groq**: Free tier, blazing fast Llama 3.3 70B inference.
  - **🌟 Google Gemini**: Free tier with Gemini 2.5 Flash / 1.5 Flash.
  - **🦙 Ollama**: 100% private, local LLMs (`llama3.2`, `mistral`, `deepseek-r1`) with zero API key needed.
  - **🤖 OpenAI**: GPT-4o-mini and GPT-4o.
  - **🧠 Anthropic Claude**: Claude 3.5 Sonnet and Haiku.
  - **🔒 Local Extractive Mode**: 100% offline, zero-key fallback mode that works out-of-the-box.
- **Interactive Source Cards**: Displays relevance match score (%), page badges, and expandable preview of original text excerpts.
- **1-Click Demo Notes**: Includes a built-in 4-page sample college course syllabus and notes PDF (`CS101_Data_Structures_Notes.pdf`) covering Big-O complexity, linear structures, AVL trees, and grading policies so you can test instantly.

---

## 🛠️ Project Structure

```
ask_my_notes/
├── app.py                      # Main Streamlit web application
├── pdf_loader.py               # Page-by-page PDF text extraction with pypdf
├── text_splitter.py            # ~500-word context-preserving chunker with metadata
├── embeddings.py               # Local SentenceTransformer embeddings manager
├── vector_store.py             # FAISS index with cosine similarity search & stats
├── llm_client.py               # Multi-provider LLM connector (Groq, Gemini, Ollama, OpenAI, Claude, Offline)
├── rag_chain.py                # Retrieval-Augmented Generation pipeline & grounded prompts
├── create_sample_pdf.py        # Generates realistic 4-page CS101 lecture notes PDF
├── test_pipeline.py            # Automated end-to-end verification test suite
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows 1-click launcher
└── README.md                   # Complete documentation
```

---

## 💻 Quick Start

### 1. Installation

Ensure you have Python 3.10+ installed:

```bash
pip install -r requirements.txt
```

### 2. Launch the Application

Run the Streamlit app:

```bash
streamlit run app.py
```
*(Or double-click `run.bat` on Windows)*

The web UI will open in your browser at `http://localhost:8501`.

---

## 🧪 Testing Immediately (No PDF needed!)

1. Launch the app (`streamlit run app.py`).
2. In the sidebar under **2. Document Upload**, click the **"🧪 Demo Notes"** button.
3. This generates and indexes a 4-page college document: **`CS101_Data_Structures_Notes.pdf`**.
4. Test with the quick question buttons:
   - *"What is the penalty for late assignments?"* ➔ Cites **Page 4** (Grading & Policies).
   - *"How do arrays differ from linked lists in memory?"* ➔ Cites **Page 2** (Linear Structures).
   - *"What are AVL trees and their worst case search complexity?"* ➔ Cites **Page 3** (Trees & Graphs).

---

## 🔑 LLM Provider Options

| Provider | Cost / Setup | Recommended Model | Best For |
|---|---|---|---|
| **⚡ Groq** | Free API key at [console.groq.com](https://console.groq.com/keys) | `llama-3.3-70b-versatile` | Ultra-fast responses (< 1s) |
| **🌟 Google Gemini** | Free tier at [aistudio.google.com](https://aistudio.google.com/app/apikey) | `gemini-2.5-flash` | High accuracy, long context |
| **🦙 Ollama** | Free, runs locally via [ollama.ai](https://ollama.ai) | `llama3.2` | 100% offline & private |
| **🤖 OpenAI** | Paid API key at [platform.openai.com](https://platform.openai.com) | `gpt-4o-mini` | Standard reliable results |
| **🧠 Claude** | Paid API key at [console.anthropic.com](https://console.anthropic.com) | `claude-3-5-haiku-20241022` | Exceptional synthesis quality |
| **🔒 Offline Extractive** | No key, no setup needed | Local synthesizer | Direct chunk extraction |

---

## 🧪 Running the Automated Verification Test

You can verify the entire ingestion, embedding, FAISS indexing, and retrieval pipeline from the command line anytime:

```bash
python test_pipeline.py
```
