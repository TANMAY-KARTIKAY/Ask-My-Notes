"""
End-to-end verification script for Ask My Notes RAG pipeline.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pdf_loader import extract_pages_from_pdf
from text_splitter import chunk_document_pages
from embeddings import EmbeddingManager
from vector_store import FAISSVectorStore
from rag_chain import AskMyNotesRAG
from llm_client import LLMClient

def test_full_pipeline():
    pdf_path = "CS101_Data_Structures_Notes.pdf"
    print(f"Loading {pdf_path}...")
    pages = extract_pages_from_pdf(pdf_path, filename="CS101_Data_Structures_Notes.pdf")
    print(f"Extracted {len(pages)} pages.")
    for p in pages:
        print(f"  Page {p['page_number']}: {p['word_count']} words, {p['char_count']} chars")

    chunks = chunk_document_pages(pages, target_words=400, overlap_words=50)
    print(f"\nGenerated {len(chunks)} chunks.")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1}: Page {c['page_number']}, {c['word_count']} words")

    print("\nInitializing embedding manager...")
    embedder = EmbeddingManager()
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(chunk_texts)
    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    print("\nBuilding FAISS vector index...")
    store = FAISSVectorStore(dimension=embedder.dimension)
    store.add_chunks(chunks, embeddings)
    print(f"Vector store ready: {store.get_stats()}")

    # Test Query 1: Late policy (should match Page 4)
    q1 = "What is the penalty for late assignment submission?"
    print(f"\n--- Testing Query: '{q1}' ---")
    results = store.search(embedder.embed_query(q1), top_k=2)
    for r in results:
        print(f"  Result: Document '{r['document_name']}', Page {r['page_number']}, Score: {r['similarity_percentage']}")
        print(f"  Excerpt: {r['text'][:120]}...\n")
    assert any(r['page_number'] == 4 for r in results), "Failed: expected Page 4 in results"

    # Test Query 2: AVL Tree (should match Page 3)
    q2 = "What are AVL trees and what is their worst case search complexity?"
    print(f"--- Testing Query: '{q2}' ---")
    results2 = store.search(embedder.embed_query(q2), top_k=2)
    for r in results2:
        print(f"  Result: Document '{r['document_name']}', Page {r['page_number']}, Score: {r['similarity_percentage']}")
        print(f"  Excerpt: {r['text'][:120]}...\n")
    assert any(r['page_number'] == 3 for r in results2), "Failed: expected Page 3 in results"

    # Test RAG Chain with Offline Synthesizer
    print("--- Testing RAG Chain Answer Generation (Offline Mode) ---")
    llm = LLMClient(provider="offline")
    rag = AskMyNotesRAG(vector_store=store, embedding_manager=embedder, llm_client=llm)
    ans = rag.query(q1)
    print("Answer generated:")
    print(ans["answer"][:300] + "...")
    print("\nAll pipeline tests PASSED successfully!")

if __name__ == "__main__":
    test_full_pipeline()
