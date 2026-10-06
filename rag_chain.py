"""
RAG Chain engine for 'Ask My Notes'.
Connects vector retrieval with strict grounded prompt generation and citation extraction.
"""

from typing import List, Dict, Any, Generator, Tuple
import re
from embeddings import EmbeddingManager
from vector_store import FAISSVectorStore
from llm_client import LLMClient

SYSTEM_PROMPT = """You are "Ask My Notes", a grounded college study assistant.
Your goal is to answer the student's question clearly, thoroughly, and accurately using ONLY the provided document excerpts.

CRITICAL RULES:
1. Rely EXCLUSIVELY on the provided notes context. Do NOT invent, assume, or pull outside facts.
2. In your answer, cite the specific page numbers whenever you state a fact, like [Page X] or [Document Name, Page X].
3. If the provided excerpts do NOT contain enough information to answer the question, explicitly state:
   "Based on the uploaded notes, I cannot find information regarding this question."
   Then briefly mention what topics were found in the closest matching sections.
4. Format your response cleanly using Markdown headings, bullet points, and bold text for readability.
"""


class AskMyNotesRAG:
    """
    Orchestrates the full RAG pipeline:
    User Question -> Embedding -> FAISS Vector Search -> Context Prompt -> LLM Completion -> Source Citations
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_manager: EmbeddingManager,
        llm_client: LLMClient,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client

    def retrieve(self, query: str, top_k: int = 4, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        Embeds the query and retrieves top_k matching chunks.
        """
        if self.vector_store.total_chunks == 0:
            return []

        query_vec = self.embedding_manager.embed_query(query)
        chunks = self.vector_store.search(
            query_embedding=query_vec,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        return chunks

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks into clear, distinct labeled passages for the LLM.
        """
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            doc = chunk.get("document_name", "Unknown")
            page = chunk.get("page_number", "Unknown")
            score = chunk.get("similarity_percentage", "N/A")
            text = chunk.get("text", "").strip()

            block = (
                f"[Source {i}: {doc} | Page {page} | Relevance {score}]\n"
                f"{text}\n"
            )
            context_blocks.append(block)

        return "\n----------------------------------------\n".join(context_blocks)

    def build_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Combines retrieved context and user query into the final grounded prompt.
        """
        context_text = self.format_context(chunks)

        prompt = f"""=== RETRIEVED NOTES CONTEXT ===
{context_text}
=== END CONTEXT ===

STUDENT'S QUESTION:
{query}

Please answer the question using the context above. Include inline citations such as [Page X]."""
        return prompt

    def query(
        self,
        query: str,
        top_k: int = 4,
        min_similarity: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Runs the RAG pipeline synchronously.
        """
        chunks = self.retrieve(query, top_k=top_k, min_similarity=min_similarity)

        if not chunks:
            return {
                "answer": "⚠️ No relevant content found in your notes, or no documents have been uploaded yet. Please upload your PDFs or load sample notes first.",
                "sources": [],
                "prompt": "",
            }

        prompt = self.build_prompt(query, chunks)
        answer = self.llm_client.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)

        return {
            "answer": answer,
            "sources": chunks,
            "prompt": prompt,
        }

    def query_stream(
        self,
        query: str,
        top_k: int = 4,
        min_similarity: float = 0.1,
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """
        Runs the RAG pipeline and returns a stream generator along with retrieved sources.
        """
        chunks = self.retrieve(query, top_k=top_k, min_similarity=min_similarity)

        if not chunks:
            def empty_stream():
                yield "⚠️ No relevant content found in your notes, or no documents have been uploaded yet. Please upload your PDFs or load sample notes first."

            return empty_stream(), []

        prompt = self.build_prompt(query, chunks)
        stream = self.llm_client.stream_generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        return stream, chunks
