"""
Vector Store module for 'Ask My Notes'.
Uses FAISS (IndexFlatIP) for cosine-similarity semantic search over document chunks.
Stores chunk metadata (document name, page number, text) alongside embeddings.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-backed vector store for document chunks.
    Uses Inner Product (IP) metric on L2-normalized embeddings (cosine similarity).
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks_metadata: List[Dict[str, Any]] = []

    @property
    def total_chunks(self) -> int:
        return len(self.chunks_metadata)

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        Adds chunk metadata and their corresponding embeddings to the FAISS index.

        Args:
            chunks: List of chunk metadata dicts
            embeddings: (N, dimension) float32 numpy array
        """
        if len(chunks) == 0:
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match chunk count ({len(chunks)})"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension})"
            )

        # Ensure float32 format
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Add to FAISS index
        self.index.add(vectors)
        self.chunks_metadata.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks. Index now contains {self.total_chunks} chunks.")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Searches the index for the top_k most similar chunks.

        Args:
            query_embedding: (1, dimension) or (dimension,) float32 array
            top_k: Number of nearest neighbors to retrieve
            min_similarity: Minimum cosine similarity score threshold (0.0 to 1.0)

        Returns:
            List of chunk dicts sorted by similarity, with 'similarity_score' included.
        """
        if self.total_chunks == 0:
            return []

        # Ensure 2D float32 shape
        query_vec = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if len(query_vec.shape) == 1:
            query_vec = np.expand_dims(query_vec, axis=0)

        k = min(top_k, self.total_chunks)
        scores, indices = self.index.search(query_vec, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks_metadata):
                continue

            sim_score = float(score)
            if sim_score < min_similarity:
                continue

            chunk_copy = dict(self.chunks_metadata[idx])
            chunk_copy["similarity_score"] = round(sim_score, 4)
            chunk_copy["similarity_percentage"] = f"{max(0.0, sim_score) * 100:.1f}%"
            results.append(chunk_copy)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns summary statistics about indexed documents.
        """
        doc_stats: Dict[str, set] = {}
        for chunk in self.chunks_metadata:
            doc_name = chunk.get("document_name", "Unknown")
            page_num = chunk.get("page_number", 1)
            if doc_name not in doc_stats:
                doc_stats[doc_name] = set()
            doc_stats[doc_name].add(page_num)

        documents_summary = [
            {
                "name": doc_name,
                "pages_indexed": len(pages),
                "page_list": sorted(list(pages)),
            }
            for doc_name, pages in doc_stats.items()
        ]

        return {
            "total_chunks": self.total_chunks,
            "total_documents": len(doc_stats),
            "documents": documents_summary,
        }

    def clear(self) -> None:
        """
        Resets the FAISS index and metadata store.
        """
        self.index.reset()
        self.chunks_metadata = []
        logger.info("Cleared FAISS index and metadata.")

    def save(self, directory: Union[str, Path]) -> None:
        """
        Saves the FAISS index and metadata JSON to disk.
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = save_dir / "faiss_index.bin"
        meta_path = save_dir / "chunks_metadata.json"

        faiss.write_index(self.index, str(faiss_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, ensure_ascii=False, indent=2)

    def load(self, directory: Union[str, Path]) -> bool:
        """
        Loads the FAISS index and metadata from disk if available.
        """
        save_dir = Path(directory)
        faiss_path = save_dir / "faiss_index.bin"
        meta_path = save_dir / "chunks_metadata.json"

        if not faiss_path.exists() or not meta_path.exists():
            return False

        self.index = faiss.read_index(str(faiss_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks_metadata = json.load(f)
        return True
