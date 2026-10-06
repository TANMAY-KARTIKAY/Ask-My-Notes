"""
Embeddings module for 'Ask My Notes'.
Generates dense vector embeddings using SentenceTransformers (default: all-MiniLM-L6-v2).
Embeddings are L2-normalized so that inner product equals cosine similarity.
"""

from typing import List, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages loading and running SentenceTransformer embedding models.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384  # Default for all-MiniLM-L6-v2

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            # Determine dimension
            dummy = self._model.encode(["test"], normalize_embeddings=True)
            self._dimension = dummy.shape[1]
        return self._model

    @property
    def dimension(self) -> int:
        if self._model is None:
            # Trigger lazy load
            _ = self.model
        return self._dimension

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeds a list of texts and returns an (N, dimension) float32 numpy array.
        Vectors are L2-normalized so that inner product equals cosine similarity.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string and returns a (1, dimension) float32 numpy array.
        """
        if not query or not query.strip():
            query = " "
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)
