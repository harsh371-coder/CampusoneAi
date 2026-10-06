from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """
    Creates semantic embeddings for document chunks and user queries.

    This is an embedding model, not a conversational LLM.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the size of each generated embedding vector."""
        dimension = self.model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                "Could not determine embedding dimension."
            )

        return int(dimension)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Convert document chunks into normalized embedding vectors.
        """

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        cleaned_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned_texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            cleaned_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Convert one user query into a normalized embedding vector.
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embedding[0],
            dtype=np.float32,
        )


__all__ = [
    "EmbeddingService",
    "DEFAULT_EMBEDDING_MODEL",
]