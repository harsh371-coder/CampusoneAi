from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np


class FaissVectorStore:
    """
    FAISS-based vector store for CampusOne AI.

    Stores:
        1. FAISS similarity index
        2. Metadata for every indexed chunk

    The vectors are expected to be L2-normalized embeddings.
    Therefore, inner-product similarity can be used as cosine similarity.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be greater than 0.")

        self.dimension = dimension

        # Inner-product search.
        # With normalized vectors, this is equivalent to cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)

        # Metadata position must always correspond to FAISS vector position.
        self.metadata: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        """Return the number of vectors currently stored."""
        return self.index.ntotal

    def add(
        self,
        embeddings: np.ndarray,
        metadata: Sequence[dict[str, Any]],
    ) -> None:
        """
        Add embeddings and their corresponding metadata.
        """

        vectors = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings with dimension "
                f"{self.dimension}, got {vectors.shape[1]}."
            )

        if len(vectors) != len(metadata):
            raise ValueError(
                "Number of embeddings must match number of metadata records."
            )

        if len(vectors) == 0:
            return

        # FAISS expects float32 vectors.
        self.index.add(vectors)

        self.metadata.extend(
            dict(item)
            for item in metadata
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search the vector store and return the most similar chunks.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if self.size == 0:
            return []

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2 or query.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding must have shape "
                f"(1, {self.dimension}) or ({self.dimension},)."
            )

        # We cannot retrieve more records than are stored.
        k = min(top_k, self.size)

        scores, indices = self.index.search(
            query,
            k,
        )

        results: list[dict[str, Any]] = []

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):
            # FAISS may return -1 for missing results.
            if index_id < 0:
                continue

            record = dict(self.metadata[index_id])

            record["score"] = float(score)
            record["index_id"] = int(index_id)

            results.append(record)

        return results

    def save(
        self,
        directory: str | Path,
    ) -> None:
        """
        Save FAISS index and metadata to disk.
        """

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        faiss.write_index(
            self.index,
            str(index_path),
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "dimension": self.dimension,
                    "metadata": self.metadata,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "FaissVectorStore":
        """
        Load a previously saved FAISS vector store.
        """

        directory = Path(directory)

        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        index = faiss.read_index(
            str(index_path)
        )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        dimension = int(data["dimension"])

        store = cls(dimension)

        store.index = index
        store.metadata = list(
            data["metadata"]
        )

        if store.index.ntotal != len(store.metadata):
            raise ValueError(
                "FAISS index size does not match metadata size."
            )

        return store


__all__ = [
    "FaissVectorStore",
]