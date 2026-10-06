from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from backend.router.schemas import RouteResult

from .embeddings import EmbeddingService
from .retriever import (
    DEFAULT_MIN_SCORE,
    DomainRetriever,
)


DEFAULT_VECTOR_STORE_PATH = (
    "data/processed/vector_stores"
)

DEFAULT_TOP_K = 5


class RetrievalService:
    """
    Public retrieval interface for CampusOne AI.
    """

    def __init__(
        self,
        vector_store_path: str | Path = DEFAULT_VECTOR_STORE_PATH,
        embedding_model: str | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:

        if embedding_model:
            self.embedding_service = EmbeddingService(
                model_name=embedding_model
            )
        else:
            self.embedding_service = EmbeddingService()

        self.vector_store_path = Path(
            vector_store_path
        )

        self.min_score = min_score

        self.retriever: DomainRetriever | None = None

    def load(self) -> None:
        """Load the persisted vector stores."""

        self.retriever = DomainRetriever.load(
            directory=self.vector_store_path,
            embedding_service=self.embedding_service,
        )

    def retrieve(
        self,
        query: str,
        domains: Sequence[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks."""

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if self.retriever is None:
            raise RuntimeError(
                "RetrievalService is not loaded. "
                "Call load() before retrieve()."
            )

        return self.retriever.retrieve(
            query=query,
            domains=domains,
            top_k=top_k,
            min_score=self.min_score,
        )

    def retrieve_for_route(
        self,
        query: str,
        route_result: RouteResult,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Retrieve documents using Member 1's RouteResult.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if route_result.is_out_of_scope:
            return []

        if route_result.needs_clarification:
            return []

        if not route_result.domains:
            return []

        return self.retrieve(
            query=query,
            domains=route_result.domains,
            top_k=top_k,
        )

    def domain_sizes(self) -> dict[str, int]:
        """Return indexed chunk counts."""

        if self.retriever is None:
            raise RuntimeError(
                "RetrievalService is not loaded."
            )

        return {
            domain: self.retriever.size(domain)
            for domain in (
                "IT",
                "HR_ADMIN",
                "FEES",
                "FACILITIES",
            )
        }


__all__ = [
    "RetrievalService",
    "DEFAULT_VECTOR_STORE_PATH",
    "DEFAULT_TOP_K",
]