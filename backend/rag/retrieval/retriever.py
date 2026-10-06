from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .embeddings import EmbeddingService
from .vector_store import FaissVectorStore


SUPPORTED_DOMAINS = (
    "IT",
    "HR_ADMIN",
    "FEES",
    "FACILITIES",
)

DEFAULT_MIN_SCORE = 0.25


class DomainRetriever:
    """
    Domain-aware retrieval layer for CampusOne AI.

    Maintains one FAISS vector store for each supported domain.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
    ) -> None:
        self.embedding_service = embedding_service

        self.stores: dict[str, FaissVectorStore] = {
            domain: FaissVectorStore(
                dimension=embedding_service.dimension
            )
            for domain in SUPPORTED_DOMAINS
        }

    def build(
        self,
        documents: Sequence[dict[str, Any]],
    ) -> None:
        """
        Build domain-specific vector stores.

        Each document must contain:
            - domain
            - text

        Other metadata fields are preserved.
        """

        grouped_documents: dict[
            str,
            list[dict[str, Any]]
        ] = {
            domain: []
            for domain in SUPPORTED_DOMAINS
        }

        for document in documents:
            domain = document.get("domain")
            text = document.get("text")

            if domain not in SUPPORTED_DOMAINS:
                raise ValueError(
                    f"Unsupported domain: {domain}"
                )

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Every document must contain non-empty text."
                )

            grouped_documents[domain].append(
                dict(document)
            )

        for domain, domain_documents in grouped_documents.items():
            if not domain_documents:
                continue

            texts = [
                document["text"]
                for document in domain_documents
            ]

            embeddings = self.embedding_service.embed_documents(
                texts
            )

            self.stores[domain].add(
                embeddings=embeddings,
                metadata=domain_documents,
            )

    def retrieve(
        self,
        query: str,
        domains: Sequence[str],
        top_k: int = 5,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant chunks from the requested domains.

        Results below min_score are discarded.

        Multiple domains are supported for multi-topic questions.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if not 0.0 <= min_score <= 1.0:
            raise ValueError(
                "min_score must be between 0 and 1."
            )

        if not domains:
            return []

        invalid_domains = [
            domain
            for domain in domains
            if domain not in SUPPORTED_DOMAINS
        ]

        if invalid_domains:
            raise ValueError(
                f"Unsupported domains: {invalid_domains}"
            )

        query_embedding = self.embedding_service.embed_query(
            query
        )

        results: list[dict[str, Any]] = []

        for domain in domains:
            store = self.stores[domain]

            domain_results = store.search(
                query_embedding=query_embedding,
                top_k=top_k,
            )

            for result in domain_results:
                if result["score"] < min_score:
                    continue

                result["retrieved_domain"] = domain
                results.append(result)

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]

    def size(
        self,
        domain: str,
    ) -> int:
        """Return the number of chunks stored for a domain."""

        if domain not in SUPPORTED_DOMAINS:
            raise ValueError(
                f"Unsupported domain: {domain}"
            )

        return self.stores[domain].size

    def save(
        self,
        directory: str | Path,
    ) -> None:
        """
        Save all domain-specific FAISS stores to disk.
        """

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for domain, store in self.stores.items():
            domain_directory = directory / domain
            store.save(domain_directory)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        embedding_service: EmbeddingService,
    ) -> "DomainRetriever":
        """
        Load previously saved domain-specific FAISS stores.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Retriever directory not found: {directory}"
            )

        retriever = cls(
            embedding_service=embedding_service
        )

        for domain in SUPPORTED_DOMAINS:
            domain_directory = directory / domain

            index_path = domain_directory / "index.faiss"
            metadata_path = domain_directory / "metadata.json"

            if not index_path.exists() or not metadata_path.exists():
                continue

            retriever.stores[domain] = (
                FaissVectorStore.load(
                    domain_directory
                )
            )

        return retriever


__all__ = [
    "DomainRetriever",
    "SUPPORTED_DOMAINS",
    "DEFAULT_MIN_SCORE",
]