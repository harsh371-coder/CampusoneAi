from backend.rag.retrieval.dataset_loader import (
    load_and_normalize_chunks,
)
from backend.rag.retrieval.embeddings import EmbeddingService
from backend.rag.retrieval.retriever import DomainRetriever


CHUNK_FILE = (
    "data/processed/scset_ug_chunks.jsonl"
)

VECTOR_STORE_PATH = (
    "data/processed/vector_stores"
)


def main() -> None:
    print("Loading Member 2 chunks...")

    documents = load_and_normalize_chunks(
        CHUNK_FILE
    )

    print(
        f"Loaded {len(documents)} normalized chunks."
    )

    print("\nDomains found:")

    domain_counts: dict[str, int] = {}

    for document in documents:
        domain = document["domain"]

        domain_counts[domain] = (
            domain_counts.get(domain, 0) + 1
        )

    for domain, count in domain_counts.items():
        print(
            f"  {domain}: {count}"
        )

    embedding_service = EmbeddingService()

    retriever = DomainRetriever(
        embedding_service=embedding_service
    )

    print("\nBuilding FAISS indexes...")

    retriever.build(documents)

    print("\nFinal index sizes:")

    for domain in (
        "IT",
        "HR_ADMIN",
        "FEES",
        "FACILITIES",
    ):
        print(
            f"  {domain}: "
            f"{retriever.size(domain)} chunks"
        )

    retriever.save(
        VECTOR_STORE_PATH
    )

    print(
        f"\nReal vector stores saved to: "
        f"{VECTOR_STORE_PATH}"
    )


if __name__ == "__main__":
    main()