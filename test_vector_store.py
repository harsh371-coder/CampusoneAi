from backend.rag.retrieval.embeddings import EmbeddingService
from backend.rag.retrieval.vector_store import FaissVectorStore


def main() -> None:
    embedding_service = EmbeddingService()

    documents = [
        {
            "chunk_id": "it_001",
            "domain": "IT",
            "source_title": "University WiFi Guide",
            "source_url": "https://example.edu/wifi",
            "page": 1,
            "text": (
                "Students can connect to the university WiFi "
                "using their university credentials."
            ),
        },
        {
            "chunk_id": "fees_001",
            "domain": "FEES",
            "source_title": "Fee Payment Policy",
            "source_url": "https://example.edu/fees",
            "page": 2,
            "text": (
                "Semester fees must be paid before the fee "
                "payment deadline published by the university."
            ),
        },
        {
            "chunk_id": "admin_001",
            "domain": "HR_ADMIN",
            "source_title": "Certificate Procedure",
            "source_url": "https://example.edu/certificates",
            "page": 3,
            "text": (
                "Students can request academic certificates "
                "through the university administration."
            ),
        },
    ]

    texts = [
        document["text"]
        for document in documents
    ]

    embeddings = embedding_service.embed_documents(texts)

    store = FaissVectorStore(
        dimension=embedding_service.dimension
    )

    store.add(
        embeddings=embeddings,
        metadata=documents,
    )

    print("FAISS vectors stored:", store.size)

    query = "My campus WiFi is not connecting."

    query_embedding = embedding_service.embed_query(
        query
    )

    results = store.search(
        query_embedding=query_embedding,
        top_k=2,
    )

    print("\nQuery:", query)
    print("\nTop results:")

    for result in results:
        print(
            f"- {result['chunk_id']} | "
            f"{result['domain']} | "
            f"score={result['score']:.4f}"
        )
        print(f"  {result['text']}")

    # Save the store
    output_directory = (
        "data/processed/vector_store_test"
    )

    store.save(output_directory)

    print(
        f"\nVector store saved to: "
        f"{output_directory}"
    )

    # Test loading it back
    loaded_store = FaissVectorStore.load(
        output_directory
    )

    print(
        "Reloaded vectors:",
        loaded_store.size
    )


if __name__ == "__main__":
    main()