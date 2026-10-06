from backend.rag.retrieval.embeddings import EmbeddingService
from backend.rag.retrieval.retriever import DomainRetriever


VECTOR_STORE_PATH = "data/processed/vector_stores"


def build_test_documents() -> list[dict]:
    return [
        {
            "chunk_id": "it_001",
            "domain": "IT",
            "source_title": "University WiFi Guide",
            "page": 1,
            "text": (
                "Students can connect to the university WiFi "
                "using their university credentials."
            ),
        },
        {
            "chunk_id": "it_002",
            "domain": "IT",
            "source_title": "University Email Guide",
            "page": 2,
            "text": (
                "Students can access university email using "
                "their official university account."
            ),
        },
        {
            "chunk_id": "fees_001",
            "domain": "FEES",
            "source_title": "Fee Payment Policy",
            "page": 1,
            "text": (
                "Semester fees must be paid before the fee "
                "payment deadline published by the university."
            ),
        },
        {
            "chunk_id": "admin_001",
            "domain": "HR_ADMIN",
            "source_title": "Certificate Procedure",
            "page": 1,
            "text": (
                "Students can request academic certificates "
                "through university administration."
            ),
        },
        {
            "chunk_id": "facility_001",
            "domain": "FACILITIES",
            "source_title": "Hostel Information",
            "page": 1,
            "text": (
                "Students living in university hostels can "
                "contact the hostel office for assistance."
            ),
        },
    ]


def main() -> None:
    embedding_service = EmbeddingService()

    # ---------------------------------------------------------
    # 1. Build the retriever
    # ---------------------------------------------------------

    retriever = DomainRetriever(
        embedding_service=embedding_service
    )

    documents = build_test_documents()

    retriever.build(documents)

    print("Original retriever:")
    for domain in (
        "IT",
        "HR_ADMIN",
        "FEES",
        "FACILITIES",
    ):
        print(
            f"  {domain}: {retriever.size(domain)} chunks"
        )

    # ---------------------------------------------------------
    # 2. Save all domain stores
    # ---------------------------------------------------------

    retriever.save(VECTOR_STORE_PATH)

    print(
        f"\nRetriever saved to: {VECTOR_STORE_PATH}"
    )

    # ---------------------------------------------------------
    # 3. Create a NEW retriever by loading from disk
    # ---------------------------------------------------------

    loaded_retriever = DomainRetriever.load(
        VECTOR_STORE_PATH,
        embedding_service,
    )

    print("\nLoaded retriever:")

    for domain in (
        "IT",
        "HR_ADMIN",
        "FEES",
        "FACILITIES",
    ):
        print(
            f"  {domain}: "
            f"{loaded_retriever.size(domain)} chunks"
        )

    # ---------------------------------------------------------
    # 4. Test retrieval from loaded indexes
    # ---------------------------------------------------------

    query = "My campus WiFi is not working."

    results = loaded_retriever.retrieve(
        query=query,
        domains=["IT"],
        top_k=2,
    )

    print("\nRetrieval after loading from disk:")
    print("Query:", query)

    for result in results:
        print(
            f"- {result['chunk_id']} | "
            f"{result['retrieved_domain']} | "
            f"score={result['score']:.4f}"
        )

        print(
            f"  Source: {result['source_title']}"
        )

        print(
            f"  {result['text']}"
        )


if __name__ == "__main__":
    main()