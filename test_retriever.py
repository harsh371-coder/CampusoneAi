from backend.rag.retrieval.embeddings import EmbeddingService
from backend.rag.retrieval.retriever import DomainRetriever


def main() -> None:
    embedding_service = EmbeddingService()

    documents = [
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

    retriever = DomainRetriever(
        embedding_service=embedding_service
    )

    retriever.build(documents)

    print("Domain sizes:")

    for domain in (
        "IT",
        "HR_ADMIN",
        "FEES",
        "FACILITIES",
    ):
        print(
            f"  {domain}: {retriever.size(domain)} chunks"
        )

    query = "My university WiFi is not working."

    results = retriever.retrieve(
        query=query,
        domains=["IT"],
        top_k=3,
    )

    print("\nIT-only retrieval")
    print("Query:", query)

    for result in results:
        print(
            f"- {result['chunk_id']} | "
            f"{result['retrieved_domain']} | "
            f"score={result['score']:.4f}"
        )
        print(f"  {result['text']}")

    # Multi-domain retrieval test
    multi_query = (
        "I cannot connect to WiFi and I need to know "
        "when my semester fees are due."
    )

    multi_results = retriever.retrieve(
        query=multi_query,
        domains=["IT", "FEES"],
        top_k=4,
    )

    print("\nMulti-domain retrieval")
    print("Query:", multi_query)

    for result in multi_results:
        print(
            f"- {result['chunk_id']} | "
            f"{result['retrieved_domain']} | "
            f"score={result['score']:.4f}"
        )
        print(f"  {result['text']}")


if __name__ == "__main__":
    main()