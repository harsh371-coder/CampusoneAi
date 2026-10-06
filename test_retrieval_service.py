from backend.rag.retrieval.service import RetrievalService


def main() -> None:
    service = RetrievalService()

    # Load persisted indexes.
    service.load()

    print("Indexed chunks:")

    for domain, size in service.domain_sizes().items():
        print(f"  {domain}: {size}")

    query = "My university WiFi is not working."

    results = service.retrieve(
        query=query,
        domains=["IT"],
        top_k=3,
    )

    print("\nQuery:")
    print(query)

    print("\nRetrieved results:")

    for result in results:
        print(
            f"- Chunk: {result['chunk_id']}"
        )
        print(
            f"  Domain: {result['retrieved_domain']}"
        )
        print(
            f"  Score: {result['score']:.4f}"
        )
        print(
            f"  Source: {result['source_title']}"
        )
        print(
            f"  Page: {result['page']}"
        )
        print(
            f"  Text: {result['text']}"
        )
        print()


if __name__ == "__main__":
    main()