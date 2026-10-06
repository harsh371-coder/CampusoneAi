from backend.rag.retrieval.service import RetrievalService


def main() -> None:
    service = RetrievalService(
        vector_store_path="data/processed/vector_stores",
        min_score=0.25,
    )

    service.load()

    query = "My university WiFi is not working."

    results = service.retrieve(
        query=query,
        domains=["IT"],
        top_k=5,
    )

    print("QUERY:")
    print(query)

    print("\nRESULTS AFTER SCORE FILTER:")

    for result in results:
        print(
            f"- {result['chunk_id']} | "
            f"{result['retrieved_domain']} | "
            f"score={result['score']:.4f}"
        )

    print(
        f"\nMinimum accepted score: "
        f"{service.min_score}"
    )

    # Every returned result must meet the threshold.
    assert all(
        result["score"] >= service.min_score
        for result in results
    )

    print("\nQuality check: PASSED")


if __name__ == "__main__":
    main()