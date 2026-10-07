from backend.rag.retrieval.service import RetrievalService


def main() -> None:
    service = RetrievalService()
    service.load()

    query = (
        "What semester system is used for "
        "B.Tech CSE, BCA and B.Sc programs?"
    )

    results = service.retrieve(
        query=query,
        domains=["HR_ADMIN"],
        top_k=3,
    )

    print("QUERY:")
    print(query)

    print(f"\nRESULTS: {len(results)}")

    for result in results:
        print("\n------------------------------")
        print("Chunk ID:", result["chunk_id"])
        print("Domain:", result["domain"])
        print("Score:", f"{result['score']:.4f}")
        print("Source:", result["source_title"])
        print("Page:", result["page"])
        print("Source URL:", result.get("source_url"))
        print("Text:")
        print(result["text"][:1000])


if __name__ == "__main__":
    main()