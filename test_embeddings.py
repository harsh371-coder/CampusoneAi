from backend.rag.retrieval.embeddings import EmbeddingService


def main() -> None:
    service = EmbeddingService()

    texts = [
        "Students can connect to the university WiFi using their credentials.",
        "The semester fee payment deadline is published by the university.",
        "Students can request academic certificates through the administration.",
    ]

    embeddings = service.embed_documents(texts)

    print("Model:", service.model_name)
    print("Embedding dimension:", service.dimension)
    print("Number of vectors:", len(embeddings))
    print("Embedding shape:", embeddings.shape)

    query = "How do I connect to campus WiFi?"
    query_embedding = service.embed_query(query)

    print("Query vector shape:", query_embedding.shape)


if __name__ == "__main__":
    main()