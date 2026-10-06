from backend.rag.retrieval.service import RetrievalService
from backend.router.schemas import RouteResult


VECTOR_STORE_PATH = "data/processed/vector_stores"


def main() -> None:
    service = RetrievalService(
        vector_store_path=VECTOR_STORE_PATH
    )

    service.load()

    # ---------------------------------------------------------
    # 1. Normal IT query
    # ---------------------------------------------------------

    it_route = RouteResult(
        domains=["IT"],
        primary_domain="IT",
        intent="wifi_issue",
        confidence=0.95,
        needs_clarification=False,
        clarification_question=None,
        is_out_of_scope=False,
        routing_method="llm",
        reason=None,
    )

    results = service.retrieve_for_route(
        query="My university WiFi is not working.",
        route_result=it_route,
        top_k=3,
    )

    print("IT ROUTING TEST")
    print("Results:", len(results))

    for result in results:
        print(
            result["chunk_id"],
            "|",
            result["retrieved_domain"],
            "|",
            f"{result['score']:.4f}",
        )

    # ---------------------------------------------------------
    # 2. Multi-domain query
    # ---------------------------------------------------------

    multi_route = RouteResult(
        domains=["IT", "FEES"],
        primary_domain="IT",
        intent="multiple_topics",
        confidence=0.90,
        needs_clarification=False,
        clarification_question=None,
        is_out_of_scope=False,
        routing_method="llm",
        reason=None,
    )

    results = service.retrieve_for_route(
        query=(
            "My WiFi is not working and "
            "I want to know when fees are due."
        ),
        route_result=multi_route,
        top_k=4,
    )

    print("\nMULTI-DOMAIN TEST")
    print("Results:", len(results))

    for result in results:
        print(
            result["chunk_id"],
            "|",
            result["retrieved_domain"],
            "|",
            f"{result['score']:.4f}",
        )

    # ---------------------------------------------------------
    # 3. Out-of-scope query
    # ---------------------------------------------------------

    out_of_scope_route = RouteResult(
        domains=[],
        primary_domain=None,
        intent="out_of_scope",
        confidence=0.98,
        needs_clarification=False,
        clarification_question=None,
        is_out_of_scope=True,
        routing_method="llm",
        reason="Question is unrelated to supported university services.",
    )

    results = service.retrieve_for_route(
        query="Write a C++ sorting program.",
        route_result=out_of_scope_route,
    )

    print("\nOUT-OF-SCOPE TEST")
    print("Results:", len(results))

    # ---------------------------------------------------------
    # 4. Clarification test
    # ---------------------------------------------------------

    clarification_route = RouteResult(
        domains=[],
        primary_domain=None,
        intent="ambiguous",
        confidence=0.50,
        needs_clarification=True,
        clarification_question="Which payment are you referring to?",
        is_out_of_scope=False,
        routing_method="llm",
        reason="The request does not clearly identify a supported domain.",
    )

    results = service.retrieve_for_route(
        query="My payment is not working.",
        route_result=clarification_route,
    )

    print("\nCLARIFICATION TEST")
    print("Results:", len(results))


if __name__ == "__main__":
    main()