from backend.schemas.api import ChatRequest, ChatResponse, SourceReference


def main() -> None:
    request = ChatRequest(
        message="My university WiFi is not working.",
        session_id="test-session-001",
    )

    print("REQUEST")
    print(request.model_dump())

    source = SourceReference(
        chunk_id="it_001",
        domain="IT",
        source_title="University WiFi Guide",
        page=1,
        score=0.6231,
        source_url="https://example.edu/wifi",
    )

    response = ChatResponse(
        answer=(
            "Please follow the university WiFi connection "
            "procedure."
        ),
        domains=["IT"],
        confidence=0.95,
        needs_clarification=False,
        clarification_question=None,
        is_out_of_scope=False,
        sources=[source],
        session_id=request.session_id,
    )

    print("\nRESPONSE")
    print(response.model_dump())

    print("\nSchema test: PASSED")


if __name__ == "__main__":
    main()