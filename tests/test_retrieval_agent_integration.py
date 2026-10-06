"""
Integration test for Member 3 retrieval + Member 4 agent.

This test uses:
    - Member 3's real EmbeddingService
    - Member 3's real DomainRetriever
    - Member 3's real FAISS vector store
    - Member 4's real GroqDomainAgent

The FAISS store is built in memory, so no persisted
vector-store files are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from backend.agents import GroqDomainAgent
from backend.rag.retrieval.embeddings import EmbeddingService
from backend.rag.retrieval.retriever import DomainRetriever


def test_retriever_output_can_feed_agent():
    """
    Verify that retrieval results from Member 3 can be
    consumed by Member 4's domain agent.
    """

    # ---------------------------------------------------------
    # 1. Create Member 3's embedding service
    # ---------------------------------------------------------

    embedding_service = EmbeddingService()

    # ---------------------------------------------------------
    # 2. Create Member 3's real domain-aware retriever
    # ---------------------------------------------------------

    retriever = DomainRetriever(
        embedding_service=embedding_service
    )

    # ---------------------------------------------------------
    # 3. Add sample Bennett knowledge
    #
    # This is only test data. Later it will be replaced by
    # the real processed Bennett documents.
    # ---------------------------------------------------------

    documents = [
        {
            "domain": "IT",
            "text": (
                "Students experiencing Wi-Fi connectivity "
                "problems should contact the university IT "
                "helpdesk for technical assistance."
            ),
            "source_title": "IT Support Guide",
            "source_url": "https://www.bennett.edu.in/",
            "page": 2,
            "section": "Wi-Fi Support",
        },
        {
            "domain": "FEES",
            "text": (
                "Students should refer to the official fee "
                "documentation for current fee information "
                "and payment procedures."
            ),
            "source_title": "Fee Guidelines",
            "source_url": "https://www.bennett.edu.in/",
            "page": 5,
            "section": "Fee Information",
        },
    ]

    retriever.build(
        documents
    )

    # ---------------------------------------------------------
    # 4. Perform REAL retrieval
    # ---------------------------------------------------------

    retrieved = retriever.retrieve(
        query="My university Wi-Fi is not working",
        domains=["IT"],
        top_k=3,
        min_score=0.0,
    )

    # Retrieval should return at least one result.
    assert retrieved

    # Verify Member 3's actual metadata is present.
    first_result = retrieved[0]

    assert "text" in first_result
    assert "source_title" in first_result
    assert "source_url" in first_result
    assert "page" in first_result
    assert "section" in first_result
    assert "score" in first_result
    assert "retrieved_domain" in first_result

    assert first_result["retrieved_domain"] == "IT"

    # ---------------------------------------------------------
    # 5. Create YOUR real Groq agent
    #
    # We replace the network client with a mock for this test,
    # so the test itself does not consume Groq API credits.
    # ---------------------------------------------------------

    agent = GroqDomainAgent(
        domain="IT",
        api_key="test-api-key",
        model="openai/gpt-oss-20b",
    )

    # ---------------------------------------------------------
    # 6. Mock Groq's response
    # ---------------------------------------------------------

    agent.client = MagicMock()

    agent.client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=(
                        '{'
                        '"answer": '
                        '"Students should contact the university IT '
                        'helpdesk for Wi-Fi assistance.",'
                        '"sources": ['
                        '{'
                        '"source_title": "IT Support Guide",'
                        '"page": 2,'
                        '"section": "Wi-Fi Support",'
                        '"source_url": '
                        '"https://www.bennett.edu.in/"'
                        '}'
                        '],'
                        '"confidence": 0.95'
                        '}'
                    )
                )
            )
        ]
    )

    # ---------------------------------------------------------
    # 7. Pass Member 3's REAL retrieval output directly
    #    into YOUR agent.
    # ---------------------------------------------------------

    result = agent.generate_answer(
        question="My university Wi-Fi is not working",
        retrieved_knowledge=retrieved,
    )

    # ---------------------------------------------------------
    # 8. Verify YOUR agent consumed the retrieval correctly.
    # ---------------------------------------------------------

    assert result.answer == (
        "Students should contact the university IT "
        "helpdesk for Wi-Fi assistance."
    )

    assert result.confidence == 0.95

    assert len(result.sources) == 1

    assert (
        result.sources[0]["source_title"]
        == "IT Support Guide"
    )

    # ---------------------------------------------------------
    # 9. Verify Groq was actually called.
    # ---------------------------------------------------------

    agent.client.chat.completions.create.assert_called_once()