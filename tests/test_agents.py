"""
Tests for CampusOne-AI domain agents.

These tests mock the Groq API, so they do not consume API credits.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.agents import GroqDomainAgent


SUPPORTED_DOMAINS = [
    "IT",
    "HR_ADMIN",
    "FEES",
    "FACILITIES",
]


def make_fake_groq_response(
    answer: str,
    confidence: float,
    sources: list[dict] | None = None,
):
    """
    Create a fake Groq response matching the structure
    expected by GroqDomainAgent.
    """

    payload = {
        "answer": answer,
        "sources": sources or [],
        "confidence": confidence,
    }

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(payload)
                )
            )
        ]
    )


# ============================================================
# DOMAIN TESTS
# ============================================================

@pytest.mark.parametrize(
    "domain",
    SUPPORTED_DOMAINS,
)
def test_supported_domains(domain):
    """All four supported domains should initialize correctly."""

    agent = GroqDomainAgent(
        domain=domain,
        api_key="test-api-key",
        model="openai/gpt-oss-20b",
    )

    assert agent.DOMAIN == domain
    assert agent.model == "openai/gpt-oss-20b"


def test_invalid_domain():
    """Unsupported domains should raise ValueError."""

    with pytest.raises(ValueError):
        GroqDomainAgent(
            domain="UNKNOWN_DOMAIN",
            api_key="test-api-key",
        )


# ============================================================
# INPUT VALIDATION
# ============================================================

def test_empty_question():
    """An empty question should be rejected."""

    agent = GroqDomainAgent(
        domain="IT",
        api_key="test-api-key",
    )

    with pytest.raises(ValueError):
        agent.generate_answer(
            question="",
            retrieved_knowledge=[
                {
                    "text": "IT support information.",
                    "source_title": "IT Guide",
                    "page": 1,
                    "section": "Support",
                    "source_url": "https://example.com",
                    "relevance_score": 0.90,
                    "domain": "IT",
                }
            ],
        )


def test_empty_retrieval():
    """
    If retrieval returns no knowledge, Groq should not be called.
    """

    agent = GroqDomainAgent(
        domain="IT",
        api_key="test-api-key",
    )

    agent.client = MagicMock()

    result = agent.generate_answer(
        question="How do I fix Wi-Fi?",
        retrieved_knowledge=[],
    )

    assert result.confidence == 0.0
    assert result.sources == []

    assert (
        "don't have enough retrieved"
        in result.answer
    )

    agent.client.chat.completions.create.assert_not_called()


# ============================================================
# GROQ RESPONSE TEST
# ============================================================

def test_generate_grounded_answer():
    """
    A normal retrieved context should produce
    an AgentResponse with answer, sources and confidence.
    """

    agent = GroqDomainAgent(
        domain="IT",
        api_key="test-api-key",
        model="openai/gpt-oss-20b",
    )

    fake_response = make_fake_groq_response(
        answer=(
            "Students should contact the university IT "
            "helpdesk for Wi-Fi assistance."
        ),
        confidence=0.95,
        sources=[
            {
                "source_title": "IT Support Guide",
                "page": 2,
                "section": "Wi-Fi Support",
                "source_url": "https://www.bennett.edu.in/",
            }
        ],
    )

    agent.client = MagicMock()

    agent.client.chat.completions.create.return_value = (
        fake_response
    )

    result = agent.generate_answer(
        question=(
            "What should I do if my university Wi-Fi "
            "is not working?"
        ),
        retrieved_knowledge=[
            {
                "text": (
                    "Students experiencing Wi-Fi connectivity "
                    "problems should contact the university "
                    "IT helpdesk for technical assistance."
                ),
                "source_title": "IT Support Guide",
                "page": 2,
                "section": "Wi-Fi Support",
                "source_url": "https://www.bennett.edu.in/",
                "relevance_score": 0.95,
                "domain": "IT",
            }
        ],
    )

    assert result.answer == (
        "Students should contact the university IT "
        "helpdesk for Wi-Fi assistance."
    )

    assert result.confidence == 0.95

    assert len(result.sources) == 1

    assert result.sources[0]["source_title"] == (
        "IT Support Guide"
    )

    assert result.sources[0]["page"] == 2

    assert result.sources[0]["section"] == (
        "Wi-Fi Support"
    )

    assert result.sources[0]["source_url"] == (
        "https://www.bennett.edu.in/"
    )

    agent.client.chat.completions.create.assert_called_once()


# ============================================================
# CONVERSATION TEST
# ============================================================

def test_conversation_context_is_included():
    """
    Conversation history should be included
    in the user prompt.
    """

    agent = GroqDomainAgent(
        domain="IT",
        api_key="test-api-key",
    )

    prompt = agent.build_user_prompt(
        question="What should I do next?",
        retrieved_knowledge=[
            {
                "text": "Contact IT support.",
                "source_title": "IT Guide",
                "page": 1,
                "section": "Support",
                "source_url": None,
                "relevance_score": 0.90,
                "domain": "IT",
            }
        ],
        conversation_context=[
            {
                "role": "user",
                "content": "My Wi-Fi stopped working.",
            },
            {
                "role": "assistant",
                "content": (
                    "Please describe the problem."
                ),
            },
        ],
    )

    assert "My Wi-Fi stopped working." in prompt

    assert (
        "Please describe the problem."
        in prompt
    )


# ============================================================
# RETRIEVAL CONTRACT TEST
# ============================================================

def test_retrieved_knowledge_is_included_in_prompt():
    """
    Every important field from the retriever contract
    should reach the LLM prompt.
    """

    agent = GroqDomainAgent(
        domain="IT",
        api_key="test-api-key",
    )

    retrieved_knowledge = [
        {
            "text": (
                "Students should contact the IT helpdesk "
                "for Wi-Fi connectivity issues."
            ),
            "source_title": "IT Support Guide",
            "page": 2,
            "section": "Wi-Fi Support",
            "source_url": "https://www.bennett.edu.in/",
            "relevance_score": 0.95,
            "domain": "IT",
        }
    ]

    prompt = agent.build_user_prompt(
        question=(
            "My Wi-Fi is not working. "
            "What should I do?"
        ),
        retrieved_knowledge=retrieved_knowledge,
    )

    # Text
    assert (
        "Students should contact the IT helpdesk"
        in prompt
    )

    # Source metadata
    assert "IT Support Guide" in prompt
    assert "Wi-Fi Support" in prompt
    assert "https://www.bennett.edu.in/" in prompt

    # Retrieval metadata
    assert "0.95" in prompt
    assert "IT" in prompt


# ============================================================
# CONFIDENCE TESTS
# ============================================================

def test_confidence_is_normalized():
    """
    Confidence must always remain between 0 and 1.
    """

    agent = GroqDomainAgent(
        domain="IT",
        api_key="test-api-key",
    )

    assert (
        agent._normalize_confidence(0.5)
        == 0.5
    )

    assert (
        agent._normalize_confidence(-1)
        == 0.0
    )

    assert (
        agent._normalize_confidence(2)
        == 1.0
    )

    assert (
        agent._normalize_confidence(None)
        is None
    )


# ============================================================
# SOURCE HANDLING TEST
# ============================================================

def test_multiple_sources_are_preserved():
    """
    Multiple sources returned by Groq should be
    preserved by the agent.
    """

    agent = GroqDomainAgent(
        domain="FEES",
        api_key="test-api-key",
        model="openai/gpt-oss-20b",
    )

    fake_response = make_fake_groq_response(
        answer=(
            "The fee information is available in the "
            "official university documentation."
        ),
        confidence=0.88,
        sources=[
            {
                "source_title": "Fee Structure",
                "page": 5,
                "section": "Tuition Fees",
                "source_url": "https://example.com/fees",
            },
            {
                "source_title": "Payment Guidelines",
                "page": 3,
                "section": "Payment Process",
                "source_url": "https://example.com/payment",
            },
        ],
    )

    agent.client = MagicMock()

    agent.client.chat.completions.create.return_value = (
        fake_response
    )

    result = agent.generate_answer(
        question="Where can I find fee information?",
        retrieved_knowledge=[
            {
                "text": "Fee information is documented here.",
                "source_title": "Fee Structure",
                "page": 5,
                "section": "Tuition Fees",
                "source_url": "https://example.com/fees",
                "relevance_score": 0.91,
                "domain": "FEES",
            }
        ],
    )

    assert len(result.sources) == 2

    assert (
        result.sources[0]["source_title"]
        == "Fee Structure"
    )

    assert (
        result.sources[1]["source_title"]
        == "Payment Guidelines"
    )

    assert result.confidence == 0.88