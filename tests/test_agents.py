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
    Build a fake Groq SDK response matching the structure
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


@pytest.mark.parametrize("domain", SUPPORTED_DOMAINS)
def test_supported_domains(domain):
    """Every project domain should create successfully."""

    agent = GroqDomainAgent(
        domain=domain,
        api_key="test-api-key",
    )

    assert agent.DOMAIN == domain
    assert agent.model == "openai/gpt-oss-20b"


def test_invalid_domain():
    """Unsupported domains should raise an error."""

    with pytest.raises(ValueError):
        GroqDomainAgent(
            domain="UNKNOWN_DOMAIN",
            api_key="test-api-key",
        )


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
                    "chunk_text": "IT support information.",
                    "document_name": "IT Guide",
                    "page": 1,
                    "section": "Support",
                    "source_url": "https://example.com",
                }
            ],
        )


def test_empty_retrieval():
    """
    No retrieved knowledge means we should not call Groq.
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
    assert "don't have enough retrieved" in result.answer

    agent.client.chat.completions.create.assert_not_called()


def test_generate_grounded_answer():
    """A normal retrieved context should produce an AgentResponse."""

    agent = GroqDomainAgent(
        domain="IT",
        api_key="test-api-key",
    )

    fake_response = make_fake_groq_response(
        answer=(
            "Students should contact the university IT helpdesk "
            "for Wi-Fi assistance."
        ),
        confidence=0.95,
        sources=[
            {
                "document_name": "IT Support Guide",
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
        question="What should I do if Wi-Fi is not working?",
        retrieved_knowledge=[
            {
                "chunk_text": (
                    "Students experiencing Wi-Fi connectivity "
                    "problems should contact the university "
                    "IT helpdesk for technical assistance."
                ),
                "document_name": "IT Support Guide",
                "page": 2,
                "section": "Wi-Fi Support",
                "source_url": "https://www.bennett.edu.in/",
            }
        ],
    )

    assert result.answer == (
        "Students should contact the university IT helpdesk "
        "for Wi-Fi assistance."
    )

    assert result.confidence == 0.95

    assert len(result.sources) == 1
    assert result.sources[0]["document_name"] == (
        "IT Support Guide"
    )

    agent.client.chat.completions.create.assert_called_once()


def test_conversation_context_is_included():
    """Conversation history should be included in the user prompt."""

    agent = GroqDomainAgent(
        domain="IT",
        api_key="test-api-key",
    )

    prompt = agent.build_user_prompt(
        question="What should I do next?",
        retrieved_knowledge=[
            {
                "chunk_text": "Contact IT support.",
                "document_name": "IT Guide",
                "page": 1,
                "section": "Support",
                "source_url": None,
            }
        ],
        conversation_context=[
            {
                "role": "user",
                "content": "My Wi-Fi stopped working.",
            },
            {
                "role": "assistant",
                "content": "Please describe the problem.",
            },
        ],
    )

    assert "My Wi-Fi stopped working." in prompt
    assert "Please describe the problem." in prompt


def test_confidence_is_normalized():
    """Confidence should always remain between 0 and 1."""

    agent = GroqDomainAgent(
        domain="IT",
        api_key="test-api-key",
    )

    assert agent._normalize_confidence(0.5) == 0.5
    assert agent._normalize_confidence(-1) == 0.0
    assert agent._normalize_confidence(2) == 1.0
    assert agent._normalize_confidence(None) is None

def test_retrieved_knowledge_is_included_in_prompt():
    """Retrieved Bennett content must reach the LLM prompt."""

    agent = GroqDomainAgent(
        domain="IT",
        api_key="test-api-key",
    )

    retrieved_knowledge = [
        {
            "chunk_text": (
                "Students should contact the IT helpdesk "
                "for Wi-Fi connectivity issues."
            ),
            "document_name": "IT Support Guide",
            "page": 2,
            "section": "Wi-Fi Support",
            "source_url": "https://www.bennett.edu.in/",
        }
    ]

    prompt = agent.build_user_prompt(
        question="My Wi-Fi is not working. What should I do?",
        retrieved_knowledge=retrieved_knowledge,
    )

    assert "IT Support Guide" in prompt
    assert "Wi-Fi Support" in prompt
    assert "Students should contact the IT helpdesk" in prompt
    assert "https://www.bennett.edu.in/" in prompt