"""
Base classes and shared response models for CampusOne domain agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResponse:
    """
    Standard response returned by a CampusOne domain agent.

    Attributes:
        answer: Final grounded response.
        sources: Supporting source metadata.
        confidence: Confidence that the retrieved evidence supports
            the generated answer.
    """

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None


class BaseDomainAgent:
    """
    Base interface for all CampusOne domain agents.

    Supported domains:
        IT
        HR_ADMIN
        FEES
        FACILITIES
    """

    DOMAIN: str = ""

    def __init__(
        self,
        llm_client: Any,
        model: str,
    ) -> None:
        self.client = llm_client
        self.model = model

    def build_system_prompt(self) -> str:
        """Return the system prompt for the domain."""
        raise NotImplementedError

    def build_user_prompt(
        self,
        question: str,
        retrieved_knowledge: list[dict[str, Any]],
        conversation_context: list[dict[str, str]] | None = None,
    ) -> str:
        """Build the prompt sent to the LLM."""
        raise NotImplementedError

    def generate_answer(
        self,
        question: str,
        retrieved_knowledge: list[dict[str, Any]],
        conversation_context: list[dict[str, str]] | None = None,
    ) -> AgentResponse:
        """Generate a grounded answer."""
        raise NotImplementedError