"""
Groq-powered domain agent for CampusOne-AI.

Supported domains:
    IT
    HR_ADMIN
    FEES
    FACILITIES

Flow:
    User question
          +
    Retrieved Bennett knowledge
          +
    Conversation history
          ↓
    Groq / GPT-OSS 20B
          ↓
    Answer + sources + confidence
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from .base_agent import AgentResponse, BaseDomainAgent
from .prompts import get_domain_prompt


# Load variables from the project's .env file.
load_dotenv()


class GroqDomainAgent(BaseDomainAgent):
    """
    Domain-specific agent powered by Groq.

    Retrieval is NOT handled here.
    This class receives retrieved knowledge from Member 3's
    retrieval layer and uses it to generate a grounded response.
    """

    SUPPORTED_DOMAINS = {
        "IT",
        "HR_ADMIN",
        "FEES",
        "FACILITIES",
    }

    def __init__(
        self,
        domain: str,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:

        # Validate the domain.
        if domain not in self.SUPPORTED_DOMAINS:
            raise ValueError(
                f"Unsupported domain '{domain}'. "
                f"Supported domains: "
                f"{sorted(self.SUPPORTED_DOMAINS)}"
            )

        # Make sure a prompt exists for the domain.
        get_domain_prompt(domain)

        # Use the team's configured model.
        selected_model = (
            model
            or os.getenv("GROQ_AGENT_MODEL")
            or os.getenv("GROQ_ROUTER_MODEL")
            or "openai/gpt-oss-20b"
        )

        # Keep compatibility with the BaseDomainAgent
        # we created earlier.
        super().__init__(
            llm_client=None,
            model=selected_model,
        )

        self.DOMAIN = domain

        # Prefer explicitly supplied API key.
        # Otherwise load it from .env.
        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file."
            )

        self.client = Groq(
            api_key=self.api_key
        )

    def build_system_prompt(self) -> str:
        """Return the domain-specific system prompt."""

        return get_domain_prompt(
            self.DOMAIN
        )

    def build_user_prompt(
        self,
        question: str,
        retrieved_knowledge: list[dict[str, Any]],
        conversation_context: list[dict[str, str]]
        | None = None,
    ) -> str:
        """
        Build the user prompt containing:
        - question
        - retrieved knowledge
        - conversation history
        """

        knowledge_text = (
            self._format_retrieved_knowledge(
                retrieved_knowledge
            )
        )

        conversation_text = (
            self._format_conversation(
                conversation_context
            )
        )

        return f"""
User question:
{question}

Retrieved Bennett University knowledge:
{knowledge_text}

Conversation context:
{conversation_text}

Answer the user's question using the retrieved Bennett
University knowledge above.

IMPORTANT RULES:

1. The retrieved Bennett knowledge is the primary source
   of truth.

2. Do NOT invent Bennett University policies, procedures,
   fees, deadlines, contacts, facilities, or other facts.

3. Do NOT rely on unsupported general knowledge when the
   retrieved information is insufficient.

4. If the retrieved information does not adequately answer
   the question, clearly say that there is not enough
   available information to answer reliably.

5. Only provide sources that are actually present in the
   retrieved knowledge.

6. Confidence must represent how strongly the retrieved
   evidence supports the answer.

Confidence guidelines:
- 0.90 to 1.00 = directly supported by retrieved evidence
- 0.70 to 0.89 = mostly supported but somewhat incomplete
- 0.40 to 0.69 = partially supported
- 0.00 to 0.39 = insufficient evidence
""".strip()

    def generate_answer(
        self,
        question: str,
        retrieved_knowledge: list[dict[str, Any]],
        conversation_context: list[dict[str, str]]
        | None = None,
    ) -> AgentResponse:
        """
        Generate a grounded answer using Groq.
        """

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # Don't call the LLM if retrieval found nothing.
        if not retrieved_knowledge:
            return AgentResponse(
                answer=(
                    "I don't have enough retrieved "
                    "Bennett University information "
                    "to answer that reliably."
                ),
                sources=[],
                confidence=0.0,
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        self.build_system_prompt()
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        self.build_user_prompt(
                            question=question,
                            retrieved_knowledge=(
                                retrieved_knowledge
                            ),
                            conversation_context=(
                                conversation_context
                            ),
                        )
                    ),
                },
            ],
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "campusone_agent_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string"
                            },
                            "sources": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "document_name": {
                                            "type": "string"
                                        },
                                        "page": {
                                            "type": [
                                                "integer",
                                                "null",
                                            ]
                                        },
                                        "section": {
                                            "type": [
                                                "string",
                                                "null",
                                            ]
                                        },
                                        "source_url": {
                                            "type": [
                                                "string",
                                                "null",
                                            ]
                                        },
                                    },
                                    "required": [
                                        "document_name",
                                        "page",
                                        "section",
                                        "source_url",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "required": [
                            "answer",
                            "sources",
                            "confidence",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )

        content = (
            response.choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Groq returned invalid JSON: {content}"
            ) from exc

        return AgentResponse(
            answer=str(
                result.get("answer", "")
            ).strip(),
            sources=result.get(
                "sources",
                [],
            ),
            confidence=self._normalize_confidence(
                result.get("confidence")
            ),
        )

    @staticmethod
    def _format_retrieved_knowledge(
        retrieved_knowledge: list[dict[str, Any]],
    ) -> str:
        """
        Convert retrieval results into readable LLM context.
        """

        chunks: list[str] = []

        for index, chunk in enumerate(
            retrieved_knowledge,
            start=1,
        ):
            chunks.append(
                f"""
--- Retrieved Chunk {index} ---

Document:
{chunk.get("document_name", "Unknown document")}

Page:
{chunk.get("page")}

Section:
{chunk.get("section")}

Source URL:
{chunk.get("source_url")}

Content:
{chunk.get("chunk_text", "")}
""".strip()
            )

        return "\n\n".join(chunks)

    @staticmethod
    def _format_conversation(
        conversation_context: list[dict[str, str]]
        | None,
    ) -> str:
        """Format previous conversation messages."""

        if not conversation_context:
            return "No previous conversation."

        messages: list[str] = []

        for message in conversation_context:
            role = message.get(
                "role",
                "unknown",
            )

            content = message.get(
                "content",
                "",
            )

            messages.append(
                f"{role.upper()}: {content}"
            )

        return "\n".join(messages)

    @staticmethod
    def _normalize_confidence(
        value: Any,
    ) -> float | None:
        """Keep confidence safely between 0 and 1."""

        if value is None:
            return None

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None

        return max(
            0.0,
            min(1.0, confidence),
        )