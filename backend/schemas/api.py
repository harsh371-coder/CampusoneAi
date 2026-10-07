from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.router.schemas import Domain


class ChatRequest(BaseModel):
    """
    Request sent by the frontend to the CampusOne AI API.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The student's question or request.",
        examples=[
            "My university WiFi is not working."
        ],
    )

    session_id: str | None = Field(
        default=None,
        description=(
            "Optional conversation/session identifier "
            "for future multi-turn conversations."
        ),
        examples=["student-session-001"],
    )


class SourceReference(BaseModel):
    """
    Citation/source information returned with an answer.
    """

    chunk_id: str = Field(
        ...,
        description="Unique identifier of the retrieved chunk.",
    )

    domain: Domain = Field(
        ...,
        description="CampusOne AI knowledge domain.",
    )

    source_title: str = Field(
        ...,
        description="Title of the source document.",
    )

    page: int | None = Field(
        default=None,
        description="Page number when available.",
    )

    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Retriever similarity score.",
    )

    source_url: str | None = Field(
        default=None,
        description="Official source URL when available.",
    )


class ChatResponse(BaseModel):
    """
    Response returned by the CampusOne AI API.
    """

    answer: str = Field(
        ...,
        description="Final grounded answer shown to the student.",
    )

    domains: list[Domain] = Field(
        default_factory=list,
        description="Domains involved in the request.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Routing confidence.",
    )

    needs_clarification: bool = Field(
        default=False,
        description="Whether the student needs to clarify the request.",
    )

    clarification_question: str | None = Field(
        default=None,
        description="Question to ask when clarification is required.",
    )

    is_out_of_scope: bool = Field(
        default=False,
        description="Whether the request is outside CampusOne AI's scope.",
    )

    sources: list[SourceReference] = Field(
        default_factory=list,
        description="Sources supporting the answer.",
    )

    session_id: str | None = Field(
        default=None,
        description="Conversation/session identifier.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional non-essential API metadata.",
    )