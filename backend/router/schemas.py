from typing import Literal

from pydantic import BaseModel, Field


Domain = Literal[
    "IT",
    "HR_ADMIN",
    "FEES",
    "FACILITIES",
]

RoutingMethod = Literal[
    "llm",
    "keyword_fallback",
]


class LLMRouteResult(BaseModel):
    """
    Structure Gemini must return.

    This model contains only the information
    that the LLM is responsible for deciding.
    """

    domains: list[Domain] = Field(
        default_factory=list
    )

    primary_domain: Domain | None = None

    intent: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    needs_clarification: bool = False

    clarification_question: str | None = None

    is_out_of_scope: bool = False

    reason: str | None = None


class RouteResult(LLMRouteResult):
    """
    Final routing result used by the rest of the backend.

    routing_method is added by our Python code,
    not by Gemini.
    """

    routing_method: RoutingMethod