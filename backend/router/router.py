import os

from dotenv import load_dotenv
from groq import Groq

from .fallback import apply_routing_policy, keyword_route
from .prompts import build_router_prompt
from .schemas import LLMRouteResult, RouteResult


load_dotenv()


# Groq structured-output schema
# We define this manually because Groq strict JSON schema
# has stricter requirements than a raw Pydantic-generated schema.
ROUTE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "domains": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "IT",
                    "HR_ADMIN",
                    "FEES",
                    "FACILITIES",
                ],
            },
        },
        "primary_domain": {
            "type": ["string", "null"],
            "enum": [
                "IT",
                "HR_ADMIN",
                "FEES",
                "FACILITIES",
                None,
            ],
        },
        "intent": {
            "type": "string",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "needs_clarification": {
            "type": "boolean",
        },
        "clarification_question": {
            "type": ["string", "null"],
        },
        "is_out_of_scope": {
            "type": "boolean",
        },
        "reason": {
            "type": ["string", "null"],
        },
    },
    "required": [
        "domains",
        "primary_domain",
        "intent",
        "confidence",
        "needs_clarification",
        "clarification_question",
        "is_out_of_scope",
        "reason",
    ],
    "additionalProperties": False,
}


class IntentRouter:
    """
    CampusOne AI intent/domain router.

    Primary routing:
        Groq cloud LLM

    Fallback routing:
        Local deterministic keyword router

    Supported domains:
        IT
        HR_ADMIN
        FEES
        FACILITIES
    """

    def __init__(self, model: str | None = None) -> None:
        # Read Groq API key from .env
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add your Groq API key to the .env file."
            )

        # Use supplied model or .env model
        self.model = model or os.getenv(
            "GROQ_ROUTER_MODEL",
            "openai/gpt-oss-20b",
        )

        # Create Groq client
        self.client = Groq(api_key=api_key)

        # Build routing system prompt
        self.system_prompt = build_router_prompt()

    def route(self, question: str) -> RouteResult:
        """
        Route a student's question to the appropriate domain.

        Groq is used first.
        If Groq fails, the local keyword fallback router is used.
        """

        question = question.strip()

        # Handle empty input before calling Groq
        if not question:
            return RouteResult(
                domains=[],
                primary_domain=None,
                intent="empty_query",
                confidence=0.0,
                needs_clarification=True,
                clarification_question="What would you like help with?",
                is_out_of_scope=False,
                routing_method="llm",
                reason="The user submitted an empty message.",
            )

        try:
            # Call Groq Chat Completions API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "campusone_route",
                        "strict": True,
                        "schema": ROUTE_JSON_SCHEMA,
                    },
                },
            )

            # Extract model response
            content = response.choices[0].message.content

            if not content:
                raise RuntimeError(
                    "Groq returned an empty response."
                )

            # Validate Groq JSON against our Pydantic schema
            llm_result = LLMRouteResult.model_validate_json(
                content
            )

            # Convert LLM result into final router result
            result = RouteResult(
                **llm_result.model_dump(),
                routing_method="llm",
            )

            # Apply CampusOne routing policy
            return apply_routing_policy(result)

        except Exception as exc:
            # Groq failure → deterministic local fallback
            fallback_result = keyword_route(question)

            fallback_result.reason = (
                "Groq routing was unavailable, "
                "so the local fallback router was used. "
                f"Error type: {type(exc).__name__}."
            )

            return fallback_result