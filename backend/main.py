from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import RateLimitError

from backend.agents.groq_agent import GroqDomainAgent
from backend.rag.retrieval.service import RetrievalService
from backend.router.router import IntentRouter
from backend.schemas.api import (
    ChatRequest,
    ChatResponse,
    SourceReference,
)


app = FastAPI(
    title="CampusOne AI",
    description=(
        "One Front Door for Everything — "
        "University Student Assistant API"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Core components
# -------------------------------------------------------------------

intent_router = IntentRouter()

from pathlib import Path

from backend.rag.retrieval.dataset_loader import (
    load_and_normalize_chunks,
)
from backend.rag.retrieval.retriever import (
    SUPPORTED_DOMAINS,
    DomainRetriever,
)


CHUNK_FILE = Path(
    "data/processed/scset_ug_chunks.jsonl"
)

VECTOR_STORE_PATH = Path(
    "data/processed/vector_stores"
)


def initialize_retrieval() -> None:
    """
    Load existing FAISS indexes.

    If no indexes exist, build them automatically from
    Member 2's processed JSONL dataset.
    """

    existing_index = any(
        (
            VECTOR_STORE_PATH
            / domain
            / "index.faiss"
        ).exists()
        for domain in SUPPORTED_DOMAINS
    )

    if existing_index:
        retrieval_service.load()
        return

    if not CHUNK_FILE.exists():
        raise RuntimeError(
            "No FAISS indexes were found and the Member 2 "
            f"chunk file is missing: {CHUNK_FILE}"
        )

    documents = load_and_normalize_chunks(
        CHUNK_FILE
    )

    retriever = DomainRetriever(
        embedding_service=(
            retrieval_service.embedding_service
        )
    )

    retriever.build(documents)

    retriever.save(
        VECTOR_STORE_PATH
    )

    retrieval_service.retriever = retriever


retrieval_service = RetrievalService()

initialize_retrieval()


SUPPORTED_DOMAINS = (
    "IT",
    "HR_ADMIN",
    "FEES",
    "FACILITIES",
)


def create_domain_agents() -> dict[str, GroqDomainAgent]:
    """
    Create one Groq agent for each CampusOne AI domain.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set."
        )

    model = os.getenv(
        "GROQ_ROUTER_MODEL",
        "openai/gpt-oss-20b",
    )

    return {
        domain: GroqDomainAgent(
            domain=domain,
            api_key=api_key,
            model=model,
        )
        for domain in SUPPORTED_DOMAINS
    }


domain_agents = create_domain_agents()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def normalize_sources(
    agent_sources: list[dict[str, Any]],
    retrieved_knowledge: list[dict[str, Any]],
    domain: str,
) -> list[SourceReference]:
    """
    Convert Member 4's agent sources into the common API format.

    Only sources that can be matched to actual retrieved knowledge
    are returned.
    """

    normalized: list[SourceReference] = []

    for source in agent_sources:
        document_name = source.get(
            "document_name",
            "",
        )

        page = source.get("page")

        section = source.get(
            "section"
        )

        source_url = source.get(
            "source_url"
        )

        for retrieved in retrieved_knowledge:
            retrieved_title = retrieved.get(
                "source_title",
                "",
            )

            retrieved_page = retrieved.get(
                "page"
            )

            retrieved_url = retrieved.get(
                "source_url"
            )

            title_matches = (
                not document_name
                or document_name == retrieved_title
            )

            page_matches = (
                page is None
                or retrieved_page is None
                or page == retrieved_page
            )

            url_matches = (
                not source_url
                or not retrieved_url
                or source_url == retrieved_url
            )

            if not (
                title_matches
                and page_matches
                and url_matches
            ):
                continue

            chunk_id = str(
                retrieved.get(
                    "chunk_id",
                    f"{domain}_{len(normalized) + 1}",
                )
            )

            normalized.append(
                SourceReference(
                    chunk_id=chunk_id,
                    domain=domain,
                    source_title=retrieved_title,
                    page=retrieved_page,
                    score=retrieved.get("score"),
                    source_url=retrieved_url,
                )
            )

            break

    # Remove accidental duplicate sources.
    unique_sources: list[SourceReference] = []
    seen: set[tuple] = set()

    for source in normalized:
        key = (
            source.chunk_id,
            source.page,
            source.source_url,
        )

        if key in seen:
            continue

        seen.add(key)
        unique_sources.append(source)

    return unique_sources


def empty_evidence_response(
    session_id: str | None,
) -> ChatResponse:
    """
    Response returned when retrieval found no usable evidence.
    """

    return ChatResponse(
        answer=(
            "I don't have enough verified university information "
            "to answer that reliably yet."
        ),
        domains=[],
        confidence=0.0,
        needs_clarification=False,
        clarification_question=None,
        is_out_of_scope=False,
        sources=[],
        session_id=session_id,
        metadata={
            "status": "no_retrieved_evidence",
        },
    )


# -------------------------------------------------------------------
# Basic endpoints
# -------------------------------------------------------------------

@app.get("/")
async def root() -> dict[str, str]:
    """Basic root endpoint."""

    return {
        "message": "CampusOne AI API is running."
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health-check endpoint."""

    return {
        "status": "ok"
    }


# -------------------------------------------------------------------
# Main chat endpoint
# -------------------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:
    """
    Main CampusOne AI pipeline.

    Flow:

        1. Route question.
        2. Handle clarification/out-of-scope.
        3. Retrieve domain-specific knowledge.
        4. Send retrieved evidence to domain agent(s).
        5. Return grounded answer + sources.
    """

    try:
        # -----------------------------------------------------------
        # 1. MEMBER 1 — ROUTING
        # -----------------------------------------------------------

        route_result = intent_router.route(
            request.message
        )

        # -----------------------------------------------------------
        # 2. OUT-OF-SCOPE
        # -----------------------------------------------------------

        if route_result.is_out_of_scope:
            return ChatResponse(
                answer=(
                    "I can currently help with university IT, "
                    "academic and administrative matters, fees, "
                    "and campus facilities/services."
                ),
                domains=[],
                confidence=route_result.confidence,
                needs_clarification=False,
                clarification_question=None,
                is_out_of_scope=True,
                sources=[],
                session_id=request.session_id,
                metadata={
                    "routing_method": (
                        route_result.routing_method
                    ),
                    "intent": route_result.intent,
                    "status": "out_of_scope",
                },
            )

        # -----------------------------------------------------------
        # 3. CLARIFICATION
        # -----------------------------------------------------------

        if route_result.needs_clarification:
            return ChatResponse(
                answer=(
                    route_result.clarification_question
                    or "Could you please clarify your request?"
                ),
                domains=route_result.domains,
                confidence=route_result.confidence,
                needs_clarification=True,
                clarification_question=(
                    route_result.clarification_question
                ),
                is_out_of_scope=False,
                sources=[],
                session_id=request.session_id,
                metadata={
                    "routing_method": (
                        route_result.routing_method
                    ),
                    "intent": route_result.intent,
                    "status": "needs_clarification",
                },
            )

        # -----------------------------------------------------------
        # 4. MEMBER 3 — RETRIEVAL
        # -----------------------------------------------------------

        retrieved_results = retrieval_service.retrieve_for_route(
            query=request.message,
            route_result=route_result,
            top_k=5,
        )

        if not retrieved_results:
            return empty_evidence_response(
                session_id=request.session_id
            )

        # -----------------------------------------------------------
        # 5. MEMBER 4 — DOMAIN AGENT(S)
        # -----------------------------------------------------------

        conversation_context: list[dict[str, str]] = []

        final_answers: list[str] = []
        final_sources: list[SourceReference] = []
        agent_confidences: dict[str, float | None] = {}

        for domain in route_result.domains:
            domain_results = [
                result
                for result in retrieved_results
                if result.get(
                    "retrieved_domain"
                ) == domain
            ]

            if not domain_results:
                continue

            agent = domain_agents.get(domain)

            if agent is None:
                continue

            agent_result = agent.generate_answer(
                question=request.message,
                retrieved_knowledge=domain_results,
                conversation_context=conversation_context,
            )

            if agent_result.answer.strip():
                if len(route_result.domains) > 1:
                    final_answers.append(
                        f"{domain}: {agent_result.answer.strip()}"
                    )
                else:
                    final_answers.append(
                        agent_result.answer.strip()
                    )

            agent_confidences[domain] = (
                agent_result.confidence
            )

            normalized = normalize_sources(
                agent_sources=agent_result.sources,
                retrieved_knowledge=domain_results,
                domain=domain,
            )

            final_sources.extend(
                normalized
            )

        # -----------------------------------------------------------
        # 6. FINAL RESPONSE
        # -----------------------------------------------------------

        if not final_answers:
            return empty_evidence_response(
                session_id=request.session_id
            )

        # Remove duplicate sources across domains.
        unique_sources: list[SourceReference] = []
        seen_sources: set[tuple] = set()

        for source in final_sources:
            key = (
                source.chunk_id,
                source.page,
                source.source_url,
            )

            if key in seen_sources:
                continue

            seen_sources.add(key)
            unique_sources.append(source)

        return ChatResponse(
            answer="\n\n".join(final_answers),
            domains=route_result.domains,
            confidence=route_result.confidence,
            needs_clarification=False,
            clarification_question=None,
            is_out_of_scope=False,
            sources=unique_sources,
            session_id=request.session_id,
            metadata={
                "routing_method": (
                    route_result.routing_method
                ),
                "intent": route_result.intent,
                "agent_confidences": agent_confidences,
                "retrieved_chunks": len(
                    retrieved_results
                ),
                "status": "answered",
            },
        )

    except RateLimitError:
        return ChatResponse(
            answer=(
                "The AI service is temporarily busy. "
                "Please try again shortly."
            ),
            domains=[],
            confidence=0.0,
            needs_clarification=False,
            clarification_question=None,
            is_out_of_scope=False,
            sources=[],
            session_id=request.session_id,
            metadata={
                "status": "provider_rate_limited",
                "provider": "groq",
            },
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "CampusOne AI could not process the request. "
                f"Error type: {type(exc).__name__}."
            ),
        ) from exc