from backend.router.fallback import keyword_route
from backend.router.schemas import RouteResult


def test_wifi_routes_to_it() -> None:
    result = keyword_route(
        "My university WiFi is not working."
    )

    assert result.domains == ["IT"]
    assert result.primary_domain == "IT"
    assert result.intent == "wifi_issue"


def test_fee_deadline_routes_to_fees() -> None:
    result = keyword_route(
        "When is my semester fee deadline?"
    )

    assert result.domains == ["FEES"]
    assert result.primary_domain == "FEES"
    assert result.intent == "fee_deadline"


def test_revaluation_routes_to_hr_admin() -> None:
    result = keyword_route(
        "How can I apply for revaluation?"
    )

    assert result.domains == ["HR_ADMIN"]
    assert result.primary_domain == "HR_ADMIN"
    assert result.intent == "revaluation"


def test_multi_topic_query() -> None:
    result = keyword_route(
        "My WiFi is not working and when are semester fees due?"
    )

    assert set(result.domains) == {"IT", "FEES"}
    assert result.intent == "multiple_topics"


def test_out_of_scope_query() -> None:
    result = keyword_route(
        "Write a C++ sorting program."
    )

    assert result.is_out_of_scope is True
    assert result.domains == []
    assert result.primary_domain is None


def test_empty_query_needs_clarification() -> None:
    result = keyword_route("")

    assert result.needs_clarification is True
    assert result.is_out_of_scope is False
    assert result.domains == []


def test_route_result_schema_accepts_valid_result() -> None:
    result = RouteResult(
        domains=["IT"],
        primary_domain="IT",
        intent="wifi_issue",
        confidence=0.90,
        needs_clarification=False,
        clarification_question=None,
        is_out_of_scope=False,
        routing_method="keyword_fallback",
        reason=None,
    )

    assert result.primary_domain == "IT"
    assert result.confidence == 0.90
    assert result.routing_method == "keyword_fallback"