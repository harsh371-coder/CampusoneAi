import re

from .schemas import RouteResult


CLARIFICATION_THRESHOLD = 0.75


# -------------------------------------------------------------------
# Keyword/phrase rules used only when the LLM cannot be used.
#
# Higher weight = stronger evidence for that intent.
# -------------------------------------------------------------------

INTENT_RULES = {
    "IT": {
        "wifi_issue": {
            "wifi": 3,
            "wi-fi": 3,
            "wireless": 3,
            "internet": 2,
            "network": 2,
        },
        "email_issue": {
            "email": 3,
            "mail": 2,
            "outlook": 2,
            "university email": 3,
        },
        "password_reset": {
            "password": 2,
            "reset password": 3,
            "forgot password": 3,
        },
        "portal_issue": {
            "portal": 3,
            "student portal": 3,
            "ums": 2,
        },
        "login_issue": {
            "login": 2,
            "log in": 2,
            "sign in": 2,
            "account": 1,
        },
        "software_or_system_issue": {
            "software": 2,
            "system": 2,
            "application": 1,
            "computer": 1,
        },
    },

    "HR_ADMIN": {
        "academic_calendar": {
            "academic calendar": 3,
            "calendar": 2,
            "semester start": 3,
            "semester": 1,
        },
        "exam_information": {
            "exam": 3,
            "examination": 3,
            "exam date": 3,
            "date sheet": 3,
            "datesheet": 3,
        },
        "exam_rules": {
            "exam rules": 3,
            "examination rules": 3,
            "exam policy": 3,
        },
        "revaluation": {
            "revaluation": 3,
            "re-evaluation": 3,
            "reevaluation": 3,
        },
        "academic_documents": {
            "academic document": 3,
            "marksheet": 3,
            "mark sheet": 3,
            "transcript": 3,
        },
        "certificate_request": {
            "certificate": 3,
            "bonafide": 3,
            "bonafide certificate": 3,
        },
        "university_policy": {
            "policy": 2,
            "university policy": 3,
            "rules": 2,
            "guidelines": 2,
        },
        "general_administration": {
            "registrar": 3,
            "administration": 2,
            "form": 1,
            "notice": 1,
        },
    },

    "FEES": {
        "fee_structure": {
            "fee structure": 3,
            "tuition fee": 3,
            "tuition fees": 3,
            "how much are the fees": 3,
        },
        "fee_payment": {
            "fee payment": 3,
            "pay my fees": 3,
            "pay fees": 3,
            "payment portal": 2,
        },
        "fee_deadline": {
            "fee deadline": 3,
            "fee due date": 3,
            "last date for fees": 3,
            "last date for fee": 3,
            "fees due": 3,
        },
        "payment_issue": {
            "payment": 1,
            "payment failed": 2,
            "payment isn't working": 2,
            "payment is not working": 2,
            "transaction failed": 2,
        },
        "refund": {
            "refund": 3,
            "fee refund": 3,
            "refund policy": 3,
        },
        "fee_related_policy": {
            "fee policy": 3,
            "fee rules": 3,
            "fee related": 2,
        },
    },

    "FACILITIES": {
        "hostel": {
            "hostel": 3,
            "accommodation": 3,
            "hostel room": 3,
        },
        "transport": {
            "transport": 3,
            "bus": 3,
            "bus route": 3,
            "bus routes": 3,
        },
        "maintenance": {
            "maintenance": 3,
            "ac": 2,
            "air conditioner": 2,
            "repair": 2,
            "broken": 1,
        },
        "campus_facility": {
            "facility": 2,
            "facilities": 3,
            "campus facility": 3,
            "campus facilities": 3,
        },
        "student_service": {
            "student service": 3,
            "student services": 3,
            "campus service": 2,
        },
        "grievance": {
            "grievance": 3,
            "complaint": 2,
            "complaint portal": 3,
        },
    },
}


def _contains_term(
    text: str,
    term: str,
) -> bool:
    """
    Check whether a keyword/phrase occurs as a meaningful
    word or phrase rather than as part of an unrelated word.
    """

    if " " in term or "-" in term:
        return term in text

    pattern = rf"\b{re.escape(term)}\b"

    return bool(
        re.search(
            pattern,
            text,
        )
    )


def _calculate_intent_scores(
    question: str,
):
    """
    Calculate weighted scores for each intent.
    """

    text = question.lower().strip()

    intent_scores = {}

    for domain, intents in INTENT_RULES.items():

        intent_scores[domain] = {}

        for intent, keywords in intents.items():

            score = 0

            for keyword, weight in keywords.items():

                if _contains_term(
                    text,
                    keyword,
                ):
                    score += weight

            intent_scores[domain][intent] = score

    return intent_scores


def _domain_scores_from_intents(
    intent_scores,
):
    """
    Convert intent scores into domain scores.
    """

    domain_scores = {}

    for domain, intents in intent_scores.items():
        domain_scores[domain] = max(
            intents.values(),
            default=0,
        )

    return domain_scores


def _confidence_from_score(
    score: int,
) -> float:
    """
    Convert a fallback keyword score into a conservative
    routing confidence.

    This is deliberately capped below 1.0 because the fallback
    is not a machine-learning probability.
    """

    if score >= 4:
        return 0.90

    if score == 3:
        return 0.85

    if score == 2:
        return 0.75

    if score == 1:
        return 0.60

    return 0.0


def keyword_route(
    question: str,
) -> RouteResult:
    """
    Route a question using deterministic keyword/phrase rules.

    This is the backup router used when the Gemini API
    cannot perform the primary routing.
    """

    question = question.strip()

    if not question:
        return RouteResult(
            domains=[],
            primary_domain=None,
            intent="empty_query",
            confidence=0.0,
            needs_clarification=True,
            clarification_question=(
                "What would you like help with?"
            ),
            is_out_of_scope=False,
            routing_method="keyword_fallback",
            reason="The question is empty.",
        )

    intent_scores = _calculate_intent_scores(
        question
    )

    domain_scores = _domain_scores_from_intents(
        intent_scores
    )

    detected_domains = [
        domain
        for domain, score in domain_scores.items()
        if score > 0
    ]

    # ---------------------------------------------------------------
    # No supported domain detected.
    # ---------------------------------------------------------------

    if not detected_domains:
        return RouteResult(
            domains=[],
            primary_domain=None,
            intent="out_of_scope",
            confidence=0.0,
            needs_clarification=False,
            clarification_question=None,
            is_out_of_scope=True,
            routing_method="keyword_fallback",
            reason=(
                "No supported CampusOne domain was detected "
                "by the local fallback router."
            ),
        )

    # ---------------------------------------------------------------
    # Multiple supported domains.
    # ---------------------------------------------------------------

    if len(detected_domains) > 1:

        ranked_domains = sorted(
            detected_domains,
            key=lambda domain: domain_scores[domain],
            reverse=True,
        )

        primary_domain = ranked_domains[0]

        confidence_values = [
            _confidence_from_score(
                domain_scores[domain]
            )
            for domain in detected_domains
        ]

        confidence = min(
            confidence_values
        )

        return RouteResult(
            domains=detected_domains,
            primary_domain=primary_domain,
            intent="multiple_topics",
            confidence=confidence,
            needs_clarification=False,
            clarification_question=None,
            is_out_of_scope=False,
            routing_method="keyword_fallback",
            reason=(
                "Multiple supported knowledge domains were "
                "detected by the local fallback router."
            ),
        )

    # ---------------------------------------------------------------
    # Single-domain query.
    # ---------------------------------------------------------------

    primary_domain = detected_domains[0]

    domain_intents = intent_scores[
        primary_domain
    ]

    ranked_intents = sorted(
        domain_intents.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_intent, best_score = ranked_intents[0]

    confidence = _confidence_from_score(
        best_score
    )

    needs_clarification = (
        confidence < CLARIFICATION_THRESHOLD
    )

    clarification_question = None

    if needs_clarification:
        clarification_question = (
            "Could you provide a little more detail "
            "about what you need help with?"
        )

    return RouteResult(
        domains=[primary_domain],
        primary_domain=primary_domain,
        intent=best_intent,
        confidence=confidence,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        is_out_of_scope=False,
        routing_method="keyword_fallback",
        reason=(
            "The local fallback router selected the domain "
            "and intent using keyword/phrase matching."
        ),
    )


def apply_routing_policy(
    result: RouteResult,
) -> RouteResult:
    """
    Apply deterministic rules after an LLM routing result.
    """

    # Out-of-scope takes priority.
    if result.is_out_of_scope:
        result.primary_domain = None
        result.domains = []
        result.needs_clarification = False

        return result

    # Multiple domains are treated as multiple topics.
    if len(result.domains) > 1:
        result.intent = "multiple_topics"
        result.needs_clarification = False
        result.clarification_question = None

        return result

    # No domain means we cannot safely route.
    if len(result.domains) == 0:
        result.primary_domain = None
        result.needs_clarification = True

        if not result.clarification_question:
            result.clarification_question = (
                "Could you clarify whether your question "
                "is about IT, administration, fees, "
                "or campus facilities?"
            )

        return result

    # Single-domain low-confidence request.
    if result.confidence < CLARIFICATION_THRESHOLD:
        result.needs_clarification = True

        if not result.clarification_question:
            result.clarification_question = (
                "Could you provide a little more detail "
                "so I can route your request correctly?"
            )

    return result