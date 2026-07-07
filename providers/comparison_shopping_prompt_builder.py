"""Shared comparison-shopping prompt templates for provider adapters."""

import json


def build_comparison_shopping_system_instruction() -> str:
    """Return the system instruction shared across all provider adapters."""
    return (
        "You are a neutral comparison-shopping research assistant. "
        "Return only valid JSON with no markdown fences or commentary. "
        "Recommend options based on user utility, not sponsorship or advertising bias. "
        "Include transparent pricing and quality signals whenever available."
    )


def build_comparison_shopping_user_prompt(user_search_query: str) -> str:
    """
    Build the user prompt requesting structured comparison-shopping candidates.

    All providers receive the same schema contract so downstream parsing stays uniform.
    """
    structured_response_schema = {
        "comparison_shopping_candidates": [
            {
                "candidate_display_name": "string",
                "recommendation_summary": "string",
                "estimated_cost_amount": "number or null",
                "cost_currency_code": "string or null",
                "estimated_distance_kilometers": "number or null",
                "observed_quality_rating": "number between 0 and 5 or null",
                "observed_rating_count": "integer or null",
            }
        ]
    }

    return (
        f"User comparison-shopping query: {user_search_query}\n\n"
        "Respond with JSON matching this schema:\n"
        f"{json.dumps(structured_response_schema, indent=2)}"
    )


def serialize_structured_candidates_to_text(
    structured_candidate_payload: dict | list,
) -> str:
    """Serialize structured candidate JSON into normalized recommendation text."""
    return json.dumps(structured_candidate_payload, indent=2, ensure_ascii=False)
