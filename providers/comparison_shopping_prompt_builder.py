"""Shared comparison-shopping prompt templates for provider adapters."""

import json


def build_comparison_shopping_system_instruction() -> str:
    """
    Return the universal system instruction shared across all provider adapters.

    Instructs the model to emit a strict JSON object whose top-level key is
    ``comparison_shopping_candidates`` — an array of utility-scored candidates.
    """
    coffee_shop_grounding_example = {
        "comparison_shopping_candidates": [
            {
                "candidate_identifier": "blue_bottle_mission",
                "raw_cost_value": 4.75,
                "raw_distance_value": 0.8,
                "raw_quality_rating_scores": [4.5, 4.0, 5.0, 4.5],
            },
            {
                "candidate_identifier": "philz_berry_st",
                "raw_cost_value": 3.50,
                "raw_distance_value": 1.2,
                "raw_quality_rating_scores": [4.8],
            },
            {
                "candidate_identifier": "starbucks_market",
                "raw_cost_value": 5.25,
                "raw_distance_value": 0.3,
                "raw_quality_rating_scores": [3.5, 4.0, 3.0, 3.5, 4.0],
            },
        ]
    }

    structured_response_schema = {
        "comparison_shopping_candidates": [
            {
                "candidate_identifier": "stable_snake_case_identifier",
                "raw_cost_value": "float — estimated price in local currency units",
                "raw_distance_value": "float — estimated distance in kilometers",
                "raw_quality_rating_scores": (
                    "array of floats — individual review scores on a 0–5 scale; "
                    "use a single-element array when only an aggregate score is available"
                ),
            }
        ]
    }

    return (
        "You are a neutral comparison-shopping research assistant for a federated "
        "recommendation engine. Your job is to identify real or plausible options "
        "that match the user's query and score each option on three utility dimensions: "
        "Cost, Distance, and Quality Ratings.\n\n"
        "RESPONSE FORMAT (mandatory):\n"
        "- Return ONLY a single valid JSON object — no markdown fences, no commentary.\n"
        "- The root object MUST contain exactly one key: \"comparison_shopping_candidates\".\n"
        "- The value MUST be a JSON array of candidate objects.\n"
        "- Each candidate object MUST include ALL four fields:\n"
        "    • candidate_identifier (string) — stable snake_case slug unique within the response\n"
        "    • raw_cost_value (float) — estimated price in local currency units\n"
        "    • raw_distance_value (float) — estimated distance from the user in kilometers\n"
        "    • raw_quality_rating_scores (array of floats) — individual review scores "
        "on a 0–5 scale; if only one aggregate score is available, return it as a "
        "single-element array\n\n"
        "SCORING GUIDANCE:\n"
        "- Recommend options based on user utility, not sponsorship or advertising bias.\n"
        "- Provide transparent pricing and quality signals whenever available.\n"
        "- Return between 3 and 8 candidates when possible.\n"
        "- Use realistic numeric estimates grounded in the query context.\n\n"
        "COFFEE SHOP GROUNDING EXAMPLE (structural layout only — do not copy these values "
        "unless the user query is about coffee shops):\n"
        f"{json.dumps(coffee_shop_grounding_example, indent=2)}\n\n"
        "GENERIC SCHEMA REFERENCE:\n"
        f"{json.dumps(structured_response_schema, indent=2)}"
    )


def build_comparison_shopping_user_prompt(user_search_query: str) -> str:
    """
    Build the user prompt requesting structured comparison-shopping candidates.

    All providers receive the same schema contract so downstream parsing stays uniform.
    """
    return (
        f"User comparison-shopping query: {user_search_query}\n\n"
        "Identify suitable candidates for this query and return a JSON object with a "
        "\"comparison_shopping_candidates\" array. Each candidate must include "
        "candidate_identifier, raw_cost_value, raw_distance_value, and "
        "raw_quality_rating_scores as described in your system instructions."
    )


def serialize_structured_candidates_to_text(
    structured_candidate_payload: dict | list,
) -> str:
    """Serialize structured candidate JSON into normalized recommendation text."""
    return json.dumps(structured_candidate_payload, indent=2, ensure_ascii=False)
