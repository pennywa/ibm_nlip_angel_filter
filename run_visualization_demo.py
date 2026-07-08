"""
Quick local runner for the Angel Filter 3D vector-space visualization.

Builds or loads lifecycle telemetry, renders an interactive Plotly chart,
saves ``angel_filter_space.html``, and opens it in the default browser.
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Any

from matrix_ranker.matrix_normalization_service import (
    build_candidate_decision_matrix_from_provider_payloads,
)
from matrix_ranker.ranking_engine import CandidateRankingEngine, UserAbsoluteUtilityProfile
from matrix_ranker.vector_visualization_service import generate_3d_distance_visualization
from providers.base_provider import ProviderResponsePayload

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_HTML_PATH = PROJECT_ROOT / "angel_filter_space.html"
TELEMETRY_JSON_PATH = PROJECT_ROOT / "recent_telemetry.json"

DEMO_CANDIDATE_PAYLOAD: dict[str, Any] = {
    "comparison_shopping_candidates": [
        {
            "candidate_display_name": "TravelPro Budget Carry-On",
            "estimated_cost_amount": 79.99,
            "estimated_distance_kilometers": 12.5,
            "observed_quality_rating": 4.2,
            "observed_rating_count": 318,
        },
        {
            "candidate_display_name": "AmazonBasics Expandable Spinner",
            "estimated_cost_amount": 54.99,
            "estimated_distance_kilometers": 8.0,
            "observed_quality_rating": 3.8,
            "observed_rating_count": 1240,
        },
        {
            "candidate_display_name": "Samsonite Lite-Weight Tote",
            "estimated_cost_amount": 99.0,
            "estimated_distance_kilometers": 4.2,
            "observed_quality_rating": 4.6,
            "observed_rating_count": 89,
        },
        {
            "candidate_display_name": "Watson Value Roller",
            "estimated_cost_amount": 64.5,
            "estimated_distance_kilometers": 15.0,
            "observed_quality_rating": 4.0,
            "observed_rating_count": 210,
        },
    ]
}


def build_mock_lifecycle_telemetry() -> dict[str, Any]:
    """Construct lifecycle telemetry using the matrix ranker demo fixture."""
    provider_labels = ("openai", "ollama", "gemini", "watson")
    mock_provider_response_payloads = [
        ProviderResponsePayload(
            provider_identifier=provider_labels[candidate_index % len(provider_labels)],
            provider_model_name=f"{provider_labels[candidate_index % len(provider_labels)]}-demo",
            raw_recommendation_text=json.dumps(
                {"comparison_shopping_candidates": [candidate_record]},
            ),
        )
        for candidate_index, candidate_record in enumerate(
            DEMO_CANDIDATE_PAYLOAD["comparison_shopping_candidates"],
        )
    ]

    candidate_decision_matrix_rows = build_candidate_decision_matrix_from_provider_payloads(
        successful_provider_response_payloads=mock_provider_response_payloads,
    )

    candidate_ranking_result = CandidateRankingEngine().rank_candidate_decision_matrix(
        candidate_decision_matrix_rows=candidate_decision_matrix_rows,
        user_absolute_utility_profile=UserAbsoluteUtilityProfile(),
    )

    ranked_candidate_records: list[dict[str, Any]] = []
    for ranked_matrix_row in candidate_ranking_result.ranked_candidate_decision_matrix_rows:
        candidate_utility_metrics = ranked_matrix_row.candidate_utility_metrics
        ranked_candidate_records.append(
            {
                "final_rank_position": ranked_matrix_row.final_rank_position,
                "candidate_identifier": candidate_utility_metrics.candidate_identifier,
                "candidate_display_name": candidate_utility_metrics.candidate_display_name,
                "weighted_vector_distance_score": ranked_matrix_row.weighted_vector_distance_score,
                "source_provider_identifiers": ranked_matrix_row.source_provider_identifiers,
                "normalized_cost_score": (
                    candidate_utility_metrics.normalized_cost_score.normalized_cost_score
                ),
                "normalized_distance_score": (
                    candidate_utility_metrics.normalized_distance_score.normalized_distance_score
                ),
                "bayesian_quality_rating": (
                    candidate_utility_metrics.bayesian_quality_rating.bayesian_quality_rating
                ),
            },
        )

    return {
        "ranked_candidate_records": ranked_candidate_records,
        "upstream_provider_identifiers": ["openai", "ollama", "gemini", "watson"],
        "excluded_provider_identifiers": [],
        "skipped_provider_identifiers": [],
        "global_prior_mean_rating": candidate_ranking_result.global_prior_mean_rating,
        "global_smoothing_constant": candidate_ranking_result.global_smoothing_constant,
    }


def load_or_build_telemetry_payload() -> dict[str, Any]:
    """Prefer a saved telemetry JSON file; otherwise build a local mock payload."""
    if TELEMETRY_JSON_PATH.is_file():
        print(f"Loading telemetry from {TELEMETRY_JSON_PATH}")
        return json.loads(TELEMETRY_JSON_PATH.read_text(encoding="utf-8"))

    print("No recent_telemetry.json found; building mock lifecycle telemetry.")
    return build_mock_lifecycle_telemetry()


def main() -> int:
    """Generate the visualization HTML and launch it in the default browser."""
    try:
        telemetry_payload = load_or_build_telemetry_payload()
        visualization_html = generate_3d_distance_visualization(telemetry_payload)
    except ImportError as missing_plotly_error:
        print(f"ERROR: {missing_plotly_error}")
        return 1
    except (ValueError, KeyError, TypeError) as visualization_error:
        print(f"ERROR: Failed to build visualization: {visualization_error}")
        return 1

    OUTPUT_HTML_PATH.write_text(visualization_html, encoding="utf-8")
    print(f"Saved visualization to {OUTPUT_HTML_PATH}")

    opened_in_browser = webbrowser.open(OUTPUT_HTML_PATH.as_uri())
    if opened_in_browser:
        print("Opened visualization in the default browser.")
    else:
        print(
            "Could not launch a browser automatically. "
            f"Open this file manually: {OUTPUT_HTML_PATH}",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
