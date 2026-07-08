"""Dashboard lifecycle orchestration and 3D vector visualization assembly."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from matrix_ranker.matrix_normalization_service import (
    build_candidate_decision_matrix_from_provider_payloads,
)
from matrix_ranker.ranking_engine import (
    CandidateRankingEngine,
    CandidateRankingResult,
    UserAbsoluteUtilityProfile,
)
from matrix_ranker.vector_visualization_service import generate_3d_distance_visualization
from providers.provider_orchestrator import ProviderOrchestrator
from providers.provider_settings import ProviderSettings

logger = logging.getLogger(__name__)

DEFAULT_DASHBOARD_USER_SEARCH_QUERY: str = "coffee places"

_PROVIDER_DISPLAY_LABELS: dict[str, str] = {
    "openai": "OpenAI",
    "ollama": "Local Assistant (Ollama)",
    "gemini": "Google Gemini",
    "watson": "IBM Watson",
}


@dataclass(frozen=True)
class DashboardQueryProfile:
    """Resolved query and utility weights for a dashboard render request."""

    user_search_query: str
    user_absolute_utility_profile: UserAbsoluteUtilityProfile
    low_cost_selected: bool
    proximity_selected: bool
    quality_selected: bool


@dataclass(frozen=True)
class DashboardRecommendationSummary:
    """User-friendly recommendation row for the dashboard results list."""

    final_rank_position: int
    candidate_display_name: str
    provider_display_label: str
    friendly_match_score: int


@dataclass(frozen=True)
class DashboardVisualizationContext:
    """Successful dashboard visualization payload and summary metadata."""

    lifecycle_telemetry: dict[str, Any]
    plotly_html_fragment: str
    ranked_recommendation_summaries: list[DashboardRecommendationSummary]


def map_preference_checkboxes_to_utility_profile(
    low_cost_selected: bool,
    proximity_selected: bool,
    quality_selected: bool,
) -> UserAbsoluteUtilityProfile:
    """
    Map friendly checkbox selections to backend utility weight vectors.

    When no preferences are selected, apply an equal one-third distribution.
  Otherwise split weight evenly across each selected dimension.
    """
    selected_dimensions: list[str] = []
    if low_cost_selected:
        selected_dimensions.append("cost")
    if proximity_selected:
        selected_dimensions.append("distance")
    if quality_selected:
        selected_dimensions.append("quality")

    if not selected_dimensions:
        equal_weight = 1.0 / 3.0
        return UserAbsoluteUtilityProfile(
            user_preference_weight_cost=equal_weight,
            user_preference_weight_distance=equal_weight,
            user_preference_weight_quality=equal_weight,
        )

    dimension_weight = 1.0 / len(selected_dimensions)
    return UserAbsoluteUtilityProfile(
        user_preference_weight_cost=(
            dimension_weight if "cost" in selected_dimensions else 0.0
        ),
        user_preference_weight_distance=(
            dimension_weight if "distance" in selected_dimensions else 0.0
        ),
        user_preference_weight_quality=(
            dimension_weight if "quality" in selected_dimensions else 0.0
        ),
    )


def resolve_dashboard_query_profile_from_form(
    requested_user_search_query: str | None,
    low_cost_selected: bool,
    proximity_selected: bool,
    quality_selected: bool,
) -> DashboardQueryProfile:
    """Resolve the active dashboard query and checkbox-derived utility profile."""
    normalized_user_search_query = (
        requested_user_search_query.strip()
        if requested_user_search_query and requested_user_search_query.strip()
        else DEFAULT_DASHBOARD_USER_SEARCH_QUERY
    )

    return DashboardQueryProfile(
        user_search_query=normalized_user_search_query,
        user_absolute_utility_profile=map_preference_checkboxes_to_utility_profile(
            low_cost_selected=low_cost_selected,
            proximity_selected=proximity_selected,
            quality_selected=quality_selected,
        ),
        low_cost_selected=low_cost_selected,
        proximity_selected=proximity_selected,
        quality_selected=quality_selected,
    )


def resolve_dashboard_query_profile(
    requested_user_search_query: str | None,
    user_preference_weight_cost: float | None = None,
    user_preference_weight_distance: float | None = None,
    user_preference_weight_quality: float | None = None,
) -> DashboardQueryProfile:
    """Resolve query profile from explicit fractional weights (legacy query routes)."""
    normalized_user_search_query = (
        requested_user_search_query.strip()
        if requested_user_search_query and requested_user_search_query.strip()
        else DEFAULT_DASHBOARD_USER_SEARCH_QUERY
    )

    return DashboardQueryProfile(
        user_search_query=normalized_user_search_query,
        user_absolute_utility_profile=UserAbsoluteUtilityProfile(
            user_preference_weight_cost=(
                user_preference_weight_cost
                if user_preference_weight_cost is not None
                else 1.0 / 3.0
            ),
            user_preference_weight_distance=(
                user_preference_weight_distance
                if user_preference_weight_distance is not None
                else 1.0 / 3.0
            ),
            user_preference_weight_quality=(
                user_preference_weight_quality
                if user_preference_weight_quality is not None
                else 1.0 / 3.0
            ),
        ),
        low_cost_selected=False,
        proximity_selected=False,
        quality_selected=False,
    )


def _format_provider_display_label(source_provider_identifiers: list[str]) -> str:
    """Convert provider identifiers into a friendly source label."""
    if not source_provider_identifiers:
        return "Multiple Sources"

    if len(source_provider_identifiers) == 1:
        provider_identifier = source_provider_identifiers[0]
        return _PROVIDER_DISPLAY_LABELS.get(
            provider_identifier,
            provider_identifier.replace("_", " ").title(),
        )

    friendly_labels = [
        _PROVIDER_DISPLAY_LABELS.get(provider_identifier, provider_identifier.title())
        for provider_identifier in sorted(source_provider_identifiers)
    ]
    return ", ".join(friendly_labels)


def _compute_friendly_match_score(
    weighted_vector_distance_score: float,
    all_weighted_vector_distance_scores: list[float],
) -> int:
    """Translate backend distance into a simple, higher-is-better match percentage."""
    if not all_weighted_vector_distance_scores:
        return 0

    minimum_distance_score = min(all_weighted_vector_distance_scores)
    maximum_distance_score = max(all_weighted_vector_distance_scores)

    if maximum_distance_score <= minimum_distance_score:
        return 100

    normalized_closeness = 1.0 - (
        (weighted_vector_distance_score - minimum_distance_score)
        / (maximum_distance_score - minimum_distance_score)
    )
    return max(1, min(100, round(normalized_closeness * 100)))


def build_ranked_recommendation_summaries(
    lifecycle_telemetry: dict[str, Any],
) -> list[DashboardRecommendationSummary]:
    """Build user-friendly recommendation cards from lifecycle telemetry."""
    ranked_candidate_records: list[dict[str, Any]] = lifecycle_telemetry.get(
        "ranked_candidate_records",
        [],
    )
    all_weighted_vector_distance_scores = [
        float(candidate_record["weighted_vector_distance_score"])
        for candidate_record in ranked_candidate_records
    ]

    ranked_recommendation_summaries: list[DashboardRecommendationSummary] = []
    for candidate_record in ranked_candidate_records:
        ranked_recommendation_summaries.append(
            DashboardRecommendationSummary(
                final_rank_position=int(candidate_record["final_rank_position"]),
                candidate_display_name=str(candidate_record["candidate_display_name"]),
                provider_display_label=_format_provider_display_label(
                    candidate_record.get("source_provider_identifiers", []),
                ),
                friendly_match_score=_compute_friendly_match_score(
                    weighted_vector_distance_score=float(
                        candidate_record["weighted_vector_distance_score"],
                    ),
                    all_weighted_vector_distance_scores=all_weighted_vector_distance_scores,
                ),
            ),
        )

    return ranked_recommendation_summaries


def _detect_skipped_cloud_provider_identifiers(
    provider_settings: ProviderSettings,
) -> list[str]:
    """Return cloud provider identifiers omitted from fan-out due to missing credentials."""
    skipped_cloud_provider_identifiers: list[str] = []

    if not provider_settings.openai_provider_is_configured:
        skipped_cloud_provider_identifiers.append("openai")
    if not provider_settings.gemini_provider_is_configured:
        skipped_cloud_provider_identifiers.append("gemini")
    if not provider_settings.watson_provider_is_configured:
        skipped_cloud_provider_identifiers.append("watson")

    return skipped_cloud_provider_identifiers


def build_comparison_shopping_lifecycle_telemetry(
    candidate_ranking_result: CandidateRankingResult,
    successful_provider_response_payloads,
    excluded_provider_identifiers: list[str],
    skipped_provider_identifiers: list[str],
    user_search_query: str,
) -> dict[str, Any]:
    """Assemble lifecycle telemetry for dashboard visualization and audit display."""
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

    upstream_provider_identifiers = [
        provider_response_payload.provider_identifier
        for provider_response_payload in successful_provider_response_payloads
    ]

    return {
        "user_search_query": user_search_query,
        "ranked_candidate_records": ranked_candidate_records,
        "upstream_provider_identifiers": upstream_provider_identifiers,
        "excluded_provider_identifiers": excluded_provider_identifiers,
        "skipped_provider_identifiers": skipped_provider_identifiers,
        "global_prior_mean_rating": candidate_ranking_result.global_prior_mean_rating,
        "global_smoothing_constant": candidate_ranking_result.global_smoothing_constant,
    }


async def execute_dashboard_comparison_shopping_lifecycle(
    dashboard_query_profile: DashboardQueryProfile,
    provider_settings: ProviderSettings | None = None,
) -> dict[str, Any]:
    """Run provider fan-out, normalization, and ranking for dashboard telemetry."""
    resolved_provider_settings = provider_settings or ProviderSettings()

    provider_orchestrator = ProviderOrchestrator.create_from_provider_settings(
        provider_settings=resolved_provider_settings,
    )

    (
        successful_provider_response_payloads,
        excluded_provider_identifiers,
    ) = await provider_orchestrator.execute_all_providers(
        user_search_query=dashboard_query_profile.user_search_query,
    )

    if not successful_provider_response_payloads:
        raise RuntimeError(
            "We could not gather recommendations right now. Please try again in a moment."
        )

    candidate_decision_matrix_rows = build_candidate_decision_matrix_from_provider_payloads(
        successful_provider_response_payloads=successful_provider_response_payloads,
    )

    if not candidate_decision_matrix_rows:
        raise RuntimeError(
            "We found responses, but none contained usable shopping suggestions.",
        )

    candidate_ranking_engine = CandidateRankingEngine()
    candidate_ranking_result = candidate_ranking_engine.rank_candidate_decision_matrix(
        candidate_decision_matrix_rows=candidate_decision_matrix_rows,
        user_absolute_utility_profile=dashboard_query_profile.user_absolute_utility_profile,
    )

    skipped_provider_identifiers = _detect_skipped_cloud_provider_identifiers(
        provider_settings=resolved_provider_settings,
    )

    return build_comparison_shopping_lifecycle_telemetry(
        candidate_ranking_result=candidate_ranking_result,
        successful_provider_response_payloads=successful_provider_response_payloads,
        excluded_provider_identifiers=excluded_provider_identifiers,
        skipped_provider_identifiers=skipped_provider_identifiers,
        user_search_query=dashboard_query_profile.user_search_query,
    )


async def build_dashboard_visualization_context(
    dashboard_query_profile: DashboardQueryProfile,
) -> DashboardVisualizationContext:
    """Execute the lifecycle and render inline Plotly plus friendly result cards."""
    lifecycle_telemetry = await execute_dashboard_comparison_shopping_lifecycle(
        dashboard_query_profile=dashboard_query_profile,
    )

    ranked_candidate_records = lifecycle_telemetry["ranked_candidate_records"]
    if not ranked_candidate_records:
        raise RuntimeError("No ranked recommendations were produced for this search.")

    plotly_html_fragment = generate_3d_distance_visualization(lifecycle_telemetry)

    return DashboardVisualizationContext(
        lifecycle_telemetry=lifecycle_telemetry,
        plotly_html_fragment=plotly_html_fragment,
        ranked_recommendation_summaries=build_ranked_recommendation_summaries(
            lifecycle_telemetry=lifecycle_telemetry,
        ),
    )
