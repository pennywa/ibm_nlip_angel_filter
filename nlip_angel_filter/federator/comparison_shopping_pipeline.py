"""End-to-end comparison-shopping federator pipeline orchestration."""

import logging
import os
from uuid import uuid4

from matrix_ranker.matrix_normalization_service import (
    build_candidate_decision_matrix_from_provider_payloads,
)
from matrix_ranker.ollama_fiduciary_validator import (
    OllamaFiduciaryValidationResult,
    OllamaFiduciaryValidator,
)
from matrix_ranker.ranking_engine import (
    CandidateRankingEngine,
    CandidateRankingResult,
    UserAbsoluteUtilityProfile,
)
from protocol.nlip_message_envelope_schemas import (
    NLIPComparisonShoppingQueryPayload,
    NLIPFederatorRoutingMetadata,
    NLIPMessageEnvelope,
    NLIPMessageHeader,
    NLIPMessageKind,
    NLIPMessageOrigin,
    NLIPNormalizedRecommendationPayload,
    NLIPProviderAttributionRecord,
)
from providers.provider_orchestrator import ProviderOrchestrator

logger = logging.getLogger(__name__)

FEDERATOR_INSTANCE_IDENTIFIER: str = os.environ.get(
    "FEDERATOR_INSTANCE_IDENTIFIER",
    "nlip-angel-filter-federator",
)


class ComparisonShoppingPipelineOrchestrator:
    """
    Bridges the full Angel Filter pipeline from provider fan-out to Ollama validation.

    Pipeline stages:
        1. Multi-provider parallel fan-out
        2. Matrix normalization
        3. Bayesian smoothing and weighted vector distance ranking
        4. Local Ollama fiduciary validation
        5. NLIP response envelope assembly
    """

    def __init__(
        self,
        provider_orchestrator: ProviderOrchestrator | None = None,
        candidate_ranking_engine: CandidateRankingEngine | None = None,
        ollama_fiduciary_validator: OllamaFiduciaryValidator | None = None,
    ) -> None:
        self._provider_orchestrator = (
            provider_orchestrator or ProviderOrchestrator.create_from_provider_settings()
        )
        self._candidate_ranking_engine = candidate_ranking_engine or CandidateRankingEngine()
        self._ollama_fiduciary_validator = (
            ollama_fiduciary_validator or OllamaFiduciaryValidator()
        )

    async def execute_comparison_shopping_pipeline(
        self,
        comparison_shopping_query_payload: NLIPComparisonShoppingQueryPayload,
        authenticated_github_username: str | None = None,
    ) -> NLIPMessageEnvelope:
        """Execute the full federator pipeline and return an NLIP response envelope."""
        user_search_query = comparison_shopping_query_payload.user_search_query

        logger.info(
            "Starting comparison-shopping pipeline for query: %s (user: %s)",
            user_search_query,
            authenticated_github_username or "anonymous",
        )

        (
            successful_provider_response_payloads,
            excluded_provider_identifiers,
        ) = await self._provider_orchestrator.execute_all_providers(
            user_search_query=user_search_query,
        )

        if not successful_provider_response_payloads:
            raise RuntimeError(
                "All provider adapters failed. Cannot build candidate decision matrix. "
                f"Excluded providers: {excluded_provider_identifiers}"
            )

        candidate_decision_matrix_rows = build_candidate_decision_matrix_from_provider_payloads(
            successful_provider_response_payloads=successful_provider_response_payloads,
        )

        if not candidate_decision_matrix_rows:
            raise RuntimeError(
                "Provider payloads contained no parseable comparison-shopping candidates."
            )

        user_absolute_utility_profile = UserAbsoluteUtilityProfile(
            user_preference_weight_cost=comparison_shopping_query_payload.user_preference_weight_cost,
            user_preference_weight_distance=comparison_shopping_query_payload.user_preference_weight_distance,
            user_preference_weight_quality=comparison_shopping_query_payload.user_preference_weight_quality,
        )

        candidate_ranking_result: CandidateRankingResult = (
            self._candidate_ranking_engine.rank_candidate_decision_matrix(
                candidate_decision_matrix_rows=candidate_decision_matrix_rows,
                user_absolute_utility_profile=user_absolute_utility_profile,
            )
        )

        ollama_fiduciary_validation_result: OllamaFiduciaryValidationResult = (
            await self._ollama_fiduciary_validator.validate_mathematical_ranking_and_generate_report(
                user_search_query=user_search_query,
                candidate_ranking_result=candidate_ranking_result,
            )
        )

        upstream_provider_identifiers = [
            provider_response_payload.provider_identifier
            for provider_response_payload in successful_provider_response_payloads
        ]

        provider_attribution_records = [
            NLIPProviderAttributionRecord(
                provider_identifier=provider_response_payload.provider_identifier,
                provider_model_name=provider_response_payload.provider_model_name,
            )
            for provider_response_payload in successful_provider_response_payloads
        ]

        ranked_candidate_identifiers = [
            ranked_row.candidate_utility_metrics.candidate_identifier
            for ranked_row in candidate_ranking_result.ranked_candidate_decision_matrix_rows
        ]

        federator_routing_metadata = NLIPFederatorRoutingMetadata(
            federator_instance_identifier=FEDERATOR_INSTANCE_IDENTIFIER,
            upstream_provider_identifiers=upstream_provider_identifiers,
            excluded_provider_identifiers=excluded_provider_identifiers,
            correlation_trace_identifier=uuid4(),
        )

        normalized_recommendation_payload = NLIPNormalizedRecommendationPayload(
            ranked_candidate_identifiers=ranked_candidate_identifiers,
            provider_attribution_records=provider_attribution_records,
            federator_routing_metadata=federator_routing_metadata,
        )

        response_message_payload = normalized_recommendation_payload.model_dump(mode="json")
        response_message_payload["fiduciary_validation_report"] = (
            ollama_fiduciary_validation_result.final_user_centric_recommendation_report
        )
        response_message_payload["mathematical_top_pick_candidate_identifier"] = (
            ollama_fiduciary_validation_result.mathematical_top_pick_candidate_identifier
        )
        response_message_payload["fiduciary_confirmed_mathematical_ranking"] = (
            ollama_fiduciary_validation_result.fiduciary_confirmed_mathematical_ranking
        )
        response_message_payload["flagged_steered_deviation_candidate_identifiers"] = (
            ollama_fiduciary_validation_result.flagged_steered_deviation_candidate_identifiers
        )
        response_message_payload["ollama_fiduciary_model_name"] = (
            ollama_fiduciary_validation_result.ollama_fiduciary_model_name
        )
        response_message_payload["ranked_candidate_matrix"] = [
            {
                "final_rank_position": ranked_row.final_rank_position,
                "candidate_identifier": (
                    ranked_row.candidate_utility_metrics.candidate_identifier
                ),
                "candidate_display_name": (
                    ranked_row.candidate_utility_metrics.candidate_display_name
                ),
                "weighted_vector_distance_score": ranked_row.weighted_vector_distance_score,
            }
            for ranked_row in candidate_ranking_result.ranked_candidate_decision_matrix_rows
        ]

        return NLIPMessageEnvelope(
            message_header=NLIPMessageHeader(
                message_kind=NLIPMessageKind.NORMALIZED_RECOMMENDATION,
                message_origin=NLIPMessageOrigin.FEDERATOR,
                authenticated_github_username=authenticated_github_username,
            ),
            message_payload=response_message_payload,
        )
