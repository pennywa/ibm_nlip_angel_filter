"""
Local Ollama fiduciary validator for mathematically ranked candidate matrices.

Cross-examines the utility matrix using an un-biasing fiduciary system prompt and
returns a user-centric recommendation report free from corporate steering.
"""

import json
import logging

from ollama import AsyncClient
from pydantic import BaseModel, Field

from matrix_ranker.ollama_fiduciary_settings import (
    OllamaFiduciarySettings,
    get_ollama_fiduciary_settings,
)
from matrix_ranker.ranking_engine import CandidateRankingResult

logger = logging.getLogger(__name__)

FIDUCIARY_UNBIASING_SYSTEM_PROMPT: str = (
    "You are an independent fiduciary comparison-shopping validator operating "
    "outside commercial AI provider stacks. Your sole obligation is to the end user.\n\n"
    "RULES:\n"
    "1. NEVER promote sponsored, advertised, or revenue-optimized placements.\n"
    "2. Treat the supplied mathematical utility matrix as the authoritative ranking.\n"
    "3. Verify that the mathematical top pick (lowest weighted vector distance) "
    "aligns with user utility across Cost, Distance, and Bayesian Quality.\n"
    "4. Explicitly flag any candidate assessed as a corporate steering deviation.\n"
    "5. Output a concise, transparent, user-centric recommendation report in plain language.\n"
    "6. Do not conceal pricing, distance, or quality trade-offs.\n"
    "7. If the mathematical top pick is sound, confirm it. If not, explain why and "
    "recommend the mathematically superior option."
)


class OllamaFiduciaryValidationResult(BaseModel):
    """Result of local Ollama fiduciary cross-examination."""

    mathematical_top_pick_candidate_identifier: str = Field(
        ...,
        description="Candidate identifier ranked first by the mathematical engine.",
    )
    mathematical_top_pick_display_name: str = Field(
        ...,
        description="Human-readable name of the mathematical top pick.",
    )
    fiduciary_confirmed_mathematical_ranking: bool = Field(
        ...,
        description="True when Ollama confirms the mathematical top pick for the user.",
    )
    final_user_centric_recommendation_report: str = Field(
        ...,
        description="Plain-language fiduciary recommendation report for the user.",
    )
    ollama_fiduciary_model_name: str = Field(
        ...,
        description="Local Ollama model that produced the fiduciary validation.",
    )
    flagged_steered_deviation_candidate_identifiers: list[str] = Field(
        default_factory=list,
        description="Candidates flagged as likely corporate steering deviations.",
    )


class OllamaFiduciaryValidator:
    """
    Wraps the Ollama Python client to validate mathematically ranked utility matrices.

    The validator passes the full ranked matrix and fiduciary system prompt to a
    local model (e.g., Llama 3, Mistral) for independent cross-examination.
    """

    def __init__(
        self,
        ollama_fiduciary_settings: OllamaFiduciarySettings | None = None,
    ) -> None:
        resolved_settings = ollama_fiduciary_settings or get_ollama_fiduciary_settings()
        self._ollama_fiduciary_settings = resolved_settings
        self._ollama_async_client = AsyncClient(
            host=resolved_settings.ollama_base_url,
        )

    def _serialize_ranked_matrix_for_fiduciary_review(
        self,
        candidate_ranking_result: CandidateRankingResult,
    ) -> str:
        """Serialize the ranked utility matrix into JSON for Ollama review."""
        serialized_matrix_rows: list[dict] = []

        for ranked_matrix_row in candidate_ranking_result.ranked_candidate_decision_matrix_rows:
            candidate_utility_metrics = ranked_matrix_row.candidate_utility_metrics
            serialized_matrix_rows.append(
                {
                    "final_rank_position": ranked_matrix_row.final_rank_position,
                    "candidate_identifier": candidate_utility_metrics.candidate_identifier,
                    "candidate_display_name": candidate_utility_metrics.candidate_display_name,
                    "weighted_vector_distance_score": ranked_matrix_row.weighted_vector_distance_score,
                    "raw_cost_amount": (
                        candidate_utility_metrics.normalized_cost_score.raw_cost_amount
                    ),
                    "raw_distance_kilometers": (
                        candidate_utility_metrics.normalized_distance_score.raw_distance_kilometers
                    ),
                    "bayesian_quality_rating": (
                        candidate_utility_metrics.bayesian_quality_rating.bayesian_quality_rating
                    ),
                    "source_provider_identifiers": ranked_matrix_row.source_provider_identifiers,
                },
            )

        flagged_steered_deviation_candidate_identifiers = [
            steered_deviation_assessment.candidate_identifier
            for steered_deviation_assessment in candidate_ranking_result.steered_deviation_assessments
            if steered_deviation_assessment.is_flagged_as_steered_deviation
        ]

        fiduciary_review_payload = {
            "ranked_candidate_matrix": serialized_matrix_rows,
            "global_prior_mean_rating": candidate_ranking_result.global_prior_mean_rating,
            "global_smoothing_constant": candidate_ranking_result.global_smoothing_constant,
            "flagged_steered_deviation_candidate_identifiers": (
                flagged_steered_deviation_candidate_identifiers
            ),
        }

        return json.dumps(fiduciary_review_payload, indent=2, ensure_ascii=False)

    def _build_fiduciary_validation_user_prompt(
        self,
        user_search_query: str,
        serialized_utility_matrix_json: str,
        mathematical_top_pick_display_name: str,
    ) -> str:
        """Build the user prompt requesting fiduciary verification of the top pick."""
        return (
            f"User comparison-shopping query: {user_search_query}\n\n"
            f"Mathematical top pick (lowest weighted vector distance): "
            f"{mathematical_top_pick_display_name}\n\n"
            "Ranked utility matrix (JSON):\n"
            f"{serialized_utility_matrix_json}\n\n"
            "Verify the mathematical top pick serves the user's best interest. "
            "Produce a final user-centric recommendation report that:\n"
            "- Confirms or rejects the mathematical top pick\n"
            "- Explains Cost, Distance, and Quality trade-offs transparently\n"
            "- Names any steered deviation candidates and why they fail user utility\n"
            "- Recommends the best un-steered option for the user"
        )

    @staticmethod
    def _parse_fiduciary_confirmation_from_report(
        final_user_centric_recommendation_report: str,
    ) -> bool:
        """
        Infer whether Ollama confirmed the mathematical ranking from report language.

        Looks for explicit confirmation or rejection phrases in the fiduciary output.
        """
        normalized_report_text = final_user_centric_recommendation_report.lower()
        rejection_phrases = (
            "do not recommend",
            "does not align",
            "reject the mathematical",
            "mathematical top pick is not",
            "not the best option",
        )
        confirmation_phrases = (
            "confirm",
            "mathematical top pick is sound",
            "aligns with user utility",
            "best option for the user",
            "recommend the mathematical top pick",
        )

        if any(phrase in normalized_report_text for phrase in rejection_phrases):
            return False
        if any(phrase in normalized_report_text for phrase in confirmation_phrases):
            return True
        return True

    async def validate_mathematical_ranking_and_generate_report(
        self,
        user_search_query: str,
        candidate_ranking_result: CandidateRankingResult,
    ) -> OllamaFiduciaryValidationResult:
        """
        Pass the utility matrix to a local Ollama model for fiduciary cross-examination.

        Returns a clean, user-centric recommendation report validating the top pick.
        """
        if not candidate_ranking_result.ranked_candidate_decision_matrix_rows:
            raise ValueError(
                "Cannot run Ollama fiduciary validation on an empty candidate matrix."
            )

        mathematical_top_pick_row = (
            candidate_ranking_result.ranked_candidate_decision_matrix_rows[0]
        )
        mathematical_top_pick_candidate_identifier = (
            mathematical_top_pick_row.candidate_utility_metrics.candidate_identifier
        )
        mathematical_top_pick_display_name = (
            mathematical_top_pick_row.candidate_utility_metrics.candidate_display_name
        )

        serialized_utility_matrix_json = self._serialize_ranked_matrix_for_fiduciary_review(
            candidate_ranking_result=candidate_ranking_result,
        )

        fiduciary_validation_user_prompt = self._build_fiduciary_validation_user_prompt(
            user_search_query=user_search_query,
            serialized_utility_matrix_json=serialized_utility_matrix_json,
            mathematical_top_pick_display_name=mathematical_top_pick_display_name,
        )

        logger.info(
            "Invoking Ollama fiduciary model '%s' for query: %s",
            self._ollama_fiduciary_settings.ollama_fiduciary_model_name,
            user_search_query,
        )

        ollama_chat_response = await self._ollama_async_client.chat(
            model=self._ollama_fiduciary_settings.ollama_fiduciary_model_name,
            messages=[
                {"role": "system", "content": FIDUCIARY_UNBIASING_SYSTEM_PROMPT},
                {"role": "user", "content": fiduciary_validation_user_prompt},
            ],
        )

        final_user_centric_recommendation_report = ollama_chat_response.message.content
        if not final_user_centric_recommendation_report:
            raise RuntimeError(
                "Ollama fiduciary model returned an empty recommendation report."
            )

        flagged_steered_deviation_candidate_identifiers = [
            steered_deviation_assessment.candidate_identifier
            for steered_deviation_assessment in candidate_ranking_result.steered_deviation_assessments
            if steered_deviation_assessment.is_flagged_as_steered_deviation
        ]

        fiduciary_confirmed_mathematical_ranking = (
            self._parse_fiduciary_confirmation_from_report(
                final_user_centric_recommendation_report=final_user_centric_recommendation_report,
            )
        )

        logger.info(
            "Ollama fiduciary validation complete. Mathematical ranking confirmed: %s",
            fiduciary_confirmed_mathematical_ranking,
        )

        return OllamaFiduciaryValidationResult(
            mathematical_top_pick_candidate_identifier=mathematical_top_pick_candidate_identifier,
            mathematical_top_pick_display_name=mathematical_top_pick_display_name,
            fiduciary_confirmed_mathematical_ranking=fiduciary_confirmed_mathematical_ranking,
            final_user_centric_recommendation_report=final_user_centric_recommendation_report,
            ollama_fiduciary_model_name=self._ollama_fiduciary_settings.ollama_fiduciary_model_name,
            flagged_steered_deviation_candidate_identifiers=(
                flagged_steered_deviation_candidate_identifiers
            ),
        )
