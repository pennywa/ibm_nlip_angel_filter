"""
End-to-end integration verification for the local Ollama comparison-shopping pipeline.

Exercises:
    1. Mocked Ollama markdown-fence JSON extraction through matrix normalization.
    2. Live provider fan-out, normalization, and weighted Euclidean ranking lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from matrix_ranker.matrix_normalization_service import (
    build_candidate_decision_matrix_from_provider_payloads,
)
from matrix_ranker.ranking_engine import (
    CandidateRankingEngine,
    UserAbsoluteUtilityProfile,
)
from providers.ollama_adapter import (
    OllamaProviderAdapter,
    extract_json_payload_from_local_model_response,
)
from providers.provider_orchestrator import ProviderOrchestrator
from providers.provider_settings import ProviderSettings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

INTEGRATION_USER_SEARCH_QUERY: str = (
    "best budget-friendly carry-on luggage under 100 dollars"
)

ALL_CLOUD_PROVIDER_IDENTIFIERS: tuple[str, ...] = ("openai", "gemini", "watson")

MOCK_MARKDOWN_FENCED_OLLAMA_RESPONSE_TEXT: str = """\
Here is your data for the comparison-shopping query:

```json
{
  "comparison_shopping_candidates": [
    {
      "candidate_display_name": "TravelPro Budget Carry-On",
      "recommendation_summary": "Lightweight hard-shell carry-on with strong value.",
      "estimated_cost_amount": 79.99,
      "cost_currency_code": "USD",
      "estimated_distance_kilometers": 12.5,
      "observed_quality_rating": 4.2,
      "observed_rating_count": 318
    },
    {
      "candidate_display_name": "AmazonBasics Expandable Spinner",
      "recommendation_summary": "Affordable expandable spinner with decent reviews.",
      "estimated_cost_amount": 54.99,
      "cost_currency_code": "USD",
      "estimated_distance_kilometers": 8.0,
      "observed_quality_rating": 3.8,
      "observed_rating_count": 1240
    }
  ]
}
```

Hope this helps!
"""


def detect_skipped_cloud_provider_identifiers(
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
    candidate_ranking_result,
    successful_provider_response_payloads,
    excluded_provider_identifiers: list[str],
    skipped_provider_identifiers: list[str],
) -> dict[str, Any]:
    """
    Assemble lifecycle telemetry matching federator routing metadata expectations.

    The returned dictionary tracks sorted candidates, successful upstream providers,
    and providers excluded or skipped during orchestration.
    """
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
        "ranked_candidate_records": ranked_candidate_records,
        "upstream_provider_identifiers": upstream_provider_identifiers,
        "excluded_provider_identifiers": excluded_provider_identifiers,
        "skipped_provider_identifiers": skipped_provider_identifiers,
        "global_prior_mean_rating": candidate_ranking_result.global_prior_mean_rating,
        "global_smoothing_constant": candidate_ranking_result.global_smoothing_constant,
    }


async def execute_comparison_shopping_lifecycle(
    user_search_query: str,
    provider_settings: ProviderSettings | None = None,
    user_absolute_utility_profile: UserAbsoluteUtilityProfile | None = None,
) -> dict[str, Any]:
    """
    Run the full comparison-shopping data lifecycle from provider fan-out to ranking.

    Stages:
        ProviderOrchestrator fan-out
        -> matrix normalization
        -> CandidateRankingEngine weighted Euclidean sorting
        -> lifecycle telemetry dictionary
    """
    resolved_provider_settings = provider_settings or ProviderSettings()
    resolved_user_utility_profile = user_absolute_utility_profile or UserAbsoluteUtilityProfile()

    provider_orchestrator = ProviderOrchestrator.create_from_provider_settings(
        provider_settings=resolved_provider_settings,
    )

    (
        successful_provider_response_payloads,
        excluded_provider_identifiers,
    ) = await provider_orchestrator.execute_all_providers(
        user_search_query=user_search_query,
    )

    if not successful_provider_response_payloads:
        raise RuntimeError(
            "Comparison-shopping lifecycle failed: no provider returned candidates. "
            f"Excluded providers: {excluded_provider_identifiers}"
        )

    candidate_decision_matrix_rows = build_candidate_decision_matrix_from_provider_payloads(
        successful_provider_response_payloads=successful_provider_response_payloads,
    )

    if not candidate_decision_matrix_rows:
        raise RuntimeError(
            "Comparison-shopping lifecycle failed: normalization produced zero matrix rows."
        )

    candidate_ranking_engine = CandidateRankingEngine()
    candidate_ranking_result = candidate_ranking_engine.rank_candidate_decision_matrix(
        candidate_decision_matrix_rows=candidate_decision_matrix_rows,
        user_absolute_utility_profile=resolved_user_utility_profile,
    )

    skipped_provider_identifiers = detect_skipped_cloud_provider_identifiers(
        provider_settings=resolved_provider_settings,
    )

    return build_comparison_shopping_lifecycle_telemetry(
        candidate_ranking_result=candidate_ranking_result,
        successful_provider_response_payloads=successful_provider_response_payloads,
        excluded_provider_identifiers=excluded_provider_identifiers,
        skipped_provider_identifiers=skipped_provider_identifiers,
    )


def assert_lifecycle_telemetry_structure(lifecycle_telemetry: dict[str, Any]) -> None:
    """Assert the lifecycle telemetry dictionary exposes required tracking fields."""
    required_top_level_keys = {
        "ranked_candidate_records",
        "upstream_provider_identifiers",
        "excluded_provider_identifiers",
        "skipped_provider_identifiers",
    }
    assert required_top_level_keys.issubset(lifecycle_telemetry.keys())

    ranked_candidate_records = lifecycle_telemetry["ranked_candidate_records"]
    assert isinstance(ranked_candidate_records, list)
    assert len(ranked_candidate_records) > 0

    previous_weighted_vector_distance_score: float | None = None
    for ranked_candidate_record in ranked_candidate_records:
        required_candidate_keys = {
            "final_rank_position",
            "candidate_identifier",
            "candidate_display_name",
            "weighted_vector_distance_score",
            "source_provider_identifiers",
        }
        assert required_candidate_keys.issubset(ranked_candidate_record.keys())

        current_weighted_vector_distance_score = ranked_candidate_record[
            "weighted_vector_distance_score"
        ]
        assert current_weighted_vector_distance_score is not None

        if previous_weighted_vector_distance_score is not None:
            assert current_weighted_vector_distance_score >= previous_weighted_vector_distance_score

        previous_weighted_vector_distance_score = current_weighted_vector_distance_score

    upstream_provider_identifiers = lifecycle_telemetry["upstream_provider_identifiers"]
    excluded_provider_identifiers = lifecycle_telemetry["excluded_provider_identifiers"]
    skipped_provider_identifiers = lifecycle_telemetry["skipped_provider_identifiers"]

    assert isinstance(upstream_provider_identifiers, list)
    assert isinstance(excluded_provider_identifiers, list)
    assert isinstance(skipped_provider_identifiers, list)
    assert len(upstream_provider_identifiers) > 0

    tracked_provider_identifiers = set(upstream_provider_identifiers) | set(
        excluded_provider_identifiers,
    )
    for skipped_provider_identifier in skipped_provider_identifiers:
        assert skipped_provider_identifier not in tracked_provider_identifiers


async def is_local_ollama_endpoint_reachable(ollama_base_url: str) -> bool:
    """Return True when the local Ollama HTTP API responds to a tags probe."""
    normalized_ollama_base_url = ollama_base_url.rstrip("/")
    ollama_tags_probe_url = f"{normalized_ollama_base_url}/api/tags"

    try:
        ollama_health_response = httpx.get(ollama_tags_probe_url, timeout=3.0)
        return ollama_health_response.status_code == 200
    except httpx.HTTPError:
        return False


class MockOllamaMarkdownFenceIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """Verify markdown-fence JSON extraction and matrix normalization for Ollama."""

    async def test_ollama_adapter_extracts_markdown_fenced_json_into_matrix_rows(self) -> None:
        """Mock a messy Ollama response and assert pristine normalized matrix rows."""
        extracted_json_payload = extract_json_payload_from_local_model_response(
            local_model_response_text=MOCK_MARKDOWN_FENCED_OLLAMA_RESPONSE_TEXT,
        )

        comparison_shopping_candidates = extracted_json_payload["comparison_shopping_candidates"]
        self.assertEqual(len(comparison_shopping_candidates), 2)
        self.assertEqual(
            comparison_shopping_candidates[0]["candidate_display_name"],
            "TravelPro Budget Carry-On",
        )

        mock_ollama_chat_response = MagicMock()
        mock_ollama_chat_response.message.content = MOCK_MARKDOWN_FENCED_OLLAMA_RESPONSE_TEXT

        mock_ollama_async_client = MagicMock()
        mock_ollama_async_client.chat = AsyncMock(return_value=mock_ollama_chat_response)

        with patch(
            "providers.ollama_adapter.AsyncClient",
            return_value=mock_ollama_async_client,
        ):
            ollama_provider_adapter = OllamaProviderAdapter(
                ollama_base_url="http://127.0.0.1:11434",
                ollama_comparison_shopping_model_name="llama3.2",
            )

            provider_response_payload = (
                await ollama_provider_adapter.execute_comparison_shopping_query(
                    user_search_query=INTEGRATION_USER_SEARCH_QUERY,
                )
            )

        self.assertEqual(provider_response_payload.provider_identifier, "ollama")

        parsed_provider_json_payload = json.loads(
            provider_response_payload.raw_recommendation_text,
        )
        self.assertIn("comparison_shopping_candidates", parsed_provider_json_payload)

        candidate_decision_matrix_rows = build_candidate_decision_matrix_from_provider_payloads(
            successful_provider_response_payloads=[provider_response_payload],
        )

        self.assertEqual(len(candidate_decision_matrix_rows), 2)

        first_matrix_row = candidate_decision_matrix_rows[0]
        first_candidate_utility_metrics = first_matrix_row.candidate_utility_metrics

        self.assertEqual(
            first_candidate_utility_metrics.candidate_display_name,
            "TravelPro Budget Carry-On",
        )
        self.assertEqual(first_matrix_row.source_provider_identifiers, ["ollama"])
        self.assertGreater(
            first_candidate_utility_metrics.normalized_cost_score.normalized_cost_score,
            0.0,
        )
        self.assertLessEqual(
            first_candidate_utility_metrics.normalized_cost_score.normalized_cost_score,
            1.0,
        )

        candidate_ranking_engine = CandidateRankingEngine()
        candidate_ranking_result = candidate_ranking_engine.rank_candidate_decision_matrix(
            candidate_decision_matrix_rows=candidate_decision_matrix_rows,
            user_absolute_utility_profile=UserAbsoluteUtilityProfile(),
        )

        lifecycle_telemetry = build_comparison_shopping_lifecycle_telemetry(
            candidate_ranking_result=candidate_ranking_result,
            successful_provider_response_payloads=[provider_response_payload],
            excluded_provider_identifiers=[],
            skipped_provider_identifiers=["openai", "gemini", "watson"],
        )

        assert_lifecycle_telemetry_structure(lifecycle_telemetry)
        self.assertEqual(lifecycle_telemetry["upstream_provider_identifiers"], ["ollama"])


class LiveOllamaIntegrationLifecycleTest(unittest.IsolatedAsyncioTestCase):
    """Live integration against a running local Ollama instance in development mode."""

    async def test_live_development_lifecycle_with_local_ollama(self) -> None:
        """Fan out to active providers, normalize, rank, and verify lifecycle telemetry."""
        live_provider_settings = ProviderSettings()

        if not live_provider_settings.is_development_environment:
            self.skipTest(
                "Live Ollama integration requires APPLICATION_ENVIRONMENT=development."
            )

        ollama_endpoint_is_reachable = await is_local_ollama_endpoint_reachable(
            ollama_base_url=live_provider_settings.ollama_base_url,
        )
        if not ollama_endpoint_is_reachable:
            self.skipTest(
                f"Local Ollama endpoint is not reachable at "
                f"{live_provider_settings.ollama_base_url}."
            )

        lifecycle_telemetry = await execute_comparison_shopping_lifecycle(
            user_search_query=INTEGRATION_USER_SEARCH_QUERY,
            provider_settings=live_provider_settings,
        )

        assert_lifecycle_telemetry_structure(lifecycle_telemetry)

        upstream_provider_identifiers = lifecycle_telemetry["upstream_provider_identifiers"]
        self.assertIn(
            "ollama",
            upstream_provider_identifiers,
            "Development lifecycle must include a successful Ollama provider response.",
        )

        logger.info(
            "Live lifecycle telemetry: upstream=%s excluded=%s skipped=%s ranked_count=%d",
            lifecycle_telemetry["upstream_provider_identifiers"],
            lifecycle_telemetry["excluded_provider_identifiers"],
            lifecycle_telemetry["skipped_provider_identifiers"],
            len(lifecycle_telemetry["ranked_candidate_records"]),
        )


async def run_live_integration_verification_block() -> int:
    """
    Execute the live development lifecycle block and print telemetry for manual inspection.

    Returns process exit code (0 success, 1 failure).
    """
    live_provider_settings = ProviderSettings()

    print("=== Live Ollama Integration Lifecycle Verification ===")
    print(f"APPLICATION_ENVIRONMENT={live_provider_settings.application_environment}")
    print(f"OLLAMA_BASE_URL={live_provider_settings.ollama_base_url}")
    print(
        f"OLLAMA_COMPARISON_SHOPPING_MODEL_NAME="
        f"{live_provider_settings.ollama_comparison_shopping_model_name}"
    )

    if not live_provider_settings.is_development_environment:
        print(
            "SKIP: Set APPLICATION_ENVIRONMENT=development in .env to run the live block."
        )
        return 0

    ollama_endpoint_is_reachable = await is_local_ollama_endpoint_reachable(
        ollama_base_url=live_provider_settings.ollama_base_url,
    )
    if not ollama_endpoint_is_reachable:
        print(
            f"SKIP: Local Ollama is not reachable at {live_provider_settings.ollama_base_url}."
        )
        return 0

    try:
        lifecycle_telemetry = await execute_comparison_shopping_lifecycle(
            user_search_query=INTEGRATION_USER_SEARCH_QUERY,
            provider_settings=live_provider_settings,
        )
    except RuntimeError as lifecycle_runtime_error:
        print(f"FAIL: {lifecycle_runtime_error}")
        return 1

    assert_lifecycle_telemetry_structure(lifecycle_telemetry)

    print(json.dumps(lifecycle_telemetry, indent=2, default=str))
    print("PASS: Live Ollama integration lifecycle verification succeeded.")
    return 0


def main() -> int:
    """Run mocked unit tests, then optional live lifecycle verification."""
    unittest_runner = unittest.TextTestRunner(verbosity=2)
    unittest_test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    unittest_result = unittest_runner.run(unittest_test_suite)

    if not unittest_result.wasSuccessful():
        return 1

    live_verification_exit_code = asyncio.run(run_live_integration_verification_block())
    return live_verification_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
