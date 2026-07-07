"""
Isolated verification gate for the providers subsystem.

Validates graceful exclusion when API keys are missing, mock fan-out execution,
and JSON payload structure for successful provider responses.

Run from the repository root:
    py -3 test_providers_subsystem.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from providers.base_provider import BaseProvider, ProviderResponsePayload
from providers.gemini_provider import GeminiProvider
from providers.openai_provider import OpenAIProvider
from providers.provider_orchestrator import ProviderOrchestrator
from providers.provider_settings import ProviderSettings, get_provider_settings
from providers.watson_provider import WatsonProvider

DEMO_USER_SEARCH_QUERY: str = (
    "Find the best coffee shops near downtown by cost, distance, and ratings"
)

ALL_PROVIDER_IDENTIFIERS: tuple[str, ...] = (
    OpenAIProvider.PROVIDER_IDENTIFIER,
    GeminiProvider.PROVIDER_IDENTIFIER,
    WatsonProvider.PROVIDER_IDENTIFIER,
)

MOCK_COMPARISON_SHOPPING_CANDIDATES_PAYLOAD: dict[str, Any] = {
    "comparison_shopping_candidates": [
        {
            "candidate_identifier": "mock_blue_bottle_downtown",
            "raw_cost_value": 4.50,
            "raw_distance_value": 0.6,
            "raw_quality_rating_scores": [4.5, 4.0, 4.5],
        },
        {
            "candidate_identifier": "mock_philz_civic_center",
            "raw_cost_value": 3.75,
            "raw_distance_value": 1.1,
            "raw_quality_rating_scores": [4.8],
        },
        {
            "candidate_identifier": "mock_ritual_mission",
            "raw_cost_value": 5.00,
            "raw_distance_value": 0.9,
            "raw_quality_rating_scores": [4.2, 4.6, 4.0],
        },
    ]
}

MOCK_RAW_RECOMMENDATION_TEXT: str = json.dumps(
    MOCK_COMPARISON_SHOPPING_CANDIDATES_PAYLOAD,
    indent=2,
    ensure_ascii=False,
)


class _VerificationGateFailure(Exception):
    """Raised when an isolated verification gate does not pass."""


def _build_mock_provider_adapter(
    provider_identifier: str,
    provider_model_name: str,
    raw_recommendation_text: str,
) -> BaseProvider:
    """Return a provider adapter that always succeeds with a fixed payload."""

    class _MockComparisonShoppingProvider(BaseProvider):
        @property
        def provider_identifier(self) -> str:
            return provider_identifier

        async def execute_comparison_shopping_query(
            self,
            user_search_query: str,
        ) -> ProviderResponsePayload:
            return ProviderResponsePayload(
                provider_identifier=provider_identifier,
                raw_recommendation_text=raw_recommendation_text,
                provider_model_name=provider_model_name,
            )

    return _MockComparisonShoppingProvider()


def _validate_comparison_shopping_candidates_payload(
    raw_recommendation_text: str,
    provider_identifier: str,
) -> list[str]:
    """
    Validate that raw provider text decodes into the expected candidate schema.

    Returns a list of human-readable validation errors (empty when valid).
    """
    validation_errors: list[str] = []

    try:
        decoded_provider_payload = json.loads(raw_recommendation_text)
    except json.JSONDecodeError as json_decode_error:
        return [
            f"{provider_identifier}: raw_recommendation_text is not valid JSON "
            f"({json_decode_error})"
        ]

    if not isinstance(decoded_provider_payload, dict):
        return [
            f"{provider_identifier}: decoded payload must be a JSON object, "
            f"got {type(decoded_provider_payload).__name__}"
        ]

    comparison_shopping_candidates = decoded_provider_payload.get(
        "comparison_shopping_candidates"
    )
    if comparison_shopping_candidates is None:
        return [
            f"{provider_identifier}: missing top-level key "
            f"'comparison_shopping_candidates'"
        ]

    if not isinstance(comparison_shopping_candidates, list):
        return [
            f"{provider_identifier}: 'comparison_shopping_candidates' must be an array"
        ]

    if not comparison_shopping_candidates:
        return [
            f"{provider_identifier}: 'comparison_shopping_candidates' must not be empty"
        ]

    for candidate_index, comparison_shopping_candidate in enumerate(
        comparison_shopping_candidates
    ):
        candidate_path = f"{provider_identifier}.comparison_shopping_candidates[{candidate_index}]"

        if not isinstance(comparison_shopping_candidate, dict):
            validation_errors.append(f"{candidate_path} must be a JSON object")
            continue

        candidate_identifier = comparison_shopping_candidate.get("candidate_identifier")
        if not isinstance(candidate_identifier, str) or not candidate_identifier.strip():
            validation_errors.append(
                f"{candidate_path}.candidate_identifier must be a non-empty string"
            )

        raw_cost_value = comparison_shopping_candidate.get("raw_cost_value")
        if not isinstance(raw_cost_value, (int, float)):
            validation_errors.append(
                f"{candidate_path}.raw_cost_value must be a float"
            )

        raw_distance_value = comparison_shopping_candidate.get("raw_distance_value")
        if not isinstance(raw_distance_value, (int, float)):
            validation_errors.append(
                f"{candidate_path}.raw_distance_value must be a float"
            )

        raw_quality_rating_scores = comparison_shopping_candidate.get(
            "raw_quality_rating_scores"
        )
        if not isinstance(raw_quality_rating_scores, list) or not raw_quality_rating_scores:
            validation_errors.append(
                f"{candidate_path}.raw_quality_rating_scores must be a non-empty array"
            )
        else:
            for rating_index, raw_quality_rating_score in enumerate(
                raw_quality_rating_scores
            ):
                if not isinstance(raw_quality_rating_score, (int, float)):
                    validation_errors.append(
                        f"{candidate_path}.raw_quality_rating_scores[{rating_index}] "
                        f"must be a float"
                    )

    return validation_errors


async def _verify_missing_api_keys_are_excluded() -> None:
    """Gate 1: missing keys must land in excluded_provider_identifiers without crashing."""
    print("\n[GATE 1] Key absence — all providers excluded safely")

    provider_settings_with_no_api_keys = ProviderSettings(
        openai_api_key="",
        gemini_api_key="",
        watson_api_key="",
        watson_project_id="",
    )
    provider_orchestrator = ProviderOrchestrator.create_from_provider_settings(
        provider_settings=provider_settings_with_no_api_keys,
    )

    try:
        (
            successful_provider_response_payloads,
            excluded_provider_identifiers,
        ) = await provider_orchestrator.execute_all_providers(
            user_search_query=DEMO_USER_SEARCH_QUERY,
        )
    except Exception as unhandled_orchestrator_exception:
        raise _VerificationGateFailure(
            "ProviderOrchestrator raised an unhandled exception when all API keys "
            f"were missing: {unhandled_orchestrator_exception}"
        ) from unhandled_orchestrator_exception

    if successful_provider_response_payloads:
        raise _VerificationGateFailure(
            "Expected zero successful payloads when all API keys are missing, "
            f"got {len(successful_provider_response_payloads)}"
        )

    missing_excluded_provider_identifiers = set(ALL_PROVIDER_IDENTIFIERS) - set(
        excluded_provider_identifiers
    )
    if missing_excluded_provider_identifiers:
        raise _VerificationGateFailure(
            "Expected all provider identifiers in excluded_provider_identifiers "
            f"when keys are absent; missing: {sorted(missing_excluded_provider_identifiers)}"
        )

    print(f"  PASS — excluded: {sorted(excluded_provider_identifiers)}")


async def _verify_selective_key_absence_is_excluded() -> None:
    """Gate 1b: a single missing key excludes only that provider."""
    print("\n[GATE 1b] Selective key absence — only missing provider excluded")

    provider_settings_with_partial_api_keys = ProviderSettings(
        openai_api_key="configured-openai-key",
        gemini_api_key="",
        watson_api_key="configured-watson-key",
        watson_project_id="configured-watson-project",
    )
    provider_orchestrator = ProviderOrchestrator.create_from_provider_settings(
        provider_settings=provider_settings_with_partial_api_keys,
    )

    mock_openai_provider_adapter = _build_mock_provider_adapter(
        provider_identifier=OpenAIProvider.PROVIDER_IDENTIFIER,
        provider_model_name="mock-openai",
        raw_recommendation_text=MOCK_RAW_RECOMMENDATION_TEXT,
    )
    mock_watson_provider_adapter = _build_mock_provider_adapter(
        provider_identifier=WatsonProvider.PROVIDER_IDENTIFIER,
        provider_model_name="mock-watson",
        raw_recommendation_text=MOCK_RAW_RECOMMENDATION_TEXT,
    )

    provider_orchestrator_with_mocks = ProviderOrchestrator(
        registered_provider_adapters=[
            mock_openai_provider_adapter,
            provider_orchestrator._registered_provider_adapters[1],
            mock_watson_provider_adapter,
        ],
        provider_query_timeout_seconds=provider_settings_with_partial_api_keys.provider_query_timeout_seconds,
    )

    try:
        (
            successful_provider_response_payloads,
            excluded_provider_identifiers,
        ) = await provider_orchestrator_with_mocks.execute_all_providers(
            user_search_query=DEMO_USER_SEARCH_QUERY,
        )
    except Exception as unhandled_orchestrator_exception:
        raise _VerificationGateFailure(
            "ProviderOrchestrator raised an unhandled exception during selective "
            f"key absence test: {unhandled_orchestrator_exception}"
        ) from unhandled_orchestrator_exception

    if GeminiProvider.PROVIDER_IDENTIFIER not in excluded_provider_identifiers:
        raise _VerificationGateFailure(
            "Expected Gemini provider in excluded_provider_identifiers when "
            "GEMINI_API_KEY is missing"
        )

    if len(successful_provider_response_payloads) != 2:
        raise _VerificationGateFailure(
            "Expected two successful mock providers (openai, watson), "
            f"got {len(successful_provider_response_payloads)}"
        )

    print(
        f"  PASS — excluded: {sorted(excluded_provider_identifiers)}; "
        f"succeeded: {[payload.provider_identifier for payload in successful_provider_response_payloads]}"
    )


async def _verify_mock_execute_pipeline() -> None:
    """Gate 2: mock fan-out executes the demo query without crashing."""
    print("\n[GATE 2] Mock execute pipeline — demo coffee-shop query")

    mock_registered_provider_adapters: list[BaseProvider] = [
        _build_mock_provider_adapter(
            provider_identifier=OpenAIProvider.PROVIDER_IDENTIFIER,
            provider_model_name="mock-gpt-4.1-nano",
            raw_recommendation_text=MOCK_RAW_RECOMMENDATION_TEXT,
        ),
        _build_mock_provider_adapter(
            provider_identifier=GeminiProvider.PROVIDER_IDENTIFIER,
            provider_model_name="mock-gemini-1.5-flash",
            raw_recommendation_text=MOCK_RAW_RECOMMENDATION_TEXT,
        ),
        _build_mock_provider_adapter(
            provider_identifier=WatsonProvider.PROVIDER_IDENTIFIER,
            provider_model_name="mock-llama-3-8b-instruct",
            raw_recommendation_text=MOCK_RAW_RECOMMENDATION_TEXT,
        ),
    ]

    provider_orchestrator = ProviderOrchestrator(
        registered_provider_adapters=mock_registered_provider_adapters,
        provider_query_timeout_seconds=10.0,
    )

    try:
        (
            successful_provider_response_payloads,
            excluded_provider_identifiers,
        ) = await provider_orchestrator.execute_all_providers(
            user_search_query=DEMO_USER_SEARCH_QUERY,
        )
    except Exception as unhandled_orchestrator_exception:
        raise _VerificationGateFailure(
            "Mock execute pipeline raised an unhandled exception: "
            f"{unhandled_orchestrator_exception}"
        ) from unhandled_orchestrator_exception

    if excluded_provider_identifiers:
        raise _VerificationGateFailure(
            f"Mock pipeline should exclude no providers, got: {excluded_provider_identifiers}"
        )

    if len(successful_provider_response_payloads) != len(ALL_PROVIDER_IDENTIFIERS):
        raise _VerificationGateFailure(
            "Mock pipeline should return one payload per provider, "
            f"got {len(successful_provider_response_payloads)}"
        )

    print(
        f"  PASS — {len(successful_provider_response_payloads)} payloads returned for "
        f"query: {DEMO_USER_SEARCH_QUERY!r}"
    )

    await _verify_payload_structure_for_successes(successful_provider_response_payloads)


async def _verify_payload_structure_for_successes(
    successful_provider_response_payloads: list[ProviderResponsePayload],
) -> None:
    """Gate 3: successful payloads decode into comparison_shopping_candidates schema."""
    print("\n[GATE 3] Payload structure — comparison_shopping_candidates schema")

    aggregated_validation_errors: list[str] = []

    for provider_response_payload in successful_provider_response_payloads:
        payload_validation_errors = _validate_comparison_shopping_candidates_payload(
            raw_recommendation_text=provider_response_payload.raw_recommendation_text,
            provider_identifier=provider_response_payload.provider_identifier,
        )
        aggregated_validation_errors.extend(payload_validation_errors)

        if not payload_validation_errors:
            decoded_payload = json.loads(provider_response_payload.raw_recommendation_text)
            candidate_count = len(
                decoded_payload["comparison_shopping_candidates"]
            )
            print(
                f"  PASS — {provider_response_payload.provider_identifier}: "
                f"{candidate_count} candidates validated"
            )

    if aggregated_validation_errors:
        formatted_errors = "\n    ".join(aggregated_validation_errors)
        raise _VerificationGateFailure(
            f"Payload structure validation failed:\n    {formatted_errors}"
        )


async def _verify_live_execute_pipeline_if_configured() -> None:
    """Optional gate: live fan-out when real API keys are present in the environment."""
    get_provider_settings.cache_clear()
    resolved_provider_settings = get_provider_settings()

    configured_provider_identifiers: list[str] = []
    if resolved_provider_settings.openai_api_key.strip():
        configured_provider_identifiers.append(OpenAIProvider.PROVIDER_IDENTIFIER)
    if resolved_provider_settings.gemini_api_key.strip():
        configured_provider_identifiers.append(GeminiProvider.PROVIDER_IDENTIFIER)
    if (
        resolved_provider_settings.watson_api_key.strip()
        and resolved_provider_settings.watson_project_id.strip()
    ):
        configured_provider_identifiers.append(WatsonProvider.PROVIDER_IDENTIFIER)

    if not configured_provider_identifiers:
        print("\n[GATE 4] Live execute pipeline — SKIPPED (no API keys in environment)")
        return

    print(
        f"\n[GATE 4] Live execute pipeline — providers with keys: "
        f"{configured_provider_identifiers}"
    )

    provider_orchestrator = ProviderOrchestrator.create_from_provider_settings(
        provider_settings=resolved_provider_settings,
    )

    try:
        (
            successful_provider_response_payloads,
            excluded_provider_identifiers,
        ) = await provider_orchestrator.execute_all_providers(
            user_search_query=DEMO_USER_SEARCH_QUERY,
        )
    except Exception as unhandled_orchestrator_exception:
        raise _VerificationGateFailure(
            "Live execute pipeline raised an unhandled exception: "
            f"{unhandled_orchestrator_exception}"
        ) from unhandled_orchestrator_exception

    print(
        f"  INFO — succeeded: "
        f"{[payload.provider_identifier for payload in successful_provider_response_payloads]}; "
        f"excluded: {sorted(excluded_provider_identifiers)}"
    )

    if not successful_provider_response_payloads:
        print("  WARN — no live providers succeeded; structure gate skipped for live path")
        return

    await _verify_payload_structure_for_successes(successful_provider_response_payloads)
    print("  PASS — live provider payloads validated")


async def _run_all_verification_gates() -> None:
    """Execute every isolated verification gate in sequence."""
    print("=" * 72)
    print("PROVIDERS SUBSYSTEM — ISOLATED VERIFICATION GATE")
    print("=" * 72)

    await _verify_missing_api_keys_are_excluded()
    await _verify_selective_key_absence_is_excluded()
    await _verify_mock_execute_pipeline()
    await _verify_live_execute_pipeline_if_configured()

    print("\n" + "=" * 72)
    print("ALL VERIFICATION GATES PASSED")
    print("=" * 72)


def main() -> None:
    """Entry point for the isolated providers subsystem verification script."""
    try:
        asyncio.run(_run_all_verification_gates())
    except _VerificationGateFailure as verification_gate_failure:
        print(f"\nVERIFICATION FAILED: {verification_gate_failure}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nVERIFICATION INTERRUPTED", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
