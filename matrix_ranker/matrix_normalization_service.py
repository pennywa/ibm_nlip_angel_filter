"""Parse provider response payloads into candidate decision matrix rows."""

import json
import re
from typing import Any

from matrix_ranker.candidate_utility_metric_schemas import (
    BayesianQualityRating,
    CandidateDecisionMatrixRow,
    CandidateUtilityMetrics,
    NormalizedCostScore,
    NormalizedDistanceScore,
)
from providers.base_provider import ProviderResponsePayload

DEFAULT_MISSING_COST_AMOUNT: float = 999_999.0
DEFAULT_MISSING_DISTANCE_KILOMETERS: float = 999.0
DEFAULT_MISSING_QUALITY_RATING: float = 3.0
DEFAULT_MISSING_RATING_COUNT: int = 0


def _slugify_candidate_identifier_source_text(source_text: str) -> str:
    """Convert display text into a stable lowercase identifier fragment."""
    normalized_source_text = source_text.strip().lower()
    slugified_text = re.sub(r"[^a-z0-9]+", "-", normalized_source_text)
    return slugified_text.strip("-") or "unknown-candidate"


def _normalize_cost_to_unit_interval(
    raw_cost_amount: float,
    minimum_observed_cost_amount: float,
    maximum_observed_cost_amount: float,
) -> float:
    """Map a raw cost amount into the zero-to-one normalized cost interval."""
    if maximum_observed_cost_amount <= minimum_observed_cost_amount:
        return 0.5
    normalized_cost_score = (raw_cost_amount - minimum_observed_cost_amount) / (
        maximum_observed_cost_amount - minimum_observed_cost_amount
    )
    return max(0.0, min(1.0, normalized_cost_score))


def _normalize_distance_to_unit_interval(
    raw_distance_kilometers: float,
    minimum_observed_distance_kilometers: float,
    maximum_observed_distance_kilometers: float,
) -> float:
    """Map a raw distance value into the zero-to-one normalized distance interval."""
    if maximum_observed_distance_kilometers <= minimum_observed_distance_kilometers:
        return 0.5
    normalized_distance_score = (
        raw_distance_kilometers - minimum_observed_distance_kilometers
    ) / (
        maximum_observed_distance_kilometers - minimum_observed_distance_kilometers
    )
    return max(0.0, min(1.0, normalized_distance_score))


def _parse_structured_provider_candidate_records(
    provider_response_payload: ProviderResponsePayload,
) -> list[dict[str, Any]]:
    """Extract comparison-shopping candidate records from a provider JSON payload."""
    try:
        parsed_provider_json_payload = json.loads(
            provider_response_payload.raw_recommendation_text,
        )
    except json.JSONDecodeError as json_decode_error:
        raise ValueError(
            f"Provider '{provider_response_payload.provider_identifier}' returned "
            "non-JSON recommendation text."
        ) from json_decode_error

    structured_candidate_records = parsed_provider_json_payload.get(
        "comparison_shopping_candidates",
        [],
    )
    if not isinstance(structured_candidate_records, list):
        raise ValueError(
            f"Provider '{provider_response_payload.provider_identifier}' JSON payload "
            "missing a comparison_shopping_candidates array."
        )
    return structured_candidate_records


def build_candidate_decision_matrix_from_provider_payloads(
    successful_provider_response_payloads: list[ProviderResponsePayload],
) -> list[CandidateDecisionMatrixRow]:
    """
    Normalize successful provider payloads into a unified candidate decision matrix.

    Deduplicates candidates by display name while preserving provider attribution.
    """
    aggregated_candidate_records: dict[str, dict[str, Any]] = {}

    for provider_response_payload in successful_provider_response_payloads:
        structured_candidate_records = _parse_structured_provider_candidate_records(
            provider_response_payload=provider_response_payload,
        )

        for candidate_record_index, structured_candidate_record in enumerate(
            structured_candidate_records,
        ):
            candidate_display_name = str(
                structured_candidate_record.get("candidate_display_name", "Unknown"),
            )
            candidate_identifier_slug = _slugify_candidate_identifier_source_text(
                candidate_display_name,
            )
            candidate_identifier = (
                f"{provider_response_payload.provider_identifier}-"
                f"{candidate_identifier_slug}-{candidate_record_index}"
            )

            if candidate_identifier not in aggregated_candidate_records:
                aggregated_candidate_records[candidate_identifier] = {
                    "candidate_identifier": candidate_identifier,
                    "candidate_display_name": candidate_display_name,
                    "estimated_cost_amount": structured_candidate_record.get(
                        "estimated_cost_amount",
                    ),
                    "cost_currency_code": structured_candidate_record.get(
                        "cost_currency_code",
                    ),
                    "estimated_distance_kilometers": structured_candidate_record.get(
                        "estimated_distance_kilometers",
                    ),
                    "observed_quality_rating": structured_candidate_record.get(
                        "observed_quality_rating",
                    ),
                    "observed_rating_count": structured_candidate_record.get(
                        "observed_rating_count",
                        DEFAULT_MISSING_RATING_COUNT,
                    ),
                    "source_provider_identifiers": [
                        provider_response_payload.provider_identifier,
                    ],
                }
            else:
                existing_record = aggregated_candidate_records[candidate_identifier]
                existing_record["source_provider_identifiers"].append(
                    provider_response_payload.provider_identifier,
                )

    if not aggregated_candidate_records:
        return []

    observed_cost_amounts = [
        float(record["estimated_cost_amount"])
        for record in aggregated_candidate_records.values()
        if record["estimated_cost_amount"] is not None
    ]
    observed_distance_kilometers_values = [
        float(record["estimated_distance_kilometers"])
        for record in aggregated_candidate_records.values()
        if record["estimated_distance_kilometers"] is not None
    ]

    minimum_observed_cost_amount = min(observed_cost_amounts) if observed_cost_amounts else 0.0
    maximum_observed_cost_amount = (
        max(observed_cost_amounts) if observed_cost_amounts else DEFAULT_MISSING_COST_AMOUNT
    )
    minimum_observed_distance_kilometers = (
        min(observed_distance_kilometers_values)
        if observed_distance_kilometers_values
        else 0.0
    )
    maximum_observed_distance_kilometers = (
        max(observed_distance_kilometers_values)
        if observed_distance_kilometers_values
        else DEFAULT_MISSING_DISTANCE_KILOMETERS
    )

    candidate_decision_matrix_rows: list[CandidateDecisionMatrixRow] = []

    for aggregated_candidate_record in aggregated_candidate_records.values():
        candidate_identifier = aggregated_candidate_record["candidate_identifier"]
        raw_cost_amount = aggregated_candidate_record["estimated_cost_amount"]
        if raw_cost_amount is None:
            raw_cost_amount = DEFAULT_MISSING_COST_AMOUNT
        raw_distance_kilometers = aggregated_candidate_record["estimated_distance_kilometers"]
        if raw_distance_kilometers is None:
            raw_distance_kilometers = DEFAULT_MISSING_DISTANCE_KILOMETERS
        observed_quality_rating = aggregated_candidate_record["observed_quality_rating"]
        if observed_quality_rating is None:
            observed_quality_rating = DEFAULT_MISSING_QUALITY_RATING
        observed_rating_count = int(
            aggregated_candidate_record.get(
                "observed_rating_count",
                DEFAULT_MISSING_RATING_COUNT,
            ),
        )

        normalized_cost_score_value = _normalize_cost_to_unit_interval(
            raw_cost_amount=float(raw_cost_amount),
            minimum_observed_cost_amount=minimum_observed_cost_amount,
            maximum_observed_cost_amount=maximum_observed_cost_amount,
        )
        normalized_distance_score_value = _normalize_distance_to_unit_interval(
            raw_distance_kilometers=float(raw_distance_kilometers),
            minimum_observed_distance_kilometers=minimum_observed_distance_kilometers,
            maximum_observed_distance_kilometers=maximum_observed_distance_kilometers,
        )

        candidate_utility_metrics = CandidateUtilityMetrics(
            candidate_identifier=candidate_identifier,
            candidate_display_name=aggregated_candidate_record["candidate_display_name"],
            normalized_cost_score=NormalizedCostScore(
                candidate_identifier=candidate_identifier,
                normalized_cost_score=normalized_cost_score_value,
                raw_cost_amount=float(raw_cost_amount),
                cost_currency_code=aggregated_candidate_record.get("cost_currency_code"),
            ),
            normalized_distance_score=NormalizedDistanceScore(
                candidate_identifier=candidate_identifier,
                normalized_distance_score=normalized_distance_score_value,
                raw_distance_kilometers=float(raw_distance_kilometers),
            ),
            bayesian_quality_rating=BayesianQualityRating(
                candidate_identifier=candidate_identifier,
                bayesian_quality_rating=float(observed_quality_rating),
                observed_rating_count=observed_rating_count,
                raw_average_rating=float(observed_quality_rating),
            ),
        )

        candidate_decision_matrix_rows.append(
            CandidateDecisionMatrixRow(
                candidate_utility_metrics=candidate_utility_metrics,
                source_provider_identifiers=aggregated_candidate_record[
                    "source_provider_identifiers"
                ],
            ),
        )

    return candidate_decision_matrix_rows
