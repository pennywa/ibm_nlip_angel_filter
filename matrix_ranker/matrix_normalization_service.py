"""
Transform upstream provider JSON into uniform candidate decision matrix rows.

Supports the production demo schema (``comparison_shopping_candidates`` with
``raw_cost_value``, ``raw_distance_value``, ``raw_quality_rating_scores``) and
flexible field aliases for arbitrary consumer categories.
"""

from __future__ import annotations

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

CANDIDATE_IDENTIFIER_FIELD_ALIASES: tuple[str, ...] = (
    "candidate_identifier",
    "candidate_id",
    "id",
    "identifier",
)

CANDIDATE_DISPLAY_NAME_FIELD_ALIASES: tuple[str, ...] = (
    "candidate_display_name",
    "display_name",
    "name",
    "title",
    "label",
)

RAW_COST_FIELD_ALIASES: tuple[str, ...] = (
    "raw_cost_value",
    "raw_cost_amount",
    "estimated_cost_amount",
    "cost",
    "price",
)

RAW_DISTANCE_FIELD_ALIASES: tuple[str, ...] = (
    "raw_distance_value",
    "raw_distance_kilometers",
    "estimated_distance_kilometers",
    "distance",
    "distance_kilometers",
)

RAW_QUALITY_RATINGS_FIELD_ALIASES: tuple[str, ...] = (
    "raw_quality_rating_scores",
    "quality_rating_scores",
    "quality_ratings",
    "ratings",
    "review_scores",
)

CANDIDATE_COLLECTION_ROOT_KEYS: tuple[str, ...] = (
    "comparison_shopping_candidates",
    "candidates",
    "recommendations",
    "items",
    "results",
)


def _slugify_candidate_identifier_source_text(source_text: str) -> str:
    """Convert display text into a stable lowercase identifier fragment."""
    normalized_source_text = source_text.strip().lower()
    slugified_text = re.sub(r"[^a-z0-9]+", "_", normalized_source_text)
    return slugified_text.strip("_") or "unknown_candidate"


def _extract_first_matching_field(
    candidate_record: dict[str, Any],
    field_aliases: tuple[str, ...],
) -> Any | None:
    """Return the first non-null value found under any alias key."""
    for field_alias in field_aliases:
        if field_alias in candidate_record and candidate_record[field_alias] is not None:
            return candidate_record[field_alias]
    return None


def _coerce_positive_float(raw_numeric_value: Any, field_description: str) -> float:
    """Parse a numeric provider field into a non-negative float."""
    try:
        parsed_numeric_value = float(raw_numeric_value)
    except (TypeError, ValueError) as numeric_parse_error:
        raise ValueError(
            f"Expected numeric {field_description}, received {raw_numeric_value!r}."
        ) from numeric_parse_error
    if parsed_numeric_value < 0.0:
        raise ValueError(
            f"Expected non-negative {field_description}, received {parsed_numeric_value}."
        )
    return parsed_numeric_value


def _extract_observed_quality_ratings(
    candidate_record: dict[str, Any],
) -> list[float]:
    """
    Extract individual quality ratings from flexible provider record shapes.

    Accepts rating arrays or a single aggregate score field.
    """
    raw_quality_field_value = _extract_first_matching_field(
        candidate_record=candidate_record,
        field_aliases=RAW_QUALITY_RATINGS_FIELD_ALIASES,
    )

    if isinstance(raw_quality_field_value, list):
        if not raw_quality_field_value:
            return []
        return [
            _coerce_positive_float(
                raw_numeric_value=rating_value,
                field_description="quality rating score",
            )
            for rating_value in raw_quality_field_value
        ]

    if raw_quality_field_value is not None:
        aggregate_rating_value = _coerce_positive_float(
            raw_numeric_value=raw_quality_field_value,
            field_description="aggregate quality rating",
        )
        explicit_rating_count = candidate_record.get("observed_rating_count")
        if explicit_rating_count is not None and int(explicit_rating_count) > 1:
            return [aggregate_rating_value] * int(explicit_rating_count)
        return [aggregate_rating_value]

    return []


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


def _resolve_candidate_identifier(
    candidate_record: dict[str, Any],
    *,
    fallback_display_name: str,
    provider_identifier: str | None,
    candidate_record_index: int,
) -> str:
    """Resolve a stable candidate identifier from explicit or derived fields."""
    explicit_candidate_identifier = _extract_first_matching_field(
        candidate_record=candidate_record,
        field_aliases=CANDIDATE_IDENTIFIER_FIELD_ALIASES,
    )
    if explicit_candidate_identifier is not None:
        return str(explicit_candidate_identifier).strip()

    slugified_display_name = _slugify_candidate_identifier_source_text(
        source_text=fallback_display_name,
    )
    if provider_identifier is not None:
        return f"{provider_identifier}_{slugified_display_name}_{candidate_record_index}"
    return f"{slugified_display_name}_{candidate_record_index}"


def _resolve_candidate_display_name(
    candidate_record: dict[str, Any],
    candidate_identifier: str,
) -> str:
    """Resolve a human-readable display name for a candidate record."""
    explicit_display_name = _extract_first_matching_field(
        candidate_record=candidate_record,
        field_aliases=CANDIDATE_DISPLAY_NAME_FIELD_ALIASES,
    )
    if explicit_display_name is not None:
        return str(explicit_display_name).strip()
    return candidate_identifier.replace("_", " ").title()


def parse_raw_json_candidate_block(
    raw_json_payload: dict[str, Any] | list[Any] | str,
    *,
    source_provider_identifier: str | None = None,
) -> list[dict[str, Any]]:
    """
    Parse an arbitrary JSON block into a list of normalized intermediate records.

    The returned dictionaries are merged downstream into ``CandidateDecisionMatrixRow``
    instances.
    """
    if isinstance(raw_json_payload, str):
        try:
            parsed_json_payload = json.loads(raw_json_payload)
        except json.JSONDecodeError as json_decode_error:
            raise ValueError("Raw JSON candidate block is not valid JSON.") from json_decode_error
        return parse_raw_json_candidate_block(
            raw_json_payload=parsed_json_payload,
            source_provider_identifier=source_provider_identifier,
        )

    if isinstance(raw_json_payload, list):
        structured_candidate_records = raw_json_payload
    elif isinstance(raw_json_payload, dict):
        structured_candidate_records = None
        for collection_root_key in CANDIDATE_COLLECTION_ROOT_KEYS:
            candidate_collection_value = raw_json_payload.get(collection_root_key)
            if isinstance(candidate_collection_value, list):
                structured_candidate_records = candidate_collection_value
                break
        if structured_candidate_records is None:
            if _extract_first_matching_field(
                candidate_record=raw_json_payload,
                field_aliases=CANDIDATE_IDENTIFIER_FIELD_ALIASES,
            ) is not None or _extract_first_matching_field(
                candidate_record=raw_json_payload,
                field_aliases=RAW_COST_FIELD_ALIASES,
            ) is not None:
                structured_candidate_records = [raw_json_payload]
            else:
                raise ValueError(
                    "JSON payload does not contain a recognized candidate collection key."
                )
    else:
        raise ValueError("Raw JSON candidate block must be a dict, list, or JSON string.")

    normalized_intermediate_records: list[dict[str, Any]] = []

    for candidate_record_index, structured_candidate_record in enumerate(
        structured_candidate_records,
    ):
        if not isinstance(structured_candidate_record, dict):
            raise ValueError(
                f"Candidate record at index {candidate_record_index} must be a JSON object."
            )

        provisional_display_name = _resolve_candidate_display_name(
            candidate_record=structured_candidate_record,
            candidate_identifier=f"candidate_{candidate_record_index}",
        )
        candidate_identifier = _resolve_candidate_identifier(
            candidate_record=structured_candidate_record,
            fallback_display_name=provisional_display_name,
            provider_identifier=source_provider_identifier,
            candidate_record_index=candidate_record_index,
        )
        candidate_display_name = _resolve_candidate_display_name(
            candidate_record=structured_candidate_record,
            candidate_identifier=candidate_identifier,
        )

        raw_cost_field_value = _extract_first_matching_field(
            candidate_record=structured_candidate_record,
            field_aliases=RAW_COST_FIELD_ALIASES,
        )
        raw_distance_field_value = _extract_first_matching_field(
            candidate_record=structured_candidate_record,
            field_aliases=RAW_DISTANCE_FIELD_ALIASES,
        )
        observed_individual_quality_ratings = _extract_observed_quality_ratings(
            candidate_record=structured_candidate_record,
        )

        raw_cost_amount = (
            _coerce_positive_float(
                raw_numeric_value=raw_cost_field_value,
                field_description="cost value",
            )
            if raw_cost_field_value is not None
            else DEFAULT_MISSING_COST_AMOUNT
        )
        raw_distance_kilometers = (
            _coerce_positive_float(
                raw_numeric_value=raw_distance_field_value,
                field_description="distance value",
            )
            if raw_distance_field_value is not None
            else DEFAULT_MISSING_DISTANCE_KILOMETERS
        )

        if observed_individual_quality_ratings:
            observed_rating_count = len(observed_individual_quality_ratings)
            raw_average_rating = sum(observed_individual_quality_ratings) / observed_rating_count
        else:
            observed_rating_count = DEFAULT_MISSING_RATING_COUNT
            raw_average_rating = DEFAULT_MISSING_QUALITY_RATING

        normalized_intermediate_records.append(
            {
                "candidate_identifier": candidate_identifier,
                "candidate_display_name": candidate_display_name,
                "raw_cost_amount": raw_cost_amount,
                "raw_distance_kilometers": raw_distance_kilometers,
                "observed_individual_quality_ratings": observed_individual_quality_ratings,
                "observed_rating_count": observed_rating_count,
                "raw_average_rating": raw_average_rating,
                "cost_currency_code": structured_candidate_record.get("cost_currency_code"),
                "source_provider_identifiers": (
                    [source_provider_identifier] if source_provider_identifier else []
                ),
            }
        )

    return normalized_intermediate_records


def _merge_intermediate_candidate_records(
    existing_candidate_record: dict[str, Any],
    incoming_candidate_record: dict[str, Any],
) -> dict[str, Any]:
    """Merge duplicate candidate records while preserving provider attribution."""
    merged_source_provider_identifiers = list(
        existing_candidate_record.get("source_provider_identifiers", [])
    )
    for provider_identifier in incoming_candidate_record.get(
        "source_provider_identifiers",
        [],
    ):
        if provider_identifier not in merged_source_provider_identifiers:
            merged_source_provider_identifiers.append(provider_identifier)

    merged_quality_ratings = list(
        existing_candidate_record.get("observed_individual_quality_ratings", [])
    )
    merged_quality_ratings.extend(
        incoming_candidate_record.get("observed_individual_quality_ratings", [])
    )

    if merged_quality_ratings:
        merged_observed_rating_count = len(merged_quality_ratings)
        merged_raw_average_rating = sum(merged_quality_ratings) / merged_observed_rating_count
    else:
        merged_observed_rating_count = max(
            existing_candidate_record.get("observed_rating_count", 0),
            incoming_candidate_record.get("observed_rating_count", 0),
        )
        merged_raw_average_rating = (
            existing_candidate_record.get("raw_average_rating", DEFAULT_MISSING_QUALITY_RATING)
            + incoming_candidate_record.get("raw_average_rating", DEFAULT_MISSING_QUALITY_RATING)
        ) / 2.0

    return {
        "candidate_identifier": existing_candidate_record["candidate_identifier"],
        "candidate_display_name": existing_candidate_record["candidate_display_name"],
        "raw_cost_amount": min(
            existing_candidate_record["raw_cost_amount"],
            incoming_candidate_record["raw_cost_amount"],
        ),
        "raw_distance_kilometers": min(
            existing_candidate_record["raw_distance_kilometers"],
            incoming_candidate_record["raw_distance_kilometers"],
        ),
        "observed_individual_quality_ratings": merged_quality_ratings,
        "observed_rating_count": merged_observed_rating_count,
        "raw_average_rating": merged_raw_average_rating,
        "cost_currency_code": existing_candidate_record.get("cost_currency_code")
        or incoming_candidate_record.get("cost_currency_code"),
        "source_provider_identifiers": merged_source_provider_identifiers,
    }


def build_candidate_decision_matrix_from_intermediate_records(
    intermediate_candidate_records: list[dict[str, Any]],
) -> list[CandidateDecisionMatrixRow]:
    """Materialize intermediate parsed records into decision matrix rows."""
    if not intermediate_candidate_records:
        return []

    observed_cost_amounts = [
        float(record["raw_cost_amount"])
        for record in intermediate_candidate_records
        if record["raw_cost_amount"] is not None
    ]
    observed_distance_kilometers_values = [
        float(record["raw_distance_kilometers"])
        for record in intermediate_candidate_records
        if record["raw_distance_kilometers"] is not None
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

    for intermediate_candidate_record in intermediate_candidate_records:
        candidate_identifier = intermediate_candidate_record["candidate_identifier"]
        raw_cost_amount = float(intermediate_candidate_record["raw_cost_amount"])
        raw_distance_kilometers = float(intermediate_candidate_record["raw_distance_kilometers"])
        observed_rating_count = int(intermediate_candidate_record["observed_rating_count"])
        raw_average_rating = float(intermediate_candidate_record["raw_average_rating"])

        normalized_cost_score_value = _normalize_cost_to_unit_interval(
            raw_cost_amount=raw_cost_amount,
            minimum_observed_cost_amount=minimum_observed_cost_amount,
            maximum_observed_cost_amount=maximum_observed_cost_amount,
        )
        normalized_distance_score_value = _normalize_distance_to_unit_interval(
            raw_distance_kilometers=raw_distance_kilometers,
            minimum_observed_distance_kilometers=minimum_observed_distance_kilometers,
            maximum_observed_distance_kilometers=maximum_observed_distance_kilometers,
        )

        candidate_utility_metrics = CandidateUtilityMetrics(
            candidate_identifier=candidate_identifier,
            candidate_display_name=intermediate_candidate_record["candidate_display_name"],
            normalized_cost_score=NormalizedCostScore(
                candidate_identifier=candidate_identifier,
                normalized_cost_score=normalized_cost_score_value,
                raw_cost_amount=raw_cost_amount,
                cost_currency_code=intermediate_candidate_record.get("cost_currency_code"),
            ),
            normalized_distance_score=NormalizedDistanceScore(
                candidate_identifier=candidate_identifier,
                normalized_distance_score=normalized_distance_score_value,
                raw_distance_kilometers=raw_distance_kilometers,
            ),
            bayesian_quality_rating=BayesianQualityRating(
                candidate_identifier=candidate_identifier,
                bayesian_quality_rating=raw_average_rating,
                observed_rating_count=observed_rating_count,
                raw_average_rating=raw_average_rating,
            ),
        )

        candidate_decision_matrix_rows.append(
            CandidateDecisionMatrixRow(
                candidate_utility_metrics=candidate_utility_metrics,
                source_provider_identifiers=intermediate_candidate_record.get(
                    "source_provider_identifiers",
                    [],
                ),
            )
        )

    return candidate_decision_matrix_rows


def build_candidate_decision_matrix_from_provider_payloads(
    successful_provider_response_payloads: list[ProviderResponsePayload],
) -> list[CandidateDecisionMatrixRow]:
    """
    Normalize successful provider payloads into a unified candidate decision matrix.

    Candidates sharing the same ``candidate_identifier`` are merged across providers.
    """
    aggregated_candidate_records: dict[str, dict[str, Any]] = {}

    for provider_response_payload in successful_provider_response_payloads:
        try:
            parsed_provider_json_payload = json.loads(
                provider_response_payload.raw_recommendation_text,
            )
        except json.JSONDecodeError as json_decode_error:
            raise ValueError(
                f"Provider '{provider_response_payload.provider_identifier}' returned "
                "non-JSON recommendation text."
            ) from json_decode_error

        intermediate_candidate_records = parse_raw_json_candidate_block(
            raw_json_payload=parsed_provider_json_payload,
            source_provider_identifier=provider_response_payload.provider_identifier,
        )

        for intermediate_candidate_record in intermediate_candidate_records:
            candidate_identifier = intermediate_candidate_record["candidate_identifier"]
            if candidate_identifier not in aggregated_candidate_records:
                aggregated_candidate_records[candidate_identifier] = intermediate_candidate_record
            else:
                aggregated_candidate_records[candidate_identifier] = (
                    _merge_intermediate_candidate_records(
                        existing_candidate_record=aggregated_candidate_records[
                            candidate_identifier
                        ],
                        incoming_candidate_record=intermediate_candidate_record,
                    )
                )

    return build_candidate_decision_matrix_from_intermediate_records(
        intermediate_candidate_records=list(aggregated_candidate_records.values()),
    )


def build_candidate_decision_matrix_from_raw_json(
    raw_json_payload: dict[str, Any] | list[Any] | str,
    *,
    source_provider_identifier: str | None = None,
) -> list[CandidateDecisionMatrixRow]:
    """Parse a standalone JSON block into ranked-ready decision matrix rows."""
    intermediate_candidate_records = parse_raw_json_candidate_block(
        raw_json_payload=raw_json_payload,
        source_provider_identifier=source_provider_identifier,
    )
    return build_candidate_decision_matrix_from_intermediate_records(
        intermediate_candidate_records=intermediate_candidate_records,
    )
