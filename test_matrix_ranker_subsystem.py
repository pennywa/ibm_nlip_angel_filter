"""
Isolated verification gate for the matrix_ranker subsystem.

Validates Bayesian volume smoothing, zero-variance Z-score guards, flexible
JSON field aliasing, and user-preference-weight-driven ranking order.

Run from the repository root:
    py -3 test_matrix_ranker_subsystem.py
"""

from __future__ import annotations

import math
import sys
from typing import Any

from matrix_ranker.matrix_normalization_service import (
    build_candidate_decision_matrix_from_raw_json,
    parse_raw_json_candidate_block,
)
from matrix_ranker.ranking_engine import (
    DEFAULT_GLOBAL_SMOOTHING_CONSTANT,
    MINIMUM_DIMENSION_STANDARD_DEVIATION,
    CandidateRankingEngine,
)

BAYESIAN_GUARD_GLOBAL_PRIOR_MEAN_RATING: float = 4.0
LOW_SAMPLE_OBSERVED_RATINGS: list[float] = [5.0]
HIGH_VOLUME_OBSERVED_RATINGS: list[float] = [4.7] * 20


class _VerificationGateFailure(Exception):
    """Raised when an isolated verification gate does not pass."""


def _assert_finite_numeric_value(
    numeric_value: float | None,
    value_description: str,
) -> None:
    """Assert that a computed score is a finite real number."""
    if numeric_value is None:
        raise _VerificationGateFailure(
            f"{value_description} was None; expected a finite float."
        )
    if not math.isfinite(numeric_value):
        raise _VerificationGateFailure(
            f"{value_description} was {numeric_value!r}; expected a finite float."
        )


def _run_bayesian_smoothing_guard_check() -> None:
    """
    Gate 1: volume-weighted Bayesian smoothing must favor high-sample ratings.

    A single 5.0 review must NOT outrank twenty 4.7 reviews after smoothing.
    """
    print("Gate 1: Bayesian Smoothing Guard Check")

    candidate_ranking_engine = CandidateRankingEngine(
        global_smoothing_constant=DEFAULT_GLOBAL_SMOOTHING_CONSTANT,
    )

    (
        low_sample_bayesian_quality_rating,
        low_sample_observed_rating_count,
    ) = candidate_ranking_engine.compute_bayesian_average_rating(
        global_smoothing_constant_c=DEFAULT_GLOBAL_SMOOTHING_CONSTANT,
        global_prior_mean_rating_m=BAYESIAN_GUARD_GLOBAL_PRIOR_MEAN_RATING,
        observed_individual_ratings=LOW_SAMPLE_OBSERVED_RATINGS,
    )
    (
        high_volume_bayesian_quality_rating,
        high_volume_observed_rating_count,
    ) = candidate_ranking_engine.compute_bayesian_average_rating(
        global_smoothing_constant_c=DEFAULT_GLOBAL_SMOOTHING_CONSTANT,
        global_prior_mean_rating_m=BAYESIAN_GUARD_GLOBAL_PRIOR_MEAN_RATING,
        observed_individual_ratings=HIGH_VOLUME_OBSERVED_RATINGS,
    )

    if low_sample_observed_rating_count != 1:
        raise _VerificationGateFailure(
            "Low-sample guard fixture must contain exactly one observed rating."
        )
    if high_volume_observed_rating_count != 20:
        raise _VerificationGateFailure(
            "High-volume guard fixture must contain exactly twenty observed ratings."
        )
    if high_volume_bayesian_quality_rating <= low_sample_bayesian_quality_rating:
        raise _VerificationGateFailure(
            "Bayesian smoothing failed to rank the high-volume candidate higher: "
            f"high_volume={high_volume_bayesian_quality_rating:.6f}, "
            f"low_sample={low_sample_bayesian_quality_rating:.6f}."
        )

    bayesian_guard_candidate_payload: dict[str, Any] = {
        "comparison_shopping_candidates": [
            {
                "candidate_identifier": "low_sample_single_five_star",
                "raw_cost_value": 10.0,
                "raw_distance_value": 1.0,
                "raw_quality_rating_scores": LOW_SAMPLE_OBSERVED_RATINGS,
            },
            {
                "candidate_identifier": "high_volume_twenty_four_point_seven",
                "raw_cost_value": 10.0,
                "raw_distance_value": 1.0,
                "raw_quality_rating_scores": HIGH_VOLUME_OBSERVED_RATINGS,
            },
        ]
    }
    bayesian_guard_matrix_rows = build_candidate_decision_matrix_from_raw_json(
        raw_json_payload=bayesian_guard_candidate_payload,
    )
    candidate_utility_metrics_list = [
        matrix_row.candidate_utility_metrics for matrix_row in bayesian_guard_matrix_rows
    ]
    smoothed_candidate_utility_metrics_list = (
        candidate_ranking_engine.apply_bayesian_quality_smoothing(
            candidate_utility_metrics_list=candidate_utility_metrics_list,
            global_prior_mean_rating_m=BAYESIAN_GUARD_GLOBAL_PRIOR_MEAN_RATING,
        )
    )

    smoothed_rating_by_candidate_identifier = {
        candidate_utility_metrics.candidate_identifier: (
            candidate_utility_metrics.bayesian_quality_rating.bayesian_quality_rating
        )
        for candidate_utility_metrics in smoothed_candidate_utility_metrics_list
    }

    if (
        smoothed_rating_by_candidate_identifier["high_volume_twenty_four_point_seven"]
        <= smoothed_rating_by_candidate_identifier["low_sample_single_five_star"]
    ):
        raise _VerificationGateFailure(
            "Matrix-level Bayesian smoothing did not favor the high-volume candidate."
        )

    print(
        "  PASS  low_sample="
        f"{low_sample_bayesian_quality_rating:.4f}, "
        f"high_volume={high_volume_bayesian_quality_rating:.4f}"
    )


def _run_zero_variance_vector_guard_check() -> None:
    """
    Gate 2: identical cost or distance values must not crash Z-score normalization.

    Verifies the MINIMUM_DIMENSION_STANDARD_DEVIATION floor and finite distances.
    """
    print("Gate 2: Zero-Variance Vector Guard Check")

    candidate_ranking_engine = CandidateRankingEngine(
        global_smoothing_constant=DEFAULT_GLOBAL_SMOOTHING_CONSTANT,
    )

    identical_cost_feature_values = [50.0, 50.0, 50.0]
    identical_distance_feature_values = [2.5, 2.5, 2.5]

    identical_cost_sample_standard_deviation = (
        candidate_ranking_engine.compute_sample_standard_deviation(
            dimension_feature_values=identical_cost_feature_values,
        )
    )
    identical_distance_sample_standard_deviation = (
        candidate_ranking_engine.compute_sample_standard_deviation(
            dimension_feature_values=identical_distance_feature_values,
        )
    )

    if not math.isclose(
        identical_cost_sample_standard_deviation,
        MINIMUM_DIMENSION_STANDARD_DEVIATION,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise _VerificationGateFailure(
            "Identical cost batch must floor to MINIMUM_DIMENSION_STANDARD_DEVIATION; "
            f"received {identical_cost_sample_standard_deviation}."
        )
    if not math.isclose(
        identical_distance_sample_standard_deviation,
        MINIMUM_DIMENSION_STANDARD_DEVIATION,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise _VerificationGateFailure(
            "Identical distance batch must floor to MINIMUM_DIMENSION_STANDARD_DEVIATION; "
            f"received {identical_distance_sample_standard_deviation}."
        )

    zero_variance_cost_payload: dict[str, Any] = {
        "comparison_shopping_candidates": [
            {
                "candidate_identifier": "uniform_cost_near",
                "raw_cost_value": 50.0,
                "raw_distance_value": 1.0,
                "raw_quality_rating_scores": [4.0],
            },
            {
                "candidate_identifier": "uniform_cost_mid",
                "raw_cost_value": 50.0,
                "raw_distance_value": 2.0,
                "raw_quality_rating_scores": [4.5],
            },
            {
                "candidate_identifier": "uniform_cost_far",
                "raw_cost_value": 50.0,
                "raw_distance_value": 3.0,
                "raw_quality_rating_scores": [3.5],
            },
        ]
    }
    zero_variance_distance_payload: dict[str, Any] = {
        "comparison_shopping_candidates": [
            {
                "candidate_identifier": "uniform_distance_cheap",
                "raw_cost_value": 10.0,
                "raw_distance_value": 2.5,
                "raw_quality_rating_scores": [4.0],
            },
            {
                "candidate_identifier": "uniform_distance_mid",
                "raw_cost_value": 20.0,
                "raw_distance_value": 2.5,
                "raw_quality_rating_scores": [4.2],
            },
            {
                "candidate_identifier": "uniform_distance_premium",
                "raw_cost_value": 30.0,
                "raw_distance_value": 2.5,
                "raw_quality_rating_scores": [4.4],
            },
        ]
    }

    zero_variance_cost_matrix_rows = build_candidate_decision_matrix_from_raw_json(
        raw_json_payload=zero_variance_cost_payload,
    )
    zero_variance_distance_matrix_rows = build_candidate_decision_matrix_from_raw_json(
        raw_json_payload=zero_variance_distance_payload,
    )

    zero_variance_cost_ranking_result = candidate_ranking_engine.rank_candidate_decision_matrix(
        candidate_decision_matrix_rows=zero_variance_cost_matrix_rows,
        user_preference_weight_cost=0.33,
        user_preference_weight_distance=0.33,
        user_preference_weight_quality=0.34,
    )
    zero_variance_distance_ranking_result = (
        candidate_ranking_engine.rank_candidate_decision_matrix(
            candidate_decision_matrix_rows=zero_variance_distance_matrix_rows,
            user_preference_weight_cost=0.33,
            user_preference_weight_distance=0.33,
            user_preference_weight_quality=0.34,
        )
    )

    for matrix_row in (
        zero_variance_cost_ranking_result.ranked_candidate_decision_matrix_rows
        + zero_variance_distance_ranking_result.ranked_candidate_decision_matrix_rows
    ):
        _assert_finite_numeric_value(
            numeric_value=matrix_row.weighted_vector_distance_score,
            value_description=(
                "weighted_vector_distance_score for "
                f"{matrix_row.candidate_utility_metrics.candidate_identifier}"
            ),
        )

    print(
        "  PASS  MINIMUM_DIMENSION_STANDARD_DEVIATION floor active; "
        "all weighted_vector_distance_score values finite"
    )


def _run_flexible_aliasing_check() -> None:
    """
    Gate 3: diverse product-domain keys must normalize into uniform matrix rows.
    """
    print("Gate 3: Flexible Aliasing Check")

    electronics_alias_payload: dict[str, Any] = {
        "items": [
            {
                "id": "wireless_mouse_a",
                "name": "Ergo Wireless Mouse A",
                "price": 29.99,
                "distance": 4.2,
                "ratings": [4.6, 4.4, 4.5],
            },
            {
                "candidate_identifier": "mechanical_keyboard_b",
                "display_name": "Mechanical Keyboard B",
                "cost": 89.0,
                "distance_kilometers": 6.1,
                "quality_rating_scores": [4.8, 4.7],
            },
        ]
    }

    parsed_intermediate_records = parse_raw_json_candidate_block(
        raw_json_payload=electronics_alias_payload,
    )
    electronics_matrix_rows = build_candidate_decision_matrix_from_raw_json(
        raw_json_payload=electronics_alias_payload,
    )

    if len(parsed_intermediate_records) != 2:
        raise _VerificationGateFailure(
            f"Expected 2 aliased product records, received {len(parsed_intermediate_records)}."
        )
    if len(electronics_matrix_rows) != 2:
        raise _VerificationGateFailure(
            f"Expected 2 decision matrix rows, received {len(electronics_matrix_rows)}."
        )

    matrix_row_by_candidate_identifier = {
        matrix_row.candidate_utility_metrics.candidate_identifier: matrix_row
        for matrix_row in electronics_matrix_rows
    }

    wireless_mouse_row = matrix_row_by_candidate_identifier["wireless_mouse_a"]
    mechanical_keyboard_row = matrix_row_by_candidate_identifier["mechanical_keyboard_b"]

    if wireless_mouse_row.candidate_utility_metrics.candidate_display_name != (
        "Ergo Wireless Mouse A"
    ):
        raise _VerificationGateFailure(
            "Aliased 'name' field was not mapped to candidate_display_name."
        )
    if not math.isclose(
        wireless_mouse_row.candidate_utility_metrics.normalized_cost_score.raw_cost_amount,
        29.99,
        rel_tol=1e-9,
    ):
        raise _VerificationGateFailure("Aliased 'price' field was not mapped to raw_cost_amount.")
    if not math.isclose(
        wireless_mouse_row.candidate_utility_metrics.normalized_distance_score.raw_distance_kilometers,
        4.2,
        rel_tol=1e-9,
    ):
        raise _VerificationGateFailure(
            "Aliased 'distance' field was not mapped to raw_distance_kilometers."
        )
    if (
        wireless_mouse_row.candidate_utility_metrics.bayesian_quality_rating.observed_rating_count
        != 3
    ):
        raise _VerificationGateFailure("Aliased 'ratings' array length was not preserved.")

    if not math.isclose(
        mechanical_keyboard_row.candidate_utility_metrics.normalized_cost_score.raw_cost_amount,
        89.0,
        rel_tol=1e-9,
    ):
        raise _VerificationGateFailure("Aliased 'cost' field was not mapped to raw_cost_amount.")

    print("  PASS  electronics alias payload normalized into uniform matrix rows")


def _run_weighted_distance_sorting_check() -> None:
    """
    Gate 4: ranking order must shift when user preference weights change.
    """
    print("Gate 4: Weighted Distance Sorting Check")

    candidate_ranking_engine = CandidateRankingEngine(
        global_smoothing_constant=DEFAULT_GLOBAL_SMOOTHING_CONSTANT,
        global_prior_mean_rating=BAYESIAN_GUARD_GLOBAL_PRIOR_MEAN_RATING,
    )

    tradeoff_candidate_payload: dict[str, Any] = {
        "comparison_shopping_candidates": [
            {
                "candidate_identifier": "budget_option",
                "raw_cost_value": 200.0,
                "raw_distance_value": 5.0,
                "raw_quality_rating_scores": [3.0, 3.5, 3.0],
            },
            {
                "candidate_identifier": "premium_option",
                "raw_cost_value": 1200.0,
                "raw_distance_value": 5.0,
                "raw_quality_rating_scores": [4.8, 4.9, 4.7, 4.8, 4.9],
            },
        ]
    }

    tradeoff_matrix_rows = build_candidate_decision_matrix_from_raw_json(
        raw_json_payload=tradeoff_candidate_payload,
    )

    cost_prioritized_ranking_result = candidate_ranking_engine.rank_candidate_decision_matrix(
        candidate_decision_matrix_rows=tradeoff_matrix_rows,
        user_preference_weight_cost=0.70,
        user_preference_weight_distance=0.15,
        user_preference_weight_quality=0.15,
    )
    quality_prioritized_ranking_result = candidate_ranking_engine.rank_candidate_decision_matrix(
        candidate_decision_matrix_rows=tradeoff_matrix_rows,
        user_preference_weight_cost=0.15,
        user_preference_weight_distance=0.15,
        user_preference_weight_quality=0.70,
    )

    cost_prioritized_rank_order = [
        matrix_row.candidate_utility_metrics.candidate_identifier
        for matrix_row in cost_prioritized_ranking_result.ranked_candidate_decision_matrix_rows
    ]
    quality_prioritized_rank_order = [
        matrix_row.candidate_utility_metrics.candidate_identifier
        for matrix_row in quality_prioritized_ranking_result.ranked_candidate_decision_matrix_rows
    ]

    if cost_prioritized_rank_order[0] != "budget_option":
        raise _VerificationGateFailure(
            "Cost-maximized weights did not rank budget_option first: "
            f"{cost_prioritized_rank_order}."
        )
    if quality_prioritized_rank_order[0] != "premium_option":
        raise _VerificationGateFailure(
            "Quality-maximized weights did not rank premium_option first: "
            f"{quality_prioritized_rank_order}."
        )
    if cost_prioritized_rank_order == quality_prioritized_rank_order:
        raise _VerificationGateFailure(
            "Ranking order did not change when user preference weights shifted."
        )

    print(
        "  PASS  cost-first="
        f"{cost_prioritized_rank_order}, quality-first={quality_prioritized_rank_order}"
    )


def main() -> int:
    """Execute all matrix_ranker verification gates."""
    print("=" * 72)
    print("matrix_ranker subsystem verification gate")
    print("=" * 72)

    verification_gates = (
        _run_bayesian_smoothing_guard_check,
        _run_zero_variance_vector_guard_check,
        _run_flexible_aliasing_check,
        _run_weighted_distance_sorting_check,
    )

    try:
        for verification_gate in verification_gates:
            verification_gate()
            print()
    except _VerificationGateFailure as verification_gate_failure:
        print(f"\nFAILED  {verification_gate_failure}")
        return 1
    except Exception as unexpected_error:
        print(f"\nFAILED  Unexpected error: {unexpected_error}")
        raise

    print("=" * 72)
    print("ALL GATES PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
