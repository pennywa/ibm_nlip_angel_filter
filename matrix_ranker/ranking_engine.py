"""
Mathematical ranking engine for multi-criteria candidate optimization.

Implements Bayesian quality smoothing, batch Z-score normalization with
directionality inversion, and weighted vector distance ranking.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

from matrix_ranker.candidate_utility_metric_schemas import (
    BayesianQualityRating,
    CandidateDecisionMatrixRow,
    CandidateUtilityMetrics,
)

DEFAULT_GLOBAL_SMOOTHING_CONSTANT: float = 10.0
DEFAULT_GLOBAL_PRIOR_MEAN_RATING: float = 3.0
MINIMUM_DIMENSION_STANDARD_DEVIATION: float = 1e-9


class UserPreferenceWeightProfile(BaseModel):
    """User preference weights for Cost, Distance, and Bayesian Quality dimensions."""

    user_preference_weight_cost: float = Field(
        default=0.33,
        ge=0.0,
        le=1.0,
        description="User weight for the Cost utility dimension.",
    )
    user_preference_weight_distance: float = Field(
        default=0.33,
        ge=0.0,
        le=1.0,
        description="User weight for the Distance utility dimension.",
    )
    user_preference_weight_quality: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description="User weight for the Bayesian Quality utility dimension.",
    )

    @model_validator(mode="after")
    def validate_preference_weights_sum_to_one(self) -> UserPreferenceWeightProfile:
        """Ensure preference weights form a valid convex combination."""
        total_preference_weight = (
            self.user_preference_weight_cost
            + self.user_preference_weight_distance
            + self.user_preference_weight_quality
        )
        if not math.isclose(total_preference_weight, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "User preference weights must sum to 1.0 across cost, distance, and quality."
            )
        return self


class CandidateRankingResult(BaseModel):
    """Complete output from the ranking engine for a candidate decision matrix."""

    ranked_candidate_decision_matrix_rows: list[CandidateDecisionMatrixRow] = Field(
        default_factory=list,
        description="Candidates ordered by ascending weighted vector distance.",
    )
    global_prior_mean_rating: float = Field(
        ...,
        description="Dataset prior mean (m) used during Bayesian smoothing.",
    )
    global_smoothing_constant: float = Field(
        ...,
        description="Confidence threshold (C) used during Bayesian smoothing.",
    )


class CandidateRankingEngine:
    """
    Multi-criteria optimization engine for user-aligned candidate ranking.

  Pipeline:
      1. Bayesian quality smoothing from observed review arrays.
      2. Batch Z-score normalization with directionality inversion.
      3. Weighted Euclidean distance to the ideal oriented target vector.
      4. Ascending sort by weighted_vector_distance_score (Rank #1 = closest).
    """

    def __init__(
        self,
        global_smoothing_constant: float = DEFAULT_GLOBAL_SMOOTHING_CONSTANT,
        global_prior_mean_rating: float | None = None,
    ) -> None:
        self._global_smoothing_constant = global_smoothing_constant
        self._global_prior_mean_rating = global_prior_mean_rating

    @staticmethod
    def compute_bayesian_average_rating(
        global_smoothing_constant_c: float,
        global_prior_mean_rating_m: float,
        observed_individual_ratings: list[float],
    ) -> tuple[float, int]:
        """
        Compute Bayesian-adjusted quality rating for one candidate.

        Formula:
            bayesian_quality_rating = (C * m + sum(r_k)) / (C + n)
        """
        observed_rating_count_n = len(observed_individual_ratings)
        sum_observed_ratings = sum(observed_individual_ratings)

        if observed_rating_count_n == 0:
            return global_prior_mean_rating_m, 0

        bayesian_quality_rating = (
            global_smoothing_constant_c * global_prior_mean_rating_m
            + sum_observed_ratings
        ) / (global_smoothing_constant_c + observed_rating_count_n)

        return bayesian_quality_rating, observed_rating_count_n

    @staticmethod
    def compute_bayesian_average_from_aggregate_ratings(
        global_smoothing_constant_c: float,
        global_prior_mean_rating_m: float,
        raw_average_rating: float,
        observed_rating_count_n: int,
    ) -> float:
        """Compute Bayesian average when only aggregate mean and count are available."""
        if observed_rating_count_n <= 0:
            return global_prior_mean_rating_m

        sum_observed_ratings = raw_average_rating * observed_rating_count_n
        return (
            global_smoothing_constant_c * global_prior_mean_rating_m
            + sum_observed_ratings
        ) / (global_smoothing_constant_c + observed_rating_count_n)

    @staticmethod
    def compute_dataset_prior_mean_rating(
        candidate_utility_metrics_list: list[CandidateUtilityMetrics],
    ) -> float:
        """Compute the global prior mean (m) across all candidates in the matrix."""
        raw_average_rating_values: list[float] = []

        for candidate_utility_metrics in candidate_utility_metrics_list:
            bayesian_quality_rating_record = (
                candidate_utility_metrics.bayesian_quality_rating
            )
            if bayesian_quality_rating_record.raw_average_rating is not None:
                raw_average_rating_values.append(
                    bayesian_quality_rating_record.raw_average_rating
                )
            else:
                raw_average_rating_values.append(
                    bayesian_quality_rating_record.bayesian_quality_rating
                )

        if not raw_average_rating_values:
            return DEFAULT_GLOBAL_PRIOR_MEAN_RATING

        return sum(raw_average_rating_values) / len(raw_average_rating_values)

    def apply_bayesian_quality_smoothing(
        self,
        candidate_utility_metrics_list: list[CandidateUtilityMetrics],
        global_prior_mean_rating_m: float | None = None,
    ) -> list[CandidateUtilityMetrics]:
        """Apply Bayesian smoothing to every candidate's quality dimension."""
        resolved_global_prior_mean_rating_m = (
            global_prior_mean_rating_m
            if global_prior_mean_rating_m is not None
            else self._global_prior_mean_rating
            if self._global_prior_mean_rating is not None
            else self.compute_dataset_prior_mean_rating(
                candidate_utility_metrics_list=candidate_utility_metrics_list,
            )
        )

        smoothed_candidate_utility_metrics_list: list[CandidateUtilityMetrics] = []

        for candidate_utility_metrics in candidate_utility_metrics_list:
            existing_bayesian_quality_rating = (
                candidate_utility_metrics.bayesian_quality_rating
            )

            if existing_bayesian_quality_rating.observed_rating_count > 0:
                smoothed_bayesian_quality_rating_value = (
                    self.compute_bayesian_average_from_aggregate_ratings(
                        global_smoothing_constant_c=self._global_smoothing_constant,
                        global_prior_mean_rating_m=resolved_global_prior_mean_rating_m,
                        raw_average_rating=(
                            existing_bayesian_quality_rating.raw_average_rating
                            if existing_bayesian_quality_rating.raw_average_rating is not None
                            else existing_bayesian_quality_rating.bayesian_quality_rating
                        ),
                        observed_rating_count_n=(
                            existing_bayesian_quality_rating.observed_rating_count
                        ),
                    )
                )
            else:
                smoothed_bayesian_quality_rating_value = resolved_global_prior_mean_rating_m

            updated_bayesian_quality_rating = BayesianQualityRating(
                candidate_identifier=candidate_utility_metrics.candidate_identifier,
                bayesian_quality_rating=smoothed_bayesian_quality_rating_value,
                observed_rating_count=existing_bayesian_quality_rating.observed_rating_count,
                raw_average_rating=existing_bayesian_quality_rating.raw_average_rating,
                global_prior_mean_rating=resolved_global_prior_mean_rating_m,
                confidence_threshold_rating_count=int(self._global_smoothing_constant),
            )

            smoothed_candidate_utility_metrics_list.append(
                candidate_utility_metrics.model_copy(
                    update={"bayesian_quality_rating": updated_bayesian_quality_rating},
                )
            )

        return smoothed_candidate_utility_metrics_list

    @staticmethod
    def _extract_cost_feature_value(
        candidate_utility_metrics: CandidateUtilityMetrics,
    ) -> float:
        """Extract the raw cost feature used in vector distance calculations."""
        if candidate_utility_metrics.normalized_cost_score.raw_cost_amount is not None:
            return candidate_utility_metrics.normalized_cost_score.raw_cost_amount
        return candidate_utility_metrics.normalized_cost_score.normalized_cost_score

    @staticmethod
    def _extract_distance_feature_value(
        candidate_utility_metrics: CandidateUtilityMetrics,
    ) -> float:
        """Extract the raw distance feature used in vector distance calculations."""
        if (
            candidate_utility_metrics.normalized_distance_score.raw_distance_kilometers
            is not None
        ):
            return (
                candidate_utility_metrics.normalized_distance_score.raw_distance_kilometers
            )
        return candidate_utility_metrics.normalized_distance_score.normalized_distance_score

    @staticmethod
    def _extract_quality_feature_value(
        candidate_utility_metrics: CandidateUtilityMetrics,
    ) -> float:
        """Extract the Bayesian quality feature used in vector distance calculations."""
        return candidate_utility_metrics.bayesian_quality_rating.bayesian_quality_rating

    @staticmethod
    def compute_batch_mean_value(dimension_feature_values: list[float]) -> float:
        """Compute the arithmetic mean of a utility dimension across the candidate batch."""
        if not dimension_feature_values:
            return 0.0
        return sum(dimension_feature_values) / len(dimension_feature_values)

    @staticmethod
    def compute_sample_standard_deviation(
        dimension_feature_values: list[float],
    ) -> float:
        """
        Compute sample standard deviation (n-1 denominator) for a utility dimension.

        Returns a minimum floor to prevent division-by-zero during Z-score scaling.
        """
        candidate_count = len(dimension_feature_values)
        if candidate_count <= 1:
            return 1.0

        dimension_batch_mean_value = sum(dimension_feature_values) / candidate_count
        sum_of_squared_deviations = sum(
            (feature_value - dimension_batch_mean_value) ** 2
            for feature_value in dimension_feature_values
        )
        sample_variance = sum_of_squared_deviations / (candidate_count - 1)
        sample_standard_deviation = math.sqrt(sample_variance)

        return max(sample_standard_deviation, MINIMUM_DIMENSION_STANDARD_DEVIATION)

    @staticmethod
    def compute_statistical_z_score(
        raw_feature_value: float,
        dimension_batch_mean_value: float,
        dimension_sample_standard_deviation: float,
        lower_raw_value_is_preferred: bool,
    ) -> float:
        """
        Normalize a raw feature into an oriented statistical Z-score.

        Lower-is-better dimensions (cost, distance) are inverted so that
        higher oriented Z-scores always indicate better user utility.
        """
        if not math.isfinite(raw_feature_value):
            return 0.0

        protected_standard_deviation = max(
            dimension_sample_standard_deviation,
            MINIMUM_DIMENSION_STANDARD_DEVIATION,
        )
        statistical_z_score = (
            raw_feature_value - dimension_batch_mean_value
        ) / protected_standard_deviation

        if lower_raw_value_is_preferred:
            return -statistical_z_score
        return statistical_z_score

    def compute_weighted_vector_distance_score(
        self,
        oriented_cost_z_score: float,
        oriented_distance_z_score: float,
        oriented_quality_z_score: float,
        ideal_oriented_cost_z_score: float,
        ideal_oriented_distance_z_score: float,
        ideal_oriented_quality_z_score: float,
        user_preference_weight_cost: float,
        user_preference_weight_distance: float,
        user_preference_weight_quality: float,
    ) -> float:
        """
        Compute weighted Euclidean distance to the ideal oriented target vector.

        Formula:
            weighted_vector_distance_score = sqrt(
                w_cost   * (z_cost   - z_cost*)^2 +
                w_dist   * (z_dist   - z_dist*)^2 +
                w_quality* (z_quality- z_quality*)^2
            )
        """
        weighted_sum_of_squared_deviations = (
            user_preference_weight_cost
            * (oriented_cost_z_score - ideal_oriented_cost_z_score) ** 2
            + user_preference_weight_distance
            * (oriented_distance_z_score - ideal_oriented_distance_z_score) ** 2
            + user_preference_weight_quality
            * (oriented_quality_z_score - ideal_oriented_quality_z_score) ** 2
        )

        return math.sqrt(weighted_sum_of_squared_deviations)

    def rank_candidate_decision_matrix(
        self,
        candidate_decision_matrix_rows: list[CandidateDecisionMatrixRow],
        user_preference_weight_cost: float = 0.33,
        user_preference_weight_distance: float = 0.33,
        user_preference_weight_quality: float = 0.34,
    ) -> CandidateRankingResult:
        """
        Rank candidates by minimizing weighted vector distance to user utility.

        Pipeline:
            1. Apply Bayesian quality smoothing across the matrix.
            2. Z-score normalize Cost, Distance, and Quality with direction inversion.
            3. Compute weighted_vector_distance_score for each candidate.
            4. Sort ascending (lower distance = Rank #1).
        """
        user_preference_weight_profile = UserPreferenceWeightProfile(
            user_preference_weight_cost=user_preference_weight_cost,
            user_preference_weight_distance=user_preference_weight_distance,
            user_preference_weight_quality=user_preference_weight_quality,
        )

        if not candidate_decision_matrix_rows:
            return CandidateRankingResult(
                ranked_candidate_decision_matrix_rows=[],
                global_prior_mean_rating=DEFAULT_GLOBAL_PRIOR_MEAN_RATING,
                global_smoothing_constant=self._global_smoothing_constant,
            )

        candidate_utility_metrics_list = [
            matrix_row.candidate_utility_metrics
            for matrix_row in candidate_decision_matrix_rows
        ]

        global_prior_mean_rating_m = self.compute_dataset_prior_mean_rating(
            candidate_utility_metrics_list=candidate_utility_metrics_list,
        )

        smoothed_candidate_utility_metrics_list = self.apply_bayesian_quality_smoothing(
            candidate_utility_metrics_list=candidate_utility_metrics_list,
            global_prior_mean_rating_m=global_prior_mean_rating_m,
        )

        cost_feature_values = [
            self._extract_cost_feature_value(candidate_utility_metrics)
            for candidate_utility_metrics in smoothed_candidate_utility_metrics_list
        ]
        distance_feature_values = [
            self._extract_distance_feature_value(candidate_utility_metrics)
            for candidate_utility_metrics in smoothed_candidate_utility_metrics_list
        ]
        quality_feature_values = [
            self._extract_quality_feature_value(candidate_utility_metrics)
            for candidate_utility_metrics in smoothed_candidate_utility_metrics_list
        ]

        cost_batch_mean_value = self.compute_batch_mean_value(
            dimension_feature_values=cost_feature_values,
        )
        distance_batch_mean_value = self.compute_batch_mean_value(
            dimension_feature_values=distance_feature_values,
        )
        quality_batch_mean_value = self.compute_batch_mean_value(
            dimension_feature_values=quality_feature_values,
        )

        cost_sample_standard_deviation = self.compute_sample_standard_deviation(
            dimension_feature_values=cost_feature_values,
        )
        distance_sample_standard_deviation = self.compute_sample_standard_deviation(
            dimension_feature_values=distance_feature_values,
        )
        quality_sample_standard_deviation = self.compute_sample_standard_deviation(
            dimension_feature_values=quality_feature_values,
        )

        oriented_cost_z_scores: list[float] = []
        oriented_distance_z_scores: list[float] = []
        oriented_quality_z_scores: list[float] = []

        for matrix_row_index in range(len(smoothed_candidate_utility_metrics_list)):
            oriented_cost_z_scores.append(
                self.compute_statistical_z_score(
                    raw_feature_value=cost_feature_values[matrix_row_index],
                    dimension_batch_mean_value=cost_batch_mean_value,
                    dimension_sample_standard_deviation=cost_sample_standard_deviation,
                    lower_raw_value_is_preferred=True,
                )
            )
            oriented_distance_z_scores.append(
                self.compute_statistical_z_score(
                    raw_feature_value=distance_feature_values[matrix_row_index],
                    dimension_batch_mean_value=distance_batch_mean_value,
                    dimension_sample_standard_deviation=distance_sample_standard_deviation,
                    lower_raw_value_is_preferred=True,
                )
            )
            oriented_quality_z_scores.append(
                self.compute_statistical_z_score(
                    raw_feature_value=quality_feature_values[matrix_row_index],
                    dimension_batch_mean_value=quality_batch_mean_value,
                    dimension_sample_standard_deviation=quality_sample_standard_deviation,
                    lower_raw_value_is_preferred=False,
                )
            )

        ideal_oriented_cost_z_score = max(oriented_cost_z_scores)
        ideal_oriented_distance_z_score = max(oriented_distance_z_scores)
        ideal_oriented_quality_z_score = max(oriented_quality_z_scores)

        scored_candidate_rows: list[tuple[CandidateDecisionMatrixRow, float]] = []

        for matrix_row_index, matrix_row in enumerate(candidate_decision_matrix_rows):
            smoothed_candidate_utility_metrics = smoothed_candidate_utility_metrics_list[
                matrix_row_index
            ]

            weighted_vector_distance_score = self.compute_weighted_vector_distance_score(
                oriented_cost_z_score=oriented_cost_z_scores[matrix_row_index],
                oriented_distance_z_score=oriented_distance_z_scores[matrix_row_index],
                oriented_quality_z_score=oriented_quality_z_scores[matrix_row_index],
                ideal_oriented_cost_z_score=ideal_oriented_cost_z_score,
                ideal_oriented_distance_z_score=ideal_oriented_distance_z_score,
                ideal_oriented_quality_z_score=ideal_oriented_quality_z_score,
                user_preference_weight_cost=(
                    user_preference_weight_profile.user_preference_weight_cost
                ),
                user_preference_weight_distance=(
                    user_preference_weight_profile.user_preference_weight_distance
                ),
                user_preference_weight_quality=(
                    user_preference_weight_profile.user_preference_weight_quality
                ),
            )

            updated_matrix_row = matrix_row.model_copy(
                update={
                    "candidate_utility_metrics": smoothed_candidate_utility_metrics,
                    "weighted_vector_distance_score": weighted_vector_distance_score,
                },
            )
            scored_candidate_rows.append(
                (updated_matrix_row, weighted_vector_distance_score),
            )

        scored_candidate_rows.sort(key=lambda scored_row: scored_row[1])

        ranked_candidate_decision_matrix_rows: list[CandidateDecisionMatrixRow] = []
        for rank_position, (matrix_row, weighted_vector_distance_score) in enumerate(
            scored_candidate_rows,
            start=1,
        ):
            ranked_candidate_decision_matrix_rows.append(
                matrix_row.model_copy(
                    update={
                        "final_rank_position": rank_position,
                        "weighted_vector_distance_score": weighted_vector_distance_score,
                    },
                ),
            )

        return CandidateRankingResult(
            ranked_candidate_decision_matrix_rows=ranked_candidate_decision_matrix_rows,
            global_prior_mean_rating=global_prior_mean_rating_m,
            global_smoothing_constant=self._global_smoothing_constant,
        )
