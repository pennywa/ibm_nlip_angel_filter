"""
Mathematical ranking engine for multi-criteria candidate optimization.

Implements Bayesian quality smoothing and weighted vector distance alignment
to rank comparison-shopping candidates by user utility while surfacing
potential corporate steering deviations.
"""

import math
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from matrix_ranker.candidate_utility_metric_schemas import (
    BayesianQualityRating,
    CandidateDecisionMatrixRow,
    CandidateUtilityMetrics,
)

DEFAULT_GLOBAL_SMOOTHING_CONSTANT: float = 10.0
MINIMUM_DIMENSION_STANDARD_DEVIATION: float = 1e-9
DEFAULT_STEERED_DEVIATION_STANDARD_DEVIATION_MULTIPLIER: float = 1.5


class UtilityDimensionName(str, Enum):
    """Named utility dimensions used in weighted vector distance calculations."""

    COST = "cost"
    DISTANCE = "distance"
    QUALITY = "quality"


class UserAbsoluteUtilityProfile(BaseModel):
    """User preference weights and ideal utility targets for ranking alignment."""

    user_preference_weight_cost: float = Field(
        default=0.33,
        ge=0.0,
        le=1.0,
        description="Importance weight for the Cost utility dimension.",
    )
    user_preference_weight_distance: float = Field(
        default=0.33,
        ge=0.0,
        le=1.0,
        description="Importance weight for the Distance utility dimension.",
    )
    user_preference_weight_quality: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description="Importance weight for the Bayesian Quality utility dimension.",
    )
    user_ideal_cost_target: float | None = Field(
        default=None,
        description="Ideal cost feature target; derived as minimum when omitted.",
    )
    user_ideal_distance_target: float | None = Field(
        default=None,
        description="Ideal distance feature target; derived as minimum when omitted.",
    )
    user_ideal_quality_target: float | None = Field(
        default=None,
        description="Ideal quality feature target; derived as maximum when omitted.",
    )

    @model_validator(mode="after")
    def validate_preference_weights_sum_to_one(self) -> "UserAbsoluteUtilityProfile":
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


class SteeredDeviationAssessment(BaseModel):
    """Assessment of whether a candidate likely reflects corporate steering bias."""

    candidate_identifier: str = Field(
        ...,
        description="Candidate evaluated for steering deviation.",
    )
    weighted_vector_distance_score: float = Field(
        ...,
        ge=0.0,
        description="Computed distance from the user's absolute utility profile.",
    )
    is_flagged_as_steered_deviation: bool = Field(
        ...,
        description="True when the candidate exceeds the steering deviation threshold.",
    )
    steered_deviation_reason: str | None = Field(
        default=None,
        description="Human-readable explanation when steering deviation is flagged.",
    )


class CandidateRankingResult(BaseModel):
    """Complete output from the ranking engine for a candidate decision matrix."""

    ranked_candidate_decision_matrix_rows: list[CandidateDecisionMatrixRow] = Field(
        default_factory=list,
        description="Candidates ordered by ascending weighted vector distance.",
    )
    steered_deviation_assessments: list[SteeredDeviationAssessment] = Field(
        default_factory=list,
        description="Steering deviation flags for each evaluated candidate.",
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
    Multi-criteria optimization engine for fair, user-aligned candidate ranking.

    Applies Bayesian average quality smoothing and weighted vector distance
    minimization to counter opaque provider steering.
    """

    def __init__(
        self,
        global_smoothing_constant: float = DEFAULT_GLOBAL_SMOOTHING_CONSTANT,
        steered_deviation_standard_deviation_multiplier: float = (
            DEFAULT_STEERED_DEVIATION_STANDARD_DEVIATION_MULTIPLIER
        ),
    ) -> None:
        self._global_smoothing_constant = global_smoothing_constant
        self._steered_deviation_standard_deviation_multiplier = (
            steered_deviation_standard_deviation_multiplier
        )

    @staticmethod
    def compute_bayesian_average_rating(
        global_smoothing_constant_c: float,
        global_prior_mean_rating_m: float,
        observed_individual_ratings: list[float],
    ) -> tuple[float, int]:
        """
        Compute Bayesian-adjusted quality rating for one candidate.

        Formula:
            R̄_i = (C · m + Σ r_k) / (C + n)

        Args:
            global_smoothing_constant_c: Confidence threshold (C).
            global_prior_mean_rating_m: Dataset prior mean rating (m).
            observed_individual_ratings: Individual observed ratings (r_k).

        Returns:
            Tuple of (bayesian_quality_rating, observed_rating_count).
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
        """
        Compute Bayesian average when only aggregate mean and count are available.

        Uses Σ r_k = raw_average_rating × n.
        """
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
        """
        Compute the global prior mean (m) across all candidates in the matrix.

        Uses raw average ratings when present; otherwise falls back to existing
        bayesian_quality_rating values.
        """
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
            return 0.0

        return sum(raw_average_rating_values) / len(raw_average_rating_values)

    def apply_bayesian_quality_smoothing(
        self,
        candidate_utility_metrics_list: list[CandidateUtilityMetrics],
        global_prior_mean_rating_m: float | None = None,
    ) -> list[CandidateUtilityMetrics]:
        """
        Apply Bayesian smoothing to every candidate's quality dimension.

        Returns a new list with updated ``BayesianQualityRating`` records.
        """
        resolved_global_prior_mean_rating_m = (
            global_prior_mean_rating_m
            if global_prior_mean_rating_m is not None
            else self.compute_dataset_prior_mean_rating(
                candidate_utility_metrics_list=candidate_utility_metrics_list,
            )
        )

        smoothed_candidate_utility_metrics_list: list[CandidateUtilityMetrics] = []

        for candidate_utility_metrics in candidate_utility_metrics_list:
            existing_bayesian_quality_rating = (
                candidate_utility_metrics.bayesian_quality_rating
            )

            if existing_bayesian_quality_rating.raw_average_rating is not None:
                smoothed_bayesian_quality_rating_value = (
                    self.compute_bayesian_average_from_aggregate_ratings(
                        global_smoothing_constant_c=self._global_smoothing_constant,
                        global_prior_mean_rating_m=resolved_global_prior_mean_rating_m,
                        raw_average_rating=existing_bayesian_quality_rating.raw_average_rating,
                        observed_rating_count_n=existing_bayesian_quality_rating.observed_rating_count,
                    )
                )
            else:
                smoothed_bayesian_quality_rating_value = (
                    existing_bayesian_quality_rating.bayesian_quality_rating
                )

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
        """Extract the cost feature used in vector distance calculations."""
        if candidate_utility_metrics.normalized_cost_score.raw_cost_amount is not None:
            return candidate_utility_metrics.normalized_cost_score.raw_cost_amount
        return candidate_utility_metrics.normalized_cost_score.normalized_cost_score

    @staticmethod
    def _extract_distance_feature_value(
        candidate_utility_metrics: CandidateUtilityMetrics,
    ) -> float:
        """Extract the distance feature used in vector distance calculations."""
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
    def compute_dimension_standard_deviation(
        dimension_feature_values: list[float],
    ) -> float:
        """
        Compute σ_k — standard deviation of a utility dimension across candidates.

        Returns a minimum floor to prevent division-by-zero during z-score scaling.
        """
        if len(dimension_feature_values) <= 1:
            return 1.0

        dimension_mean_value = sum(dimension_feature_values) / len(
            dimension_feature_values
        )
        dimension_variance = sum(
            (feature_value - dimension_mean_value) ** 2
            for feature_value in dimension_feature_values
        ) / len(dimension_feature_values)
        dimension_standard_deviation = math.sqrt(dimension_variance)

        return max(dimension_standard_deviation, MINIMUM_DIMENSION_STANDARD_DEVIATION)

    def compute_weighted_vector_distance(
        self,
        candidate_cost_feature: float,
        candidate_distance_feature: float,
        candidate_quality_feature: float,
        user_absolute_utility_profile: UserAbsoluteUtilityProfile,
        user_ideal_cost_target: float,
        user_ideal_distance_target: float,
        user_ideal_quality_target: float,
        cost_dimension_standard_deviation: float,
        distance_dimension_standard_deviation: float,
        quality_dimension_standard_deviation: float,
    ) -> float:
        """
        Compute weighted vector distance from the user's absolute utility profile.

        Formula:
            d_j = sqrt( Σ w_k · ((x_{k,j} - x_k*) / σ_k)² )
        """
        standardized_cost_deviation = (
            candidate_cost_feature - user_ideal_cost_target
        ) / cost_dimension_standard_deviation
        standardized_distance_deviation = (
            candidate_distance_feature - user_ideal_distance_target
        ) / distance_dimension_standard_deviation
        standardized_quality_deviation = (
            candidate_quality_feature - user_ideal_quality_target
        ) / quality_dimension_standard_deviation

        weighted_sum_of_squared_deviations = (
            user_absolute_utility_profile.user_preference_weight_cost
            * standardized_cost_deviation**2
            + user_absolute_utility_profile.user_preference_weight_distance
            * standardized_distance_deviation**2
            + user_absolute_utility_profile.user_preference_weight_quality
            * standardized_quality_deviation**2
        )

        return math.sqrt(weighted_sum_of_squared_deviations)

    def _resolve_user_ideal_utility_targets(
        self,
        candidate_utility_metrics_list: list[CandidateUtilityMetrics],
        user_absolute_utility_profile: UserAbsoluteUtilityProfile,
    ) -> tuple[float, float, float]:
        """Derive or validate user ideal targets (x_k*) for each dimension."""
        cost_feature_values = [
            self._extract_cost_feature_value(candidate_utility_metrics)
            for candidate_utility_metrics in candidate_utility_metrics_list
        ]
        distance_feature_values = [
            self._extract_distance_feature_value(candidate_utility_metrics)
            for candidate_utility_metrics in candidate_utility_metrics_list
        ]
        quality_feature_values = [
            self._extract_quality_feature_value(candidate_utility_metrics)
            for candidate_utility_metrics in candidate_utility_metrics_list
        ]

        user_ideal_cost_target = (
            user_absolute_utility_profile.user_ideal_cost_target
            if user_absolute_utility_profile.user_ideal_cost_target is not None
            else min(cost_feature_values)
        )
        user_ideal_distance_target = (
            user_absolute_utility_profile.user_ideal_distance_target
            if user_absolute_utility_profile.user_ideal_distance_target is not None
            else min(distance_feature_values)
        )
        user_ideal_quality_target = (
            user_absolute_utility_profile.user_ideal_quality_target
            if user_absolute_utility_profile.user_ideal_quality_target is not None
            else max(quality_feature_values)
        )

        return (
            user_ideal_cost_target,
            user_ideal_distance_target,
            user_ideal_quality_target,
        )

    def identify_steered_deviation_candidates(
        self,
        candidate_distance_score_pairs: list[tuple[str, float]],
    ) -> list[SteeredDeviationAssessment]:
        """
        Flag candidates whose utility misalignment suggests corporate steering.

        Candidates exceeding mean + k·σ of weighted vector distance scores are
        flagged as likely steered deviations from the user's utility profile.
        """
        if not candidate_distance_score_pairs:
            return []

        weighted_vector_distance_scores = [
            distance_score for _, distance_score in candidate_distance_score_pairs
        ]
        mean_weighted_vector_distance = sum(weighted_vector_distance_scores) / len(
            weighted_vector_distance_scores
        )
        distance_score_variance = sum(
            (distance_score - mean_weighted_vector_distance) ** 2
            for distance_score in weighted_vector_distance_scores
        ) / len(weighted_vector_distance_scores)
        distance_score_standard_deviation = math.sqrt(distance_score_variance)

        steering_deviation_threshold = mean_weighted_vector_distance + (
            self._steered_deviation_standard_deviation_multiplier
            * max(distance_score_standard_deviation, MINIMUM_DIMENSION_STANDARD_DEVIATION)
        )

        steered_deviation_assessments: list[SteeredDeviationAssessment] = []

        for (
            candidate_identifier,
            weighted_vector_distance_score,
        ) in candidate_distance_score_pairs:
            is_flagged_as_steered_deviation = (
                weighted_vector_distance_score > steering_deviation_threshold
            )
            steered_deviation_reason = None
            if is_flagged_as_steered_deviation:
                steered_deviation_reason = (
                    "Candidate weighted vector distance exceeds the steering "
                    f"deviation threshold ({steering_deviation_threshold:.4f}), "
                    "indicating misalignment with the user's absolute utility profile."
                )

            steered_deviation_assessments.append(
                SteeredDeviationAssessment(
                    candidate_identifier=candidate_identifier,
                    weighted_vector_distance_score=weighted_vector_distance_score,
                    is_flagged_as_steered_deviation=is_flagged_as_steered_deviation,
                    steered_deviation_reason=steered_deviation_reason,
                )
            )

        return steered_deviation_assessments

    def rank_candidate_decision_matrix(
        self,
        candidate_decision_matrix_rows: list[CandidateDecisionMatrixRow],
        user_absolute_utility_profile: UserAbsoluteUtilityProfile,
    ) -> CandidateRankingResult:
        """
        Rank candidates by minimizing weighted vector distance to user utility.

        Pipeline:
            1. Apply Bayesian quality smoothing across the matrix.
            2. Standardize Cost, Distance, and Quality dimensions by σ_k.
            3. Compute weighted vector distance d_j for each candidate.
            4. Sort ascending by d_j (lower = better user alignment).
            5. Flag steered deviations exceeding the statistical threshold.
        """
        if not candidate_decision_matrix_rows:
            return CandidateRankingResult(
                ranked_candidate_decision_matrix_rows=[],
                steered_deviation_assessments=[],
                global_prior_mean_rating=0.0,
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

        (
            user_ideal_cost_target,
            user_ideal_distance_target,
            user_ideal_quality_target,
        ) = self._resolve_user_ideal_utility_targets(
            candidate_utility_metrics_list=smoothed_candidate_utility_metrics_list,
            user_absolute_utility_profile=user_absolute_utility_profile,
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

        cost_dimension_standard_deviation = self.compute_dimension_standard_deviation(
            dimension_feature_values=cost_feature_values,
        )
        distance_dimension_standard_deviation = self.compute_dimension_standard_deviation(
            dimension_feature_values=distance_feature_values,
        )
        quality_dimension_standard_deviation = self.compute_dimension_standard_deviation(
            dimension_feature_values=quality_feature_values,
        )

        scored_candidate_rows: list[tuple[CandidateDecisionMatrixRow, float]] = []

        for matrix_row_index, matrix_row in enumerate(candidate_decision_matrix_rows):
            smoothed_candidate_utility_metrics = smoothed_candidate_utility_metrics_list[
                matrix_row_index
            ]

            weighted_vector_distance_score = self.compute_weighted_vector_distance(
                candidate_cost_feature=self._extract_cost_feature_value(
                    smoothed_candidate_utility_metrics
                ),
                candidate_distance_feature=self._extract_distance_feature_value(
                    smoothed_candidate_utility_metrics
                ),
                candidate_quality_feature=self._extract_quality_feature_value(
                    smoothed_candidate_utility_metrics
                ),
                user_absolute_utility_profile=user_absolute_utility_profile,
                user_ideal_cost_target=user_ideal_cost_target,
                user_ideal_distance_target=user_ideal_distance_target,
                user_ideal_quality_target=user_ideal_quality_target,
                cost_dimension_standard_deviation=cost_dimension_standard_deviation,
                distance_dimension_standard_deviation=distance_dimension_standard_deviation,
                quality_dimension_standard_deviation=quality_dimension_standard_deviation,
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

        candidate_distance_score_pairs = [
            (
                matrix_row.candidate_utility_metrics.candidate_identifier,
                matrix_row.weighted_vector_distance_score or 0.0,
            )
            for matrix_row in ranked_candidate_decision_matrix_rows
        ]

        steered_deviation_assessments = self.identify_steered_deviation_candidates(
            candidate_distance_score_pairs=candidate_distance_score_pairs,
        )

        return CandidateRankingResult(
            ranked_candidate_decision_matrix_rows=ranked_candidate_decision_matrix_rows,
            steered_deviation_assessments=steered_deviation_assessments,
            global_prior_mean_rating=global_prior_mean_rating_m,
            global_smoothing_constant=self._global_smoothing_constant,
        )
