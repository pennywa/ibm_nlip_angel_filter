"""
Pydantic schemas for parsing raw provider text into uniform utility metric attributes.

These models define the structural contract between provider adapters and the
ranking engine. Deep scoring math is deferred to a subsequent implementation phase.
"""

from pydantic import BaseModel, Field, field_validator


class RawProviderRecommendationRecord(BaseModel):
    """Single recommendation line extracted from a provider's raw text response."""

    candidate_identifier: str = Field(
        ...,
        description="Stable identifier for the recommended product or service.",
    )
    candidate_display_name: str = Field(
        ...,
        description="Human-readable name of the recommended candidate.",
    )
    raw_recommendation_text: str = Field(
        ...,
        description="Original unparsed text fragment describing this candidate.",
    )
    source_provider_identifier: str = Field(
        ...,
        description="Provider adapter that supplied this recommendation record.",
    )


class ParsedProviderTextResponse(BaseModel):
    """Structured parse result for an entire provider raw text response."""

    source_provider_identifier: str = Field(
        ...,
        description="Provider adapter that produced the raw response.",
    )
    user_search_query: str = Field(
        ...,
        description="Comparison-shopping query that generated this response.",
    )
    raw_provider_recommendation_records: list[RawProviderRecommendationRecord] = Field(
        default_factory=list,
        description="Individual recommendation records extracted from the raw text.",
    )


class NormalizedCostScore(BaseModel):
    """Cost utility dimension for a single candidate."""

    candidate_identifier: str = Field(
        ...,
        description="Candidate this cost score applies to.",
    )
    normalized_cost_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Zero-to-one normalized cost score (lower raw cost yields lower score).",
    )
    raw_cost_amount: float | None = Field(
        default=None,
        ge=0.0,
        description="Original monetary cost before normalization, if extracted.",
    )
    cost_currency_code: str | None = Field(
        default=None,
        description="ISO 4217 currency code for raw_cost_amount, if known.",
    )


class NormalizedDistanceScore(BaseModel):
    """Distance utility dimension for a single candidate."""

    candidate_identifier: str = Field(
        ...,
        description="Candidate this distance score applies to.",
    )
    normalized_distance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Zero-to-one normalized distance score (closer yields lower score).",
    )
    raw_distance_kilometers: float | None = Field(
        default=None,
        ge=0.0,
        description="Original distance in kilometers before normalization, if extracted.",
    )


class BayesianQualityRating(BaseModel):
    """Bayesian-adjusted quality rating for a single candidate."""

    candidate_identifier: str = Field(
        ...,
        description="Candidate this quality rating applies to.",
    )
    bayesian_quality_rating: float = Field(
        ...,
        ge=0.0,
        description="Bayesian-averaged quality score balancing rating volume.",
    )
    observed_rating_count: int = Field(
        default=0,
        ge=0,
        description="Number of raw ratings observed for this candidate.",
    )
    raw_average_rating: float | None = Field(
        default=None,
        ge=0.0,
        description="Unadjusted arithmetic mean of observed ratings, if available.",
    )
    global_prior_mean_rating: float | None = Field(
        default=None,
        ge=0.0,
        description="Population prior mean used in the Bayesian average formula.",
    )
    confidence_threshold_rating_count: int | None = Field(
        default=None,
        ge=0,
        description="Minimum effective rating count (C) used in the Bayesian formula.",
    )


class CandidateUtilityMetrics(BaseModel):
    """Unified utility metric attributes for one candidate across all dimensions."""

    candidate_identifier: str = Field(
        ...,
        description="Candidate these utility metrics describe.",
    )
    candidate_display_name: str = Field(
        ...,
        description="Human-readable name of the candidate.",
    )
    normalized_cost_score: NormalizedCostScore = Field(
        ...,
        description="Cost dimension metrics for this candidate.",
    )
    normalized_distance_score: NormalizedDistanceScore = Field(
        ...,
        description="Distance dimension metrics for this candidate.",
    )
    bayesian_quality_rating: BayesianQualityRating = Field(
        ...,
        description="Bayesian quality dimension metrics for this candidate.",
    )

    @field_validator("normalized_cost_score")
    @classmethod
    def validate_cost_candidate_identifier_matches(
        cls,
        normalized_cost_score: NormalizedCostScore,
        validation_info,
    ) -> NormalizedCostScore:
        """Ensure nested cost score references the same candidate identifier."""
        candidate_identifier = validation_info.data.get("candidate_identifier")
        if (
            candidate_identifier is not None
            and normalized_cost_score.candidate_identifier != candidate_identifier
        ):
            raise ValueError(
                "normalized_cost_score.candidate_identifier must match "
                "CandidateUtilityMetrics.candidate_identifier."
            )
        return normalized_cost_score

    @field_validator("normalized_distance_score")
    @classmethod
    def validate_distance_candidate_identifier_matches(
        cls,
        normalized_distance_score: NormalizedDistanceScore,
        validation_info,
    ) -> NormalizedDistanceScore:
        """Ensure nested distance score references the same candidate identifier."""
        candidate_identifier = validation_info.data.get("candidate_identifier")
        if (
            candidate_identifier is not None
            and normalized_distance_score.candidate_identifier != candidate_identifier
        ):
            raise ValueError(
                "normalized_distance_score.candidate_identifier must match "
                "CandidateUtilityMetrics.candidate_identifier."
            )
        return normalized_distance_score

    @field_validator("bayesian_quality_rating")
    @classmethod
    def validate_quality_candidate_identifier_matches(
        cls,
        bayesian_quality_rating: BayesianQualityRating,
        validation_info,
    ) -> BayesianQualityRating:
        """Ensure nested quality rating references the same candidate identifier."""
        candidate_identifier = validation_info.data.get("candidate_identifier")
        if (
            candidate_identifier is not None
            and bayesian_quality_rating.candidate_identifier != candidate_identifier
        ):
            raise ValueError(
                "bayesian_quality_rating.candidate_identifier must match "
                "CandidateUtilityMetrics.candidate_identifier."
            )
        return bayesian_quality_rating


class CandidateDecisionMatrixRow(BaseModel):
    """One row in the decision matrix passed to the Ollama re-ranker engine."""

    candidate_utility_metrics: CandidateUtilityMetrics = Field(
        ...,
        description="Parsed utility metrics for this matrix row.",
    )
    source_provider_identifiers: list[str] = Field(
        default_factory=list,
        description="Providers that recommended this candidate.",
    )
    weighted_vector_distance_score: float | None = Field(
        default=None,
        ge=0.0,
        description="Computed weighted distance from user ideal (deferred calculation).",
    )
    final_rank_position: int | None = Field(
        default=None,
        ge=1,
        description="Final rank assigned after re-ranking (deferred assignment).",
    )
