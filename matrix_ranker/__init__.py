"""Decision matrix construction, normalization, and mathematical ranking."""

from matrix_ranker.candidate_utility_metric_schemas import (
    BayesianQualityRating,
    CandidateDecisionMatrixRow,
    CandidateUtilityMetrics,
    NormalizedCostScore,
    NormalizedDistanceScore,
    ParsedProviderTextResponse,
    RawProviderRecommendationRecord,
)
from matrix_ranker.matrix_normalization_service import (
    build_candidate_decision_matrix_from_provider_payloads,
    build_candidate_decision_matrix_from_raw_json,
    parse_raw_json_candidate_block,
)
from matrix_ranker.ranking_engine import (
    CandidateRankingEngine,
    CandidateRankingResult,
    UserPreferenceWeightProfile,
)

__all__ = [
    "BayesianQualityRating",
    "CandidateDecisionMatrixRow",
    "CandidateRankingEngine",
    "CandidateRankingResult",
    "CandidateUtilityMetrics",
    "NormalizedCostScore",
    "NormalizedDistanceScore",
    "ParsedProviderTextResponse",
    "RawProviderRecommendationRecord",
    "UserPreferenceWeightProfile",
    "build_candidate_decision_matrix_from_provider_payloads",
    "build_candidate_decision_matrix_from_raw_json",
    "parse_raw_json_candidate_block",
]
