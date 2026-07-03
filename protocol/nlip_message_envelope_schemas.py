"""
Pydantic schemas for the ECMA-430 NLIP messaging envelope.

These models isolate protocol wire-framing from ranking calculations and provider
adapter logic. Field names follow NLIP federator conventions for interoperability.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class NLIPMessageKind(str, Enum):
    """Discriminator for NLIP envelope payload types handled by the federator."""

    COMPARISON_SHOPPING_QUERY = "comparison_shopping_query"
    NORMALIZED_RECOMMENDATION = "normalized_recommendation"
    FEDERATOR_ERROR = "federator_error"
    FEDERATOR_STATUS = "federator_status"


class NLIPMessageOrigin(str, Enum):
    """Identifies which layer produced an NLIP envelope."""

    CLIENT = "client"
    FEDERATOR = "federator"
    PROVIDER_ADAPTER = "provider_adapter"
    MATRIX_RANKER = "matrix_ranker"
    OLLAMA_RANKER = "ollama_ranker"


class NLIPMessageHeader(BaseModel):
    """Transport header attached to every ECMA-430 NLIP federator message."""

    nlip_protocol_version: str = Field(
        default="ECMA-430",
        description="NLIP protocol version identifier carried on the wire.",
    )
    message_identifier: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for correlation across federator hops.",
    )
    message_kind: NLIPMessageKind = Field(
        ...,
        description="Payload type discriminator for envelope routing.",
    )
    message_origin: NLIPMessageOrigin = Field(
        ...,
        description="Layer that authored this envelope.",
    )
    created_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the envelope was constructed.",
    )
    authenticated_github_username: str | None = Field(
        default=None,
        description="GitHub identity associated with the request, if authenticated.",
    )


class NLIPFederatorRoutingMetadata(BaseModel):
    """Federator-specific routing and trace metadata for NLIP envelopes."""

    federator_instance_identifier: str = Field(
        ...,
        description="Logical identifier of the federator instance handling the message.",
    )
    upstream_provider_identifiers: list[str] = Field(
        default_factory=list,
        description="Provider adapters consulted while building this envelope.",
    )
    excluded_provider_identifiers: list[str] = Field(
        default_factory=list,
        description="Providers that failed and were excluded from the result set.",
    )
    correlation_trace_identifier: UUID | None = Field(
        default=None,
        description="Optional distributed trace identifier for observability.",
    )


class NLIPComparisonShoppingQueryPayload(BaseModel):
    """Inbound comparison-shopping query body carried inside an NLIP envelope."""

    user_search_query: str = Field(
        ...,
        min_length=1,
        description="Natural-language comparison-shopping query from the user.",
    )
    maximum_candidate_count: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Upper bound on recommendation candidates to return.",
    )
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


class NLIPProviderAttributionRecord(BaseModel):
    """Records which external provider contributed a normalized recommendation."""

    provider_identifier: str = Field(
        ...,
        description="Stable identifier for the originating provider adapter.",
    )
    provider_model_name: str | None = Field(
        default=None,
        description="Specific model name returned by the provider, if available.",
    )
    raw_provider_response_reference: str | None = Field(
        default=None,
        description="Optional opaque reference to the stored raw provider payload.",
    )


class NLIPNormalizedRecommendationPayload(BaseModel):
    """Outbound normalized recommendation body carried inside an NLIP envelope."""

    ranked_candidate_identifiers: list[str] = Field(
        default_factory=list,
        description="Ordered list of candidate identifiers after federator re-ranking.",
    )
    provider_attribution_records: list[NLIPProviderAttributionRecord] = Field(
        default_factory=list,
        description="Attribution for each provider that contributed candidates.",
    )
    federator_routing_metadata: NLIPFederatorRoutingMetadata = Field(
        ...,
        description="Routing metadata describing federator processing.",
    )


class NLIPMessageEnvelope(BaseModel):
    """
    Top-level ECMA-430 NLIP message envelope.

    The ``message_payload`` field holds the body whose structure is determined
    by ``message_header.message_kind``.
    """

    message_header: NLIPMessageHeader = Field(
        ...,
        description="Transport and routing header for the NLIP message.",
    )
    message_payload: dict[str, Any] = Field(
        ...,
        description="Kind-specific payload serialized as a JSON object.",
    )

    def wrap_comparison_shopping_query(
        self,
        comparison_shopping_query: NLIPComparisonShoppingQueryPayload,
        *,
        authenticated_github_username: str | None = None,
    ) -> "NLIPMessageEnvelope":
        """Construct an inbound query envelope (factory helper; body deferred elsewhere)."""
        return NLIPMessageEnvelope(
            message_header=NLIPMessageHeader(
                message_kind=NLIPMessageKind.COMPARISON_SHOPPING_QUERY,
                message_origin=NLIPMessageOrigin.CLIENT,
                authenticated_github_username=authenticated_github_username,
            ),
            message_payload=comparison_shopping_query.model_dump(mode="json"),
        )
