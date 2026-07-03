"""Abstract base contract for external comparison-shopping provider adapters."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class ProviderResponsePayload(BaseModel):
    """Normalized raw response returned by a single provider adapter."""

    provider_identifier: str = Field(
        ...,
        description="Stable identifier for the provider that produced this payload.",
    )
    raw_recommendation_text: str = Field(
        ...,
        description="Unparsed recommendation text returned by the provider.",
    )
    provider_model_name: str | None = Field(
        default=None,
        description="Specific model identifier reported by the provider SDK.",
    )


class ProviderExecutionResult(BaseModel):
    """Outcome of a single provider execution attempt."""

    provider_identifier: str = Field(
        ...,
        description="Stable identifier for the provider that was invoked.",
    )
    provider_response_payload: ProviderResponsePayload | None = Field(
        default=None,
        description="Successful response payload, if the provider call succeeded.",
    )
    provider_failure_message: str | None = Field(
        default=None,
        description="Human-readable failure description when the provider call failed.",
    )

    @property
    def provider_execution_succeeded(self) -> bool:
        """Return True when the provider returned a usable response payload."""
        return self.provider_response_payload is not None


class BaseProvider(ABC):
    """
    Unified execution contract for all external AI provider adapters.

    Concrete implementations must handle SDK initialization and map provider-specific
    responses into ``ProviderResponsePayload`` instances.
    """

    @property
    @abstractmethod
    def provider_identifier(self) -> str:
        """Return the stable identifier used in logs, NLIP attribution, and orchestration."""

    @abstractmethod
    async def execute_comparison_shopping_query(
        self,
        user_search_query: str,
    ) -> ProviderResponsePayload:
        """
        Execute a comparison-shopping query against the external provider.

        Raises:
            Exception: Propagates provider SDK failures to the orchestrator,
                which logs and excludes the provider without crashing the pipeline.
        """

    async def safe_execute_comparison_shopping_query(
        self,
        user_search_query: str,
    ) -> ProviderExecutionResult:
        """
        Execute the provider query inside a guarded boundary.

        Failures are captured and returned as ``ProviderExecutionResult`` rather
        than crashing the federator orchestrator.
        """
        try:
            provider_response_payload = await self.execute_comparison_shopping_query(
                user_search_query=user_search_query,
            )
            return ProviderExecutionResult(
                provider_identifier=self.provider_identifier,
                provider_response_payload=provider_response_payload,
            )
        except Exception as provider_exception:
            return ProviderExecutionResult(
                provider_identifier=self.provider_identifier,
                provider_failure_message=str(provider_exception),
            )
