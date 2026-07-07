"""Fan-out orchestration across provider adapters with graceful failure isolation."""

import asyncio
import logging

from providers.base_provider import BaseProvider, ProviderExecutionResult, ProviderResponsePayload
from providers.gemini_provider import GeminiProvider
from providers.openai_provider import OpenAIProvider
from providers.provider_settings import ProviderSettings, get_provider_settings
from providers.watson_provider import WatsonProvider

logger = logging.getLogger(__name__)


class ProviderOrchestrator:
    """
    Execute comparison-shopping queries across multiple providers in parallel.

    Providers that fail are logged and recorded in ``excluded_provider_identifiers``
    without crashing the federator pipeline.
    """

    def __init__(
        self,
        registered_provider_adapters: list[BaseProvider],
        provider_query_timeout_seconds: float = 60.0,
    ) -> None:
        self._registered_provider_adapters = registered_provider_adapters
        self._provider_query_timeout_seconds = provider_query_timeout_seconds

    @classmethod
    def create_from_provider_settings(
        cls,
        provider_settings: ProviderSettings | None = None,
    ) -> "ProviderOrchestrator":
        """Build an orchestrator wired to OpenAI, Gemini, and Watsonx adapters."""
        resolved_provider_settings = provider_settings or get_provider_settings()

        registered_provider_adapters: list[BaseProvider] = [
            OpenAIProvider(openai_api_key=resolved_provider_settings.openai_api_key),
            GeminiProvider(gemini_api_key=resolved_provider_settings.gemini_api_key),
            WatsonProvider(
                watson_api_key=resolved_provider_settings.watson_api_key,
                watson_project_id=resolved_provider_settings.watson_project_id,
                watson_api_base_url=resolved_provider_settings.watson_api_base_url,
                watson_foundation_model_id=resolved_provider_settings.watson_foundation_model_id,
            ),
        ]

        return cls(
            registered_provider_adapters=registered_provider_adapters,
            provider_query_timeout_seconds=resolved_provider_settings.provider_query_timeout_seconds,
        )

    async def _execute_provider_with_timeout(
        self,
        provider_adapter: BaseProvider,
        user_search_query: str,
    ) -> ProviderExecutionResult:
        """Execute a single provider inside a timeout boundary."""
        try:
            return await asyncio.wait_for(
                provider_adapter.safe_execute_comparison_shopping_query(
                    user_search_query=user_search_query,
                ),
                timeout=self._provider_query_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return ProviderExecutionResult(
                provider_identifier=provider_adapter.provider_identifier,
                provider_failure_message=(
                    f"Provider timed out after {self._provider_query_timeout_seconds:.0f} seconds."
                ),
            )

    async def execute_all_providers(
        self,
        user_search_query: str,
    ) -> tuple[list[ProviderResponsePayload], list[str]]:
        """
        Fan out the query to every registered provider and collect successes.

        Returns:
            A tuple of (successful_provider_response_payloads, excluded_provider_identifiers).
        """
        if not self._registered_provider_adapters:
            logger.warning(
                "Provider orchestrator invoked with zero registered provider adapters."
            )
            return [], []

        logger.info(
            "Broadcasting comparison-shopping query to %d providers: %s",
            len(self._registered_provider_adapters),
            user_search_query,
        )

        provider_execution_tasks = [
            self._execute_provider_with_timeout(
                provider_adapter=provider_adapter,
                user_search_query=user_search_query,
            )
            for provider_adapter in self._registered_provider_adapters
        ]

        provider_execution_results: list[ProviderExecutionResult] = await asyncio.gather(
            *provider_execution_tasks,
        )

        successful_provider_response_payloads: list[ProviderResponsePayload] = []
        excluded_provider_identifiers: list[str] = []

        for provider_execution_result in provider_execution_results:
            if provider_execution_result.provider_execution_succeeded:
                assert provider_execution_result.provider_response_payload is not None
                successful_provider_response_payloads.append(
                    provider_execution_result.provider_response_payload
                )
                logger.info(
                    "Provider '%s' succeeded for comparison-shopping query.",
                    provider_execution_result.provider_identifier,
                )
            else:
                excluded_provider_identifiers.append(
                    provider_execution_result.provider_identifier
                )
                logger.warning(
                    "Provider '%s' excluded due to failure: %s",
                    provider_execution_result.provider_identifier,
                    provider_execution_result.provider_failure_message,
                )

        logger.info(
            "Provider fan-out complete: %d succeeded, %d excluded.",
            len(successful_provider_response_payloads),
            len(excluded_provider_identifiers),
        )

        return successful_provider_response_payloads, excluded_provider_identifiers
