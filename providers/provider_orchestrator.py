"""Fan-out orchestration across provider adapters with graceful failure isolation."""

import asyncio
import logging

from providers.base_provider import BaseProvider, ProviderExecutionResult, ProviderResponsePayload

logger = logging.getLogger(__name__)


class ProviderOrchestrator:
    """
    Execute comparison-shopping queries across multiple providers in parallel.

    Providers that fail are logged and recorded in ``excluded_provider_identifiers``
    without crashing the federator pipeline.
    """

    def __init__(self, registered_provider_adapters: list[BaseProvider]) -> None:
        self._registered_provider_adapters = registered_provider_adapters

    async def execute_all_providers(
        self,
        user_search_query: str,
    ) -> tuple[list[ProviderResponsePayload], list[str]]:
        """
        Fan out the query to every registered provider and collect successes.

        Returns:
            A tuple of (successful_provider_response_payloads, excluded_provider_identifiers).
        """
        provider_execution_tasks = [
            provider_adapter.safe_execute_comparison_shopping_query(
                user_search_query=user_search_query,
            )
            for provider_adapter in self._registered_provider_adapters
        ]

        provider_execution_results: list[ProviderExecutionResult] = await asyncio.gather(
            *provider_execution_tasks
        )

        successful_provider_response_payloads: list[ProviderResponsePayload] = []
        excluded_provider_identifiers: list[str] = []

        for provider_execution_result in provider_execution_results:
            if provider_execution_result.provider_execution_succeeded:
                assert provider_execution_result.provider_response_payload is not None
                successful_provider_response_payloads.append(
                    provider_execution_result.provider_response_payload
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

        return successful_provider_response_payloads, excluded_provider_identifiers
