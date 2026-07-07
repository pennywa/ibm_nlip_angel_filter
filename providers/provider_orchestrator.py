"""Fan-out orchestration across provider adapters with graceful failure isolation."""

import asyncio
import logging

from providers.base_provider import BaseProvider, ProviderExecutionResult, ProviderResponsePayload
from providers.gemini_provider import GeminiProvider
from providers.ollama_adapter import OllamaProviderAdapter
from providers.openai_provider import OpenAIProvider
from providers.provider_settings import ProviderSettings, get_provider_settings
from providers.watson_provider import WatsonProvider

logger = logging.getLogger(__name__)


class ProviderOrchestrator:
    """
    Execute comparison-shopping queries across multiple providers in parallel.

    Providers that fail are logged and recorded in ``excluded_provider_identifiers``
    without crashing the federator pipeline.

    When ``APPLICATION_ENVIRONMENT`` is ``development``, or when cloud upstream
    credentials are absent, the local Ollama adapter is registered for fan-out.
    If every cloud provider fails at runtime, Ollama is invoked as a fallback.
    """

    def __init__(
        self,
        registered_provider_adapters: list[BaseProvider],
        ollama_fallback_provider_adapter: OllamaProviderAdapter | None = None,
    ) -> None:
        self._registered_provider_adapters = registered_provider_adapters
        self._ollama_fallback_provider_adapter = ollama_fallback_provider_adapter

    @staticmethod
    def _build_ollama_provider_adapter(
        provider_settings: ProviderSettings,
    ) -> OllamaProviderAdapter:
        """Construct a configured Ollama provider adapter from environment settings."""
        return OllamaProviderAdapter(
            ollama_base_url=provider_settings.ollama_base_url,
            ollama_comparison_shopping_model_name=(
                provider_settings.ollama_comparison_shopping_model_name
            ),
        )

    @classmethod
    def _build_cloud_provider_adapters(
        cls,
        provider_settings: ProviderSettings,
    ) -> list[BaseProvider]:
        """Register cloud provider adapters only when credentials are configured."""
        registered_cloud_provider_adapters: list[BaseProvider] = []

        if provider_settings.openai_provider_is_configured:
            registered_cloud_provider_adapters.append(
                OpenAIProvider(openai_api_key=provider_settings.openai_api_key),
            )
        else:
            logger.info(
                "Skipping OpenAI provider registration: API key is missing or placeholder."
            )

        if provider_settings.gemini_provider_is_configured:
            registered_cloud_provider_adapters.append(
                GeminiProvider(gemini_api_key=provider_settings.gemini_api_key),
            )
        else:
            logger.info(
                "Skipping Gemini provider registration: API key is missing or placeholder."
            )

        if provider_settings.watson_provider_is_configured:
            registered_cloud_provider_adapters.append(
                WatsonProvider(
                    watson_api_key=provider_settings.watson_api_key,
                    watson_project_id=provider_settings.watson_project_id,
                    watson_api_base_url=provider_settings.watson_api_base_url,
                    watson_foundation_model_id=provider_settings.watson_foundation_model_id,
                ),
            )
        else:
            logger.info(
                "Skipping Watsonx provider registration: API key or project ID is missing."
            )

        return registered_cloud_provider_adapters

    @classmethod
    def create_from_provider_settings(
        cls,
        provider_settings: ProviderSettings | None = None,
    ) -> "ProviderOrchestrator":
        """
        Build an orchestrator wired to available cloud adapters and local Ollama.

        Ollama is registered for parallel fan-out when running in development or
        when no cloud provider credentials are configured.
        """
        resolved_provider_settings = provider_settings or get_provider_settings()

        registered_provider_adapters: list[BaseProvider] = cls._build_cloud_provider_adapters(
            provider_settings=resolved_provider_settings,
        )

        ollama_provider_adapter = cls._build_ollama_provider_adapter(
            provider_settings=resolved_provider_settings,
        )
        ollama_fallback_provider_adapter: OllamaProviderAdapter | None = None

        ollama_should_participate_in_fan_out = (
            resolved_provider_settings.is_development_environment
            or not resolved_provider_settings.any_cloud_provider_is_configured
        )

        if ollama_should_participate_in_fan_out:
            registered_provider_adapters.append(ollama_provider_adapter)
            logger.info(
                "Registered local Ollama provider for parallel fan-out "
                "(environment=%s, cloud_providers_configured=%s).",
                resolved_provider_settings.application_environment,
                resolved_provider_settings.any_cloud_provider_is_configured,
            )
        else:
            ollama_fallback_provider_adapter = ollama_provider_adapter
            logger.info(
                "Local Ollama provider reserved for cloud-failure fallback "
                "(environment=%s).",
                resolved_provider_settings.application_environment,
            )

        return cls(
            registered_provider_adapters=registered_provider_adapters,
            ollama_fallback_provider_adapter=ollama_fallback_provider_adapter,
        )

    async def _execute_ollama_fallback_provider(
        self,
        user_search_query: str,
    ) -> ProviderExecutionResult | None:
        """Invoke the local Ollama adapter when all cloud providers have failed."""
        if self._ollama_fallback_provider_adapter is None:
            return None

        logger.warning(
            "All cloud providers failed; falling back to local Ollama for query: %s",
            user_search_query,
        )

        return await self._ollama_fallback_provider_adapter.safe_execute_comparison_shopping_query(
            user_search_query=user_search_query,
        )

    async def execute_all_providers(
        self,
        user_search_query: str,
    ) -> tuple[list[ProviderResponsePayload], list[str]]:
        """
        Broadcast the user inquiry to all registered providers simultaneously.

        When every registered provider fails and an Ollama fallback adapter is
        configured, a single local inference attempt is made before returning.

        Returns:
            A tuple of (successful_provider_response_payloads, excluded_provider_identifiers).
        """
        if not self._registered_provider_adapters:
            logger.warning(
                "Provider orchestrator invoked with zero registered provider adapters."
            )
            ollama_fallback_execution_result = await self._execute_ollama_fallback_provider(
                user_search_query=user_search_query,
            )
            if (
                ollama_fallback_execution_result is not None
                and ollama_fallback_execution_result.provider_execution_succeeded
            ):
                assert ollama_fallback_execution_result.provider_response_payload is not None
                return [ollama_fallback_execution_result.provider_response_payload], []
            if ollama_fallback_execution_result is not None:
                return [], [ollama_fallback_execution_result.provider_identifier]
            return [], []

        logger.info(
            "Broadcasting comparison-shopping query to %d providers: %s",
            len(self._registered_provider_adapters),
            user_search_query,
        )

        provider_execution_tasks = [
            provider_adapter.safe_execute_comparison_shopping_query(
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

        if not successful_provider_response_payloads:
            ollama_fallback_execution_result = await self._execute_ollama_fallback_provider(
                user_search_query=user_search_query,
            )
            if (
                ollama_fallback_execution_result is not None
                and ollama_fallback_execution_result.provider_execution_succeeded
            ):
                assert ollama_fallback_execution_result.provider_response_payload is not None
                successful_provider_response_payloads.append(
                    ollama_fallback_execution_result.provider_response_payload,
                )
                logger.info(
                    "Ollama fallback provider succeeded for comparison-shopping query."
                )
            elif ollama_fallback_execution_result is not None:
                excluded_provider_identifiers.append(
                    ollama_fallback_execution_result.provider_identifier,
                )
                logger.warning(
                    "Ollama fallback provider excluded due to failure: %s",
                    ollama_fallback_execution_result.provider_failure_message,
                )

        logger.info(
            "Provider fan-out complete: %d succeeded, %d excluded.",
            len(successful_provider_response_payloads),
            len(excluded_provider_identifiers),
        )

        return successful_provider_response_payloads, excluded_provider_identifiers
