"""OpenAI provider adapter skeleton."""

import logging

from providers.base_provider import BaseProvider, ProviderResponsePayload

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """Adapter for OpenAI comparison-shopping recommendation requests."""

    PROVIDER_IDENTIFIER: str = "openai"

    def __init__(self, openai_api_key: str) -> None:
        self._openai_api_key = openai_api_key
        self._openai_client = None  # Deferred: initialize openai.AsyncOpenAI client.

    @property
    def provider_identifier(self) -> str:
        return self.PROVIDER_IDENTIFIER

    async def execute_comparison_shopping_query(
        self,
        user_search_query: str,
    ) -> ProviderResponsePayload:
        """
        Query OpenAI for comparison-shopping recommendations.

        Implementation deferred: SDK invocation and response mapping.
        """
        logger.debug(
            "OpenAI provider invoked for query (not yet implemented): %s",
            user_search_query,
        )
        raise NotImplementedError(
            "OpenAI comparison-shopping query execution is not yet implemented."
        )
