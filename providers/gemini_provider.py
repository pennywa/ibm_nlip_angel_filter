"""Google Gemini provider adapter skeleton."""

import logging

from providers.base_provider import BaseProvider, ProviderResponsePayload

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Adapter for Google Gemini comparison-shopping recommendation requests."""

    PROVIDER_IDENTIFIER: str = "gemini"

    def __init__(self, gemini_api_key: str) -> None:
        self._gemini_api_key = gemini_api_key
        self._gemini_client = None  # Deferred: initialize google.genai.Client.

    @property
    def provider_identifier(self) -> str:
        return self.PROVIDER_IDENTIFIER

    async def execute_comparison_shopping_query(
        self,
        user_search_query: str,
    ) -> ProviderResponsePayload:
        """
        Query Google Gemini for comparison-shopping recommendations.

        Implementation deferred: SDK invocation and response mapping.
        """
        logger.debug(
            "Gemini provider invoked for query (not yet implemented): %s",
            user_search_query,
        )
        raise NotImplementedError(
            "Gemini comparison-shopping query execution is not yet implemented."
        )
