"""IBM Watson provider adapter skeleton."""

import logging

from providers.base_provider import BaseProvider, ProviderResponsePayload

logger = logging.getLogger(__name__)


class WatsonProvider(BaseProvider):
    """Adapter for IBM Watson comparison-shopping recommendation requests."""

    PROVIDER_IDENTIFIER: str = "watson"

    def __init__(self, watson_api_key: str) -> None:
        self._watson_api_key = watson_api_key
        self._watson_service_client = None  # Deferred: initialize ibm-watson client.

    @property
    def provider_identifier(self) -> str:
        return self.PROVIDER_IDENTIFIER

    async def execute_comparison_shopping_query(
        self,
        user_search_query: str,
    ) -> ProviderResponsePayload:
        """
        Query IBM Watson for comparison-shopping recommendations.

        Implementation deferred: SDK invocation and response mapping.
        """
        logger.debug(
            "Watson provider invoked for query (not yet implemented): %s",
            user_search_query,
        )
        raise NotImplementedError(
            "Watson comparison-shopping query execution is not yet implemented."
        )
