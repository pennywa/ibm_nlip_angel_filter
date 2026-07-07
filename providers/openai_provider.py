"""OpenAI provider adapter using cost-efficient structured JSON output."""

import json
import logging

from openai import AsyncOpenAI, NotFoundError

from providers.base_provider import BaseProvider, ProviderResponsePayload
from providers.comparison_shopping_prompt_builder import (
    build_comparison_shopping_system_instruction,
    build_comparison_shopping_user_prompt,
    serialize_structured_candidates_to_text,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """Adapter for OpenAI comparison-shopping recommendation requests."""

    PROVIDER_IDENTIFIER: str = "openai"
    OPENAI_PRIMARY_MODEL: str = "gpt-4.1-nano"
    OPENAI_FALLBACK_MODEL: str = "gpt-4o-mini"

    def __init__(self, openai_api_key: str) -> None:
        self._openai_api_key = openai_api_key
        self._openai_async_client: AsyncOpenAI | None = None

    @property
    def provider_identifier(self) -> str:
        return self.PROVIDER_IDENTIFIER

    def _get_openai_async_client(self) -> AsyncOpenAI:
        """Lazily initialize the OpenAI async SDK client."""
        if self._openai_async_client is None:
            self._openai_async_client = AsyncOpenAI(api_key=self._openai_api_key)
        return self._openai_async_client

    async def _request_openai_structured_completion(
        self,
        user_search_query: str,
        openai_model_name: str,
    ):
        """Issue a single structured JSON chat completion against the given model."""
        openai_async_client = self._get_openai_async_client()

        return await openai_async_client.chat.completions.create(
            model=openai_model_name,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": build_comparison_shopping_system_instruction(),
                },
                {
                    "role": "user",
                    "content": build_comparison_shopping_user_prompt(
                        user_search_query=user_search_query,
                    ),
                },
            ],
        )

    async def execute_comparison_shopping_query(
        self,
        user_search_query: str,
    ) -> ProviderResponsePayload:
        """Query OpenAI for structured comparison-shopping recommendations."""
        if not self._openai_api_key.strip():
            raise ValueError("OPENAI_API_KEY is not configured.")

        resolved_openai_model_name = self.OPENAI_PRIMARY_MODEL

        try:
            openai_chat_completion_response = await self._request_openai_structured_completion(
                user_search_query=user_search_query,
                openai_model_name=resolved_openai_model_name,
            )
        except NotFoundError:
            logger.warning(
                "OpenAI model '%s' unavailable; falling back to '%s'.",
                self.OPENAI_PRIMARY_MODEL,
                self.OPENAI_FALLBACK_MODEL,
            )
            resolved_openai_model_name = self.OPENAI_FALLBACK_MODEL
            openai_chat_completion_response = await self._request_openai_structured_completion(
                user_search_query=user_search_query,
                openai_model_name=resolved_openai_model_name,
            )

        openai_message_content = openai_chat_completion_response.choices[0].message.content
        if not openai_message_content:
            raise RuntimeError(
                "OpenAI returned an empty message for the comparison-shopping query."
            )

        try:
            structured_openai_candidate_payload = json.loads(openai_message_content)
        except json.JSONDecodeError as json_decode_error:
            raise RuntimeError(
                "OpenAI response was not valid JSON for comparison-shopping candidates."
            ) from json_decode_error

        raw_recommendation_text = serialize_structured_candidates_to_text(
            structured_candidate_payload=structured_openai_candidate_payload,
        )

        logger.info(
            "OpenAI provider (%s) returned structured candidates for query: %s",
            resolved_openai_model_name,
            user_search_query,
        )

        return ProviderResponsePayload(
            provider_identifier=self.provider_identifier,
            raw_recommendation_text=raw_recommendation_text,
            provider_model_name=resolved_openai_model_name,
        )
