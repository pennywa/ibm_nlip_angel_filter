"""Google Gemini provider adapter using the google-genai SDK."""

import json
import logging

from google import genai
from google.genai import types

from providers.base_provider import BaseProvider, ProviderResponsePayload
from providers.comparison_shopping_prompt_builder import (
    build_comparison_shopping_system_instruction,
    build_comparison_shopping_user_prompt,
    serialize_structured_candidates_to_text,
)

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Adapter for Google Gemini comparison-shopping recommendation requests."""

    PROVIDER_IDENTIFIER: str = "gemini"
    GEMINI_COMPARISON_SHOPPING_MODEL: str = "gemini-1.5-pro"

    def __init__(self, gemini_api_key: str) -> None:
        self._gemini_api_key = gemini_api_key
        self._gemini_client: genai.Client | None = None

    @property
    def provider_identifier(self) -> str:
        return self.PROVIDER_IDENTIFIER

    def _get_gemini_client(self) -> genai.Client:
        """Lazily initialize the google-genai client."""
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self._gemini_api_key)
        return self._gemini_client

    async def execute_comparison_shopping_query(
        self,
        user_search_query: str,
    ) -> ProviderResponsePayload:
        """Query Gemini 1.5 Pro for structured comparison-shopping recommendations."""
        gemini_client = self._get_gemini_client()

        gemini_generation_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            system_instruction=build_comparison_shopping_system_instruction(),
        )

        gemini_generate_content_response = (
            await gemini_client.aio.models.generate_content(
                model=self.GEMINI_COMPARISON_SHOPPING_MODEL,
                contents=build_comparison_shopping_user_prompt(
                    user_search_query=user_search_query,
                ),
                config=gemini_generation_config,
            )
        )

        gemini_response_text = gemini_generate_content_response.text
        if not gemini_response_text:
            raise RuntimeError(
                "Gemini returned an empty response for the comparison-shopping query."
            )

        try:
            structured_gemini_candidate_payload = json.loads(gemini_response_text)
        except json.JSONDecodeError as json_decode_error:
            raise RuntimeError(
                "Gemini response was not valid JSON for comparison-shopping candidates."
            ) from json_decode_error

        raw_recommendation_text = serialize_structured_candidates_to_text(
            structured_candidate_payload=structured_gemini_candidate_payload,
        )

        logger.info(
            "Gemini provider returned structured candidates for query: %s",
            user_search_query,
        )

        return ProviderResponsePayload(
            provider_identifier=self.provider_identifier,
            raw_recommendation_text=raw_recommendation_text,
            provider_model_name=self.GEMINI_COMPARISON_SHOPPING_MODEL,
        )
