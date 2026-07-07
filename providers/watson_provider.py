"""IBM Watsonx provider adapter using ibm-watson IAM authentication and Watsonx REST API."""

import asyncio
import json
import logging

import httpx
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from providers.base_provider import BaseProvider, ProviderResponsePayload
from providers.comparison_shopping_prompt_builder import (
    build_comparison_shopping_system_instruction,
    build_comparison_shopping_user_prompt,
    serialize_structured_candidates_to_text,
)

logger = logging.getLogger(__name__)


class WatsonProvider(BaseProvider):
    """Adapter for IBM Watsonx comparison-shopping recommendation requests."""

    PROVIDER_IDENTIFIER: str = "watson"
    WATSONX_TEXT_GENERATION_API_VERSION: str = "2024-05-31"
    WATSONX_MAX_NEW_TOKENS: int = 2048

    def __init__(
        self,
        watson_api_key: str,
        watson_project_id: str,
        watson_api_base_url: str = "https://us-south.ml.cloud.ibm.com",
        watson_foundation_model_id: str = "ibm/granite-3-8b-instruct",
    ) -> None:
        self._watson_api_key = watson_api_key
        self._watson_project_id = watson_project_id
        self._watson_api_base_url = watson_api_base_url.rstrip("/")
        self._watson_foundation_model_id = watson_foundation_model_id
        self._watson_iam_authenticator = IAMAuthenticator(apikey=self._watson_api_key)

    @property
    def provider_identifier(self) -> str:
        return self.PROVIDER_IDENTIFIER

    def _build_watsonx_comparison_shopping_prompt(
        self,
        user_search_query: str,
    ) -> str:
        """Combine system and user instructions into a single Watsonx text prompt."""
        return (
            f"{build_comparison_shopping_system_instruction()}\n\n"
            f"{build_comparison_shopping_user_prompt(user_search_query=user_search_query)}"
        )

    def _fetch_watson_iam_access_token(self) -> str:
        """Retrieve a fresh IBM Cloud IAM bearer token via the ibm-watson SDK stack."""
        watson_iam_access_token = self._watson_iam_authenticator.token_manager.get_token()
        if not watson_iam_access_token:
            raise RuntimeError(
                "IBM Watson IAM authenticator did not return an access token."
            )
        return watson_iam_access_token

    def _execute_watsonx_text_generation_request(
        self,
        watsonx_comparison_shopping_prompt: str,
    ) -> str:
        """Execute a synchronous Watsonx text generation request."""
        watson_iam_access_token = self._fetch_watson_iam_access_token()

        watsonx_text_generation_url = (
            f"{self._watson_api_base_url}/ml/v1/text/generation"
            f"?version={self.WATSONX_TEXT_GENERATION_API_VERSION}"
        )

        watsonx_request_payload = {
            "model_id": self._watson_foundation_model_id,
            "project_id": self._watson_project_id,
            "input": watsonx_comparison_shopping_prompt,
            "parameters": {
                "max_new_tokens": self.WATSONX_MAX_NEW_TOKENS,
                "temperature": 0.2,
            },
        }

        watsonx_http_response = httpx.post(
            watsonx_text_generation_url,
            headers={
                "Authorization": f"Bearer {watson_iam_access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=watsonx_request_payload,
            timeout=60.0,
        )
        watsonx_http_response.raise_for_status()
        watsonx_response_payload = watsonx_http_response.json()

        watsonx_generation_results = watsonx_response_payload.get("results", [])
        if not watsonx_generation_results:
            raise RuntimeError(
                "Watsonx text generation response did not include any results."
            )

        watsonx_generated_text = watsonx_generation_results[0].get("generated_text", "")
        if not watsonx_generated_text.strip():
            raise RuntimeError(
                "Watsonx text generation returned an empty generated_text field."
            )

        return watsonx_generated_text.strip()

    async def execute_comparison_shopping_query(
        self,
        user_search_query: str,
    ) -> ProviderResponsePayload:
        """Query IBM Watsonx for structured comparison-shopping recommendations."""
        watsonx_comparison_shopping_prompt = self._build_watsonx_comparison_shopping_prompt(
            user_search_query=user_search_query,
        )

        watsonx_generated_text = await asyncio.to_thread(
            self._execute_watsonx_text_generation_request,
            watsonx_comparison_shopping_prompt,
        )

        try:
            structured_watson_candidate_payload = json.loads(watsonx_generated_text)
        except json.JSONDecodeError as json_decode_error:
            raise RuntimeError(
                "Watsonx response was not valid JSON for comparison-shopping candidates."
            ) from json_decode_error

        raw_recommendation_text = serialize_structured_candidates_to_text(
            structured_candidate_payload=structured_watson_candidate_payload,
        )

        logger.info(
            "Watsonx provider returned structured candidates for query: %s",
            user_search_query,
        )

        return ProviderResponsePayload(
            provider_identifier=self.provider_identifier,
            raw_recommendation_text=raw_recommendation_text,
            provider_model_name=self._watson_foundation_model_id,
        )
