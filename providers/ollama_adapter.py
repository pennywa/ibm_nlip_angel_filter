"""Local Ollama provider adapter for zero-cost comparison-shopping candidate generation."""

import json
import logging
import re

from ollama import AsyncClient

from providers.base_provider import BaseProvider, ProviderResponsePayload
from providers.comparison_shopping_prompt_builder import (
    build_comparison_shopping_system_instruction,
    build_comparison_shopping_user_prompt,
    serialize_structured_candidates_to_text,
)

logger = logging.getLogger(__name__)

MARKDOWN_JSON_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_balanced_json_object_text(candidate_json_text: str) -> str | None:
    """Extract the first balanced JSON object substring from arbitrary model output."""
    brace_depth = 0
    json_object_start_index: int | None = None

    for character_index, current_character in enumerate(candidate_json_text):
        if current_character == "{":
            if brace_depth == 0:
                json_object_start_index = character_index
            brace_depth += 1
        elif current_character == "}":
            if brace_depth == 0:
                continue
            brace_depth -= 1
            if brace_depth == 0 and json_object_start_index is not None:
                return candidate_json_text[json_object_start_index : character_index + 1]

    return None


def extract_json_payload_from_local_model_response(
    local_model_response_text: str,
) -> dict:
    """
    Parse structured comparison-shopping JSON from local model output.

    Local models frequently wrap JSON in markdown fences or prepend commentary.
    This routine attempts direct parsing first, then progressively more
    permissive extraction strategies.
    """
    stripped_local_model_response_text = local_model_response_text.strip()
    if not stripped_local_model_response_text:
        raise RuntimeError(
            "Ollama returned an empty message for the comparison-shopping query."
        )

    direct_parse_candidates: list[str] = [stripped_local_model_response_text]

    markdown_fence_match_groups = MARKDOWN_JSON_FENCE_PATTERN.findall(
        stripped_local_model_response_text,
    )
    direct_parse_candidates.extend(
        fenced_json_fragment.strip()
        for fenced_json_fragment in markdown_fence_match_groups
        if fenced_json_fragment.strip()
    )

    first_opening_brace_index = stripped_local_model_response_text.find("{")
    if first_opening_brace_index >= 0:
        balanced_json_object_text = _extract_balanced_json_object_text(
            stripped_local_model_response_text[first_opening_brace_index:],
        )
        if balanced_json_object_text:
            direct_parse_candidates.append(balanced_json_object_text)

    last_json_decode_error: json.JSONDecodeError | None = None
    for json_candidate_text in direct_parse_candidates:
        try:
            extracted_json_payload = json.loads(json_candidate_text)
        except json.JSONDecodeError as json_decode_error:
            last_json_decode_error = json_decode_error
            continue

        if not isinstance(extracted_json_payload, dict):
            continue

        comparison_shopping_candidates = extracted_json_payload.get(
            "comparison_shopping_candidates",
        )
        if comparison_shopping_candidates is None:
            continue

        if not isinstance(comparison_shopping_candidates, list):
            raise RuntimeError(
                "Ollama JSON payload contained a non-array "
                "comparison_shopping_candidates field."
            )

        return extracted_json_payload

    raise RuntimeError(
        "Ollama response did not contain parseable comparison-shopping JSON."
    ) from last_json_decode_error


def build_ollama_comparison_shopping_system_instruction() -> str:
    """
    Build an Ollama-specific system instruction for structured candidate output.

    Smaller local models benefit from explicit schema reminders and strict
    formatting constraints beyond the shared provider instruction.
    """
    shared_system_instruction = build_comparison_shopping_system_instruction()
    ollama_structured_output_constraints = (
        "\n\nOLLAMA OUTPUT CONTRACT:\n"
        "- Respond with a single JSON object only.\n"
        "- The root object must contain exactly one key: comparison_shopping_candidates.\n"
        "- comparison_shopping_candidates must be a JSON array of candidate objects.\n"
        "- Each candidate object must include: candidate_display_name, "
        "recommendation_summary, estimated_cost_amount, cost_currency_code, "
        "estimated_distance_kilometers, observed_quality_rating, "
        "observed_rating_count.\n"
        "- Use null for unknown numeric or currency fields instead of omitting keys.\n"
        "- Do not wrap the JSON in markdown code fences.\n"
        "- Do not include explanatory text before or after the JSON object."
    )
    return f"{shared_system_instruction}{ollama_structured_output_constraints}"


class OllamaProviderAdapter(BaseProvider):
    """
    Adapter for local Ollama comparison-shopping recommendation requests.

    Interfaces with a locally running Ollama instance to produce structured
    comparison-shopping candidates without cloud API dependencies.
    """

    PROVIDER_IDENTIFIER: str = "ollama"

    def __init__(
        self,
        ollama_base_url: str,
        ollama_comparison_shopping_model_name: str,
    ) -> None:
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._ollama_comparison_shopping_model_name = ollama_comparison_shopping_model_name
        self._ollama_async_client = AsyncClient(host=self._ollama_base_url)

    @property
    def provider_identifier(self) -> str:
        return self.PROVIDER_IDENTIFIER

    async def execute_comparison_shopping_query(
        self,
        user_search_query: str,
    ) -> ProviderResponsePayload:
        """Query a local Ollama model for structured comparison-shopping recommendations."""
        ollama_system_instruction = build_ollama_comparison_shopping_system_instruction()
        ollama_user_prompt = build_comparison_shopping_user_prompt(
            user_search_query=user_search_query,
        )

        logger.info(
            "Invoking Ollama comparison-shopping model '%s' at '%s' for query: %s",
            self._ollama_comparison_shopping_model_name,
            self._ollama_base_url,
            user_search_query,
        )

        ollama_chat_response = await self._ollama_async_client.chat(
            model=self._ollama_comparison_shopping_model_name,
            messages=[
                {"role": "system", "content": ollama_system_instruction},
                {"role": "user", "content": ollama_user_prompt},
            ],
        )

        local_model_response_text = ollama_chat_response.message.content
        if not local_model_response_text:
            raise RuntimeError(
                "Ollama returned an empty message for the comparison-shopping query."
            )

        structured_ollama_candidate_payload = extract_json_payload_from_local_model_response(
            local_model_response_text=local_model_response_text,
        )

        raw_recommendation_text = serialize_structured_candidates_to_text(
            structured_candidate_payload=structured_ollama_candidate_payload,
        )

        logger.info(
            "Ollama provider returned structured candidates for query: %s",
            user_search_query,
        )

        return ProviderResponsePayload(
            provider_identifier=self.provider_identifier,
            raw_recommendation_text=raw_recommendation_text,
            provider_model_name=self._ollama_comparison_shopping_model_name,
        )
