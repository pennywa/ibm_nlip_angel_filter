"""Environment-backed configuration for external AI provider adapters."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderSettings(BaseSettings):
    """Credentials and endpoint configuration for federator provider adapters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(
        default="",
        description="OpenAI API authentication key.",
    )
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API authentication key.",
    )
    watson_api_key: str = Field(
        default="",
        description="IBM Cloud IAM API key for Watsonx access.",
    )
    watson_project_id: str = Field(
        default="",
        description="IBM Watsonx project identifier for text generation requests.",
    )
    watson_api_base_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        description="Regional Watsonx API base URL.",
    )
    watson_foundation_model_id: str = Field(
        default="meta-llama/llama-3-8b-instruct",
        description="Lightweight Watsonx foundation model for comparison-shopping queries.",
    )
    provider_query_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Per-provider wall-clock timeout for comparison-shopping fan-out.",
    )


@lru_cache
def get_provider_settings() -> ProviderSettings:
    """Return a cached ProviderSettings instance loaded from the environment."""
    return ProviderSettings()
