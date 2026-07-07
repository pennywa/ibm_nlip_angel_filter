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
        populate_by_name=True,
    )

    openai_api_key: str = Field(
        ...,
        validation_alias="OPENAI_API_KEY",
        min_length=1,
        description="OpenAI API authentication key.",
    )
    gemini_api_key: str = Field(
        ...,
        validation_alias="GEMINI_API_KEY",
        min_length=1,
        description="Google Gemini API authentication key.",
    )
    watson_api_key: str = Field(
        ...,
        validation_alias="WATSON_API_KEY",
        min_length=1,
        description="IBM Cloud IAM API key for Watsonx access.",
    )
    watson_project_id: str = Field(
        ...,
        validation_alias="WATSON_PROJECT_ID",
        min_length=1,
        description="IBM Watsonx project identifier for text generation requests.",
    )
    watson_api_base_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        validation_alias="WATSON_API_BASE_URL",
        description="Regional Watsonx API base URL.",
    )
    watson_foundation_model_id: str = Field(
        default="ibm/granite-3-8b-instruct",
        validation_alias="WATSON_FOUNDATION_MODEL_ID",
        description="Watsonx foundation model identifier for comparison-shopping queries.",
    )


@lru_cache
def get_provider_settings() -> ProviderSettings:
    """Return a cached ProviderSettings instance loaded from the environment."""
    return ProviderSettings()
