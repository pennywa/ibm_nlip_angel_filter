"""Environment-backed configuration for external AI provider adapters."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_API_KEY_PREFIX: str = "replace-with-"


class ProviderSettings(BaseSettings):
    """Credentials and endpoint configuration for federator provider adapters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    application_environment: str = Field(
        default="production",
        validation_alias="APPLICATION_ENVIRONMENT",
        description="Runtime environment label controlling local Ollama orchestration.",
    )
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API authentication key.",
    )
    gemini_api_key: str = Field(
        default="",
        validation_alias="GEMINI_API_KEY",
        description="Google Gemini API authentication key.",
    )
    watson_api_key: str = Field(
        default="",
        validation_alias="WATSON_API_KEY",
        description="IBM Cloud IAM API key for Watsonx access.",
    )
    watson_project_id: str = Field(
        default="",
        validation_alias="WATSON_PROJECT_ID",
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
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias="OLLAMA_BASE_URL",
        description="Base URL for the local Ollama inference server.",
    )
    ollama_comparison_shopping_model_name: str = Field(
        default="llama3.2",
        validation_alias="OLLAMA_COMPARISON_SHOPPING_MODEL_NAME",
        description="Local Ollama model used for comparison-shopping candidate generation.",
    )

    @property
    def is_development_environment(self) -> bool:
        """Return True when APPLICATION_ENVIRONMENT is set to development."""
        return self.application_environment.strip().lower() == "development"

    @staticmethod
    def cloud_provider_credential_is_configured(credential_value: str) -> bool:
        """Return True when a cloud provider credential is present and not a placeholder."""
        normalized_credential_value = credential_value.strip()
        if not normalized_credential_value:
            return False
        if normalized_credential_value.lower().startswith(PLACEHOLDER_API_KEY_PREFIX):
            return False
        return True

    @property
    def openai_provider_is_configured(self) -> bool:
        """Return True when OpenAI credentials are available for fan-out."""
        return self.cloud_provider_credential_is_configured(self.openai_api_key)

    @property
    def gemini_provider_is_configured(self) -> bool:
        """Return True when Gemini credentials are available for fan-out."""
        return self.cloud_provider_credential_is_configured(self.gemini_api_key)

    @property
    def watson_provider_is_configured(self) -> bool:
        """Return True when Watsonx credentials are available for fan-out."""
        return (
            self.cloud_provider_credential_is_configured(self.watson_api_key)
            and self.cloud_provider_credential_is_configured(self.watson_project_id)
        )

    @property
    def any_cloud_provider_is_configured(self) -> bool:
        """Return True when at least one cloud provider adapter can be registered."""
        return (
            self.openai_provider_is_configured
            or self.gemini_provider_is_configured
            or self.watson_provider_is_configured
        )


@lru_cache
def get_provider_settings() -> ProviderSettings:
    """Return a cached ProviderSettings instance loaded from the environment."""
    return ProviderSettings()
