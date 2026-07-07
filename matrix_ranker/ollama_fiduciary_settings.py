"""Configuration for the local Ollama fiduciary validation layer."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaFiduciarySettings(BaseSettings):
    """Environment-backed settings for the Ollama fiduciary validator."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias="OLLAMA_BASE_URL",
        description="Base URL for the local Ollama inference server.",
    )
    ollama_fiduciary_model_name: str = Field(
        default="llama3.2",
        validation_alias="OLLAMA_FIDUCIARY_MODEL_NAME",
        description="Local Ollama model used for fiduciary cross-examination.",
    )


@lru_cache
def get_ollama_fiduciary_settings() -> OllamaFiduciarySettings:
    """Return a cached OllamaFiduciarySettings instance loaded from the environment."""
    return OllamaFiduciarySettings()
