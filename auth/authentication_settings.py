"""Pydantic-settings configuration for the GitHub OAuth and JWT auth gateway."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthenticationSettings(BaseSettings):
    """Environment-backed settings for OAuth, JWT signing, and username allowlisting."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    jwt_secret_key: str = Field(
        ...,
        description="HMAC signing secret used to mint and verify federator JWT tokens.",
    )
    allowed_github_usernames: str = Field(
        ...,
        description="Comma-separated GitHub usernames permitted to authenticate.",
    )
    github_oauth_client_id: str = Field(
        default="",
        description="GitHub OAuth application client identifier.",
    )
    github_oauth_client_secret: str = Field(
        default="",
        description="GitHub OAuth application client secret.",
    )
    github_oauth_redirect_uri: str = Field(
        default="",
        description="Registered OAuth callback URL for the federator application.",
    )
    jwt_token_expiration_seconds: int = Field(
        default=3600,
        description="Lifetime of issued JWT access tokens in seconds.",
    )

    def parse_allowed_github_username_set(self) -> set[str]:
        """Return the allowlisted GitHub usernames as a normalized lookup set."""
        return {
            github_username.strip().lower()
            for github_username in self.allowed_github_usernames.split(",")
            if github_username.strip()
        }


@lru_cache
def get_authentication_settings() -> AuthenticationSettings:
    """Return a cached AuthenticationSettings instance loaded from the environment."""
    return AuthenticationSettings()
