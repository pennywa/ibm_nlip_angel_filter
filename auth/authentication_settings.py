"""Pydantic-settings configuration for the GitHub OAuth and JWT auth gateway."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthenticationSettings(BaseSettings):
    """Environment-backed settings for OAuth, JWT signing, and username allowlisting."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    jwt_secret_key: str = Field(
        ...,
        validation_alias="JWT_SECRET_KEY",
        min_length=32,
        description="HMAC signing secret used to mint and verify federator JWT tokens.",
    )
    allowed_github_usernames: str = Field(
        ...,
        validation_alias="ALLOWED_GITHUB_USERNAMES",
        min_length=1,
        description="Comma-separated GitHub usernames permitted to authenticate.",
    )
    github_oauth_client_id: str = Field(
        ...,
        validation_alias="GITHUB_OAUTH_CLIENT_ID",
        min_length=1,
        description="GitHub OAuth application client identifier.",
    )
    github_oauth_client_secret: str = Field(
        ...,
        validation_alias="GITHUB_OAUTH_CLIENT_SECRET",
        min_length=1,
        description="GitHub OAuth application client secret.",
    )
    github_oauth_redirect_uri: str = Field(
        ...,
        validation_alias="GITHUB_OAUTH_REDIRECT_URI",
        min_length=1,
        description="Registered OAuth callback URL for the federator application.",
    )
    jwt_token_expiration_seconds: int = Field(
        default=3600,
        validation_alias="JWT_TOKEN_EXPIRATION_SECONDS",
        ge=60,
        le=86400,
        description="Lifetime of issued JWT access tokens in seconds.",
    )

    @field_validator("allowed_github_usernames")
    @classmethod
    def validate_allowed_github_usernames_not_blank(
        cls,
        allowed_github_usernames_value: str,
    ) -> str:
        """Ensure at least one non-empty username exists in the allowlist string."""
        parsed_usernames = [
            github_username.strip()
            for github_username in allowed_github_usernames_value.split(",")
            if github_username.strip()
        ]
        if not parsed_usernames:
            raise ValueError(
                "ALLOWED_GITHUB_USERNAMES must contain at least one GitHub username."
            )
        return allowed_github_usernames_value

    def parse_allowed_github_username_set(self) -> set[str]:
        """Return the allowlisted GitHub usernames as a normalized lowercase lookup set."""
        return {
            github_username.strip().lower()
            for github_username in self.allowed_github_usernames.split(",")
            if github_username.strip()
        }


@lru_cache
def get_authentication_settings() -> AuthenticationSettings:
    """Return a cached AuthenticationSettings instance loaded from the environment."""
    return AuthenticationSettings()
