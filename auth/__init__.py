"""GitHub OAuth gateway, JWT lifecycle, and allowlist enforcement."""

from auth.allowlist_validation_middleware import AllowlistValidationMiddleware
from auth.authentication_settings import AuthenticationSettings, get_authentication_settings
from auth.github_oauth_routes import github_oauth_router
from auth.jwt_token_service import (
    JWT_GITHUB_USERNAME_CLAIM,
    JWT_SUBJECT_CLAIM,
    JwtTokenValidationError,
    decode_and_validate_federator_access_token,
    extract_bearer_token_from_authorization_header,
    mint_federator_access_token,
)

__all__ = [
    "AllowlistValidationMiddleware",
    "AuthenticationSettings",
    "JWT_GITHUB_USERNAME_CLAIM",
    "JWT_SUBJECT_CLAIM",
    "JwtTokenValidationError",
    "decode_and_validate_federator_access_token",
    "extract_bearer_token_from_authorization_header",
    "get_authentication_settings",
    "github_oauth_router",
    "mint_federator_access_token",
]
