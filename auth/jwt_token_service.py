"""HS256 JWT minting and verification for the federator auth gateway."""

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from jwt.exceptions import InvalidTokenError

from auth.authentication_settings import AuthenticationSettings

JWT_SIGNING_ALGORITHM: str = "HS256"
JWT_GITHUB_USERNAME_CLAIM: str = "github_username"
JWT_SUBJECT_CLAIM: str = "sub"


class JwtTokenValidationError(Exception):
    """Raised when an inbound JWT fails signature, expiry, or claim validation."""


def mint_federator_access_token(
    authenticated_github_username: str,
    authentication_settings: AuthenticationSettings,
) -> str:
    """
    Mint a signed HS256 JWT for an allowlisted GitHub identity.

    Args:
        authenticated_github_username: GitHub login name confirmed via OAuth.
        authentication_settings: Environment-backed signing configuration.

    Returns:
        Encoded JWT string suitable for ``Authorization: Bearer`` headers.
    """
    issued_at_utc = datetime.now(timezone.utc)
    expiration_at_utc = issued_at_utc + timedelta(
        seconds=authentication_settings.jwt_token_expiration_seconds,
    )

    jwt_payload: dict[str, Any] = {
        JWT_SUBJECT_CLAIM: authenticated_github_username,
        JWT_GITHUB_USERNAME_CLAIM: authenticated_github_username,
        "iat": issued_at_utc,
        "exp": expiration_at_utc,
    }

    return jwt.encode(
        jwt_payload,
        authentication_settings.jwt_secret_key,
        algorithm=JWT_SIGNING_ALGORITHM,
    )


def decode_and_validate_federator_access_token(
    federator_access_token: str,
    authentication_settings: AuthenticationSettings,
) -> dict[str, Any]:
    """
    Decode and validate an inbound federator JWT.

    Args:
        federator_access_token: Raw JWT from the Authorization header.
        authentication_settings: Environment-backed verification configuration.

    Returns:
        Validated JWT claims dictionary.

    Raises:
        JwtTokenValidationError: When the token is invalid or expired.
    """
    try:
        decoded_jwt_claims = jwt.decode(
            federator_access_token,
            authentication_settings.jwt_secret_key,
            algorithms=[JWT_SIGNING_ALGORITHM],
        )
    except InvalidTokenError as invalid_token_error:
        raise JwtTokenValidationError(
            f"Federator access token validation failed: {invalid_token_error}"
        ) from invalid_token_error

    authenticated_github_username = decoded_jwt_claims.get(JWT_GITHUB_USERNAME_CLAIM)
    if not authenticated_github_username or not isinstance(
        authenticated_github_username,
        str,
    ):
        raise JwtTokenValidationError(
            "Federator access token is missing a valid github_username claim."
        )

    return decoded_jwt_claims


def extract_bearer_token_from_authorization_header(
    authorization_header_value: str | None,
) -> str | None:
    """
    Parse a Bearer token from the HTTP Authorization header value.

    Returns:
        The raw JWT string, or None when the header is absent or malformed.
    """
    if not authorization_header_value:
        return None

    authorization_header_parts = authorization_header_value.split(" ", maxsplit=1)
    if len(authorization_header_parts) != 2:
        return None

    authorization_scheme, bearer_token_value = authorization_header_parts
    if authorization_scheme.lower() != "bearer" or not bearer_token_value.strip():
        return None

    return bearer_token_value.strip()
