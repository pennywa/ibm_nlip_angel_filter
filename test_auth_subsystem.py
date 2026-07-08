"""
Isolated sanity checks for the auth/ security subsystem.

Validates AuthenticationSettings constraints, JWT mint/decode round-trips, and
simulated allowlist gate behavior without starting the federator application.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

from pydantic import ValidationError

from auth.authentication_settings import AuthenticationSettings
from auth.jwt_token_service import (
    JWT_GITHUB_USERNAME_CLAIM,
    JWT_SUBJECT_CLAIM,
    decode_and_validate_federator_access_token,
    mint_federator_access_token,
)

AUTHORIZED_GITHUB_USERNAME: str = "pennywa"
UNAUTHORIZED_GITHUB_USERNAME: str = "not-on-allowlist-user"

VALID_AUTHENTICATION_SETTINGS_KWARGS: dict[str, Any] = {
    "jwt_secret_key": "x" * 32,
    "allowed_github_usernames": f"{AUTHORIZED_GITHUB_USERNAME},other-user",
    "github_oauth_client_id": "test-github-oauth-client-id",
    "github_oauth_client_secret": "test-github-oauth-client-secret",
    "github_oauth_redirect_uri": "http://127.0.0.1:8000/auth/github/callback",
}


def _run_sanity_check(
    sanity_check_name: str,
    sanity_check_callable: Callable[[], None],
) -> bool:
    """Execute a single sanity check and print a pass/fail line."""
    try:
        sanity_check_callable()
    except Exception as sanity_check_error:
        print(f"FAIL  {sanity_check_name}: {sanity_check_error}")
        return False

    print(f"PASS  {sanity_check_name}")
    return True


def _build_valid_authentication_settings() -> AuthenticationSettings:
    """Return AuthenticationSettings populated with valid test credentials."""
    return AuthenticationSettings(**VALID_AUTHENTICATION_SETTINGS_KWARGS)


def test_authentication_settings_rejects_short_jwt_secret_key() -> None:
    """JWT_SECRET_KEY shorter than 32 characters must raise ValidationError."""
    invalid_settings_kwargs = {
        **VALID_AUTHENTICATION_SETTINGS_KWARGS,
        "jwt_secret_key": "too-short-secret-key",
    }

    try:
        AuthenticationSettings(**invalid_settings_kwargs)
    except ValidationError:
        return

    raise AssertionError(
        "Expected ValidationError when jwt_secret_key is under 32 characters."
    )


def test_authentication_settings_rejects_empty_allowlist() -> None:
    """An empty ALLOWED_GITHUB_USERNAMES value must raise ValidationError."""
    invalid_settings_kwargs = {
        **VALID_AUTHENTICATION_SETTINGS_KWARGS,
        "allowed_github_usernames": "   ,  , ",
    }

    try:
        AuthenticationSettings(**invalid_settings_kwargs)
    except ValidationError:
        return

    raise AssertionError(
        "Expected ValidationError when allowed_github_usernames parses to an empty set."
    )


def test_authentication_settings_parses_lowercase_allowlist_set() -> None:
    """parse_allowed_github_username_set() must normalize usernames to lowercase."""
    authentication_settings = _build_valid_authentication_settings()

    allowed_github_username_set = (
        authentication_settings.parse_allowed_github_username_set()
    )

    assert allowed_github_username_set == {
        AUTHORIZED_GITHUB_USERNAME.lower(),
        "other-user",
    }


def test_jwt_token_service_mint_and_decode_round_trip() -> None:
    """Minting and decoding a token must preserve the expected JWT claim payload."""
    authentication_settings = _build_valid_authentication_settings()

    federator_access_token = mint_federator_access_token(
        authenticated_github_username=AUTHORIZED_GITHUB_USERNAME,
        authentication_settings=authentication_settings,
    )

    decoded_jwt_claims = decode_and_validate_federator_access_token(
        federator_access_token=federator_access_token,
        authentication_settings=authentication_settings,
    )

    assert decoded_jwt_claims[JWT_SUBJECT_CLAIM] == AUTHORIZED_GITHUB_USERNAME
    assert decoded_jwt_claims[JWT_GITHUB_USERNAME_CLAIM] == AUTHORIZED_GITHUB_USERNAME
    assert isinstance(decoded_jwt_claims["iat"], int)
    assert isinstance(decoded_jwt_claims["exp"], int)
    assert decoded_jwt_claims["exp"] > decoded_jwt_claims["iat"]


def simulate_allowlist_validation_check(
    authenticated_github_username: str,
    authentication_settings: AuthenticationSettings,
) -> tuple[bool, str | None]:
    """
    Mirror the allowlist gate from AllowlistValidationMiddleware without ASGI.

    Returns:
        A tuple of (is_authorized, rejection_detail_message).
    """
    allowed_github_username_set = (
        authentication_settings.parse_allowed_github_username_set()
    )
    normalized_github_username = authenticated_github_username.lower()

    if normalized_github_username not in allowed_github_username_set:
        return (
            False,
            "GitHub username is not authorized to access this resource.",
        )

    return True, None


def test_simulated_allowlist_validation_rejects_unauthorized_username() -> None:
    """A non-allowlisted GitHub username must be flagged as unauthorized."""
    authentication_settings = _build_valid_authentication_settings()

    federator_access_token = mint_federator_access_token(
        authenticated_github_username=UNAUTHORIZED_GITHUB_USERNAME,
        authentication_settings=authentication_settings,
    )

    decoded_jwt_claims = decode_and_validate_federator_access_token(
        federator_access_token=federator_access_token,
        authentication_settings=authentication_settings,
    )

    authenticated_github_username = decoded_jwt_claims[JWT_GITHUB_USERNAME_CLAIM]

    is_authorized, rejection_detail_message = simulate_allowlist_validation_check(
        authenticated_github_username=authenticated_github_username,
        authentication_settings=authentication_settings,
    )

    assert is_authorized is False
    assert rejection_detail_message is not None
    assert "not authorized" in rejection_detail_message.lower()


def test_simulated_allowlist_validation_accepts_authorized_username() -> None:
    """An allowlisted GitHub username must pass the simulated gate."""
    authentication_settings = _build_valid_authentication_settings()

    is_authorized, rejection_detail_message = simulate_allowlist_validation_check(
        authenticated_github_username=AUTHORIZED_GITHUB_USERNAME,
        authentication_settings=authentication_settings,
    )

    assert is_authorized is True
    assert rejection_detail_message is None


def main() -> int:
    """Run all auth subsystem sanity checks and return a process exit code."""
    sanity_checks: list[tuple[str, Callable[[], None]]] = [
        (
            "AuthenticationSettings rejects short JWT_SECRET_KEY",
            test_authentication_settings_rejects_short_jwt_secret_key,
        ),
        (
            "AuthenticationSettings rejects empty allowlist",
            test_authentication_settings_rejects_empty_allowlist,
        ),
        (
            "AuthenticationSettings parses lowercase allowlist set",
            test_authentication_settings_parses_lowercase_allowlist_set,
        ),
        (
            "JWT token mint/decode preserves expected claims",
            test_jwt_token_service_mint_and_decode_round_trip,
        ),
        (
            "Simulated allowlist gate rejects unauthorized username",
            test_simulated_allowlist_validation_rejects_unauthorized_username,
        ),
        (
            "Simulated allowlist gate accepts authorized username",
            test_simulated_allowlist_validation_accepts_authorized_username,
        ),
    ]

    passed_sanity_check_count = sum(
        _run_sanity_check(sanity_check_name, sanity_check_callable)
        for sanity_check_name, sanity_check_callable in sanity_checks
    )
    total_sanity_check_count = len(sanity_checks)

    print(
        f"\n{passed_sanity_check_count}/{total_sanity_check_count} "
        "auth subsystem sanity checks passed."
    )

    return 0 if passed_sanity_check_count == total_sanity_check_count else 1


if __name__ == "__main__":
    sys.exit(main())
