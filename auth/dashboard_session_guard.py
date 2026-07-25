"""Cookie-first JWT session validation for the federator HTML dashboard."""

from __future__ import annotations

from dataclasses import dataclass

from starlette.requests import Request

from auth.authentication_settings import AuthenticationSettings, get_authentication_settings
from auth.jwt_token_service import (
    JWT_GITHUB_USERNAME_CLAIM,
    JwtTokenValidationError,
    decode_and_validate_federator_access_token,
    extract_bearer_token_from_authorization_header,
    extract_federator_access_token_from_request_cookies,
)


@dataclass(frozen=True)
class DashboardSessionContext:
    """Validated dashboard session for an allowlisted GitHub identity."""

    authenticated_github_username: str


@dataclass(frozen=True)
class DashboardAccessDenied:
    """Dashboard access rejection with a human-readable reason."""

    denial_reason: str


def resolve_dashboard_session_from_request(
    request: Request,
    authentication_settings: AuthenticationSettings | None = None,
) -> DashboardSessionContext | DashboardAccessDenied:
    """
    Validate the inbound browser session JWT and enforce allowlist membership.

    Cookie-based credentials are preferred for dashboard navigation; Bearer
    tokens are accepted as a fallback for programmatic access.
    """
    resolved_authentication_settings = (
        authentication_settings or get_authentication_settings()
    )
    allowed_github_username_set = (
        resolved_authentication_settings.parse_allowed_github_username_set()
    )

    federator_access_token = extract_federator_access_token_from_request_cookies(
        request_cookies=dict(request.cookies),
    )
    if federator_access_token is None:
        federator_access_token = extract_bearer_token_from_authorization_header(
            authorization_header_value=request.headers.get("Authorization"),
        )

    if federator_access_token is None:
        return DashboardAccessDenied(
            denial_reason="No valid session JWT cookie was found for this browser.",
        )

    try:
        decoded_jwt_claims = decode_and_validate_federator_access_token(
            federator_access_token=federator_access_token,
            authentication_settings=resolved_authentication_settings,
        )
    except JwtTokenValidationError as jwt_validation_error:
        return DashboardAccessDenied(denial_reason=str(jwt_validation_error))

    authenticated_github_username = decoded_jwt_claims[JWT_GITHUB_USERNAME_CLAIM]
    normalized_github_username = authenticated_github_username.lower()

    if normalized_github_username not in allowed_github_username_set:
        return DashboardAccessDenied(
            denial_reason=(
                "GitHub username is not on the federator allowlist "
                f"({authenticated_github_username})."
            ),
        )

    return DashboardSessionContext(
        authenticated_github_username=authenticated_github_username,
    )
