"""FastAPI routing for the GitHub OAuth2 authentication loop and JWT session management."""

import logging
import secrets
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse

from auth.authentication_settings import AuthenticationSettings, get_authentication_settings
from auth.jwt_token_service import (
    FEDERATOR_ACCESS_TOKEN_COOKIE_NAME,
    JWT_GITHUB_USERNAME_CLAIM,
    JwtTokenValidationError,
    decode_and_validate_federator_access_token,
    extract_bearer_token_from_authorization_header,
    mint_federator_access_token,
)

logger = logging.getLogger(__name__)

github_oauth_router = APIRouter(
    prefix="/auth/github",
    tags=["GitHub OAuth Authentication"],
)

GITHUB_OAUTH_AUTHORIZE_URL: str = "https://github.com/login/oauth/authorize"
GITHUB_OAUTH_ACCESS_TOKEN_URL: str = "https://github.com/login/oauth/access_token"
GITHUB_USER_PROFILE_API_URL: str = "https://api.github.com/user"
GITHUB_OAUTH_USER_SCOPE: str = "read:user"
OAUTH_CSRF_STATE_COOKIE_NAME: str = "github_oauth_csrf_state"


def _build_github_oauth_authorization_url(
    authentication_settings: AuthenticationSettings,
    oauth_csrf_state_token: str,
) -> str:
    """Construct the GitHub OAuth authorize redirect URL with CSRF state."""
    oauth_query_parameters = {
        "client_id": authentication_settings.github_oauth_client_id,
        "redirect_uri": authentication_settings.github_oauth_redirect_uri,
        "scope": GITHUB_OAUTH_USER_SCOPE,
        "state": oauth_csrf_state_token,
    }
    return f"{GITHUB_OAUTH_AUTHORIZE_URL}?{urlencode(oauth_query_parameters)}"


async def _exchange_github_authorization_code_for_access_token(
    authorization_code: str,
    authentication_settings: AuthenticationSettings,
) -> str:
    """Exchange a temporary GitHub authorization code for an OAuth access token."""
    token_exchange_payload = {
        "client_id": authentication_settings.github_oauth_client_id,
        "client_secret": authentication_settings.github_oauth_client_secret,
        "code": authorization_code,
        "redirect_uri": authentication_settings.github_oauth_redirect_uri,
    }

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        token_exchange_response = await http_client.post(
            GITHUB_OAUTH_ACCESS_TOKEN_URL,
            headers={"Accept": "application/json"},
            data=token_exchange_payload,
        )
        token_exchange_response.raise_for_status()
        token_exchange_body = token_exchange_response.json()

    github_access_token = token_exchange_body.get("access_token")
    if not github_access_token:
        oauth_error_description = token_exchange_body.get(
            "error_description",
            "GitHub did not return an access token.",
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"GitHub OAuth token exchange failed: {oauth_error_description}",
        )

    return github_access_token


async def _fetch_authenticated_github_username(
    github_access_token: str,
) -> str:
    """Retrieve the GitHub login name for the authenticated OAuth identity."""
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        github_user_profile_response = await http_client.get(
            GITHUB_USER_PROFILE_API_URL,
            headers={
                "Authorization": f"Bearer {github_access_token}",
                "Accept": "application/vnd.github+json",
            },
        )
        github_user_profile_response.raise_for_status()
        github_user_profile_payload = github_user_profile_response.json()

    authenticated_github_username = github_user_profile_payload.get("login")
    if not authenticated_github_username:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="GitHub user profile response did not include a login username.",
        )

    return authenticated_github_username


@github_oauth_router.get(
    "/login",
    summary="Initiate GitHub OAuth authorization",
    response_class=RedirectResponse,
)
async def initiate_github_oauth_authorization(
    request: Request,
    authentication_settings: AuthenticationSettings = Depends(get_authentication_settings),
) -> RedirectResponse:
    """Redirect the user to GitHub's OAuth authorize endpoint with a CSRF state token."""
    oauth_csrf_state_token = secrets.token_urlsafe(32)
    github_authorization_url = _build_github_oauth_authorization_url(
        authentication_settings=authentication_settings,
        oauth_csrf_state_token=oauth_csrf_state_token,
    )

    oauth_redirect_response = RedirectResponse(
        url=github_authorization_url,
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    )
    oauth_redirect_response.set_cookie(
        key=OAUTH_CSRF_STATE_COOKIE_NAME,
        value=oauth_csrf_state_token,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        max_age=600,
    )
    return oauth_redirect_response


@github_oauth_router.get(
    "/callback",
    summary="Handle GitHub OAuth callback and exchange authorization code",
    response_class=RedirectResponse,
)
async def handle_github_oauth_callback(
    request: Request,
    authorization_code: str = Query(
        ...,
        alias="code",
        description="Temporary authorization code issued by GitHub OAuth.",
    ),
    oauth_csrf_state_parameter: str = Query(
        ...,
        alias="state",
        description="CSRF state token echoed by GitHub OAuth.",
    ),
    authentication_settings: AuthenticationSettings = Depends(get_authentication_settings),
) -> RedirectResponse:
    """
    Exchange the GitHub authorization code, validate the allowlist, and mint a JWT.

    Sets a browser session cookie and redirects to the federator dashboard.
    """
    stored_oauth_csrf_state_token = request.cookies.get(OAUTH_CSRF_STATE_COOKIE_NAME)
    if (
        not stored_oauth_csrf_state_token
        or not secrets.compare_digest(
            stored_oauth_csrf_state_token,
            oauth_csrf_state_parameter,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OAuth CSRF state validation failed.",
        )

    github_access_token = await _exchange_github_authorization_code_for_access_token(
        authorization_code=authorization_code,
        authentication_settings=authentication_settings,
    )

    authenticated_github_username = await _fetch_authenticated_github_username(
        github_access_token=github_access_token,
    )

    allowed_github_username_set = (
        authentication_settings.parse_allowed_github_username_set()
    )
    normalized_github_username = authenticated_github_username.lower()

    if normalized_github_username not in allowed_github_username_set:
        logger.warning(
            "GitHub username '%s' rejected: not on allowlist.",
            authenticated_github_username,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="GitHub username is not on the federator allowlist.",
        )

    federator_access_token = mint_federator_access_token(
        authenticated_github_username=authenticated_github_username,
        authentication_settings=authentication_settings,
    )

    dashboard_redirect_response = RedirectResponse(
        url="/dashboard",
        status_code=status.HTTP_303_SEE_OTHER,
    )
    dashboard_redirect_response.set_cookie(
        key=FEDERATOR_ACCESS_TOKEN_COOKIE_NAME,
        value=federator_access_token,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        max_age=authentication_settings.jwt_token_expiration_seconds,
    )
    dashboard_redirect_response.delete_cookie(key=OAUTH_CSRF_STATE_COOKIE_NAME)

    return dashboard_redirect_response


@github_oauth_router.get(
    "/session",
    summary="Return the authenticated session profile for a valid JWT",
)
async def retrieve_authenticated_session_profile(
    request: Request,
    authentication_settings: AuthenticationSettings = Depends(get_authentication_settings),
) -> dict[str, Any]:
    """Validate the Bearer JWT on the incoming request and return active session claims."""
    authorization_header_value = request.headers.get("Authorization")
    federator_access_token = extract_bearer_token_from_authorization_header(
        authorization_header_value=authorization_header_value,
    )

    if federator_access_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header with Bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        decoded_jwt_claims = decode_and_validate_federator_access_token(
            federator_access_token=federator_access_token,
            authentication_settings=authentication_settings,
        )
    except JwtTokenValidationError as jwt_validation_error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(jwt_validation_error),
            headers={"WWW-Authenticate": "Bearer"},
        ) from jwt_validation_error

    authenticated_github_username = decoded_jwt_claims[JWT_GITHUB_USERNAME_CLAIM]

    return {
        "session_active": True,
        "github_username": authenticated_github_username,
        "issued_at_unix_timestamp": decoded_jwt_claims.get("iat"),
        "expires_at_unix_timestamp": decoded_jwt_claims.get("exp"),
    }
