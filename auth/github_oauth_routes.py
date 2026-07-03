"""Structural FastAPI routing boilerplate for the GitHub OAuth2 authentication loop."""

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse

from auth.authentication_settings import AuthenticationSettings, get_authentication_settings

github_oauth_router = APIRouter(
    prefix="/auth/github",
    tags=["GitHub OAuth Authentication"],
)


@github_oauth_router.get(
    "/login",
    summary="Initiate GitHub OAuth authorization",
    response_class=RedirectResponse,
)
async def initiate_github_oauth_authorization(
    authentication_settings: AuthenticationSettings = Depends(get_authentication_settings),
) -> RedirectResponse:
    """
    Redirect the user to GitHub's OAuth authorize endpoint.

    Implementation deferred: construct the authorization URL from
    ``github_oauth_client_id`` and ``github_oauth_redirect_uri``.
    """
    raise NotImplementedError(
        "GitHub OAuth authorization redirect is not yet implemented."
    )


@github_oauth_router.get(
    "/callback",
    summary="Handle GitHub OAuth callback and exchange authorization code",
)
async def handle_github_oauth_callback(
    authorization_code: str,
    authentication_settings: AuthenticationSettings = Depends(get_authentication_settings),
) -> dict[str, Any]:
    """
    Exchange the GitHub authorization code for an access token, resolve the
    authenticated GitHub username, validate against the allowlist, and mint a JWT.

    Implementation deferred: code exchange, profile lookup, allowlist gate,
    and JWT issuance will be wired in a subsequent phase.
    """
    raise NotImplementedError(
        "GitHub OAuth callback and code exchange are not yet implemented."
    )


@github_oauth_router.get(
    "/session",
    summary="Return the authenticated session profile for a valid JWT",
)
async def retrieve_authenticated_session_profile(
    request: Request,
    authentication_settings: AuthenticationSettings = Depends(get_authentication_settings),
) -> dict[str, Any]:
    """
    Validate the Bearer JWT on the incoming request and return session claims.

    Implementation deferred: JWT verification and claim extraction.
    """
    raise NotImplementedError(
        "Authenticated session profile retrieval is not yet implemented."
    )
