"""ASGI middleware gate for JWT authentication and GitHub username allowlist enforcement."""

import json
import logging
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from auth.authentication_settings import get_authentication_settings
from auth.jwt_token_service import (
    JWT_GITHUB_USERNAME_CLAIM,
    JwtTokenValidationError,
    decode_and_validate_federator_access_token,
    extract_bearer_token_from_authorization_header,
)

logger = logging.getLogger(__name__)


class AllowlistValidationMiddleware(BaseHTTPMiddleware):
    """
    Intercept protected requests and enforce JWT authentication plus allowlist screening.

    Unauthorized traffic receives a strict ``401 Unauthorized`` response.
    """

    PROTECTED_PATH_PREFIXES: tuple[str, ...] = (
        "/federator/",
        "/comparison-shopping/",
    )

    EXEMPT_PATH_PREFIXES: tuple[str, ...] = (
        "/auth/github/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    )

    REQUEST_STATE_GITHUB_USERNAME_KEY: str = "authenticated_github_username"

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Decode inbound JWT credentials and block unauthorized protected requests."""
        if self._is_exempt_request_path(request.url.path):
            return await call_next(request)

        if not self._is_protected_request_path(request.url.path):
            return await call_next(request)

        authentication_settings = get_authentication_settings()
        allowed_github_username_set = (
            authentication_settings.parse_allowed_github_username_set()
        )

        authorization_header_value = request.headers.get("Authorization")
        federator_access_token = extract_bearer_token_from_authorization_header(
            authorization_header_value=authorization_header_value,
        )

        if federator_access_token is None:
            return self._build_unauthorized_response(
                unauthorized_detail_message=(
                    "Authorization header with Bearer token is required."
                ),
            )

        try:
            decoded_jwt_claims = decode_and_validate_federator_access_token(
                federator_access_token=federator_access_token,
                authentication_settings=authentication_settings,
            )
        except JwtTokenValidationError as jwt_validation_error:
            logger.info(
                "JWT validation failed for protected path '%s': %s",
                request.url.path,
                jwt_validation_error,
            )
            return self._build_unauthorized_response(
                unauthorized_detail_message=str(jwt_validation_error),
            )

        authenticated_github_username = decoded_jwt_claims[JWT_GITHUB_USERNAME_CLAIM]
        normalized_github_username = authenticated_github_username.lower()

        if normalized_github_username not in allowed_github_username_set:
            logger.warning(
                "Allowlist rejection for GitHub username '%s' on path '%s'.",
                authenticated_github_username,
                request.url.path,
            )
            return self._build_unauthorized_response(
                unauthorized_detail_message=(
                    "GitHub username is not authorized to access this resource."
                ),
            )

        setattr(
            request.state,
            self.REQUEST_STATE_GITHUB_USERNAME_KEY,
            authenticated_github_username,
        )

        return await call_next(request)

    def _build_unauthorized_response(
        self,
        unauthorized_detail_message: str,
    ) -> Response:
        """Return a strict 401 Unauthorized JSON response."""
        response_body = json.dumps({"detail": unauthorized_detail_message})
        return Response(
            content=response_body,
            status_code=401,
            media_type="application/json",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def _is_exempt_request_path(self, request_path: str) -> bool:
        """Return True when the request path bypasses authentication screening."""
        return any(
            request_path.startswith(exempt_prefix)
            for exempt_prefix in self.EXEMPT_PATH_PREFIXES
        )

    def _is_protected_request_path(self, request_path: str) -> bool:
        """Return True when the request path requires JWT allowlist validation."""
        return any(
            request_path.startswith(protected_prefix)
            for protected_prefix in self.PROTECTED_PATH_PREFIXES
        )
