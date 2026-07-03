"""ASGI middleware skeleton for GitHub username allowlist validation."""

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from auth.authentication_settings import get_authentication_settings


class AllowlistValidationMiddleware(BaseHTTPMiddleware):
    """
    Intercept protected requests and ensure the authenticated GitHub identity
    appears in ``ALLOWED_GITHUB_USERNAMES`` before the request reaches handlers.

    Validation logic is deferred; this middleware defines the structural contract.
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

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """
        Run allowlist screening for protected routes.

        Implementation deferred: extract JWT claims, resolve GitHub username,
        and reject requests whose identity is not on the allowlist.
        """
        if self._is_exempt_request_path(request.url.path):
            return await call_next(request)

        if self._is_protected_request_path(request.url.path):
            authentication_settings = get_authentication_settings()
            _allowed_github_username_set = (
                authentication_settings.parse_allowed_github_username_set()
            )
            # Placeholder: authenticated_github_username will be extracted from JWT.
            _authenticated_github_username: str | None = None
            if _authenticated_github_username is None:
                return Response(
                    content='{"detail":"Authentication required."}',
                    status_code=401,
                    media_type="application/json",
                )
            if (
                _authenticated_github_username.lower()
                not in _allowed_github_username_set
            ):
                return Response(
                    content='{"detail":"GitHub username not on allowlist."}',
                    status_code=403,
                    media_type="application/json",
                )

        return await call_next(request)

    def _is_exempt_request_path(self, request_path: str) -> bool:
        """Return True when the request path bypasses allowlist screening."""
        return any(
            request_path.startswith(exempt_prefix)
            for exempt_prefix in self.EXEMPT_PATH_PREFIXES
        )

    def _is_protected_request_path(self, request_path: str) -> bool:
        """Return True when the request path requires allowlist validation."""
        return any(
            request_path.startswith(protected_prefix)
            for protected_prefix in self.PROTECTED_PATH_PREFIXES
        )
