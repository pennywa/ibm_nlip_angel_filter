"""
Root orchestration entrypoint for the NLIP Angel Filter federator.

Re-exports the ASGI application at the Docker-configured import path:
    nlip_angel_filter.federator.api_server:angel_filter_fastapi_application
"""

from nlip_angel_filter.federator.api_server import angel_filter_fastapi_application

__all__ = ["angel_filter_fastapi_application"]
