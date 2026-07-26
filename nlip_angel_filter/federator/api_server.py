"""NLIP Angel Filter federator API server."""

import logging

from fastapi import FastAPI, Form, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from auth.allowlist_validation_middleware import AllowlistValidationMiddleware
from auth.dashboard_session_guard import (
    DashboardAccessDenied,
    DashboardSessionContext,
    resolve_dashboard_session_from_request,
)
from auth.github_oauth_routes import github_oauth_router
from nlip_angel_filter.federator.comparison_shopping_pipeline import (
    ComparisonShoppingPipelineOrchestrator,
)
from nlip_angel_filter.federator.dashboard_service import (
    DEFAULT_DASHBOARD_USER_SEARCH_QUERY,
    build_dashboard_visualization_context,
    resolve_dashboard_query_profile,
    resolve_dashboard_query_profile_from_form,
)
from nlip_angel_filter.federator.dashboard_templates import (
    render_angel_filter_dashboard_page,
    render_dashboard_access_denied_page,
    render_dashboard_pipeline_error_page,
)
from protocol.nlip_message_envelope_schemas import (
    NLIPComparisonShoppingQueryPayload,
    NLIPMessageEnvelope,
    NLIPMessageKind,
)

logger = logging.getLogger(__name__)

angel_filter_fastapi_application = FastAPI(
    title="NLIP Angel Filter Federator",
    description=(
        "ECMA-430 NLIP Federator comparison-shopping engine that intercepts "
        "steered AI recommendations and re-ranks them by user utility."
    ),
    version="0.1.0",
)

@angel_filter_fastapi_application.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root hits directly to the dashboard."""
    return RedirectResponse(url="/dashboard")

angel_filter_fastapi_application.add_middleware(AllowlistValidationMiddleware)
angel_filter_fastapi_application.include_router(github_oauth_router)

_comparison_shopping_pipeline_orchestrator = ComparisonShoppingPipelineOrchestrator()


def _build_dashboard_access_denied_response(
    dashboard_access_denied: DashboardAccessDenied,
) -> HTMLResponse:
    """Return a stylized HTML 403 response for failed dashboard session checks."""
    return HTMLResponse(
        content=render_dashboard_access_denied_page(
            denial_reason=dashboard_access_denied.denial_reason,
        ),
        status_code=status.HTTP_403_FORBIDDEN,
    )


def _require_dashboard_session(request: Request) -> DashboardSessionContext | HTMLResponse:
    """Validate cookie-based JWT credentials and allowlist membership for dashboard routes."""
    dashboard_session_resolution = resolve_dashboard_session_from_request(request=request)
    if isinstance(dashboard_session_resolution, DashboardAccessDenied):
        return _build_dashboard_access_denied_response(dashboard_session_resolution)
    return dashboard_session_resolution


def _parse_checkbox_form_value(checkbox_form_value: str | None) -> bool:
    """Return True when an HTML checkbox field was submitted as selected."""
    return checkbox_form_value is not None


def _render_dashboard_page_response(
    *,
    authenticated_github_username: str,
    dashboard_query_profile,
    ranked_recommendation_summaries=None,
    plotly_html_fragment: str | None = None,
    pipeline_error_message: str | None = None,
    show_results: bool = False,
) -> HTMLResponse:
    """Build the unified Angel Filter dashboard HTML response."""
    return HTMLResponse(
        content=render_angel_filter_dashboard_page(
            authenticated_github_username=authenticated_github_username,
            active_user_search_query=dashboard_query_profile.user_search_query,
            low_cost_selected=dashboard_query_profile.low_cost_selected,
            proximity_selected=dashboard_query_profile.proximity_selected,
            quality_selected=dashboard_query_profile.quality_selected,
            ranked_recommendation_summaries=ranked_recommendation_summaries,
            plotly_html_fragment=plotly_html_fragment,
            pipeline_error_message=pipeline_error_message,
            show_results=show_results,
        ),
        status_code=status.HTTP_200_OK,
    )


@angel_filter_fastapi_application.get(
    "/dashboard",
    tags=["Dashboard"],
    summary="Angel Filter comparison-shopping dashboard",
    response_class=HTMLResponse,
)
async def render_angel_filter_dashboard(
    request: Request,
    user_search_query: str | None = Query(
        default=None,
        alias="q",
        description="Optional pre-filled search query (defaults to 'coffee places').",
    ),
) -> HTMLResponse:
    """
    Render the friendly Angel Filter dashboard shell.

    Protected by cookie-based JWT session verification and GitHub allowlist screening.
    """
    dashboard_session_resolution = _require_dashboard_session(request=request)
    if isinstance(dashboard_session_resolution, HTMLResponse):
        return dashboard_session_resolution

    dashboard_query_profile = resolve_dashboard_query_profile_from_form(
        requested_user_search_query=user_search_query,
        low_cost_selected=False,
        proximity_selected=False,
        quality_selected=False,
    )

    return _render_dashboard_page_response(
        authenticated_github_username=(
            dashboard_session_resolution.authenticated_github_username
        ),
        dashboard_query_profile=dashboard_query_profile,
        show_results=False,
    )


@angel_filter_fastapi_application.post(
    "/dashboard",
    tags=["Dashboard"],
    summary="Run the Angel Filter evaluation pipeline from the dashboard",
    response_class=HTMLResponse,
)
async def run_angel_filter_dashboard(
    request: Request,
    user_search_query: str = Form(default=""),
    pref_low_cost: str | None = Form(default=None),
    pref_proximity: str | None = Form(default=None),
    pref_quality: str | None = Form(default=None),
) -> HTMLResponse:
    """
    Execute comparison shopping with friendly preference checkboxes and render results.

    Output order:
        1. Ranked recommendation list
        2. Inline 3D vector visualization arena
    """
    dashboard_session_resolution = _require_dashboard_session(request=request)
    if isinstance(dashboard_session_resolution, HTMLResponse):
        return dashboard_session_resolution

    dashboard_query_profile = resolve_dashboard_query_profile_from_form(
        requested_user_search_query=user_search_query or DEFAULT_DASHBOARD_USER_SEARCH_QUERY,
        low_cost_selected=_parse_checkbox_form_value(pref_low_cost),
        proximity_selected=_parse_checkbox_form_value(pref_proximity),
        quality_selected=_parse_checkbox_form_value(pref_quality),
    )

    try:
        dashboard_visualization_context = await build_dashboard_visualization_context(
            dashboard_query_profile=dashboard_query_profile,
        )
    except RuntimeError as dashboard_pipeline_runtime_error:
        logger.error(
            "Dashboard pipeline failed for user '%s': %s",
            dashboard_session_resolution.authenticated_github_username,
            dashboard_pipeline_runtime_error,
        )
        return _render_dashboard_page_response(
            authenticated_github_username=(
                dashboard_session_resolution.authenticated_github_username
            ),
            dashboard_query_profile=dashboard_query_profile,
            pipeline_error_message=str(dashboard_pipeline_runtime_error),
            show_results=False,
        )
    except ImportError as missing_plotly_error:
        return _render_dashboard_page_response(
            authenticated_github_username=(
                dashboard_session_resolution.authenticated_github_username
            ),
            dashboard_query_profile=dashboard_query_profile,
            pipeline_error_message=str(missing_plotly_error),
            show_results=False,
        )

    return _render_dashboard_page_response(
        authenticated_github_username=(
            dashboard_session_resolution.authenticated_github_username
        ),
        dashboard_query_profile=dashboard_query_profile,
        ranked_recommendation_summaries=(
            dashboard_visualization_context.ranked_recommendation_summaries
        ),
        plotly_html_fragment=dashboard_visualization_context.plotly_html_fragment,
        show_results=True,
    )


@angel_filter_fastapi_application.get(
    "/dashboard/visualization",
    tags=["Dashboard"],
    summary="Legacy embedded 3D Plotly vector visualization fragment",
    response_class=HTMLResponse,
)
async def render_angel_filter_dashboard_visualization(
    request: Request,
    user_search_query: str | None = Query(
        default=None,
        alias="q",
        description="Comparison-shopping query; defaults permanently to 'coffee places'.",
    ),
    user_preference_weight_cost: float | None = Query(default=None, alias="w_cost"),
    user_preference_weight_distance: float | None = Query(default=None, alias="w_distance"),
    user_preference_weight_quality: float | None = Query(default=None, alias="w_quality"),
) -> HTMLResponse:
    """Render the iframe-hosted Plotly 3D utility vector workspace."""
    dashboard_session_resolution = _require_dashboard_session(request=request)
    if isinstance(dashboard_session_resolution, HTMLResponse):
        return dashboard_session_resolution

    dashboard_query_profile = resolve_dashboard_query_profile(
        requested_user_search_query=user_search_query,
        user_preference_weight_cost=user_preference_weight_cost,
        user_preference_weight_distance=user_preference_weight_distance,
        user_preference_weight_quality=user_preference_weight_quality,
    )

    try:
        dashboard_visualization_context = await build_dashboard_visualization_context(
            dashboard_query_profile=dashboard_query_profile,
        )
    except RuntimeError as dashboard_pipeline_runtime_error:
        logger.error(
            "Dashboard visualization pipeline failed: %s",
            dashboard_pipeline_runtime_error,
        )
        return HTMLResponse(
            content=render_dashboard_pipeline_error_page(
                error_message=str(dashboard_pipeline_runtime_error),
            ),
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
    except ImportError as missing_plotly_error:
        return HTMLResponse(
            content=render_dashboard_pipeline_error_page(
                error_message=str(missing_plotly_error),
            ),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return HTMLResponse(
        content=(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<script src='https://cdn.tailwindcss.com'></script></head>"
            f"<body class='bg-slate-950 p-2'>"
            f"{dashboard_visualization_context.plotly_html_fragment}"
            "</body></html>"
        ),
        status_code=status.HTTP_200_OK,
    )


@angel_filter_fastapi_application.get(
    "/health",
    tags=["Health"],
    summary="Federator health check",
)
async def federator_health_check() -> dict[str, str]:
    """Return a simple health status for load balancers and Render routing."""
    return {"federator_status": "healthy"}


@angel_filter_fastapi_application.post(
    "/federator/comparison-shopping/query",
    tags=["Comparison Shopping Federator"],
    summary="Execute the full un-steered comparison-shopping pipeline",
    response_model=NLIPMessageEnvelope,
)
async def execute_comparison_shopping_federator_query(
    inbound_nlip_message_envelope: NLIPMessageEnvelope,
    request: Request,
) -> NLIPMessageEnvelope:
    """
    Bridge the full Angel Filter pipeline:

    1. Validate inbound NLIP envelope (auth enforced by middleware)
    2. Fan out to OpenAI, Gemini, and Watsonx in parallel
    3. Normalize matrix and apply Bayesian + vector distance ranking
    4. Cross-examine with local Ollama fiduciary validator
    5. Return final un-steered NLIP response packet
    """
    if (
        inbound_nlip_message_envelope.message_header.message_kind
        != NLIPMessageKind.COMPARISON_SHOPPING_QUERY
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Inbound NLIP envelope must have message_kind "
                f"'{NLIPMessageKind.COMPARISON_SHOPPING_QUERY.value}'."
            ),
        )

    try:
        comparison_shopping_query_payload = NLIPComparisonShoppingQueryPayload.model_validate(
            inbound_nlip_message_envelope.message_payload,
        )
    except Exception as payload_validation_error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid comparison-shopping query payload: {payload_validation_error}",
        ) from payload_validation_error

    authenticated_github_username = getattr(
        request.state,
        AllowlistValidationMiddleware.REQUEST_STATE_GITHUB_USERNAME_KEY,
        inbound_nlip_message_envelope.message_header.authenticated_github_username,
    )

    try:
        outbound_nlip_message_envelope = (
            await _comparison_shopping_pipeline_orchestrator.execute_comparison_shopping_pipeline(
                comparison_shopping_query_payload=comparison_shopping_query_payload,
                authenticated_github_username=authenticated_github_username,
            )
        )
    except RuntimeError as pipeline_runtime_error:
        logger.error(
            "Comparison-shopping pipeline failed: %s",
            pipeline_runtime_error,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(pipeline_runtime_error),
        ) from pipeline_runtime_error
    except Exception as pipeline_unexpected_error:
        logger.exception(
            "Unexpected comparison-shopping pipeline failure.",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred in the comparison-shopping pipeline.",
        ) from pipeline_unexpected_error

    logger.info(
        "Comparison-shopping pipeline completed for user: %s",
        authenticated_github_username or "anonymous",
    )

    return outbound_nlip_message_envelope
