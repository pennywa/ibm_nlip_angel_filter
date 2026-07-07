"""NLIP Angel Filter federator API server."""

import logging

from fastapi import FastAPI, HTTPException, Request, status

from auth.allowlist_validation_middleware import AllowlistValidationMiddleware
from auth.github_oauth_routes import github_oauth_router
from nlip_angel_filter.federator.comparison_shopping_pipeline import (
    ComparisonShoppingPipelineOrchestrator,
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

angel_filter_fastapi_application.add_middleware(AllowlistValidationMiddleware)
angel_filter_fastapi_application.include_router(github_oauth_router)

_comparison_shopping_pipeline_orchestrator = ComparisonShoppingPipelineOrchestrator()


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
