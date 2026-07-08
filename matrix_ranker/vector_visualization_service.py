"""
Interactive 3D vector-space visualization for Angel Filter distance ranking.

Maps ranked candidate telemetry into a Cartesian utility space and renders
proximity vectors from the user's ideal target anchor to each candidate.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import plotly.graph_objects as go
except ImportError as import_error:  # pragma: no cover - optional dependency
    raise ImportError(
        "plotly is required for 3D vector visualization. "
        "Install it with: pip install plotly"
    ) from import_error

_PROVIDER_COLOR_MAP: dict[str, str] = {
    "openai": "#10A37F",
    "ollama": "#FF6B35",
    "gemini": "#8E44AD",
    "watson": "#0F62FE",
}

_DEFAULT_PROVIDER_COLOR: str = "#6C757D"
_TARGET_ANCHOR_COLOR: str = "#FFD700"
_TARGET_ANCHOR_HALO_COLOR: str = "rgba(255, 215, 0, 0.35)"
_PROXIMITY_LINE_COLOR: str = "rgba(148, 163, 184, 0.55)"


def _resolve_primary_provider_identifier(source_provider_identifiers: list[str]) -> str:
    """Return a single provider label for color-coding and legend grouping."""
    if not source_provider_identifiers:
        return "unknown"
    if len(source_provider_identifiers) == 1:
        return source_provider_identifiers[0]
    return "+".join(sorted(source_provider_identifiers))


def _resolve_provider_color(provider_identifier: str) -> str:
    """Map a provider identifier to a distinct visualization color."""
    if "+" in provider_identifier:
        return _DEFAULT_PROVIDER_COLOR
    return _PROVIDER_COLOR_MAP.get(provider_identifier, _DEFAULT_PROVIDER_COLOR)


def _derive_ideal_target_vector(
    ranked_candidate_records: list[dict[str, Any]],
    telemetry_payload: dict[str, Any],
) -> dict[str, float]:
    """
    Resolve the user's optimal target anchor in plotted utility space.

    Prefers an explicit ``ideal_target_vector`` entry in the telemetry payload;
    otherwise derives min cost, min distance, and max quality from candidates.
    """
    explicit_ideal_target_vector = telemetry_payload.get("ideal_target_vector")
    if isinstance(explicit_ideal_target_vector, dict):
        return {
            "normalized_cost_score": float(
                explicit_ideal_target_vector["normalized_cost_score"]
            ),
            "normalized_distance_score": float(
                explicit_ideal_target_vector["normalized_distance_score"]
            ),
            "bayesian_quality_rating": float(
                explicit_ideal_target_vector["bayesian_quality_rating"]
            ),
        }

    normalized_cost_scores = [
        float(record["normalized_cost_score"]) for record in ranked_candidate_records
    ]
    normalized_distance_scores = [
        float(record["normalized_distance_score"]) for record in ranked_candidate_records
    ]
    bayesian_quality_ratings = [
        float(record["bayesian_quality_rating"]) for record in ranked_candidate_records
    ]

    return {
        "normalized_cost_score": min(normalized_cost_scores),
        "normalized_distance_score": min(normalized_distance_scores),
        "bayesian_quality_rating": max(bayesian_quality_ratings),
    }


def _build_candidate_hover_text(candidate_record: dict[str, Any]) -> str:
    """Build rich multi-line hover text for a candidate marker."""
    primary_provider = _resolve_primary_provider_identifier(
        candidate_record.get("source_provider_identifiers", []),
    )
    provider_list_text = ", ".join(
        candidate_record.get("source_provider_identifiers", []) or ["unknown"],
    )

    return (
        f"<b>{candidate_record['candidate_display_name']}</b><br>"
        f"Provider: {primary_provider}<br>"
        f"All Sources: {provider_list_text}<br>"
        f"Final Rank: #{candidate_record['final_rank_position']}<br>"
        f"Weighted Vector Distance: "
        f"{float(candidate_record['weighted_vector_distance_score']):.4f}<br>"
        f"Normalized Cost: {float(candidate_record['normalized_cost_score']):.4f}<br>"
        f"Normalized Distance: "
        f"{float(candidate_record['normalized_distance_score']):.4f}<br>"
        f"Bayesian Quality: {float(candidate_record['bayesian_quality_rating']):.4f}"
    )


def _euclidean_distance(
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
) -> float:
    """Compute 3D Euclidean distance between two plotted coordinate tuples."""
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2
        + (point_a[1] - point_b[1]) ** 2
        + (point_a[2] - point_b[2]) ** 2,
    )


def generate_3d_distance_visualization(telemetry_payload: dict) -> str:
    """
    Build an interactive 3D Plotly visualization from lifecycle telemetry.

    Axes:
        X -> normalized_cost_score
        Y -> normalized_distance_score
        Z -> bayesian_quality_rating

    The glowing target anchor marks the user's optimal utility vector. Dotted
    proximity lines stretch from that anchor to each candidate; hover text and
  annotations surface the backend ``weighted_vector_distance_score``.

    Args:
        telemetry_payload: Standard comparison-shopping lifecycle telemetry,
            including ``ranked_candidate_records`` with normalized utility metrics.

    Returns:
        Standalone interactive HTML fragment suitable for embedding or saving.
    """
    ranked_candidate_records: list[dict[str, Any]] = telemetry_payload.get(
        "ranked_candidate_records",
        [],
    )
    if not ranked_candidate_records:
        raise ValueError(
            "telemetry_payload must include a non-empty 'ranked_candidate_records' list.",
        )

    ideal_target_vector = _derive_ideal_target_vector(
        ranked_candidate_records=ranked_candidate_records,
        telemetry_payload=telemetry_payload,
    )
    anchor_position = (
        ideal_target_vector["normalized_cost_score"],
        ideal_target_vector["normalized_distance_score"],
        ideal_target_vector["bayesian_quality_rating"],
    )

    figure = go.Figure()

    for candidate_record in ranked_candidate_records:
        candidate_position = (
            float(candidate_record["normalized_cost_score"]),
            float(candidate_record["normalized_distance_score"]),
            float(candidate_record["bayesian_quality_rating"]),
        )
        weighted_vector_distance_score = float(
            candidate_record["weighted_vector_distance_score"],
        )
        geometric_line_distance = _euclidean_distance(anchor_position, candidate_position)
        primary_provider = _resolve_primary_provider_identifier(
            candidate_record.get("source_provider_identifiers", []),
        )
        provider_color = _resolve_provider_color(primary_provider)

        figure.add_trace(
            go.Scatter3d(
                x=[anchor_position[0], candidate_position[0]],
                y=[anchor_position[1], candidate_position[1]],
                z=[anchor_position[2], candidate_position[2]],
                mode="lines",
                line=dict(
                    color=_PROXIMITY_LINE_COLOR,
                    width=3,
                    dash="dot",
                ),
                hoverinfo="text",
                hovertext=(
                    f"<b>Proximity Vector</b><br>"
                    f"Candidate: {candidate_record['candidate_display_name']}<br>"
                    f"Weighted Vector Distance: {weighted_vector_distance_score:.4f}<br>"
                    f"3D Geometric Span: {geometric_line_distance:.4f}"
                ),
                showlegend=False,
            ),
        )

    candidates_by_provider: dict[str, list[dict[str, Any]]] = {}
    for candidate_record in ranked_candidate_records:
        primary_provider = _resolve_primary_provider_identifier(
            candidate_record.get("source_provider_identifiers", []),
        )
        candidates_by_provider.setdefault(primary_provider, []).append(candidate_record)

    for provider_identifier in sorted(candidates_by_provider.keys()):
        provider_candidate_records = candidates_by_provider[provider_identifier]
        provider_color = _resolve_provider_color(provider_identifier)

        figure.add_trace(
            go.Scatter3d(
                x=[
                    float(record["normalized_cost_score"])
                    for record in provider_candidate_records
                ],
                y=[
                    float(record["normalized_distance_score"])
                    for record in provider_candidate_records
                ],
                z=[
                    float(record["bayesian_quality_rating"])
                    for record in provider_candidate_records
                ],
                mode="markers",
                name=f"{provider_identifier} candidates",
                marker=dict(
                    size=9,
                    color=provider_color,
                    symbol="circle",
                    line=dict(width=1.5, color="#FFFFFF"),
                    opacity=0.95,
                ),
                text=[
                    record["candidate_display_name"]
                    for record in provider_candidate_records
                ],
                hovertext=[
                    _build_candidate_hover_text(record)
                    for record in provider_candidate_records
                ],
                hoverinfo="text",
            ),
        )

    figure.add_trace(
        go.Scatter3d(
            x=[anchor_position[0]],
            y=[anchor_position[1]],
            z=[anchor_position[2]],
            mode="markers",
            name="Ideal Target Anchor",
            marker=dict(
                size=22,
                color=_TARGET_ANCHOR_HALO_COLOR,
                symbol="diamond",
                line=dict(width=0),
            ),
            hoverinfo="text",
            hovertext=(
                "<b>Ideal Target Anchor</b><br>"
                f"Normalized Cost: {anchor_position[0]:.4f}<br>"
                f"Normalized Distance: {anchor_position[1]:.4f}<br>"
                f"Bayesian Quality: {anchor_position[2]:.4f}<br>"
                "User-optimal utility vector derived from search weights."
            ),
        ),
    )
    figure.add_trace(
        go.Scatter3d(
            x=[anchor_position[0]],
            y=[anchor_position[1]],
            z=[anchor_position[2]],
            mode="markers",
            name="Ideal Target Anchor (core)",
            marker=dict(
                size=14,
                color=_TARGET_ANCHOR_COLOR,
                symbol="diamond",
                line=dict(width=2, color="#FFFFFF"),
            ),
            hoverinfo="skip",
            showlegend=False,
        ),
    )

    upstream_providers = telemetry_payload.get("upstream_provider_identifiers", [])
    excluded_providers = telemetry_payload.get("excluded_provider_identifiers", [])
    skipped_providers = telemetry_payload.get("skipped_provider_identifiers", [])
    ranked_count = len(ranked_candidate_records)

    subtitle_parts = [
        f"{ranked_count} ranked candidates",
        f"active providers: {', '.join(upstream_providers) or 'n/a'}",
    ]
    if excluded_providers:
        subtitle_parts.append(f"excluded: {', '.join(excluded_providers)}")
    if skipped_providers:
        subtitle_parts.append(f"skipped: {', '.join(skipped_providers)}")

    figure.update_layout(
        title=dict(
            text=(
                "<b>Angel Filter Vector Space</b><br>"
                f"<sup>{' | '.join(subtitle_parts)}</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        font=dict(color="#E2E8F0", family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(
            title=dict(text="Candidates by Source"),
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="rgba(148, 163, 184, 0.35)",
            borderwidth=1,
            x=0.02,
            y=0.98,
            traceorder="normal",
        ),
        scene=dict(
            xaxis=dict(
                title="Normalized Cost Score",
                backgroundcolor="#111827",
                gridcolor="rgba(148, 163, 184, 0.25)",
                zerolinecolor="rgba(148, 163, 184, 0.35)",
            ),
            yaxis=dict(
                title="Normalized Distance Score",
                backgroundcolor="#111827",
                gridcolor="rgba(148, 163, 184, 0.25)",
                zerolinecolor="rgba(148, 163, 184, 0.35)",
            ),
            zaxis=dict(
                title="Bayesian Quality Rating",
                backgroundcolor="#111827",
                gridcolor="rgba(148, 163, 184, 0.25)",
                zerolinecolor="rgba(148, 163, 184, 0.35)",
            ),
            bgcolor="#0B1120",
            aspectmode="data",
        ),
    )

    return figure.to_html(full_html=False, include_plotlyjs="cdn")
