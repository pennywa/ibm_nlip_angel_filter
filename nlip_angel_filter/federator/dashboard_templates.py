"""HTML templates for the Angel Filter federator dashboard shell."""

from __future__ import annotations

import html
from typing import Any

from nlip_angel_filter.federator.dashboard_service import DashboardRecommendationSummary

DEFAULT_DASHBOARD_USER_SEARCH_QUERY: str = "coffee places"

_TAILWIND_CDN_SCRIPT: str = (
    '<script src="https://cdn.tailwindcss.com"></script>'
    "<script>"
    "tailwind.config = {"
    '  theme: { extend: { colors: {'
    '    ink: "#0f172a",'
    '    mist: "#f8fafc",'
    '    glow: "#fbbf24",'
    '    bloom: "#38bdf8"'
    "  }}}"
    "};"
    "</script>"
)


def _render_dashboard_header(authenticated_github_username: str) -> str:
    escaped_github_username = html.escape(authenticated_github_username)
    return f"""
    <header class="border-b border-slate-800/80 bg-slate-950/90 backdrop-blur sticky top-0 z-20">
      <div class="mx-auto flex max-w-5xl items-center justify-between gap-4 px-4 py-4 sm:px-6">
        <div class="flex items-center gap-3">
          <div class="flex h-10 w-10 items-center justify-center rounded-2xl bg-amber-400/15 text-xl">✨</div>
          <div>
            <p class="text-sm font-semibold text-white">Angel Filter</p>
            <p class="text-xs text-slate-400">Fair comparison shopping</p>
          </div>
          <span class="hidden sm:inline-flex items-center gap-2 rounded-full bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-emerald-300">
            🛡️ Angel Filter Active
          </span>
        </div>
        <p class="text-sm text-slate-400">Signed in as <span class="font-semibold text-white">@{escaped_github_username}</span></p>
      </div>
    </header>
    """


def render_dashboard_access_denied_page(denial_reason: str | None = None) -> str:
    """Render a stylized 403 page for unauthenticated or non-allowlisted users."""
    resolved_denial_reason = denial_reason or (
        "Your GitHub identity is not authorized to access the Angel Filter dashboard."
    )
    escaped_denial_reason = html.escape(resolved_denial_reason)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>403 Access Denied | Angel Filter</title>
  {_TAILWIND_CDN_SCRIPT}
</head>
<body class="min-h-screen bg-gradient-to-b from-slate-900 via-slate-950 to-black text-slate-100">
  <main class="mx-auto flex min-h-screen max-w-lg items-center px-4 py-10">
    <section class="w-full rounded-3xl border border-rose-500/30 bg-slate-900/80 p-8 shadow-2xl shadow-black/40">
      <p class="text-xs font-bold uppercase tracking-[0.2em] text-rose-300">🛡️ Security Gate</p>
      <h1 class="mt-3 text-3xl font-bold text-white">403 Access Denied</h1>
      <p class="mt-3 text-slate-300">Unauthorized Identity</p>
      <p class="mt-4 text-sm leading-relaxed text-slate-400">
        This dashboard is only available to approved GitHub accounts with a valid session cookie.
      </p>
      <div class="mt-5 rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
        {escaped_denial_reason}
      </div>
      <a href="/auth/github/login"
         class="mt-6 inline-flex items-center justify-center rounded-2xl bg-amber-400 px-5 py-3 text-sm font-bold text-slate-950 transition hover:bg-amber-300">
        Sign in with GitHub
      </a>
    </section>
  </main>
</body>
</html>"""


def _render_recommendation_results_html(
    ranked_recommendation_summaries: list[DashboardRecommendationSummary],
) -> str:
    """Render the friendly ordered recommendation list."""
    if not ranked_recommendation_summaries:
        return ""

    recommendation_cards: list[str] = []
    for recommendation_summary in ranked_recommendation_summaries:
        escaped_name = html.escape(recommendation_summary.candidate_display_name)
        escaped_provider = html.escape(recommendation_summary.provider_display_label)
        rank_badge_class = (
            "bg-amber-400 text-slate-950"
            if recommendation_summary.final_rank_position == 1
            else "bg-slate-700 text-slate-100"
        )

        recommendation_cards.append(
            f"""
            <li class="flex items-start gap-4 rounded-2xl border border-slate-800 bg-slate-900/70 p-4 shadow-sm">
              <div class="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl text-sm font-bold {rank_badge_class}">
                #{recommendation_summary.final_rank_position}
              </div>
              <div class="min-w-0 flex-1">
                <h3 class="text-base font-semibold text-white">{escaped_name}</h3>
                <p class="mt-1 text-sm text-slate-400">Suggested by {escaped_provider}</p>
              </div>
              <div class="text-right">
                <p class="text-xs uppercase tracking-wide text-slate-500">Match</p>
                <p class="text-lg font-bold text-sky-300">{recommendation_summary.friendly_match_score}%</p>
              </div>
            </li>
            """
        )

    return f"""
    <section class="mt-8">
      <div class="mb-4">
        <h2 class="text-xl font-semibold text-white">Your Top Picks</h2>
        <p class="mt-1 text-sm text-slate-400">Ranked for you — clearest matches appear first.</p>
      </div>
      <ol class="space-y-3">
        {''.join(recommendation_cards)}
      </ol>
    </section>
    """


def _render_visualization_arena_html(plotly_html_fragment: str | None) -> str:
    """Render the inline 3D Plotly arena beneath the recommendation list."""
    if not plotly_html_fragment:
        return ""

    return f"""
    <section class="mt-8">
      <div class="mb-4">
        <h2 class="text-xl font-semibold text-white">Visual Match Arena</h2>
        <p class="mt-1 text-sm text-slate-400">
          The glowing gold star is your ideal pick profile. Shorter lines mean a closer, fairer match.
        </p>
      </div>
      <div class="overflow-hidden rounded-3xl border border-slate-800 bg-slate-950 shadow-inner">
        <div class="min-h-[520px] w-full">{plotly_html_fragment}</div>
      </div>
    </section>
    """


def render_angel_filter_dashboard_page(
    authenticated_github_username: str,
    active_user_search_query: str,
    low_cost_selected: bool,
    proximity_selected: bool,
    quality_selected: bool,
    ranked_recommendation_summaries: list[DashboardRecommendationSummary] | None = None,
    plotly_html_fragment: str | None = None,
    pipeline_error_message: str | None = None,
    show_results: bool = False,
) -> str:
    """Render the polished, non-technical Angel Filter dashboard experience."""
    escaped_query_value = html.escape(active_user_search_query)
    escaped_pipeline_error = html.escape(pipeline_error_message or "")

    low_cost_checked = "checked" if low_cost_selected else ""
    proximity_checked = "checked" if proximity_selected else ""
    quality_checked = "checked" if quality_selected else ""

    error_banner_html = ""
    if pipeline_error_message:
        error_banner_html = f"""
        <div class="mt-6 rounded-2xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
          {escaped_pipeline_error}
        </div>
        """

    results_html = ""
    if show_results and not pipeline_error_message:
        results_html = (
            _render_recommendation_results_html(
                ranked_recommendation_summaries or [],
            )
            + _render_visualization_arena_html(plotly_html_fragment)
        )

    welcome_html = ""
    if not show_results and not pipeline_error_message:
        welcome_html = """
        <div class="mt-6 rounded-2xl border border-slate-800 bg-slate-900/50 px-4 py-4 text-sm text-slate-400">
          Tell us what you are shopping for, choose what matters most, then tap
          <span class="font-semibold text-white">Run Angel Filter</span> to see fair picks and a visual proof map.
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Angel Filter Dashboard</title>
  {_TAILWIND_CDN_SCRIPT}
</head>
<body class="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-100">
  {_render_dashboard_header(authenticated_github_username)}

  <main class="mx-auto max-w-5xl px-4 py-8 sm:px-6">
    <section class="text-center">
      <p class="text-sm font-medium uppercase tracking-[0.25em] text-amber-300/90">Comparison Shopping, Un-steered</p>
      <h1 class="mt-3 text-3xl font-bold text-white sm:text-4xl">Find the fairest options for you</h1>
      <p class="mx-auto mt-3 max-w-2xl text-sm leading-relaxed text-slate-400 sm:text-base">
        Angel Filter gathers suggestions from multiple assistants, removes hidden bias, and ranks what truly fits your priorities.
      </p>
    </section>

    <section class="mx-auto mt-8 max-w-3xl rounded-3xl border border-slate-800 bg-slate-900/70 p-6 shadow-2xl shadow-black/30 sm:p-8">
      <form id="angel-filter-form" method="post" action="/dashboard" class="space-y-6">
        <div>
          <label for="user_search_query" class="block text-sm font-semibold text-white">
            What are you comparison shopping for today?
          </label>
          <input
            id="user_search_query"
            name="user_search_query"
            type="text"
            value="{escaped_query_value}"
            placeholder="{DEFAULT_DASHBOARD_USER_SEARCH_QUERY}"
            class="mt-2 w-full rounded-2xl border border-slate-700 bg-slate-950 px-4 py-3 text-base text-white placeholder:text-slate-500 focus:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-400/30"
          />
        </div>

        <fieldset>
          <legend class="text-sm font-semibold text-white">What matters most to you? (Select all that apply)</legend>
          <p class="mt-1 text-xs text-slate-500">Leave all unchecked if everything matters equally.</p>
          <div class="mt-4 grid gap-3 sm:grid-cols-3">
            <label class="flex cursor-pointer items-start gap-3 rounded-2xl border border-slate-800 bg-slate-950/70 px-4 py-3 transition hover:border-sky-500/40">
              <input type="checkbox" name="pref_low_cost" value="on" class="mt-1 h-4 w-4 rounded border-slate-600 bg-slate-900 text-sky-400 focus:ring-sky-400" {low_cost_checked} />
              <span>
                <span class="block text-sm font-medium text-white">Low Cost</span>
                <span class="block text-xs text-slate-500">Best price wins</span>
              </span>
            </label>
            <label class="flex cursor-pointer items-start gap-3 rounded-2xl border border-slate-800 bg-slate-950/70 px-4 py-3 transition hover:border-sky-500/40">
              <input type="checkbox" name="pref_proximity" value="on" class="mt-1 h-4 w-4 rounded border-slate-600 bg-slate-900 text-sky-400 focus:ring-sky-400" {proximity_checked} />
              <span>
                <span class="block text-sm font-medium text-white">Convenient Location</span>
                <span class="block text-xs text-slate-500">Closer is better</span>
              </span>
            </label>
            <label class="flex cursor-pointer items-start gap-3 rounded-2xl border border-slate-800 bg-slate-950/70 px-4 py-3 transition hover:border-sky-500/40">
              <input type="checkbox" name="pref_quality" value="on" class="mt-1 h-4 w-4 rounded border-slate-600 bg-slate-900 text-sky-400 focus:ring-sky-400" {quality_checked} />
              <span>
                <span class="block text-sm font-medium text-white">Highest Community Ratings</span>
                <span class="block text-xs text-slate-500">Trusted favorites</span>
              </span>
            </label>
          </div>
        </fieldset>

        <button
          id="run-button"
          type="submit"
          class="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-amber-300 to-amber-400 px-5 py-3 text-base font-bold text-slate-950 shadow-lg shadow-amber-500/20 transition hover:from-amber-200 hover:to-amber-300 disabled:cursor-not-allowed disabled:opacity-70 sm:w-auto"
        >
          Run Angel Filter
        </button>
      </form>

      {welcome_html}
      {error_banner_html}
    </section>

    <div id="loading-state" class="hidden mx-auto mt-8 max-w-3xl rounded-3xl border border-sky-500/30 bg-sky-500/10 px-6 py-8 text-center">
      <div class="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-4 border-sky-300/30 border-t-sky-300"></div>
      <p class="text-lg font-semibold text-white">Working on your fair ranking…</p>
      <p class="mt-2 text-sm text-sky-100/90">Stalking hidden biases… routing to fallback arrays…</p>
    </div>

    <div id="results-panel">
      {results_html}
    </div>
  </main>

  <script>
    (function () {{
      const angelFilterForm = document.getElementById("angel-filter-form");
      const loadingState = document.getElementById("loading-state");
      const runButton = document.getElementById("run-button");
      const resultsPanel = document.getElementById("results-panel");

      if (!angelFilterForm || !loadingState || !runButton) {{
        return;
      }}

      angelFilterForm.addEventListener("submit", function () {{
        loadingState.classList.remove("hidden");
        runButton.disabled = true;
        if (resultsPanel) {{
          resultsPanel.classList.add("opacity-40");
        }}
        window.scrollTo({{ top: loadingState.offsetTop - 24, behavior: "smooth" }});
      }});
    }})();
  </script>
</body>
</html>"""


def render_dashboard_pipeline_error_page(error_message: str) -> str:
    """Render a compact pipeline error panel for legacy visualization routes."""
    escaped_error_message = html.escape(error_message)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pipeline Error</title>
  {_TAILWIND_CDN_SCRIPT}
</head>
<body class="min-h-screen bg-slate-950 p-6 text-slate-100">
  <div class="mx-auto max-w-xl rounded-2xl border border-rose-500/40 bg-rose-500/10 p-5 text-sm text-rose-100">
    <h1 class="text-base font-semibold text-white">We could not finish this search</h1>
    <p class="mt-2">{escaped_error_message}</p>
  </div>
</body>
</html>"""
