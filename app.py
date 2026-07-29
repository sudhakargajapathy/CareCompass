"""CareCompass - AI-Powered Healthcare Provider Matching System

A Streamlit application that uses multi-agent AI to find and rank healthcare providers
based on user preferences and requirements.
"""

import copy
import html
import os
import queue
import re
import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Preference weight levels: raw values are normalized to sum to 1.0 in
# render_search_form. All three default to Medium, so an untouched form
# weights location/ratings/experience equally (~0.33 each).
WEIGHT_LEVELS = {"Low": 1.0, "Medium": 1.5, "High": 2.0}
WEIGHT_LEVEL_OPTIONS = list(WEIGHT_LEVELS.keys())

# Which model powers each agent — surfaced on the live progress lines and in
# the How-it-works strip so the multi-model orchestration is visible.
AGENT_MODELS = {
    "DataGathererAgent": "Claude Haiku 4.5 + Tavily",
    "PreferenceScorerAgent": "GPT-5.6 Terra",
    "CriticValidatorAgent": "Claude Opus 4.8",
}

# Portfolio link shown in the sidebar and the How-it-works section
PORTFOLIO_GITHUB_URL = os.getenv("PORTFOLIO_GITHUB_URL", "https://github.com/sudhakargajapathy")

# Import our agent system
from agents.orchestrator import create_orchestrator
from agents.critic_validator import is_judge_concern
# The card's "Closest" chip must compare the same effective distance the
# scorer ranks on, or the badge contradicts the ordering beside it.
from agents.preference_scorer import CITY_CENTROID_MARGIN_MILES
from utils.config import get_config, check_environment
from utils.vector_store import get_vector_store
from utils.auth import get_authenticator
from utils.provenance import label_source, linkable, source_domain
from utils.audit_log import log_audit_event
from utils.rate_limit import get_rate_limiter
from utils.security_headers import inject_security_headers
from utils.security import InputValidator
from utils.theme import (
    HEARTH,
    inject_theme,
)


def render_html(markup: str) -> None:
    """Render an HTML fragment safely through st.markdown.

    Collapses the fragment to a single line first: Markdown treats indented
    lines after a blank line as code blocks, which made closing tags render
    literally (e.g. a visible "</div>") whenever an optional section like the
    NPI line was empty.
    """
    flattened = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flattened, unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "workflow_results" not in st.session_state:
        st.session_state.workflow_results = None
    if "search_executed" not in st.session_state:
        st.session_state.search_executed = False
    if "show_agent_logs" not in st.session_state:
        st.session_state.show_agent_logs = False
    if "last_search_params" not in st.session_state:
        st.session_state.last_search_params = None
    if "use_cache" not in st.session_state:
        # Default ON: reuse is the point. The toggle exists to force a cold run
        # for demos and for verifying the cache against a live fetch.
        st.session_state.use_cache = True
    if "fhir_enabled" not in st.session_state:
        st.session_state.fhir_enabled = False
    if "network_payer" not in st.session_state:
        st.session_state.network_payer = ""
    if "fast_demo" not in st.session_state:
        # Default OFF: field testing showed the basic-depth/cap-3 demo profile
        # produced visibly weaker review coverage; results quality is the
        # default, the toggle is the cost-saver.
        st.session_state.fast_demo = False


@st.cache_resource
def get_orchestrator(fhir_enabled: bool, fast_mode: bool):
    """Create (and cache) the multi-agent orchestrator for a given configuration.

    Both flags must be cache keys: agents snapshot config via get_config() at
    init time, so the env vars have to be set before create_orchestrator()
    runs and a different flag combination needs a fresh orchestrator.
    """
    os.environ["FHIR_ENABLED"] = str(fhir_enabled).lower()
    os.environ["TAVILY_SEARCH_DEPTH"] = "basic" if fast_mode else "advanced"
    # Enrichment runs in parallel, so 3 candidates cost the same wall time as
    # 1. Since round 10 this knob is the RESEARCH BUDGET, not just an enrichment
    # cap: it also bounds the judge and the critic, so lowering it for fast-demo
    # cuts the two stages that dominate a run's cost (~74%) rather than the one
    # that is ~9% of it.
    # (The tiered pool this once described — top-8 second opinions, zero-pair
    # providers below the boundary — was deleted in Phase 2.)
    os.environ["MAX_PROVIDERS_TO_ENRICH"] = "3" if fast_mode else "10"
    return create_orchestrator()


def execute_with_live_progress(orchestrator, search_params: Dict[str, Any], status, progress_bar) -> Dict[str, Any]:
    """Run the workflow while rendering live agent progress.

    LangGraph may invoke the progress callback from its own executor threads,
    where Streamlit calls fail silently (no ScriptRunContext). The callback
    therefore only enqueues updates; this main script thread drains the queue
    and owns every st.* call.
    """
    updates: "queue.Queue[Dict[str, Any]]" = queue.Queue()

    def render_update(update: Dict[str, Any]) -> None:
        agent = update.get("agent_name", "")
        action = update.get("action", "")
        model = AGENT_MODELS.get(agent)
        label = f"**{agent}** · {model}" if model else f"**{agent}**"
        status.write(f"{label} — {action}")
        progress_bar.progress(min(max(update.get("progress_percentage", 0), 0), 100))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            orchestrator.execute_workflow_streaming,
            specialty=search_params["specialty"],
            location=search_params["location"],
            preferences=search_params["preferences"],
            progress_callback=updates.put,  # thread-safe; no Streamlit calls in workers
            use_cache=search_params.get("use_cache", True),
        )

        while True:
            try:
                render_update(updates.get(timeout=0.2))
                continue
            except queue.Empty:
                pass
            if future.done():
                break

        while not updates.empty():
            render_update(updates.get_nowait())

        return future.result()


def _format_wait(seconds: int) -> str:
    """Human wait time for the demo-budget message (seconds -> min/hours)."""
    if seconds < 90:
        return f"{seconds} seconds"
    if seconds < 2 * 3600:
        return f"{max(1, round(seconds / 60))} minutes"
    return f"about {max(1, round(seconds / 3600))} hours"


def check_api_keys() -> bool:
    """Check if all required API keys are configured."""
    is_valid, missing_keys = check_environment()
    if not is_valid:
        # Map internal key names to the actual environment variable names the
        # app reads (see utils/config.py) so the message is accurate. The
        # Anthropic key uses an APP_ prefix, so a naive f"{key.upper()}_API_KEY"
        # would print a variable name the app never reads.
        env_var_names = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "APP_ANTHROPIC_API_KEY",
            "tavily": "TAVILY_API_KEY",
        }
        st.error("🚨 Missing API Keys")
        st.markdown("Please set up the following API keys in your `.env` file:")
        for key in missing_keys:
            st.markdown(f"- `{env_var_names.get(key, f'{key.upper()}_API_KEY')}`")
        st.markdown("Copy `.env.example` to `.env` and add your API keys.")
        return False
    return True


def render_header():
    """Render the application header."""
    st.set_page_config(
        page_title="CareCompass",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_security_headers()
    inject_theme()

    render_html(
        """
        <div class="cc-hero">
            <div class="cc-hero-eyebrow">AI Care Navigator</div>
            <h1 class="cc-hero-title">CareCompass</h1>
            <div class="cc-hero-sub">Multi-agent AI — Claude + GPT orchestrated in one LangGraph pipeline — finding, ranking, and validating healthcare providers for you.</div>
        </div>
        """
    )

    # How the multi-agent pipeline works (also the architecture story for
    # anyone reviewing the project)
    with st.expander("How it works"):
        render_html(
            """
            <div class="cc-pipe">
                <div class="cc-pipe-step">
                    <div class="cc-pipe-name">Data Gatherer</div>
                    <div class="cc-pipe-model">Claude Haiku 4.5 + Tavily search</div>
                </div>
                <div class="cc-pipe-arrow">&rarr;</div>
                <div class="cc-pipe-step">
                    <div class="cc-pipe-name">Preference Scorer</div>
                    <div class="cc-pipe-model">GPT-5.6 Terra &middot; rubric judge</div>
                </div>
                <div class="cc-pipe-arrow">&rarr;</div>
                <div class="cc-pipe-step">
                    <div class="cc-pipe-name">Critic Validator</div>
                    <div class="cc-pipe-model">Claude Sonnet 5 &middot; bias &amp; red flags</div>
                </div>
                <div class="cc-pipe-arrow">&rarr;</div>
                <div class="cc-pipe-step">
                    <div class="cc-pipe-name">Refined ranking</div>
                    <div class="cc-pipe-model">Deterministic &middot; no extra LLM calls</div>
                </div>
            </div>
            """
        )
        st.markdown(
            "A LangGraph orchestrator coordinates agents from two model labs: provider data is "
            "gathered from live web sources, scored 70% by your weighted preferences and 30% by "
            "a rubric-scored AI judge that reads review evidence with citations, then "
            "independently challenged for bias and overlooked red flags — the critique is "
            "folded back into the final order. Coverage is verified via a payer-directory "
            "(FHIR) prototype in the sidebar — never scored from scraped data."
        )
        st.markdown(
            f"[View the project on GitHub]({PORTFOLIO_GITHUB_URL}) &nbsp;·&nbsp; "
            "v2 — an agentic care-navigation companion (FastAPI + React) — is in active development."
        )


def render_search_form() -> Optional[Dict[str, Any]]:
    """Render the provider search form."""
    st.header("Find Healthcare Providers")

    with st.form("provider_search"):
        col1, col2 = st.columns(2)

        with col1:
            specialty = st.selectbox(
                "Medical Specialty",
                [
                    "Neurology",
                    "Cardiology",
                    "Dermatology",
                    "Orthopedics",
                    "Gastroenterology",
                    "Endocrinology",
                    "Psychiatry",
                    "Oncology",
                    "Rheumatology",
                    "Pulmonology",
                    "Family Medicine",
                    "Internal Medicine"
                ],
                index=0
            )

        with col2:
            location = st.text_input(
                "Location (City, State ZIP)",
                placeholder="e.g., Phoenix, AZ 85004",
                value="Phoenix, AZ",
                help="ZIP is optional — adding it gives more accurate distance ranking."
            )

        st.subheader("What matters most to you?")
        st.caption("Weight each factor — we normalize them behind the scenes.")

        col3, col4, col5 = st.columns(3)
        with col3:
            location_level = st.segmented_control(
                "Location", WEIGHT_LEVEL_OPTIONS, default="Medium", key="w_location"
            )
        with col4:
            rating_level = st.segmented_control(
                "Ratings", WEIGHT_LEVEL_OPTIONS, default="Medium", key="w_rating"
            )
        with col5:
            experience_level = st.segmented_control(
                "Experience", WEIGHT_LEVEL_OPTIONS, default="Medium", key="w_experience"
            )

        # All three start Medium so the ranking is balanced until the user
        # tells us what matters; segmented_control returns None on deselect,
        # so fall back to Medium too.
        location_weight = WEIGHT_LEVELS[location_level or "Medium"]
        rating_weight = WEIGHT_LEVELS[rating_level or "Medium"]
        experience_weight = WEIGHT_LEVELS[experience_level or "Medium"]

        submitted = st.form_submit_button("Find Providers", type="primary", use_container_width=True)
        st.caption(
            "A full agent run takes about 30–60 seconds and costs a few cents — "
            "you'll see each agent report its progress live."
        )

        if submitted:
            if not specialty or not location:
                st.error("Please provide both specialty and location.")
                return None

            # Validate inputs using security module
            validator = InputValidator()

            # Validate specialty
            safe_specialty = validator.sanitize_specialty(specialty)
            if not safe_specialty:
                st.error("Invalid specialty selected. Please choose from the dropdown list.")
                return None

            # Validate location
            safe_location = validator.sanitize_location(location)
            if not safe_location:
                st.error("Invalid location format. Please use: City, State with optional ZIP (e.g., Phoenix, AZ 85004)")
                return None

            # Normalize weights to sum to 1.0
            total_weight = location_weight + rating_weight + experience_weight
            if total_weight > 0:
                location_weight /= total_weight
                rating_weight /= total_weight
                experience_weight /= total_weight

            # Deliberately no insurance and no free-text requirements here:
            # scraped "accepted insurance" is unverifiable marketing (the
            # payer lives in the sidebar, feeding only the FHIR network
            # check), and free-form asks the judge can't verify skewed
            # results while opening the prompts to arbitrary input.
            return {
                "specialty": safe_specialty,
                "location": safe_location,
                "preferences": {
                    "location_weight": location_weight,
                    "rating_weight": rating_weight,
                    "experience_weight": experience_weight,
                },
                # Per-search, not a get_orchestrator() cache key: that factory
                # is keyed on (fhir_enabled, fast_demo) and sets os.environ at
                # construction, so routing this through it would rebuild the
                # orchestrator on every toggle.
                "use_cache": st.session_state.get("use_cache", True),
            }

    return None



def render_execution_timeline(execution_log: List[Dict[str, Any]]) -> None:
    """Render agent execution timeline from execution log.

    Args:
        execution_log: List of execution log entries from workflow
    """
    if not execution_log:
        st.info("No execution log available")
        return

    st.subheader("Agent Execution Timeline")

    # Group log entries by step
    step_groups = {}
    for entry in execution_log:
        step = entry.get("step", "unknown")
        if step not in step_groups:
            step_groups[step] = []
        step_groups[step].append(entry)

    # Step metadata. Insertion order here is the DISPLAY order (see the sort
    # below): review enrichment runs inside the scoring node, so its "started"
    # entry lands after the scorer's in the log, and grouping by first
    # appearance would print it after the step it happens before.
    step_info = {
        "initialize": {"icon": "🎯", "name": "Initialization", "agent": "Orchestrator"},
        "gather_data": {"icon": "🔍", "name": "Data Gathering", "agent": "DataGathererAgent"},
        # Its own row, and its own agent. This is a Tavily search plus a Haiku
        # extraction per provider — DataGathererAgent work that runs inside the
        # scoring node because the core ranking decides who gets it. Folded into
        # "Preference Scoring" it read as a slow scorer: 70.1s against the
        # scorer's actual ~15s share on 2026-07-28.
        "enrich_reviews": {"icon": "📚", "name": "Review Enrichment", "agent": "DataGathererAgent"},
        "score_providers": {"icon": "📊", "name": "Preference Scoring", "agent": "PreferenceScorerAgent"},
        "validate_rankings": {"icon": "🛡️", "name": "Validation", "agent": "CriticValidatorAgent"},
        "finalize_results": {"icon": "✨", "name": "Finalizing", "agent": "Orchestrator"},
        "handle_error": {"icon": "❌", "name": "Error Handling", "agent": "Orchestrator"}
    }

    # Canonical pipeline order, with anything unrecognised kept in log order
    # after it rather than dropped.
    _order = list(step_info)
    step_groups = dict(sorted(
        step_groups.items(),
        key=lambda kv: _order.index(kv[0]) if kv[0] in _order else len(_order),
    ))

    # Display each step
    for step, entries in step_groups.items():
        info = step_info.get(step, {"icon": "•", "name": step.title(), "agent": "Unknown"})

        # Find started and completed entries
        started_entry = next((e for e in entries if e.get("status") == "started"), None)
        completed_entry = next((e for e in entries if e.get("status") == "completed"), None)
        failed_entry = next((e for e in entries if e.get("status") == "failed"), None)

        # Real per-step duration recorded by the orchestrator
        elapsed_s = ((completed_entry or failed_entry or {}).get("details", {}) or {}).get("elapsed_s")
        duration_label = f" — {elapsed_s:.1f}s" if isinstance(elapsed_s, (int, float)) else ""

        with st.expander(f"{info['icon']} **{info['name']}** ({info['agent']}){duration_label}", expanded=False):
            # Status
            if failed_entry:
                st.error(f"❌ Failed: {failed_entry.get('details', {}).get('error', 'Unknown error')}")
            elif completed_entry:
                st.success("✅ Completed")
            elif started_entry:
                st.info("⏳ Started")

            # Display details from completed or started entry
            details_entry = completed_entry or started_entry
            if details_entry:
                details = details_entry.get("details", {})

                # Show key metrics
                if details:
                    # st.metric raises TypeError on a list or dict, which
                    # takes down the whole results page rather than one tile.
                    # finalize_results logs the workflow summary, whose fourth
                    # key is `other_providers` — a list — so simply enabling
                    # "Show agent internals" after any successful search
                    # replaced the page with a traceback.
                    scalars = [
                        (k, v) for k, v in details.items()
                        if isinstance(v, (str, int, float)) or v is None
                    ]
                    complex_items = [
                        (k, v) for k, v in details.items()
                        if not (isinstance(v, (str, int, float)) or v is None)
                    ]
                    if scalars:
                        metric_cols = st.columns(min(len(scalars), 4))
                        for i, (key, value) in enumerate(scalars[:4]):
                            with metric_cols[i]:
                                metric_name = key.replace("_", " ").title()
                                st.metric(metric_name, value)

                    # Show remaining details in a simple list
                    remaining = scalars[4:] + complex_items
                    if remaining:
                        st.markdown("**Additional Details:**")
                        for key, value in remaining:
                            if isinstance(value, (list, tuple)):
                                value = f"{len(value)} item(s)"
                            st.text(f"  • {key.replace('_', ' ').title()}: {value}")

            # Show all log entries in a simple list
            if len(entries) > 1:
                st.markdown("**All Events:**")
                for entry in entries:
                    st.caption(f"• {entry.get('status', 'unknown').title()}: {entry.get('timestamp', 'N/A')}")
                    if entry.get('details'):
                        st.json(entry['details'], expanded=False)


def render_cost_card(cost_summary: Dict[str, Any], search_metadata: Dict[str, Any] = None) -> None:
    """Render the per-search cost and timing summary card.

    `search_metadata` supplies the discovery query count. Ring expansion to
    nearby cities is the one part of a search whose cost varies with a decision
    the code makes silently, and until now nothing surfaced whether it fired:
    the log line goes to stdout rather than `logs/` (only `audit.log` is written
    there), and the debug tab read `search_metadata["query"]` — the singular
    representative — never `query_count`. A run's Tavily bill could double with
    no visible reason.
    """
    if not cost_summary:
        return

    total = cost_summary.get("total_usd", 0.0)
    elapsed = cost_summary.get("elapsed_s", 0.0)
    llm = cost_summary.get("llm", {})
    tavily = cost_summary.get("tavily", {})
    embeddings = cost_summary.get("embeddings", {})
    timings = cost_summary.get("step_timings", {})

    rows = []
    for model, stats in llm.get("by_model", {}).items():
        rows.append(
            f"<tr><td>{html.escape(model)}</td>"
            f"<td>{stats.get('calls', 0)} calls · "
            f"{stats.get('input_tokens', 0):,} in / {stats.get('output_tokens', 0):,} out</td>"
            f"<td>${stats.get('cost_usd', 0.0):.4f}</td></tr>"
        )
    if tavily.get("searches"):
        # Name the discovery queries separately when the ring fired: without it
        # the extra searches are indistinguishable from enrichment traffic.
        discovery_queries = (search_metadata or {}).get("query_count")
        discovery_note = ""
        if discovery_queries:
            discovery_note = f" · {discovery_queries} discovery queries"
            if (search_metadata or {}).get("ring_expanded"):
                discovery_note += " (expanded to nearby cities)"
        rows.append(
            f"<tr><td>Tavily web search</td>"
            f"<td>{tavily['searches']} searches · {tavily.get('credits', 0)} credits"
            f"{html.escape(discovery_note)}</td>"
            f"<td>${tavily.get('cost_usd', 0.0):.4f}</td></tr>"
        )
    if embeddings.get("tokens"):
        rows.append(
            f"<tr><td>{html.escape(embeddings.get('model', 'embeddings'))}</td>"
            f"<td>{embeddings['tokens']:,} tokens</td>"
            f"<td>${embeddings.get('cost_usd', 0.0):.4f}</td></tr>"
        )

    # A cache hit is invisible in a cost table — it shows up only as calls that
    # did NOT happen. State it explicitly, or the saving cannot be verified
    # without reading logs.
    cache = cost_summary.get("cache", {})
    if cache.get("lookups"):
        hits, misses = cache.get("hits", 0), cache.get("misses", 0)
        rows.append(
            f"<tr><td>Provider cache</td>"
            f"<td>{hits} reused · {misses} fetched live</td>"
            f"<td>&mdash;</td></tr>"
        )

    step_chips = "".join(
        f'<span class="cc-chip">{html.escape(step.replace("_", " "))} · {seconds:.1f}s</span>'
        for step, seconds in timings.items()
    )

    render_html(
        f"""
        <div class="cc-cost-card">
            <div class="cc-cost-head">
                <span class="cc-cost-total">${total:.4f}</span>
                <span class="cc-cost-sub">estimated cost of this search · {elapsed:.1f}s end to end</span>
            </div>
            <table class="cc-cost-table">{"".join(rows)}</table>
            <div class="cc-step-chips">{step_chips}</div>
            <div class="cc-cost-note">Estimated from list prices; Tavily approximated per credit. Actual billing may differ.</div>
        </div>
        """
    )


def _match_ring_svg(score: float) -> str:
    """Inline SVG match-score ring (accent stroke on a bone track)."""
    try:
        pct = max(0.0, min(float(score), 100.0))
    except (TypeError, ValueError):
        pct = 0.0
    circumference = 2 * 3.14159 * 27
    dash = pct / 100 * circumference
    return (
        f'<svg width="64" height="64" viewBox="0 0 64 64" role="img" aria-label="Match score {pct:.0f}">'
        f'<circle cx="32" cy="32" r="27" fill="none" stroke="{HEARTH["bone_200"]}" stroke-width="5"/>'
        f'<circle cx="32" cy="32" r="27" fill="none" stroke="{HEARTH["accent_500"]}" stroke-width="5" '
        f'stroke-dasharray="{dash:.1f} {circumference:.1f}" stroke-linecap="round" transform="rotate(-90 32 32)"/>'
        f'<text x="32" y="37" text-anchor="middle" font-size="15" font-weight="600" '
        f'fill="{HEARTH["clay_800"]}" font-family="Inter, sans-serif">{pct:.0f}</text>'
        f"</svg>"
    )


def _stars_markup(rating: float) -> str:
    """Star glyphs for a 0-5 rating."""
    try:
        value = max(0.0, min(float(rating), 5.0))
    except (TypeError, ValueError):
        return ""
    full = int(value)
    half = "½" if (value - full) >= 0.5 else ""
    return f'<span class="cc-stars">{"★" * full}{half}</span>'


def render_network_check(recommendations: List[Dict[str, Any]], workflow_results: Dict[str, Any]) -> None:
    """Verify top recommendations against the payer FHIR directory (prototype).

    Zero effect on ranking: results render as labeled evidence. Verification
    results are cached on the provider dicts, so Streamlit reruns don't repeat
    directory lookups.
    """
    from fhir.verify import verify_network

    payer = st.session_state.get("network_payer") or ""
    if not payer:
        st.caption("Pick an insurance in the sidebar to run the network check.")
        return

    search_metadata = (
        workflow_results.get("agent_outputs", {})
        .get("data_gatherer", {})
        .get("search_metadata", {})
    )
    specialty = search_metadata.get("specialty")
    location = search_metadata.get("location")

    esc = html.escape
    chips = []
    source = None
    for recommendation in recommendations[:5]:
        provider = recommendation.get("provider", {})
        # Cache per payer (provider dicts persist in session state across
        # reruns), and mirror the active payer's result into the single
        # `network_check` slot the card chips read — a payer switch must
        # never leave a stale verdict on the cards.
        checks = provider.setdefault("network_checks", {})
        if payer not in checks:
            checks[payer] = verify_network(
                provider.get("name", ""), payer, specialty=specialty, location=location
            )
        provider["network_check"] = checks[payer]
        check = checks[payer]
        source = check.get("source") or source

        name = esc(str(provider.get("name", "Unknown")))
        status = check.get("status")
        if status == "verified":
            chips.append(f'<span class="cc-chip cc-chip--moss">{name} &middot; in-network</span>')
        elif status == "no_record":
            chips.append(f'<span class="cc-chip">{name} &middot; no record</span>')
        else:
            chips.append(f'<span class="cc-chip">{name} &middot; unavailable</span>')

    if source == "sandbox":
        note = (
            "Prototype queried the sandbox FHIR directory — “no record” means "
            "not in demo data, never “not in network”. A real Plan-Net endpoint "
            "plugs in via FHIR_USE_MOCK=false."
        )
    else:
        note = "Queried the payer's live FHIR directory. Confirm coverage with the office before booking."

    render_html(
        f"""
        <div class="cc-card">
            <div class="cc-panel-head">
                <span class="cc-panel-title">Network check &middot; {esc(str(payer))}</span>
                <span class="cc-chip">FHIR directory prototype</span>
            </div>
            <div class="cc-chips" style="margin-top:12px">{"".join(chips)}</div>
            <div class="cc-cost-note">{note}</div>
        </div>
        """
    )


def _reorder_reconciliation(workflow_results: Dict[str, Any]) -> str:
    """One line telling the reader which positions the bias prose predates.

    The bias analysis names providers by RANK, and `refine_rankings` runs after
    it, so those ordinals can be stale by the time they render. Rather than
    rewriting the model's prose, state what changed underneath it — the moves
    are already computed and this reads the same structure the refinement note
    does, so the two cannot disagree.

    Empty when nothing moved, like every other conditional note in the panel.
    """
    moves = ((workflow_results.get("workflow_summary") or {}).get("refinement") or {}).get("moves") or []
    if not moves:
        return ""
    changed = ", ".join(
        f"{html.escape(str(m.get('name', 'Unknown')))} is now #{m.get('to', '?')}"
        for m in moves[:4]
    )
    return (
        f"<div style='margin-top:6px;opacity:.8'><i>Positions above are from before "
        f"the independent review. After it: {changed}.</i></div>"
    )


def refinement_note_markup(refinement: Dict[str, Any]) -> str:
    """How critic feedback re-ordered the ranking, as panel-ready markup.

    Returns markup rather than rendering, so the note lives INSIDE the
    Responsible-AI panel beside the bias and judge notes. It used to render as
    a sibling block immediately after the panel's own `render_html` — visually
    adjacent, structurally unrelated, and the comment at that call site already
    argued it "belongs to" the panel.

    Empty string when there is nothing to report, matching the conditional
    discipline of the other two notes: a permanent row that says nothing
    trains the eye to skip the row that matters.
    """
    refinement = refinement or {}
    moves = refinement.get("moves", [])

    if not moves:
        # The critique loop still ran — say so, or the story disappears on
        # runs where the validator agreed with the original order.
        if refinement.get("applied"):
            return (
                '<div class="cc-why" style="margin-top:12px">'
                "<b>Refined by critic review:</b> the validator challenged the ranking "
                "and confirmed the original order — no changes needed.</div>"
            )
        return ""

    items = []
    for move in moves:
        name = html.escape(str(move.get("name", "Unknown")))
        # A move with NO reasons is a DISPLACEMENT: this provider's score never
        # changed, they rose or fell because someone else's did. The fallback
        # here read "critic feedback", which asserts the validator said
        # something about them — on 2026-07-28 it labelled two such providers
        # that way, one of whom moved three places purely because Dr. Khan fell.
        # Same class as round 6's invented causality, on the same panel.
        reasons = move.get("reasons") or []
        detail = (
            html.escape("; ".join(reasons)) if reasons
            else "<i>no change to their own score — moved as others were re-scored</i>"
        )
        items.append(
            f"<div style='margin-top:4px'>&bull; <b>{name}</b> "
            f"#{move.get('from', '?')} &rarr; #{move.get('to', '?')} — {detail}</div>"
        )

    # The COUNT is of providers the validator actually adjusted, not of rows
    # that moved. `len(moves)` said "re-ordered 4 recommendation(s)" for a run
    # in which the critic had docked exactly one; the other three were that
    # one's wake. `adjusted_count` was already computed three lines away in
    # refine_rankings and never read.
    adjusted = refinement.get("adjusted_count")
    if not isinstance(adjusted, int):
        adjusted = sum(1 for m in moves if m.get("reasons"))
    displaced = len(moves) - adjusted
    headline = f"the validator's findings changed {adjusted} recommendation(s)"
    if displaced > 0:
        headline += f", which moved {displaced} more"

    return (
        f'<div class="cc-why" style="margin-top:12px">'
        f"<b>Refined by critic review:</b> {headline} — at no extra API cost or "
        f"latency. Match rings show the critic-adjusted score, so the order you "
        f"see follows the scores you see."
        f'{"".join(items)}</div>'
    )


def _score_breakdown_markup(provider: Dict[str, Any]) -> str:
    """Weighted-contribution bars for the deterministic 70% of the score."""
    breakdown = provider.get("score_breakdown")
    if not isinstance(breakdown, dict) or not breakdown:
        return ""

    rows = []
    for key, label in (("location", "Location"), ("rating", "Ratings"), ("experience", "Experience")):
        part = breakdown.get(key)
        if not isinstance(part, dict):
            continue
        try:
            score = max(0.0, min(float(part.get("score", 0)), 100.0))
            weight = float(part.get("weight", 0))
        except (TypeError, ValueError):
            continue
        rows.append(
            f'<div class="cc-bar-row"><span class="cc-bar-label">{label}</span>'
            f'<span class="cc-bar-track"><span class="cc-bar-fill" style="width:{score:.0f}%"></span></span>'
            f'<span class="cc-bar-val">{score:.0f}/100 &middot; weight {weight:.0%}</span></div>'
        )
    return "".join(rows)


# Fourth element is what to show when the judge cites nothing for a criterion.
# A blank line under a mid-range bar read as a rendering bug, so each band
# names its own absence.
#
# These describe the JUDGE'S CITATION, not the source material. The earlier
# wording ("...found in the sources", "...across the sources") asserted a
# property of the corpus, and on a live card that assertion was flatly false:
# practical_access rendered "No details on scheduling, wait times, or office
# responsiveness in the sources" directly beneath a review summary describing
# long wait times, which the judge had itself quoted under red_flags. The only
# thing this branch actually observes is that no snippet arrived for this key.
#
# They are also score-independent on purpose: the branch below reads only the
# evidence string, so a high score can reach this text too (48/50 beside a
# "none" sentinel). "Nothing was cited" stays true there; "nothing was found"
# would not.
_RUBRIC_DISPLAY = (
    ("review_substance", "Review substance", 50,
     "No review text was cited for this criterion."),
    ("red_flags", "Red flags & consistency", 30,
     "No consistency concerns were cited."),
    ("practical_access", "Practical access", 20,
     "No scheduling or wait-time evidence was cited."),
)

# What the judge emits when a neutral band applies (plus the near-misses a
# model reaches for), compared case-folded and stripped of trailing marks.
_NO_EVIDENCE_SENTINELS = {"no evidence", "none", "n/a", "na", "not applicable"}


def _rubric_markup(provider: Dict[str, Any]) -> str:
    """Bars + cited evidence for the AI judge's rubric (the 30% side)."""
    rubric = provider.get("ai_rubric")
    if not isinstance(rubric, dict) or not rubric:
        return ""

    evidence = provider.get("ai_evidence") or {}
    rows = []
    for key, label, cap, absent_note in _RUBRIC_DISPLAY:
        try:
            value = max(0.0, min(float(rubric.get(key, 0)), float(cap)))
        except (TypeError, ValueError):
            value = 0.0
        pct = value / cap * 100 if cap else 0
        rows.append(
            f'<div class="cc-bar-row"><span class="cc-bar-label" style="width:170px">{label}</span>'
            f'<span class="cc-bar-track"><span class="cc-bar-fill" style="width:{pct:.0f}%"></span></span>'
            f'<span class="cc-bar-val">{value:g}/{cap}</span></div>'
        )
        quote = str(evidence.get(key, "") or "").strip()
        if quote and quote.strip(" .!\"'").lower() not in _NO_EVIDENCE_SENTINELS:
            rows.append(f'<div class="cc-bar-quote">&ldquo;{html.escape(quote)}&rdquo;</div>')
        else:
            rows.append(
                f'<div class="cc-bar-quote cc-bar-nodata">{html.escape(absent_note)}</div>'
            )
    return "".join(rows)


def _strip_consider_prefix(text: Any) -> str:
    """Drop the mechanical "Consider " the critic's blind spots are wrapped in.

    `important_considerations` is built as `f"Consider {factor}"` over
    `blind_spots.missing_factors`. The factors are noun phrases describing what
    our ranking fails to model ("Recency and trend of reviews"), so the prefix
    produced "Consider Sentiment specificity vs volume:" — ungrammatical
    because it was never written as advice.
    """
    entry = re.sub(r"^consider\b[\s:]*", "", str(text or "").strip(), flags=re.IGNORECASE)
    entry = entry.strip()
    if entry:
        entry = entry[0].upper() + entry[1:]
    return entry


def _rating_without_count_pages(coverage: List[Dict[str, Any]]) -> int:
    """How many fetched pages gave a rating but no review count.

    That shape is the specific one that puts `— listing page` on a card. A
    rating-only observation loses the same-domain collapse on `has_pair`, the
    FIRST element of the strength tuple, so page kind is never consulted and a
    city directory takes the headline — even though the doctor's own profile
    was fetched and read.

    Counted rather than left in the JSON below because on 2026-07-29 reaching
    this conclusion took two rounds of inference over a 60c run, and the state
    it describes is a data-coverage gap, not a bug in the collapse.
    """
    return sum(
        1
        for row in coverage or []
        if isinstance(row, dict)
        for source in (row.get("sources") or [])
        if isinstance(source, dict)
        and isinstance(source.get("yielded"), dict)
        and source["yielded"].get("rating") is not None
        and not source["yielded"].get("review_count")
    )


def _judge_findings(validation_results: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(provider_name, finding) for every non-empty `recommendation_adjustments`.

    The critic's audit of the JUDGE, not of the provider: cases where a rubric
    criterion sits in its neutral band although the summary contains evidence
    for it, or a cited snippet has no basis in the text. These move NO score by
    design — `refine_rankings` never reads this field — so they are reported,
    never charged to the provider.

    Read from the raw critic entries rather than each provider's
    `critic_review["judge_findings"]`: the critic validates every provider the
    JUDGE scored while the cards render only the top 5, and a signal about our
    own scoring must not stop at rank 5.

    Round 10 bounded that set to the research budget — a provider the judge
    never scored has no rubric to audit — but the reasoning is unchanged inside
    it, and reading the per-card field would still truncate at 5.
    """
    entries = (
        (validation_results or {}).get("top_provider_validation", {}) or {}
    ).get("top_provider_validations", []) or []

    findings: List[Tuple[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        finding = str(entry.get("recommendation_adjustments", "") or "").strip()
        # The critic writes PASS verdicts into this field too, despite the
        # prompt asking for it to stay empty when scoring matches evidence.
        # Counting those told a patient "10 inconsistencies were found" when
        # the true count of judge errors was zero.
        if finding and is_judge_concern(finding):
            findings.append((str(entry.get("provider_name", "") or "Unknown"), finding))
    return findings


def _judge_findings_note(findings: List[Tuple[str, str]]) -> str:
    """Patient-facing callout for judge-consistency findings; "" when none.

    Carries the COUNT only. The finding text names internal criteria
    ("practical_access scored 10/20 …") and the judge is explicitly forbidden
    from putting that vocabulary in front of a patient — so the raw text stays
    on the developer surfaces (Detailed Agent Analysis, audit log) and no
    provider is named beside an admission that our own judge slipped.
    """
    if not findings:
        return ""

    count = len(findings)
    noun = "inconsistency" if count == 1 else "inconsistencies"
    verb = "was" if count == 1 else "were"
    return (
        '<div class="cc-why" style="margin-top:12px"><b>Judge review:</b> '
        "our AI judge's scoring was checked against the same patient-review text "
        f"the independent critic read. {count} {noun} {verb} found and logged for "
        "review — no provider's ranking was changed by it.</div>"
    )


def _withheld_note(withheld: Dict[str, Any]) -> str:
    """Patient-facing callout for providers kept out of the recommendations.

    A recommendation asserts three things completed for that provider: we found
    their details, our judge scored them, and our independent reviewer checked
    them. Providers missing any of those are still LISTED — in "Other providers
    considered", with their own reason — but they cannot be carded.

    Two registers, deliberately separated. Missing data is a gap in what the web
    holds and is nobody's fault; a stage of OURS not completing is our defect,
    and this panel is where the system reports its own errors. Collapsing them
    would tell a patient a provider was unverifiable when in fact we simply
    failed to score them.

    Counts only — no provider names beside an admission that our pipeline
    slipped, matching `_judge_findings_note`. Renders "" when nothing was
    withheld: a permanent "0 withheld" row trains the eye to skip the row that
    matters.
    """
    if not isinstance(withheld, dict):
        return ""

    no_data = int(withheld.get("no_data") or 0)
    ours = int(withheld.get("pipeline_failures") or 0)
    if not (no_data or ours):
        return ""

    sentences = []
    if no_data:
        verb = "was" if no_data == 1 else "were"
        noun = "provider" if no_data == 1 else "providers"
        sentences.append(
            f"{no_data} {noun} {verb} left out of the recommendations because we "
            "couldn't find enough verified information about them."
        )
    if ours:
        verb = "was" if ours == 1 else "were"
        noun = "provider" if ours == 1 else "providers"
        sentences.append(
            f"{ours} {noun} {verb} researched successfully, but our own scoring "
            "didn't finish for them, so we've held them back rather than show an "
            "incomplete recommendation."
        )

    return (
        '<div class="cc-why" style="margin-top:12px"><b>Not recommended:</b> '
        + " ".join(sentences)
        + " They're all listed under &ldquo;Other providers considered&rdquo;.</div>"
    )


def _pool_highlights(providers: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """At-a-glance superlatives across the shortlist — one winner each for
    proximity, review volume, and tenure, keyed by the provider's index.

    Each award needs at least two comparable providers, otherwise the
    superlative carries no information (being "closest" of one is trivial).
    Ties break toward the higher-ranked (earlier) provider.
    """
    highlights: Dict[int, List[str]] = {}

    def _award(idx: int, label: str) -> None:
        highlights.setdefault(idx, []).append(label)

    def _review_count(p: Dict[str, Any]) -> float:
        for key in ("blended_review_count", "review_count"):
            value = p.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        return 0.0

    def _effective_miles(p: Dict[str, Any]) -> Optional[float]:
        """The distance the SCORER compares, not the nominal one.

        A city-precision figure is one centroid shared by every provider in
        that city, so the scorer adds CITY_CENTROID_MARGIN_MILES before
        comparing it against a ZIP-measured distance. Reading the raw number
        here let an 8.0 mi city ESTIMATE take "Closest" from an 8.5 mi
        ZIP MEASUREMENT that the algorithm had ranked nearer — the chip
        contradicting the ranking beside it — and, when a whole pool shares
        one centroid, handed the badge to whichever tied provider happened to
        sort first.
        """
        miles = p.get("computed_distance_miles")
        if not isinstance(miles, (int, float)):
            return None
        if p.get("distance_precision") == "city":
            return float(miles) + CITY_CENTROID_MARGIN_MILES
        return float(miles)

    distances = [
        (i, m) for i, m in ((i, _effective_miles(p)) for i, p in enumerate(providers))
        if m is not None
    ]
    # An exact tie is not a superlative: providers sharing one city centroid
    # are genuinely indistinguishable, and picking one would overstate what we
    # measured — the same error as calling the centroid a measurement.
    if len(distances) >= 2:
        nearest = min(distances, key=lambda t: t[1])
        if sum(1 for _, m in distances if m == nearest[1]) == 1:
            _award(nearest[0], "Closest")

    reviewed = [(i, _review_count(p)) for i, p in enumerate(providers) if _review_count(p) > 0]
    if len(reviewed) >= 2:
        _award(max(reviewed, key=lambda t: t[1])[0], "Most reviewed")

    experienced = [
        (i, p.get("years_experience")) for i, p in enumerate(providers)
        if isinstance(p.get("years_experience"), (int, float)) and p.get("years_experience") > 0
    ]
    if len(experienced) >= 2:
        _award(max(experienced, key=lambda t: t[1])[0], "Most experienced")

    return highlights


def render_other_providers(others: List[Dict[str, Any]]) -> None:
    """Compact, transparency-only list of providers scored below the top 5."""
    others = [o for o in (others or []) if isinstance(o, dict)]
    if not others:
        return

    tier_labels = {
        "same_zip": "same ZIP", "same_city": "same city",
        "same_state": "same state", "different": "farther out",
    }
    # Why a provider is unrated. "We searched and found nothing" and "we never
    # searched" produce identical-looking rows otherwise, and only one of them
    # is a statement about the provider.
    unrated_reason = {
        "no_profile_found": "no reviews found",
        "identity_rejected": "reviews found but not verifiably theirs",
        "over_budget": "not searched — pool limit",
        "failed": "review lookup failed",
    }

    # Three groups, split by RECOMMENDABILITY. A recommendation asserts three
    # things completed: we found the provider's details, our judge scored them,
    # and our independent reviewer checked them. Anything less cannot be carded,
    # but is still shown here with its own reason.
    #
    # Note the first two groups ARE on one score scale — a provider we searched
    # and found nothing for was still judged and could still be docked by the
    # critic. Only `over_budget` is genuinely off-scale: it reached no model at
    # all, so its number is pure imputation, which is why the 2026-07-27 run put
    # four of them at an identical 64 above three researched providers at
    # 60/58/57 (the whole gap being one critic -8 each). The headings say
    # "why not recommended", not "why not comparable".
    eligible = [o for o in others if not o.get("withheld_reason")]
    no_data = [o for o in others
               if o.get("withheld_reason") in ("no_profile_found", "identity_rejected",
                                               "failed", "not_judged", "not_critiqued")]
    unresearched = [o for o in others if o.get("withheld_reason") == "over_budget"]
    groups = [g for g in (eligible, no_data, unresearched) if g]

    with st.expander(f"Other providers considered ({len(others)})", expanded=False):
        multiple = len(groups) > 1

        if eligible:
            if multiple:
                st.markdown(f"**Ranked below the top 5 ({len(eligible)})**")
            st.caption(
                "Fully researched, scored and independently reviewed — simply "
                "ranked below the shortlist."
            )
            _render_other_rows(eligible, tier_labels, unrated_reason)

        if no_data:
            if multiple:
                st.markdown(f"**Researched, but not recommendable ({len(no_data)})**")
            st.caption(
                "We searched for these providers but couldn't complete a full "
                "assessment — the reason is shown against each one. They're scored "
                "on the same scale, but we won't recommend a provider we can't "
                "show you the evidence for."
            )
            _render_other_rows(no_data, tier_labels, unrated_reason)

        if unresearched:
            budget = get_config().MAX_PROVIDERS_TO_ENRICH
            if multiple:
                st.markdown(f"**Found but not researched ({len(unresearched)})**")
            st.caption(
                f"These ranked below the top {budget} on the preliminary score, so they "
                f"fell outside this search's research budget. No reviews were retrieved "
                f"and no independent review was run — their match score is provisional "
                f"and not comparable with the scores above."
            )
            _render_other_rows(unresearched, tier_labels, unrated_reason)


def _render_other_rows(
    rows: List[Dict[str, Any]],
    tier_labels: Dict[str, str],
    unrated_reason: Dict[str, str],
) -> None:
    """One compact two-line entry per provider, shared by both groups."""
    for other in rows:
        name = str(other.get("name", "Unknown"))
        rank = str(other.get("rank", "?"))

        score = other.get("refined_score", other.get("final_score"))
        try:
            score_text = f"{float(score):.0f}"
        except (TypeError, ValueError):
            score_text = "—"

        bits = []
        specialty = str(other.get("specialty", "") or "").strip()
        if specialty:
            bits.append(specialty)

        blended = other.get("blended_rating")
        if blended is not None:
            rating_bit = f"★ {blended}/5 weighted"
            if other.get("blended_review_count"):
                rating_bit += f" · {other['blended_review_count']} reviews"
            if other.get("blended_platform_count"):
                rating_bit += f" · {other['blended_platform_count']} platforms"
            bits.append(rating_bit)
        elif other.get("rating"):
            rating_bit = f"★ {other['rating']}/5"
            if other.get("review_count"):
                rating_bit += f" ({other['review_count']} reviews)"
            bits.append(rating_bit)
        else:
            reason = unrated_reason.get(str(other.get("enrichment_outcome") or ""))
            bits.append(f"Unrated — {reason}" if reason else "Unrated")

        distance = other.get("computed_distance_miles")
        if isinstance(distance, (int, float)):
            # Same precision honesty as the top cards: a city centroid is
            # one coordinate shared by everyone in that city.
            suffix = " (city-level)" if other.get("distance_precision") == "city" else ""
            bits.append(f"~{distance} mi{suffix}")
        elif tier_labels.get(str(other.get("location_match") or "")):
            bits.append(tier_labels[str(other["location_match"])])

        critic_status = str(other.get("critic_status") or "").strip().lower()
        if critic_status and critic_status != "approved":
            bits.append(f"critic: {critic_status}")

        # Why this provider isn't a recommendation. The rating branch above
        # already says "Unrated — no reviews found" for the coverage cases, so
        # only add a reason it hasn't covered — otherwise the row says the same
        # thing twice.
        reason = str(other.get("withheld_reason") or "")
        if reason in ("not_judged", "not_critiqued"):
            label = str(other.get("withheld_label") or "").strip()
            if label:
                bits.append(f"withheld — {label}")

        # Two lines per row, with the score set apart — the old single run-on
        # line put the number that decided the ordering last, after the
        # least important field. `html.escape` into plain `st.markdown` also
        # double-escaped, so an "&" in a practice name rendered as "&amp;".
        st.markdown(f"**#{rank} · {name}** &nbsp;·&nbsp; **{score_text}** match")
        if bits:
            st.caption(" · ".join(bits))



def render_provider_card(
    provider: Dict[str, Any], rank: int, highlights: Optional[List[str]] = None
) -> None:
    """Render a Hearth-styled provider card.

    Every provider-derived string is escaped: values originate from web
    scraping and LLM extraction, and this HTML renders unescaped.

    highlights: at-a-glance superlatives this provider won across the
    shortlist (e.g. "Closest", "Most reviewed"), computed by the caller.
    """
    esc = html.escape

    name = esc(str(provider.get("name", "Unknown Provider")))
    specialty = esc(str(provider.get("specialty", "")))

    # Meta row: location · distance · phone · experience · rating
    meta_bits = []
    if provider.get("location"):
        meta_bits.append(esc(str(provider["location"])))
    # Prefer the code-computed straight-line distance (utils/geo.py) over a
    # page-stated one; the "~" signals it's approximate, not driving distance.
    # A city-precision figure is one centroid shared by every provider in that
    # city, so it says so — otherwise a whole pool showing an identical "8.0 mi"
    # reads as ten measurements that happen to agree.
    if provider.get("computed_distance_miles") is not None:
        miles = esc(str(provider["computed_distance_miles"]))
        if provider.get("distance_precision") == "city":
            meta_bits.append(f"~{miles} mi (city-level)")
        else:
            meta_bits.append(f"~{miles} mi")
    elif provider.get("distance"):
        meta_bits.append(esc(str(provider["distance"])))
    if provider.get("phone"):
        meta_bits.append(esc(str(provider["phone"])))
    if provider.get("years_experience"):
        meta_bits.append(f"{esc(str(provider['years_experience']))} yrs experience")

    rating = provider.get("rating", 0) or 0
    review_count = provider.get("review_count")
    # Labeled source: self-published pages are flagged "— practice site"
    review_source = label_source(provider.get("review_source_url"), provider.get("website"))
    blended = provider.get("blended_rating")
    if blended is not None:
        # The ranking scores this count-weighted figure — show the same
        # number the score uses; the traceable single-source headline moves
        # to the reviews expander ("Best single source")
        rating_bit = (
            f"{_stars_markup(blended)} {esc(str(blended))}/5 weighted · "
            f"{esc(str(provider.get('blended_review_count')))} reviews · "
            f"{esc(str(provider.get('blended_platform_count')))} platforms"
        )
        meta_bits.append(rating_bit)
    elif rating:
        rating_bit = f"{_stars_markup(rating)} {esc(str(rating))}/5"
        if review_count and review_source:
            rating_bit += f" ({esc(str(review_count))} reviews · {esc(review_source)})"
        elif review_count:
            rating_bit += f" ({esc(str(review_count))} reviews)"
        meta_bits.append(rating_bit)
    else:
        meta_bits.append("Unrated")

    meta_html = "".join(f"<span>{bit}</span>" for bit in meta_bits)

    # "Why this match" callout from the scorer's reasoning
    why_html = ""
    ai_reasoning = str(provider.get("ai_reasoning", "") or "").strip()
    if ai_reasoning:
        why_html = f'<div class="cc-why"><b>Why this match:</b> {esc(ai_reasoning)}</div>'

    # Chips are for signals worth knowing at a glance: network check, pool
    # superlatives, the critic's verdict, sentiment, caveats, rank movement.
    # Insurance lists are NOT chips — they are unverified directory data and
    # live in the AI-analysis section; only the FHIR network check speaks to
    # coverage on the card face.
    chips = []
    network_check = provider.get("network_check") or {}
    if network_check.get("status") == "verified":
        source = "sandbox" if network_check.get("source") == "sandbox" else "payer directory"
        chips.append(f'<span class="cc-chip cc-chip--moss">In {esc(source)}: in-network</span>')
    elif network_check.get("status") == "no_record" and network_check.get("source") == "sandbox":
        chips.append('<span class="cc-chip">Sandbox directory: no record</span>')

    # Pool superlatives — the one distinguishing fact about this provider
    # relative to the shortlist (closest / most reviewed / most experienced)
    for highlight in (highlights or []):
        chips.append(f'<span class="cc-chip cc-chip--moss">{esc(str(highlight))}</span>')

    # The critic's own verdict on this provider, surfaced at a glance
    card_critic = provider.get("critic_review") or {}
    critic_status = str(card_critic.get("status", "") or "").lower()
    if critic_status == "approved":
        verdict_label = "Critic approved"
        critic_conf = str(card_critic.get("confidence", "") or "").lower()
        if critic_conf:
            verdict_label += f" · {critic_conf} confidence"
        chips.append(f'<span class="cc-chip cc-chip--moss">{esc(verdict_label)}</span>')
    elif critic_status == "conditional":
        chips.append('<span class="cc-chip">Critic: conditional</span>')
    elif critic_status == "rejected":
        chips.append('<span class="cc-chip cc-chip--rust">Critic flagged</span>')

    sentiment = str(provider.get("review_sentiment", "unknown") or "unknown").lower()
    if sentiment == "positive":
        chips.append('<span class="cc-chip cc-chip--moss">Positive reviews</span>')
    elif sentiment == "negative":
        chips.append('<span class="cc-chip cc-chip--rust">Negative reviews</span>')
    elif sentiment == "mixed":
        chips.append('<span class="cc-chip">Mixed reviews</span>')

    data_warnings = provider.get("data_warnings", []) or []
    if data_warnings:
        chips.append(f'<span class="cc-chip cc-chip--rust">Data caveats ({len(data_warnings)})</span>')

    # Rank movement from the critic-feedback refinement pass
    pre_rank = provider.get("pre_refinement_rank")
    if isinstance(pre_rank, int) and pre_rank != rank:
        if pre_rank > rank:
            chips.append(f'<span class="cc-chip cc-chip--moss">Moved up from #{pre_rank}</span>')
        else:
            chips.append(f'<span class="cc-chip cc-chip--rust">Moved down from #{pre_rank}</span>')

    # Expectation management: an 80-ring on a weak pool must not read as an
    # endorsement. When the TOP pick still carries mixed/negative reviews or
    # critic red flags, say "best available" out loud.
    if rank == 1:
        top_critic_flags = (provider.get("critic_review") or {}).get("red_flags")
        if sentiment in ("mixed", "negative"):
            chips.append('<span class="cc-chip cc-chip--rust">Best available — reviews mixed</span>')
        elif top_critic_flags:
            chips.append('<span class="cc-chip cc-chip--rust">Best available — critic flagged concerns</span>')

    # The ring must show the score the final order is actually sorted by:
    # the critic-adjusted score when refinement ran, else the scorer's
    # composite. Showing the raw composite made rank and score disagree
    # whenever the critic re-ordered providers.
    try:
        display_score = float(provider.get("refined_score", provider.get("final_score", 0)) or 0)
    except (TypeError, ValueError):
        display_score = 0.0
    try:
        adjustment = float(provider.get("refinement_adjustment") or 0)
    except (TypeError, ValueError):
        adjustment = 0.0
    adjustment_html = ""
    if adjustment:
        tint = HEARTH["moss_600"] if adjustment > 0 else HEARTH["rust_600"]
        adjustment_html = f'<div class="cc-ring-label" style="color:{tint}">Critic {adjustment:+g}</div>'

    npi_html = ""
    fhir_metadata = provider.get("fhir_metadata", {}) or {}
    if fhir_metadata.get("npi"):
        npi_html = f'<div class="cc-ring-label">NPI {esc(str(fhir_metadata["npi"]))}</div>'

    render_html(
        f"""
        <div class="cc-card">
          <div class="cc-card-top">
            <div style="flex:1; min-width:0;">
              <div class="cc-name"><span class="cc-rank">{rank}.</span>{name}</div>
              <div class="cc-spec">{specialty}</div>
              <div class="cc-meta">{meta_html}</div>
              {why_html}
              <div class="cc-chips">{"".join(chips)}</div>
            </div>
            <div class="cc-ring-wrap">
              {_match_ring_svg(display_score)}
              <div class="cc-ring-label">Match</div>
              {adjustment_html}
              {npi_html}
            </div>
          </div>
        </div>
        """
    )

    # Review Summary (only when actual patient feedback exists). The flag is
    # reused by the AI-analysis tail: the review source link belongs in
    # exactly one place, and this expander is the better one when it renders.
    review_summary = provider.get("review_summary", "")
    reviews_shown = bool(
        review_summary
        and review_summary != "No reviews available"
        and sentiment != "unknown"
        and len(review_summary.strip()) > 0
    )
    if reviews_shown:
        with st.expander("Patient reviews summary", expanded=False):
            st.markdown(f"**Overall sentiment:** {sentiment.title()}")
            st.markdown(f"> {review_summary}")
            review_url = provider.get("review_source_url")
            # When the meta row shows the cross-platform blend, this line
            # keeps the single most authoritative page traceable
            if blended is not None and rating:
                src_label = "Best single source"
                headline_bit = f" {rating}/5" + (f" ({review_count} reviews)" if review_count else "")
            else:
                src_label = "Source"
                headline_bit = ""
            if review_source and linkable(review_url):
                st.caption(f"{src_label}: [{review_source}]({review_url}){headline_bit}")
            elif review_source:
                st.caption(f"{src_label}: {review_source}{headline_bit}")

            # Platforms disagree about doctors more often than you'd hope —
            # show every stated rating so one platform's outlier can't hide
            observations = [
                o for o in (provider.get("review_observations") or [])
                if o.get("rating") is not None or o.get("review_count")
            ]
            if len(observations) >= 2:
                bits = []
                for obs in observations[:4]:
                    labeled = label_source(obs.get("source_url"), provider.get("website")) or "unknown"
                    text = labeled
                    if obs.get("rating") is not None:
                        text += f" {obs['rating']}/5"
                    if obs.get("review_count"):
                        text += f" ({obs['review_count']} reviews)"
                    if linkable(obs.get("source_url")):
                        bits.append(f"[{text}]({obs['source_url']})")
                    else:
                        bits.append(text)
                st.caption("Across platforms: " + " · ".join(bits))

    # AI analysis: reasoning, the two score graphs, judge strengths/concerns,
    # the critic's review (+ how it moved the rank), insurance, caveats, sources
    breakdown_markup = _score_breakdown_markup(provider)
    if (ai_reasoning or data_warnings or breakdown_markup
            or provider.get("critic_review") or provider.get("refinement_reasons")
            or provider.get("insurance_accepted")):
        with st.expander("AI analysis", expanded=False):
            if ai_reasoning:
                st.markdown("**Why this provider was ranked here:**")
                st.markdown(ai_reasoning)

            # Graph 1 — the deterministic 70%: the user's own weighted factors
            if breakdown_markup:
                st.markdown("**Weighted preference algorithm — 70% of final score:**")
                render_html(breakdown_markup)

            # Graph 2 — the AI judge's 30%: rubric-scored review evidence
            rubric_markup = _rubric_markup(provider)
            if rubric_markup:
                st.markdown(
                    "**Rubric-scored AI judge — 30% of final score** "
                    "(anchored criteria with cited evidence):"
                )
                render_html(rubric_markup)

            # Strengths / considerations are the AI JUDGE's read of the review
            # evidence — labeled so their origin is unambiguous vs the critic
            if provider.get("ai_strengths"):
                st.markdown("**Strengths (AI judge):**")
                for strength in provider.get("ai_strengths", []):
                    st.markdown(f"- {strength}")

            if provider.get("ai_concerns"):
                st.markdown("**Considerations (AI judge):**")
                for concern in provider.get("ai_concerns", []):
                    st.markdown(f"- {concern}")

            # The critic's independent review of this provider AND how its
            # verdict re-ordered the ranking — one block so the whole critic
            # story reads together (the refinement note was its own heading).
            critic_review = provider.get("critic_review") or {}
            refinement_reasons = provider.get("refinement_reasons") or []
            if (critic_review.get("notes") or critic_review.get("red_flags")
                    or critic_review.get("considerations") or refinement_reasons):
                header = "Critic's independent review"
                status = str(critic_review.get("status", "") or "").replace("_", " ")
                confidence = str(critic_review.get("confidence", "") or "")
                qualifiers = [q for q in (status, f"{confidence} confidence" if confidence else "") if q]
                if qualifiers:
                    header += f" ({', '.join(qualifiers)})"
                # Verdict-rubric legend lives in the ? tooltip so the section
                # stays scannable — same entry criteria the critic is held to
                st.markdown(
                    f"**{header}:**",
                    help=(
                        "How to read this — **approved**: evidence is consistent, no "
                        "disqualifying signal (the expected verdict for a clean "
                        "provider). **conditional**: one named concern plus what would "
                        "resolve it. **rejected**: quotable disqualifying evidence. "
                        "Confidence reflects evidence volume: **high** = 2+ independent "
                        "platforms agreeing, **low** = no independent platform evidence "
                        "at all. Missing data is never counted as a red flag."
                    ),
                )
                if critic_review.get("notes"):
                    st.markdown(critic_review["notes"])
                for flag in critic_review.get("red_flags", []) or []:
                    st.markdown(f"- ⚠️ {flag}")
                # (the critic's patient-facing note closes the whole expander —
                #  see the coverage block below)
                # How the critic's verdict moved this provider in the ranking
                if refinement_reasons:
                    effect_label = (
                        f"Ranking effect ({adjustment:+g} points):" if adjustment
                        else "Ranking effect:"
                    )
                    st.markdown(f"_{effect_label}_")
                    for reason in refinement_reasons:
                        st.markdown(f"- {reason}")

            if data_warnings:
                st.markdown("**Data quality notes:**")
                for warning in data_warnings:
                    st.caption(f"- {warning}")

            # Coverage closes the expander: the critic's patient-facing note,
            # the directory plan list, and ONE verification instruction, in a
            # single line. Scraped plan names are unverified, so they are a
            # patient note here and never a card chip — the sidebar FHIR
            # network check is the only thing that confirms coverage.
            insurance_list = provider.get("insurance_accepted", [])
            if isinstance(insurance_list, list):
                insurance_names = [str(x).strip() for x in insurance_list if str(x).strip()]
            elif insurance_list:
                insurance_names = [str(insurance_list).strip()]
            else:
                insurance_names = []

            considerations = str(critic_review.get("considerations", "") or "").strip()
            note_bits = [considerations] if considerations else []
            if insurance_names:
                note_bits.append("Directories list: " + " · ".join(insurance_names[:8]) + ".")
            if note_bits:
                # The critic often closes its own note with the same advice;
                # say it once, whoever said it.
                if "verify" not in " ".join(note_bits).lower():
                    note_bits.append("Verify coverage with the provider directly.")
                st.caption("For patients: " + " ".join(note_bits))

            # Where the displayed facts came from. The insurance page always
            # earns a link here; the review page only when the Patient-reviews
            # expander did not already show it (it can be set from a bare
            # rating with no review text, which that expander suppresses).
            source_pairs = []
            if not reviews_shown:
                source_pairs.append(("Review source", provider.get("review_source_url")))
            source_pairs.append(("Insurance source", provider.get("insurance_source_url")))
            for label, url in source_pairs:
                labeled = label_source(url, provider.get("website"))
                if not labeled:
                    continue
                link = f"[{labeled}]({url})" if linkable(url) else labeled
                st.caption(f"{label}: {link}")

            # Naming plans while silently omitting where they came from asks the
            # patient to trust an unattributed claim. Same rule the rubric bands
            # follow: when the evidence is missing, say it is missing.
            if insurance_names and not provider.get("insurance_source_url"):
                st.caption(
                    "Insurance source: not recorded for this provider — treat the "
                    "plan list above as unverified."
                )


def _render_withheld_detail(
    withheld: Dict[str, Any], others: List[Dict[str, Any]]
) -> None:
    """Name every provider held back from the recommendations, and why.

    Developer surface. The reasons name internal pipeline stages, so the
    patient-facing panel gets the count only (`_withheld_note`) and the detail
    lives here — the same split the judge-consistency findings use.

    Our own failures are listed FIRST and marked, because they are the
    actionable ones: a provider whose data we successfully found but whom our
    judge or critic never scored represents a stage we paid for and did not get.
    """
    if not isinstance(withheld, dict) or not withheld.get("total"):
        return

    rows = [o for o in others if isinstance(o, dict) and o.get("withheld_reason")]
    if not rows:
        return

    ours = [r for r in rows if r.get("withheld_reason") in ("not_judged", "not_critiqued")]
    theirs = [r for r in rows if r not in ours]

    st.markdown(f"**Withheld from recommendations ({len(rows)}):**")
    for row in ours + theirs:
        marker = " ⚠️" if row in ours else ""
        label = str(row.get("withheld_label") or row.get("withheld_reason") or "unknown")
        st.markdown(f"• **{row.get('name', 'Unknown')}**{marker} — {label}")
    if ours:
        st.caption(
            f"⚠️ {len(ours)} of these were researched successfully — the missing step "
            "is ours, not a gap in what the web holds. Every provider listed here is "
            "still shown under \"Other providers considered\"."
        )
    else:
        st.caption(
            "All of these are gaps in what we could find or verify, not pipeline "
            "failures. They remain listed under \"Other providers considered\"."
        )


def render_agent_workflow(workflow_results: Dict[str, Any]) -> None:
    """Render agent workflow visualization."""
    st.header("Agent Decision Process")

    # Workflow overview
    execution_log = workflow_results.get("execution_log", [])
    agent_outputs = workflow_results.get("agent_outputs", {})

    # Agent status indicators
    col1, col2, col3 = st.columns(3)

    with col1:
        data_status = agent_outputs.get("data_gatherer", {}).get("status", "unknown")
        status_color = "🟢" if data_status == "success" else "🔴"
        st.markdown(f"{status_color} **Data Gatherer**")
        st.markdown(f"Status: {data_status.title()}")
        if data_status == "success":
            data_meta = agent_outputs.get("data_gatherer", {}).get("search_metadata", {})
            provider_count = len(agent_outputs.get("data_gatherer", {}).get("providers", []))
            fhir_count = data_meta.get("fhir_count", 0)
            st.markdown(f"Found: {provider_count} providers")
            if fhir_count > 0:
                st.markdown(f":green[FHIR verified: {fhir_count}]")

    with col2:
        score_status = agent_outputs.get("preference_scorer", {}).get("status", "unknown")
        status_color = "🟢" if score_status == "success" else "🔴"
        st.markdown(f"{status_color} **Preference Scorer**")
        st.markdown(f"Status: {score_status.title()}")
        if score_status == "success":
            top_provider = agent_outputs.get("preference_scorer", {}).get("scoring_metadata", {}).get("top_provider")
            if top_provider:
                st.markdown(f"Top: {top_provider}")

    with col3:
        validation_status = agent_outputs.get("critic_validator", {}).get("status", "unknown")
        status_color = "🟢" if validation_status == "success" else "🔴"
        st.markdown(f"{status_color} **Critic Validator**")
        st.markdown(f"Status: {validation_status.title()}")
        if validation_status == "success":
            confidence = agent_outputs.get("critic_validator", {}).get("validation_metadata", {}).get("ranking_confidence", "unknown")
            st.markdown(f"Confidence: {confidence.title()}")

    # Providers held back from the recommendations, named and itemised. The
    # Responsible-AI panel carries the COUNT for patients; the reasons name
    # internal pipeline stages ("our rubric scoring did not complete"), so the
    # per-provider detail belongs on this developer surface — same split the
    # judge-consistency findings use.
    _render_withheld_detail(
        (workflow_results.get("workflow_summary") or {}).get("withheld") or {},
        (workflow_results.get("workflow_summary") or {}).get("other_providers") or [],
    )

    # Detailed agent outputs
    with st.expander("🔍 Detailed Agent Analysis"):
        tab1, tab2, tab3 = st.tabs(["Data Gatherer", "Preference Scorer", "Critic Validator"])

        with tab1:
            data_output = agent_outputs.get("data_gatherer", {})
            search_metadata = data_output.get("search_metadata", {})

            st.markdown("**Search Parameters:**")
            st.json({
                # `queries` plural, not the singular representative: three
                # phrasings run at home and two more if the ring fires, and the
                # old singular read made that difference invisible.
                "queries": search_metadata.get("queries", []),
                "query_count": search_metadata.get("query_count", 0),
                "ring_expanded": search_metadata.get("ring_expanded", False),
                "specialty": search_metadata.get("specialty", "N/A"),
                "location": search_metadata.get("location", "N/A"),
                "total_found": search_metadata.get("total_found", 0)
            })

            # What ring expansion bought, not just that it fired. `ring_expanded`
            # is a boolean; it never said whether the two extra searches
            # contributed anything that reached a patient. The ring also FILLS
            # the research budget, so each provider it adds costs an enrichment
            # search, a judge slot and an Opus verdict on top of the discovery
            # spend — which makes "added, but none shortlisted" the signal that
            # MIN_CANDIDATE_POOL is set too high.
            ring = (workflow_results.get("workflow_summary") or {}).get("ring_contribution") or {}
            if search_metadata.get("ring_expanded"):
                st.markdown("**Ring expansion — what it bought:**")
                st.caption(
                    f"{ring.get('added', 0)} candidate(s) added from nearby cities · "
                    f"{ring.get('researched', 0)} researched · "
                    f"{ring.get('shortlisted', 0)} reached the recommendations"
                )

            # Review coverage per researched provider. Three different failures
            # produce the same finished card — the platform's profile was never
            # returned by the search, it was returned but yielded no
            # observation, or it yielded one that lost the same-domain collapse
            # to a directory listing — and they need different fixes. The
            # 2026-07-28 run carded "healthgrades.com — listing page" as the
            # best single source for two providers with nothing recording which
            # of the three had happened.
            #
            # `profile_backed_platforms` restores the coverage measure
            # `_has_profile_source` provided until the 2026-07-25
            # enrichment-uniformity phase deleted it with the tier predicates it
            # sat among. It is also the only thing that measures ratemds's
            # probation criterion ("a clean profile-based rating+count
            # pair"), which has been assessed by eye for three rounds.
            coverage = (workflow_results.get("workflow_summary") or {}).get("review_coverage") or []
            if coverage:
                listing_only = [
                    row for row in coverage
                    if row.get("platform_pairs") and not row.get("profile_backed_platforms")
                ]
                rating_only = _rating_without_count_pages(coverage)
                st.markdown("**Review source coverage:**")
                st.caption(
                    f"{len(coverage)} researched · "
                    f"{len(listing_only)} with platform ratings but no profile-backed source"
                    + (f" · {rating_only} page(s) gave a rating with no count"
                       if rating_only else "")
                )
                st.json(coverage)

        with tab2:
            score_output = agent_outputs.get("preference_scorer", {})
            scoring_metadata = score_output.get("scoring_metadata", {})

            st.markdown("**Scoring Summary:**")
            st.json({
                "total_providers": scoring_metadata.get("total_providers", 0),
                "scoring_method": scoring_metadata.get("scoring_method", "N/A"),
                "score_range": scoring_metadata.get("score_range", {})
            })

        with tab3:
            validation_output = agent_outputs.get("critic_validator", {})
            validation_results = validation_output.get("validation_results", {})

            # Show bias analysis
            if validation_results.get("bias_analysis"):
                bias_assessment = validation_results["bias_analysis"].get("bias_assessment", {})
                st.markdown("**Bias Assessment:**")
                st.markdown(f"**Severity:** {bias_assessment.get('severity', 'unknown').title()}")

                detected_biases = bias_assessment.get('detected_biases', [])
                if detected_biases:
                    st.markdown("**Detected Biases:**")
                    for bias in detected_biases:
                        st.markdown(f"• {bias}")

                # Developer surface: prefer the technical register, which
                # carries the field names and weighted_contribution arithmetic
                # the patient-facing copy is forbidden from using. The fallback
                # runs ONLY in this direction — an unconstrained string must
                # never reach the panel by default.
                technical = str(bias_assessment.get("technical_explanation", "") or "").strip()
                explanation = technical or bias_assessment.get("explanation") or "No explanation provided"
                st.markdown(f"**Explanation:** {explanation}")
                if technical and bias_assessment.get("explanation"):
                    st.caption("Patient-facing wording differs — see the Responsible AI panel.")

            # The critic's audit of the JUDGE — raw text, which names internal
            # rubric criteria and so belongs here rather than in the
            # patient-facing panel. Absent entirely when the judge's scoring
            # matched its evidence, which is the expected case.
            judge_findings = _judge_findings(validation_results)
            if judge_findings:
                st.markdown(f"**Judge consistency findings ({len(judge_findings)}):**")
                for provider_name, finding in judge_findings:
                    st.markdown(f"• **{provider_name}** — {finding}")
                st.caption(
                    "These are faults in our own scoring, not the provider's — they are "
                    "reported and logged, and move no provider's score or rank."
                )


def render_validation_insights(workflow_results: Dict[str, Any]) -> None:
    """Render the critic's review as a public Responsible-AI panel.

    Surfaces the bias check, ranking confidence, and red flags the validator
    already produces — previously reachable only through admin-gated views.
    """
    validation_output = workflow_results.get("agent_outputs", {}).get("critic_validator", {})
    validation_results = validation_output.get("validation_results", {})
    if not validation_results:
        return

    esc = html.escape

    final_recommendations = validation_results.get("final_recommendations", {}) or {}
    confidence = str(final_recommendations.get("recommendation_confidence", "medium") or "medium").lower()

    bias = (validation_results.get("bias_analysis", {}) or {}).get("bias_assessment", {}) or {}
    severity = str(bias.get("severity", "unknown") or "unknown").lower()
    detected_biases = [str(b).strip() for b in bias.get("detected_biases", []) or [] if str(b).strip()]
    # Patient-facing. `technical_explanation` deliberately does NOT fall back
    # into this slot: an unconstrained string must never reach the panel by
    # default. The fallback runs the other way, on the developer surface only.
    bias_explanation = str(bias.get("explanation", "") or "").strip()

    red_flags = []
    top_validation = validation_results.get("top_provider_validation", {}) or {}
    for validation in top_validation.get("top_provider_validations", []) or []:
        red_flags.extend(str(f).strip() for f in validation.get("red_flags", []) or [] if str(f).strip())

    confidence_chip = {"high": "cc-chip--moss", "low": "cc-chip--rust"}.get(confidence, "")
    if detected_biases:
        _n = len(detected_biases)
        bias_value = f"{_n} potential bias{'' if _n == 1 else 'es'} flagged"
    elif severity in ("low", "medium"):
        bias_value = f"{esc(severity.title())} severity &middot; none blocking"
    else:
        bias_value = "Reviewed"
    flags_value = f"{len(red_flags)} raised on top picks" if red_flags else "None raised"

    # The bias analysis runs BEFORE `refine_rankings` — it has to, because the
    # verdicts that drive the refinement come from the critic's other call. So
    # every ordinal in its prose describes the PRE-refinement order, and the
    # panel then renders it above cards numbered by the final one.
    #
    # On 2026-07-28 that put "Dr. Khan (ranked 3rd) has a higher review score"
    # three lines above "Dr. Mohammad B. Khan, MD #3 -> #6", with his card
    # numbered 6. The panel contradicted itself on one screen.
    #
    # Labelled and reconciled rather than rewritten: the moves are already in
    # hand, and post-processing ordinals out of model prose is the kind of
    # regex-over-narrative that breaks silently.
    bias_note = ""
    if bias_explanation and (detected_biases or severity in ("medium", "high")):
        bullets = "".join(f"<div style='margin-top:4px'>&bull; {esc(b)}</div>" for b in detected_biases[:4])
        bias_note = (
            f'<div class="cc-why" style="margin-top:12px"><b>Bias check</b> '
            f"<span style='opacity:.7'>(read before the independent review re-ordered "
            f"the list)</span><b>:</b> {esc(bias_explanation)}{bullets}"
            f"{_reorder_reconciliation(workflow_results)}</div>"
        )

    # The critic's audit of our own judge. Renders only when a concern exists —
    # a permanent "0 inconsistencies" tile would train the eye to skip the row
    # that matters. Count only, no jargon: see _judge_findings_note.
    judge_note = _judge_findings_note(_judge_findings(validation_results))
    # Providers kept OFF the cards, and why. Same conditional discipline and the
    # same count-only rule as the judge note above it.
    withheld_note = _withheld_note(
        (workflow_results.get("workflow_summary") or {}).get("withheld") or {}
    )
    # The refinement note is the CONSEQUENCE of the two notes above it, so it
    # renders after them inside the same panel.
    refinement_note = refinement_note_markup(
        workflow_results.get("workflow_summary", {}).get("refinement", {})
    )

    # These are the bias analysis's `blind_spots.missing_factors` — gaps in OUR
    # OWN ranking, which the critic prompt defines as "what important factors
    # might be missing". Presenting them under "Before you book" turned our
    # self-critique into patient advice nobody could act on ("the ranking
    # treats them as interchangeable with no clinical-fit dimension"), and the
    # mechanical "Consider " prefix is why they read ungrammatically. Named
    # honestly, this becomes the most useful section on the panel.
    considerations = [
        _strip_consider_prefix(c)
        for c in final_recommendations.get("important_considerations", []) or []
        if str(c).strip()
    ]
    considerations = [c for c in considerations if c]

    # `user_guidance` is the one genuinely patient-facing list the critic
    # produces, and until now it was computed and rendered NOWHERE — real
    # guidance discarded while the blind spots above were dressed up as
    # guidance. It belongs here: the gaps say what the ranking misses, and
    # these say what to do about it. Deduped case-insensitively so a repeat of
    # a blind spot doesn't appear twice.
    _seen = {c.lower() for c in considerations}
    guidance = []
    for item in final_recommendations.get("user_guidance", []) or []:
        entry = str(item).strip()
        if entry and entry.lower() not in _seen:
            _seen.add(entry.lower())
            guidance.append(entry)

    considerations_html = ""
    if considerations or guidance:
        items = "".join(f"<div style='margin-top:4px'>&bull; {esc(c)}</div>"
                        for c in considerations[:3] + guidance[:2])
        considerations_html = (
            '<div style="margin-top:12px"><b>What this ranking doesn\'t capture</b>'
            f'{items}</div>'
        )

    render_html(
        f"""
        <div class="cc-card">
            <div class="cc-panel-head">
                <span class="cc-panel-title">Responsible AI review</span>
                <span class="cc-chip {confidence_chip}">{esc(confidence.title())} confidence</span>
            </div>
            <div class="cc-trust-grid">
                <div class="cc-trust-item">
                    <div class="cc-trust-label">Bias check</div>
                    <div class="cc-trust-value">{bias_value}</div>
                </div>
                <div class="cc-trust-item">
                    <div class="cc-trust-label">Red flags</div>
                    <div class="cc-trust-value">{flags_value}</div>
                </div>
            </div>
            {bias_note}
            {judge_note}
            {withheld_note}
            {refinement_note}
            {considerations_html}
            <div class="cc-cost-note">
                Specialties are checked against an allowlist before reaching any model, and
                every provider we recommend has been reviewed independently by a second AI.
            </div>
        </div>
        """
    )


def data_gatherer_status(workflow_results: Dict[str, Any]) -> str:
    """The data-gathering step's outcome status for a finished workflow.

    "no_results" is a benign outcome (the search worked, nobody matched) and
    must not be presented like a system failure. Exception paths return empty
    agent_outputs, so they yield "" and fall through to the error rendering.
    """
    return (
        workflow_results.get("agent_outputs", {})
        .get("data_gatherer", {})
        .get("status", "")
    )


def main():
    """Main application function."""
    init_session_state()
    render_header()

    config = get_config()
    rate_limiter = get_rate_limiter()

    # Authentication check
    authenticator = get_authenticator()

    # If not authenticated, show login form
    if not authenticator.require_authentication():
        authenticator.login_form()
        return

    # Check API keys
    if not check_api_keys():
        return

    # Sidebar configuration
    with st.sidebar:
        # Account block. The public demo auto-logs-in as dev_admin; that
        # should read as "demo", not leak internal account plumbing.
        username = st.session_state.get("username", "User")
        role = st.session_state.get("user_role", "user")
        if username == "dev_admin":
            st.markdown("### Demo mode")
            st.caption("Explore freely — no account needed.")
        else:
            st.markdown(f"### {username}")
            st.caption(f"Role: {role}")
            if st.button("Logout", use_container_width=True):
                authenticator.logout()

        st.markdown("---")
        st.header("Configuration")

        # Fast demo mode: shallower web search + lighter enrichment
        st.session_state.fast_demo = st.toggle(
            "Fast demo mode",
            value=st.session_state.fast_demo,
            help="Basic search depth and lighter review enrichment for quick, "
                 "inexpensive demos. Platform review lookups always run at "
                 "full depth. Turn off for the deepest provider search."
        )

        # FHIR network-check toggle (verification prototype — not a data source)
        st.session_state.fhir_enabled = st.toggle(
            "Network check (FHIR prototype)",
            value=st.session_state.fhir_enabled,
            help="Verify top matches against the payer's FHIR directory. Runs on "
                 "sandbox data by default; a real Plan-Net endpoint plugs in via "
                 "FHIR_USE_MOCK=false. Never affects ranking."
        )

        # The payer lives HERE, not in the search form: scraped "accepted
        # insurance" lists are unverifiable marketing (never filtered or
        # scored on), so insurance feeds only the directory verification —
        # and switching payers re-checks the SAME results with no new search.
        if st.session_state.fhir_enabled:
            st.session_state.network_payer = st.selectbox(
                "Insurance for network check",
                ["", "Aetna", "Blue Cross Blue Shield", "Cigna", "UnitedHealth",
                 "Medicare", "Medicaid", "Other"],
                index=0,
                help="Checked against the payer directory only — not used in "
                     "search or scoring."
            )

        # Enrichment cache: reuse review/tenure/insurance evidence gathered by
        # earlier searches instead of paying Tavily + Haiku for it again.
        st.session_state.use_cache = st.toggle(
            "Use cached provider data",
            value=st.session_state.get("use_cache", True),
            help=f"Reuse enrichment stored by earlier searches (refreshed every "
                 f"{int(get_config().PROVIDER_CACHE_TTL_DAYS)} days). Turn off to "
                 "force a full live fetch — slower and a few cents more, but "
                 "useful for demos and for checking the cache against a cold run. "
                 "Distance is always recomputed for your location, never reused."
        )

        # Agent internals toggle (admins only; silent for everyone else)
        if authenticator.is_admin():
            st.session_state.show_agent_logs = st.toggle("Show agent internals", value=False)
        else:
            st.session_state.show_agent_logs = False

        # Clear results button
        if st.button("Clear Results"):
            st.session_state.workflow_results = None
            st.session_state.search_executed = False
            st.rerun()

        # Cache clearing is DESTRUCTIVE and lives below a divider, away from
        # "Clear Results": one drops this session's view, the other discards
        # stored evidence every future search would have reused. Adjacent
        # buttons with one-word differences are how that mistake happens.
        st.markdown("---")
        try:
            cached_count = get_vector_store().get_collection_stats().get("total_providers", 0)
        except Exception:
            cached_count = 0

        st.caption(f"Provider cache: {cached_count} stored")
        confirm_clear = st.checkbox(
            "Confirm cache clear",
            key="confirm_cache_clear",
            help="The next search will run cold and cost a few cents more."
        )
        if st.button("Clear provider cache", disabled=not confirm_clear):
            try:
                get_vector_store().clear_collection()
                st.session_state.confirm_cache_clear = False
                st.success(f"Cleared {cached_count} cached providers.")
            except Exception as e:
                st.error(f"Could not clear the cache: {e}")
            st.rerun()

        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("CareCompass v1 — multi-agent provider matching with LangGraph, Claude, and GPT.")
        st.markdown(f"[View the project on GitHub]({PORTFOLIO_GITHUB_URL})")
        st.caption("v2 — an agentic care-navigation companion (FastAPI + React) — is in active development.")

    # Main content
    search_params = render_search_form()

    # A submitted form IS the trigger: `render_search_form` returns None unless
    # the button was pressed, so a truthy `search_params` already means "the
    # user just clicked". Gating that on `search_params_changed` made an
    # IDENTICAL re-submit a silent no-op — the page simply repainted stale
    # results with no feedback. That blocked every workflow the sidebar
    # advertises, including the cold-vs-warm cache comparison the "Use cached
    # provider data" help text describes, since a repeat run is by definition
    # the same search.
    if search_params:
        st.session_state.search_executed = False
        st.session_state.last_search_params = copy.deepcopy(search_params)

    # Execute search if parameters provided
    if search_params and not st.session_state.search_executed:
        user = st.session_state.get("username", "anonymous")
        allowed, retry_after = rate_limiter.check(
            f"workflow:{user}",
            config.RATE_LIMIT_MAX_REQUESTS,
            config.RATE_LIMIT_WINDOW_SECONDS
        )
        if not allowed:
            # On the public demo every visitor shares one identity, so this
            # sliding window doubles as the global demo budget. Ask for the
            # contact at the moment the demo has already sold itself.
            st.warning(
                f"Today's demo budget is used up — searches reopen in "
                f"{_format_wait(retry_after)}. The README walks through the full "
                f"flow in the meantime, or [reach out on GitHub]({PORTFOLIO_GITHUB_URL}) "
                "for a live walkthrough."
            )
            log_audit_event(
                "rate_limit_exceeded",
                user=user,
                success=False,
                details={"retry_after": retry_after}
            )
            return

        log_audit_event(
            "workflow_started",
            user=user,
            details={
                "specialty": search_params.get("specialty"),
                "location": search_params.get("location"),
                "insurance": search_params.get("insurance")
            }
        )

        status = st.status("CareCompass agents are working...", expanded=True)
        progress_bar = status.progress(0)

        try:
            # Orchestrator is cached per (FHIR, fast-demo) configuration
            orchestrator = get_orchestrator(
                st.session_state.fhir_enabled, st.session_state.fast_demo
            )

            started_at = time.perf_counter()
            workflow_results = execute_with_live_progress(
                orchestrator, search_params, status, progress_bar
            )
            elapsed_s = time.perf_counter() - started_at

            st.session_state.workflow_results = workflow_results
            st.session_state.search_executed = True

            if workflow_results.get("success"):
                provider_count = len(workflow_results.get("final_recommendations", []))
                status.update(
                    label=f"Search complete in {elapsed_s:.1f}s",
                    state="complete",
                    expanded=False,
                )
                log_audit_event(
                    "workflow_completed",
                    user=user,
                    details={"providers": provider_count, "elapsed_s": round(elapsed_s, 1)}
                )

                # Durable record of the critic's audit of our judge. The
                # in-process WARNING goes to stderr with no FileHandler, so it
                # dies with the container; this survives for diffing across
                # searches. Written only when a finding exists — audit noise on
                # every clean run would bury the ones that matter. Kept here
                # rather than inside the agent: every log_audit_event call in
                # this codebase lives in app.py or utils/auth.py, and agents
                # stay free of file I/O.
                judge_findings = _judge_findings(
                    workflow_results.get("agent_outputs", {})
                    .get("critic_validator", {})
                    .get("validation_results", {})
                )
                if judge_findings:
                    log_audit_event(
                        "judge_evidence_inconsistency",
                        user=user,
                        success=False,   # a detected inconsistency, not a failed action
                        details={
                            "count": len(judge_findings),
                            "findings": [
                                {"provider": name, "finding": text}
                                for name, text in judge_findings
                            ],
                        },
                    )
            elif data_gatherer_status(workflow_results) == "no_results":
                status.update(label="No providers found", state="complete", expanded=False)
                log_audit_event(
                    "workflow_completed",
                    user=user,
                    details={"providers": 0, "elapsed_s": round(elapsed_s, 1)}
                )
            else:
                status.update(label="Search failed", state="error", expanded=False)
                log_audit_event("workflow_failed", user=user, success=False)

            # No st.rerun() here. `status.update()` only ENQUEUES its message,
            # and RerunException clears the unflushed queue on the next script
            # start — so the collapse was raced away and the panel stayed on
            # screen reading "CareCompass agents are working..." with the whole
            # step list expanded, forever. The tell was that the exception path
            # below, which has no rerun, collapsed correctly. Results render in
            # this same pass, so the rerun bought nothing.

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            log_audit_event("workflow_failed", user=user, success=False, details={"error": str(e)})
            status.update(label="Search failed", state="error", expanded=False)
            st.error("An error occurred while processing your search. Please try again later.")

            # Show details only in debug mode
            if config.DEBUG:
                with st.expander("Error Details (Debug Mode)"):
                    st.code(str(e))

    # Display results if available
    if st.session_state.workflow_results:
        workflow_results = st.session_state.workflow_results

        if workflow_results.get("success"):
            recommendations = workflow_results.get("final_recommendations", [])

            if recommendations:
                st.header("Provider Recommendations")

                # Network verification prototype (runs before the cards so
                # their chips reflect the directory answers). With the toggle
                # off, clear the active slot so cards can't show a verdict
                # from a previously selected payer.
                if st.session_state.fhir_enabled:
                    render_network_check(recommendations, workflow_results)
                else:
                    for recommendation in recommendations:
                        recommendation.get("provider", {}).pop("network_check", None)

                # At-a-glance superlatives are computed across the shortlist,
                # so the card loop needs them before rendering any single card.
                shortlist = [rec.get("provider", {}) for rec in recommendations]
                pool_highlights = _pool_highlights(shortlist)

                # Display provider cards
                for index, recommendation in enumerate(recommendations):
                    provider = recommendation.get("provider", {})
                    rank = recommendation.get("rank", 0)
                    render_provider_card(provider, rank, highlights=pool_highlights.get(index, []))

                # Trust layer: bias check, confidence, red flags, and the
                # critic-refinement note (rendered inside this panel)
                render_validation_insights(workflow_results)

                # Ranks 6+ that were scored but didn't make the shortlist
                render_other_providers(
                    workflow_results.get("workflow_summary", {}).get("other_providers", [])
                )

                # What this search cost (tokens, API credits, timing)
                render_cost_card(
                    workflow_results.get("cost_summary", {}),
                    workflow_results.get("agent_outputs", {})
                                    .get("data_gatherer", {})
                                    .get("search_metadata", {}),
                )

                # Agent internals (admin/debug)
                if st.session_state.show_agent_logs:
                    render_agent_workflow(workflow_results)

                    execution_log = workflow_results.get("execution_log", [])
                    if execution_log:
                        st.markdown("---")
                        render_execution_timeline(execution_log)

            else:
                st.warning("No provider recommendations found. Try adjusting your search criteria.")

        elif data_gatherer_status(workflow_results) == "no_results":
            # The search ran fine and found nobody — an outcome, not an error
            st.warning(
                "We couldn't find any providers matching your search. Try a broader "
                "location or a different specialty."
            )
            render_cost_card(
                workflow_results.get("cost_summary", {}),
                workflow_results.get("agent_outputs", {})
                                .get("data_gatherer", {})
                                .get("search_metadata", {}),
            )

        else:
            st.error("Search failed. Please check your input and try again.")
            error_messages = workflow_results.get("error_messages", [])
            if error_messages:
                with st.expander("Error Details"):
                    for error in error_messages:
                        st.error(error)

    # Always-visible trust footer
    render_html(
        '<div class="cc-footer">CareCompass is a portfolio demonstration, not medical advice. '
        "Provider information is AI-extracted from public web sources and may be inaccurate or "
        "out of date — verify details directly with providers and your insurer. "
        "No personal health information is collected or stored.</div>"
    )


if __name__ == "__main__":
    main()