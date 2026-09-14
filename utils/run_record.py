"""The per-run health record: one flat set of aggregate fields per run.

Why this exists. Tavily's August 2026 search overhaul degraded the pipeline
(the Chandler pool ~100+ → 38, enrichment 7-of-8 `no_profile_found`) and it
took about three weeks for anyone to notice — while every number that would
have caught it in a day (pool size, rows per platform, enrichment outcomes,
empty bodies, credits) was ALREADY computed on every run and discarded at
the end of it. The run record is that set of numbers kept, so a chart over
months shows a cliff the day it happens instead of a screenshot three weeks
later.

Where it lives. On the run's Langfuse trace: every scalar MEASUREMENT below
is attached to the root trace as a score (numeric / boolean), the
DESCRIPTORS (who ran it, with what config, on what input) ride as trace
metadata and tags, and the STRUCTURED fields (histograms, timings) ride as
metadata too, because a score is one value and a histogram is not. The
weekly report job then exports the rows it charts into the repository
beside the report — the durable copy that outlives the trace store's
retention window and any vendor decision.

There is deliberately NO database. A Postgres store (a Supabase
`run_metrics` table, DDL, psycopg2 writer) was scaffolded and retired the
same day, 2026-09-12: at a few hundred runs a month it added a secret in
three places, a pooler/SSL/IPv6 story, a store the development sandbox could
not reach at all (HTTPS-only egress), and a free tier that pauses after a
week idle — for a history the trace store already holds for ~90 days and the
exported rows hold indefinitely.

Contract:
  * AGGREGATES ONLY. No provider names, no review text, no page bodies, no
    URLs — per-run detail is the trace's spans. Inputs are allowlisted
    values (specialty, city, state), never free text. A guard test refuses
    any field whose name suggests identity: the exported rows ship to the
    PUBLIC repository.
  * The field list lives HERE, once. Score names on the trace, the export's
    columns and the report's tables all derive from this tuple, so a field
    added in one place cannot be missing from another.
  * `SCHEMA_VERSION` rides on every record; fields that arrive before a
    version bump go in `extra`.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

SCHEMA_VERSION = 1

# Who produced the run. user = a person on the deployed app; canary = a
# scheduled synthetic run; eval = an evaluation-harness run; smoke = the
# per-deploy check; verify = the account-verification script — a trace
# proving the project accepts writes, filtered out of every chart.
RUN_SOURCES: Tuple[str, ...] = ("user", "canary", "eval", "smoke", "verify")

RUN_RECORD_FIELDS: Tuple[str, ...] = (
    # identity
    "run_id", "ts", "source", "case_id", "git_sha", "schema_version",
    # config snapshot: what lets a step-change in a chart be explained
    "tavily_mode", "gatherer_model", "judge_model", "critic_model", "budget",
    "radius_miles",
    # input (allowlisted values only)
    "specialty", "city", "state", "zip_present",
    # discovery
    "fetch_mode", "fallback_fired", "pages_planned", "pages_fetched",
    "pages_failed", "empty_bodies", "rows_hg", "rows_wm", "rows_vi",
    "pool_raw", "pool_after_specialty", "pool_after_radius", "radius_dropped",
    "ring_fired", "ring_added", "credits_discovery",
    # enrichment
    "n_enriched", "n_no_profile_found", "n_failed", "n_cached",
    "n_over_budget", "n_identity_rejected", "pairs_hist", "profile_backed_hist",
    "coverage_hg", "coverage_wm", "coverage_vi", "all_three", "via_parser",
    "via_llm", "empty_body_rate", "credits_enrichment",
    # judge
    "judge_shards_ok", "judge_truncated", "judge_applied", "neutral_band_rate",
    "evidence_present_rate",
    # critic
    "critic_shards_failed", "validation_collapsed", "verdict_hist",
    "judge_findings", "conditional_rate", "identity_contradictions",
    # output
    "shortlist_size", "withheld_coverage", "withheld_ours", "withheld_budget",
    # economics
    "cost_usd", "tavily_credits", "tokens_in", "tokens_out", "latency_s",
    "stage_timings",
    # forward-compatible spillover
    "extra",
)

# Fields that identify or configure the run rather than measure it. They
# ride on the trace as metadata/tags, never as scores — a score named
# "judge_model" would be a string pretending to be a measurement.
DESCRIPTOR_FIELDS = frozenset(
    {
        "run_id", "ts", "source", "case_id", "git_sha", "schema_version",
        "tavily_mode", "gatherer_model", "judge_model", "critic_model",
        "budget", "radius_miles", "specialty", "city", "state", "zip_present",
        "fetch_mode",
    }
)

# Non-scalar fields: trace metadata, and nested objects in the exported rows.
STRUCTURED_FIELDS = frozenset(
    {
        "pairs_hist",
        "profile_backed_hist",
        "empty_body_rate",
        "verdict_hist",
        "stage_timings",
        "extra",
    }
)


def measurement_fields() -> Tuple[str, ...]:
    """The scalar measurements — the fields that become scores on the trace."""
    return tuple(
        f for f in RUN_RECORD_FIELDS
        if f not in DESCRIPTOR_FIELDS and f not in STRUCTURED_FIELDS
    )


# --------------------------------------------------------------------------
# Building the record from a finished workflow state
# --------------------------------------------------------------------------

def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def _hist(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value if value is not None else "none")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


_PLATFORM_KEYS = (("healthgrades", "hg"), ("webmd", "wm"), ("vitals", "vi"))


def _weights(preferences: Any) -> Optional[Dict[str, float]]:
    if not isinstance(preferences, dict):
        return None
    out = {}
    for key in ("rating_weight", "location_weight", "experience_weight"):
        value = preferences.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key.replace("_weight", "")] = round(float(value), 3)
    return out or None


def _shortlist_hashes(recommendations: Iterable[Any]) -> list:
    import hashlib

    from utils.provider_key import normalized_name

    out = []
    for entry in recommendations:
        name = entry.get("name") if isinstance(entry, dict) else None
        if name:
            out.append(hashlib.sha256(normalized_name(name).encode("utf-8")).hexdigest()[:12])
    return out


def _platform_of(text: Any) -> Optional[str]:
    lowered = str(text or "").lower()
    for needle, short in _PLATFORM_KEYS:
        if needle in lowered:
            return short
    return None


def build_run_record(
    state: Mapping[str, Any], config: Any = None, providers: Optional[List[Dict[str, Any]]] = None,
    pipeline: bool = False,
) -> Dict[str, Any]:
    """Every RUN_RECORD_FIELDS entry computed from the final workflow state.

    A field the state cannot answer is None — never a guessed zero, because
    "we did not measure" and "we measured nothing" chart differently, and
    the plan's alerts compare against baselines that must not be diluted
    by unmeasured runs. Aggregates only: nothing here names a provider.

    `providers` is the REFINED pool when the caller has it. The critic's
    verdicts (`critic_review`) live on the copies `refine_rankings` returns,
    never on `scored_providers.ranked_providers`; the first live Tier B
    canary (2026-09-13) recorded `verdict_hist` None and `judge_findings`
    None beside a critic log full of both, because the builder read the
    pre-refinement list. The finalize step passes the refined pool; a
    caller without it gets the enrichment and judge fields and None for
    the critic's, which is the honest shape.

    `pipeline` says whether the ORCHESTRATOR ran (discovery, enrichment,
    judge, critic) as opposed to a discovery-only measurement; it is a fact
    about the caller, so the caller states it. The weekly report reads it
    to keep fetch canaries and pipeline runs in different tables — the
    first report put the pipeline canary in neither, because the tier was
    stamped on the canary's local copy after the trace's copy was sealed.
    """
    from datetime import datetime, timezone

    from utils.client_context import public_subset
    from utils.geo import parse_location
    from utils.tracing import git_sha

    if config is None:
        from utils.config import get_config

        config = get_config()

    gathered = state.get("gathered_data") or {}
    meta = gathered.get("search_metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    scored = state.get("scored_providers") or {}
    ranked = providers if isinstance(providers, list) else None
    if ranked is None:
        ranked = scored.get("ranked_providers") if isinstance(scored, dict) else None
    if not isinstance(ranked, list):
        ranked = gathered.get("providers") or []
    summary = state.get("workflow_summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    cost = summary.get("cost_summary") or {}
    if not isinstance(cost, dict):
        cost = {}
    withheld = summary.get("withheld") or {}
    if not isinstance(withheld, dict):
        withheld = {}
    fetch = meta.get("fetch_stats") if isinstance(meta.get("fetch_stats"), dict) else {}
    fetch_stages = fetch.get("fetch") if isinstance(fetch.get("fetch"), dict) else {}
    discovery_fetch = fetch_stages.get("discovery") if isinstance(fetch_stages.get("discovery"), dict) else None
    enrichment_fetch = fetch_stages.get("enrichment") if isinstance(fetch_stages.get("enrichment"), dict) else None
    listing_rows = fetch.get("listing_rows") if isinstance(fetch.get("listing_rows"), dict) else {}

    preferences = state.get("preferences") or {}
    parts = parse_location(str(state.get("location") or ""))
    radius = preferences.get("search_radius_miles") if isinstance(preferences, dict) else None
    if radius is None:
        radius = meta.get("radius_miles", getattr(config, "DEFAULT_SEARCH_RADIUS", None))

    # ---- enrichment, over the providers that were inside the budget
    researched = [
        p for p in ranked
        if isinstance(p, dict) and p.get("enrichment_outcome") not in (None, "", "over_budget")
    ]
    outcomes = _hist(p.get("enrichment_outcome") for p in researched)
    # `via` rides inside each source's `yielded` block (the gatherer annotates
    # what a page produced and how); the first Tier B canary read 0 / 0 for
    # eight enriched providers because the builder looked one level up.
    via = _hist(
        (src.get("yielded") or {}).get("via") if isinstance(src.get("yielded"), dict) else src.get("via")
        for p in researched
        for src in (p.get("enrichment_sources") or []) if isinstance(src, dict)
    )
    coverage = {"hg": 0, "wm": 0, "vi": 0}
    all_three = 0
    for p in researched:
        urls = p.get("platform_profile_urls") or {}
        platforms = {_platform_of(k) or _platform_of(v) for k, v in urls.items()} if isinstance(urls, dict) else set()
        platforms.discard(None)
        for short in platforms:
            coverage[short] = coverage.get(short, 0) + 1
        if len(platforms) == 3:
            all_three += 1
    empty_body_rate = None
    if enrichment_fetch and isinstance(enrichment_fetch.get("by_domain"), dict):
        empty_body_rate = {}
        for domain, counts in enrichment_fetch["by_domain"].items():
            short = _platform_of(domain) or domain
            fetched = _safe_int(counts.get("fetched")) or 0
            empty = _safe_int(counts.get("empty")) or 0
            empty_body_rate[short] = _rate(empty, fetched + empty)

    # ---- judge, over the providers it was asked to score
    judged = [p for p in researched if p.get("ai_judged") is not False]
    applied = [p for p in judged if p.get("ai_rubric")]
    evidence_entries = [
        str(v or "") for p in applied
        for v in ((p.get("ai_evidence") or {}).values() if isinstance(p.get("ai_evidence"), dict) else [])
    ]
    neutral = sum(1 for e in evidence_entries if "no evidence" in e.lower())
    present = sum(1 for e in evidence_entries if e.strip() and "no evidence" not in e.lower())
    judge_shards = (scored.get("scoring_metadata") or {}).get("judge_shards") if isinstance(scored, dict) else None
    if not isinstance(judge_shards, dict):
        judge_shards = {}

    # ---- critic
    reviews = [p.get("critic_review") for p in researched if isinstance(p.get("critic_review"), dict)]
    verdicts = _hist((r.get("status") or "").lower() or "none" for r in reviews)
    conditional = sum(1 for r in reviews if "conditional" in str(r.get("status") or "").lower())
    validation = state.get("validation_results") or {}
    call_timings = (validation.get("validation_metadata") or {}).get("call_timings") if isinstance(validation, dict) else None
    shards_failed = (
        sum(1 for t in call_timings if isinstance(t, dict) and t.get("failed"))
        if isinstance(call_timings, list) else None
    )
    collapsed = None
    for entry in state.get("execution_log") or []:
        if isinstance(entry, dict) and entry.get("step") == "validate_rankings" and entry.get("status") == "completed":
            collapsed = bool((entry.get("details") or {}).get("validation_collapsed"))

    # ---- economics
    stage_timings = {
        entry.get("step"): (entry.get("details") or {}).get("elapsed_s")
        for entry in (state.get("execution_log") or [])
        if isinstance(entry, dict) and entry.get("status") == "completed"
        and (entry.get("details") or {}).get("elapsed_s") is not None
    }
    credits_by_stage = (cost.get("tavily") or {}).get("credits_by_stage") or {}

    record: Dict[str, Any] = {
        "run_id": state.get("run_id") or state.get("workflow_id"),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": state.get("run_source") or "user",
        "case_id": state.get("case_id"),
        "git_sha": git_sha(),
        "schema_version": SCHEMA_VERSION,
        "tavily_mode": getattr(config, "TAVILY_MODE", None),
        "gatherer_model": getattr(config, "GATHERER_MODEL", None),
        "judge_model": getattr(config, "JUDGE_MODEL", None),
        "critic_model": getattr(config, "CRITIC_MODEL", None),
        "budget": getattr(config, "MAX_PROVIDERS_TO_ENRICH", None),
        "radius_miles": radius,
        "specialty": state.get("specialty"),
        "city": parts.get("city"),
        "state": parts.get("state"),
        "zip_present": bool(parts.get("zip")),
        "fetch_mode": meta.get("fetch_mode"),
        "fallback_fired": bool(meta.get("fetch_mode_fallback")) if "fetch_mode_fallback" in meta else None,
        "pages_planned": _safe_int(discovery_fetch.get("planned")) if discovery_fetch else None,
        "pages_fetched": _safe_int(discovery_fetch.get("fetched")) if discovery_fetch else None,
        "pages_failed": _safe_int(discovery_fetch.get("failed")) if discovery_fetch else None,
        "empty_bodies": _safe_int(discovery_fetch.get("empty")) if discovery_fetch else None,
        "rows_hg": sum(v for k, v in listing_rows.items() if _platform_of(k) == "hg") if listing_rows else None,
        "rows_wm": sum(v for k, v in listing_rows.items() if _platform_of(k) == "wm") if listing_rows else None,
        "rows_vi": sum(v for k, v in listing_rows.items() if _platform_of(k) == "vi") if listing_rows else None,
        "pool_raw": None,
        "pool_after_specialty": None,
        "pool_after_radius": _safe_int(meta.get("total_found")),
        "radius_dropped": _safe_int(meta.get("radius_dropped")),
        "ring_fired": bool(meta.get("ring_expanded")) if "ring_expanded" in meta else None,
        "ring_added": _safe_int(meta.get("ring_added")),
        "credits_discovery": _safe_int(credits_by_stage.get("discovery")) if credits_by_stage else None,
        "n_enriched": outcomes.get("enriched", 0) if researched else None,
        "n_no_profile_found": outcomes.get("no_profile_found", 0) if researched else None,
        "n_failed": outcomes.get("failed", 0) if researched else None,
        "n_cached": outcomes.get("cached", 0) if researched else None,
        "n_over_budget": sum(1 for p in ranked if isinstance(p, dict) and p.get("enrichment_outcome") == "over_budget"),
        "n_identity_rejected": outcomes.get("identity_rejected", 0) if researched else None,
        "pairs_hist": _hist(p.get("platform_pair_count", 0) for p in researched) if researched else None,
        "profile_backed_hist": _hist(p.get("profile_backed_platforms", 0) for p in researched) if researched else None,
        "coverage_hg": coverage["hg"] if researched else None,
        "coverage_wm": coverage["wm"] if researched else None,
        "coverage_vi": coverage["vi"] if researched else None,
        "all_three": all_three if researched else None,
        "via_parser": via.get("profile_parser", 0) if researched else None,
        "via_llm": via.get("llm", 0) if researched else None,
        "empty_body_rate": empty_body_rate,
        "credits_enrichment": _safe_int(credits_by_stage.get("enrichment")) if credits_by_stage else None,
        "judge_shards_ok": _safe_int(judge_shards.get("ok")),
        "judge_truncated": bool(judge_shards.get("truncated")) if judge_shards else None,
        "judge_applied": len(applied) if judged else None,
        "neutral_band_rate": _rate(neutral, len(evidence_entries)),
        "evidence_present_rate": _rate(present, len(evidence_entries)),
        "critic_shards_failed": shards_failed,
        "validation_collapsed": collapsed,
        "verdict_hist": verdicts if reviews else None,
        "judge_findings": sum(1 for r in reviews if r.get("judge_findings")) if reviews else None,
        "conditional_rate": _rate(conditional, len(reviews)),
        "identity_contradictions": len(meta.get("identity_contradictions") or []) if isinstance(meta.get("identity_contradictions"), list) else None,
        "shortlist_size": len(state.get("final_recommendations") or []),
        "withheld_coverage": _safe_int(withheld.get("no_data")),
        "withheld_ours": _safe_int(withheld.get("pipeline_failures")),
        "withheld_budget": _safe_int(withheld.get("not_researched")),
        "cost_usd": cost.get("total_usd"),
        "tavily_credits": _safe_int((cost.get("tavily") or {}).get("credits")),
        "tokens_in": _safe_int((cost.get("llm") or {}).get("input_tokens")),
        "tokens_out": _safe_int((cost.get("llm") or {}).get("output_tokens")),
        "latency_s": cost.get("elapsed_s"),
        "stage_timings": stage_timings or None,
        "extra": {
            "pipeline": bool(pipeline),
            # The search criteria beyond the allowlisted inputs: the three
            # weights as the user set them (normalized), so a report row says
            # what was asked, not only where.
            "weights": _weights(preferences),
            # Coarse visitor facts only — country / region / timezone /
            # locale. The salted visitor id stays on the trace's own
            # metadata; the rows are public.
            "client": public_subset(state.get("client")),
            # The shortlist as ORDERED HASHES of normalized names (12 hex each):
            # enough for the weekly report to say "4 of 5 shared with last
            # week, 3 in the same slot" — the rolling stability signal — and
            # nothing a reader could turn back into a doctor.
            "shortlist_hashes": _shortlist_hashes(state.get("final_recommendations") or []),
            "workflow_failed": bool(summary.get("workflow_failed")),
            "errors": len(state.get("error_messages") or []),
            "specialty_rows_dropped": _safe_int(fetch.get("specialty_rows_dropped")),
            "data_status": gathered.get("status"),
        },
    }
    # The pool BEFORE the radius bound is what discovery produced; the
    # specialty bound runs on parsed listing ROWS before dedupe (its drops are
    # in `extra.specialty_rows_dropped`), so at provider level the pool after
    # it is the same number — recorded as such rather than left null.
    if record["pool_after_radius"] is not None:
        record["pool_raw"] = record["pool_after_radius"] + (record["radius_dropped"] or 0)
        record["pool_after_specialty"] = record["pool_raw"]
    missing = [f for f in RUN_RECORD_FIELDS if f not in record]
    if missing:  # a field added to the tuple and not here must fail loudly in tests
        raise KeyError(f"build_run_record does not compute: {missing}")
    return record
