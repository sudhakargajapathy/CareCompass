"""Alert thresholds: measured baselines, one severity each, one first step each.

Every threshold here has a date and a number behind it — the September 2026
outage, replayed: the pool fell 120 → 38 (pool floor 60 would have fired on
day one), 7 of 8 researched providers came back `no_profile_found` (floor 3),
webmd vanished from every card (platform coverage zero), and none of it was
seen for three weeks. The severity vocabulary is exactly three words:

  P1  vendor health — something upstream is broken or unpaid; nothing else
      can be trusted until it is fixed (auth/credit errors, a canary that
      fetched nothing, a pipeline run with no shortlist)
  P2  drift — the pipeline runs but its yield moved past a floor
  P3  cost & freshness — money, time, or a monitor that stopped reporting

An alert nobody can act on is noise nobody reads, so every finding names
its FIRST diagnostic step; those steps are the runbook, inline.

Two rules the checks follow:

  * A finding's IDENTITY is (check, case, detail) — one GitHub issue per
    identity, so a check that fires per platform (`platform_rows_zero`)
    carries the platform in `detail` and a webmd outage and a vitals outage
    are two issues with two lifetimes. Everything else leaves `detail`
    empty and is one issue per case.
  * A run only VOUCHES for the checks it evaluated. `checks_for_tier` is
    the recovery scope: a Tier A (fetch-only) run never closes a Tier B
    issue such as `no_shortlist`, because it did not look — silence is not
    a recovery.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence

from evals.cases import PLATFORM_LABELS, CanaryCase

SEVERITIES = ("P1", "P2", "P3")

# Bands measured 2026-08 → 2026-09 (cold $0.5592 / warm $0.3497; 50-70 s).
COST_BAND_USD = (0.25, 0.90)
LATENCY_CEILING_S = 150.0
NO_PROFILE_FLOOR = 3          # of the 8-provider research budget
EMPTY_BODY_RATE_CEILING = 0.30
# The weekly report is written on Mondays; eight days covers one missed
# Monday plus slack for a late run — GitHub disables idle crons silently,
# and this is the watchdog for the watcher.
REPORT_MAX_AGE_DAYS = 8

# Gather statuses that mean the measurement never happened. "no_results" and
# "no_providers_extracted" are NOT here: those are measurements (a pool of
# zero), and the pool floor reports them.
FAILED_GATHER_STATUSES = ("error", "invalid_input", "crashed")

FIRST_STEPS = {
    "vendor_error": (
        "Open the failing generation/tool span's error text on the trace; check the vendor's "
        "credits/billing page. TAVILY_MODE=search is the rollback lever for a FETCH-side "
        "failure only."
    ),
    "gather_failed": (
        "The gatherer did not complete (an exception escaped discovery, or the input failed "
        "validation): the canary output carries the redacted error and the trace's "
        "step.gather_data span its status. A live search would be failing the same way."
    ),
    "canary_did_not_run": (
        "Open the workflow run: a missing secret or a crashed script. The canary must produce "
        "its JSON before anything else can be judged."
    ),
    "pipeline_failed": (
        "The workflow did not complete (an exception escaped a node): the canary output carries "
        "the redacted error and the trace root its status. Every live search is failing the "
        "same way."
    ),
    "report_stale": (
        "The weekly report has not been written: GitHub Actions → is the weekly-report "
        "schedule disabled (crons auto-disable after 60 idle days)? Then the job's last log — "
        "a Langfuse auth or rate-limit failure reads differently from a git push rejection."
    ),
    "golden_identity_failure": (
        "The URL now names someone else (or its slug contradicts the key): open the page — "
        "a retired doctor's URL reassigned, or a platform merge. Retire the entry "
        "(evals/refresh_fixtures.py) rather than refresh it; the app's identity guards are the "
        "same check, so verify a card for that doctor."
    ),
    "template_drift": (
        "The production parser reads no rating or count from today's copy of a page it read "
        "last week: diff the private body artifact against the saved copy, fix "
        "utils/profile_parser off the FETCHED bytes, add today's copy as a new fixture beside "
        "the old one (both templates must keep parsing)."
    ),
    "golden_page_missing": (
        "A key page came back empty or absent: fetch it by hand. Two weeks running → the "
        "doctor moved or the URL died; propose a replacement from the same platform."
    ),
    "golden_needs_review": (
        "A number moved past its tolerance (count down, rating ≥ 0.3, tenure changed): read "
        "both copies; if the world changed, refresh the entry; if the parser read a "
        "neighbour, that is a parser bug with a stranger's stars behind it."
    ),
    "golden_pipeline_mismatch": (
        "The live pipeline's read of a golden-set page disagrees with the key (or the page was "
        "attached to a differently named provider): open the Tier B trace's enrichment.provider "
        "span for that doctor beside the page's saved copy. A rating that moved is a refresh; a "
        "different name is the identity guard's bug — a stranger's page on someone's card."
    ),
    "no_pages_fetched": (
        "Open the trace's tavily.extract spans: every batch empty or failed means the vendor "
        "or the constructed URLs, not the parsers. Try one listing URL by hand."
    ),
    "pool_size_low": (
        "Open the trace's discovery step: pages fetched vs empty bodies vs listing rows per "
        "platform tell which layer shrank."
    ),
    "platform_rows_zero": (
        "That platform's listing page parsed to zero rows: fetch it by hand and read the "
        "markup against utils/listing_parser — a redesign, or a slug that now 404s."
    ),
    "fetch_mode_fallback": (
        "The constructed listing URLs yielded nothing and discovery fell back to search: "
        "check the specialty/city slug tables in utils/platform_urls."
    ),
    "empty_body_rate_high": (
        "The vendor is returning pages with empty bodies again (the August class): compare "
        "the trace's per-page digests with a direct /extract of the same URL."
    ),
    "no_shortlist": (
        "Zero recommendations: read the run's withheld reasons on the trace output — a "
        "collapsed validation (vendor) reads differently from a coverage gap."
    ),
    "validation_collapsed": (
        "Every critic shard failed: the shard spans carry the error text (credits, a rejected "
        "parameter, a model id)."
    ),
    "no_profile_found_high": (
        "Researched providers without a profile page: check the enrichment.provider spans' "
        "known_profile_urls and the tavily.extract yields under them."
    ),
    "platform_coverage_zero": (
        "No researched provider carries a URL on that platform: listing rows or the dedupe "
        "URL union dropped it (the 2026-09-02 webmd case)."
    ),
    "judge_truncated": (
        "A judge shard hit its token ceiling: raise _judge_token_budget or check the pool size "
        "that call was handed."
    ),
    "critic_shard_failed": (
        "A deep-validation shard returned its error fallback: its span carries the reason."
    ),
    "cost_out_of_band": (
        "Compare the trace's generation costs by agent with the baseline split; a runaway "
        "loop shows as call COUNT, a vendor price change as cost per call."
    ),
    "latency_high": (
        "The trace's step spans say which stage grew; compare call_timings on validation."
    ),
}

CHECK_TITLES = {
    "vendor_error": "Vendor error",
    "gather_failed": "Gather failed",
    "canary_did_not_run": "Canary did not run",
    "pipeline_failed": "Pipeline failed",
    "report_stale": "Weekly report stale",
    "golden_identity_failure": "Golden identity failure",
    "template_drift": "Template drift",
    "golden_page_missing": "Golden page missing",
    "golden_needs_review": "Golden set needs review",
    "golden_pipeline_mismatch": "Pipeline read disagrees with the golden key",
    "no_pages_fetched": "No pages fetched",
    "pool_size_low": "Pool size low",
    "platform_rows_zero": "Platform listing rows zero",
    "fetch_mode_fallback": "Fetch-mode fallback fired",
    "empty_body_rate_high": "Empty-body rate high",
    "no_shortlist": "No shortlist",
    "validation_collapsed": "Validation collapsed",
    "no_profile_found_high": "no_profile_found high",
    "platform_coverage_zero": "Platform coverage zero",
    "judge_truncated": "Judge truncated",
    "critic_shard_failed": "Critic shard failed",
    "cost_out_of_band": "Cost out of band",
    "latency_high": "Latency high",
}

# Recovery scopes. The watcher's own check rides in every tier: a case that
# produced a result this run has, by definition, run.
FETCH_CHECKS: FrozenSet[str] = frozenset({
    "vendor_error", "gather_failed", "no_pages_fetched", "pool_size_low",
    "platform_rows_zero", "fetch_mode_fallback", "empty_body_rate_high",
})
PIPELINE_CHECKS: FrozenSet[str] = FETCH_CHECKS | frozenset({
    "pipeline_failed", "no_shortlist", "validation_collapsed", "no_profile_found_high",
    "platform_coverage_zero", "judge_truncated", "critic_shard_failed",
    "cost_out_of_band", "latency_high", "golden_pipeline_mismatch",
})
WATCHER_CHECKS: FrozenSet[str] = frozenset({"canary_did_not_run"})
# Not per case: the weekly report is one artifact, evaluated under its own
# pseudo-case id so it gets one issue with one lifetime.
REPORT_CHECKS: FrozenSet[str] = frozenset({"report_stale"})
REPORT_CASE_ID = "weekly-report"
# The golden set is graded per page; its findings carry provider/platform in
# `detail` so each page's problem has its own issue, under one pseudo-case.
GOLDEN_CHECKS: FrozenSet[str] = frozenset({
    "golden_identity_failure", "template_drift", "golden_page_missing", "golden_needs_review",
})
GOLDEN_CASE_ID = "golden-set"
ALL_CHECKS: FrozenSet[str] = PIPELINE_CHECKS | WATCHER_CHECKS | REPORT_CHECKS | GOLDEN_CHECKS


def checks_for_tier(tier: str) -> FrozenSet[str]:
    """The checks a tier's run evaluates — and therefore may declare recovered."""
    if tier == "A":
        return FETCH_CHECKS | WATCHER_CHECKS
    if tier == "B":
        return PIPELINE_CHECKS | WATCHER_CHECKS
    raise ValueError(f"unknown tier {tier!r}: A | B")


@dataclass(frozen=True)
class Finding:
    check: str
    severity: str
    case_id: str
    observed: str
    threshold: str
    detail: str = ""

    @property
    def title(self) -> str:
        base = f"[{self.severity}] {CHECK_TITLES.get(self.check, self.check)} — {self.case_id}"
        return f"{base} ({self.detail})" if self.detail else base

    @property
    def identity(self):
        return (self.check, self.case_id, self.detail)

    @property
    def first_step(self) -> str:
        return FIRST_STEPS.get(self.check, "See the run summary.")

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["title"] = self.title
        data["first_step"] = self.first_step
        return data


def finding_from_dict(data: Mapping[str, Any]) -> Finding:
    return Finding(
        check=str(data["check"]), severity=str(data["severity"]), case_id=str(data["case_id"]),
        observed=str(data.get("observed", "")), threshold=str(data.get("threshold", "")),
        detail=str(data.get("detail", "")),
    )


def _platform_rows(record: Mapping[str, Any], short: str) -> Optional[int]:
    return record.get(f"rows_{short}")


def _vendor_finding(
    cid: str, vendor_errors: Sequence[Mapping[str, Any]], fetched: Optional[int]
) -> Optional[Finding]:
    """One vendor_error per case: every fatal error, or transient ones only
    when nothing came back at all.

    A timeout the gatherer's own retry recovered from is not vendor health —
    it is noise the summary's notes column carries — but ten timeouts and an
    empty run is the vendor down, whatever the exception class said.
    """
    fatal = [e for e in vendor_errors if e.get("kind") == "fatal"]
    transient = [e for e in vendor_errors if e.get("kind") != "fatal"]

    def describe(errors: Sequence[Mapping[str, Any]]) -> str:
        seen, parts = set(), []
        for e in errors:
            key = (e.get("vendor"), e.get("error"))
            if key in seen:
                continue
            seen.add(key)
            message = str(e.get("message") or "").strip()
            parts.append(f"{e.get('vendor')}: {e.get('error')}" + (f" — {message[:120]}" if message else ""))
        return "; ".join(parts)

    if fatal:
        return Finding("vendor_error", "P1", cid, describe(fatal), "no vendor errors")
    if transient and not fetched:
        return Finding(
            "vendor_error", "P1", cid,
            f"{len(transient)} transient error(s) and nothing fetched: {describe(transient)}",
            "no vendor errors",
        )
    return None


def evaluate_fetch(
    record: Mapping[str, Any],
    case: CanaryCase,
    fetch_stats: Optional[Mapping[str, Any]] = None,
    vendor_errors: Optional[Sequence[Mapping[str, Any]]] = None,
    gather_status: Optional[str] = None,
) -> List[Finding]:
    """Tier A (discovery-only) findings for one canary case.

    Ordered by what explains what: a vendor error explains a failed gather,
    a failed gather explains an empty pool, an empty fetch explains every
    zero-row platform — each of those returns early so one outage is one
    finding (one issue, one email) rather than a cascade of consequences.
    """
    findings: List[Finding] = []
    cid = case.case_id
    planned, fetched = record.get("pages_planned"), record.get("pages_fetched")
    vendor = _vendor_finding(cid, vendor_errors or [], fetched)
    if vendor:
        findings.append(vendor)
    if gather_status in FAILED_GATHER_STATUSES:
        if not vendor:
            findings.append(Finding(
                "gather_failed", "P1", cid, f"status={gather_status}", "status=success",
            ))
        return findings
    if planned and not fetched:
        if not vendor:
            findings.append(Finding("no_pages_fetched", "P1", cid, f"0 of {planned} pages", "≥ 1 page"))
        return findings  # everything below would only restate this
    pool = record.get("pool_raw")
    if pool is not None and pool < case.pool_floor:
        findings.append(Finding("pool_size_low", "P2", cid, str(pool), f"≥ {case.pool_floor}"))
    if fetched:
        for short in case.platforms_expected:
            rows = _platform_rows(record, short) or 0
            if rows == 0:
                findings.append(Finding(
                    "platform_rows_zero", "P2", cid, f"{PLATFORM_LABELS[short]}: 0 rows", "≥ 1 row",
                    detail=PLATFORM_LABELS[short],
                ))
    if record.get("fallback_fired"):
        findings.append(Finding("fetch_mode_fallback", "P2", cid, "fired", "never"))
    discovery = ((fetch_stats or {}).get("fetch") or {}).get("discovery") or {}
    for domain, counts in (discovery.get("by_domain") or {}).items():
        got, empty = int(counts.get("fetched") or 0), int(counts.get("empty") or 0)
        total = got + empty
        if total >= 2 and empty / total > EMPTY_BODY_RATE_CEILING:
            findings.append(Finding(
                "empty_body_rate_high", "P2", cid, f"{domain}: {empty}/{total} empty",
                f"≤ {int(EMPTY_BODY_RATE_CEILING * 100)}%", detail=domain,
            ))
    return findings


def evaluate_pipeline(
    record: Mapping[str, Any],
    case: CanaryCase,
    vendor_errors: Optional[Sequence[Mapping[str, Any]]] = None,
    gather_status: Optional[str] = None,
    fetch_stats: Optional[Mapping[str, Any]] = None,
    cross_check: Optional[Mapping[str, Any]] = None,
) -> List[Finding]:
    """Tier B (full pipeline) findings — the fetch checks plus the model-side ones.

    A failed STATUS here means the workflow did not complete, which is a
    different first step from a failed gather (the whole node graph, not
    discovery), so it gets its own check; the fetch checks then run on
    whatever the run did measure.
    """
    cid = case.case_id
    if gather_status in FAILED_GATHER_STATUSES:
        vendor = _vendor_finding(cid, vendor_errors or [], record.get("pages_fetched"))
        if vendor:
            return [vendor]
        return [Finding("pipeline_failed", "P1", cid, f"status={gather_status}", "status=success")]
    findings = evaluate_fetch(record, case, fetch_stats, vendor_errors, None)
    if any(f.check in ("no_pages_fetched", "vendor_error") for f in findings):
        return findings
    shortlist = record.get("shortlist_size")
    if shortlist == 0:
        findings.append(Finding("no_shortlist", "P1", cid, "0 recommendations", "≥ 1"))
    if record.get("validation_collapsed"):
        findings.append(Finding("validation_collapsed", "P1", cid, "every critic shard failed", "none"))
    npf = record.get("n_no_profile_found")
    if npf is not None and npf >= NO_PROFILE_FLOOR:
        findings.append(Finding("no_profile_found_high", "P2", cid, f"{npf} of the budget", f"< {NO_PROFILE_FLOOR}"))
    if record.get("n_enriched") is not None:
        for short in case.platforms_expected:
            if (record.get(f"coverage_{short}") or 0) == 0:
                findings.append(Finding(
                    "platform_coverage_zero", "P2", cid, f"{PLATFORM_LABELS[short]}: 0 providers",
                    "≥ 1 researched provider", detail=PLATFORM_LABELS[short],
                ))
    if record.get("judge_truncated"):
        findings.append(Finding("judge_truncated", "P2", cid, "a judge shard hit max tokens", "never"))
    if (record.get("critic_shards_failed") or 0) > 0:
        findings.append(Finding("critic_shard_failed", "P2", cid, f"{record['critic_shards_failed']} shard(s)", "0"))
    cost = record.get("cost_usd")
    if cost is not None and not (COST_BAND_USD[0] <= float(cost) <= COST_BAND_USD[1]):
        findings.append(Finding("cost_out_of_band", "P3", cid, f"${float(cost):.2f}", f"${COST_BAND_USD[0]:.2f}–${COST_BAND_USD[1]:.2f}"))
    latency = record.get("latency_s")
    if latency is not None and float(latency) > LATENCY_CEILING_S:
        findings.append(Finding("latency_high", "P3", cid, f"{float(latency):.0f} s", f"≤ {LATENCY_CEILING_S:.0f} s"))
    for page in (cross_check or {}).get("pages") or []:
        if not isinstance(page, dict) or page.get("status") not in ("disagree", "identity"):
            continue
        notes = "; ".join(str(n) for n in (page.get("notes") or [])) or str(page.get("status"))
        findings.append(Finding(
            "golden_pipeline_mismatch", "P2", cid, f"{page.get('status')}: {notes}"[:200],
            "rating within 0.3 of a saved copy, same person", detail=f"{page.get('provider_id')}/{page.get('platform')}",
        ))
    return findings


def evaluate_report_freshness(latest: Optional[Mapping[str, Any]], now: Optional[datetime] = None) -> List[Finding]:
    """P3 when the last weekly report is older than REPORT_MAX_AGE_DAYS.

    `latest` is the report job's own stamp (`reports/latest.json`); with
    no stamp there has never been a report, and "never" is the weekly job
    being red on its first Monday, not a staleness — nothing fires.
    """
    if not latest or not latest.get("generated_at"):
        return []
    now = now or datetime.now(timezone.utc)
    try:
        generated = datetime.fromisoformat(str(latest["generated_at"]).replace("Z", "+00:00"))
    except ValueError:
        return [Finding("report_stale", "P3", REPORT_CASE_ID, "unreadable generated_at", f"≤ {REPORT_MAX_AGE_DAYS} days")]
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    age = now - generated
    if age > timedelta(days=REPORT_MAX_AGE_DAYS):
        return [Finding(
            "report_stale", "P3", REPORT_CASE_ID, f"{age.days} days old (week {latest.get('week', '?')})",
            f"≤ {REPORT_MAX_AGE_DAYS} days",
        )]
    return []


_GOLDEN_STATUS_CHECK = {
    "identity": ("golden_identity_failure", "P2"),
    "template_drift": ("template_drift", "P2"),
    "missing": ("golden_page_missing", "P3"),
    "needs_review": ("golden_needs_review", "P3"),
}


def evaluate_golden(payload: Optional[Mapping[str, Any]]) -> List[Finding]:
    """One finding per graded page that is not ok/world_drift.

    World drift is not a finding: the key's floor moved the way the world
    moves (a count went up, a rating nudged), and the summary lists it as
    a refresh proposal. Everything else is a person's decision or a
    parser's blindness, and gets an issue whose detail names the page.
    """
    findings: List[Finding] = []
    for page in (payload or {}).get("pages") or []:
        if not isinstance(page, dict):
            continue
        mapped = _GOLDEN_STATUS_CHECK.get(str(page.get("status")))
        if not mapped:
            continue
        check, severity = mapped
        detail = f"{page.get('provider_id')}/{page.get('platform')}"
        notes = "; ".join(str(n) for n in (page.get("notes") or [])) or page.get("status")
        expected = page.get("expected") or {}
        threshold = (
            f"rating {expected.get('rating')} / count {expected.get('review_count')}"
            if check in ("golden_needs_review", "template_drift") else "matches the key"
        )
        findings.append(Finding(check, severity, GOLDEN_CASE_ID, str(notes)[:200], threshold, detail=detail))
    return findings


def worst_severity(findings: Sequence[Finding]) -> Optional[str]:
    for severity in SEVERITIES:
        if any(f.severity == severity for f in findings):
            return severity
    return None
