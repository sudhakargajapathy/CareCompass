"""Critic Validator Agent: challenges and validates provider rankings.

Runs on CRITIC_MODEL (Claude Opus 4.8 by default) — deliberately a different
model family from the OpenAI judge it audits, so the validator does not share
the scorer's blind spots.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
import json
import re
from anthropic import Anthropic

# The judge's own rubric, imported rather than restated. The critic audits the
# judge's per-criterion scores, and an auditor working from a paraphrase of the
# standard is not an independent check — see JUDGE_RUBRIC's own note. Direction
# is deliberate and acyclic: preference_scorer imports nothing from here.
from agents.preference_scorer import JUDGE_RUBRIC
from utils.config import get_config
from utils.cost_tracker import get_cost_tracker, safe_usage
from utils.excerpt import SUMMARY_MAX_CHARS, clip_words
from utils.provenance import source_domain
from utils.provider_key import normalize_name_tokens, normalized_name
from utils.shard import round_robin_shards
from utils.security import InputValidator

logger = logging.getLogger(__name__)

# Refinement adjustments (score points on the 0-100 match scale), sized so a
# single caution or red flag can reorder adjacent providers but not catapult
# a provider across the whole list.
_STATUS_PENALTY_REJECTED = 15.0
_STATUS_PENALTY_NOT_APPROVED = 8.0
# Output-token allowance for one deep-validation call, scaled to the pool.
#
# It was a flat 6500 with a comment reading "8 entries with capped notes fit
# comfortably" — true for the pool of the day, and exactly the assumption
# DESIGN §10.17 records as the way a budget fails: the pool is now a knob
# (MAX_PROVIDERS_TO_ENRICH, env-tunable upward) and nothing re-derived this.
#
# Each entry carries a verdict, confidence, validation_notes, red_flags,
# patient_considerations and recommendation_adjustments — heavier than a judge
# entry, hence the larger per-provider figure. The floor keeps small pools from
# being starved by a cap that scales below the fixed JSON envelope.
_VALIDATION_BASE_TOKENS = 2500
_VALIDATION_TOKENS_PER_PROVIDER = 500
_VALIDATION_MAX_TOKENS = 16000


def _validation_token_budget(provider_count: int) -> int:
    """Scaled ceiling for the deep-validation response."""
    return max(
        4000,
        min(
            _VALIDATION_BASE_TOKENS + _VALIDATION_TOKENS_PER_PROVIDER * max(provider_count, 0),
            _VALIDATION_MAX_TOKENS,
        ),
    )


# How many concurrent calls run the deep validation, and the pool below which
# it stays one call. Unlike the judge this needs no config knob: the verdict
# rubric scores each provider against fixed entry criteria, never against the
# other providers in the call, so a split cannot move a verdict the way a
# differently-calibrated judge shard could.
#
# The floor is set by the DIFFERENTIATION CHECK, which is the one part of this
# prompt that IS about the group: below 4 providers a shard holds one, and
# "find the real differences between these providers" has no meaning for a
# call holding a single record.
_VALIDATION_SHARDS = 2
_MIN_PROVIDERS_TO_SPLIT_VALIDATION = 4

# Least-to-most confident. Used to merge shards conservatively — see below.
_CONFIDENCE_ORDER = ("low", "medium", "high")


def _merge_validation_shards(shards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine per-shard validation responses into one result.

    `top_provider_validations` simply concatenates — the entries are
    per-provider and carry global ranks, so order is irrelevant to every
    consumer (`refine_rankings` binds by name tokens; the final-recommendation
    builder looks up `rank`).

    `overall_ranking_validity` is the hard part, and the honest answer is that
    NO shard saw the whole ranking it describes. Its `confidence` reaches the
    patient — a "low" adds a caution line to user_guidance — so it merges to
    the LEAST confident shard rather than an average: a ranking is only as
    validated as its weakest half. Averaging would let a confident half mask a
    half the critic could not vouch for.
    """
    validations: List[Dict[str, Any]] = []
    confidences: List[str] = []
    statuses: List[str] = []
    summaries: List[str] = []
    suggestions: List[str] = []

    for shard in shards:
        if not isinstance(shard, dict):
            continue
        entries = shard.get("top_provider_validations")
        if isinstance(entries, list):
            validations.extend(entries)
        validity = shard.get("overall_ranking_validity")
        if not isinstance(validity, dict):
            continue
        # A shard that returned nothing has no opinion on the ranking, and
        # counting its fallback "low" would let one failed call drag the whole
        # verdict down while its providers are separately marked not_critiqued.
        if not entries:
            continue
        if validity.get("confidence") in _CONFIDENCE_ORDER:
            confidences.append(validity["confidence"])
        if validity.get("status"):
            statuses.append(str(validity["status"]))
        if validity.get("summary"):
            summaries.append(str(validity["summary"]))
        for item in validity.get("improvement_suggestions") or []:
            if item not in suggestions:
                suggestions.append(item)

    if not validations:
        return {
            "top_provider_validations": [],
            "overall_ranking_validity": {
                "status": "error",
                "confidence": "low",
                "summary": "Validation could not be completed",
                "improvement_suggestions": [],
            },
        }

    usable_statuses = [s for s in statuses if s != "error"]
    return {
        "top_provider_validations": validations,
        "overall_ranking_validity": {
            "status": usable_statuses[0] if usable_statuses else "error",
            "confidence": (
                min(confidences, key=_CONFIDENCE_ORDER.index) if confidences else "low"
            ),
            "summary": " ".join(summaries) if summaries else "",
            "improvement_suggestions": suggestions,
        },
    }


_RED_FLAG_PENALTY = 4.0
_RED_FLAG_PENALTY_CAP = 12.0

# Name-token overlap required before a critic verdict binds to a provider.
# This replaced the positional rank fallback in `refine_rankings` — see the
# note there.
#
# Matches the gatherer's dedup threshold rather than the judge's 0.5, because
# the roles differ: the judge's 0.5 is a CROSS-CHECK on an entry already bound
# by index, where a loose pass costs nothing. Here the overlap IS the binding,
# and it carries penalties up to -15. At 0.5 a shared surname alone qualifies —
# {david, kim} against {jane, kim} is exactly 0.5 — so one Kim would collect
# the other's rejection. Realistic name pairs only ever score 0.5, 0.67 or
# 1.0, so this sits in the gap: a dropped middle name still binds
# ({hemant, pandey} vs {hemant, kumar, pandey} = 1.0) while two different
# people sharing a surname do not.
_VERDICT_NAME_THRESHOLD = 0.8

# `recommendation_adjustments` is specified as "leave it empty when the judge's
# scoring matches the evidence". The 2026-07-25 live run shows the critic
# ignoring that: 9 of 10 entries were PASS verdicts ("scoring matches
# evidence", one ending verbatim "No correction needed."), and the panel
# reported "10 inconsistencies were found" when the true count of judge errors
# was zero.
#
# So this predicate is the PRIMARY guard, not a backstop — the prompt already
# carried the instruction and was disobeyed.
_JUDGE_PASS_RE = re.compile(
    r"match(?:es|ed)?\s+(?:the\s+)?evidence"
    r"|scoring\s+(?:is|was)\s+consistent"
    r"|is\s+consistent\s+with"
    r"|correctly\s+scored"
    r"|no\s+correction\s+needed"
    r"|no\s+(?:judge\s+)?(?:inconsistenc|discrepanc|issue|error)"
    r"|fairly\s+(?:reflects|credits|represents)"
    r"|appropriately\s+(?:reflects|scored|credits)"
    r"|(?:score|scoring)\s+is\s+accept",
    re.IGNORECASE,
)

# An UNAMBIGUOUS verdict, checked before anything else. One live-run entry read
# "Judge parked practical_access at 10 ... the neutral score is acceptable ...
# No correction needed." — descriptive problem vocabulary wrapped around an
# explicit all-clear. A stated verdict outranks inferred tone.
_JUDGE_STRONG_PASS_RE = re.compile(
    r"no\s+correction\s+needed"
    r"|no\s+(?:judge\s+)?(?:inconsistenc|discrepanc|error|correction)s?\s+(?:found|needed|required)"
    r"|scoring\s+match(?:es|ed)?\s+(?:the\s+)?evidence",
    re.IGNORECASE,
)

# Otherwise, anything asserting a PROBLEM wins over a pass phrase, because a
# real finding can quote the very language a confirmation uses:
# "practical_access does not match the evidence" contains "match the evidence"
# and must not be filtered.
#
# "parked" is deliberately NOT here: the critic uses it descriptively ("Judge
# parked practical_access at 10") in verdicts that go either way. A genuine
# neutral-band finding is caught by "while/though the summary" instead, which
# is what actually makes it a finding.
# A model asked for "" that writes a word meaning "nothing to report". These
# match the WHOLE entry, so a null answer can never be mistaken for content.
_JUDGE_NULL_ANSWER_RE = re.compile(
    r"[\s.\-–—]*"
    r"(?:none|n/?a|nil|null|nothing|no\s+(?:adjustments?|changes?|corrections?|"
    r"concerns?|issues?|findings?|comments?|notes?)"
    r"(?:\s+(?:needed|required|noted|found|necessary))?)"
    r"[\s.\-–—]*",
    re.IGNORECASE,
)

_JUDGE_CONCERN_RE = re.compile(
    r"\b(?:does|did|do)\s+not\b|\bdoesn'?t\b|\bdidn'?t\b|\bisn'?t\b|\bwasn'?t\b"
    r"|\bno\s+basis\b|\bunsupported\b|\bfabricat|\bmissed\b|\bmisses\b"
    r"|\bshould\s+have\b|\bthough\s+the\s+summary\b|\bdespite\b|\bwhile\s+the\s+summary\b"
    r"|\bnot\s+supported\b|\binconsistent\s+with\b|\bmismatch|\bcontradict"
    r"|\boverstat|\bunderstat|\bfails?\s+to\b",
    re.IGNORECASE,
)

# Contrastive pivots. Whatever precedes one, the clause AFTER it is the
# finding — so this is the ONE thing that outranks an explicit all-clear.
# The `[\w-]` lookarounds are load-bearing: a plain `\bbut\b` matched inside
# "mixed-but-mostly-positive" and turned a real live-run PASS verdict into a
# reported inconsistency.
_JUDGE_MIXED_RE = re.compile(
    r"(?<![\w-])(?:but|however|whereas|although)(?![\w-])"
    r"|(?<![\w-])(?:ignor(?:es|ed|ing)|overlook(?:s|ed|ing)?)(?![\w-])",
    re.IGNORECASE,
)


def _judge_finding_or_empty(raw: Any, provider_name: str = "") -> str:
    """The entry if it asserts a problem, "" if it is a pass verdict.

    Filtering is logged rather than silent: if a prompt change starts producing
    concerns in unrecognized phrasing, or the critic floods the field with
    confirmations again, that regression should be visible in the log instead
    of quietly shrinking the count.
    """
    entry = str(raw or "").strip()
    if not entry:
        return ""
    if is_judge_concern(entry):
        return entry
    logger.info(
        "Dropped a judge PASS verdict for %r (not an inconsistency): %s",
        provider_name, entry[:160],
    )
    return ""


def is_judge_concern(text: str) -> bool:
    """Does this `recommendation_adjustments` entry assert an actual problem?

    Unrecognized phrasing counts as a concern. The asymmetry is deliberate:
    over-reporting shows a developer one extra line, while under-reporting
    silently discards the signal this whole mechanism exists to surface.

    Precedence, and why it is in this order. A MIXED verdict — "no correction
    needed for red_flags, but practical_access should have been lowered" —
    used to match its leading all-clear and be dropped, taking a real finding
    out of all three destinations. Only a CONTRASTIVE PIVOT now outranks a
    strong pass; general concern vocabulary still does not, because an
    all-clear is frequently phrased with negation ("judge did not misread
    anything here... no correction needed") and reordering that wholesale
    turns clean passes back into false counts.

    The null-answer check is the other half. The prompt asks for an empty
    string, and a model that writes "None" or "N/A" instead is complying in
    spirit — but neither string matches a pass pattern, so the
    default-to-concern fallback classified all of them as inconsistencies and
    reproduced the exact false patient-facing count this predicate was built
    to eliminate, through the guard rather than around it.
    """
    entry = str(text or "").strip()
    if not entry:
        return False
    if not any(ch.isalnum() for ch in entry):
        return False  # "-", "—", "..." — punctuation is not a finding
    if _JUDGE_NULL_ANSWER_RE.fullmatch(entry):
        return False
    if _JUDGE_MIXED_RE.search(entry):
        return True
    if _JUDGE_STRONG_PASS_RE.search(entry):
        return False
    if _JUDGE_CONCERN_RE.search(entry):
        return True
    return not _JUDGE_PASS_RE.search(entry)


_CONFIDENCE_ADJUSTMENT = {"high": 2.0, "low": -4.0}

# "approved" and "rejected" as bare substrings both misread their own
# negations. Verified against the prompt's own vocabulary plus the drift a
# model actually produces: "conditional approval" contained "approv" and so
# escaped the conditional penalty entirely, scoring identically to a clean
# approval; "not approved", "unapproved" and "disapproved" did the same; and
# "not rejected" — an explicit clearing — took the full -15.
_NEGATED_APPROVAL_RE = re.compile(r"\b(?:not|un|dis)\s*-?\s*approv", re.IGNORECASE)
_NEGATED_REJECT_RE = re.compile(r"\bnot\s+reject", re.IGNORECASE)


def _verdict_class(status: Any) -> str:
    """Classify a critic verdict as approved / conditional / rejected / other.

    `conditional` is tested before `approved` so "conditional approval" and
    "approved with conditions" land where they belong.
    """
    text = str(status or "").strip().lower()
    if not text:
        return "other"
    if "reject" in text and not _NEGATED_REJECT_RE.search(text):
        return "rejected"
    if "condition" in text:
        return "conditional"
    if "approv" in text and not _NEGATED_APPROVAL_RE.search(text):
        return "approved"
    return "other"


def _score_contributions(provider: Dict[str, Any]) -> Dict[str, Any]:
    """Per-dimension weighted contributions to this provider's core score.

    Without these the bias analyst sees only the INPUTS (rating, years,
    distance) and the final TOTAL, and has to guess which dimension drove the
    ordering. In the 2026-07-25 run it guessed wrong and told patients "review
    VOLUME is silently amplifying the rating dimension" — while the arithmetic
    says Dr. An led on experience (+4.36) and location (+3.00) despite LOSING
    rating (-4.07). The scorer already computes all of this and attaches it as
    `score_breakdown`; it simply never reached this payload.
    """
    breakdown = provider.get("score_breakdown")
    if not isinstance(breakdown, dict):
        return {}

    contributions: Dict[str, Any] = {}
    for dimension in ("rating", "experience", "location"):
        entry = breakdown.get(dimension)
        if not isinstance(entry, dict):
            continue
        try:
            score = float(entry.get("score", 0) or 0)
            weight = float(entry.get("weight", 0) or 0)
        except (TypeError, ValueError):
            continue
        contributions[dimension] = {
            "score": round(score, 1),
            "weight": round(weight, 3),
            "weighted_contribution": round(score * weight, 2),
        }
    return contributions


def _location_evidence(provider: Dict[str, Any]) -> str:
    """Human-readable location basis for the critic, read from the scorer's
    OWN breakdown — so the critic critiques address COVERAGE, not a phantom
    'N/A' (the old payload sent the rare page-stated `distance` field, which
    was almost always absent even for providers we had measured). A fallback
    label is an already-penalized imputation, not unearned leniency.
    """
    breakdown = provider.get("score_breakdown")
    location = breakdown.get("location") if isinstance(breakdown, dict) else None
    if isinstance(location, dict):
        basis = location.get("basis")
        value = location.get("value")
        if basis == "computed_distance" and isinstance(value, (int, float)):
            return f"{value} mi (measured from ZIP coordinates)"
        if basis == "city_estimate" and isinstance(value, (int, float)):
            # Says plainly that this figure is shared by every provider in that
            # city, so "all providers show the same distance" reads as the
            # precision limit it is rather than a scoring artifact.
            return f"~{value} mi (city-centre estimate, shared by that city; no street address resolved)"
        if basis == "stated_distance" and value:
            return f"~{value} (page-stated)"
        if basis == "same_zip":
            return "same ZIP — tier fallback, address not resolved to coordinates"
        if basis == "same_city":
            return "same-city tier fallback — street address not resolved"
        if basis == "same_state":
            return "same-state tier fallback — city not resolved"
        if basis == "different":
            return "different area — tier fallback"
        if basis == "missing":
            return "unknown — neutral imputation, no address on any page"
    # No breakdown (pre-scoring path): fall back to the raw fields
    computed = provider.get("computed_distance_miles")
    if isinstance(computed, (int, float)):
        return f"{computed} mi (computed straight-line)"
    stated = provider.get("distance")
    if stated and str(stated).upper() != "N/A":
        return f"~{stated} (page-stated)"
    return "not disclosed"


def _adjusted_rating(provider: Dict[str, Any]) -> Any:
    """The Bayesian-shrunk rating actually scored (from the scorer's
    breakdown), so the critic verifies small-sample claims against the number
    the algorithm used, not the raw headline star value."""
    breakdown = provider.get("score_breakdown")
    rating = breakdown.get("rating") if isinstance(breakdown, dict) else None
    if isinstance(rating, dict):
        return rating.get("adjusted_rating")
    return None


def _normalize_name(name: Any) -> str:
    """Normalize provider names for matching across agent outputs.

    Delegates to the ONE normalization in utils/provider_key — the same
    function the gatherer's dedup and the enrichment cache key use, because
    "is this the same physician?" must not get three different answers in one
    pipeline. The local version this replaced stripped "dr." and "dr " and
    nothing else, so every credential suffix survived: it scored
    "Dr. Hussam Seif-Eddeine, MD" against "Hussam Seif-Eddeine" as
    "hussam seif eddeine md" vs "hussam seif eddeine" — a miss, and likewise a
    miss for "Andrea An, M.D." ("andrea an m d"), for every "…, MD", and for
    "Jane O'Brien DO". The name index was therefore dead in practice and
    essentially every verdict was matched through the positional rank
    fallback below, which silently attaches the wrong critic verdict — and its
    red-flag penalty — whenever the model omits or reorders an entry.
    """
    return normalized_name(name)


def refine_rankings(
    ranked_providers: List[Dict[str, Any]],
    validation_results: Dict[str, Any],
) -> tuple:
    """Fold the critic's structured findings back into the final ranking.

    Pure post-processing over the per-provider validation verdicts (status,
    red flags, confidence). No additional LLM calls, so refinement adds no
    latency or cost. Only the user's own stated weights and the critic's
    evidence-bound verdicts move scores.

    Args:
        ranked_providers: Providers in the preference scorer's order
        validation_results: Output of validate_rankings (wrapped or inner shape)

    Returns:
        (refined_providers, summary) where summary lists the rank moves and
        each refined provider carries pre_refinement_rank / refined_score /
        refinement_reasons
    """
    if not ranked_providers:
        return [], {"applied": False, "moves": [], "adjusted_count": 0}

    inner = (validation_results or {}).get("validation_results", validation_results) or {}
    top_validations = (inner.get("top_provider_validation") or {}).get("top_provider_validations", []) or []

    # Matched by canonical name ONLY. There used to be a positional fallback on
    # the entry's "rank", justified as "the entry's rank refers to the scorer's
    # pre-refinement order" — true until round 10's research budget, which made
    # the critic audit a SUBSET while this function still walks the whole list.
    # The critic numbers its entries 1..N over the providers it saw; this loop
    # numbered providers 1..M over everyone. Every unjudged provider sorting
    # above a judged one shifts the two spaces apart by one:
    #
    #   full list          critic's rank space
    #   1 Alpha  judged  -> 1 Alpha
    #   2 Bravo  UNJUDGED     (never audited)
    #   3 Chen   judged  -> 2 Chen
    #   4 Diaz   judged  -> 3 Diaz
    #
    #   lookup by full position: Bravo->rank 2 = CHEN's verdict (and its -8, on
    #   a provider nobody reviewed); Chen->rank 3 = DIAZ's; Diaz->rank 4 = ...
    #
    # It happened not to fire on 2026-07-27 only because all four unjudged
    # providers sorted below all ten judged ones. Keeping two coordinate systems
    # in step is the coupling that broke; removing one is the fix. A name that
    # cannot be matched now yields NO verdict rather than someone else's — a
    # missing adjustment degrades gracefully, a misattributed one does not.
    # Matched by NAME TOKENS, each entry claimable once. The exact-string name
    # index this replaces could not absorb a dropped middle name
    # ("Dr. Hemant Pandey" for "Dr. Hemant Kumar Pandey, MD"), which is why a
    # positional fallback on the entry's "rank" existed underneath it. Token
    # overlap handles that case directly — {hemant, pandey} against
    # {hemant, kumar, pandey} is 1.0 — so the fallback has nothing left to do.
    # It is the same predicate the judge uses for the same purpose
    # (`preference_scorer._same_provider_name`, threshold 0.5).
    entries = [
        (normalize_name_tokens(entry.get("provider_name")), entry)
        for entry in top_validations
    ]
    entries = [(tokens, entry) for tokens, entry in entries if tokens]

    claimed: set = set()
    refined: List[Dict[str, Any]] = []
    for index, provider in enumerate(ranked_providers):
        original_rank = index + 1
        provider_tokens = normalize_name_tokens(provider.get("name"))
        adjustment = 0.0
        reasons: List[str] = []
        # Adjustments the critic made because it FOUND something, as distinct
        # from a score that merely moved. The panel reports this as "the
        # validator's findings changed N recommendation(s)", and counting any
        # non-zero delta made that number the pool size: the +2 for high
        # confidence is the EXPECTED verdict for a clean record and lands on
        # nearly everyone, so the 2026-07-29 run reported 8 when exactly one
        # provider had been docked. A term with no variance across the pool
        # carries no ranking information and is not a finding — the same
        # reasoning §10.42 applied one level up, where the count was of rows
        # that moved.
        findings = 0

        validation, best_overlap, best_position = None, 0.0, None
        for position, (entry_tokens, entry) in enumerate(entries):
            if position in claimed:
                continue
            shared = provider_tokens & entry_tokens
            overlap = len(shared) / max(min(len(provider_tokens), len(entry_tokens)), 1)
            if overlap >= _VERDICT_NAME_THRESHOLD and overlap > best_overlap:
                validation, best_overlap, best_position = entry, overlap, position
        if best_position is not None:
            claimed.add(best_position)
        critic_review = None
        if validation:
            status = str(validation.get("validation_status", "")).lower()
            verdict = _verdict_class(status)
            if verdict == "rejected":
                adjustment -= _STATUS_PENALTY_REJECTED
                findings += 1
                reasons.append(f"critic rejected this recommendation (-{_STATUS_PENALTY_REJECTED:g})")
            elif verdict in ("conditional", "other") and status:
                adjustment -= _STATUS_PENALTY_NOT_APPROVED
                findings += 1
                reasons.append(f"critic marked it '{status}' (-{_STATUS_PENALTY_NOT_APPROVED:g})")

            red_flags = [flag for flag in validation.get("red_flags", []) or [] if str(flag).strip()]
            if red_flags:
                penalty = min(len(red_flags) * _RED_FLAG_PENALTY, _RED_FLAG_PENALTY_CAP)
                adjustment -= penalty
                findings += 1
                reasons.append(f"{len(red_flags)} red flag(s) from critic review (-{penalty:g})")

            confidence = str(validation.get("confidence_in_recommendation", "")).lower()
            confidence_adj = _CONFIDENCE_ADJUSTMENT.get(confidence, 0.0)
            if confidence_adj:
                adjustment += confidence_adj
                reasons.append(f"{confidence} critic confidence ({confidence_adj:+g})")
                # A "low" is a finding — the rubric reserves it for a provider
                # with no independent platform evidence at all, or directly
                # conflicting evidence. A "high" is the ABSENCE of one: the
                # expected verdict for a clean record, handed out pool-wide.
                if confidence_adj < 0:
                    findings += 1

            # Surface the critic's own words for this provider — the UI shows
            # them in the AI-analysis section, not just the score delta
            critic_review = {
                "status": str(validation.get("validation_status", "") or ""),
                "confidence": confidence,
                "notes": str(validation.get("validation_notes", "") or ""),
                "red_flags": red_flags,
                "considerations": str(validation.get("patient_considerations", "") or ""),
                "judge_findings": _judge_finding_or_empty(
                    validation.get("recommendation_adjustments"),
                    provider.get("name", "Unknown"),
                ),
            }

            # A judge/summary inconsistency is a fault in OUR pipeline, not
            # information a patient needs — so it goes to the log, not the
            # card. It took five field-test rounds for a human to notice by
            # eye that the judge was scoring "no access evidence" beside a
            # summary describing long waits; this makes the next one grep-able.
            # Deliberately does not move any score: see the prompt's routing
            # rule (red_flags demote the provider, this must not).
            if critic_review["judge_findings"]:
                logger.warning(
                    "Critic flagged a judge/evidence inconsistency for %r: %s",
                    provider.get("name", "Unknown"), critic_review["judge_findings"],
                )

        try:
            base_score = float(provider.get("final_score", 0) or 0)
        except (TypeError, ValueError):
            base_score = 0.0

        refined_provider = dict(provider)
        refined_provider["pre_refinement_rank"] = original_rank
        refined_provider["refined_score"] = round(max(0.0, min(base_score + adjustment, 100.0)), 1)
        refined_provider["refinement_adjustment"] = round(adjustment, 1)
        refined_provider["refinement_findings"] = findings
        refined_provider["refinement_reasons"] = reasons
        refined_provider["critic_review"] = critic_review
        refined.append(refined_provider)

    if top_validations and not any(p.get("critic_review") for p in refined):
        logger.warning(
            "Critic produced %d validation(s) but none matched a provider by "
            "name or rank — per-provider critic reviews will be empty",
            len(top_validations),
        )

    # A verdict that bound to nobody is a silent loss of the critic's work — it
    # was previously masked by the rank fallback, which always found SOMETHING
    # to attach to (that was the bug, not the cure).
    unbound = [
        str(entry.get("provider_name") or "?")
        for position, (_tokens, entry) in enumerate(entries) if position not in claimed
    ]
    if unbound:
        logger.warning(
            "%d critic verdict(s) matched no provider by name and were dropped: %s",
            len(unbound), ", ".join(sorted(unbound)),
        )

    # Stable sort keeps the scorer's order for untouched providers
    refined.sort(key=lambda p: -p["refined_score"])

    moves = []
    for index, provider in enumerate(refined):
        refined_rank = index + 1
        provider["final_rank"] = refined_rank
        if refined_rank != provider["pre_refinement_rank"]:
            moves.append({
                "name": provider.get("name", "Unknown"),
                "from": provider["pre_refinement_rank"],
                "to": refined_rank,
                "reasons": provider["refinement_reasons"],
            })

    summary = {
        "applied": bool(moves),
        "moves": moves,
        # Providers the critic FOUND something about — not providers whose
        # score moved. See `findings` above.
        "adjusted_count": sum(1 for p in refined if p["refinement_findings"]),
    }
    if moves:
        logger.info(f"Critic refinement re-ordered {len(moves)} provider(s)")
    return refined, summary


class CriticValidatorAgent:
    """Agent responsible for critically evaluating and validating provider rankings with sophisticated reasoning."""

    def __init__(self):
        """Initialize the critic validator with Anthropic client."""
        self.config = get_config()
        self.anthropic_client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            if not self.config.ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not found in configuration")

            self.anthropic_client = Anthropic(api_key=self.config.ANTHROPIC_API_KEY)
            logger.info("Critic validator client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize critic validator client: {e}")
            raise

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from markdown-wrapped responses.

        Args:
            response_text: Raw response text from Claude

        Returns:
            Cleaned JSON string
        """
        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            # Extract JSON from markdown code block
            match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if match:
                return match.group(1).strip()
        elif "```" in response_text:
            # Extract from generic code block
            match = re.search(r'```\s*([\s\S]*?)\s*```', response_text)
            if match:
                return match.group(1).strip()

        # If no code blocks, try to find JSON object or array
        # Look for JSON object {...} or array [...]
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response_text)
        if json_match:
            return json_match.group(1).strip()

        return response_text.strip()

    def _parse_json_with_repair(self, json_text: str, context: str):
        """Parse LLM JSON with escalating repairs; None only when unrecoverable.

        Sonnet's long free-text fields occasionally break strict JSON —
        unescaped inner quotes, raw newlines inside strings, trailing commas.
        Chain: direct parse -> mechanical repairs (trailing commas removed,
        strict=False tolerates control chars in strings) -> one cheap Haiku
        "fix syntax only" call. Every stage preserves content; the LLM stage
        is instructed to change nothing but syntax.
        """
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

        mechanical = re.sub(r',\s*([}\]])', r'\1', json_text)
        try:
            result = json.loads(mechanical, strict=False)
            logger.info(f"{context}: JSON recovered by mechanical repair")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"{context}: mechanical JSON repair insufficient ({e}); trying LLM repair")

        try:
            llm_started = time.perf_counter()
            response = self.anthropic_client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=6000,
                messages=[{
                    "role": "user",
                    "content": (
                        "The text below is meant to be one valid JSON document but has syntax "
                        "errors (typically unescaped double quotes inside strings). Return the "
                        "SAME content as strictly valid JSON. Do not add, remove, reword, or "
                        "reorder anything — fix syntax only. Return ONLY the JSON.\n\n"
                        f"{json_text}"
                    ),
                }],
            )
            in_tokens, out_tokens = safe_usage(response)
            get_cost_tracker().record_llm(
                "claude-haiku-4-5", in_tokens, out_tokens,
                agent="critic_validator", duration_s=time.perf_counter() - llm_started
            )
            repaired = self._extract_json_from_response(response.content[0].text.strip())
            result = json.loads(repaired, strict=False)
            logger.info(f"{context}: JSON recovered via LLM repair")
            return result
        except Exception as e:
            logger.error(f"{context}: JSON unrecoverable after all repairs ({e})")
            return None

    def _analyze_ranking_bias(self, ranked_providers: List[Dict[str, Any]], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential biases in the current rankings.

        Args:
            ranked_providers: List of ranked provider dictionaries
            preferences: User preferences used for ranking

        Returns:
            Dictionary containing bias analysis
        """
        try:
            # Sanitize preferences first
            safe_preferences = InputValidator.validate_preferences(preferences)

            # Prepare data for analysis
            ranking_data = []
            # The research budget, not a second hardcoded one. This was `[:10]`
            # while the shipped budget moved to 8 — a parallel cap that had
            # ALREADY diverged, so the bias analyst reasoned over a pool the rest
            # of the pipeline had not researched.
            for i, provider in enumerate(ranked_providers[:self.config.MAX_PROVIDERS_TO_ENRICH]):
                ranking_data.append({
                    "rank": i + 1,
                    "name": provider.get("name", "Unknown"),
                    "rating": provider.get("rating", 0),
                    "review_count": provider.get("review_count"),
                    "blended_rating": provider.get("blended_rating"),
                    "blended_review_count": provider.get("blended_review_count"),
                    "blended_platform_count": provider.get("blended_platform_count", 0),
                    "adjusted_rating": _adjusted_rating(provider),
                    "location_evidence": _location_evidence(provider),
                    # `insurance` is NOT sent — see the note at the deep-
                    # validation payload below. The judge was already denied it.
                    "final_score": provider.get("final_score", 0),
                    "ai_reasoning": provider.get("ai_reasoning", "No reasoning"),
                    "years_experience": provider.get("years_experience", "N/A"),
                    # What actually moved this provider. Without it the analyst
                    # sees inputs and a total, and must guess at the middle.
                    "score_contributions": _score_contributions(provider),
                })

            prompt = f"""As a critical healthcare analytics expert, analyze this provider ranking for potential biases and blind spots.

<user_preferences>
{json.dumps(safe_preferences, indent=2)}
</user_preferences>

CURRENT TOP RANKINGS:
{json.dumps(ranking_data, indent=2)}

SCORING MECHANICS (ground your bias claims in these facts, not assumptions):
- Ratings are Bayesian-shrunk by review volume in the deterministic core — a 5.0 on a handful of reviews scores well below a 4.8 on hundreds; small-sample ratings do NOT outrank large ones on stars alone.
- adjusted_rating IS the star value actually scored (post-shrinkage). When arguing a thin-evidence rating is over-rewarded, compare adjusted_rating values, not the raw 5.0 vs 4.8 headlines — the shrinkage has already happened.
- blended_rating/blended_review_count/blended_platform_count show cross-platform evidence volume; cite these numbers when claiming a rating rests on thin evidence.
- location_evidence is each provider's ACTUAL distance basis. A "tier fallback" or "not resolved" value is an already-penalized imputation (its tier score sits BELOW a comparable measured distance), NOT unearned leniency — critique missing address COVERAGE if you like, but never claim an unresolved provider received a non-penalizing "N/A distance".
- Insurance is verification-only BY DESIGN: a sidebar payer-directory (FHIR) check, deliberately not a ranking factor. Never flag its absence from the scoring weights as a bias or gap.
- score_contributions shows what ACTUALLY moved each provider: per dimension, its 0-100 score, its weight, and the weighted_contribution those produce. The core score is the sum of the three contributions.
- ANY claim about which dimension drove the ordering MUST cite weighted_contribution values and MUST NOT be inferred from the raw inputs. Compare contributions between the providers you are contrasting, and say which dimension supplied the margin.
- A weight cannot be "silently exceeded" or "amplified beyond its nominal value": the weight multiplies the dimension score and nothing else. Bayesian shrinkage moves the rating VALUE toward the prior, which NARROWS the gap between a thin 5.0 and a well-evidenced 4.2 — it never increases the rating dimension's influence. If the top-ranked provider is not the highest-rated, find the dimension whose weighted_contribution supplies the margin and name that one.

WRITING FOR TWO AUDIENCES — this matters as much as the analysis:
- "explanation" and every entry in "detected_biases" are shown DIRECTLY TO PATIENTS on the results page. Write them in plain language a non-technical person can act on. NEVER use internal field names (adjusted_rating, blended_review_count, years_experience, ai_reasoning, score_contributions, weighted_contribution), snake_case, raw internal scores (e.g. 89.96), or scoring jargon ("post-shrinkage", "quantization", "monotonic"). Say "review score" not "adjusted_rating"; "how far away they are" not "location_evidence"; "how many reviews back it up" not "blended_review_count".
- "technical_explanation" is for DEVELOPERS ONLY and is never shown to patients. Put the field names, the weighted_contribution arithmetic, and the exact numbers there. Be as precise and technical as you like.
- Say the same thing in both. They are two registers of one finding, not two different findings.

CRITICAL ANALYSIS REQUIRED:

1. BIAS DETECTION:
   - Are rankings overly influenced by any single factor?
   - Do preferences create unfair advantages/disadvantages?
   - Are there geographic or demographic biases?

2. BLIND SPOTS:
   - What important factors might be missing?
   - Are there hidden quality indicators not considered?
   - Could the ranking mislead patients?

3. RANKING VALIDITY:
   - Do top-ranked providers truly serve user needs?
   - Are lower-ranked providers unfairly penalized?
   - Is the ranking methodology sound?

IMPORTANT: Return ONLY valid JSON. No markdown, no explanations, just pure JSON.
Ensure all strings are properly escaped. Use double quotes for all keys and string values.

Return analysis as this exact JSON structure:
{{
  "bias_assessment": {{
    "detected_biases": ["bias1", "bias2"],
    "severity": "low",
    "explanation": "Plain-language summary for a PATIENT",
    "technical_explanation": "The numeric reasoning, for developers"
  }},
  "blind_spots": {{
    "missing_factors": ["factor1", "factor2"],
    "impact": "Impact description",
    "recommendations": ["rec1", "rec2"]
  }},
  "validity_concerns": {{
    "ranking_issues": ["issue1", "issue2"],
    "misleading_aspects": ["aspect1", "aspect2"],
    "confidence_level": "medium"
  }},
  "overall_assessment": "Brief assessment summary"
}}

Be thorough and critical. Return ONLY the JSON object, nothing else."""

            llm_started = time.perf_counter()
            response = self.anthropic_client.messages.create(
                model=self.config.CRITIC_MODEL,
                max_tokens=4500,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}]
            )
            in_tokens, out_tokens = safe_usage(response)
            get_cost_tracker().record_llm(
                self.config.CRITIC_MODEL, in_tokens, out_tokens,
                agent="critic_validator", duration_s=time.perf_counter() - llm_started
            )

            response_text = response.content[0].text.strip()
            # Extract JSON from markdown if needed
            cleaned_response = self._extract_json_from_response(response_text)

            bias_analysis = self._parse_json_with_repair(cleaned_response, "bias analysis")
            if isinstance(bias_analysis, dict):
                logger.info("Bias analysis completed successfully")
                return bias_analysis

            logger.error(f"Bias analysis response unusable; preview: {cleaned_response[:500]}...")
            return {
                "bias_assessment": {"detected_biases": [], "severity": "unknown", "explanation": "Analysis failed"},
                "blind_spots": {"missing_factors": [], "impact": "Unknown", "recommendations": []},
                "validity_concerns": {"ranking_issues": [], "misleading_aspects": [], "confidence_level": "low"},
                "overall_assessment": "Critical analysis could not be completed"
            }

        except Exception as e:
            logger.error(f"Bias analysis failed: {e}")
            return {
                "bias_assessment": {"detected_biases": [], "severity": "unknown", "explanation": "Analysis failed"},
                "blind_spots": {"missing_factors": [], "impact": "Unknown", "recommendations": []},
                "validity_concerns": {"ranking_issues": [], "misleading_aspects": [], "confidence_level": "low"},
                "overall_assessment": f"Error during analysis: {str(e)}"
            }

    def _validate_top_recommendations(self, ranked_providers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the top recommended providers with detailed scrutiny.

        Runs in `_VALIDATION_SHARDS` CONCURRENT calls when the pool is big
        enough. This was 36.9s of a 97.8s run — the single most expensive step
        — and the verdicts are genuinely per-provider: nothing in the rubric
        scores one provider relative to another, and `refine_rankings` binds
        verdicts back by NAME TOKENS, so a shard's entries carry no positional
        dependency on the call they came from.

        Shards are ROUND-ROBIN. `ranked_providers` arrives best-first, so
        contiguous halves would hand one call only strong providers and the
        other only weak ones — and the DIFFERENTIATION CHECK below asks the
        critic to find real differences among the providers in front of it. A
        shard with no spread has none to find. Dealing preserves the spread in
        both. (The bias analysis is the opposite case and stays ONE call: it
        reasons about the ORDERING, which no half of the pool contains.)

        Args:
            ranked_providers: List of ranked providers

        Returns:
            Dictionary with validation results for top providers
        """
        try:
            # Validate EVERY ranked provider, not a top slice. A partial
            # audit is a flat tax on the audited: live, the critic
            # red-flagged all 8 validated leaders ~equally and the
            # never-audited pre-#9 rose to #1 by default in a 2.4-point
            # field. Uniform coverage makes penalties move RELATIVE
            # differences only — nobody wins by escaping review. The
            # notes cap keeps output bounded: 3 sentences cost ~$0.05 over a
            # top-8 slice, and 2 is where Phase 2 put it — the notes are
            # rendered as a single card line, where the third sentence was
            # routinely a restatement of the first.
            validation_data = []

            for index, provider in enumerate(ranked_providers):
                validation_data.append({
                    "name": provider.get("name", "Unknown"),
                    # Positional rank in the scorer's order — refine_rankings
                    # matches entries back by this (final_rank doesn't exist
                    # yet at validation time)
                    "rank": index + 1,
                    "rating": provider.get("rating", 0),
                    "review_count": provider.get("review_count", 0),
                    # Byte-identical to what the judge was given — same helper,
                    # same bound, same default. The critic used to read the
                    # summary in full while the judge read a 400-char stub, so
                    # the two "disagreed" about evidence only one of them could
                    # see. A critic that reviews different text than the agent
                    # it is auditing is not an independent check; it is a
                    # second opinion on a different question.
                    "review_summary": clip_words(
                        provider.get("review_summary") or "No reviews available",
                        SUMMARY_MAX_CHARS,
                    ),
                    "review_sentiment": provider.get("review_sentiment", "unknown"),
                    "review_source": source_domain(provider.get("review_source_url")) or "unknown",
                    # Cross-platform blend when ≥2 platform pairs exist — the
                    # confidence rubric keys off blended_platform_count, so the
                    # critic can ground "evidence volume" in actual data
                    "blended_rating": provider.get("blended_rating"),
                    "blended_review_count": provider.get("blended_review_count"),
                    "blended_platform_count": provider.get("blended_platform_count", 0),
                    # Each platform's own numbers, so disagreement findings can
                    # name the outlier platform instead of just noting that
                    # the blend sits far from the headline
                    "platform_observations": [
                        f"{source_domain(obs.get('source_url')) or 'unknown'} "
                        f"{obs.get('rating')}/5"
                        + (f" ({obs.get('review_count')} reviews)"
                           if obs.get("review_count") else "")
                        for obs in (provider.get("review_observations") or [])[:5]
                        if isinstance(obs, dict) and obs.get("rating") is not None
                    ],
                    "final_score": provider.get("final_score", 0),
                    "ai_reasoning": provider.get("ai_reasoning", ""),
                    # The judge's ACTUAL per-criterion output. This slot used to
                    # hold `ai_confidence`, a field no agent has ever written —
                    # so every provider on every search arrived carrying the
                    # literal default 50, and the checklist line asking whether
                    # confidence tracked quality was auditing a constant.
                    # Sending the rubric instead lets the critic check the
                    # judge's scores against the same review text (see the
                    # JUDGE CONSISTENCY step below) — the one contradiction it
                    # was structurally unable to catch before.
                    "ai_rubric": provider.get("ai_rubric") or {},
                    "ai_evidence": provider.get("ai_evidence") or {},
                    "strengths": provider.get("ai_strengths", []),
                    "concerns": provider.get("ai_concerns", []),
                    # `insurance` is deliberately absent, matching the judge.
                    #
                    # It was load-bearing for nothing: no checklist step, rubric
                    # or verdict criterion referenced it, and the prompt named it
                    # only in a descriptive list. What it DID produce was the
                    # critic asserting coverage as fact in patient_considerations
                    # ("accepts Aetna, Cigna, Humana, Medicare"), which the card
                    # then contradicted two clauses later with the same scraped
                    # list presented as unverified plus "verify coverage with the
                    # provider directly". Only the FHIR network check is
                    # supposed to speak to coverage; this was the one path
                    # laundering directory data into an assertion.
                    "location": provider.get("location", ""),
                    "location_evidence": _location_evidence(provider)
                })

            shards = (
                round_robin_shards(validation_data, _VALIDATION_SHARDS)
                if len(validation_data) >= _MIN_PROVIDERS_TO_SPLIT_VALIDATION
                else [validation_data]
            )
            if len(shards) <= 1:
                return self._validate_shard(validation_data)

            logger.info(
                "Critic deep validation split across %d concurrent calls "
                "(%s providers each)",
                len(shards), "/".join(str(len(s)) for s in shards),
            )
            with ThreadPoolExecutor(max_workers=len(shards)) as executor:
                futures = [
                    executor.submit(self._validate_shard, shard) for shard in shards
                ]
                return _merge_validation_shards([future.result() for future in futures])

        except Exception as e:
            logger.error(f"Top provider validation failed: {e}")
            return {
                "top_provider_validations": [],
                "overall_ranking_validity": {
                    "status": "error",
                    "confidence": "low",
                    "summary": f"Validation error: {str(e)}",
                    "improvement_suggestions": []
                }
            }

    def _validate_shard(self, validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run ONE deep-validation call over these provider records.

        Split out of `_validate_top_recommendations` so several can run
        concurrently; the payload is built there, once, with GLOBAL ranks.
        """
        try:
            prompt = f"""As a senior healthcare quality auditor, rigorously validate these {len(validation_data)} provider recommendations (listed in ranking order, best first).

TOP PROVIDERS TO VALIDATE:
{json.dumps(validation_data, indent=2)}

Each provider includes:
- Basic info (name, rating, review_count)
- Review data (review_summary with patient feedback themes, review_sentiment: positive/mixed/negative, review_source: the domain the summary came from)
- Cross-platform blend when available (blended_rating over blended_review_count reviews across blended_platform_count independent platforms; blended_platform_count 0 or null = only one platform's numbers exist)
- platform_observations: each independent platform's OWN rating and count — when platforms disagree, cite the specific platforms and numbers, not just "sources vary"
- The upstream AI judge's output: ai_reasoning, strengths, concerns, plus ai_rubric (its per-criterion scores) and ai_evidence (the snippet it cited for each). You and the judge were given the SAME review_summary text AND the same rubric — it is reproduced below, so audit against it rather than against your own idea of what each criterion should cover.
- Practical factors (location, location_evidence). location_evidence is the provider's actual distance basis; a "tier fallback"/"not resolved" value is an already-penalized imputation, never a red flag.
- You are given NO insurance or plan data. Never state, imply, or speculate about which plans a provider accepts — not in patient_considerations, not anywhere. The interface presents plan lists separately with their own "unverified, confirm with the provider" framing, and a confident claim here contradicts it on the same card.

THE RUBRIC THE JUDGE WAS SCORED AGAINST (verbatim — its routing rules bind your audit too):
{JUDGE_RUBRIC}

VALIDATION CHECKLIST (where to look):

1. RECOMMENDATION QUALITY:
   - Is each provider truly suitable for the user's needs?
   - Are the rankings justified and logical?
   - Does the review sentiment align with the rating and AI assessment?

2. JUDGE CONSISTENCY (you and the judge read the SAME review_summary — so any gap is a real error, not a difference of sources):
   - Does each ai_evidence snippet actually appear in, or fairly paraphrase, that provider's review_summary? An evidence snippet with no basis in the summary is a fabrication — flag it.
   - Did the judge score a criterion in its NEUTRAL band while the summary plainly contains evidence for it? practical_access sitting at 8-11 ("no access signals either way") when the summary describes wait times, scheduling problems, or an unreachable office is the clearest case. Name the criterion and quote the summary line the judge missed.
   - A score is only an error if the RUBRIC ABOVE says so. Its routing rules are part of the standard: access complaints belong to practical_access and must NOT also be charged to red_flags, and distance is never scored by the judge at all. A judge that scored delays under practical_access and left red_flags untouched has FOLLOWED the rubric — calling that "generous" asks for the double-charge the rubric exists to prevent. Read the bands before calling a number wrong.
   - Does ai_reasoning describe the evidence it was given, or does it complain about the evidence being absent/incomplete when the summary in fact contains it?
   - WHERE THESE FINDINGS GO: a judge mistake is a SYSTEM fault, not the provider's — put it in "recommendation_adjustments" and NEVER in "red_flags" or "validation_status", which move the provider's rank. The underlying patient evidence is judged separately on its own merits: if the summary really does describe repeated long waits, that is a provider red flag in its own right (step 3) whether or not the judge noticed it.

3. REVIEW DATA ANALYSIS:
   - Are there red flags in the review summaries (e.g., "long wait times", "rude staff", "poor communication")?
   - Does review sentiment contradict the high rating or ranking?
   - Do platforms disagree with each other? platform_observations lists each platform's numbers — name the outlier (e.g. "healthgrades 2.1/13 vs vitals 3.5/16"), never just "ratings vary".

4. USER SAFETY:
   - Would you personally recommend these providers to a family member based on reviews and data?
   - Any risk factors from reviews that patients should know about?

5. RANKING ACCURACY:
   - Is the #1 provider truly the best choice considering reviews?
   - Are the lower ranks appropriately positioned relative to each other?

VERDICT RUBRIC — every output field has entry criteria. A label without its required evidence is wrong, even if it feels cautious:

validation_status (exactly one of "approved" / "conditional" / "rejected"):
- "rejected" — ONLY with specific disqualifying evidence you can quote from the data above: a direct contradiction of the user's stated requirement, a sub-3.0 rating on independent platforms, a specialty mismatch, or a concrete safety signal in the review text. Name that evidence in validation_notes.
- "conditional" — ONLY when you can name ONE specific concern AND what evidence would resolve it. Generic caveats ("limited data", "could be more complete", "web sources may be incomplete") do NOT qualify — they apply to every provider equally and carry zero ranking information.
- "approved" — the EXPECTED verdict for a provider whose evidence is consistent and shows no disqualifying signal. Approving a clean provider is not negligence; withholding approval from one is a rubric violation, not caution.

red_flags (each one costs the provider ranking points, so this list is for the PROVIDER's conduct only):
- Each flag must cite the specific observation it comes from ("healthgrades 2.1/13 contradicts vitals 3.5/16", "reviews repeatedly mention billing disputes") — never a category label ("data quality", "limited reviews").
- Missing data is NEVER a red flag: unrated or thinly-reviewed providers already pay for missing evidence through neutral scores upstream — flagging it again double-penalizes. An empty red_flags list is a normal, common output.
- A mistake by the upstream judge is NEVER a red flag either: the provider did not cause it and must not be demoted for it. Those go in recommendation_adjustments.

recommendation_adjustments (free text; affects NO score — this is where system-level findings go):
- Any JUDGE CONSISTENCY finding from step 2: an unsupported ai_evidence snippet, a criterion parked in its neutral band while the summary contains evidence for it, or ai_reasoning complaining about evidence it was actually given. Name the criterion and quote the line.
- EMPTY STRING is the expected output. Write here ONLY to report a judge MISTAKE.
- NEVER write a confirmation. If the judge's scoring matches the evidence, return "" — do not write "scoring matches evidence", "correctly scored", "no correction needed", "consistent with the summary", or any other all-clear. This field is counted and shown to patients as "inconsistencies found"; a pass verdict written here reports an error that did not happen.

confidence_in_recommendation (evidence volume, not mood):
- "high" — blended_platform_count >= 2 with a consistent story, or one credible platform pair plus review text that agrees with it.
- "medium" — one platform pair, or minor cross-source inconsistencies.
- "low" — RESERVED for providers with no independent platform evidence at all, or directly conflicting evidence. If most providers here have platform data, most should NOT be "low".

DIFFERENTIATION CHECK: these providers are being ranked AGAINST EACH OTHER. If your verdicts assign the same status, flag count, and confidence to most of them, you have not done the analysis — go back and find the real differences. Uniform penalties cancel out and transmit zero ranking information.

IMPORTANT: Return ONLY valid JSON. No markdown, no explanations, just pure JSON.
Ensure all strings are properly escaped. Use double quotes for all keys and string values.
Return one entry PER PROVIDER, echoing each provider's given "rank" value exactly.
"rank" is the provider's position in the FULL ranking, so these values may not be consecutive and may not start at 1 — that is expected and is not an error to report. Never renumber them.
Keep validation_notes to at most 2 sentences per provider.

Return detailed validation as this exact JSON structure:
{{
  "top_provider_validations": [
    {{
      "provider_name": "Provider Name",
      "rank": 1,
      "validation_status": "approved",
      "confidence_in_recommendation": "high",
      "validation_notes": "Brief assessment citing the evidence behind the verdict",
      "red_flags": [],
      "recommendation_adjustments": "Brief adjustments",
      "patient_considerations": "Brief considerations"
    }}
  ],
  "overall_ranking_validity": {{
    "status": "validated",
    "confidence": "medium",
    "summary": "Brief overall assessment",
    "improvement_suggestions": ["suggestion1", "suggestion2"]
  }}
}}

Be rigorous and evidence-bound — every verdict must survive the rubric above. Return ONLY the JSON object, nothing else."""

            llm_started = time.perf_counter()
            budget = _validation_token_budget(len(validation_data))
            response = self.anthropic_client.messages.create(
                model=self.config.CRITIC_MODEL,
                max_tokens=budget,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}]
            )
            in_tokens, out_tokens = safe_usage(response)
            get_cost_tracker().record_llm(
                self.config.CRITIC_MODEL, in_tokens, out_tokens,
                agent="critic_validator", duration_s=time.perf_counter() - llm_started
            )

            # Nothing checked this, so a response cut off mid-array reported
            # itself as a generic parse failure — and round 13 raised the
            # stakes: a critic response that cannot be recovered leaves every
            # provider `not_critiqued`, which correctly empties the shortlist.
            # The right failure with the wrong log is still a bug: the operator
            # sees zero recommendations and no cause. Mirrors the judge's
            # `finish_reason` handler (preference_scorer), which was added for
            # the identical incident one agent over.
            if getattr(response, "stop_reason", None) == "max_tokens":
                logger.error(
                    "Critic validation hit the %s-token ceiling for %d providers and was "
                    "TRUNCATED. The repair chain below salvages what it can; raise the "
                    "budget in _validation_token_budget if this recurs. An unrecoverable "
                    "response leaves every provider not_critiqued and EMPTIES the "
                    "shortlist.",
                    budget, len(validation_data),
                )

            response_text = response.content[0].text.strip()
            # Extract JSON from markdown if needed
            cleaned_response = self._extract_json_from_response(response_text)

            validation_results = self._parse_json_with_repair(cleaned_response, "top provider validation")
            if isinstance(validation_results, dict):
                logger.info("Top provider validation completed")
                return validation_results

            logger.error(f"Validation response unusable; preview: {cleaned_response[:500]}")
            return {
                "top_provider_validations": [],
                "overall_ranking_validity": {
                    "status": "error",
                    "confidence": "low",
                    "summary": "Validation could not be completed",
                    "improvement_suggestions": []
                }
            }

        # Contained here so one failed shard costs only its own providers a
        # verdict (they end up `not_critiqued` and are named as withheld) while
        # the other shard's verdicts still apply. Letting it escape would make
        # `future.result()` raise into the caller and empty the shortlist for
        # the whole pool.
        except Exception as e:
            logger.error(
                f"Deep validation failed for a {len(validation_data)}-provider shard: {e}"
            )
            return {
                "top_provider_validations": [],
                "overall_ranking_validity": {
                    "status": "error",
                    "confidence": "low",
                    "summary": f"Validation error: {str(e)}",
                    "improvement_suggestions": []
                }
            }

    def validate_rankings(self, ranked_providers: List[Dict[str, Any]], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to critically validate and challenge provider rankings.

        Args:
            ranked_providers: List of ranked provider dictionaries
            preferences: User preferences used for original ranking

        Returns:
            Dictionary containing comprehensive validation results and critiques
        """
        try:
            # Sanitize preferences first
            safe_preferences = InputValidator.validate_preferences(preferences)

            logger.info(f"Starting critical validation of {len(ranked_providers)} ranked providers")

            if not ranked_providers:
                return {
                    "validation_results": {
                        "bias_analysis": {},
                        "top_provider_validation": {},
                        "final_recommendations": []
                    },
                    "validation_metadata": {
                        "total_providers_analyzed": 0,
                        "validation_method": "claude_sonnet_critical_analysis",
                        "validation_timestamp": "N/A"
                    },
                    "status": "no_providers",
                    "message": "No providers to validate"
                }

            # The two analyses are independent Claude calls; run them
            # concurrently so validation wall time is the slower call instead
            # of the sum.
            logger.info("Running bias analysis and top-provider validation in parallel...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                bias_future = executor.submit(self._analyze_ranking_bias, ranked_providers, safe_preferences)
                top_future = executor.submit(self._validate_top_recommendations, ranked_providers)

                bias_analysis = bias_future.result()
                top_validation = top_future.result()

            # Fold the two analyses into final critical recommendations
            final_recommendations = self._generate_final_recommendations(
                ranked_providers, bias_analysis, top_validation
            )

            result = {
                "validation_results": {
                    "bias_analysis": bias_analysis,
                    "top_provider_validation": top_validation,
                    "final_recommendations": final_recommendations
                },
                "validation_metadata": {
                    "total_providers_analyzed": len(ranked_providers),
                    "validation_method": "claude_sonnet_critical_analysis",
                    "bias_severity": bias_analysis.get("bias_assessment", {}).get("severity", "unknown"),
                    "ranking_confidence": top_validation.get("overall_ranking_validity", {}).get("confidence", "unknown")
                },
                "status": "success",
                "message": f"Critical validation completed for {len(ranked_providers)} providers"
            }

            logger.info(f"Validation completed: {result['message']}")
            return result

        except Exception as e:
            logger.error(f"Rankings validation failed: {e}")
            return {
                "validation_results": {
                    "bias_analysis": {},
                    "top_provider_validation": {},
                    "final_recommendations": []
                },
                "validation_metadata": {
                    "total_providers_analyzed": len(ranked_providers),
                    "validation_method": "claude_sonnet_critical_analysis"
                },
                "status": "error",
                "message": f"Error during validation: {str(e)}"
            }

    def _generate_final_recommendations(
        self,
        ranked_providers: List[Dict[str, Any]],
        bias_analysis: Dict[str, Any],
        top_validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final recommendations based on all validation analyses.

        Args:
            ranked_providers: Original ranked providers
            bias_analysis: Bias analysis results
            top_validation: Top provider validation results

        Returns:
            Final recommendations dictionary
        """
        try:
            # Determine recommendation confidence
            bias_severity = bias_analysis.get("bias_assessment", {}).get("severity", "medium")
            validation_confidence = top_validation.get("overall_ranking_validity", {}).get("confidence", "medium")

            # Generate final recommendations
            recommendations = {
                "recommendation_confidence": "high" if bias_severity == "low" and validation_confidence == "high" else "medium",
                "key_findings": [],
                "user_guidance": [],
                "provider_recommendations": [],
                "important_considerations": []
            }

            # key_findings deliberately stays empty. It held two hardcoded
            # strings — "Detected potential biases in ranking methodology" and
            # "Top provider validation status: <status>" — which restated the
            # tiles directly above them in worse language. No model output ever
            # reached the section, and the panel no longer renders it.

            # user_guidance carries EARNED entries only, for the same reason
            # key_findings above carries none.
            #
            # It used to open with an unconditional "Review detailed provider
            # information beyond just rankings" — from no model output, appended
            # on every run. Round 7 wired this field to the panel's "What this
            # ranking doesn't capture", whose other entries are real gaps the
            # critic identified in OUR ranking; wiring it revealed that its
            # first element was filler, and filler under that heading reads as a
            # gap we found and can't articulate.
            #
            # The line below is conditional on low validation confidence, so it
            # says something true when it appears.
            if validation_confidence == "low":
                recommendations["user_guidance"].append("Exercise additional caution in provider selection")

            # Add provider recommendations
            for i, provider in enumerate(ranked_providers[:3]):
                validation_info = None
                for val in top_validation.get("top_provider_validations", []):
                    if val.get("rank") == i + 1:
                        validation_info = val
                        break

                provider_rec = {
                    "name": provider.get("name", "Unknown"),
                    "rank": i + 1,
                    "recommendation": "proceed with confidence" if validation_info and validation_info.get("validation_status") == "approved" else "consider carefully",
                    "key_considerations": validation_info.get("patient_considerations", "No specific considerations") if validation_info else "No validation available"
                }
                recommendations["provider_recommendations"].append(provider_rec)

            # Add important considerations
            blind_spots = bias_analysis.get("blind_spots", {}).get("missing_factors", [])
            if blind_spots:
                recommendations["important_considerations"].extend([f"Consider {factor}" for factor in blind_spots[:3]])

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate final recommendations: {e}")
            return {
                "recommendation_confidence": "low",
                "key_findings": ["Error generating recommendations"],
                "user_guidance": ["Manual review recommended"],
                "provider_recommendations": [],
                "important_considerations": []
            }


def create_critic_validator() -> CriticValidatorAgent:
    """Factory function to create a CriticValidatorAgent instance.

    Returns:
        CriticValidatorAgent instance
    """
    return CriticValidatorAgent()