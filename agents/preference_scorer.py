"""Preference Scorer Agent: deterministic weighted core (70%) plus a
rubric-scored judge (30%, JUDGE_MODEL — gpt-5.6-terra by default)."""

import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
import json
from openai import OpenAI

from utils.config import get_config
from utils.cost_tracker import get_cost_tracker, safe_usage
from utils.excerpt import SUMMARY_MAX_CHARS, clip_words
from utils.provenance import source_domain
from utils.provider_key import normalize_name_tokens, resolve_cache_key
from utils.shard import round_robin_shards

logger = logging.getLogger(__name__)


def _same_provider_name(claimed: Any, actual: Any) -> bool:
    """Do two spellings of a provider name refer to the same person?

    Uses the canonical tokenizer so credentials and punctuation don't matter
    ("Dr. Andrea An, M.D." == "Andrea An MD"). Missing or unparseable names
    degrade to accept: this guard exists to catch a judge binding its answer
    to the WRONG provider, not to discard a correct answer over spelling.
    """
    a, b = normalize_name_tokens(claimed), normalize_name_tokens(actual)
    if not a or not b:
        return True
    return len(a & b) / max(min(len(a), len(b)), 1) >= 0.5


def _judge_token_budget(provider_count: int) -> int:
    """Output-token allowance for one judge call, scaled to the pool.

    Each entry costs roughly 150-250 tokens (three subscores, three cited
    snippets, a patient-facing paragraph, strengths and concerns), and on a
    reasoning model the reasoning trace is drawn from the same allowance — so
    the headroom below is deliberate rather than generous.
    """
    return max(4000, min(_JUDGE_BASE_TOKENS + _JUDGE_TOKENS_PER_PROVIDER * max(provider_count, 0),
                         _JUDGE_MAX_TOKENS))


def _salvage_json_objects(text: str) -> List[Dict[str, Any]]:
    """Recover complete top-level JSON objects from a truncated array.

    A response cut off mid-array is unparseable as a whole but the entries
    before the cut are intact. Losing all twenty because the twentieth was
    clipped is the difference between a degraded ranking and no judge at all.
    The critic already recovers from malformed JSON (`_parse_json_with_repair`);
    the scorer had no equivalent despite making the more expensive call.
    """
    objects: List[Dict[str, Any]] = []
    depth = 0
    start = None
    in_string = False
    escaped = False
    for position, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = position
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        parsed = json.loads(text[start:position + 1])
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        objects.append(parsed)
                    start = None
    return objects


def _parse_ranking_response(response_text: str) -> List[Dict[str, Any]]:
    """Judge entries from a response, tolerating fences, wrappers and truncation.

    Replaces hand-rolled `split("```json")` which was case-sensitive, mis-sliced
    when a fence appeared inside an evidence quote, and had no fallback at all:
    one unparseable character discarded the entire paid call.
    """
    text = str(response_text or "").strip()
    if not text:
        return []

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text

    for attempt in (candidate, text):
        try:
            parsed = json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            return [entry for entry in parsed if isinstance(entry, dict)]
        if isinstance(parsed, dict):
            # A model that wrapped the array, e.g. {"rankings": [...]}
            for value in parsed.values():
                if isinstance(value, list) and any(isinstance(v, dict) for v in value):
                    return [entry for entry in value if isinstance(entry, dict)]
            return [parsed]

    salvaged = _salvage_json_objects(candidate) or _salvage_json_objects(text)
    if salvaged:
        logger.warning(
            "Judge response did not parse as JSON; salvaged %d complete entries "
            "from it (likely truncated mid-array)", len(salvaged),
        )
    return salvaged


def _matches_other_provider(
    claimed: Any, providers: List[Dict[str, Any]], skip_index: int
) -> bool:
    """Does `claimed` positively name a DIFFERENT provider in this pool?

    A POSITIVE match only — unlike `_same_provider_name`, an unparseable name on
    either side counts as no match. This is the evidence that separates a judge
    mis-binding its answer (drop it) from a judge fumbling the name field (keep
    it): the first is a wrong provider being scored, the second is only a
    cosmetic slip in a field that exists purely to detect the first.
    """
    wanted = normalize_name_tokens(claimed)
    if not wanted:
        return False
    for index, provider in enumerate(providers):
        if index == skip_index:
            continue
        other = normalize_name_tokens(provider.get("name", ""))
        if not other:
            continue
        if len(wanted & other) / max(min(len(wanted), len(other)), 1) >= 0.5:
            return True
    return False

# Upper bound on a judge evidence snippet. The rubric asks for one line, so
# this is a guard against a pathological response, not a formatting budget.
_EVIDENCE_MAX_CHARS = 400

# The judge's rubric, module-level so the CRITIC can be handed the same text.
#
# It was inline in the judge's f-string, and the critic — which audits the
# judge's per-criterion scores — was given only the criterion NAMES and maxima
# plus the phrase "neutral band". So it audited against a standard it had to
# infer. On 2026-07-28 it reported, and the panel published to the patient:
#
#   "practical_access at 2.0 ... is not a neutral-band error, but red_flags at
#    18.0 is generous given the summary cites both incomplete workups and
#    REPEATED DELAYS"
#
# The judge had scored the delays under practical_access (2.0) and correctly
# NOT charged them again under red_flags — which is what this rubric's own
# routing rule requires ("Access complaints ... belong to practical_access —
# do NOT also score them here, or one signal is charged twice"). The critic was
# asking for the double-charge, because it had never been shown the rule.
#
# Same doctrine as DESIGN §10.9's payload parity, applied to the STANDARD
# rather than the evidence: an auditor reading a different standard than the
# agent it audits is not an independent check. The import direction is
# deliberate — the critic reads the judge's rubric, so the two cannot drift.
JUDGE_RUBRIC = """<rubric>
Every band below covers a contiguous range. If your judgment falls between two anchors,
pick the band whose description fits the evidence and score inside it — you should never
need a score that no band describes.
1. review_substance (0-50) — substance and credibility of patient feedback.
   41-50: multiple specific, consistent themes (outcomes, communication, diagnosis accuracy) across meaningful review volume on independent platforms.
   28-40: positive but generic or thin evidence.
   21-27: no review text available — neutral. Never score absence below this band.
   11-20: mixed feedback with substantive concerns.
   0-10: credible pattern of serious complaints.
   Source credibility: when review_source is the provider's OWN practice site (not an
   independent platform like healthgrades/vitals/webmd/zocdoc), the evidence is
   self-published marketing — cap review_substance at the 28-40 band and say so; the top
   band requires independent-platform evidence.
2. red_flags (0-30) — internal consistency and absence of credible red flags.
   28-30: clean and consistent ON SUBSTANTIVE EVIDENCE — there is enough review text that
   a real problem would have surfaced, and none did. The top band is a finding, not a
   default: it requires evidence you could have found a problem in.
   25-27: no evidence either way — too little review text to assess consistency at all.
   Nothing turned up, but nothing was there to turn up. Absence is NOT a red flag and is
   never scored below this band; it simply is not the same claim as "verified clean".
   15-24: minor inconsistencies (high rating but lukewarm text; big claims on sparse data).
   0-14: credible red-flag patterns (billing disputes, misdiagnosis reports) — cite them.
   Strong cross-platform disagreement in review_observations (e.g. 5/5 on one platform,
   1.2/5 on another) is a consistency signal — score it here and cite the platforms; do
   not silently ignore it.
   Access complaints (waits, scheduling, reachability) belong to practical_access — do NOT
   also score them here, or one signal is charged twice.
3. practical_access (0-20) — access signals in the review text and page evidence.
   Access means scheduling, wait times, office reachability, follow-up, and how much
   time the provider gives a visit.
   17-20: concrete access positives cited (easy scheduling, short waits, responsive
   office, accepting new patients) and no friction reported — quote them.
   14-16: mild or incidental access positives ("appointments are not rushed", "willing
   to spend time with patients", "staff got back to me") — cite them.
   12-13: MIXED — the text cites access positives AND access friction. Quote BOTH sides.
   Most real feedback is two-sided; this band exists so you never have to collapse it to
   neutral. If the friction is the dominant theme rather than balanced against the
   positives, score it as friction (4-7 or 0-3) and still quote both.
   8-11: no access signals either way — the text says NOTHING about scheduling, waits,
   reachability, follow-up or visit length. If you can quote anything at all on those
   subjects, you are NOT in this band. Absence of access mentions is NOT an access
   problem; never score absence below this band.
   4-7: isolated access friction — one or two mentions the overall feedback does not treat
   as a pattern — cite them.
   0-3: repeated access complaints (long waits, unreachable office, scheduling chaos) —
   cite them.
   Distance/proximity is scored by the deterministic core with the user's own weights —
   NEVER score it here, in either direction.
</rubric>
"""


# Judge output-token budget, scaled to the pool. See `_judge_token_budget`.
_JUDGE_BASE_TOKENS = 2000
_JUDGE_TOKENS_PER_PROVIDER = 320
_JUDGE_MAX_TOKENS = 16000

# How many concurrent calls score the pool, and the pool size below which the
# split is refused. Scoring 8 providers in one call was 24.4s of a 97.8s run.
#
# Two, and deliberately not a tuning knob for "more": every extra shard
# re-sends the whole rubric (~1.3k tokens) AND narrows the set of providers
# any one call can be internally consistent across. The rubric's bands are
# anchored precisely so scores are absolute rather than relative to the pool —
# that is the argument that makes splitting safe at all — but the anchors are
# prose, and prose calibrates against the examples in front of it.
#
# The floor is the sharp edge of the same concern: at 3 providers a shard
# holds one or two, and a rubric applied to a single provider has nothing in
# its own call to be consistent with. 4 is the smallest pool where both
# shards hold at least two. Below it, one call.
_JUDGE_SHARDS = 2
_MIN_PROVIDERS_TO_SPLIT_JUDGE = 4


def _clip_evidence(value: Any) -> str:
    """Bound a judge evidence snippet without cutting mid-word."""
    return clip_words(value, _EVIDENCE_MAX_CHARS)


def _core_rank_order(scored: List[Dict[str, Any]]) -> List[int]:
    """Indices of `scored` best-first by core score, ties broken deterministically.

    The tie-break is the provider's cache key — stable across runs and
    independent of the order discovery happened to return. Exact ties are
    ordinary (the 2026-07-25 pool had 53/53 and 50/50 adjacent), and this
    ranking decides who gets the enrichment budget, so `sorted`'s input-order
    fallback meant an identical search could research a different set.

    It carries NO claim about which of two tied providers deserves the slot.
    An earlier draft ordered by fewest platform pairs — spend enrichment where
    it can change the answer — but that only holds if "no reviews found" is
    noise rather than signal, which is unmeasured. The opposite reading (a
    rated provider below the line earned its rank; an unrated one is propped up
    by the imputation) is equally defensible on today's evidence. Encoding
    either would repeat the assumption we declined to make in the unrated
    imputation itself.
    """
    return sorted(
        range(len(scored)),
        key=lambda i: (-float(scored[i].get("base_score", 0) or 0),
                       resolve_cache_key(scored[i])),
    )


# The prior every measured rating is shrunk toward, and the score it maps to.
# Stated once so the unrated imputation is DERIVED from the prior rather than
# chosen independently — the same discipline as "same_city 82 == a verified
# ~9 mi" and "EXPERIENCE_UNKNOWN_SCORE == a verified 10 years".
RATING_PRIOR = 3.5
RATING_UNKNOWN_SCORE = (RATING_PRIOR / 5.0) * 100  # 70.0


def calculate_rating_score_with_confidence(rating: float, review_count: Optional[int] = None) -> Dict[str, Any]:
    """Calculate rating score with confidence adjustment using Bayesian average.

    Low review counts are pulled toward `RATING_PRIOR` to avoid over-weighting
    ratings with insufficient sample size.

    Args:
        rating: Provider rating (0-5)
        review_count: Number of reviews (None if not available)

    Returns:
        Dictionary with adjusted score, confidence level, and reliability
    """
    if rating <= 0:
        # Unrated is "insufficient evidence", not "worst possible": use the
        # imputation band from interpret_rating_status (the prior when unrated,
        # 50 on a data anomaly) instead of zeroing new / low-web-presence
        # providers.
        status = interpret_rating_status(rating, review_count)
        return {
            'score': status['base_score'],
            'confidence': 'no_rating',
            'adjusted_rating': 0,
            'original_rating': 0,
            'review_count': review_count or 0,
            'reliability': 'unknown'
        }

    # Default review count if missing
    if review_count is None:
        review_count = 15  # Assume moderate confidence for verified providers
        confidence_level = 'medium_assumed'
    elif review_count < 5:
        confidence_level = 'low'
    elif review_count < 20:
        confidence_level = 'medium'
    else:
        confidence_level = 'high'

    # Bayesian average calculation
    global_avg_rating = 3.5  # Conservative global average
    confidence_weight = 10   # Weight of prior belief

    # Calculate weighted rating (more reviews = closer to actual rating, fewer = closer to global avg)
    adjusted_rating = (
        (confidence_weight * global_avg_rating + review_count * rating) /
        (confidence_weight + review_count)
    )

    # Convert to 0-100 score
    score = (adjusted_rating / 5.0) * 100

    return {
        'score': round(score, 2),
        'confidence': confidence_level,
        'adjusted_rating': round(adjusted_rating, 2),
        'original_rating': rating,
        'review_count': review_count,
        'reliability': 'high' if review_count >= 20 else 'moderate' if review_count >= 5 else 'low'
    }


def calculate_missing_data_score(preference_weight: float, default_score: int = 50) -> int:
    """Calculate score for missing data based on user preference weight.

    When data is missing, penalize more heavily if user cares about that factor.

    Args:
        preference_weight: How important this factor is to user (0.0-1.0)
        default_score: Default neutral score (typically 50)

    Returns:
        Adjusted score for missing data
    """
    if preference_weight >= 0.5:
        # User prioritizes this factor - significant penalty
        return int(default_score * 0.4)  # 40% of default
    elif preference_weight >= 0.3:
        # Medium priority - moderate penalty
        return int(default_score * 0.7)  # 70% of default
    else:
        # Low priority - minimal penalty
        return default_score


def interpret_rating_status(rating: float, review_count: Optional[int]) -> Dict[str, Any]:
    """Distinguish between no rating vs poor rating.

    Args:
        rating: Provider rating (0-5)
        review_count: Number of reviews

    Returns:
        Dictionary with status, display text, scoring approach, and warning
    """
    if rating == 0 and (review_count is None or review_count == 0):
        # Imputed at the PRIOR, stated as its measured equivalence:
        # RATING_UNKNOWN_SCORE == a verified 3.5 stars.
        #
        # The old value, 40, was equivalent to a measured 2.0 stars — BELOW the
        # 2.5 this same function calls 'poor_quality' twenty lines down. A
        # provider nobody had looked at scored worse than one measured as bad,
        # and the comment "slightly below neutral" described a 30-point penalty.
        #
        # That penalty was self-fulfilling: the rating dimension is ~1/3 of the
        # core score, the core ranking decides who gets enriched, and enrichment
        # is what produces ratings. An unrated provider sorted low, the low tail
        # was cut, and the cut was the thing that would have rated them. Same
        # defect round 7 fixed for tenure (45 == a measured 2.5 years); rating
        # was never swept.
        #
        # The prior is neutral by construction — a measured 4.5 still beats it,
        # so supplying data remains the way to score well. It does not assert
        # the provider is average; it asserts we do not know, and declines to
        # charge them for our own extraction coverage.
        return {
            'status': 'unrated',
            'display': 'No reviews yet',
            'score_approach': 'prior',
            'base_score': RATING_UNKNOWN_SCORE,
            'warning': 'New provider or limited online presence'
        }
    elif rating == 0 and review_count and review_count > 0:
        # Unusual - might be data error
        return {
            'status': 'data_anomaly',
            'display': 'Rating unavailable',
            'score_approach': 'neutral',
            'base_score': 50,
            'warning': 'Rating data may be incomplete'
        }
    elif 0 < rating < 2.5 and review_count and review_count >= 5:
        return {
            'status': 'poor_quality',
            'display': f'{rating}/5.0 (⚠️ Low)',
            'score_approach': 'penalize',
            'base_score': (rating / 5.0) * 100,
            'warning': 'Below average patient ratings'
        }
    else:
        return {
            'status': 'valid_rating',
            'display': f'{rating}/5.0',
            'score_approach': 'normal',
            'base_score': (rating / 5.0) * 100,
            'warning': None
        }


# Imputation for unknown tenure, stated as its measured equivalence the way
# the location tiers are ("same_city 82 == a verified ~9 mi").
#
# EXPERIENCE_UNKNOWN_SCORE == a verified 10 years.
#
# The old value, 45, was equivalent to a measured 2.5 YEARS — so a provider
# whose tenure our extractor simply failed to find was scored as more junior
# than almost any practising physician, and at High weight that cost ~8.6 core
# points against a 26-year provider. Years are read from excerpt anchors
# hitting a profile header, so absence reflects OUR extraction coverage, not
# the provider's career: that is a data-availability penalty wearing a quality
# signal's clothing, which is exactly the calibration location's tiers were
# given and experience never was.
#
# 10 years is the pessimistic edge, not the median: a provider with a MEASURED
# 10+ years still beats the imputation, so supplying data remains the way to
# score well. Providers measured below it lose to an unknown, which is the same
# trade the location tiers already make.
# Uncertainty added to a city-centroid distance before scoring — roughly a
# small city's radius. Any positive margin preserves the invariant (measured
# beats estimated at the same nominal distance); this magnitude keeps the
# penalty proportionate rather than punitive.
CITY_CENTROID_MARGIN_MILES = 3.0

EXPERIENCE_UNKNOWN_SCORE = 70.0
EXPERIENCE_UNKNOWN_EQUIV_YEARS = 10

# The curve flattens after the knee and stops at the cap.
#
# Experience is the only dimension with NO shrinkage — a free monotone climb —
# while every rating is pulled toward RATING_PRIOR by its own review volume. On
# the 2026-07-25 run that let the top provider score 96 on tenure against 73 on
# rating, and the critic's bias analysis flagged it twice: an UNVERIFIED career
# length was out-swinging measured patient experience.
#
# Two properties are deliberate:
#
#   * EXPERIENCE_CAP sits BELOW what a strong measured rating reaches (4.75 with
#     100 reviews scores ~93), so tenure can no longer beat reviews outright.
#     Under the old cap of 100 it always could.
#   * The knee is at 15 years because the marginal year stops meaning much
#     there. The old linear curve put a 20-year and a 35-year physician 30
#     points apart — a precision the underlying evidence (a scraped profile
#     header) cannot support.
#
# The knee is placed ABOVE EXPERIENCE_UNKNOWN_EQUIV_YEARS so the round-7
# equivalence survives unchanged: EXPERIENCE_UNKNOWN_SCORE is still exactly a
# verified 10 years. Changing the curve without preserving that would turn a
# documented imputation into an undocumented one.
#
# ROUND 16 — the SPREAD, not the cap. Round 14 stopped tenure beating reviews
# outright; it did not stop tenure OUT-SWINGING them at equal weight, and the
# critic's bias analysis named that in three consecutive live runs, the last
# one unprompted and in patient-facing copy ("the sixth-ranked doctor has
# strong patient feedback but ranks lower largely because she has fewer years
# in practice").
#
# Equal WEIGHTS do not produce equal INFLUENCE. What matters is the span each
# dimension actually realizes across a real pool, and rating is compressed
# twice — Bayesian shrinkage toward RATING_PRIOR, and real doctors genuinely
# cluster 4.0-5.0 — while experience was an unshrunk linear ramp on a single
# scraped integer:
#
#     realistic input range        old span      new span     rating's span
#     10 -> 30 years               60 -> 85      70 -> 85
#                                    (25)          (15)
#     4.0 -> 4.5 stars                                        78.6 -> 87.1
#                                                                (8.5)
#
# So experience carried ~3x rating's leverage and now carries ~1.8x. NOT parity:
# the remaining gap is honest, because a 30-year career IS more informative than
# a half-star. Parity would need shrinkage, which needs a prior over career
# length nobody here has measured — a bigger claim than the evidence supports.
#
# The floor moves 40 -> 55 and the ramp flattens (2.0/1.0 -> 1.5/0.5 per year).
# Both round-14 properties survive: the cap still sits below what a strong
# measured rating reaches, and the knee still sits above
# EXPERIENCE_UNKNOWN_EQUIV_YEARS so EXPERIENCE_UNKNOWN_SCORE is still EXACTLY a
# verified 10 years — which is why that constant moved 60 -> 70 in the same
# edit. It is derived, not chosen; changing the curve and not the imputation
# would silently turn a documented equivalence into an undocumented one.
#
#     years   0    5    10   15   20   26   30+
#     old     40   50   60   70   75   81   85
#     new     55   62.5 70   77.5 80   83   85
EXPERIENCE_FLOOR_SCORE = 55.0
EXPERIENCE_KNEE_YEARS = 15.0
EXPERIENCE_KNEE_RATE = 1.5
EXPERIENCE_POST_KNEE_RATE = 0.5
EXPERIENCE_CAP = 85.0


def calculate_experience_score(years_experience: Any) -> Dict[str, Any]:
    """Experience component: rewards long careers, neutral when unknown.

    Known years: `EXPERIENCE_KNEE_RATE` points per year from
    `EXPERIENCE_FLOOR_SCORE` to `EXPERIENCE_KNEE_YEARS`, then
    `EXPERIENCE_POST_KNEE_RATE`, capped at `EXPERIENCE_CAP` (5y -> 62.5,
    10y -> 70, 15y -> 77.5, 20y -> 80, 30y+ -> 85). Missing/unparseable:
    `EXPERIENCE_UNKNOWN_SCORE` — a provider is never penalized as a novice for
    data the scraper didn't find.

    The ramp is deliberately flat: see the constants for the three runs of
    field evidence that a steeper one out-swings measured patient ratings at
    equal weight.
    """
    try:
        years = float(years_experience)
    except (TypeError, ValueError):
        years = None

    if years is None or years < 0:
        return {
            "score": EXPERIENCE_UNKNOWN_SCORE,
            "years": None,
            "data_quality": "missing",
            "warning": (
                "Years of experience not found — scored as an unknown "
                f"(equivalent to {EXPERIENCE_UNKNOWN_EQUIV_YEARS} years), not as a new provider"
            ),
        }

    if years <= EXPERIENCE_KNEE_YEARS:
        raw = EXPERIENCE_FLOOR_SCORE + years * EXPERIENCE_KNEE_RATE
    else:
        raw = (
            EXPERIENCE_FLOOR_SCORE
            + EXPERIENCE_KNEE_YEARS * EXPERIENCE_KNEE_RATE
            + (years - EXPERIENCE_KNEE_YEARS) * EXPERIENCE_POST_KNEE_RATE
        )

    return {
        "score": round(min(EXPERIENCE_CAP, raw), 1),
        "years": years,
        "data_quality": "known",
        "warning": None,
    }


class PreferenceScorerAgent:
    """Agent responsible for scoring and ranking healthcare providers based on user preferences."""

    def __init__(self):
        """Initialize the preference scorer with OpenAI client."""
        self.config = get_config()
        self.openai_client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            if not self.config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found in configuration")

            self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            logger.info("Preference scorer client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize preference scorer client: {e}")
            raise

    def _calculate_base_scores(self, providers: List[Dict[str, Any]], preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate base scores for providers using weighted algorithm.

        Args:
            providers: List of provider dictionaries
            preferences: User preference weights

        Returns:
            List of providers with base scores
        """
        scored_providers = []

        for provider in providers:
            score = 0.0
            score_breakdown = {}
            data_quality_flags = {}
            data_warnings = []

            # Rating score (0-100) with confidence adjustment. The score
            # hears ALL platforms when a cross-platform blend exists
            # (count-weighted by the gatherer; a vitals 3.5(16) beside a
            # healthgrades 2.1(13) scores as ~2.9, not the headline 3.5) —
            # the card keeps showing the traceable headline either way.
            if provider.get("blended_rating") is not None:
                rating = float(provider["blended_rating"])
                review_count = provider.get("blended_review_count")
                rating_basis = "cross_platform_blend"
            else:
                rating = float(provider.get("rating", 0))
                review_count = provider.get("review_count", None)
                rating_basis = "headline"

            # Check rating status first
            rating_status = interpret_rating_status(rating, review_count)
            if rating_status['warning']:
                data_warnings.append(rating_status['warning'])

            # Calculate confidence-adjusted rating score
            rating_result = calculate_rating_score_with_confidence(rating, review_count)
            rating_score = rating_result['score']

            score_breakdown["rating"] = {
                "value": rating,
                "review_count": review_count,
                "adjusted_rating": rating_result.get('adjusted_rating', rating),
                "score": rating_score,
                "weight": preferences.get("rating_weight", 0.3),
                "confidence": rating_result.get('confidence', 'unknown'),
                "reliability": rating_result.get('reliability', 'unknown'),
                "basis": rating_basis,
            }
            if rating_basis == "cross_platform_blend":
                score_breakdown["rating"]["platforms"] = provider.get("blended_platform_count")
            score += rating_score * preferences.get("rating_weight", 0.3)
            data_quality_flags['rating'] = rating_result.get('confidence', 'unknown')

            # Location score (0-100), best evidence wins. Precedence:
            #   1. computed_distance_miles — ZIP/city-centroid haversine from
            #      utils/geo.py, attached by the gatherer (never LLM-guessed)
            #   2. distance — explicitly stated on a page (rare)
            #   3. location_match tier — textual same-zip/city/state fallback
            #   4. missing-data path (weight-sensitive neutral)
            # Falloff reuses DEFAULT_SEARCH_RADIUS: score hits 0 at twice the
            # radius (2×25 = 50 mi by default, matching the old hardcoded /50).
            falloff_miles = 2 * self.config.DEFAULT_SEARCH_RADIUS
            # Tier scores are IMPUTATIONS (no measured distance), so each sits
            # at the pessimistic edge of its plausible band — a provider we've
            # actually measured must never be out-scored by one we've only
            # tiered. At radius 25: same_city 82 ≡ a verified ~9 mi; same_zip
            # 90 ≡ a verified ~5 mi. (Were same_city 90, a city-only address
            # would tie a genuinely-close measured provider — the inversion
            # that let unverified locations ride at the top.)
            tier_scores = {"same_zip": 90, "same_city": 82, "same_state": 55, "different": 25}
            # A city-centroid distance is an IMPUTATION too — every provider in
            # that city shares one coordinate — so it gets the same pessimistic
            # treatment as a tier: a margin roughly a small city's radius, which
            # guarantees a ZIP-MEASURED provider always out-scores a
            # city-ESTIMATED one at the same nominal distance. It does not
            # manufacture differentiation between providers in the same city:
            # at city precision we genuinely cannot tell them apart, and
            # inventing a spread would be the same overstatement as labelling
            # the estimate "computed straight-line".
            city_margin = CITY_CENTROID_MARGIN_MILES

            computed_distance = provider.get("computed_distance_miles")
            stated_distance = provider.get("distance", None)
            location_value = None

            if isinstance(computed_distance, (int, float)):
                if provider.get("distance_precision") == "city":
                    effective_miles = float(computed_distance) + city_margin
                    location_basis = "city_estimate"
                    data_quality_flags['distance'] = 'derived'
                else:
                    effective_miles = float(computed_distance)
                    location_basis = "computed_distance"
                    data_quality_flags['distance'] = 'complete'
                distance_score = max(0, 100 - (effective_miles / falloff_miles * 100))
                location_value = computed_distance
            else:
                try:
                    dist_value = float(str(stated_distance).replace("mi", "").replace("miles", "").strip())
                except (TypeError, ValueError):
                    dist_value = None
                if stated_distance and dist_value is not None:
                    distance_score = max(0, 100 - (dist_value / falloff_miles * 100))
                    location_basis = "stated_distance"
                    data_quality_flags['distance'] = 'complete'
                    location_value = stated_distance
                elif provider.get("location_match") in tier_scores:
                    location_basis = provider["location_match"]
                    distance_score = tier_scores[location_basis]
                    data_quality_flags['distance'] = 'derived'
                    location_value = location_basis
                else:
                    # Nothing to go on - penalize based on how important location is to user
                    distance_score = calculate_missing_data_score(
                        preferences.get("location_weight", 0.4), 50
                    )
                    location_basis = "missing"
                    data_quality_flags['distance'] = 'missing'

            score_breakdown["location"] = {
                "value": location_value,
                "score": distance_score,
                "weight": preferences.get("location_weight", 0.4),
                "basis": location_basis,
                "data_quality": data_quality_flags['distance']
            }
            score += distance_score * preferences.get("location_weight", 0.4)

            # Experience (0-100, user-weighted; neutral when unknown).
            # Insurance is deliberately NOT scored here: scraped acceptance
            # lists cannot be validated, so insurance is surfaced as labeled
            # evidence (chips + AI rubric) and verified via the FHIR
            # network-check prototype instead of moving rankings.
            experience_result = calculate_experience_score(provider.get("years_experience"))
            experience_score = experience_result["score"]
            if experience_result["warning"]:
                data_warnings.append(experience_result["warning"])

            score_breakdown["experience"] = {
                "value": experience_result["years"],
                "score": experience_score,
                "weight": preferences.get("experience_weight", 0.3),
                "data_quality": experience_result["data_quality"]
            }
            score += experience_score * preferences.get("experience_weight", 0.3)
            data_quality_flags['experience'] = experience_result["data_quality"]

            provider_copy = provider.copy()
            provider_copy["base_score"] = round(score, 2)
            provider_copy["score_breakdown"] = score_breakdown
            provider_copy["data_quality_flags"] = data_quality_flags
            provider_copy["data_warnings"] = data_warnings
            provider_copy["rating_confidence"] = rating_result
            scored_providers.append(provider_copy)

        return scored_providers

    def _generate_ai_rankings(self, providers: List[Dict[str, Any]], preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Score providers with a rubric-based LLM judge (`JUDGE_MODEL`).

        The judge reads the evidence the algorithm can't — review text and
        cross-platform observations — against THREE anchored criteria
        (review_substance /50, red_flags /30, practical_access /20) and returns
        cited subscores summing to a cardinal ai_score (0-100), the 30% side of
        the composite.

        Each criterion's bands TILE its range: every integer 0..max falls in
        exactly one, guarded by `test_every_score_lands_on_exactly_one_band`.
        A gap is where the judge improvises — `practical_access` had none for
        "isolated access friction", so a provider whose summary named wait times
        was parked at the neutral 10 and the scoring rules then made him deny
        evidence he had quoted under red_flags on the same card.

        Round 11 closed the NUMERIC gaps; the 2026-07-28 run showed the
        remaining one was SEMANTIC. The bands are a single positive→negative
        ladder, so two-sided evidence had no rung, and the neutral band absorbed
        every case the ladder could not describe — two providers landed on
        exactly 10 for opposite reasons. One had only a mild positive
        ("willingness to spend time with patients", which is the 14-16 anchor in
        different words); the other had a real positive AND a real friction
        ("timely appointments" alongside "difficulty reaching the office via
        phone"). The third, whose evidence was purely negative, scored 2 —
        correctly. The ladder works when evidence is one-signed and collapses to
        neutral when it is not.

        So `practical_access` gains a MIXED band and the neutral band is now
        unreachable with a citation: it says outright that quoting anything on
        the subject disqualifies it. The five anchors round 11 calibrated
        against `logs/audit.log` (3, 5, 10, 14, 17) all keep the band that
        described them — the mixed band was placed at 12-13 rather than
        straddling neutral precisely so that calibration survives.

        (The docstring previously described four criteria at 40/25/20/15, a
        GPT-5-mini judge, and inputs — free-text requirements, insurance — that
        were removed on 2026-07-21. None of that had been true for months.)

        Args:
            providers: List of scored provider dictionaries
            preferences: User preferences

        Returns:
            List of providers with ai_score, ai_rubric, ai_evidence, reasoning
        """
        try:
            # Prepare provider data for the rubric judge. Deliberately absent:
            # base_score (anchoring — the judge must not see the deterministic
            # score), network_verified (verification is display-only, never
            # scored), and insurance fields (coverage belongs to the FHIR
            # check, not the ranking).
            provider_summaries = []
            for i, provider in enumerate(providers):
                summary = {
                    # Named `provider_index`, matching the key the output
                    # format asks for, so the model echoes back the field it
                    # was given. It was `index` while the output format said
                    # `provider_index` and the list is SHUFFLED below, so a
                    # model returning its position in the shuffled array — the
                    # natural reading — silently attached every rubric score,
                    # citation and patient-facing sentence to a different
                    # doctor, with every value passing the range check.
                    "provider_index": i,
                    "name": provider.get("name", "Unknown"),
                    "specialty": provider.get("specialty", ""),
                    "location": provider.get("location", ""),
                    "rating": provider.get("rating", 0),
                    "review_count": provider.get("review_count"),
                    "blended_rating": (
                        f"{provider['blended_rating']}/5 across "
                        f"{provider.get('blended_review_count')} reviews on "
                        f"{provider.get('blended_platform_count')} platforms"
                        if provider.get("blended_rating") is not None else "n/a"
                    ),
                    "review_sentiment": provider.get("review_sentiment", "unknown"),
                    # The WHOLE summary, bounded only against a pathological
                    # response. A flat [:400] cut here severed every real
                    # summary at ~55% — and not randomly: the gatherer prompt
                    # asks for "most praised aspects, common complaints, and
                    # overall experience themes" IN THAT ORDER, so the caveats
                    # always live in the tail that a head-cut discards. That
                    # silently biased red_flags and practical_access (50 of the
                    # judge's 100 points, and the two criteria whose evidence
                    # sits precisely there) upward, and had the judge writing
                    # "the summary begins to mention practice-level complaints
                    # without providing their details" — the truncation
                    # describing itself — straight into patient-facing copy.
                    # Same helper and bound as the critic: see utils/excerpt.
                    "review_summary": clip_words(
                        provider.get("review_summary") or "No reviews available",
                        SUMMARY_MAX_CHARS,
                    ),
                    "review_source": source_domain(provider.get("review_source_url")) or "unknown",
                    "review_observations": " · ".join(
                        f"{source_domain(obs.get('source_url')) or 'unknown'} "
                        f"{obs['rating']}/5" + (f" ({obs['review_count']})" if obs.get("review_count") else "")
                        for obs in (provider.get("review_observations") or [])[:5]
                        if obs.get("rating") is not None
                    ) or "none",
                    # `or` would discard the single best value this field can
                    # hold: a computed 0.0 miles (provider in the user's own
                    # ZIP) is falsy, so the closest provider in the pool was
                    # the one reported to the judge as "N/A". The critic's
                    # `_location_evidence` already tests this with
                    # `isinstance(computed, (int, float))`; match it.
                    "distance": (
                        provider["computed_distance_miles"]
                        if isinstance(provider.get("computed_distance_miles"), (int, float))
                        else provider.get("distance", "N/A")
                    ),
                    "years_experience": provider.get("years_experience", "N/A"),
                    "data_warnings": provider.get("data_warnings", []),
                }
                provider_summaries.append(summary)

            # Fan out. `provider_index` is the GLOBAL index built above, so a
            # shard carries its providers' real positions and every guard in
            # `_apply_judge_rankings` keeps working unchanged.
            #
            # Round-robin, not contiguous: `providers` arrives in core-rank
            # order (the caller pinned it before enrichment), so contiguous
            # halves would give one call only strong providers and the other
            # only weak ones. The rubric is anchored, which is the argument for
            # splitting at all — but a shard containing no strong provider has
            # nothing to calibrate the top of the scale against, and that is
            # precisely the divergence this split risks.
            shards = (
                round_robin_shards(provider_summaries, _JUDGE_SHARDS)
                if self._should_split_judge(len(provider_summaries))
                else [provider_summaries]
            )

            ranked_providers = providers.copy()
            seen_indices: set = set()

            if len(shards) <= 1:
                entries = self._judge_shard(provider_summaries)
                seen_indices |= self._apply_judge_rankings(ranked_providers, entries, None)
            else:
                logger.info(
                    "Rubric judge split across %d concurrent calls (%s providers each)",
                    len(shards), "/".join(str(len(s)) for s in shards),
                )
                with ThreadPoolExecutor(max_workers=len(shards)) as executor:
                    futures = [executor.submit(self._judge_shard, shard) for shard in shards]
                    entries_per_shard = [future.result() for future in futures]
                for shard, entries in zip(shards, entries_per_shard):
                    # An entry claiming an index this shard was never shown is
                    # a mis-binding, not a slip: the model can only have
                    # invented it. Without this the two calls share one index
                    # space, so shard B echoing index 0 would overwrite the
                    # rubric shard A wrote for a provider B never read.
                    allowed = {s["provider_index"] for s in shard}
                    seen_indices |= self._apply_judge_rankings(
                        ranked_providers, entries, allowed
                    )

            # Providers the judge skipped score neutral, never punitive
            for provider in ranked_providers:
                provider.setdefault("ai_score", 50.0)

            # Report what was APPLIED, not the input length. The old line
            # printed the pool size even when every entry had been dropped,
            # so "Rubric judging completed for 16 providers" was the log a
            # total judge failure produced.
            applied = len(seen_indices)
            if applied < len(ranked_providers):
                logger.warning(
                    "Rubric judging applied %d of %d providers — the rest fall back "
                    "to the neutral ai_score 50", applied, len(ranked_providers),
                )
            else:
                logger.info("Rubric judging applied to all %d providers", applied)
            return ranked_providers

        except Exception as e:
            logger.error(f"AI ranking failed: {e}", exc_info=True)
            return providers

    def _should_split_judge(self, provider_count: int) -> bool:
        """Whether to score this pool in `_JUDGE_SHARDS` concurrent calls.

        Off by config (`JUDGE_PARALLEL_ENABLED=false`) reverts to the single
        call with no other change — the escape hatch this split shipped with,
        because two calls CAN calibrate differently and the only way to find
        out is a live run.

        The floor is not a tuning knob. Below it a shard holds one or two
        providers, and a rubric applied to a single provider has no other
        provider in the same call to be consistent with — the failure mode
        the round-robin partition exists to avoid, in its sharpest form.
        """
        return (
            self.config.JUDGE_PARALLEL_ENABLED
            and provider_count >= _MIN_PROVIDERS_TO_SPLIT_JUDGE
        )

    def _judge_shard(self, provider_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run ONE judge call over these provider records; return its entries.

        Returns [] on any failure — no content, unparseable, or an exception —
        so a failed shard costs its own providers a rubric (they fall back to
        the neutral 50) and never the whole pool's.
        """
        try:
            # Shuffle presentation order: LLM judges favor early positions
            shuffled_summaries = provider_summaries[:]
            random.shuffle(shuffled_summaries)

            prompt = f"""You are a healthcare-provider evaluation judge. Score every provider against the rubric below using ONLY the evidence provided. Quote evidence — never invent it.

{JUDGE_RUBRIC}
<scoring_rules>
- Use the full range the anchors allow; do not cluster scores.
- Treat the star rating as low-confidence when review_count is null or small.
- ai_score must equal the sum of the three criterion scores.
- Give a one-line quoted evidence snippet per criterion. Write "no evidence" ONLY when the
  text contains nothing at all bearing on that criterion — never merely because you landed
  in a neutral band. Whether evidence exists and which band applies are separate questions.
- If you cite a fact under one criterion you must not deny it under another. Quoting a
  wait-time complaint under red_flags and writing "no evidence" for practical_access is a
  contradiction, and it is shown to the patient on the same card.
- "reasoning" is shown directly to the patient: write it in plain, friendly language.
  NEVER use internal criterion names (review_substance, red_flags, practical_access),
  snake_case, or scoring jargon there — say "detailed patient reviews" instead of "high
  review_substance score".
- Copy each provider's own "provider_index" value into your entry for that provider.
  The list below is in RANDOM order, so the index is NOT the position in the array —
  read it from the provider's own record. Every index must appear exactly once.
- Copy that same provider's "name" value into "provider_name", exactly as written.
  It is a cross-check on the index; do not paraphrase it and do not invent one.
- Keep evidence snippets to ONE short sentence each. Long citations crowd out later
  providers in the response.
- Do NOT include any explanatory text, ONLY return the JSON array.
</scoring_rules>

<providers_data>
{json.dumps(shuffled_summaries, indent=2)}
</providers_data>

<output_format>
Return ONLY a JSON array with one entry per provider:
[
  {{
    "provider_index": 0,
    "provider_name": "Dr. Jane Smith, MD",
    "scores": {{"review_substance": 42, "red_flags": 26, "practical_access": 10}},
    "evidence": {{"review_substance": "...", "red_flags": "...", "practical_access": "..."}},
    "ai_score": 78,
    "reasoning": "2-3 sentences summarizing the judgment",
    "strengths": ["..."],
    "concerns": ["..."]
  }}
]
</output_format>"""

            # OpenAI reasoning models: max_completion_tokens replaces
            # max_tokens (reasoning tokens count against it) and non-default
            # temperature is rejected. reasoning_effort "low" is the lowest
            # tier valid on BOTH the GPT-5 scale (minimal/low/medium/high)
            # and the GPT-5.6 scale (none/low/medium/high/xhigh) — "minimal"
            # is a 400 on gpt-5.6-terra, verified live.
            # The budget MUST scale with the pool. This was a flat 4000 while the
            # judge scores every provider (up to MAX_PROVIDERS_PER_SEARCH, 20) —
            # and on a reasoning model the reasoning tokens come out of the same
            # allowance. A pool of 10 fitted; a pool of 16 did not, and the
            # overflow was invisible: truncated JSON fails to parse, the handler
            # returns the providers untouched, and every card silently falls back
            # to the neutral ai_score 50 with no judge section at all.
            #
            # Scaled to THIS SHARD, not the pool: the ceiling has to cover the
            # entries one call actually returns. Splitting therefore raises the
            # total allowance (4/4 gives 3280 twice against 4560 once), which
            # is a ceiling and not a spend — and it makes the truncation this
            # comment describes strictly less likely per call.
            budget = _judge_token_budget(len(provider_summaries))
            llm_started = time.perf_counter()
            response = self.openai_client.chat.completions.create(
                model=self.config.JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a rigorous healthcare-provider evaluation judge. Score strictly against the given rubric with cited evidence."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=budget,
                reasoning_effort="low"
            )
            in_tokens, out_tokens = safe_usage(response)
            get_cost_tracker().record_llm(
                self.config.JUDGE_MODEL, in_tokens, out_tokens,
                agent="preference_scorer", duration_s=time.perf_counter() - llm_started
            )

            # Nothing in this codebase checked finish_reason, so a response cut
            # off mid-array looked identical to a malformed one — and neither
            # named the real cause. Say it plainly, with the number that has to
            # change.
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length":
                logger.error(
                    "Judge response hit the %s-token ceiling for %d providers and was "
                    "TRUNCATED. Whatever survives is salvaged below; raise the budget "
                    "in _judge_token_budget if this recurs.",
                    budget, len(provider_summaries),
                )

            # `.strip()` used to sit outside the try, so a None content (the
            # reasoning budget consumed everything) raised AttributeError into
            # the outer handler and reported itself as a generic failure.
            raw_content = getattr(response.choices[0].message, "content", None)
            if not raw_content:
                logger.error(
                    "Judge returned no content (finish_reason=%s, budget=%s, "
                    "%d providers); every provider will fall back to the neutral "
                    "ai_score 50", finish_reason, budget, len(provider_summaries),
                )
                return []
            response_text = raw_content.strip()

            # Parse AI response
            try:
                # Extract JSON from potential markdown code blocks
                ai_rankings = _parse_ranking_response(response_text)
                if not ai_rankings:
                    logger.error(
                        "Judge response yielded no usable entries for %d providers; "
                        "all will fall back to the neutral ai_score 50. Response "
                        "began: %s", len(provider_summaries), response_text[:300],
                    )
                return ai_rankings

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI ranking response: {e}")
                logger.debug(f"AI response was: {response_text}")
                return []

        except Exception as e:
            logger.error(f"AI judge call failed: {e}", exc_info=True)
            return []

    def _apply_judge_rankings(
        self,
        ranked_providers: List[Dict[str, Any]],
        ai_rankings: List[Dict[str, Any]],
        allowed_indices: Optional[set],
    ) -> set:
        """Bind one call's entries onto the pool; return the indices claimed.

        `allowed_indices` is the set of `provider_index` values the call that
        produced these entries was actually shown — None when the whole pool
        went to one call. See the mis-binding note at the fan-out.
        """
        if not ai_rankings:
            return set()

        # Apply rubric scores to providers. The itemized subscores are
        # authoritative: each is coerced and clamped to its criterion
        # cap, and ai_score is their sum (the model's own total is
        # ignored if it disagrees).
        criteria_max = {
            "review_substance": 50.0,
            "red_flags": 30.0,
            "practical_access": 20.0,
        }
        # The presented order is shuffled, so a mis-echoed index binds
        # a provider's rubric, citations and patient-facing reasoning
        # to a DIFFERENT doctor — and every value still passes the
        # range check, so nothing downstream can notice. Two guards,
        # because the prompt alone is not one: an index may be claimed
        # once, and the echoed name must be the name at that index.
        seen_indices: set = set()
        for ranking in ai_rankings:
            provider_idx = ranking.get("provider_index")
            if isinstance(provider_idx, str) and provider_idx.strip().isdigit():
                # A stringified index is a formatting slip, not a
                # mis-binding — coerce rather than discard a paid answer
                provider_idx = int(provider_idx.strip())
            if not isinstance(provider_idx, int) or not (0 <= provider_idx < len(ranked_providers)):
                # This branch used to be the ONLY failure mode that left
                # no trace at all: a missing or malformed index dropped
                # every entry in total silence.
                logger.warning(
                    "Judge entry has an unusable provider_index %r (pool size "
                    "%d); dropping it", ranking.get("provider_index"),
                    len(ranked_providers),
                )
                continue
            if allowed_indices is not None and provider_idx not in allowed_indices:
                # In range for the POOL but not shown to THIS call — so the
                # model cannot have read that provider's record and the entry
                # is invented. Range alone stopped being sufficient the moment
                # the pool was split across calls sharing one index space.
                logger.warning(
                    "Judge entry claims provider_index %s, which this shard was "
                    "never shown (it held %s); dropping it rather than "
                    "overwriting another shard's verdict",
                    provider_idx, sorted(allowed_indices),
                )
                continue
            if provider_idx in seen_indices:
                logger.warning(
                    "Judge returned provider_index %s twice; ignoring the "
                    "duplicate (the response is not a permutation)", provider_idx
                )
                continue

            claimed = ranking.get("provider_name")
            if claimed:
                actual = ranked_providers[provider_idx].get("name", "")
                if not _same_provider_name(claimed, actual):
                    # Drop ONLY on evidence of mis-binding — a claimed
                    # name that belongs to a DIFFERENT provider in this
                    # pool. A name matching nobody is a formatting
                    # failure, not a mis-bind, and dropping it loses a
                    # paid answer for no safety gain: the first version
                    # of this guard did exactly that, and because the
                    # output_format's example value was the
                    # self-describing string "copy the name from that
                    # provider's record verbatim", a model echoing the
                    # placeholder silently cost every provider its
                    # rubric, evidence and reasoning.
                    if _matches_other_provider(claimed, ranked_providers, provider_idx):
                        logger.warning(
                            "Judge entry for index %s names %r, which is a "
                            "DIFFERENT provider in this pool (index %s is %r); "
                            "dropping it rather than scoring the wrong provider",
                            provider_idx, claimed, provider_idx, actual,
                        )
                        continue
                    logger.warning(
                        "Judge entry for index %s carries an unrecognized "
                        "provider_name %r (index %s is %r); keeping the entry — "
                        "it names no other provider, so this is a formatting "
                        "slip rather than a mis-binding",
                        provider_idx, claimed, provider_idx, actual,
                    )
            seen_indices.add(provider_idx)

            raw_scores = ranking.get("scores") or {}
            subscores = {}
            total = 0.0
            for criterion, cap in criteria_max.items():
                try:
                    value = float(raw_scores.get(criterion, 0))
                except (TypeError, ValueError):
                    value = 0.0
                value = max(0.0, min(value, cap))
                subscores[criterion] = round(value, 1)
                total += value

            # The rubric asks for a one-line quoted snippet, so 200
            # chars cut real citations mid-word (and mid-quotation,
            # leaving an unbalanced " on the card). Bound generously
            # and break on a word boundary — this is the last place
            # the full text exists, so the UI cannot undo a bad cut.
            evidence = ranking.get("evidence")
            safe_evidence = (
                {str(k): _clip_evidence(v) for k, v in evidence.items()}
                if isinstance(evidence, dict) else {}
            )

            ranked_providers[provider_idx].update({
                "ai_score": round(min(total, 100.0), 1),
                "ai_rubric": subscores,
                "ai_evidence": safe_evidence,
                "ai_reasoning": ranking.get("reasoning", "No reasoning provided"),
                "ai_strengths": ranking.get("strengths", []),
                "ai_concerns": ranking.get("concerns", [])
            })

        return seen_indices

    def score_core(self, providers: List[Dict[str, Any]], preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank providers by the deterministic core score only — no LLM call.

        Returns the ORIGINAL provider dicts (not copies) sorted best-first, so
        the orchestrator can direct the review-enrichment budget at the likely
        top of the ranking and any enrichment mutates the real objects.
        """
        if not providers:
            return []
        scored = self._calculate_base_scores(providers, preferences)  # copies, input order
        return [providers[i] for i in _core_rank_order(scored)]

    def score_providers(
        self,
        providers: List[Dict[str, Any]],
        preferences: Dict[str, Any],
        judge_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Main method to score and rank healthcare providers.

        Args:
            providers: List of provider dictionaries from data gatherer
            preferences: User preference dictionary with weights
            judge_count: Send only the FIRST `judge_count` providers, as given,
                to the rubric judge. Everyone still receives a deterministic
                core score. None judges the whole pool.

        Returns:
            Dictionary containing ranked providers and scoring metadata

        Why the judged set is positional rather than re-derived here: the
        caller pins it BEFORE enrichment, and enrichment backfills ratings that
        move core scores. A set recomputed at this point would not be the set
        that was actually researched — we would judge providers on evidence we
        never went looking for, which is the failure this parameter exists to
        end. `_calculate_base_scores` preserves input order, so "first N" is
        exactly the caller's pinned selection.
        """
        try:
            logger.info(f"Starting provider scoring for {len(providers)} providers")

            if not providers:
                return {
                    "ranked_providers": [],
                    "scoring_metadata": {
                        "total_providers": 0,
                        "preferences_used": preferences,
                        "scoring_method": "weighted_algorithm_with_ai"
                    },
                    "status": "no_providers",
                    "message": "No providers to score"
                }

            # Step 1: Calculate base scores using weighted algorithm
            scored_providers = self._calculate_base_scores(providers, preferences)

            # Step 2: Rubric judge — over the pinned selection only. Providers
            # past it keep their deterministic core score and are flagged, so
            # the UI can say they were ranked but not AI-evaluated rather than
            # implying a judge looked at them.
            if judge_count is None or judge_count >= len(scored_providers):
                to_judge, deferred = scored_providers, []
            else:
                to_judge = scored_providers[:max(judge_count, 0)]
                deferred = scored_providers[max(judge_count, 0):]
                for provider in deferred:
                    provider["ai_judged"] = False
                logger.info(
                    "Judge pool: %d of %d providers (the rest were not enriched, "
                    "so a rubric score would grade our coverage, not them)",
                    len(to_judge), len(scored_providers),
                )

            ranked_providers = self._generate_ai_rankings(to_judge, preferences) + deferred

            # Step 3: Composite — 70% deterministic core (0-100), 30% rubric
            # judge (0-100). Both cardinal, so final_score is a true 0-100.
            # Judge-less providers (parse failure, skipped entry) sit at the
            # neutral 50, which shifts everyone equally and preserves order.
            for provider in ranked_providers:
                ai_score = provider.get("ai_score", 50.0)
                composite_score = (provider.get("base_score", 0) * 0.7) + (ai_score * 0.3)

                provider["final_score"] = round(composite_score, 2)

            # Re-sort by final score to ensure rankings match scores
            ranked_providers.sort(key=lambda x: x.get("final_score", 0), reverse=True)

            # Assign final ranks based on sorted order
            for i, provider in enumerate(ranked_providers):
                provider["final_rank"] = i + 1

            result = {
                "ranked_providers": ranked_providers,
                "scoring_metadata": {
                    "total_providers": len(ranked_providers),
                    "preferences_used": preferences,
                    "scoring_method": "weighted_algorithm_with_ai",
                    "top_provider": ranked_providers[0]["name"] if ranked_providers else None,
                    "score_range": {
                        "highest": ranked_providers[0]["final_score"] if ranked_providers else 0,
                        "lowest": ranked_providers[-1]["final_score"] if ranked_providers else 0
                    }
                },
                "status": "success",
                "message": f"Successfully ranked {len(ranked_providers)} providers"
            }

            logger.info(f"Provider scoring completed: {result['message']}")
            return result

        except Exception as e:
            logger.error(f"Provider scoring failed: {e}", exc_info=True)
            return {
                "ranked_providers": [],
                "scoring_metadata": {
                    "total_providers": len(providers),
                    "preferences_used": preferences,
                    "scoring_method": "weighted_algorithm_with_ai"
                },
                "status": "error",
                "message": "Error scoring providers. Please try again."
            }

    def explain_scoring_methodology(self) -> Dict[str, Any]:
        """Explain the scoring methodology used by the agent.

        Returns:
            Dictionary explaining the scoring approach
        """
        return {
            "methodology": "Hybrid Weighted Algorithm + Rubric-Scored AI Judge",
            "components": {
                "base_scoring": {
                    "description": "Weighted algorithm using user preferences (0-100)",
                    "factors": [
                        "Provider rating (0-5 stars), Bayesian-adjusted by review volume",
                        "Distance/location proximity score (0-100)",
                        "Years of experience (0-100; unknown is imputed at 60, the equivalent of a measured 10 years — NOT a low score)"
                    ]
                },
                "ai_ranking": {
                    "description": "An OpenAI judge scores an anchored rubric with cited evidence",
                    "model": self.config.JUDGE_MODEL,
                    "rubric": {
                        "review_substance": 50,
                        "red_flags": 30,
                        "practical_access": 20
                    },
                    "provides": [
                        "Per-criterion subscores with quoted evidence",
                        "Detailed reasoning, strengths, and concerns"
                    ]
                },
                "final_scoring": {
                    "description": "final = 0.7 x weighted core + 0.3 x rubric score (true 0-100)",
                    "output": "Final ranked list with comprehensive explanations"
                }
            },
            "advantages": [
                "Scores only on validated or evidence-cited signals",
                "Transparent scoring breakdown",
                "AI-generated reasoning for each decision",
                "Customizable preference weighting"
            ]
        }


def create_preference_scorer() -> PreferenceScorerAgent:
    """Factory function to create a PreferenceScorerAgent instance.

    Returns:
        PreferenceScorerAgent instance
    """
    return PreferenceScorerAgent()