"""Data Gatherer Agent for collecting healthcare provider information using Tavily search and FHIR."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from typing import Callable, Dict, List, Any, Optional, Tuple, Union
import json
import re
from tavily import TavilyClient
from anthropic import Anthropic

from utils.config import get_config
from utils.cost_tracker import get_cost_tracker, safe_usage
from utils.excerpt import build_excerpt, clip_words
from utils.json_salvage import salvage_json_objects
from utils.listing_parser import (
    parse_listing, slug_agrees_with_name, slug_contradicts_name,
)
from utils.profile_parser import parse_profile, strip_data_uris
from utils.geo import city_state_for_zip, distance_miles, location_tier, nearby_cities, parse_location, resolution_level, strip_zip
from utils.provenance import (
    REVIEW_PLATFORM_DOMAINS, canonical_profile_url, is_profile_url,
    source_domain, url_page_kind, urls_contradict,
)
from utils.shard import contiguous_shards

# Which discovery pass surfaced a candidate. `ring_expanded` in search_metadata
# is a BOOLEAN — it says the ring fired, never what it bought, so the only way to
# judge whether the extra searches earn their cost was to eyeball a card list and
# guess which names looked out-of-town. This tags them.
#
# The question it answers is specifically "would we still have these providers
# without the ring", which is why a doctor found by BOTH passes counts as
# `home` — see the precedence rule in `_dedupe_providers`.
_DISCOVERY_HOME = "home"
_DISCOVERY_RING = "ring"

# Ordinal preference for a source page when two observations otherwise tie.
# Ordinal rather than boolean so an UNRECOGNISED url outranks a confirmed
# directory listing instead of tying with it — under the tie the winner was
# whichever was seen first, and discovery (city listings) always runs before
# enrichment (real profiles).
_PAGE_KIND_RANK = {"profile": 2, "unknown": 1, "listing": 0}


def _page_rank(url) -> int:
    """Tie-break rank for a source URL: profile 2, unknown 1, listing 0."""
    return _PAGE_KIND_RANK.get(url_page_kind(url), 1)
from utils.provider_key import (
    CACHE_KEY_FIELD,
    normalize_name_tokens,
    normalized_name,
    normalized_place,
    pin_cache_key,
    resolve_cache_key,
)
from utils.security import PromptSanitizer, validate_search_params

logger = logging.getLogger(__name__)

# Token-overlap threshold for name matching during FHIR/Tavily merge
_NAME_MATCH_THRESHOLD = 0.5

# Stricter threshold for deduping extracted candidates: at the FHIR-merge 0.5,
# "John Ortega" vs "Maria Ortega" (overlap 1/2) would wrongly merge
_DEDUP_NAME_THRESHOLD = 0.8

# Results scoring below this Tavily relevance are dropped before they consume
# one of the capped extraction slots. A floor, deliberately NOT a sort key:
# scores are query-relative, and re-sorting merged multi-query results by raw
# score would let one hot-scoring query crowd out the others again.
_MIN_RELEVANCE_SCORE = 0.3

# Independent review platforms: prioritized in extraction input, targeted by
# the platform searches, and the ONLY acceptable sources for
# rating/review_count. Canonical list lives in utils.provenance (shared with
# the scorer's cross-platform blend and the UI); alias kept for existing
# callers and tests.
_REVIEW_PLATFORM_DOMAINS = REVIEW_PLATFORM_DOMAINS

# Result blocks handed to the enrichment extractor. One per platform plus a
# spare — the search asks for 2x the platform count so each domain has a real
# chance to appear, and the round-robin below spends these slots on distinct
# domains first.
_MAX_REVIEW_BLOCKS = len(REVIEW_PLATFORM_DOMAINS) + 1

# The ENRICHMENT excerpt. Round 12 raised discovery's budget 2000 -> 6000 and
# left this one a bare literal `2000` at the call site, defended by a comment
# asserting "2000 chars over 3 windows is ample, and more windows would only
# dilute a single page's header." Two live runs on 2026-07-28 contradicted it:
# the same search ranked Dr. Andrea An #1 and then #5, because run 1 read
# webmd as "4.5/5" with no count and run 2 read "4.5/5 (61 reviews)".
#
# Measured on an 18.3K-char profile-shaped page (`_profile_page` in
# tests/unit/test_excerpt.py), with the rating header at char 993 and a
# complaint in the review prose at 76% depth. Measured AFTER the word-boundary
# anchor fix in utils/excerpt — an earlier sweep taken before it read
# differently, which is the reason this table lives here and not in a doc:
#
#   budget  head  windows   header   prose    chars   coverage
#     2000  none      3       no      no      1334      7.3%   <- shipped
#     2000  1200      3      YES      no      1602      8.8%
#     3000  none      3       no      no      2003     11.0%
#     3000  1200      3      YES      no      2103     11.5%
#     3000  1200      4      YES      no      2406     13.2%   <- chosen
#     3000  1200      6      YES      no      2406     13.2%
#     4000  none      3      YES      no      2669     14.6%
#     4000  1200      3      YES      no      2802     15.3%
#     6000  none      3      YES      no      4002     21.9%
#     6000  1200      3      YES      no      4803     26.3%
#
# What the table settles:
#
#  1. The head reservation is the CHEAP way to the rating header. A review
#     platform states its rating once, ~1000 chars into the cleaned text; the
#     reservation was `budget // max_windows` = 666 and landed ~400 short. The
#     budget alone does eventually reach it — at 4000, i.e. twice today's
#     tokens for the same one fact, and still no prose. 3000+1200 gets the
#     header AND 13.2% coverage for a third less.
#  2. Window count is a STAIRCASE, not the flat line round 12 measured for
#     discovery, because `window_size = budget // max_windows` feeds back into
#     clustering — narrower windows split the review body into more clusters:
#
#         windows  3 -> 2103 chars | 4,5,6 -> 2406 | 8,10 -> 2557
#
#     4 is chosen as the first real step. 8 buys 151 more chars (+0.8pt
#     coverage) but only by cutting the 1800 non-head chars into SEVEN 257-char
#     fragments — barely a sentence each, and a complaint split across two of
#     them is not evidence the extractor can use. Char count is not the metric;
#     usable spans are.
#  3. Coverage 7.3% -> 13.2% is review PROSE, which is what the summary is
#     written from. Run 1 wrote an all-praise summary for a provider whose
#     reviews name MRI follow-up failures; run 2 caught them.
#
# What the table does NOT fix is the prose column: a fact at 76% depth is
# unreachable at every row, because that one merged cluster centres its window
# near 50%. That is a structural limit of density selection on a uniformly
# named page — not a constant to tune; do not paper it over with a bigger
# number.
_ENRICHMENT_EXCERPT_BUDGET = 3000
_ENRICHMENT_EXCERPT_WINDOWS = 4
_ENRICHMENT_HEAD_CHARS = 1200

# Result blocks handed to the DISCOVERY extractor, and the excerpt each one
# gets. These are deliberately different from enrichment's, because the two
# passes read different documents for different reasons.
#
# Discovery reads directory and "top N" pages that spread 10-20 providers
# across 10-20K chars. `_pick_clusters` opens a NEW cluster whenever anchor
# hits are more than one window apart, so on a listicle whose entries run
# ~1000 chars EVERY ENTRY IS ITS OWN CLUSTER — under 2000/3 (666-char windows)
# the three densest landed on roughly the first three entries, and a 15-name
# page contributed three names. The home pool then came in under
# MIN_CANDIDATE_POOL and the ring expanded to cities the user never asked for,
# paying for a second search AND a second extraction to recover names that were
# already on pages we had.
#
# Measured on a synthetic 15-entry / 14K-char listicle (the test below), the
# BUDGET is the lever and the window count merely has to keep each window
# wider than one entry:
#
#     budget  windows  window  names recovered
#       2000        3     666      2   <- today
#       6000        8     750      7
#       6000       12     500      7   <- more windows, no gain: too narrow
#       8000       12     666      9
#
# So ~1 name per 1000 chars of budget, and splitting a fixed budget into more
# windows buys nothing once they fall below entry length. 6000/8 takes a
# 15-name page from 2 names to 7 — enough for 18 blocks to fill a pool of 10
# without leaving the city. `build_excerpt` takes min(budget, available), so
# the larger budget only materialises on genuinely long pages, which are
# exactly the ones being under-read.
_DISCOVERY_MAX_BLOCKS = 18
_DISCOVERY_EXCERPT_BUDGET = 6000
_DISCOVERY_EXCERPT_WINDOWS = 8

# Bound on Tavily's `content` per page block. `raw_content` has been budgeted
# since round 12, but `content` was pasted WHOLE into the same block — one
# field disciplined, the other not, in the same f-string. That asymmetry was
# harmless only while `content` was small at the vendor's default of 3 chunks;
# raising TAVILY_CHUNKS_PER_SOURCE to 5 grows it ~70% (2,169 -> 3,636 chars per
# page measured over 20 pages), and an 18-block discovery pass multiplies that
# by 18. A knob that silently scales an unbounded field is how a prompt doubles
# without anyone deciding it should.
#
# 3000, not the excerpt's 6000: `content` and the excerpt are COMPLEMENTARY,
# not redundant, and the split protects the excerpt's share. Measured over one
# discovery pass, counting occurrences across the whole pass:
#
#                      content   excerpt
#     provider entry      65        36     content wins the roster fields
#     rating+count        55        20
#     address w/ ZIP      97        37
#     years of exp         5        11     excerpt wins the two that the
#     insurance           26        47     per-domain anchors target
#
# Neither field can be dropped for the other: `content` is selected for
# relevance to a query that asks about reviews and ratings, so it is dense in
# exactly those and thin in tenure and payer lists, which is what
# `_DOMAIN_ANCHOR_HINTS` aims the excerpt windows at. Clipping is word-safe
# and leaves a visible "…", because a model handed a silent fragment reasons
# about it as though it were the whole page.
_CONTENT_MAX_CHARS = 3000

# (ADDRESS_CONFLICT_MILES, the 20-mile conflict threshold, was retired with
# the cluster vote: a multi-office group specialist's far-apart addresses are
# a confirmed practice, not a conflict, and the member-facing note now keys
# on "more than one location" rather than on any distance.)

# How many concurrent extraction calls read those blocks, and the pool size
# below which splitting stops paying. Two, not more: each shard re-sends the
# whole ~1.5k-token instruction block, so the prompt cost grows linearly with
# the shard count while the latency win is only the first halving — 2 buys
# ~12s of the 25.3s step, 4 buys ~6s more for twice the preamble.
#
# The floor exists because the win is proportional and the overhead is not.
# At 18 blocks (the cap, and the normal case) each shard reads 9 pages; below
# 6 the halves are 3 and 3, where one extra preamble buys about two pages of
# parallelism. It is NOT a correctness floor — the pages are independent at
# any count — so it is a plain threshold, not a guard.
_DISCOVERY_SHARDS = 2
_MIN_PAGES_TO_SHARD = 6

# Discovery extraction's OUTPUT budget scales with the blocks in the call —
# the judge's and critic's ceilings have scaled with pool size since round 9,
# while this call kept a flat 8000 although its truncation is the worst in
# the system: a cut array has no closing bracket, repair cannot match, the
# pool reads ZERO, and the ring rebuilds it from cities nobody asked about
# (2026-07-28: two runs of one search, 11 minutes apart, overlapped on ~1
# physician — the flat ceiling was the coin flip deciding WHO got
# recommended). A block can name several providers on a directory page and a
# full entry with a multi-sentence summary runs ~400-600 output tokens, so
# 600/block; the base covers preamble-free small calls; the FLOOR is the old
# flat value so no call gets less room than it had (at 9 blocks — a normal
# shard — the formula gives 9400). max_tokens is a ceiling, not a spend:
# unused headroom costs nothing, so the cap exists only to bound a
# pathological block count, far under Haiku's 64k output limit.
_DISCOVERY_TOKENS_BASE = 4000
_DISCOVERY_TOKENS_PER_BLOCK = 600
_DISCOVERY_TOKENS_MAX = 16000

# Where to aim the excerpt window on each platform's pages, beyond the
# generic review vocabulary: every platform has a known section whose
# capture directly feeds a score input (years -> experience subscore,
# insurance -> payer evidence, conditions -> the judge's requirements fit).
_DOMAIN_ANCHOR_HINTS = {
    # "patient rating" earns its place beside the tenure and insurance hints
    # because the REVIEW COUNT is the field healthgrades states in its header
    # and nowhere else, and it is the one the blend weighs. The generic
    # `rating` anchor was supposed to cover it and did not: bounded at both
    # ends it misses the plural "70 patient ratings" — fixed in
    # `_anchor_pattern`, but a hint aimed at the exact phrase costs nothing and
    # does not depend on that fix holding.
    'healthgrades.com': ("years of experience", "insurance accepted", "patient rating"),
    'webmd.com': ("conditions treated", "procedures"),
    'vitals.com': ("insurance",),
}


# Surnames that are also English FUNCTION words. A bare-surname anchor is a
# FREQUENCY BET: it works because a profile page mentions the doctor often and
# mentions little else by that token. For "An" the bet inverts completely — on
# review prose the indefinite article scores 360 hits to the surname's handful,
# so the priority windows scatter uniformly across the page and never reach the
# header carrying the rating. The 2026-07-29 run showed the result: a
# healthgrades profile fetched at 44,138 chars produced no rating+count pair,
# while the same doctor's webmd and vitals profiles (33,730 / 31,690 chars) did.
#
# The irony is exact. Round 15's `_MIN_ANCHOR_LEN` 3 -> 2 existed so that Dr.
# Andrea An's surname would anchor AT ALL — it had scored 1 hit against 101 for
# a normal surname. Going from 1 useless hit to 360 useless hits is the same
# failure with the sign flipped, on the same provider, in the same field.
#
# CONTENT-word surnames (Young, Price, Long, Stone) are deliberately absent:
# they appear a handful of times per page, the same order as a real surname
# mention, so excluding them would cost recall for no measured gain. Function
# words differ in KIND, not degree — they are the scaffolding of every sentence
# on the page.
_FUNCTION_WORD_SURNAMES = frozenset({
    "a", "am", "an", "and", "any", "are", "as", "at", "be", "but", "by", "can",
    "did", "do", "for", "had", "has", "he", "her", "him", "his", "how", "i",
    "if", "in", "is", "it", "its", "me", "my", "no", "nor", "not", "of", "on",
    "or", "our", "out", "she", "so", "the", "their", "them", "they", "this",
    "to", "up", "us", "was", "we", "were", "what", "when", "who", "why",
    "will", "with", "you", "your",
})


def _surname_anchors(surname: str) -> List[str]:
    """Priority excerpt anchors for a provider's surname, titled forms first.

    "Dr. An" is what a profile header and its review bodies actually write, and
    unlike the bare token it cannot collide with ordinary prose. Both the
    period and the bare-title spelling are covered because platforms differ.

    The bare surname follows only when it is not a function word — it is still
    the broadest matcher when a page omits the title, which vitals headers do.
    """
    token = str(surname or "").strip()
    if not token:
        return []
    anchors = [f"dr. {token}", f"dr {token}"]
    if token.lower() not in _FUNCTION_WORD_SURNAMES:
        anchors.append(token)
    return anchors


# The three platforms whose directory pages parse deterministically. Discovery
# fans one query across them, ONE DOMAIN PER CALL.
#
# One combined call is not equivalent, and not merely less balanced: with all
# five platforms in a single basic-depth request, `include_domains` did not hold
# and the search returned bestbuy.com, YouTube and Cambridge Dictionary — zero
# in-domain results against advanced's twenty. One domain per call holds at
# basic on all three (5 results, 5 in-domain, measured), which is also why
# discovery costs 3 credits here rather than 6.
#
# zocdoc and ratemds are absent by measurement, not preference: zocdoc returned
# ten pages with no rating on any of them and no URL our classifier could type,
# and ratemds returned two pages and no rating+count pair while already on
# probation for exactly that.
_LISTING_DOMAINS = ("healthgrades.com", "doctor.webmd.com", "vitals.com")


# The HEAD of a specialty label names the DISCIPLINE; the words in front of it
# narrow it. "Vascular Neurology" is neurology. "Neurological Surgery" is
# surgery — a different training path, a different kind of appointment — and no
# amount of shared stem makes a neurosurgeon a neurologist.
#
# A stem rule alone cannot tell those apart, and did not: a Neurology search
# returned "Neurological Surgery" at ranks 1 and 4 because "neurological"
# starts with "neurol". The critic caught it in its own words on that run —
# "the list blends brain/spine surgeons with general neurologists" — which is a
# ranking caveat standing in for a filter that should have run first.
#
# Only heads that name a whole discipline belong here. "Medicine" deliberately
# does NOT: "Sleep Medicine" is a neurology sub-specialty portals file
# neurologists under, and treating its head as a discipline would reject them.
# A label whose head is not listed falls through to the lenient check below,
# which is the behaviour this list is narrowing, not replacing.
_DISCIPLINE_HEADS = frozenset({
    "surgery", "neurology", "psychiatry", "radiology", "pathology",
    "anesthesiology", "dermatology", "oncology", "psychology", "dentistry",
    "podiatry", "optometry", "chiropractic", "obstetrics", "gynecology",
    "pediatrics", "cardiology", "urology", "ophthalmology", "otolaryngology",
    "orthopedics", "orthopaedics", "endocrinology", "gastroenterology",
    "rheumatology", "nephrology", "pulmonology", "immunology", "hematology",
    "geriatrics", "neurosurgery",
})


def _discipline_head(label: str) -> Optional[str]:
    """The discipline a specialty label resolves to, or None if unrecognised."""
    words = re.findall(r"[a-z]+", str(label or "").lower())
    for word in reversed(words):
        if word in _DISCIPLINE_HEADS:
            return "surgery" if word == "neurosurgery" else word
    return None


def _specialty_is_compatible(row_specialty: Any, target: str) -> bool:
    """Does a listing row's own specialty label admit the searched specialty?

    A directory scoped to one specialty still lists others — a vitals NEUROLOGY
    page carries "Dr. Edgardo D Zavala-Alarcon, MD — Plastic Surgery" at 4.7
    over 72 ratings, which would outrank most of the real neurologists. Beyond
    that the check is DELIBERATELY LENIENT: portals file one doctor under
    adjacent labels, and rejecting on a label mismatch alone is the failure that
    made the enrichment query drop the specialty term in the first place. A
    missing label admits the row.

    The one place leniency is NOT allowed is a different DISCIPLINE. When both
    labels name one, they must name the same one — that is what separates
    "Vascular Neurology" (a neurologist) from "Neurological Surgery" (a
    surgeon), which the stem rule below could not, because it only ever saw
    that both words begin "neurol". Symmetric by construction: searching
    "General Surgery" admits "Neurological Surgery" and rejects "Neurology".
    """
    label = str(row_specialty or "").strip().lower()
    want = str(target or "").strip().lower()
    if not label or not want:
        return True

    label_head, want_head = _discipline_head(label), _discipline_head(want)
    if label_head and want_head:
        return label_head == want_head

    label_words = set(re.findall(r"[a-z]+", label))
    want_words = set(re.findall(r"[a-z]+", want))
    if label_words & want_words:
        return True
    # Spelling variants of ONE discipline ("Neurologic" / "Neurology"). This is
    # all the stem rule was ever meant to do; the discipline check above is what
    # stops it reaching across two.
    return any(
        w.startswith(v[:6]) or v.startswith(w[:6])
        for w in label_words for v in want_words
        if len(w) >= 6 and len(v) >= 6
    )


def _listing_row_to_provider(row: Dict[str, Any], specialty: str) -> Dict[str, Any]:
    """A parsed listing row in the shape the rest of the pipeline expects.

    The rating is emitted BOTH as the headline pair and as a
    `review_observations` entry, because the blend, the same-domain collapse and
    the platform-pair count all read observations — a row that set only
    `rating`/`review_count` would score but contribute nothing to cross-platform
    agreement, which is the whole reason for reading three platforms.

    `profile_url` is carried through as the canonical identity key. It is
    stronger evidence than name matching: two records sharing a platform's own
    `/physician/dr-…` link are the same person, where `_name_token_overlap` at
    0.5 would accept two different doctors sharing a surname.
    """
    provider: Dict[str, Any] = {
        "name": row.get("name"),
        "specialty": row.get("specialty") or specialty,
        "location": row.get("location"),
        # Where the address came from, carried from the first write. A card
        # showed two different doctors at one street address and one distance,
        # and no surface could say whether that came from a listing row, a
        # parsed profile, or a model reading a group practice page.
        "location_source": f"listing_parser:{row.get('review_source_url')}",
        "years_experience": row.get("years_experience"),
        # Same provenance discipline as the address: tenure decides real
        # ranking points (a stated year vs the unknown imputation), so triage
        # needs its producer named. None when the row stated no tenure —
        # a later producer then labels its own write.
        "experience_source": (
            f"listing_parser:{row.get('review_source_url')}"
            if row.get("years_experience") is not None else None
        ),
        "profile_url": row.get("profile_url"),
        "rating": row.get("rating") or 0,
        "review_count": row.get("review_count"),
        "review_source_url": row.get("review_source_url"),
        "review_summary": "No reviews available",
        "review_sentiment": "unknown",
        "insurance_accepted": [],
        "extraction_source": "listing_parser",
    }
    if row.get("rating") is not None and (row.get("review_count") or 0) > 0:
        # The observation points at the DOCTOR'S OWN PROFILE, not at the index
        # we read it off — when the row carried one, which is the normal case
        # because every platform links the profile from the entry heading.
        #
        # A directory row's numbers are PER ROW. `Rated 3.8 out of 5 … from 88
        # ratings` sits inside Dr. Pandey's block and is his, so "listing page"
        # describes where we read it and not whom it is about. Recording the
        # index URL made the card say "doctor.webmd.com — listing page" and
        # link forty doctors deep, while the link to the one doctor it is about
        # sat unused in the same parsed row.
        #
        # `read_from_url` keeps the provenance rather than losing it: the pair
        # was read off the directory, and a later enrichment pass reading the
        # profile itself may state different numbers.
        provider["review_observations"] = [{
            "source_url": row.get("profile_url") or row.get("review_source_url"),
            "read_from_url": row.get("review_source_url"),
            "rating": row.get("rating"),
            "review_count": row.get("review_count"),
            "page_provider_name": row.get("name"),
        }]
    else:
        provider["review_observations"] = []
    return provider


def _anchors_for(url: Any, base_anchors: List[str]) -> List[str]:
    """Base anchors + the hint anchors for the result's platform, if any."""
    lowered = str(url or "").lower()
    for domain, hints in _DOMAIN_ANCHOR_HINTS.items():
        if domain in lowered:
            return list(base_anchors) + list(hints)
    return list(base_anchors)


def _describe_query_spec(spec: Dict[str, Any]) -> str:
    """One display string per search CALL, domain restriction included.

    Discovery's three calls share one query string by design — the
    differentiator is one listing domain per call, riding in
    `include_domains` where no metadata surface showed it. The dev panel
    therefore rendered "three identical queries", which photographs exactly
    like a triple-spend bug. The arrow suffix makes the per-call restriction
    visible; an unrestricted spec keeps the bare query.
    """
    query = str(spec.get("query", ""))
    domains = spec.get("include_domains") or []
    return f"{query}  →  {', '.join(domains)}" if domains else query


def _is_review_platform_url(url: Any) -> bool:
    """True when a URL belongs to one of the review platforms.

    Deliberately domain-level rather than `url_page_kind(url) == "profile"`:
    the head window is cheap insurance, an unrecognised-but-real profile is a
    known state of that classifier, and spending the window on a listing page's
    title costs far less than missing a profile header's rating.
    """
    lowered = str(url or "").lower()
    return any(domain in lowered for domain in _REVIEW_PLATFORM_DOMAINS)

# A rating+count pair needs at least this many reviews to headline over a
# rating-only observation — a single 1.0 review must not outrank an
# "Excellent (5/5)" whose total simply didn't scrape.
_MIN_CREDIBLE_COUNT = 3

# List fields where two records of the same physician hold DIFFERENT evidence
# rather than competing versions of the same fact — merge them instead of
# picking a winner. (`_select_review_observation` collapses observations to one
# voice per platform downstream, so a union here cannot double-count.)
_UNION_ON_DEDUPE = ("review_observations", "insurance_accepted")


def _union_evidence(current: Any, incoming: Any) -> Optional[List[Any]]:
    """Merge two evidence lists, first occurrence winning, order preserved."""
    if not isinstance(current, list) and not isinstance(incoming, list):
        return None
    merged: List[Any] = []
    seen = set()
    for item in list(current or []) + list(incoming or []):
        if isinstance(item, dict):
            marker = str(item.get("source_url") or item.get("platform") or item).lower()
        else:
            marker = str(item).strip().lower()
        if not marker or marker in seen:
            continue
        seen.add(marker)
        merged.append(item)
    return merged

# When only rating-only observations exist and they span more than this many
# stars (the Khan case: 1.2 on one platform vs 5.0 on another), no single
# headline is honest — decline, and let the card's "Across platforms" line
# show the disagreement.
_RATING_DISAGREEMENT_SPAN = 2.0

_RATING_NUMBER_RE = re.compile(r'(\d+(?:\.\d+)?)')


def _parse_rating(value) -> Optional[float]:
    """Normalize a stated rating to a float in (0, 5], else None.

    Pages and models phrase ratings many ways — 4.7, "4.7", "4.7/5",
    "1.2 / 5", "4 stars", "4.5 out of 5". The first number wins (denominators
    and units trail it); anything outside (0, 5] is rejected, so a stray
    review COUNT ("271") can never masquerade as a rating.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        rating = float(value)
    else:
        match = _RATING_NUMBER_RE.search(str(value))
        if not match:
            return None
        rating = float(match.group(1))
    return round(rating, 1) if 0 < rating <= 5 else None


_INT_IN_PROSE_RE = re.compile(r"\d[\d,]*\d|\d")


def _first_int(value) -> Optional[int]:
    """First integer in a value, or None.

    Profiles state numbers as prose — "30+ years of experience", "Based on 31
    reviews", "(271)". A bare int(float(...)) rejects every one of those, and
    the rejection is silent: a stated review count that failed to parse left
    the observation rating-only, so it stopped counting as a platform pair and
    dropped out of the cross-platform blend.

    Thousands separators are part of that prose and must survive the parse. A
    plain `\\d+` stops at the comma, so "1,234 reviews" read as 1 and
    "12,000 patient ratings" as 12 — the busiest, best-evidenced providers
    reduced to single-review noise. That is not a rounding error: at 4.8
    stars, a count of 1234 gives a Bayesian-adjusted 4.79 (rating score 95.8)
    and a count of 1 gives 3.62 (72.4), a 23-point swing on the dimension,
    plus a confidence downgrade from "high" to "low" and a
    count-weighted cross-platform blend that now weights this platform at 1.
    The comma is only consumed BETWEEN digits, so "4.7, 271 reviews" still
    reads 4 rather than merging across the separator.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = _INT_IN_PROSE_RE.search(str(value))
    return int(match.group().replace(",", "")) if match else None


def _run_observation_ladder(candidates):
    """The headline preference ladder over one class of observations."""
    credible_pairs = [
        o for o in candidates
        if o["rating"] is not None and (o["review_count"] or 0) >= _MIN_CREDIBLE_COUNT
    ]
    rated = [o for o in candidates if o["rating"] is not None]
    counted = [o for o in candidates if o["review_count"]]

    if credible_pairs:
        # Volume is the tiebreaker between disagreeing sources — but only among
        # sources that are ABOUT this doctor. A "Best Neurologists in Chandler"
        # index states a number that is true of the page, not attributable to
        # the person whose card it lands on, and volume alone let one win: a
        # 2026-07-31 run carded "Best single source: healthgrades.com — listing
        # page 4.6/5 (33 reviews)" for a provider whose own webmd profile stated
        # 5.0 over 22. The listing had a bigger number, so it took the headline
        # and the link.
        #
        # A listing pair still counts everywhere else — it stays in "Across
        # platforms", it feeds the blend, it counts toward the platform pair
        # count. It just cannot be the ONE source the card attributes to the
        # doctor while an attributable one exists. That is the same distinction
        # `profile_backed_platforms` was added to measure, applied at the point
        # where it decides what a patient reads.
        attributable = [o for o in credible_pairs if _page_rank(o["source_url"]) > 0]
        return max(attributable or credible_pairs, key=lambda o: o["review_count"])
    if rated:
        ratings = [o["rating"] for o in rated]
        if max(ratings) - min(ratings) > _RATING_DISAGREEMENT_SPAN:
            # Wildly conflicting rating-only claims: any single headline
            # would mislead — show nothing and let the observations speak
            return None
        return rated[0]
    if counted:
        return max(counted, key=lambda o: o["review_count"])
    return None


def _select_review_observation(observations) -> tuple:
    """Pick the headline rating/count from per-platform observations, in code.

    The model only TRANSCRIBES what each page states; selection is
    deterministic here. Independent review platforms are a strictly higher
    class than everything else: a hospital system's own 4.7 (486 surveys)
    must not out-headline healthgrades' 2.8 (16 reviews) — employer sites
    have a marketing incentive. Non-platform observations headline only when
    NO platform observation exists at all; if platform observations exist
    but decline (disagreement guard), the decline stands — self-published
    numbers don't win by forfeit. Everything remains visible in the card's
    "Across platforms" line either way.

    Returns (headline_or_None, normalized_observations).
    """
    normalized = []
    for obs in observations or []:
        if not isinstance(obs, dict):
            continue
        rating = _parse_rating(obs.get("rating"))
        count = _first_int(obs.get("review_count"))
        if count is not None and count <= 0:
            count = None
        if rating is None and count is None:
            continue
        normalized.append({
            "source_url": str(obs.get("source_url") or "").strip(),
            "rating": rating,
            "review_count": count,
        })

    # One voice per platform. Portals list one doctor under several URLs
    # (the Khan case: healthgrades' Neurology and Sleep Medicine paths both
    # carried 4.0/31) and a same-domain duplicate must never pose as a
    # second opinion — it satisfied the enrichment trigger, inflated the
    # blend's platform count, and printed twice on the card. Keep the
    # strongest observation per platform domain (a pair beats rating-only,
    # a larger count beats a smaller — headline doctrine); non-platform
    # observations pass through untouched.
    def _strength(o):
        return (
            o["rating"] is not None and (o["review_count"] or 0) > 0,
            o["review_count"] or 0,
            o["rating"] is not None,
            # Break ties toward the provider's own profile page over a
            # directory listing — same numbers, but the profile is the
            # correct link and the attributable source. Ordinal, so an
            # unrecognised URL outranks a confirmed listing instead of tying
            # with it (the loser here is DELETED from the observation list).
            _page_rank(o["source_url"]),
        )

    best_by_domain: Dict[str, Dict[str, Any]] = {}
    deduped: List[Dict[str, Any]] = []
    for o in normalized:
        domain = next(
            (d for d in _REVIEW_PLATFORM_DOMAINS if d in o["source_url"].lower()), None
        )
        if domain is None:
            deduped.append(o)
            continue
        held = best_by_domain.get(domain)
        if held is None:
            best_by_domain[domain] = o
            deduped.append(o)
        elif _strength(o) > _strength(held):
            deduped[deduped.index(held)] = o
            best_by_domain[domain] = o
    normalized = deduped

    platform_obs = [
        o for o in normalized
        if any(domain in o["source_url"].lower() for domain in _REVIEW_PLATFORM_DOMAINS)
    ]
    if platform_obs:
        headline = _run_observation_ladder(platform_obs)
    else:
        headline = _run_observation_ladder(normalized)
    return headline, normalized


def _platform_rating_pairs(observations) -> List[Dict[str, Any]]:
    """The best rating+count pair per platform DOMAIN — the only
    observations the blend can weigh, and the enrichment trigger's coverage
    measure: fewer than two means the provider's numbers are one platform's
    opinion. Collapsing per domain here (largest stated count wins) is the
    load-bearing guarantee that a same-domain duplicate can never count as
    a second opinion, wherever this helper is called."""
    best: Dict[str, Dict[str, Any]] = {}
    for o in observations or []:
        if not isinstance(o, dict):
            continue
        if o.get("rating") is None or (o.get("review_count") or 0) <= 0:
            continue
        url = str(o.get("source_url", "")).lower()
        domain = next((d for d in _REVIEW_PLATFORM_DOMAINS if d in url), None)
        if domain is None:
            continue
        held = best.get(domain)
        # Largest stated count wins (headline doctrine); at equal counts, the
        # provider's own profile page beats a directory listing so the pair's
        # recorded URL is the attributable one.
        #
        # `_page_rank` is ordinal (profile 2 / unknown 1 / listing 0), not a
        # boolean. Under the boolean an UNRECOGNISED profile URL scored the
        # same as a confirmed directory index, so the tie collapsed to
        # first-seen — and discovery runs before enrichment, which handed the
        # city-listing page the win over the real profile found later. Note
        # rating is deliberately still absent from this key: the doctrine is
        # that review VOLUME decides the headline, not the most flattering
        # number.
        cand_key = ((o.get("review_count") or 0), _page_rank(o.get("source_url")))
        held_key = (
            ((held.get("review_count") or 0), _page_rank(held.get("source_url")))
            if held is not None else None
        )
        if held is None or cand_key > held_key:
            best[domain] = o
    return list(best.values())


def _profile_backed_pairs(observations) -> List[Dict[str, Any]]:
    """The subset of `_platform_rating_pairs` whose source is a real profile.

    Restores the coverage measure `_has_profile_source` provided until the
    2026-07-25 enrichment-uniformity phase deleted it alongside the tier
    predicates it sat with. It was NOT one of them: the tiers rationed WHO got
    enriched, while this asks whether the numbers we ended up with are
    attributable to the doctor's own page or only to a "Best Neurologists in
    <city>" directory index — whose figures belong to many doctors.

    Uniform enrichment preserved the ACTION the old trigger forced (everyone in
    budget gets a per-provider pass) but silently dropped the OUTCOME check.
    `_classify_enrichment` still returns `enriched` for a provider whose only
    healthgrades URL is a directory page, so "we hold her profile" and "we hold
    a listing with her name on it" became the same value — which is how the
    2026-07-28 run carded "healthgrades.com — listing page" as the best single
    source for two providers with nothing recording it.

    `unknown` deliberately does NOT count. This measure exists to answer "did we
    reach a real profile", and an unrecognised URL is one we have not
    identified; counting it would report coverage we cannot demonstrate. That is
    the opposite of `_page_rank`'s treatment of unknown, and for the opposite
    reason: there, refusing to rank unknown BELOW a confirmed listing avoids
    asserting something false about a URL we can't classify.
    """
    return [
        pair for pair in _platform_rating_pairs(observations)
        if url_page_kind(pair.get("source_url")) == "profile"
    ]


def _annotate_source_yields(
    sources: Optional[List[Dict[str, Any]]],
    observations: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Record what each fetched page PRODUCED, alongside that it was fetched.

    `enrichment_sources` logged url / kind / raw_chars — everything about the
    fetch and nothing about the result — so three failure modes it claimed to
    separate still collapsed into one row. Each page now carries `yielded`:

        None                             nothing in this pass named this URL
        {"rating": 4.1, "review_count": None}   read, but no count -> loses the
                                         same-domain collapse on `has_pair`
                                         before page kind is ever consulted
        {"rating": 4.1, "review_count": 70}     a full pair

    Each row also carries `via` — "profile_parser" or "llm" — because the two
    fail for different reasons and the fix differs: a parser gap is a markup
    change to re-read off a fetched page, an LLM gap is an excerpt that missed
    the header. Without it, coverage looks identical either way.

    Matching is on the URL string (case- and trailing-slash-insensitive). The
    extractor echoes the URL it was shown, so a mismatch is rare — but when one
    happens the row reads `None`, which is the honest statement of what this
    function knows: no observation in this pass named that URL. It is NOT proof
    the page was unreadable.
    """
    by_url: Dict[str, Dict[str, Any]] = {}
    for obs in observations or []:
        if not isinstance(obs, dict):
            continue
        key = str(obs.get("source_url") or "").strip().lower().rstrip("/")
        if key:
            by_url.setdefault(key, obs)

    annotated = []
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        row = dict(source)
        obs = by_url.get(str(source.get("url") or "").strip().lower().rstrip("/"))
        row["yielded"] = None if obs is None else {
            "rating": _parse_rating(obs.get("rating")),
            "review_count": _first_int(obs.get("review_count")),
            "via": obs.get("extraction_source") or "llm",
        }
        annotated.append(row)
    return annotated


def _split_by_radius(
    providers: List[Dict[str, Any]], radius_miles: float
) -> tuple:
    """(near enough, too far) — by MEASURED distance only.

    A platform's city page is a radius centre, not a filter: a Chandler search
    returns Gilbert, Mesa and Phoenix entries. Nothing bounded that before, and
    distance alone could not, because location's realized span across a real
    pool is under one point — so a provider 40 miles out competed for a
    research-budget slot with one 4 miles out and lost by a rounding error.

    A radius, never a city match. On the 2026-07-31 run Gilbert at 4.5 mi was
    nearer to the searched Chandler ZIP than two of that run's own Chandler
    results at 8.0 mi, so "same city only" would have dropped the closest
    provider on the page. The city someone types is where they are, not a
    boundary they are asking to have enforced.

    An UNKNOWN distance never drops anyone. That is our geocoding coverage, not
    their location — the same rule the scorer follows when it declines to
    penalise a provider for data we failed to gather.
    """
    near, far = [], []
    for provider in providers or []:
        distance = provider.get("computed_distance_miles")
        if isinstance(distance, (int, float)) and not isinstance(distance, bool) \
                and distance > radius_miles:
            far.append(provider)
        else:
            near.append(provider)
    return near, far


def _parse_profile_pages(results: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Read every PROFILE page this search returned, deterministically.

    PROFILE pages only. These patterns describe a profile's markup, a directory
    index has its own parser, and running profile patterns over a page naming
    forty doctors is how a neighbour's rating ends up on someone's card.

    A page the parser cannot read is simply absent from the result, which is
    what routes it to the model — and so is a page it reads PARTIALLY, because
    the caller merges per field.
    """
    parsed: Dict[str, Dict[str, Any]] = {}
    for result in results or []:
        url = str(result.get("url") or "")
        if not url or url_page_kind(url) != "profile":
            continue
        facts = _parse_both_texts(url, result)
        if facts:
            parsed[url] = facts
    return parsed


def _parse_both_texts(url: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a page's `raw_content` AND Tavily's `content`, and merge them.

    ONE page, TWO extractions. `raw_content` is the full-page markdown;
    `content` is Tavily's own relevance-selected chunks of the same URL. They
    are produced differently and they do not carry the same text — which the
    round-17 sweep already measured (`content` carried 55 rating+count pairs
    across a discovery pass against the anchored excerpt's 20) — and yet only
    `raw_content` was ever parsed. The other text sat in the same result dict,
    already paid for and already going to the model in the prompt, unread by
    the deterministic path.

    It is not a hypothetical gap. A healthgrades profile came back at 5,756
    chars stating a rating and no total, while the page in a browser reads
    "4.8 Star Rating / Based on 260 reviews" — so that provider's healthgrades
    pair was dropped from the blend and the card said "2 platforms" beside
    three named ones.

    THE CHUNKS ARE THE WEAKER SOURCE and are treated as such. Chunk boundaries
    lose the heading structure the per-platform parsers use to keep a
    neighbour's numbers out, so `_parse_healthgrades`'s bound above
    "## Compare Providers" cannot fire on text where that heading was not
    selected. Two rules follow:

      - The full-page parse WINS every field it produced. Chunks only fill gaps.
      - Any field BOTH parses produced must agree, or the chunks are describing
        something the full page did not and none of their extras are trusted.
      - A COUNT from the chunks needs the full page to state a RATING. That
        rating is what names the subject: it says the page we fetched is about
        one doctor with one score, and the chunks are completing that pair
        rather than introducing a number about nobody in particular.

    That last rule USED to demand more — that the chunks RESTATE the rating and
    that the two match. It was written as deliberately too tight, with a note
    that loosening should follow a run showing it cost real coverage. The
    2026-08-04 run is that evidence. Dr. Hagevik's healthgrades profile came
    back as a thin 5,756-char body stating `4.8 Star Rating` and no total,
    while the chunks for the SAME url carried `Based on 260 reviews` — the
    exact text a browser shows. The chunks state a count and no rating, so the
    old rule refused it, healthgrades went into the blend with a rating and no
    weight, and the card read "2 platforms" beside three named ones.

    Requiring the restatement was asking the weaker text to prove something the
    stronger text had already established. Both texts are the SAME url, fetched
    once: the full page's rating is a better subject anchor than a repetition
    inside a relevance-selected excerpt. A conflicting rating in the chunks
    still discards them whole, which is the check that actually protects the
    count — a promo strip that carries someone else's total generally carries
    their stars too.
    """
    primary = parse_profile(url, str(result.get("raw_content") or ""))
    chunks = str(result.get("content") or "")
    if not chunks:
        return primary

    secondary = parse_profile(url, chunks)
    if not secondary:
        return primary

    # Any field both produced must agree, or the chunks are describing
    # something the full page did not and none of their extras are trustworthy.
    for field in ("rating", "review_count", "years_experience"):
        if (primary.get(field) is not None and secondary.get(field) is not None
                and primary[field] != secondary[field]):
            logger.info(
                "Ignoring chunk parse for %s: %s disagrees (%r vs %r)",
                url, field, primary[field], secondary[field],
            )
            return primary

    merged = dict(primary)
    for field, value in secondary.items():
        if field == "review_count" and primary.get("rating") is None:
            # No rating on the full page means nothing here names the subject,
            # so a bare count out of the chunks would be attached to whoever
            # the page happens to be about. Refuse — and say so, because this
            # is the branch that silently costs coverage.
            logger.info(
                "Chunk count %r refused for %s: the full page states no rating",
                value, url,
            )
            continue
        if field == "review_count" and primary.get("rating_pattern") == "likelihood":
            # healthgrades template B states NO total anywhere in its payload —
            # measured, and the reason the parser deliberately returns a rating
            # with no count for it. A count appearing only in this page's
            # chunks therefore cannot be the subject's; it is a neighbour's
            # "Based on N reviews" whose stars the chunk boundary cut, which is
            # exactly the shape the full-page-rating anchor cannot catch.
            # Template A pages are untouched: their payload does carry the
            # total, so a chunk count there is recovering a thin fetch's loss.
            logger.info(
                "Chunk count %r refused for %s: template-B page, its payload "
                "carries no total", value, url,
            )
            continue
        merged.setdefault(field, value)
    return merged


def _url_identifies_provider(url: Any, provider_name: str) -> bool:
    """Does this URL show the page is about THIS doctor?

    The slug must carry their name. That is the whole test, and it works the
    same way on a platform profile (`/physician/dr-harvinder-kumar-x1`) and on
    a practice site's own per-doctor page (`/physicians/harvinder-kumar/`).

    POSITIVE identification, not the absence of a contradiction — the
    distinction is the entire point here. A group practice page (`/physicians`,
    `/neurology`, `/locations/mesa`) names nobody, so it contradicts nobody, so
    `slug_contradicts_name` admits every one of them — and they are exactly the
    pages that state one address for several doctors.

    `url_page_kind` deliberately does NOT appear. It answers a different
    question — is this a profile or a directory index, for labelling a source
    on a card — and it returns "profile" for ANY url off the five review
    platforms, because an unrecognised host is not a directory. Used as an
    identity test it passes `barrowneuro.org/our-team/` and every other
    doctor's profile besides.
    """
    return bool(url) and slug_agrees_with_name(url, provider_name)


def _states_disagree(page_location: Any, known_location: Any) -> bool:
    """Do two addresses name DIFFERENT states? False whenever either is unknown.

    The one geography check that is safe to make on identity. Comparing cities
    would reject a doctor whose profile lists the practice one suburb over, and
    comparing distance needs a ZIP both sides. A state is coarse enough that
    disagreement is real evidence and fine enough to separate Chandler, AZ from
    San Antonio, TX.
    """
    page = parse_location(str(page_location or "")).get("state")
    known = parse_location(str(known_location or "")).get("state")
    return bool(page and known and page != known)


# The deterministic readers. An observation carrying one of these was read off
# page markup by a regex; anything else came from the model.
_PARSER_SOURCES = frozenset({"listing_parser", "profile_parser"})


def _merge_parser_numbers(obs: Dict[str, Any], facts: Dict[str, Any]) -> None:
    """Apply a profile parse's rating/count over an existing observation.

    The profile parse owns BOTH numbers against the MODEL, and that rule is
    unchanged: splicing a parsed rating onto a model-read count manufactures a
    pair neither source stated, on exactly the shapes where the parser withholds
    a number deliberately (healthgrades template B, a vitals comparison column
    that failed its cross-check).

    The corroboration branch below fires only when `obs` was ITSELF written by
    a parser — which, at this call site, means a second parsed page whose URL
    canonicalises to one an earlier parsed page already claimed (http/https,
    www, trailing-slash duplicates that Tavily returns as distinct results). A
    thin duplicate must not erase what the fuller copy read.

    SCOPE CORRECTION (2026-08-04, same day it shipped): this guard was written
    against Dr. Hagevik's lost healthgrades count, narrated as "the profile
    parse lands on the listing row's observation". It does not — the listing
    observation lives on the PROVIDER and never enters `_apply_parsed_profiles`,
    which receives only the enrichment pass's own (model-written) observations,
    and the real listing observation carries no `extraction_source` besides.
    The two parser reads meet in `_merge_review_data`'s union, and the fix for
    the narrated failure is `_fill_corroborated_gaps` there. This stays as the
    duplicate-page guard, with the rule it shares: a parser-established number
    survives a read that is merely SILENT about it when every shared field
    agrees; on disagreement the newer page wins outright.
    """
    incoming = {"rating": facts.get("rating"), "review_count": facts.get("review_count")}
    shared = [
        (obs.get(field), value)
        for field, value in incoming.items()
        if obs.get(field) is not None and value is not None
    ]
    corroborates = (
        obs.get("extraction_source") in _PARSER_SOURCES
        and bool(shared)
        and all(existing == value for existing, value in shared)
    )
    for field, value in incoming.items():
        if value is None and corroborates and obs.get(field) is not None:
            # Keep the earlier parser's number; this page simply didn't state it.
            continue
        obs[field] = value


def _fill_corroborated_gaps(kept: Dict[str, Any], dropped: Dict[str, Any]) -> None:
    """Two reads of the SAME url collided in a union; fill what the kept one
    is silent about, when everything they share agrees.

    Every observation union here is "first URL wins", which answers the
    question it was built for — the same page must not count twice — and
    silently answers a second question it was never meant to decide: which
    READ of that page survives. Since a parsed listing row's observation
    started pointing at the doctor's own profile url, discovery and enrichment
    collide on that url BY CONSTRUCTION, and first-wins keeps whichever pass
    ran first, richer or not. Reproduced with Dr. Hagevik's shapes: discovery
    holding `4.8/None` and enrichment recovering `4.8/260` kept the None — so
    the count, lost once to a thin fetch and once to the chunk rule, was
    recoverable a third time and still discarded.

    The rule is the corroboration doctrine the chunk merge and the
    duplicate-page guard already follow: agreement on every shared numeric
    field says the two reads describe the same subject, and only then may one
    complete the other. Strictly additive — this never overwrites a stated
    number and never clears one, so on ANY disagreement the union's existing
    behaviour (first wins, untouched) is preserved. A collision with no shared
    field (rating-only meeting count-only) fills nothing: nothing establishes
    the two reads agree, and manufacturing a pair from two half-reads is the
    exact splice the parser rules refuse.
    """
    shared = []
    for field, parse in (("rating", _parse_rating), ("review_count", _first_int)):
        kept_value, dropped_value = parse(kept.get(field)), parse(dropped.get(field))
        if kept_value is not None and dropped_value is not None:
            shared.append((kept_value, dropped_value))
    if not shared or any(a != b for a, b in shared):
        return
    for field, parse in (("rating", _parse_rating), ("review_count", _first_int)):
        if parse(kept.get(field)) is None and parse(dropped.get(field)) is not None:
            kept[field] = parse(dropped.get(field))


def _apply_parsed_profiles(
    review_data: Dict[str, Any],
    parsed: Dict[str, Dict[str, Any]],
    provider_name: str,
    overlap: Callable[[str, str], float],
    provider_location: str = "",
) -> Dict[str, Any]:
    """Let a deterministic profile parse override the model, PER FIELD.

    The parser and the extractor read the SAME pages, and where the parser can
    read one its answer is reproducible while the model's is not — a 3,000-char
    anchor-window excerpt of a 15k page, scored by the cheapest model, moved a
    provider four ranks between two runs of one search. So the parser wins where
    it has an answer and the model fills the rest.

    "Per field" is the whole contract, and a per-PAGE trigger is the bug it
    replaces: a partial parse is the NORMAL case here — healthgrades template B
    states a rating with no total, vitals keeps its count in a heading and its
    rating in a link — so "the parser returned something, skip the model" would
    have thrown away a whole rating pair on any page that yielded only tenure,
    silently.

    One exception, and it runs the other way: when the parser reads a page's
    RATING or COUNT it owns BOTH for that page. Splicing a parsed rating onto a
    model-read count manufactures a pair neither source stated, and on the two
    shapes where the parser deliberately withholds a number — healthgrades
    template B, whose count is genuinely absent from the payload, and a vitals
    comparison table whose column failed its cross-check — letting the model
    supply the missing half would undo the refusal that keeps a stranger's
    stars off the card.

    Tenure and address are taken only when the page's own H1 agrees with the
    provider we asked about, at the same 0.5 threshold the observation identity
    check uses. They are page-level facts with no `source_url` to be checked
    downstream, so this is the only place that check can happen.
    """
    if not parsed:
        return review_data

    observations = [
        o for o in (review_data.get("review_observations") or []) if isinstance(o, dict)
    ]
    by_url: Dict[str, Dict[str, Any]] = {}
    for obs in observations:
        key = canonical_profile_url(obs.get("source_url"))
        if key:
            by_url.setdefault(key, obs)

    for url, facts in parsed.items():
        # A page in a DIFFERENT STATE is a different doctor, whatever the name
        # says. healthgrades holds a "Dr. Nicole Simpkins, MD" in San Antonio,
        # TX — Clinical Neurophysiology, 5.0 over 12 reviews — and the provider
        # on the card is Dr. Nicole Alyce Simpkins in Chandler, AZ. Every guard
        # in place passes it: the names overlap, the slug agrees, and the
        # enrichment query returns her because it searches the name. Nothing
        # looked at where the page said she practises, so a stranger 870 miles
        # away would have supplied a platform rating AND overwritten the
        # address, which then re-scores the distance.
        #
        # State-level and only on positive disagreement — a doctor practising
        # in two states is real, and a page that states no address must not be
        # penalised for it. The failure modes are not symmetric: accepting puts
        # someone else's stars on a card, rejecting costs one platform.
        if _states_disagree(facts.get("location"), provider_location):
            logger.info(
                "Rejected profile page for %r: %s states an address in a "
                "different state (%s vs %s)",
                provider_name, url, facts.get("location"), provider_location,
            )
            continue

        states_a_number = facts.get("rating") is not None or facts.get("review_count") is not None
        if states_a_number:
            obs = by_url.get(canonical_profile_url(url))
            if obs is None:
                obs = {"source_url": url}
                observations.append(obs)
                by_url[canonical_profile_url(url)] = obs
            _merge_parser_numbers(obs, facts)
            if facts.get("page_provider_name"):
                obs["page_provider_name"] = facts["page_provider_name"]
            obs["extraction_source"] = "profile_parser"

        page_name = str(facts.get("page_provider_name") or "")
        if page_name and overlap(page_name, provider_name) < 0.5:
            continue
        if facts.get("years_experience") is not None:
            review_data["years_experience"] = facts["years_experience"]
            # Every producer of a ranked number names itself — triage needs
            # "where did 28 years come from" answerable per provider, the same
            # question `location_source` answers for the address.
            review_data["experience_source"] = f"profile_parser:{url}"
        if facts.get("location"):
            # EVERY gate-passing page's address, not last-wins: two platforms
            # disagreeing about where a doctor practises (healthgrades: Sun
            # City West · card: Mesa, ~40 mi apart, both AZ) is invisible to
            # the state veto and undecidable in code — a satellite office and
            # a stale profile look identical. The candidates feed a FLAG,
            # never a rejection.
            review_data.setdefault("address_candidates", []).append(
                {"address": facts["location"], "source": f"profile_parser:{url}"}
            )
            review_data["address"] = facts["location"]
            # Named so a wrong address is traceable to the page it came off.
            # The model's address carries no URL at all, which is why the two
            # are recorded differently rather than both as "enrichment".
            review_data["address_source"] = f"profile_parser:{url}"
            # ...and `address_source_url` is the field the BACKFILL GUARD reads.
            # Both producers write `address` to one key, but only the model ever
            # wrote the URL beside it, so the guard was checking a parser-read
            # address against whichever page the MODEL had quoted — refusing a
            # good address outright when the model supplied none, and validating
            # it against an unrelated page when it did. Two providers came back
            # sharing one street address and one distance.
            #
            # It also closes the hole above: the H1 check is skipped when a page
            # states no H1 at all (`page_name` empty), and pointing the guard at
            # this page makes the URL slug carry the identity test instead.
            review_data["address_source_url"] = url

        # The page's whole LOCATION SET, not just its first block. A group
        # practice lists every office on each doctor's profile (the CORE
        # Institute spans the Phoenix metro), so "the address" is a SET and
        # forcing it into one field is what made a real office on a city
        # listing look like a contradiction of the profile. Every member is a
        # candidate under the SAME source URL — one page stays one vote in the
        # conflict resolver; what the members add is REACH: an office near the
        # listing's claim corroborates that AREA even when no two strings are
        # equal. Phones ride with their own block, because Dr. Vandian's card
        # showed the Gilbert address beside the Phoenix office's phone.
        for member in facts.get("locations") or []:
            member_address = str((member or {}).get("address") or "").strip()
            if not member_address:
                continue
            entry = {"address": member_address, "source": f"profile_parser:{url}"}
            if (member or {}).get("phone"):
                entry["phone"] = str(member["phone"]).strip()
            review_data.setdefault("address_candidates", []).append(entry)

    review_data["review_observations"] = observations
    return review_data


def _blended_platform_rating(observations) -> Optional[Dict[str, Any]]:
    """Count-weighted rating across platform-stated pairs, or None.

    The headline stays the single most authoritative page (traceable,
    clickable); the SCORE should hear every platform in proportion to its
    review mass — vitals 3.5 (16) alongside healthgrades 2.1 (13) is a
    ~2.9★ doctor, not a 3.5★ one. Honesty rules: platform pairs only (a
    rating without a count has no weight) and at least two pairs (one pair IS
    the headline — nothing to blend).

    It no longer declines when the pairs DISAGREE. That gate copied its
    reasoning from the headline ladder — "averaging a 1.2 against a 5.0 would
    manufacture a middle number nobody reported" — but the ladder declines on
    RATING-ONLY observations, which carry no counts and therefore offer no
    honest way to combine. Here every input has a count, and two things follow:

      * the count weighting already resolves the case the gate feared.
        1.2 (3 reviews) against 5.0 (200) blends to 4.94 — essentially the
        high-volume platform, correctly, not a fabricated midpoint;
      * when the gate DID fire, the fallback was worse than the blend. The
        scorer drops to `provider["rating"]`, the single headline, which the
        ladder picks as the largest-count pair — i.e. ONE of the disagreeing
        sources, at full strength. Declining to average two sources because
        they conflict, then adopting one of them, is not conservatism; it is
        cherry-picking.

    A third reason is structural: at three platforms with a two-pair minimum,
    the gate fires MORE often as coverage improves, pushing a better-evidenced
    provider back onto a single source.

    The disagreement is not discarded — it is returned as `spread`, so the
    caller can caveat a contested rating instead of hiding it. What is
    deliberately NOT done is widening the Bayesian shrinkage when the spread is
    large: that is the statistically right move and it needs a prior over
    between-platform variance nobody here has measured.
    """
    pairs = _platform_rating_pairs(observations)
    if len(pairs) < 2:
        return None
    ratings = [o["rating"] for o in pairs]
    total = sum(o["review_count"] for o in pairs)
    blended = sum(o["rating"] * o["review_count"] for o in pairs) / total
    return {
        "rating": round(blended, 1),
        "review_count": total,
        "platforms": len(pairs),
        # How far apart the platforms actually were. The blend no longer
        # declines on disagreement, so the disagreement has to travel WITH the
        # number instead of suppressing it — the scorer records it, and a wide
        # spread becomes a data caveat rather than a silent average.
        "spread": round(max(ratings) - min(ratings), 1),
    }

# Filler words stripped when distilling the user's free-text requirements
# into search keywords for the per-provider enrichment query
class DataGathererAgent:
    """Agent responsible for gathering healthcare provider data using Tavily search and Claude Haiku for extraction."""

    def __init__(self):
        """Initialize the data gatherer with Tavily, Anthropic, and optionally FHIR clients."""
        self.config = get_config()
        self.tavily_client = None
        self.anthropic_client = None
        self.fhir_client = None
        self._fhir_transformer = None
        # Per-SEARCH, not per-agent: the orchestrator reuses one gatherer, so
        # without the reset in `gather_providers` a contradiction from an
        # earlier search would be reported against the current one.
        self._identity_contradictions: List[Dict[str, Any]] = []
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize Tavily, Anthropic, and optionally FHIR clients."""
        try:
            if not self.config.TAVILY_API_KEY:
                raise ValueError("Tavily API key not found in configuration")
            if not self.config.ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not found in configuration")

            self.tavily_client = TavilyClient(api_key=self.config.TAVILY_API_KEY)
            self.anthropic_client = Anthropic(api_key=self.config.ANTHROPIC_API_KEY)

            # Initialize FHIR client when enabled
            if self.config.FHIR_ENABLED:
                try:
                    from fhir.client import create_fhir_client
                    from fhir.transformer import FHIRToProviderTransformer

                    self.fhir_client = create_fhir_client()
                    self._fhir_transformer = FHIRToProviderTransformer()
                    logger.info("FHIR client initialized (mock=%s)", self.config.FHIR_USE_MOCK)
                except Exception as fhir_err:
                    logger.warning("FHIR client initialization failed, continuing without FHIR: %s", fhir_err)
                    self.fhir_client = None

            logger.info("Data gatherer clients initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize data gatherer clients: {e}")
            raise

    def _build_search_query(self, specialty: str, location: str, insurance: Optional[str] = None) -> str:
        """Build optimized search query for healthcare providers.

        Args:
            specialty: Medical specialty (e.g., "Neurology")
            location: Location (e.g., "Phoenix, AZ")
            insurance: Insurance type (optional)

        Returns:
            Optimized search query string
        """
        query_parts = []

        # Core search terms with review focus
        query_parts.append(f"{specialty} specialists in {location}")
        query_parts.append("reviews ratings patient feedback")

        # Insurance is deliberately NOT part of the discovery query: payer
        # names bias Tavily toward sparse plan-directory pages and act as an
        # implicit filter at the recall stage. Insurance evidence comes from
        # extraction + enrichment and is labeled, never searched-by.
        _ = insurance

        query = " ".join(query_parts)
        logger.debug(f"Built search query: {query}")
        return query

    def _candidate_queries(self, specialty: str, location: str) -> List[Dict[str, Any]]:
        """Discovery query specs for one city, spanning phrasings for recall.

        One phrasing returns a pool pre-clustered by whatever "best-of" pages
        exist; three surface distinct providers. The third spec is domain-
        restricted to the independent review platforms and always searches
        deep: naming platforms in the query TEXT only made their pages
        probable, so platform ratings used to arrive at enrichment (top-8
        only) or not at all. Restricting by domain guarantees listing/profile
        pages — which carry the ratings in the page body — land in the
        extraction pool itself. Payer/ZIP deliberately absent.
        """
        return [
            {
                "query": f"best {specialty} listing near {location} patient reviews ratings",
                "include_domains": [domain],
                # basic, not advanced: one domain per call holds at basic depth
                # (measured — 5 results, 5 in-domain, on all three), and the
                # pages that come back are directory listings whose entries the
                # listing parsers read deterministically. Half the credits of the
                # three advanced queries this replaces.
                "search_depth": "basic",
            }
            for domain in _LISTING_DOMAINS
        ]

    def _discover_candidates(self, queries: List[Union[str, Dict[str, Any]]], max_results: int) -> List[Dict[str, Any]]:
        """Run discovery queries in parallel; round-robin merge, dedup by URL.

        Accepts plain query strings or spec dicts ({"query", optional
        "include_domains"/"search_depth"}) so individual phrasings can be
        domain-restricted or searched deeper without a parallel-list API.

        The interleave (q1[0], q2[0], q3[0], q1[1], ...) is load-bearing:
        downstream extraction caps the merged list ([:20] then [:18]), so a
        sequential merge would let the first query's full page of results
        crowd every other phrasing out of the head — silently discarding the
        recall the extra queries paid for. First URL occurrence wins. One
        merged set feeds a single extraction call.
        """
        specs = [{"query": q} if isinstance(q, str) else q for q in queries]
        with ThreadPoolExecutor(max_workers=min(len(specs), 4)) as executor:
            per_query = [
                results or []
                for results in executor.map(
                    lambda spec: self._search_providers(
                        spec["query"],
                        max_results=max_results,
                        include_raw_content=True,
                        include_domains=spec.get("include_domains"),
                        search_depth=spec.get("search_depth"),
                    ),
                    specs,
                )
            ]

        merged: List[Dict[str, Any]] = []
        seen_urls = set()
        for rank in range(max((len(results) for results in per_query), default=0)):
            for results in per_query:
                if rank >= len(results):
                    continue
                result = results[rank]
                url = result.get("url", "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                merged.append(result)
        logger.info(f"Discovery merged {len(merged)} unique pages from {len(queries)} queries")
        return merged

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

        # If no code blocks, try to find JSON array [...]
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            return json_match.group(0).strip()

        return response_text.strip()

    def _search_providers(self, query: str, max_results: int = 10, include_raw_content: bool = False, include_domains: Optional[List[str]] = None, search_depth: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for providers using Tavily API.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            include_raw_content: Fetch full page text per result. Both callers
                pass True today: the candidate search needs page bodies because
                directory-style results name providers only in the body (a
                snippet-only pass extracted zero providers on the canonical
                demo query), and the per-provider enrichment search exists to
                read the actual review pages.
            include_domains: Restrict results to these domains (platform-
                targeted calls only; general discovery stays unrestricted so
                all relevant sources, incl. practice sites for insurance
                lists, can surface).
            search_depth: Per-call override of the global depth knob. The
                platform-targeted calls pass "advanced" — they are few, and
                they hit exactly the structured review pages where Tavily's
                deeper retrieval decides whether the profile page comes back
                at all.

        Returns:
            List of search results from Tavily
        """
        effective_depth = search_depth or self.config.TAVILY_SEARCH_DEPTH
        search_kwargs: Dict[str, Any] = {
            "query": query,
            "search_depth": effective_depth,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": include_raw_content,
        }
        if include_domains:
            search_kwargs["include_domains"] = list(include_domains)
        # `content` is relevance-selected by Tavily and has always been part of
        # both extraction prompts, so this knob steers what the extractor reads.
        #
        # Sent at BOTH depths. The parameter is widely documented as advanced-
        # only, and this line was first written to withhold it on basic so the
        # (since-removed) fast-demo toggle could not trip an API rejection.
        # Measured instead of assumed, on one query over the review platforms:
        #
        #     basic     cps=1  4,724 chars  ->  cps=5  37,298 chars
        #     advanced  cps=1  7,300 chars  ->  cps=5  37,298 chars
        #
        # Basic accepts it, honours it, and lands on the identical content
        # payload. So the guard would have cost any basic-depth deployment an
        # 8x richer extraction input to prevent a rejection that does not
        # happen — still live via the TAVILY_SEARCH_DEPTH env knob.
        search_kwargs["chunks_per_source"] = self.config.TAVILY_CHUNKS_PER_SOURCE

        # One retry on transient failures — a single network blip should not
        # become "Search failed" for the user (the whole workflow dies on []).
        for attempt in (1, 2):
            try:
                response = self.tavily_client.search(**search_kwargs)

                # Record the depth actually used, not the config value —
                # advanced costs 2 credits and overrides would misprice.
                get_cost_tracker().record_tavily(
                    depth=effective_depth, agent="data_gatherer"
                )

                results = response.get("results", [])
                logger.info(f"Found {len(results)} search results for query: {query}")
                return results

            except Exception as e:
                if attempt == 1:
                    logger.warning(f"Tavily search failed (attempt 1), retrying: {e}")
                    time.sleep(2)
                else:
                    logger.error(f"Tavily search failed after retry: {e}")
        return []

    def _extract_provider_data(self, search_results: List[Dict[str, Any]], specialty: str, location: str) -> List[Dict[str, Any]]:
        """Extract structured provider data using Claude Haiku.

        Selects and orders the pages, then reads them in `_DISCOVERY_SHARDS`
        CONCURRENT extraction calls. One call over 18 page excerpts was 25.3s
        of a 97.8s run — the single largest step — and the pages are
        independent: nothing in the prompt compares one page against another,
        and `_dedupe_providers` already runs downstream over the merged list,
        so a provider named on pages in both shards merges exactly as one
        named twice in a single response always did.

        Shards are CONTIGUOUS, not round-robin. `prioritized_results` is
        already an interleave of (review-platform, everything-else) with
        period 2, so dealing it would hand one call every profile page and the
        other every directory page — undoing round 12's fix rather than
        preserving it. See utils/shard.

        Args:
            search_results: Raw search results from Tavily
            specialty: Requested specialty
            location: Requested location

        Returns:
            List of structured provider data dictionaries
        """
        try:
            # Drop near-zero-relevance results before they consume a capped
            # slot ("a 0.11 junk page rides ahead of a 0.74 page purely by
            # domain"). Floor only — order stays the round-robin interleave.
            # If Tavily omitted scores, or the floor would empty the list,
            # keep everything.
            relevant = [
                r for r in search_results
                if not isinstance(r.get("score"), (int, float)) or r["score"] >= _MIN_RELEVANCE_SCORE
            ]
            top_results = (relevant or search_results)[:20]

            # Review-heavy sources (Healthgrades, Vitals, US News) carry
            # detailed patient feedback AND insurance-acceptance lists, so they
            # keep first pick. But they are PROFILE pages naming ONE physician,
            # while the pages below them are directories and "top N" listicles
            # naming a dozen — and this pass is paid to FIND PROVIDERS.
            #
            # Round-robin rather than hoisting. The guarantee is bounded, not
            # dramatic: because the upstream merge already interleaves the three
            # queries, only about two blocks change hands in a typical pool.
            # What it removes is the WORST case — when the domain-restricted
            # query fills the list with profile pages, hoisting can evict every
            # many-name page, and the pass that fills the candidate pool is left
            # reading pages that name one provider each.
            #
            # Ordering priority is unchanged: review-heavy still leads.
            review_heavy = [r for r in top_results if any(domain in r.get('url', '').lower() for domain in _REVIEW_PLATFORM_DOMAINS)]
            other_results = [r for r in top_results if not any(domain in r.get('url', '').lower() for domain in _REVIEW_PLATFORM_DOMAINS)]

            interleaved = []
            for pair in zip_longest(review_heavy, other_results):
                interleaved.extend(r for r in pair if r is not None)
            prioritized_results = interleaved[:_DISCOVERY_MAX_BLOCKS]

            # Nothing used to be logged between the merged-page count and the
            # extracted-provider count, which is precisely the interval where
            # the caps live — so "the ring fired" could never be attributed to
            # a thin web or to us reading a third of what we paid for.
            logger.info(
                "Discovery extraction: %d of %d pages reach the prompt "
                "(%d review-platform, %d other), %d chars of excerpt each",
                len(prioritized_results), len(search_results),
                len(review_heavy), len(other_results), _DISCOVERY_EXCERPT_BUDGET,
            )

            # PARSER FIRST. A directory page's entries are highly structured and
            # repetitive, which is what a regex is for; reading them with an LLM
            # over a 3,000-char anchored excerpt recovered a handful per page and
            # a different handful on each run. On one live search 4 of 8 fetched
            # pages yielded nothing and a provider was marked `no_profile_found`
            # while 81,000 characters of her pages sat in memory.
            #
            # The LLM is the FALLBACK, not the replacement: `parse_listing`
            # covers three platforms' current markup, and a site redesign must
            # degrade to the previous behaviour rather than empty the pool. So a
            # page the parser could not read still goes to a shard.
            parsed: List[Dict[str, Any]] = []
            unparsed: List[Dict[str, Any]] = []
            for result in prioritized_results:
                rows = parse_listing(result.get("url"), result.get("raw_content") or "")
                rows = [r for r in rows if _specialty_is_compatible(r.get("specialty"), specialty)]
                if rows:
                    parsed.extend(_listing_row_to_provider(r, specialty) for r in rows)
                else:
                    unparsed.append(result)
            if parsed:
                logger.info(
                    "Listing parsers read %d providers from %d of %d pages; "
                    "%d page(s) fall back to extraction",
                    len(parsed), len(prioritized_results) - len(unparsed),
                    len(prioritized_results), len(unparsed),
                )
            if not unparsed:
                return parsed

            shards = contiguous_shards(
                unparsed,
                _DISCOVERY_SHARDS if len(unparsed) >= _MIN_PAGES_TO_SHARD else 1,
            )
            if len(shards) <= 1:
                return parsed + (self._extract_page_shard(unparsed, specialty, location) or [])

            logger.info(
                "Discovery extraction split across %d concurrent calls (%s pages each)",
                len(shards), "/".join(str(len(s)) for s in shards),
            )
            with ThreadPoolExecutor(max_workers=len(shards)) as executor:
                futures = [
                    executor.submit(self._extract_page_shard, shard, specialty, location)
                    for shard in shards
                ]
                merged: List[Dict[str, Any]] = list(parsed)
                for future in futures:
                    merged.extend(future.result() or [])
            logger.info(
                "Discovery extraction merged %d providers (%d parsed, rest from "
                "%d shards; duplicates are resolved by _dedupe_providers)",
                len(merged), len(parsed), len(shards),
            )
            return merged

        except Exception as e:
            logger.error(f"Provider data extraction failed: {e}")
            return []

    def _extract_page_shard(self, pages: List[Dict[str, Any]], specialty: str, location: str) -> List[Dict[str, Any]]:
        """Run ONE extraction call over an already-selected block of pages.

        Split out of `_extract_provider_data` so several can run concurrently;
        page selection, ordering and the block cap all happen there, once.

        Args:
            pages: Prioritized search results — this shard's pages only
            specialty: Requested specialty
            location: Requested location

        Returns:
            List of structured provider data dictionaries from these pages
        """
        try:
            # Sanitize inputs to prevent prompt injection
            safe_specialty = PromptSanitizer.escape_for_prompt(specialty)
            safe_location = PromptSanitizer.escape_for_prompt(location)
            prioritized_results = pages

            # Prepare content for Claude with clear delimiters: the snippet
            # plus a capped excerpt of the page body when available. The
            # review-biased query surfaces directory/"top N" pages whose
            # snippets name no individual provider — the page text does.
            blocks = []
            for result in prioritized_results:
                block = (
                    f"Title: {result.get('title', '')}\n"
                    f"URL: {result.get('url', '')}\n"
                    f"Content: {clip_words(result.get('content', ''), _CONTENT_MAX_CHARS)}"
                )
                # Content-aware excerpt: boilerplate-stripped windows centered
                # where the specialty/review vocabulary actually hits, instead
                # of a blind head-truncation that mostly captured nav chrome
                raw_content = build_excerpt(
                    str(result.get("raw_content") or ""),
                    anchors=_anchors_for(
                        result.get("url"),
                        [safe_specialty, "review", "rating", "dr.", "patients"],
                    ),
                    budget=_DISCOVERY_EXCERPT_BUDGET,
                    max_windows=_DISCOVERY_EXCERPT_WINDOWS,
                )
                if raw_content:
                    block += f"\nFull page text (excerpt): {raw_content}"
                blocks.append(block)
            results_text = "\n\n".join(blocks)

            # Use structured prompt with XML-style delimiters for clarity
            prompt = f"""Extract healthcare provider information from the search results provided below.

You are a data extraction specialist. Your task is to extract healthcare provider information ONLY from the search results section.

<task_parameters>
Target Specialty: {safe_specialty}
Target Location: {safe_location}
</task_parameters>

<output_format>
Return ONLY a JSON array of provider objects with these fields:

REQUIRED FIELDS:
- name: Provider's full name
- specialty: Medical specialty
- location: Full address or city/state
- phone: Phone number (format: XXX-XXX-XXXX, or null if not stated on a page)
- rating: Rating out of 5 (numeric, e.g., 4.5, or 0 if not found)
- review_count: Number of patient reviews (integer, e.g., 127, or null if not found)
- review_summary: 3-4 sentence summary of patient feedback covering most praised aspects, common complaints, and overall experience themes (string, or "No reviews available" if no review content found)
- review_sentiment: Overall sentiment from reviews: "positive", "mixed", or "negative" (or "unknown" if no review content available)
- review_source_url: The URL of the result block the review information (rating/count/summary) came from — copy it exactly from that block's URL line; null if no review data was found
- review_observations: One entry per result block that explicitly STATES a rating and/or a review total for THIS provider, transcribed exactly: [{{"source_url": "<that block's URL line, copied exactly>", "rating": 4.0, "review_count": 31}}]. Bare JSON numbers only (4.0, not "4.0/5"). review_count null unless that page states a total ("Based on 31 reviews", "(271)"); rating null unless stated. NEVER derive a rating or count from star-percentage distributions — transcribe stated values only. [] if no block states either

HIGHLY IMPORTANT FIELDS (extract if ANY information is available):
- insurance_accepted: List of insurance names/types mentioned (e.g., ["Blue Cross", "Aetna", "Medicare"])
  * Search carefully for insurance mentions in the content
  * Look for phrases like "accepts", "takes", "insurance plans"
  * Even partial matches are valuable
  * If no insurance info found, use empty array []
- insurance_source_url: The URL of the result block the insurance names came from — copy it exactly from that block's URL line; null if insurance_accepted is empty. A plan list with no source cannot be attributed on the card, so never omit this when you populate insurance_accepted

- distance: Distance in miles from {location} ONLY if a distance is explicitly stated in the text (e.g. "2.3 miles away")
  * If no distance is explicitly stated, use null (not "N/A") — NEVER estimate or infer one
  * Distances are computed separately from addresses, so a complete address in the location field is more valuable than a guess

OPTIONAL FIELDS:
- services: List of services offered
- website: Provider's website URL
- years_experience: Years in practice (if mentioned)
- education: Medical school/credentials (if mentioned)
</output_format>

<extraction_rules>
1. Only extract providers matching the target specialty
2. Only extract providers in or near the target location
3. Directory, "best of", and "top N" listing pages often name several individual providers within the page text — extract EACH named provider matching the target specialty and location, even when per-provider detail is thin (but never invent a provider that is not named in the results)
4. Extract phone numbers in XXX-XXX-XXXX format
5. Convert ratings to numeric format (e.g., "4.5 stars" → 4.5)
6. CRITICAL: Search thoroughly for insurance and distance information - these are key factors
7. For insurance: even mentions like "accepts most major insurance" → ["Most major insurance"]
8. REVIEW ANALYSIS - CRITICAL INSTRUCTIONS:
   * ONLY extract review summaries if you find ACTUAL PATIENT FEEDBACK in the search results
   * Look for quotes, comments, or testimonials from patients (e.g., "very caring", "long wait", "listens well")
   * DO NOT make up generic statements like "has experience in X" - that's not a review summary
   * MANDATORY: If you find actual patient comments, create a comprehensive 3-4 sentence summary covering:
     1. What patients most appreciate (strengths like expertise, bedside manner, communication)
     2. What patients complain about or areas for improvement (concerns like wait times, billing, accessibility)
     3. Overall experience patterns and recommendations (professionalism, office environment, staff helpfulness)
   * Determine sentiment: "positive" (mostly good), "mixed" (both good and bad), or "negative" (mostly bad)
   * If NO actual patient review content found, use "No reviews available" for summary and "unknown" for sentiment
9. Return ONLY a JSON array of provider objects
10. If no clear providers found, return empty array []
11. Do NOT include any explanatory text, ONLY return the JSON array
</extraction_rules>

<search_results>
{results_text}
</search_results>

Response (JSON array only):"""

            llm_started = time.perf_counter()
            # Scaled, floored at the old flat 8000 — see the constants'
            # comment. The floor is load-bearing: round 12's ceiling raise
            # existed because truncation is fatal here, and a formula that
            # could dip below it would quietly reintroduce the exact
            # failure at small block counts.
            output_budget = max(8000, min(
                _DISCOVERY_TOKENS_BASE + _DISCOVERY_TOKENS_PER_BLOCK * len(pages),
                _DISCOVERY_TOKENS_MAX,
            ))
            response = self.anthropic_client.messages.create(
                model=self.config.GATHERER_MODEL,
                max_tokens=output_budget,
                messages=[{"role": "user", "content": prompt}]
            )

            # Round 10 added this check to the ENRICHMENT extraction and not
            # here, though this is the call that raised its own ceiling because
            # truncation was known to be fatal. A cut array has no closing "]",
            # the repair regex below cannot match, and `[]` is returned — so a
            # home pool of ZERO looks exactly like "the pages named nobody",
            # and the ring expands on a bug rather than on thin coverage.
            if getattr(response, "stop_reason", None) == "max_tokens":
                logger.warning(
                    "Discovery extraction hit max_tokens at %d — the response is "
                    "truncated; complete entries will be salvaged and only the "
                    "cut entry lost", output_budget
                )

            in_tokens, out_tokens = safe_usage(response)
            get_cost_tracker().record_llm(
                self.config.GATHERER_MODEL, in_tokens, out_tokens,
                agent="data_gatherer", duration_s=time.perf_counter() - llm_started
            )

            response_text = response.content[0].text.strip()

            # Extract JSON from response using helper
            try:
                # Extract JSON from markdown or raw response
                json_str = self._extract_json_from_response(response_text)

                # Try to parse JSON
                try:
                    providers = json.loads(json_str)
                except json.JSONDecodeError as parse_error:
                    # Try to fix common JSON issues
                    logger.warning(f"Initial JSON parse failed: {parse_error}. Attempting to fix...")

                    fixed_json = json_str

                    # Fix 1: Remove trailing commas before closing brackets/braces
                    fixed_json = re.sub(r',\s*([}\]])', r'\1', fixed_json)

                    # Fix 2: Remove trailing comma at end of arrays/objects (more aggressive)
                    fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)

                    # Fix 3: Try to extract only the JSON array if there's text after it
                    array_match = re.search(r'(\[[\s\S]*?\])\s*[^,}\]]*$', fixed_json)
                    if array_match:
                        fixed_json = array_match.group(1)

                    # Try parsing the fixed version
                    try:
                        providers = json.loads(fixed_json)
                        logger.info("Successfully parsed after fixing JSON syntax")
                    except json.JSONDecodeError as second_error:
                        # LAST RESORT, deliberately after full parse and regex
                        # repair: salvage keeps only complete objects, so it
                        # loses the cut entry and must never outrank a
                        # successful whole-array parse. Before this existed a
                        # truncated response returned [] here, the home pool
                        # read ZERO, and the ring rebuilt it from other
                        # cities — total loss where tail loss was available.
                        # Walking the raw text as the fallback matters: on
                        # truncation _extract_json_from_response finds no
                        # closing bracket and hands back the whole response,
                        # but a fenced-and-truncated reply can differ, and
                        # the walker is indifferent to surrounding prose.
                        salvaged = (salvage_json_objects(json_str)
                                    or salvage_json_objects(response_text))
                        if salvaged:
                            logger.warning(
                                "Salvaged %d complete provider object(s) from an "
                                "unparseable extraction response (stop_reason=%s); "
                                "entries after the cut are lost",
                                len(salvaged),
                                getattr(response, "stop_reason", None),
                            )
                            providers = salvaged
                        else:
                            logger.error(f"Could not recover from JSON error after fixing: {second_error}")
                            logger.debug(f"Original error: {parse_error}")
                            logger.debug(f"Response preview: {response_text[:1000]}...")
                            logger.debug(f"Fixed JSON preview: {fixed_json[:1000]}...")
                            return []

                # Validate that it's a list
                if not isinstance(providers, list):
                    logger.warning("Claude response was not a JSON array")
                    return []

                # Clean and validate provider data
                cleaned_providers = []
                for provider in providers:
                    if isinstance(provider, dict) and provider.get("name"):
                        # Safe float conversion helper
                        def safe_float(value, default=0.0):
                            """Convert value to float, handling 'N/A' and invalid values."""
                            if value is None or value == "" or str(value).upper() == "N/A":
                                return default
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                return default

                        # Ensure required fields
                        cleaned_provider = {
                            "name": str(provider.get("name", "")),
                            "specialty": str(provider.get("specialty", specialty)),
                            "location": str(provider.get("location", location)),
                            # "N/A" is a legacy model habit — store missing
                            # as empty so the card's truthiness check works
                            # and the enrichment backfill can fill it
                            "phone": "" if str(provider.get("phone", "") or "").strip().upper() in ("", "N/A", "NONE", "NULL")
                            else str(provider.get("phone", "")).strip(),
                            "rating": safe_float(provider.get("rating"), 0.0),
                        }

                        # Review count: keep what was found, never invent one.
                        # (A fabricated count used to inflate the Bayesian
                        # rating confidence downstream; None is handled there
                        # with a conservative prior.)
                        # `_first_int`, not a bare `int()`: this was the one
                        # count path that never got the prose-tolerant parser,
                        # and it is the exact silent rejection that parser's
                        # docstring exists to describe — `int("31 reviews")`
                        # and `int("1,234")` both raise, so the count was
                        # discarded and the provider scored as if no review
                        # volume had ever been stated.
                        review_count = provider.get("review_count")
                        cleaned_provider["review_count"] = (
                            _first_int(review_count) if review_count else None
                        )

                        # Add review summary and sentiment
                        cleaned_provider["review_summary"] = str(provider.get("review_summary", "No reviews available"))
                        cleaned_provider["review_sentiment"] = str(provider.get("review_sentiment", "unknown"))

                        # Provenance: which page the review numbers came from
                        if provider.get("review_source_url"):
                            cleaned_provider["review_source_url"] = str(provider["review_source_url"])
                        # Carried from the candidate pass too, now that the
                        # discovery prompt asks for it. Without this every
                        # discovery-sourced insurance list was structurally
                        # unattributable, and the cache then froze that state
                        # in for the whole TTL.
                        if provider.get("insurance_source_url"):
                            cleaned_provider["insurance_source_url"] = str(provider["insurance_source_url"])

                        # Per-platform observations from the discovery pass.
                        # Without these the enrichment tier-1 predicate reads
                        # zero pairs for EVERY provider, so "needs a second
                        # opinion" is unconditionally true and the budget is
                        # spent by rank instead of by need.
                        headline, observations = _select_review_observation(
                            provider.get("review_observations")
                        )
                        if observations:
                            cleaned_provider["review_observations"] = observations

                        # Add optional fields
                        for field in ["insurance_accepted", "services", "website", "education"]:
                            if provider.get(field):
                                cleaned_provider[field] = provider[field]

                        # years_experience must be an INT here. The candidate
                        # prompt doesn't constrain the type (the enrichment one
                        # asks for "a bare JSON number"), so this pass returns
                        # "26 years" or "30+" — which float() rejects, so the
                        # scorer imputed EXPERIENCE_UNKNOWN and a 26-year
                        # physician was scored as tenure-unknown. Worse, the
                        # enrichment backfill is gated on the field being None,
                        # so the unparseable string permanently blocked the
                        # real number the profile page states.
                        years = _first_int(provider.get("years_experience"))
                        if years is not None:
                            cleaned_provider["years_experience"] = years

                        # Handle distance separately with safe float conversion
                        if provider.get("distance"):
                            cleaned_provider["distance"] = safe_float(provider.get("distance"), None)

                        cleaned_providers.append(cleaned_provider)

                logger.info(f"Extracted {len(cleaned_providers)} providers from search results")
                return cleaned_providers

            except Exception as inner_error:
                logger.error(f"Error processing provider data: {inner_error}")
                return []

        # Swallowing here is what keeps one bad shard from costing the run:
        # the orchestrator reads `future.result()`, so an exception escaping
        # this method would propagate and return [] for EVERY page, not just
        # this shard's. Named distinctly from the orchestrator's own handler
        # so the log says which of the two failed.
        except Exception as e:
            logger.error(f"Provider extraction failed for a {len(pages)}-page shard: {e}")
            return []

    @staticmethod
    def _field_richness(provider: Dict[str, Any]) -> int:
        """Count meaningfully-populated fields, for picking a dedup survivor."""
        empty = (None, "", [], 0, 0.0, "No reviews available", "unknown")
        return sum(1 for value in provider.values() if value not in empty)

    def _record_identity_contradiction(
        self, provider: Dict[str, Any], kept: Dict[str, Any], overlap: float
    ) -> None:
        """Two same-named records the profile URLs prove are different people.

        This is the ONE rule in dedupe that can CREATE a duplicate rather than
        remove one. If a platform ever publishes two URLs for a single doctor,
        the veto splits that person in two and it surfaces as a mystery
        duplicate on the results page with no visible cause. Nobody has checked
        whether healthgrades does this, so the record is how it gets FOUND
        rather than guessed at weeks later.

        Recorded only when DECISIVE — the caller consults the veto after the
        names already cleared the merge threshold, so every entry here is a
        pair that name-matching alone would have merged. It used to record
        every same-platform URL difference "so the ordinary case confirms the
        rule is working": on a parsed-listing pool of ~100 candidates that was
        3,363 pairs of obviously-different doctors, which buried the decisive
        rows and put a 3,363-row JSON dump on the developer panel. The
        `names_agreed` key stays (computed, not hardcoded) because three
        surfaces filter on it and the audit log's schema should read the same
        across the change.
        """
        entry = {
            "name": provider.get("name", "Unknown"),
            "other_name": kept.get("name", "Unknown"),
            "url": provider.get("profile_url"),
            "other_url": kept.get("profile_url"),
            "domain": source_domain(provider.get("profile_url")),
            "name_overlap": round(overlap, 2),
            # The interesting half. Below the threshold this is just two
            # different doctors on one platform; at or above it, the names
            # agreed and only the URL kept them apart.
            "names_agreed": overlap >= _DEDUP_NAME_THRESHOLD,
        }
        self._identity_contradictions.append(entry)
        if entry["names_agreed"]:
            logger.warning(
                "Identity contradiction: %r and %r share a name (overlap %.2f) but hold "
                "DIFFERENT %s profile URLs (%s vs %s) — kept as separate providers. If "
                "this platform publishes two URLs for one doctor, this is where that "
                "shows up as a duplicate.",
                entry["name"], entry["other_name"], overlap, entry["domain"] or "platform",
                entry["url"], entry["other_url"],
            )
        else:
            logger.info(
                "Identity contradiction: %r vs %r on %s — different profile URLs, "
                "names also differ (overlap %.2f)",
                entry["name"], entry["other_name"], entry["domain"] or "platform", overlap,
            )

    def _dedupe_providers(self, providers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge same-person entries extracted from different pages.

        "Dr. Pritish Pawar" and "Pritish Pawar, MD" must not compete as two
        candidates. The richer entry survives; its empty fields are filled
        from the duplicate.
        """
        empty = (None, "", [], 0, 0.0)
        deduped: List[Dict[str, Any]] = []

        for provider in providers:
            # PASS 1 — certainty. Two records sharing a platform's own profile
            # link are the same person; names are not consulted at all.
            #
            # A separate pass, not a condition inside the loop below: the order
            # entries were appended is arbitrary, so a single loop that breaks
            # on the first hit lets a weak NAME match found early beat an exact
            # URL match found later. Pass 1 must see the whole list first.
            match_idx = next(
                (
                    i for i, kept in enumerate(deduped)
                    if canonical_profile_url(provider.get("profile_url"))
                    and canonical_profile_url(provider.get("profile_url"))
                    == canonical_profile_url(kept.get("profile_url"))
                ),
                None,
            )

            # PASS 2 — inference, with a VETO. Same name threshold as before,
            # but an entry whose URL CONTRADICTS this one must not merge even
            # when the names agree: a different path on the same platform
            # proves two different physician records, and no name similarity
            # can outvote that. "Dr. J Kim" and "Dr. Jane Kim" overlap 1.0.
            #
            # The name is scored FIRST and the veto consulted only when it
            # would merge. The veto used to run first and record every firing,
            # which recorded the ORDINARY case at full fidelity: on a
            # parsed-listing pool of ~100 candidates that was 3,363 pairs of
            # obviously-different doctors ("Dr. Medaa" vs "Dr. Pillai",
            # overlap 0) — pool-size arithmetic, not discovery — burying the
            # 0-3 decisive rows the record exists to surface and rendering a
            # 3,363-row JSON dump in the developer panel. The truth table is
            # unchanged by the reorder: below the threshold neither order
            # merges, above it the veto still blocks — so every recorded
            # contradiction is now DECISIVE by construction.
            if match_idx is None:
                for i, kept in enumerate(deduped):
                    overlap = self._name_token_overlap(provider.get("name", ""), kept.get("name", ""))
                    if overlap < _DEDUP_NAME_THRESHOLD:
                        continue
                    if urls_contradict(provider.get("profile_url"), kept.get("profile_url")):
                        self._record_identity_contradiction(provider, kept, overlap)
                        continue
                    match_idx = i
                    break

            if match_idx is None:
                deduped.append(provider)
                continue

            kept = deduped[match_idx]
            survivor, duplicate = (kept, provider) if self._field_richness(kept) >= self._field_richness(provider) else (provider, kept)
            for key, value in duplicate.items():
                # Evidence fields UNION. Fill-if-empty discards the
                # duplicate's list whenever the survivor has one of its own,
                # and appearing on two directory pages is the NORMAL case —
                # so a provider found on healthgrades AND vitals kept one
                # platform pair instead of two, halving platform coverage
                # before enrichment ran and starving the cross-platform
                # blend, which needs two.
                if key in _UNION_ON_DEDUPE:
                    merged = _union_evidence(survivor.get(key), value)
                    if merged:
                        survivor[key] = merged
                    continue
                current = survivor.get(key)
                placeholder = (
                    current in empty
                    or (key == "review_summary" and current == "No reviews available")
                    or (key == "review_sentiment" and current == "unknown")
                )
                if placeholder and value not in empty:
                    survivor[key] = value

            # `discovery_source` is the one field where the SURVIVOR's value is
            # the wrong answer. The survivor is chosen by field richness, not by
            # which pass found them, so a ring-sourced entry can outlive the
            # home-sourced entry for the same doctor — and the merged provider
            # would then be counted as something the ring ADDED.
            #
            # A doctor the home city already surfaced was discoverable without
            # the ring. That is the entire question this field exists to answer:
            # what did the extra searches actually buy? Home wins, always.
            if _DISCOVERY_HOME in (kept.get("discovery_source"),
                                   provider.get("discovery_source")):
                survivor["discovery_source"] = _DISCOVERY_HOME

            deduped[match_idx] = survivor
            logger.info(
                "Deduped provider '%s' into '%s'", duplicate.get("name"), survivor.get("name")
            )

        if len(deduped) < len(providers):
            logger.info("Dedup: %d extracted -> %d unique providers", len(providers), len(deduped))
        return deduped

    def _extract_review_data_only(self, search_results: List[Dict[str, Any]], provider_name: str, specialty: str = "", provider_location: str = "", prior_summary: str = "") -> Dict[str, Any]:
        """Extract only review-related data for a specific provider using Claude Haiku.

        Args:
            search_results: Search results from Tavily
            provider_name: Name of the provider to extract reviews for
            specialty: The provider's specialty — identity support only;
                portals label one doctor under adjacent specialties, so a
                differing label alone must never reject a result
            provider_location: The provider's known city/address — the
                strongest same-person signal for the identity check
            prior_summary: The summary the candidate pass already wrote, if
                any. The merge overwrites review_summary wholesale, and this
                pass only sees its own search's pages — so without this the
                narrative silently narrowed to one platform while the ratings
                beside it spanned several.

        Returns:
            Dictionary with review_summary, review_sentiment, and review_count
        """
        # Deterministic read of the profile pages, BEFORE the model runs and
        # outside the try — so a failed or unparseable Anthropic response still
        # returns whatever the parsers could establish. Previously an API error
        # here cost the provider every number on every page that was fetched.
        parsed_profiles = _parse_profile_pages(search_results)

        try:
            # Prepare content for Claude: the search snippet plus an excerpt
            # of the actual page text when available — snippets alone are one
            # or two sentences and structurally under-read reviews.
            # Same relevance floor as the candidate pass (never a sort key).
            relevant = [
                r for r in search_results
                if not isinstance(r.get("score"), (int, float)) or r["score"] >= _MIN_RELEVANCE_SCORE
            ]
            # Windows anchor on the provider's NAME as the PRIORITY (identity
            # guard doubled as excerpt targeting — generic review vocabulary
            # is only the fallback, or it would drown the name)
            name_tokens = [t for t in re.findall(r"[A-Za-z]+", provider_name) if t.lower() not in {"dr", "md", "do", "np", "pa"}]
            name_anchors = [provider_name]
            if len(name_tokens) >= 2:
                # "Andrea An" — the credential-free spelling a page header uses.
                # `provider_name` arrives as "Dr. Andrea An, MD" and matches
                # literally, so a header written without the title matched none
                # of the priority anchors and left the bare surname steering
                # them alone.
                name_anchors.append(f"{name_tokens[0]} {name_tokens[-1]}")
            if name_tokens:
                name_anchors.extend(_surname_anchors(name_tokens[-1]))
            vocab_anchors = ["review", "rating", specialty]

            # Independent review platforms lead the input; a provider's own
            # site (rich SEO, thin credibility) goes last so it can't hog the
            # model's attention
            platform_blocks_src, other_blocks_src = [], []
            for result in (relevant or search_results):
                url_lower = str(result.get("url", "")).lower()
                if any(domain in url_lower for domain in _REVIEW_PLATFORM_DOMAINS):
                    platform_blocks_src.append(result)
                else:
                    other_blocks_src.append(result)

            # Every distinct platform contributes its best page BEFORE any
            # platform contributes a second. Straight relevance order let one
            # chatty domain take every block — three healthgrades pages
            # crowding out the vitals and webmd profiles that are the whole
            # point of a second opinion. Only the strongest observation per
            # domain survives the collapse anyway, so extra same-domain pages
            # are the cheapest thing to drop.
            by_domain: Dict[str, List[Dict[str, Any]]] = {}
            for result in platform_blocks_src:
                url_lower = str(result.get("url", "")).lower()
                domain = next((d for d in _REVIEW_PLATFORM_DOMAINS if d in url_lower), "")
                by_domain.setdefault(domain, []).append(result)
            # ...and each domain's "best page" means its PROFILE, not whatever
            # Tavily ranked first. `pop(0)` took relevance order, which is a
            # relevance signal and not a quality one: on 2026-07-28 healthgrades
            # returned `/find-a-doctor/arizona/best-doctors-for-headache-in-
            # chandler` above `/physician/dr-andrea-an-2pfjn`, so the DIRECTORY
            # page took healthgrades' lead slot and the doctor's own profile was
            # deferred to a later round-robin pass.
            #
            # `_page_rank` already existed and was consulted in two downstream
            # tie-breaks — but never in SELECTION, so the profile-over-listing
            # preference only applied to whatever extraction happened to
            # produce. Sorting here is what puts a profile in front of the
            # extractor at all. Stable, so relevance still breaks ties within a
            # page kind.
            for pages in by_domain.values():
                pages.sort(key=lambda r: -_page_rank(r.get("url")))

            ordered_platforms = []
            while by_domain:
                for domain in list(by_domain):
                    ordered_platforms.append(by_domain[domain].pop(0))
                    if not by_domain[domain]:
                        del by_domain[domain]

            blocks = []
            for result in (ordered_platforms + other_blocks_src)[:_MAX_REVIEW_BLOCKS]:
                block = (
                    f"Title: {result.get('title', '')}\n"
                    f"URL: {result.get('url', '')}\n"
                    f"Content: {clip_words(result.get('content', ''), _CONTENT_MAX_CHARS)}"
                )
                raw_content = build_excerpt(
                    # Inlined image payloads first: one percent-encoded SVG (a
                    # Zocdoc logo) was 8,345 of a healthgrades profile's 15,750
                    # characters, and it starts at character 651 — so 46% of the
                    # 1,200-char head reservation meant to capture the rating
                    # header was spent on a logo. `strip_boilerplate` inside
                    # build_excerpt cannot catch it: its nav filters only apply
                    # to lines under 80 chars and this is one line of 8,345.
                    strip_data_uris(result.get("raw_content")),
                    anchors=_anchors_for(result.get("url"), vocab_anchors),
                    budget=_ENRICHMENT_EXCERPT_BUDGET,
                    max_windows=_ENRICHMENT_EXCERPT_WINDOWS,
                    priority_anchors=name_anchors,
                    # A review platform states the overall rating and total in
                    # the first ~1000 characters and nowhere else; the
                    # density-ranked windows below always went to the review
                    # body instead.
                    include_head=_is_review_platform_url(result.get("url")),
                    # ...and it has to be SIZED for that, not left to
                    # `budget // max_windows`. At 2000/3 the reservation was 666
                    # chars and a healthgrades header sits ~1000 in, so it
                    # landed ~400 short — capturing the header on some runs and
                    # not others, which is what moved a provider four ranks
                    # between two runs of the same search. See the sweep at
                    # _ENRICHMENT_HEAD_CHARS.
                    head_chars=_ENRICHMENT_HEAD_CHARS,
                )
                if raw_content:
                    block += f"\nFull page text (excerpt): {raw_content}"
                blocks.append(block)
            results_text = "\n\n".join(blocks)

            descriptors = []
            if specialty:
                descriptors.append(f"a {specialty} provider")
            if provider_location:
                descriptors.append(f"practicing in/near {provider_location}")
            identity = f'"{provider_name}"'
            if descriptors:
                identity += f" ({', '.join(descriptors)})"

            # Carry the candidate pass's findings forward. This pass reads a
            # different set of pages, and its output REPLACES the summary, so
            # anything it isn't shown is dropped from the card.
            carried = str(prior_summary or "").strip()
            prior_block = ""
            if carried and carried != "No reviews available":
                prior_block = (
                    "\nPREVIOUSLY GATHERED PATIENT FEEDBACK for this same provider, "
                    "from earlier pages NOT included below — integrate it, do not "
                    f"discard it:\n{carried}\n"
                )

            prompt = f"""Extract ONLY review information for the healthcare provider {identity} from the search results below.
CRITICAL: Return ONLY a valid JSON object with review data.

IDENTITY CHECK — decide WHO each result is about before extracting anything:
- Match on the PERSON, not on labels: same name + same city/practice = the SAME person even when a portal lists a different specialty label. Portals classify one doctor under adjacent labels (e.g. Neurology vs Sleep Medicine vs Clinical Neurophysiology) — a differing specialty label alone is NEVER grounds for rejection.
- Treat a result as a DIFFERENT person only on clear evidence: a different city/state, an unmistakably unrelated field (e.g. a dentist when looking for a neurologist), or a name that only partially matches. Ignore such results entirely.
- Clinic/practice pages that name this provider: their patient feedback may inform review_summary (present it as feedback for the provider's practice) but NEVER rating or review_count.
- If nothing clearly concerns this person, return "No reviews available", "unknown", null, null.

SOURCE QUALITY RULES:
1. Independent review platforms ({', '.join(_REVIEW_PLATFORM_DOMAINS)}, google) outrank everything else.
2. rating and review_count MUST come from an independent review platform. The provider's OWN practice website and testimonial pages NEVER supply rating or review_count — they are self-published marketing.
3. If several platforms each state a rating + review count, return the pair with the LARGEST stated count and set review_source_url to that platform's URL.
4. review_summary prefers independent-platform review text; use the practice site's testimonials only when no independent review text exists (review_source_url then points at the practice site).

Return a JSON object with these fields ONLY:
- review_summary: 3-4 sentence summary of patient feedback covering most praised aspects, common complaints, and overall experience themes (or "No reviews available" if no actual patient feedback found)
- review_sentiment: Overall sentiment: "positive", "mixed", "negative", or "unknown"
- review_observations: One entry per result block that explicitly STATES a rating and/or a review total for this provider, transcribed exactly: [{{"source_url": "<that block's URL line, copied exactly>", "page_provider_name": "<the provider name as that page writes it>", "rating": 4.0, "review_count": 31}}]. Bare JSON numbers only (4.0, not "4.0/5"). review_count null unless that page states a total ("Based on 31 reviews", "(271)"); rating null unless stated. page_provider_name is copied from the page, NOT from the name you were given — it is checked against it. NEVER derive a rating or count from star-percentage distributions — transcribe stated values only. A block's snippet text counts as its page text. [] if no block states either
- review_count: Your best single candidate from review_observations (the code makes the final pick). null if no platform states a total. NEVER the number of review snippets you happened to read
- rating: Your best single candidate rating as a bare JSON number (e.g. 4.7, not "4.7/5"); null if none stated — never estimate one
- review_source_url: The URL of the result block the rating/count came from (or, if none, the block the summary came from) — copy it exactly from that block's URL line (null if no review data)
- insurance_accepted: List of insurance plans/payers the pages explicitly state this provider accepts — directory profiles (Healthgrades, Vitals, WebMD) often have an "insurance accepted" section; empty array [] if none stated. If sources conflict, prefer the doctor's own profile page over a directory index
- insurance_source_url: The URL of the result block the insurance list came from (null if none)
- years_experience: Years in practice as a bare JSON number, ONLY if a page explicitly states it (e.g. "26 years of experience"); null otherwise — never estimate from graduation dates
- phone: The provider's office phone number if a profile page states one, formatted XXX-XXX-XXXX; null otherwise
- address: The provider's practice street address, taken ONLY from a page that is about THIS PROVIDER — their own profile on a review platform, or a practice page whose URL names them. NEVER from a group or clinic page that lists several doctors: those state one address for all of them, and it is not evidence about this one. Include city, state and ZIP when shown (e.g. "1234 W Frye Rd, Chandler, AZ 85224"); null otherwise. A full address with ZIP sharpens distance ranking
- address_source_url: The URL of the result block the address came from — copy it exactly from that block's URL line (null if no address). Required whenever address is non-null: an address that cannot be traced to a page is discarded

IMPORTANT REVIEW EXTRACTION RULES:
1. ONLY extract if you find ACTUAL PATIENT FEEDBACK (quotes, comments, testimonials) or a platform-stated rating/count
2. DO NOT make up generic statements - they must be based on real patient comments
3. Look for specific feedback like: "great bedside manner", "long wait times", "very thorough", etc.
4. If you find patient feedback, create a comprehensive 3-4 sentence summary covering:
   - What patients praise (expertise, communication, etc.)
   - What patients complain about (wait times, billing, etc.)
   - Overall experience patterns
5. If NO actual patient review content found, use "No reviews available" and "unknown"
6. review_summary must cover BOTH the previously gathered feedback (if any is shown below) and the new results — it REPLACES the earlier summary, so anything you leave out is lost. Where the two disagree, say so ("reviews on one platform note long waits, another does not") rather than dropping either. review_sentiment must reflect the combined evidence
7. A page's own headline figure IS a stated value and must be transcribed: "3.4 out of 5 (23 ratings)", "4.2 / 5 · 108 reviews", "Rated 3.4 by 23 patients" all give rating AND review_count. These usually sit at the very TOP of a profile page, above the patient comments — read the beginning of each page's text, not only the parts that mention the provider by name. This does not loosen the rule above: a star-PERCENTAGE breakdown ("48% 5-star, 39% 1-star") is still never a rating, and you must not average one
{prior_block}
SEARCH RESULTS:
{results_text}

Response (JSON object only):"""

            llm_started = time.perf_counter()
            response = self.anthropic_client.messages.create(
                model=self.config.GATHERER_MODEL,
                # Six result blocks can yield a summary, six observations, an
                # insurance list, tenure, phone and address. 1000 fitted the
                # early two-block pass and was never revisited when the block
                # count grew.
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            in_tokens, out_tokens = safe_usage(response)
            get_cost_tracker().record_llm(
                self.config.GATHERER_MODEL, in_tokens, out_tokens,
                agent="data_gatherer", duration_s=time.perf_counter() - llm_started
            )

            # Same class of defect as the judge's unchecked `finish_reason`
            # (round 9): a truncated response fails to parse, the handler
            # returns nothing, and the provider looks like one the web had no
            # data for. The gatherer was never swept for it.
            if getattr(response, "stop_reason", None) == "max_tokens":
                logger.warning(
                    "Enrichment extraction for %s hit max_tokens — the response is "
                    "truncated and may not parse; observations may be lost",
                    provider_name,
                )

            response_text = response.content[0].text.strip()

            # Strip any code fence, then take the outermost {...} directly.
            # _extract_json_from_response is array-biased: on a bare object
            # containing a nested list (insurance_accepted) it would seize
            # just the list and destroy the object.
            fence = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            json_str = fence.group(1).strip() if fence else response_text
            obj_match = re.search(r'\{[\s\S]*\}', json_str)
            if obj_match:
                json_str = obj_match.group(0)

            review_data = json.loads(json_str)

            if isinstance(review_data, dict):
                insurance_accepted = review_data.get("insurance_accepted")
                observations = review_data.get("review_observations")
                return _apply_parsed_profiles({
                    "review_summary": str(review_data.get("review_summary", "No reviews available")),
                    "review_sentiment": str(review_data.get("review_sentiment", "unknown")),
                    "review_count": review_data.get("review_count"),
                    "rating": review_data.get("rating"),
                    "review_source_url": review_data.get("review_source_url"),
                    "review_observations": observations if isinstance(observations, list) else [],
                    "insurance_accepted": insurance_accepted if isinstance(insurance_accepted, list) else [],
                    "insurance_source_url": review_data.get("insurance_source_url"),
                    "years_experience": review_data.get("years_experience"),
                    "phone": review_data.get("phone"),
                    "address": review_data.get("address"),
                    "address_source_url": review_data.get("address_source_url"),
                }, parsed_profiles, provider_name, self._name_token_overlap, provider_location)

        except Exception as e:
            logger.warning(f"Failed to extract review data for {provider_name}: {e}")

        return _apply_parsed_profiles({
            "review_summary": "No reviews available",
            "review_sentiment": "unknown",
            "review_count": None,
            "rating": None,
            "review_source_url": None,
            "review_observations": [],
            "insurance_accepted": [],
            "insurance_source_url": None,
            "years_experience": None,
            "phone": None,
            "address": None,
            "address_source_url": None,
        }, parsed_profiles, provider_name, self._name_token_overlap, provider_location)

    def _harvest_profile_urls(
        self, provider: Dict[str, Any], results: List[Dict[str, Any]]
    ) -> None:
        """Record this provider's profile URL on each platform the search reached.

        Every candidate is checked with `slug_agrees_with_name` before it is
        stored. The search is name + city and platform-restricted, so it returns
        OTHER doctors' profiles too — a page coming back for a query is not
        evidence that it is about the person queried, which is the same reason
        `_observation_is_same_person` exists.

        Never overwrites a URL discovery already established: that one came off
        a listing entry whose heading carried the name, which is stronger
        evidence than "this ranked for the query".
        """
        known = dict(provider.get("platform_profile_urls") or {})
        seeded = provider.get("profile_url")
        if seeded and source_domain(seeded) and source_domain(seeded) not in known:
            known[source_domain(seeded)] = seeded

        name = provider.get("name", "")
        for result in results or []:
            url = result.get("url")
            domain = source_domain(url)
            if not domain or domain in known:
                continue
            if not any(d in domain for d in _REVIEW_PLATFORM_DOMAINS):
                continue
            if url_page_kind(url) != "profile" or not slug_agrees_with_name(url, name):
                continue
            known[domain] = url
            logger.info("Harvested %s profile URL for %r: %s", domain, name, url)

        if known:
            provider["platform_profile_urls"] = known
            provider.setdefault("profile_url", next(iter(known.values())))

    def _observation_is_same_person(self, obs: Dict[str, Any], provider_name: str,
                                    provider: Optional[Dict[str, Any]] = None) -> bool:
        """Does this observation's page actually name our provider?

        The enrichment query is name + city with no specialty term, so the
        extractor's IDENTITY CHECK was the only thing standing between a
        same-named stranger's rating and this provider's card — a prompt, on
        the cheapest model, with nothing enforcing it. This is the code-side
        backstop.

        Silent when the model omits page_provider_name: a response that does
        not transcribe it degrades to the previous behavior rather than
        discarding every observation it found.

        STEP 4 of URL-primary identity runs FIRST, when it can. An enrichment
        observation's `source_url` IS a profile page, so where we already know
        this provider's URL on that platform the two must be equal — a
        different path on the same platform is a different physician record.
        No new field is needed and, unlike the name check below, it does not
        depend on the cheapest model transcribing anything: the URL is the
        vendor's, not the extractor's. The name check remains for observations
        whose platform URL we do not know, which is most of them until
        enrichment has run.
        """
        known = (provider or {}).get("platform_profile_urls") or {}
        source_url = obs.get("source_url")
        expected = known.get(source_domain(source_url))
        if expected and url_page_kind(source_url) == "profile" and urls_contradict(
            source_url, expected
        ):
            logger.info(
                "Rejected review observation for %r: %s is a different %s profile "
                "than the one we hold (%s)",
                provider_name, source_url, source_domain(source_url) or "platform", expected,
            )
            return False

        # The URL's own SLUG, checked before the name — because the slug is the
        # vendor's and the page name is whatever a model transcribed, and the
        # name check below degrades to ACCEPT when nothing was transcribed.
        #
        # That degrade is what let two neurosurgeons at neighbouring addresses
        # on the same street share numbers on a 2026-07-31 run: both cards read
        # "doctor.webmd.com 5.0/5 (45 reviews)" and "vitals.com (46 reviews)".
        # The enrichment query is name + city, so a colleague at the same
        # practice ranks for it, and one profile's figures were attached to the
        # other. `slug_agrees_with_name` settles it with no model in the loop —
        # every platform writes the name into the profile slug.
        #
        # PROFILE URLs only. A directory index names forty doctors and its slug
        # is a city, so it agrees with nobody and would be rejected wholesale.
        if (url_page_kind(source_url) == "profile"
                and slug_contradicts_name(source_url, provider_name)):
            logger.info(
                "Rejected review observation for %r: the profile slug in %s names "
                "someone else", provider_name, source_url,
            )
            return False

        page_name = str(obs.get("page_provider_name") or "").strip()
        if not page_name or not provider_name:
            return True
        overlap = self._name_token_overlap(page_name, provider_name)
        if overlap < _NAME_MATCH_THRESHOLD:
            logger.info(
                f"Rejected review observation for {provider_name!r}: page names "
                f"{page_name!r} (name overlap {overlap:.2f} < {_NAME_MATCH_THRESHOLD})"
            )
            return False
        return True

    # ---- derivations from review_observations -------------------------------
    #
    # `review_observations` is the single source of truth for platform
    # evidence, and it is the ONLY one of these the cache stores. The blend
    # and the superseding headline are DERIVED from it and are deliberately
    # excluded from `CACHEABLE_FIELDS` (they would otherwise go stale against
    # a changed blend rule). That makes re-derivation mandatory on a cache
    # hit, not optional: without it the same provider scores differently warm
    # than cold — the scorer falls off the blend branch to a raw headline it
    # no longer has, and the critic reads `blended_platform_count` 0, whose
    # own rubric maps that to "low" confidence and a -4.0 refinement penalty.
    # Both paths therefore go through these two helpers.

    def _apply_blend(
        self, provider: Dict[str, Any], observations: List[Dict[str, Any]]
    ) -> None:
        """Set (or clear) the cross-platform blend for the SCORE.

        Display keeps the headline; declines mirror the headline ladder's
        disagreement guard.
        """
        blend = _blended_platform_rating(observations)
        if blend:
            provider["blended_rating"] = blend["rating"]
            provider["blended_review_count"] = blend["review_count"]
            provider["blended_platform_count"] = blend["platforms"]
            # Travels with the number so a contested rating can be CAVEATED
            # rather than suppressed — see `_blended_platform_rating` for why
            # the old disagreement gate was worse than blending.
            provider["blended_rating_spread"] = blend["spread"]
            logger.info(
                f"Blend {blend['rating']}/5 over {blend['review_count']} reviews "
                f"({blend['platforms']} platforms, spread {blend['spread']}) "
                f"for {provider.get('name')}"
            )
        else:
            # Fewer than two platform pairs — a blend computed when the set was
            # larger is stale. This used to fire on DISAGREEMENT too; it no
            # longer does, so reaching here means the pair count genuinely fell.
            for key in ("blended_rating", "blended_review_count",
                        "blended_platform_count", "blended_rating_spread"):
                provider.pop(key, None)

    def _apply_source_quality(
        self, provider: Dict[str, Any], observations: List[Dict[str, Any]]
    ) -> None:
        """Record how many of this provider's platform pairs sit on a profile.

        Called from BOTH observation-derivation paths — `_merge_review_data`
        (cold) and `_rederive_from_observations` (warm) — for the same reason
        the blend is: **warm must reproduce cold exactly**. These counts are
        derived from `review_observations`, which IS cached, so deriving them on
        only one path would make a cache hit report different coverage than the
        run that populated it.

        Deliberately not cached itself (`CACHEABLE_FIELDS` is an allowlist, so
        this is already excluded): a stored count would freeze one run's page
        classification for the whole TTL, exactly the failure the unattributable
        insurance lists hit.
        """
        pairs = _platform_rating_pairs(observations)
        backed = _profile_backed_pairs(observations)
        provider["platform_pair_count"] = len(pairs)
        provider["profile_backed_platforms"] = len(backed)
        if pairs and not backed:
            # The condition round 3 shipped `_has_profile_source` to catch. It
            # is not an error — a directory index still states real numbers —
            # but the card links a patient to a "best <specialty> in <city>"
            # page instead of this doctor's, and nothing else records it.
            logger.info(
                "Source quality: %s has %d platform pair(s), NONE on a profile "
                "page (%s)", provider.get("name"), len(pairs),
                ", ".join(
                    f"{source_domain(p.get('source_url')) or '?'}"
                    f"={url_page_kind(p.get('source_url'))}" for p in pairs
                ),
            )

    def _apply_headline(self, provider: Dict[str, Any], headline: Dict[str, Any]) -> None:
        """Apply the chosen platform pair to `rating` / `review_count`."""
        headline_is_platform = any(
            domain in str(headline.get("source_url", "")).lower()
            for domain in _REVIEW_PLATFORM_DOMAINS
        )
        if (headline_is_platform and headline.get("rating") is not None
                and (_first_int(headline.get("review_count")) or 0) >= _MIN_CREDIBLE_COUNT):
            # A credible platform-stated pair is the most authoritative
            # statement we have — it SUPERSEDES numbers scraped from
            # arbitrary page text during the candidate pass (a stale
            # "(1 review)" must not block healthgrades' "4.0 (15)")
            provider["rating"] = headline["rating"]
            provider["review_count"] = _first_int(headline.get("review_count"))
            logger.info(
                f"Platform pair {provider['rating']}/{provider['review_count']} "
                f"headlines for {provider.get('name')}"
            )
        else:
            if headline.get("review_count") and not provider.get("review_count"):
                provider["review_count"] = _first_int(headline.get("review_count"))
            # Backfill the star rating when the initial extraction found
            # none — directory pages often omit it while review pages
            # state it
            if headline.get("rating") is not None and not provider.get("rating"):
                provider["rating"] = headline["rating"]
                logger.info(f"Backfilled rating {provider['rating']} for {provider.get('name')}")

    def _rederive_from_observations(self, provider: Dict[str, Any]) -> None:
        """Rebuild the blend and headline from a provider's stored observations.

        The cache-hit counterpart of the two calls `_merge_review_data` makes
        on the cold path, so a warm run reproduces a cold run's score.
        """
        stored = provider.get("review_observations") or []
        if not stored:
            return
        headline, observations = _select_review_observation(stored)
        if observations:
            provider["review_observations"] = observations
            self._apply_blend(provider, observations)
            self._apply_source_quality(provider, observations)
        if headline:
            self._apply_headline(provider, headline)

    def _merge_review_data(
        self, provider: Dict[str, Any], review_data: Dict[str, Any],
        user_location: str = "",
    ) -> None:
        """Merge secondary review data into provider object only if actual reviews found.

        Args:
            provider: Provider dictionary to update (modified in place)
            review_data: Review data extracted from secondary search
            user_location: The searching member's location — the practice-
                location selection is member-relative (nearest trusted
                office), so it needs who is asking, not just what was found
        """
        # Only merge if we found actual review content
        if review_data.get("review_summary") and review_data["review_summary"] != "No reviews available":
            provider["review_summary"] = review_data["review_summary"]
            logger.info(f"Enriched reviews for {provider.get('name')}")

        if review_data.get("review_sentiment") and review_data["review_sentiment"] != "unknown":
            provider["review_sentiment"] = review_data["review_sentiment"]

        # Headline rating/count: the model transcribes per-platform
        # observations; CODE picks deterministically (a credible pair beats a
        # single-review outlier; platform-first order breaks ties) — no more
        # run-to-run flip-flops between disagreeing platforms.
        #
        # Observations UNION across passes (first URL wins) instead of the
        # latest extraction replacing the list: the second-opinion enrichment
        # exists to ADD webmd/vitals pairs next to the healthgrades pair the
        # candidate pass found — a name-query result set that happens not to
        # re-include the original page must not erase its numbers.
        merged_obs = list(provider.get("review_observations") or [])
        # A dict, not a set: on a collision the KEPT observation has to be
        # reachable, because "first URL wins" decides which ENTRY survives and
        # must not also decide which READ does — a parsed listing row and the
        # enrichment pass now collide on the doctor's own profile url by
        # construction, and skipping the newcomer outright kept `4.8/None`
        # over a recovered `4.8/260`.
        kept_by_url = {
            u: o for u, o in (
                (str(o.get("source_url") or "").strip().lower(), o)
                for o in merged_obs if isinstance(o, dict)
            ) if u
        }
        identity_rejected = False
        for obs in review_data.get("review_observations") or []:
            if not isinstance(obs, dict):
                continue
            if not self._observation_is_same_person(obs, provider.get("name", ""), provider):
                identity_rejected = True
                continue
            url = str(obs.get("source_url") or "").strip().lower()
            if url and url in kept_by_url:
                _fill_corroborated_gaps(kept_by_url[url], obs)
                continue
            if url:
                kept_by_url[url] = obs
            merged_obs.append(obs)
        headline, observations = _select_review_observation(merged_obs)
        if observations:
            provider["review_observations"] = observations
            self._apply_blend(provider, observations)
            self._apply_source_quality(provider, observations)
        if headline is None and not observations and not identity_rejected:
            # No observations at all: fall back to the model's single pair.
            # (When observations EXIST but selection declined — conflicting
            # rating-only claims — respect the decline; the model's own pick
            # is one of those conflicting numbers.)
            #
            # `identity_rejected` gates this. The top-level rating/count/URL
            # are the model's reading of the SAME pages the per-observation
            # guard just rejected as a different physician, so falling back
            # to them re-admitted a same-surname stranger's numbers and
            # profile link through the back door — and `_classify_enrichment`
            # then reported the provider as cleanly `enriched`.
            fallback_rating = _parse_rating(review_data.get("rating"))
            fallback_count = review_data.get("review_count")
            if fallback_rating is not None or fallback_count:
                headline = {
                    "rating": fallback_rating,
                    "review_count": fallback_count,
                    "source_url": review_data.get("review_source_url"),
                }

        if headline:
            self._apply_headline(provider, headline)

        # Provenance: the page behind the headline numbers wins; else the
        # page the summary came from
        # The model-level `review_source_url` is subject to the same identity
        # gate as the numbers: when every observation was rejected as a
        # different physician, that URL points at the rejected person's
        # profile, and linking to it from this provider's card is the same
        # error as adopting their rating.
        fallback_source = None if identity_rejected else review_data.get("review_source_url")
        source_url = (headline or {}).get("source_url") or fallback_source
        if source_url:
            provider["review_source_url"] = str(source_url)

        # Directory profiles read during enrichment often list accepted
        # insurance — backfill the LIST only when the candidate pass found none
        enriched_insurance = review_data.get("insurance_accepted")
        if enriched_insurance and not provider.get("insurance_accepted"):
            provider["insurance_accepted"] = enriched_insurance
            logger.info(f"Backfilled insurance list for {provider.get('name')}")

        # The SOURCE is backfilled independently of the list. It used to be
        # nested inside the guard above, so a provider whose candidate pass
        # produced a list — the discovery prompt asked for no source URL at
        # all until now — threw away a perfectly good URL that enrichment had
        # just found, and the card showed eight named payers attributed to
        # nothing. Never downgrade a source we already have.
        if provider.get("insurance_accepted") and not provider.get("insurance_source_url"):
            enriched_source = review_data.get("insurance_source_url")
            if enriched_source:
                provider["insurance_source_url"] = str(enriched_source)
                logger.info(f"Backfilled insurance source for {provider.get('name')}")

        # Platform profiles usually state tenure — backfill the experience
        # subscore's input when the candidate pass missed it. Unscraped years
        # score the unknown imputation (60) vs up to 100 for stated years, so this
        # field decides real ranking points; never clobber an existing value.
        enriched_years = review_data.get("years_experience")
        if enriched_years is not None and provider.get("years_experience") is None:
            years = _first_int(enriched_years)
            if years is not None and 0 <= years <= 80:
                provider["years_experience"] = years
                # The parser labelled its own write; a surviving unlabelled
                # value is the model's, which reads tenure off any of six
                # blocks and carries no URL — precisely why the label matters.
                provider["experience_source"] = (
                    review_data.get("experience_source") or "enrichment_model"
                )
                logger.info(f"Backfilled {years} years experience for {provider.get('name')}")

        # Platform profiles show the office phone prominently, but most
        # candidates arrive from directory/listicle pages where it never
        # lands in the excerpt — the same capture gap years had. Backfill
        # when the candidate pass found none; never clobber.
        enriched_phone = str(review_data.get("phone") or "").strip()
        if (enriched_phone and enriched_phone.upper() != "N/A"
                and not provider.get("phone")):
            provider["phone"] = enriched_phone
            logger.info(f"Backfilled phone for {provider.get('name')}")

        # Platform profiles state a full practice address; back it in ONLY
        # when it gains ZIP precision we don't already have. City-listing
        # discovery often yields just "City, ST" (coarse same-city tier) or
        # nothing — a ZIP-resolvable address upgrades the provider to real
        # distance scoring. Never clobber a location we can already place to
        # a ZIP. The caller recomputes distance/tier after this returns.
        #
        # The state check that guards the PARSER's address applies here too,
        # and the asymmetry it closes was the bug: `_apply_parsed_profiles`
        # rejects a page in the wrong state, while this path — the one the
        # MODEL feeds — accepted whatever it read off any of six blocks with no
        # identity check at all. Those blocks include practice and group pages
        # that name several doctors, so one address could land on all of them.
        # Two providers on the 2026-08-01 run carried an identical street
        # address AND an identical distance.
        #
        # State-level, positive disagreement only, for the reasons at
        # `_states_disagree`. It cannot separate two cities in one metro, which
        # is exactly the case that prompted it — so `location_source` records
        # where the address came from, because the question "where did this
        # address come from" had no answer on any surface.
        enriched_address = str(review_data.get("address") or "").strip()
        address_url = review_data.get("address_source_url")
        if (enriched_address
                and resolution_level(enriched_address) == "zip"
                and resolution_level(provider.get("location")) != "zip"):
            # An address the model cannot attribute to a page ABOUT this doctor
            # is discarded. The prompt asks it to read only such pages, but the
            # prompt is not the guard — a code predicate is (the same reasoning
            # that put `_observation_is_same_person` behind the extractor's own
            # identity check). Without this, six blocks are in scope including
            # group practice pages that state one address for several doctors,
            # and two providers came back carrying the same street and the same
            # distance.
            if not _url_identifies_provider(address_url, provider.get("name", "")):
                logger.info(
                    "Refused address backfill for %r: %r is not traceable to a "
                    "page about them (source %r)",
                    provider.get("name"), enriched_address, address_url,
                )
            elif _states_disagree(enriched_address, provider.get("location")):
                logger.info(
                    "Refused address backfill for %r: %r is in a different state "
                    "than %r", provider.get("name"), enriched_address,
                    provider.get("location"),
                )
            else:
                previous = provider.get("location")
                provider["location"] = enriched_address
                provider["location_source"] = (
                    review_data.get("address_source")
                    or f"enrichment_model:{address_url}"
                )
                logger.info(
                    "Backfilled address for %s: %r -> %r (via %s)",
                    provider.get("name"), previous, enriched_address,
                    provider["location_source"],
                )

        self._record_practice_locations(provider, review_data)
        # Selection AFTER the record: the record is the complete inventory of
        # believable addresses, and the caller re-runs
        # `_attach_location_evidence` right after this merge returns, so a
        # relocation here re-scores distance with no extra wiring.
        self._select_nearest_trusted_location(provider, user_location)

    def _record_practice_locations(
        self, provider: Dict[str, Any], review_data: Dict[str, Any]
    ) -> None:
        """Record every believable address as the provider's LOCATION SET.

        The predecessor of this method FLAGGED addresses more than 20 miles
        apart as a "conflict" to adjudicate. Field evidence retired that
        framing: the doctors who trip it are multi-office group specialists
        (the CORE Institute lists offices across the metro on every doctor's
        own profile), so several far-apart addresses are the NORMAL state of
        a confirmed practice, not a dispute. The record is now unconditional
        on distance — every distinct believable address, with its source —
        and `_select_nearest_trusted_location` chooses which one this member
        sees. The distance threshold went with the framing: a member deserves
        the "closest of N locations" note whether the offices are 3 miles
        apart or 30, and a gate at 20 made a doctor with offices 19 miles
        apart the indefensible edge case.

        Candidates are the addresses we would actually believe: the provider's
        current location (whatever producer wrote it) plus every parsed-profile
        address that survived the state and H1 gates. The record keeps the key
        `address_conflict` for cache-schema and panel stability — renaming a
        CACHEABLE_FIELDS entry would orphan every stored row for the TTL.
        """
        candidates: List[Dict[str, str]] = []
        current = str(provider.get("location") or "").strip()
        if current:
            candidates.append({
                "address": current,
                "source": str(provider.get("location_source") or "discovery"),
            })
        for cand in review_data.get("address_candidates") or []:
            if isinstance(cand, dict) and str(cand.get("address") or "").strip():
                entry = {"address": str(cand["address"]).strip(),
                         "source": str(cand.get("source") or "profile_parser")}
                if cand.get("phone"):
                    entry["phone"] = str(cand["phone"]).strip()
                candidates.append(entry)

        # Dedupe by (address, SOURCE) — not by address alone. The same address
        # claimed by two distinct sources is CORROBORATION — it is exactly
        # what the trust check's exact-twin rule counts — and collapsing to
        # the first source would silently discard the second claimant.
        # Identical (address, source) pairs still collapse: one page restating
        # one office is one voice.
        unique: Dict[Tuple[str, str], Dict[str, str]] = {}
        for cand in candidates:
            unique.setdefault(
                (" ".join(cand["address"].split()).lower(), cand["source"]), cand
            )
        distinct = list(unique.values())
        addresses = {" ".join(c["address"].split()).lower() for c in distinct}
        if len(addresses) < 2:
            # One address (however many sources confirm it) is not a location
            # SET — nothing to select, nothing to tell the member.
            provider.pop("address_conflict", None)
            return

        provider["address_conflict"] = {
            "addresses": distinct,
            "distinct_count": len(addresses),
        }
        logger.info(
            "Practice locations for %s: %d distinct addresses from %d claims",
            provider.get("name"), len(addresses), len(distinct),
        )

    @staticmethod
    def _address_source_class(source: str) -> int:
        """Rank an address source's class for the WITHIN-cluster pick only.

        profile_parser (the doctor's own gate-checked page) beats
        listing_parser (one row on a many-doctor page — the class that gave
        two doctors one Mesa suite) beats everything else (the model's
        address carries no URL; "discovery" carries no page at all).
        """
        prefix = str(source or "").split(":", 1)[0]
        return {"profile_parser": 0, "listing_parser": 1}.get(prefix, 2)

    def _select_nearest_trusted_location(
        self, provider: Dict[str, Any], user_location: str
    ) -> None:
        """Show this member the nearest office we can TRUST the doctor is at.

        Supersedes the cluster-corroboration vote (and its banned
        nearest-to-searcher tie-break). The vote answered "which single
        address is true?", but the doctors who trip this are confirmed
        multi-office specialists — SEVERAL addresses are true — and picking
        any fixed one made the answer depend on which pages that run's fetch
        happened to foreground: Dr. Vandian carried the pool's strongest
        review record and oscillated between rank 5 at his 6.4-mile Gilbert
        office and rank 7 at a 40-mile office across two same-criteria runs.
        Finding a good specialist is hard; losing one to our own address
        pick — or to the radius bound acting on a far office — is the worse
        failure by every measure. Worst case under THIS rule is one phone
        call the card already tells the member to make.

        TRUST, then MIN — the order is load-bearing. A bare min over every
        candidate would hand the card to the single worst datapoint whenever
        it happens to be near the member (a same-named doctor's office that
        slipped the gates, a listing row's group office he never sits at),
        and gathering MORE pages would make results WORSE — each new page
        another lottery ticket for a bogus-but-near address. An address is
        trusted when the doctor's OWN profile page lists it, or when two
        distinct sources state exactly the same address. Exact, not
        same-area: the retired 20-mile "area corroboration" let two
        independently wrong addresses in one metro vouch for each other,
        and profile-listed addresses never needed a twin. The known trade —
        formatting variants ("Ste"/"Suite") fail the exact match — errs
        toward the fallback, which keeps the address we already have.

        Member-relative by design (the retired ban's reason to exist — "a
        fact about the doctor depends on who is asking" — is the point: the
        nearest of his REAL offices is precisely the member-relevant fact),
        so this runs on the warm path too and is never baked into a cached
        row. Nothing here deletes an address: the full set stays in the
        record, and the card's closest-of-N note survives selection because
        a selection is not a certification.
        """
        conflict = provider.get("address_conflict")
        if not conflict or not user_location:
            return
        entries = [
            e for e in conflict.get("addresses") or []
            if isinstance(e, dict) and str(e.get("address") or "").strip()
        ]
        if len(entries) < 2:
            return

        def norm(addr: str) -> str:
            return " ".join(str(addr).split()).lower()

        # Group claims per distinct address; the representative claim (for
        # location_source and the phone that travels with its office) is the
        # best source class, then first-seen.
        by_address: Dict[str, List[Dict[str, str]]] = {}
        order: List[str] = []
        for entry in entries:
            key = norm(entry["address"])
            if key not in by_address:
                order.append(key)
            by_address.setdefault(key, []).append(entry)

        def is_trusted(claims: List[Dict[str, str]]) -> bool:
            if any(
                self._address_source_class(c.get("source", "")) == 0
                for c in claims
            ):
                return True  # the doctor's own profile lists this office
            return len({str(c.get("source") or "") for c in claims}) >= 2

        selection: Dict[str, Any] = {"method": "nearest_trusted"}
        trusted_keys = [k for k in order if is_trusted(by_address[k])]
        selection["trusted_addresses"] = len(trusted_keys)
        if not trusted_keys:
            # Only uncorroborated singletons — the old satellite-vs-stale
            # dead end. Keep the address we already have; the closest-of-N
            # note still reaches the member because the record stands.
            selection.update({"resolved": False, "reason": "no_trusted_address"})
            conflict["selection"] = selection
            return

        scored: List[Tuple[float, int, int, str]] = []
        for index, key in enumerate(trusted_keys):
            claims = by_address[key]
            representative = min(
                range(len(claims)),
                key=lambda i: (
                    self._address_source_class(claims[i].get("source", "")), i
                ),
            )
            distance = distance_miles(
                user_location, claims[representative]["address"]
            )
            if isinstance(distance, (int, float)):
                scored.append((float(distance), index, representative, key))
        if not scored:
            # Trusted offices exist but none geocodes against the member —
            # our coverage, not their locations. Keep what we have.
            selection.update({"resolved": False, "reason": "no_resolvable_distance"})
            conflict["selection"] = selection
            return

        distance, _, representative, key = min(scored)
        chosen_entry = by_address[key][representative]
        selection.update({
            "resolved": True,
            "chosen": chosen_entry["address"],
            "chosen_source": chosen_entry.get("source"),
            "distance_miles": round(distance, 1),
        })
        conflict["selection"] = selection

        if key == norm(str(provider.get("location") or "")):
            return  # already showing the nearest trusted office

        previous = provider.get("location")
        provider["location"] = chosen_entry["address"]
        provider["location_source"] = chosen_entry.get("source") or "profile_parser"
        # The phone travels with its own office block or not at all — never
        # leave the OLD address's number beside the new address.
        if chosen_entry.get("phone"):
            provider["phone"] = chosen_entry["phone"]
        logger.info(
            "Nearest trusted office for %s: %r (%s, %.1f mi) replaces %r — "
            "distance will be recomputed",
            provider.get("name"), chosen_entry["address"],
            chosen_entry.get("source"), distance, previous,
        )

    def _attach_location_evidence(self, provider: Dict[str, Any], user_location: str) -> None:
        """Compute a provider's distance + tier from the user's location, in
        code, and null the city-centroid artifact.

        When we've placed the provider only to city precision AND it's the
        user's own city, a computed distance is a centroid coincidence (often
        ~0 mi — a fake bullseye that would out-score a provider with a real
        ZIP a few miles out, so more data would paradoxically score worse).
        Fall those to the honest same_city tier. A provider resolved to a real
        ZIP keeps its measured distance; a different-city centroid distance
        keeps the falloff (centroid error is small at inter-city scale).
        """
        provider_location = provider.get("location")
        distance = distance_miles(user_location, provider_location)
        tier = location_tier(user_location, provider_location)
        if (distance is not None
                and tier == "same_city"
                and resolution_level(provider_location) == "city"):
            distance = None
        provider["computed_distance_miles"] = distance
        provider["location_match"] = tier
        # "zip" = a real ZIP coordinate; "city" = a city centroid, which is an
        # ESTIMATE shared by every provider in that city. The 2026-07-25 run had
        # nearly the whole pool at an identical "8.0 mi (computed straight-line)"
        # — one centroid, presented as ten measurements. The scorer and the
        # critic both need to know which it is.
        provider["distance_precision"] = (
            resolution_level(provider_location) if distance is not None else None
        )

    def enrich_providers(
        self,
        providers: List[Dict[str, Any]],
        location: str,
        specialty: str = "",
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Public enrichment pass over the caller's ranked list (mutates in place).

        The orchestrator calls this AFTER deterministic core scoring with the
        FULL ranking, so when MAX_PROVIDERS_TO_ENRICH does bind it truncates
        the tail of the ranking rather than the tail of extraction order.
        Enrichment itself is uniform: every provider the cache didn't serve
        gets a live pass, and anyone past the cap is marked `over_budget`.

        Cached enrichment is applied FIRST, so a provider restored from the
        store already carries its platform pairs and is excluded from the live
        pass by its `enrichment_outcome` — the cache shrinks the work rather
        than running alongside it.
        """
        query_location = strip_zip(location) or city_state_for_zip(location) or (location or "")
        # query_location (ZIP stripped) keeps web queries clean; the original
        # `location` (with any ZIP) is kept for distance recompute when an
        # address backfill lands, so the user's ZIP precision is preserved.

        if use_cache:
            # Pin each key BEFORE enrichment, which can rewrite `location`
            # (a profile's street address gaining ZIP precision) and would
            # otherwise store the row under a key the next read never asks
            # for. See `utils.provider_key.resolve_cache_key`.
            #
            # `cache_basis` is the key's two inputs, human-readable, rendered
            # per provider in the coverage panel. A same-criteria repeat
            # search missed 7 of 8 and nothing recorded WHICH half moved
            # between the runs — the surviving name variant (middle initials
            # are kept by design, and the dedupe survivor is picked by field
            # richness) or the discovery city (a multi-site group's listing
            # shows a different office per city page). Diffing two runs'
            # bases names the moving component per provider.
            for provider in providers:
                pin_cache_key(provider)
                provider["cache_basis"] = "{}|{}".format(
                    normalized_name(provider.get("name")),
                    normalized_place(provider.get("location")),
                )

            self._apply_cached_enrichment(providers, user_location=location or "")

        self._enrich_missing_reviews(
            providers, query_location, specialty, user_location=location or ""
        )

        if use_cache:
            self._store_enrichment(providers)
            for provider in providers:
                provider.pop(CACHE_KEY_FIELD, None)

        return providers

    def _apply_cached_enrichment(
        self, providers: List[Dict[str, Any]], user_location: str = ""
    ) -> int:
        """Hydrate providers from the enrichment cache. Returns the hit count.

        Never fails a search: any cache error degrades to a cold run.
        """
        try:
            from utils.vector_store import get_vector_store

            store = get_vector_store()
            fresh, stale = store.get_cached_providers(providers)
            if stale:
                logger.info(f"Cache: {len(stale)} stale/incompatible entries ignored")
            if not fresh:
                get_cost_tracker().record_cache(hits=0, misses=len(providers))
                return 0

            hits = 0
            for provider in providers:
                key = resolve_cache_key(provider)
                payload = fresh.get(key)
                if not payload:
                    continue

                # Observations UNION; everything else is replaced.
                #
                # A blanket `update()` discarded THIS run's discovery
                # observations in favour of the stored list — the same
                # fill-if-empty mistake round 4 fixed on the cold path, where
                # keeping only one side halved platform coverage and starved
                # the blend, which needs two platforms. A cache hit is supposed
                # to save a SEARCH, not delete evidence the search already
                # produced; today's discovery pass can legitimately hold a
                # platform the stored row predates.
                live_observations = list(provider.get("review_observations") or [])
                provider.update(payload)
                if live_observations:
                    # Same union, same rule as `_merge_review_data`: the stored
                    # entry survives a collision, but a fresher discovery read
                    # of the SAME url may complete it when they agree. A stored
                    # rating-only row otherwise suppresses a listing pair for
                    # the whole TTL — the cache freezing exactly the gap the
                    # live pass just closed.
                    kept_by_url = {
                        u: o for u, o in (
                            (str(o.get("source_url") or "").strip().lower(), o)
                            for o in provider.get("review_observations") or []
                            if isinstance(o, dict)
                        ) if u
                    }
                    # `new_observations`, NEVER `fresh`: this scratch list was
                    # first named `fresh`, shadowing the HITS DICT the loop is
                    # iterating against. The first cache hit that carried live
                    # discovery observations rebound the name, the next
                    # iteration's `fresh.get(key)` raised AttributeError on a
                    # list, and the outer except — built for store outages —
                    # reported "Cache read failed, continuing cold". Every
                    # provider after that point was silently re-enriched and
                    # re-billed on every search for three days, at most ONE
                    # union-branch hit ever surviving per run, while the store
                    # underneath passed every durability test. The disguise
                    # was total: the crash wore the outage message, and the
                    # rewritten rows made the store look like it was losing
                    # data. A safety net that catches a code bug converts it
                    # into weather.
                    new_observations = []
                    for o in live_observations:
                        if not isinstance(o, dict):
                            continue
                        url = str(o.get("source_url") or "").strip().lower()
                        if url and url in kept_by_url:
                            _fill_corroborated_gaps(kept_by_url[url], o)
                            continue
                        new_observations.append(o)
                    provider["review_observations"] = (
                        list(provider.get("review_observations") or []) + new_observations
                    )
                provider["enrichment_outcome"] = "cached"
                hits += 1

                # The blend and the superseding headline are DERIVED from
                # `review_observations` and are not cached, so re-derive them
                # here. Without this a cache hit restores the observations but
                # scores as though the provider had none.
                self._rederive_from_observations(provider)

                # Distance is deliberately NOT cached — it depends on the
                # CURRENT user's location. A stored Chandler distance restored
                # into a Phoenix search would be wrong and would read as
                # measured. The cached `location` may be a sharper address than
                # discovery found, so recompute from it every time. The
                # nearest-trusted-office selection re-runs first for the same
                # reason: it is member-relative (a Glendale searcher and a
                # Chandler searcher get different offices of the same group
                # doctor), so the stored row carries the location SET and the
                # choice is made fresh per search, never served from storage.
                if user_location:
                    self._select_nearest_trusted_location(provider, user_location)
                    self._attach_location_evidence(provider, user_location)

            logger.info(f"Cache: {hits} hit(s), {len(providers) - hits} miss(es)")
            get_cost_tracker().record_cache(hits=hits, misses=len(providers) - hits)
            return hits

        except Exception as e:
            logger.warning(f"Cache read failed, continuing cold: {e}")
            return 0

    def _store_enrichment(self, providers: List[Dict[str, Any]]) -> int:
        """Persist freshly enriched providers. Never fails a search."""
        try:
            from utils.vector_store import get_vector_store

            # ONLY `enriched` is written. The filter used to be `!= "cached"`
            # (re-storing a hit unchanged would refresh its timestamp and let
            # one entry live forever unverified — still true, still excluded),
            # which let every FAILURE outcome through: the store's
            # substantive-payload guard was designed for the era when a failed
            # enrichment produced an empty payload, but discovery now supplies
            # listing-parsed observations, so a `no_profile_found` provider
            # carries "substance" and was cached anyway. On the next run the
            # hit relabelled him `cached`, which passes the recommendation
            # gate his failure had correctly failed — a withheld provider
            # laundered into recommendable on identical evidence — and his
            # profile search was never retried for the whole TTL. Failures
            # now stay uncached, so every run retries them until one
            # succeeds; `over_budget` is excluded with them (nothing was
            # learned beyond discovery, and caching discovery-grade rows
            # under enrichment pretenses is the same defect at pool scale).
            newly = [p for p in providers if p.get("enrichment_outcome") == "enriched"]
            return get_vector_store().upsert_enriched_providers(newly)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
            return 0

    def _enrich_missing_reviews(self, providers: List[Dict[str, Any]], location: str, specialty: str = "", user_location: str = "") -> List[Dict[str, Any]]:
        """Run the per-provider review pass for everyone the cache didn't serve.

        No tiering. The previous version rationed the budget across a "second
        opinion" tier and a "misranked gem" tier, but at the observed pool size
        it never actually rationed anything: MAX_PROVIDERS_TO_ENRICH was 10 at
        the time and the 2026-07-25 field run produced a pool of exactly 10, so every
        provider was already being enriched. Rank 6 arrived with no blended
        rating not because he was skipped but because his search found nothing
        usable — selection was never the problem, SUCCESS was. The tiers cost
        real complexity (three nested predicates deciding who deserved
        evidence) to solve a problem that did not exist, and their effect was
        invisible either way.

        What replaces them is an explicit outcome per provider, so "we found
        nothing for this doctor" stops being indistinguishable from "we never
        looked". Without that the scorer compares three platform pairs against
        zero as though the gap were quality rather than data availability.

        Args:
            providers: List of provider dictionaries, in ranked order
            location: Location for search context
            specialty: Target specialty, used in the extraction prompt's
                identity check so a same-named provider in another field can't
                contaminate

        Returns:
            Updated list of providers with enriched review data
        """
        # A cache hit already holds what this pass would produce. Under the old
        # tiers this exclusion was incidental — a cached provider happened to
        # carry two platform pairs, so _needs_second_opinion said no — which
        # meant it would silently break the moment the tiers went away.
        candidates = [p for p in providers if p.get("enrichment_outcome") != "cached"]

        if not candidates:
            logger.info("Enrichment: every provider served from cache")
            return providers

        # A BACKSTOP for direct callers. The orchestrator now applies the
        # research budget before calling this — pinned there because the same
        # cut must bind the judge and critic, and because enrichment moves the
        # core scores a re-derived cut would use. If it ever binds here, say so:
        # a silent truncation reads as "everyone was covered" when they were not.
        cap = self.config.MAX_PROVIDERS_TO_ENRICH
        if len(candidates) > cap:
            logger.warning(
                f"Enrichment cap reached: {len(candidates)} candidates truncated to "
                f"{cap}; {len(candidates) - cap} left unenriched"
            )
            for over in candidates[cap:]:
                over["enrichment_outcome"] = "over_budget"
            candidates = candidates[:cap]

        cached = len(providers) - len([p for p in providers if p.get("enrichment_outcome") != "cached"])
        logger.info(
            f"Enriching {len(candidates)} provider(s) live; {cached} served from cache"
        )

        # Each candidate needs a Tavily search plus a Haiku extraction; the
        # candidates are independent, so run them concurrently. Workers only
        # mutate their own provider dict.
        #
        # The width was a hardcoded 4, so a full budget of 10 ran in THREE
        # sequential waves of ~18s (search + extraction) — the third half empty
        # and still costing a full wave. That is where ~54s of the 2026-07-28
        # run's 145s went, all of it billed to "Preference Scoring" because
        # enrichment runs inside that node. See config.ENRICHMENT_MAX_WORKERS
        # for why the default is 8 rather than the budget itself.
        workers = max(1, min(len(candidates), self.config.ENRICHMENT_MAX_WORKERS))
        logger.info(
            "Enrichment concurrency: %d worker(s) over %d candidate(s) — %d wave(s)",
            workers, len(candidates), -(-len(candidates) // workers),
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(
                lambda p: self._enrich_one(p, location, specialty, user_location), candidates
            ))

        return providers

    def _enrich_one(self, provider: Dict[str, Any], location: str, specialty: str = "", user_location: str = "") -> None:
        """Run the review-enrichment search + extraction for a single provider.

        user_location is the ORIGINAL search location (with any ZIP), used to
        recompute distance if an address backfill sharpens where the provider
        is — kept separate from the ZIP-stripped query `location`.
        """
        try:
            provider_name = provider.get("name", "")
            if not provider_name:
                provider["enrichment_outcome"] = "failed"
                return

            # Query with the provider's OWN city when known — a Gilbert
            # doctor surfaced by a Chandler search has Gilbert profile pages
            provider_parts = parse_location(provider.get("location"))
            if provider_parts.get("city") and provider_parts.get("state"):
                query_location = f"{provider_parts['city']}, {provider_parts['state']}"
            else:
                query_location = location

            # Name + city only. The specialty is deliberately NOT a retrieval
            # term: portals file one doctor under adjacent labels, so asserting
            # "Neurology" made healthgrades' Neurology DIRECTORY pages (other
            # doctors) outrank the target's own Sleep-Medicine-labeled profile,
            # which then fell under the relevance floor and vanished. It also
            # pulled toward listing pages generally, fighting the
            # profile-over-listing preference. Identity is still enforced —
            # the specialty rides in the extraction prompt's IDENTITY CHECK,
            # where "adjacent label" and "unrelated field" can be told apart,
            # and _merge_review_data name-checks every observation in code.
            query = f"{provider_name} reviews {query_location}"
            logger.debug(f"Review enrichment search: {query}")

            # ONE platform-restricted advanced search. rating/review_count
            # may only come from the independent platforms anyway, so the
            # credits buy exactly those result slots — an open name query let
            # SEO aggregators and practice sites crowd the extraction input,
            # then needed a conditional rescue search when platforms didn't
            # rank (two searches for less signal). Slots are 2x the platform
            # count: at 5 the five domains contested five slots and a doctor
            # present on three of them still came back single-sourced.
            results = self._search_providers(
                query, max_results=2 * len(_REVIEW_PLATFORM_DOMAINS),
                include_raw_content=True,
                include_domains=list(_REVIEW_PLATFORM_DOMAINS),
                search_depth="advanced",
            )

            # STEP 3 of URL-primary identity: the profile URLs this provider
            # has on every platform, taken from the search we ALREADY ran.
            #
            # Discovery only learns the URL of whichever platform's listing
            # surfaced the provider — often one, and on vitals often none at
            # all. The obvious fix is a per-domain search per provider; it
            # would buy URLs already paid for. This search is platform-
            # restricted and returns the profiles: a real run for one provider
            # came back with the webmd, vitals AND healthgrades profile links
            # alongside the directory pages. Harvesting is 0 credits where
            # searching is ~10-36c a run.
            self._harvest_profile_urls(provider, results)

            # What this search actually reached, before extraction gets a say.
            # Three failures are indistinguishable on a finished card — the
            # platform's profile was never returned; it was returned but no
            # observation came out of it; it produced an observation that then
            # lost the same-domain collapse — and they need different fixes.
            # Recorded per provider so one run answers which.
            #
            # Live pass only: a cache hit runs no search, and inventing an
            # empty list for it would read as "we looked and found nothing".
            # `enrichment_outcome` == "cached" is what explains its absence.
            #
            # `raw_chars` separates a fourth failure the other three hide:
            # `build_excerpt` takes min(budget, available), so "Tavily returned
            # a thin page" and "we under-read a full one" produce the same
            # empty result. One run distinguishes them.
            #
            # `yielded` is filled in AFTER extraction (see below). Without it
            # this list claimed to separate four failure modes while recording
            # only what was FETCHED, so on 2026-07-29 a healthgrades profile at
            # 44,138 chars sat beside a listing-page headline with no way to
            # tell "produced nothing" from "produced a rating with no count"
            # from "lost the collapse". Those have three different fixes and
            # the run cost 60c to not answer the question.
            provider["enrichment_sources"] = [
                {
                    "url": r.get("url", ""),
                    "kind": url_page_kind(r.get("url", "")),
                    "raw_chars": len(str(r.get("raw_content") or "")),
                }
                for r in results if r.get("url")
            ]

            if not results:
                # Nothing on any of the five review platforms. Distinct from
                # "we never looked" — that distinction is the whole point of
                # recording an outcome.
                provider["enrichment_outcome"] = "no_profile_found"
                return

            review_data = self._extract_review_data_only(
                results, provider_name, specialty, provider.get("location", ""),
                prior_summary=provider.get("review_summary", ""),
            )
            # Annotated from THIS pass's extraction, before the merge folds in
            # discovery's observations — the question is what these pages
            # produced, not what the provider ended up holding.
            provider["enrichment_sources"] = _annotate_source_yields(
                provider.get("enrichment_sources"),
                review_data.get("review_observations"),
            )
            location_before = provider.get("location")
            self._merge_review_data(
                provider, review_data, user_location=user_location or location
            )
            # An address backfill can sharpen where the provider is —
            # recompute distance/tier from the original user location so
            # the re-score (score_providers) sees the improved evidence.
            if provider.get("location") != location_before:
                self._attach_location_evidence(provider, user_location or location)

            provider["enrichment_outcome"] = self._classify_enrichment(
                provider_name, review_data, provider
            )

        except Exception as e:
            provider["enrichment_outcome"] = "failed"
            logger.warning(f"Could not enrich reviews for {provider.get('name')}: {e}")

    def _classify_enrichment(
        self, provider_name: str, review_data: Dict[str, Any],
        provider: Optional[Dict[str, Any]] = None
    ) -> str:
        """What this provider's enrichment pass actually achieved.

        Read-only: re-runs the same identity predicate `_merge_review_data`
        applies rather than changing that function, which carries round-4
        guarantees (observation UNION across passes, prior_summary integration)
        and its own guards.

        `identity_rejected` is worth separating from `no_profile_found`: pages
        were found and thrown away because the name on them didn't match, which
        points at the identity guard or the query, whereas an empty result set
        points at coverage.
        """
        extracted = [
            o for o in (review_data.get("review_observations") or [])
            if isinstance(o, dict)
        ]
        accepted = [
            o for o in extracted
            if self._observation_is_same_person(o, provider_name, provider)
        ]

        summary = review_data.get("review_summary") or ""
        gained_summary = bool(summary) and summary != "No reviews available"

        if accepted or gained_summary:
            return "enriched"
        if extracted:
            return "identity_rejected"
        return "no_profile_found"

    # ------------------------------------------------------------------
    # FHIR integration methods
    # ------------------------------------------------------------------

    def _gather_fhir_providers(
        self, specialty: str, location: str, insurance: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query FHIR Provider Directory and return transformed provider dicts.

        Args:
            specialty: Medical specialty name
            location: City, State string
            insurance: Insurance network name (optional)

        Returns:
            List of provider dicts from FHIR data
        """
        if not self.fhir_client or not self._fhir_transformer:
            return []

        try:
            bundle = self.fhir_client.search_practitioners(
                specialty=specialty,
                location=location,
                insurance_network=insurance,
            )
            providers = self._fhir_transformer.transform_bundle(bundle)
            logger.info("FHIR returned %d providers for %s in %s", len(providers), specialty, location)
            return providers
        except Exception as e:
            logger.warning("FHIR provider gathering failed: %s", e)
            return []

    @staticmethod
    def _name_token_overlap(name_a: str, name_b: str) -> float:
        """Calculate token-overlap ratio between two provider names.

        Tokenizing is delegated to `utils.provider_key.normalize_name_tokens`,
        the SAME normalization the enrichment cache keys on — dedup and the
        cache are the same identity question asked at two moments, and the
        previous local tokenizer answered it differently. It split only on
        whitespace, so "Seif-Eddeine" was one token against "Seif Eddeine"'s
        two, overlap came out 0.0, and one physician held ranks 1 and 3 of the
        same result set.

        Args:
            name_a: First provider name
            name_b: Second provider name

        Returns:
            Overlap ratio between 0.0 and 1.0
        """
        tokens_a = normalize_name_tokens(name_a)
        tokens_b = normalize_name_tokens(name_b)

        if not tokens_a or not tokens_b:
            return 0.0

        # Dividing by the SMALLER set makes every subset a perfect 1.0, so a
        # bare surname absorbed any full name sharing it: {kim} against
        # {jane, kim} scored 1.0 and merged two different physicians. That is
        # reachable in normal operation — extraction rule 3 asks for thin
        # entries by design, and directory pages routinely print "Dr. Kim"
        # with no given name. Worse, `_field_richness` picks the survivor, so
        # the nameless row could win and the full name be discarded.
        #
        # A single-token name carries no evidence that it is the SAME person,
        # only that it is not obviously a different one, so it may merge only
        # with another single-token name. Two full names are unaffected:
        # {david, kim} vs {jane, kim} was 0.5 before and is 0.5 now.
        if (len(tokens_a) == 1) != (len(tokens_b) == 1):
            return 0.0

        intersection = tokens_a & tokens_b
        smaller = min(len(tokens_a), len(tokens_b))
        return len(intersection) / smaller if smaller else 0.0

    def _merge_fhir_and_tavily_providers(
        self,
        fhir_providers: List[Dict[str, Any]],
        tavily_providers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge FHIR and Tavily provider lists by name similarity.

        - Matched: FHIR identity/insurance + Tavily ratings/reviews
        - FHIR-only: kept with data_source="fhir"
        - Tavily-only: kept with data_source="tavily"

        Args:
            fhir_providers: Providers from FHIR directory
            tavily_providers: Providers from Tavily web search

        Returns:
            Merged list of unique providers
        """
        merged: List[Dict[str, Any]] = []
        matched_tavily_indices: set[int] = set()

        for fhir_p in fhir_providers:
            best_match_idx: Optional[int] = None
            best_score = 0.0

            for i, tavily_p in enumerate(tavily_providers):
                if i in matched_tavily_indices:
                    continue
                score = self._name_token_overlap(
                    fhir_p.get("name", ""), tavily_p.get("name", "")
                )
                if score > best_score:
                    best_score = score
                    best_match_idx = i

            if best_score >= _NAME_MATCH_THRESHOLD and best_match_idx is not None:
                # Merge: FHIR identity + Tavily ratings/reviews
                matched_tavily_indices.add(best_match_idx)
                tavily_p = tavily_providers[best_match_idx]

                combined = fhir_p.copy()
                # Overlay Tavily review data (FHIR has none)
                combined["rating"] = tavily_p.get("rating", 0.0)
                combined["review_count"] = tavily_p.get("review_count")
                combined["review_summary"] = tavily_p.get("review_summary", "No reviews available")
                combined["review_sentiment"] = tavily_p.get("review_sentiment", "unknown")
                # Tavily may have distance info
                if tavily_p.get("distance") is not None:
                    combined["distance"] = tavily_p["distance"]
                # Keep FHIR insurance but note Tavily matches too
                combined["data_source"] = "fhir+tavily"
                if combined.get("fhir_metadata"):
                    combined["fhir_metadata"]["tavily_matched"] = True
                logger.info(
                    "Merged FHIR+Tavily for %s (overlap=%.2f)",
                    combined["name"], best_score,
                )
                merged.append(combined)
            else:
                # FHIR-only provider
                merged.append(fhir_p)

        # Add unmatched Tavily providers
        for i, tavily_p in enumerate(tavily_providers):
            if i not in matched_tavily_indices:
                tavily_p["data_source"] = "tavily"
                merged.append(tavily_p)

        logger.info(
            "Merge result: %d FHIR, %d Tavily -> %d merged (%d matched)",
            len(fhir_providers),
            len(tavily_providers),
            len(merged),
            len(matched_tavily_indices),
        )
        return merged

    def gather_providers(self, specialty: str, location: str, insurance: Optional[str] = None,
                         enrich: bool = True, radius_miles: Optional[float] = None) -> Dict[str, Any]:
        """Main method to gather healthcare provider data.

        Args:
            specialty: Medical specialty to search for
            location: Location to search in (City, State with optional ZIP —
                the ZIP feeds distance computation, not the web query)
            insurance: Optional payer, recorded in search metadata for the
                audit trail only — it never rides in queries, never filters,
                never scores (coverage questions belong to the FHIR check)
            enrich: Run review enrichment inline (default True). The
                orchestrator passes False and calls enrich_providers() after
                core scoring instead, so the budget targets the likely top-K

        Returns:
            Dictionary containing providers list, search metadata, and status
        """
        self._identity_contradictions = []
        try:
            # Validate and sanitize inputs first
            validation_result = validate_search_params(specialty, location, insurance)

            if not validation_result["is_valid"]:
                errors = [e for e in validation_result["errors"] if e]
                logger.warning(f"Invalid search parameters: {errors}")
                return {
                    "providers": [],
                    "search_metadata": {
                        "query": "",
                        "specialty": specialty,
                        "location": location,
                        "insurance": insurance,
                        "total_found": 0,
                        "validation_errors": errors
                    },
                    "status": "invalid_input",
                    "message": f"Invalid search parameters: {', '.join(errors)}"
                }

            # Use sanitized values
            safe_specialty = validation_result["specialty"]
            safe_location = validation_result["location"]
            safe_insurance = validation_result["insurance"]

            logger.info(f"Starting provider search: {safe_specialty} in {safe_location}")

            # Candidates come from the live web only. The FHIR directory is a
            # verification prototype (see fhir/verify.py), not a candidate
            # source — mixing sandbox providers into real results is worse
            # than checking real results against the sandbox.
            # A raw ZIP stays out of the web query (it only adds noise there);
            # it feeds the distance computation below instead. A bare-ZIP
            # input is resolved to "City, ST" for the query.
            query_location = strip_zip(safe_location) or city_state_for_zip(safe_location) or safe_location

            # Discovery. Full page text is required, not a luxury: top results
            # are often directory/"best of" pages that name providers only in
            # the page body (raw content costs no extra Tavily credits).
            # Multi-query fans out several phrasings of the home city for
            # recall; a single query is the escape hatch.
            if self.config.MULTI_QUERY_ENABLED:
                home_specs = self._candidate_queries(safe_specialty, query_location)
                search_results = self._discover_candidates(
                    home_specs, self.config.MAX_PROVIDERS_PER_SEARCH
                )
                # Metadata carries one display string per CALL, with its
                # domain restriction visible. The three discovery calls share
                # one query string on purpose (one listing domain per call —
                # the restriction rides in include_domains, not the text),
                # and rendering the bare strings showed "three identical
                # queries" in the dev panel — which photographs exactly like
                # a triple-spend bug (owner run, 2026-08-09).
                home_queries = [_describe_query_spec(spec) for spec in home_specs]
            else:
                home_queries = [self._build_search_query(safe_specialty, query_location, safe_insurance)]
                search_results = self._search_providers(
                    home_queries[0],
                    max_results=self.config.MAX_PROVIDERS_PER_SEARCH,
                    include_raw_content=True,
                )
            query = home_queries[0]  # representative query for metadata
            queries_run = list(home_queries)

            providers: List[Dict[str, Any]] = []
            if search_results:
                providers = self._extract_provider_data(search_results, safe_specialty, safe_location)
                # Tagged BEFORE dedupe so the merge can see both sides. Tagging
                # afterwards would have nothing left to compare — the duplicate
                # is gone by then.
                for provider in providers:
                    provider["discovery_source"] = _DISCOVERY_HOME
                providers = self._dedupe_providers(providers)

                # Adaptive ring expansion: only when the home pool is thin —
                # INCLUDING empty, the thinnest case (live search results but
                # zero extractable providers is exactly the sparse-town rescue
                # scenario). Its cost (extra searches + one extraction) is paid
                # only where it helps.
                #
                # MIN_CANDIDATE_POOL equals MAX_PROVIDERS_TO_ENRICH by intent,
                # which gives this threshold a derivation rather than a
                # preference: ring out exactly when the home city cannot fill
                # the research budget. Deliberately NOT enforced as an
                # invariant — round 4's clamp between two knobs was removed on
                # purpose, and MIN_CANDIDATE_POOL=5 ("only rescue genuinely
                # sparse towns") is a legitimate setting.
                #
                # A second trigger — "or the pool is single-clustered" — was
                # deleted in round 10. It answered "distance can't tell these
                # providers apart" by importing providers from a city the user
                # didn't ask for, which is the overstatement round 7 rejected
                # one level down when it stopped city-centroid distances from
                # manufacturing a spread. Its metric was also unfixable: keyed
                # on ZIP-else-city it measured our ZIP extraction coverage
                # (10 providers in one city scored 1 or 4 depending on how many
                # addresses parsed), and keyed on city alone a single-city
                # search scores 1 and would ring out every time — on the ideal
                # outcome. The sparse-town case it was built for already trips
                # the thin-pool test above.
                if (self.config.MULTI_QUERY_ENABLED
                        and len(providers) < self.config.MIN_CANDIDATE_POOL):
                    ring = nearby_cities(
                        # The ring reaches exactly as far as the user allowed.
                        # Reaching further would import cities the radius bound
                        # then deletes — two searches and an extraction spent on
                        # rows that cannot survive.
                        query_location,
                        radius_miles or self.config.DEFAULT_SEARCH_RADIUS,
                        self.config.MAX_RING_CITIES
                    )
                    if ring:
                        logger.info(
                            f"Home pool thin ({len(providers)} providers, "
                            f"below {self.config.MIN_CANDIDATE_POOL}); ringing out to {ring}"
                        )
                        # The SAME specs the home city uses — one basic call
                        # per listing domain — not `_build_search_query`. The
                        # ring ran the old unrestricted advanced query long
                        # after home discovery moved to domain-restricted
                        # calls, so exactly the searches that fire on THIN
                        # pools returned listicles and practice sites the
                        # listing parsers cannot read, and every ring page
                        # went to the model. A combined one-call restriction
                        # is not the answer either: at basic depth
                        # include_domains measurably fails multi-domain, and
                        # at advanced one platform takes every slot — the
                        # run-to-run coverage swing round 20 killed. Costs
                        # 3 basic vs 1 advanced per ring city (+1 credit,
                        # only when the ring fires); buys parser-readable
                        # in-domain pages.
                        ring_specs = [
                            spec
                            for city in ring
                            for spec in self._candidate_queries(safe_specialty, city)
                        ]
                        queries_run.extend(_describe_query_spec(spec) for spec in ring_specs)
                        ring_results = self._discover_candidates(
                            ring_specs, self.config.MAX_PROVIDERS_PER_SEARCH
                        )
                        if ring_results:
                            ring_providers = self._extract_provider_data(
                                ring_results, safe_specialty, safe_location
                            )
                            for provider in ring_providers:
                                provider["discovery_source"] = _DISCOVERY_RING
                            home_count = len(providers)
                            providers = self._dedupe_providers(providers + ring_providers)
                            # The NET number, after dedupe and after home wins
                            # any overlap: what the two extra searches and the
                            # extra extraction actually bought. A provider both
                            # passes found is not a purchase.
                            logger.info(
                                "Ring expansion: %d home + %d ring extracted -> %d unique, "
                                "%d genuinely new",
                                home_count, len(ring_providers), len(providers),
                                sum(1 for p in providers
                                    if p.get("discovery_source") == _DISCOVERY_RING),
                            )

                if providers and enrich:
                    providers = self._enrich_missing_reviews(
                        providers, query_location, safe_specialty
                    )

            if not providers:
                return {
                    "providers": [],
                    "search_metadata": {
                        "query": query,
                        "specialty": safe_specialty,
                        "location": safe_location,
                        "insurance": safe_insurance,
                        "total_found": 0,
                        "fhir_count": 0,
                    },
                    "status": "no_results",
                    "message": "No providers found for the specified criteria"
                }

            # Honest location evidence, computed in code from vendored ZIP/
            # city centroids (utils/geo.py) — the LLM never estimates
            # distances. computed_distance_miles is None when the scraped
            # address can't be resolved; location_match is the tier fallback.
            for provider in providers:
                self._attach_location_evidence(provider, safe_location)

            # ...and then drop anyone who is not actually near the location the
            # user typed. A platform's city page is a RADIUS CENTRE, not a
            # filter — a Chandler search returns Gilbert, Mesa and Phoenix
            # entries — and until now nothing bounded that: distance was scored
            # and never used to exclude, so a provider 40 miles out competed for
            # a research-budget slot with one 4 miles out and lost by ~1 point,
            # because location's realized span across a pool is under a point.
            #
            # Deliberately a RADIUS, not a city match. Gilbert at 4.5 mi is
            # nearer to Chandler 85249 than two of that run's own Chandler
            # results at 8.0 mi, so "same city only" would have dropped the
            # closest provider on the page. The city the user typed is where
            # they are, not a boundary they want enforced.
            #
            # An UNKNOWN distance is never a reason to drop — that is our
            # geocoding coverage, not their location, and the same rule the
            # scorer follows when it declines to penalise missing data.
            # The user's chosen radius, falling back to the configured default.
            # This is the only lever that reliably keeps the research budget on
            # nearby providers: location's realized span across a pool is under
            # one point, so the WEIGHT cannot do it — the critic said as much on
            # a live run ("providers 8 miles away and providers under 4 miles
            # are separated by only a small amount in the final ordering").
            # Radius bounds the pool; weight orders it.
            radius = radius_miles or self.config.DEFAULT_SEARCH_RADIUS
            in_radius, out_of_radius = _split_by_radius(providers, radius)
            if out_of_radius:
                logger.info(
                    "Dropped %d provider(s) beyond the %d-mile search radius: %s",
                    len(out_of_radius), radius,
                    ", ".join(
                        f"{p.get('name')} ({p.get('computed_distance_miles'):.0f} mi)"
                        for p in out_of_radius[:5]
                    ),
                )
                providers = in_radius

            result = {
                "providers": providers,
                "search_metadata": {
                    "query": query,
                    "queries": queries_run,
                    "query_count": len(queries_run),
                    # Same-platform profile URLs that proved two records are
                    # different people. Surfaced rather than only logged: this
                    # is the one dedupe rule that can CREATE a duplicate, so a
                    # reader who sees the same doctor twice needs a way to find
                    # out why without reading a log file.
                    "identity_contradictions": list(self._identity_contradictions),
                    # Whether the ring fired, recorded where it is KNOWN rather
                    # than inferred downstream from a query count the UI would
                    # have to hardcode the home-phrasing total to interpret.
                    "ring_expanded": len(queries_run) > len(home_queries),
                    # How many candidates the radius bound removed, and what it
                    # was. Recorded because a filter that silently shrinks the
                    # pool is indistinguishable from a discovery failure — the
                    # symptom of both is "fewer providers than last time".
                    "radius_miles": radius,
                    "radius_dropped": len(out_of_radius),
                    # What it BOUGHT, which the boolean above never said. The
                    # ring's cost is two searches, an extraction, and — because
                    # it fills the research budget — enrichment, judge and
                    # critic on the providers it adds. Whether that is waste or
                    # the thing carrying the results is not answerable from a
                    # flag, and until this field existed the only way to guess
                    # was to read the card list and wonder which names looked
                    # out-of-town.
                    "ring_added": sum(
                        1 for p in providers
                        if p.get("discovery_source") == _DISCOVERY_RING
                    ),
                    "specialty": safe_specialty,
                    "location": safe_location,
                    "insurance": safe_insurance,
                    "total_found": len(providers),
                    "search_results_count": len(search_results),
                    "fhir_count": 0,
                    "tavily_count": len(providers),
                    "fhir_enabled": False,
                },
                "status": "success" if providers else "no_providers_extracted",
                "message": f"Found {len(providers)} {safe_specialty} providers in {safe_location}"
            }

            logger.info(f"Data gathering completed: {result['message']}")
            return result

        except Exception as e:
            logger.error(f"Provider gathering failed: {e}", exc_info=True)
            return {
                "providers": [],
                "search_metadata": {
                    "query": "",
                    "specialty": specialty,
                    "location": location,
                    "insurance": insurance,
                    "total_found": 0
                },
                "status": "error",
                "message": "Error gathering provider data. Please try again."
            }


def create_data_gatherer() -> DataGathererAgent:
    """Factory function to create a DataGathererAgent instance.

    Returns:
        DataGathererAgent instance
    """
    return DataGathererAgent()