"""Constructed review-platform listing URLs — discovery without search.

Tavily's August 2026 search overhaul (their changelog: reranking,
evidence-quality modeling, index coverage/temporal freshness) broke the
search-driven pipeline three measured ways on 2026-09-02:

  * relevance collapse — the one-domain listing query returned six dermatology
    directories, out-of-state neurology pages, and platform homepages, while
    the on-topic page (`doctor.webmd.com/providers/specialty/neurology/
    arizona/chandler`) sat in the index unreturned;
  * `include_domains` leakage — a vitals.com-restricted basic call returned
    10/20 results off-domain (YouTube, The New Yorker), falsifying the
    round-20 measurement ("one domain per call holds at basic, 5 of 5
    in-domain") that justified the search shape;
  * empty bodies — results ranking 0.98+ arrived with `raw_content` of 0
    chars while /extract returned the same pages whole.

The pages discovery actually wants have DETERMINISTIC URLs — every platform
publishes its city listing at a constructible path — so this module builds
them instead of asking a degraded index to find them. The same 2026-09-02
probe fetched all three constructed Chandler URLs via /extract in one batch
(1 credit) and the listing parsers read 114 rows, against the search path's
~38 surviving providers.

Slug tables cover `InputValidator.ALLOWED_SPECIALTIES` — the app's whitelist —
so every specialty reachable from the UI has a mapping BY CONSTRUCTION, the
same reasoning as the round-28 location pickers. Cities and states come from
the GeoNames-backed `parse_location`. A platform without a working listing
path for a specialty maps to None and contributes one page fewer — honest
degradation the caller logs, never a crash.

COVERAGE, measured by the 2026-09-02 full-allowlist /extract sweeps over
Chandler, AZ (row yield through the DETERMINISTIC listing parser — the LLM
fallback in `_extract_provider_data` still processes non-empty bodies the
parser can't read, so real discovery is at or above these numbers):

  * healthgrades: 23 of 24 specialty directories parse (pathology has no
    directory page at all, at city OR state level, in two states — mapped to
    None below rather than fetched). The FIRST sweep read only
    three; the other nineteen were `0 rows` from full 23-66 KB bodies
    because pages other than a city directory's page 1 write entry links
    host-relative and the heading regex demanded a scheme — fixed the same
    day (`listing_parser._HG_HEADING`). Nine directories render entries as
    heading + photo only (no rating in served text); those rows carry name
    + profile URL and the profile supplies the pair. All three platforms
    paginate their city listings (`listing_page_urls` — count-driven, each
    platform's own parameter, capped at 5 pages; the table beside it).
  * webmd: 24 of 24; city listing rows carry a profile URL but no rating pair.
  * vitals: 24 of 24.

Coverage re-measured 2026-09-17 by probing every allowlisted specialty on all
three platforms (`evals/probe_listing_slugs.py`). It found nine URLs that
served no directory, and every one was a SLUG rather than a gap in the
platform: six vitals stubs, two webmd near-empty bodies, and healthgrades'
pathology directory, which does not exist at any level and is mapped to None.
A wrong slug is invisible by construction — the platform answers with a
marketing page, not an error — so this sweep is the only thing that finds one,
and its result is the tables below.
"""

import re
from typing import Dict, List, Optional

# The geo module's canonical state-name table, inverted. Imported (privately)
# rather than copied so the two cannot drift — the same identity-over-copy
# rule utils/json_salvage ships under.
from utils.geo import _STATE_NAMES_TO_CODES, parse_location

_STATE_CODE_TO_NAME: Dict[str, str] = {
    code: name for name, code in _STATE_NAMES_TO_CODES.items()
}


def _slug(text: str) -> str:
    """Lowercase URL slug: non-alphanumerics collapse to single hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


# Per-platform specialty slugs, keyed by the allowlist's lowercase names.
# Only DEPARTURES from the naive `_slug(specialty)` are listed; everything
# else hyphenates as-is. Entries here were read off live URLs (the 2026-09-02
# search probes returned e.g. `cardiovascular-disease` and
# `endocrinology-diabetes-metabolism` for webmd) or verified by the
# 2026-09-02 full-allowlist /extract sweep recorded beside each table. A
# value of None means the platform has no per-city listing page for that
# specialty that we could verify — the builder returns None and discovery
# fetches one page fewer.

_HEALTHGRADES_SPECIALTY_SLUGS: Dict[str, Optional[str]] = {
    # healthgrades files primary care under "Family Practice" and merges
    # OB and GYN into one directory.
    "family medicine": "family-practice",
    "obstetrics": "obstetrics-gynecology",
    "gynecology": "obstetrics-gynecology",
    "orthopedics": "orthopedic-surgery",
    # No pathology directory exists: `/pathology-directory/<state>/<city>`,
    # the state page and a second state all returned an EMPTY body (2026-09-17).
    # None is the honest answer — discovery then fetches one page fewer for
    # this specialty instead of spending a credit on nothing.
    "pathology": None,
}

_WEBMD_SPECIALTY_SLUGS: Dict[str, Optional[str]] = {
    # Observed on live URLs 2026-09-02: webmd's vocabulary names the
    # discipline, not the colloquial specialty.
    "cardiology": "cardiovascular-disease",
    "endocrinology": "endocrinology-diabetes-metabolism",
    "obstetrics": "obstetrics-gynecology",
    "gynecology": "obstetrics-gynecology",
    "orthopedics": "orthopedic-surgery",
    # The two the 2026-09-17 allowlist sweep caught. Both served a ~6.4 KB
    # body that parsed to ZERO rows — recorded until then as "webmd serves
    # near-empty bodies for these", which was true of the WRONG SLUG and not
    # of the platform: the discipline names return 60 rows apiece.
    "general surgery": "surgery",
    "radiology": "diagnostic-radiology",
}

# vitals shares webmd's vocabulary (both are Internet Brands properties) and
# publishes it at `vitals.com/specialties`. Read off that index 2026-09-17 and
# then verified by fetching each candidate: the six specialties below served a
# 1,326-char marketing STUB under the naive slug — identical bytes in every
# city, which is why a wrong slug looks exactly like a city with no providers
# rather than like an error — and 41 to 59 parsed rows under the platform's own
# term. `family medicine` is the one that was actively WRONG rather than
# merely absent: it was set to healthgrades' "family-practice", which vitals
# does not use.
_VITALS_SPECIALTY_SLUGS: Dict[str, Optional[str]] = {
    "obstetrics": "obstetrics-gynecology",
    "gynecology": "obstetrics-gynecology",
    "orthopedics": "orthopedic-surgery",
    "family medicine": "family-medicine",
    "cardiology": "cardiovascular-disease",
    "endocrinology": "endocrinology-diabetes-metabolism",
    "general surgery": "surgery",
    "otolaryngology": "otolaryngology-head-neck-surgery",
    "radiology": "diagnostic-radiology",
}


def _specialty_slug(specialty: str, overrides: Dict[str, Optional[str]]) -> Optional[str]:
    key = " ".join((specialty or "").lower().split())
    if not key:
        return None
    if key in overrides:
        return overrides[key]
    return _slug(key)


def healthgrades_listing_url(specialty: str, city: str, state_code: str) -> Optional[str]:
    """e.g. https://www.healthgrades.com/neurology-directory/az-arizona/chandler"""
    slug = _specialty_slug(specialty, _HEALTHGRADES_SPECIALTY_SLUGS)
    state_name = _STATE_CODE_TO_NAME.get((state_code or "").upper())
    if not slug or not state_name or not city:
        return None
    return (
        f"https://www.healthgrades.com/{slug}-directory/"
        f"{state_code.lower()}-{_slug(state_name)}/{_slug(city)}"
    )


def webmd_listing_url(specialty: str, city: str, state_code: str) -> Optional[str]:
    """e.g. https://doctor.webmd.com/providers/specialty/neurology/arizona/chandler"""
    slug = _specialty_slug(specialty, _WEBMD_SPECIALTY_SLUGS)
    state_name = _STATE_CODE_TO_NAME.get((state_code or "").upper())
    if not slug or not state_name or not city:
        return None
    return (
        f"https://doctor.webmd.com/providers/specialty/{slug}/"
        f"{_slug(state_name)}/{_slug(city)}"
    )


def vitals_listing_url(specialty: str, city: str, state_code: str) -> Optional[str]:
    """e.g. https://www.vitals.com/neurology/az/chandler"""
    slug = _specialty_slug(specialty, _VITALS_SPECIALTY_SLUGS)
    if not slug or not city or not (state_code or "").strip():
        return None
    if (state_code or "").upper() not in _STATE_CODE_TO_NAME:
        return None
    return f"https://www.vitals.com/{slug}/{state_code.lower()}/{_slug(city)}"


def discovery_listing_urls(specialty: str, location: str) -> List[str]:
    """The constructed city-listing URLs for one specialty + "City, ST".

    Ordered healthgrades, webmd, vitals — the `_LISTING_DOMAINS` order the
    search-mode discovery calls use, so downstream page ordering sees the
    platforms in the same sequence either way. Empty when the location cannot
    be resolved to a known city + state (the caller treats that as "extract
    mode has nothing to fetch" and falls back to search-mode discovery).
    """
    parts = parse_location(location)
    city, state = parts.get("city"), parts.get("state")
    if not city or not state:
        return []
    urls = [
        healthgrades_listing_url(specialty, city, state),
        webmd_listing_url(specialty, city, state),
        vitals_listing_url(specialty, city, state),
    ]
    return [u for u in urls if u]


# Every platform paginates its city listing, each behind ITS OWN parameter —
# measured 2026-09-02 by reading the "Page 2" links off the pages and fetching
# what they pointed at:
#
#   healthgrades  <city>_2 … _N          20/page   (81 results = 5 pages)
#   webmd         <city>?pagenumber=N    ~50/page  (page 2: 46 rows, 43 new)
#   vitals        <city>?page=N          ~50/page  (page 2: 50 rows, 50 new)
#
# The TRAP: webmd's `?page=2` and vitals' `?pagenumber=2` both silently return
# page 1 again (0 new names) — each platform ignores the other's parameter,
# and the wrong one looks like success. Pinned by test. Five pages is the cap
# per platform: with three platforms that is at most 12 extra URLs, ~2-3
# credits, and page 1's twenty healthgrades names overlapped the webmd ∪
# vitals pool by only FOUR, so the later pages are where most of a city's
# doctors live.
LISTING_PAGE_SIZE: Dict[str, int] = {
    "healthgrades.com": 20,
    "webmd.com": 50,
    "vitals.com": 50,
}
LISTING_MAX_PAGES = 5
# Kept for the callers and comments that name them; the table above rules.
HEALTHGRADES_PER_PAGE = LISTING_PAGE_SIZE["healthgrades.com"]
HEALTHGRADES_MAX_PAGES = LISTING_MAX_PAGES
_ALREADY_PAGED = re.compile(r"(?:_\d+$|[?&](?:page|pagenumber)=\d+)", re.I)


def listing_page_urls(page_url: str, result_count: Optional[int]) -> List[str]:
    """URLs of listing pages 2..N for a page-1 URL, from the page's stated total.

    Empty when one page holds them all, when the count is unknown, when the
    platform is not one we paginate, or when the URL is already a later page
    (a page-2 URL must never plan a page 2 of its own).
    """
    lowered = (page_url or "").lower()
    domain = next((d for d in LISTING_PAGE_SIZE if d in lowered), None)
    if not domain or not result_count or _ALREADY_PAGED.search(page_url or ""):
        return []
    per_page = LISTING_PAGE_SIZE[domain]
    if int(result_count) <= per_page:
        return []
    pages = min(LISTING_MAX_PAGES, -(-int(result_count) // per_page))
    if domain == "healthgrades.com":
        return [f"{page_url}_{n}" for n in range(2, pages + 1)]
    separator = "&" if "?" in page_url else "?"
    parameter = "pagenumber" if domain == "webmd.com" else "page"
    return [f"{page_url}{separator}{parameter}={n}" for n in range(2, pages + 1)]


def healthgrades_page_urls(page_url: str, result_count: Optional[int]) -> List[str]:
    """healthgrades-only wrapper of listing_page_urls (the original hook)."""
    if "healthgrades.com" not in (page_url or "").lower():
        return []
    return listing_page_urls(page_url, result_count)
