"""Recognise a page body that arrived stripped of the content the parsers read.

Measured 2026-11-22/23. Tavily `/extract` began answering specific URLs from
a stored copy that had lost the page's substance: a healthgrades or webmd
city listing with no provider headings or profile links (16.5 KB and 14.9 KB
where the live page is 24.7 KB and 64.0 KB), a vitals profile cut to its
compare table (2.2 KB of 9.1 KB), a webmd profile without its rating block.
The stored copy came back byte-identical on every call, in 0.2-0.5 s for a
nine-URL batch — too fast to be a fetch — and the same request with
`use_cache: false` in the body took 4.5-8.3 s and returned every page whole
(9 of 9 URLs, three rounds). The listing parser read 0 rows from two of three
platforms, every researched provider carried vitals data only, and four
golden-set profiles read no rating: both canaries went red on a commit that
had been green the day before, with nothing wrong in the parsers or the
sites.

What makes the stripped copy recognisable is that it is not EMPTY. It still
carries the page's frame, so `raw_content` is non-empty and the empty-body
count stays at zero — which is why nothing upstream of the parsers noticed.
The frame is also what gives it away:

  * a LISTING still states its own result total ("We found 80 results",
    "146 Neurologist results") while holding no entry the parser can read.
    A page that says it lists 80 doctors and names none is contradicting
    itself, whatever the cause.
  * a PROFILE on a review platform reads neither a rating nor a review
    count. This one is weaker evidence — a doctor nobody has rated is a real
    thing — so it is only ever a reason to ask once more, never a verdict.
    Measured on the golden set the same day: 0 of 39 healthy profile pages
    parsed to neither number (webmd writes an unrated doctor as
    `0 (0 Ratings)`, which parses), so the extra fetch is rare on good pages.

  * a LISTING cut further still, to its frame alone. Found the day the
    refetch shipped, on the first canary run that used it: vitals' Gilbert
    family-medicine page came back as 5,476 chars holding `## Filter Family
    Physicians Results`, an explainer and an FAQ — no entry and, this time,
    NO TOTAL either, so the rule above passed it and the run reported
    "vitals: 0 rows". The same request past the cache returned 20,661 chars,
    `# 1,552 Family Physicians Found in Gilbert, AZ` and 56 rows. vitals'
    Gold Canyon neurology page had the same shape (5,458 chars, 0 rows; 13
    stated and 10 rows past the cache) — a page that had been read for
    weeks as "vitals serves this town nothing". The `## Filter … Results`
    heading is the listing's own filter panel: it is on every whole vitals
    listing measured (4 of 4) and is not on the marketing page a wrong slug
    is answered with, so a body that carries it and no row is a listing
    that lost its results, not a page that never had any.

Kept apart from the gatherer so the test for "is this body worth asking for
again" is one function with no client in it.
"""
import re
from typing import Optional

from utils.listing_parser import listing_result_count, parse_listing
from utils.profile_parser import parse_profile
from utils.provenance import url_page_kind

LISTING_WITHOUT_ROWS = "listing_without_rows"
PROFILE_WITHOUT_PAIR = "profile_without_pair"

# The three platforms whose pages have a parser. Anything else has no
# deterministic read to fail, so there is nothing to judge a body against.
_PARSED_PLATFORMS = ("healthgrades.com", "webmd.com", "vitals.com")

# vitals' filter panel heading, e.g. "## Filter Neurologists Results". Only
# vitals: the other two platforms' stripped listings kept their totals in
# every body measured, and a frame marker nobody has seen fail is a guess.
_VITALS_LISTING_FRAME = re.compile(r"^##\s+Filter\s+.+\s+Results\s*$", re.MULTILINE)


def _is_listing_frame(lowered_url: str, body: str) -> bool:
    return "vitals.com" in lowered_url and bool(_VITALS_LISTING_FRAME.search(body))


def degraded_reason(url: Optional[str], text: Optional[str]) -> Optional[str]:
    """Why this body looks stripped, or None when it reads normally.

    Profile-shaped URLs are judged as profiles and everything else on a
    parsed platform as a possible listing. The listing test does not consult
    the URL's shape at all (vitals' city listing has no path marker and is
    classified "unknown"): a body that states a result total, or carries the
    listing's filter panel, is a listing by its own account, and a profile
    has neither.
    """
    lowered = str(url or "").lower()
    body = text or ""
    if not body or not any(platform in lowered for platform in _PARSED_PLATFORMS):
        return None
    if url_page_kind(url) == "profile":
        parsed = parse_profile(url, body) or {}
        if parsed.get("rating") is None and parsed.get("review_count") is None:
            return PROFILE_WITHOUT_PAIR
        return None
    stated = listing_result_count(url, body)
    if (stated or _is_listing_frame(lowered, body)) and not parse_listing(url, body):
        return LISTING_WITHOUT_ROWS
    return None
