"""Provenance helpers: attribute extracted claims to their source pages."""

from urllib.parse import urlparse

_PRACTICE_SITE_MARKER = " — practice site"
_LISTING_PAGE_MARKER = " — listing page"

# Path fragments that mark an INDIVIDUAL provider's profile on each platform,
# as opposed to a city/specialty directory ("Best Neurologists in Chandler")
# whose ratings belong to many doctors, not the one we're attributing. Used to
# prefer profile URLs when platforms tie and to label directory links honestly.
#
# These are matched against the PATH, not the whole URL — the old version
# searched the entire lowercased string, so a host or query could satisfy a
# path marker.
#
# The original set was written in the same commit as its own tests and was
# never checked against a real URL. Two were wrong in ways that mattered:
# `vitals.com` required `/doctors/dr`, i.e. the slug had to BEGIN with "dr",
# which the repo's own fixtures (`/doctors/hodgson`) already failed; and
# `ratemds.com` (on the roster until 2026-08-04) served profiles under
# `/doctor-ratings/`, matching neither of its markers, so NO ratemds URL could
# ever be recognised as a profile — while its probation exit criterion was "a
# clean profile-based pair". It would have been dropped on an artifact of this
# table; it was ultimately dropped on a field measurement instead.
_PROFILE_PATH_MARKERS = {
    "healthgrades.com": ("/physician/", "/provider/", "/dentist/", "/doctor/"),
    "webmd.com": ("/doctor/", "/physician/"),
    "vitals.com": ("/doctors/",),
}

# Path fragments that positively mark a MULTI-provider directory page. Checked
# first: several platforms nest listings under the same root as profiles.
_LISTING_PATH_MARKERS = (
    "/usearch", "/search", "/find", "/directory", "-directory/",
    "/best-", "/browse", "/specialty/", "/city/", "/near-me",
)

# Independent patient-review platforms — the only acceptable sources for
# rating/review_count evidence (a practice's own testimonial page is
# self-published marketing). Shared by the gatherer (search targeting,
# extraction priority, headline class), the scorer (cross-platform blend),
# and the UI. This list SIZES the enrichment payload: the per-provider search
# asks for 2x this many result slots and the extractor reads len+1 blocks, so
# every domain here bills Haiku input on every enriched provider whether or
# not it ever yields a pair.
#
# Roster history: google.com deliberately absent (unscrapeable pages wasted a
# priority slot); yelp.com dropped 2026-07-20 (zero hits across every live
# run — physician coverage skews dental/chiro/urgent-care);
# health.usnews.com dropped 2026-07-21 (raw content is a 0–512-char JS shell
# even at advanced depth, yet its profile URLs rank high enough to steal
# extraction and result slots from readable platforms); zocdoc.com and
# ratemds.com dropped 2026-08-04 by the round-20 field measurement — zocdoc
# returned 10 pages with a rating on none and no classifiable URL; ratemds
# returned 2 pages and no pair, which is precisely its probation exit
# criterion ("drop for good if the next field tests still show no ratemds
# pairs" — they did). Discovery had already excluded both from
# _LISTING_DOMAINS on the same measurement; until this change enrichment
# still granted each a lead round-robin slot ahead of a second page from a
# platform that does produce pairs, and their two blocks were ~a third of
# the extractor's page payload. Cached observations from either domain now
# read as non-platform: excluded from the blend and the platform count,
# which is what the measurement says they always deserved.
REVIEW_PLATFORM_DOMAINS = (
    'healthgrades.com', 'vitals.com', 'webmd.com',
)


def label_source(url, website=None) -> str:
    """Display label for a source: its domain, flagged when self-published.

    A source hosted on the provider's own website (or an obvious testimonial
    page) is marketing, not independent review data — the label says so:
    "chandlerneurologyandsleep.com — practice site".
    """
    domain = source_domain(url)
    if not domain:
        return ""
    site_domain = source_domain(website) if website else ""
    if (site_domain and domain == site_domain) or "testimonial" in str(url).lower():
        return f"{domain}{_PRACTICE_SITE_MARKER}"
    # A platform link that lands on a directory/listing page rather than the
    # provider's own profile is labeled, so a patient isn't sent to a "best
    # neurologists in <city>" index expecting this doctor's page.
    #
    # Only a CONFIRMED listing earns the label. This used to fire on anything
    # that wasn't a recognised profile, so an unrecognised-but-real profile URL
    # was announced to the patient as a directory index — a warning that is
    # false is worse than no warning, because it trains the eye to ignore the
    # one that is true.
    if _is_review_platform(domain) and url_page_kind(url) == "listing":
        return f"{domain}{_LISTING_PAGE_MARKER}"
    return domain


def _is_review_platform(domain: str) -> bool:
    """True when a bare domain belongs to one of the review platforms."""
    return any(platform in domain for platform in REVIEW_PLATFORM_DOMAINS)


def url_page_kind(url) -> str:
    """Classify a URL as "profile", "listing" or "unknown".

    Three states, not two. The previous boolean forced every unrecognised
    shape to be called a listing, which is an assertion we cannot support: a
    URL we don't have a pattern for is a URL we haven't identified, and saying
    "listing page" about a doctor's real profile is exactly as wrong as the
    reverse. "unknown" lets the label stay silent and lets tie-breaks treat it
    as neither better nor worse than a confirmed listing.
    """
    domain = source_domain(url)
    if not domain:
        return "unknown"

    markers = None
    for platform, platform_markers in _PROFILE_PATH_MARKERS.items():
        if platform in domain:
            markers = platform_markers
            break
    if markers is None:
        # Not a review platform — a practice site or article, where the
        # profile/listing distinction is meaningless.
        return "profile"

    path = (urlparse(str(url)).path or "").lower()
    if not path or path == "/":
        return "listing"  # a bare platform domain attributes nothing
    if any(marker in path for marker in _LISTING_PATH_MARKERS):
        return "listing"
    if any(marker in path for marker in markers):
        return "profile"
    return "unknown"


def is_profile_url(url) -> bool:
    """True when a URL is a confirmed individual-provider profile.

    Kept as the boolean face of `url_page_kind` for call sites that only need
    "is this attributable to one person". Note `unknown` is False here — use
    `url_page_kind` directly wherever an unrecognised shape must not be
    treated as a confirmed listing.
    """
    return url_page_kind(url) == "profile"


def linkable(url) -> bool:
    """True when the URL is safe to embed as a markdown link target."""
    text = str(url or "").strip()
    return text.startswith(("http://", "https://")) and not any(c in text for c in " ()<>\"'")


def source_domain(url) -> str:
    """Bare display domain of a source URL ("healthgrades.com"); "" if none.

    Tolerates scheme-less URLs the extractor sometimes returns.
    """
    if not url:
        return ""
    text = str(url).strip()
    if not text:
        return ""
    try:
        netloc = urlparse(text).netloc or urlparse("https://" + text).netloc
    except ValueError:
        return ""
    domain = netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def canonical_profile_url(url) -> str:
    """A profile URL reduced to a comparable key; "" when there is none.

    Identity comparisons are string equality, so the same page reached four
    ways has to reduce to one string:

        https://www.healthgrades.com/physician/dr-hemant-pandey-xsjwm
        https://healthgrades.com/physician/dr-hemant-pandey-xsjwm/
        https://www.healthgrades.com/physician/dr-hemant-pandey-xsjwm?ref=search
        https://www.healthgrades.com/physician/dr-hemant-pandey-xsjwm#locations
            -> healthgrades.com/physician/dr-hemant-pandey-xsjwm

    The listing parsers already stop their capture at "#", but an enrichment
    observation's `source_url` comes straight from the search vendor and will
    not be clean, so the two would otherwise never compare equal.
    """
    domain = source_domain(url)
    if not domain:
        return ""
    text = str(url).strip()
    try:
        parsed = urlparse(text if "//" in text else "https://" + text)
    except ValueError:
        return ""
    path = (parsed.path or "").rstrip("/").lower()
    return f"{domain}{path}" if path else domain


def urls_contradict(url_a, url_b) -> bool:
    """True only when two URLs prove their subjects are DIFFERENT people.

    That is one case and one case only: both present, SAME platform domain,
    different paths. A platform issues one profile record per provider, so two
    different `/physician/…` paths are two different physicians — a fact no
    name comparison can establish, which is what makes this worth having.

    Everything else is silence, and the distinction matters:

      * DIFFERENT domains is not contradiction. The same doctor has a different
        URL on every platform, so treating that as a conflict would block every
        cross-platform pairing and the two-platform blend would never engage.
      * A MISSING URL is not contradiction. Roughly half of one platform's
        listing rows carry no link at all, so a rule that required one would
        quietly shrink the pool. URL identity is a preference, never a
        requirement.
    """
    key_a, key_b = canonical_profile_url(url_a), canonical_profile_url(url_b)
    if not key_a or not key_b:
        return False
    domain_a, domain_b = source_domain(url_a), source_domain(url_b)
    if domain_a != domain_b:
        return False
    return key_a != key_b
