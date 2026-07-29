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
# `ratemds.com` serves profiles under `/doctor-ratings/`, matching neither of
# its markers, so NO ratemds URL could ever be recognised as a profile — while
# ratemds sits on probation whose exit criterion is "a clean profile-based
# pair". It would have been dropped on an artifact of this table.
_PROFILE_PATH_MARKERS = {
    "healthgrades.com": ("/physician/", "/provider/", "/dentist/", "/doctor/"),
    "webmd.com": ("/doctor/", "/physician/"),
    "vitals.com": ("/doctors/",),
    "zocdoc.com": ("/doctor/", "/dentist/", "/psychiatrist/"),
    "ratemds.com": ("/doctor/", "/doctors/", "/doctor-ratings/"),
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
# and the UI. Roster history: google.com deliberately absent
# (unscrapeable pages wasted a priority slot); yelp.com dropped 2026-07-20
# (zero hits across every live run — physician coverage skews
# dental/chiro/urgent-care); health.usnews.com dropped 2026-07-21 (same
# disease as google: raw content is a 0–512-char JS shell even at advanced
# depth, yet its profile URLs rank high enough to steal extraction and
# result slots from readable platforms); ratemds.com restored 2026-07-21 ON
# PROBATION after one round out — it is readable when reached (Dr. An
# 13-review Chandler profile at score 0.82) and the last remaining
# corporate family independent of RVO Health (healthgrades) and Internet
# Brands (webmd+vitals), and with usnews's empty shells gone it competes
# for a fair slot; drop it for good if the next field tests still show no
# ratemds pairs.
REVIEW_PLATFORM_DOMAINS = (
    'healthgrades.com', 'vitals.com', 'zocdoc.com', 'webmd.com',
    'ratemds.com',
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
