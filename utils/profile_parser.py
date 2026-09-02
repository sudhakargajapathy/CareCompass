"""Deterministic parsers for a provider's own PROFILE page on a review platform.

Companion to `utils.listing_parser`. A listing gives the roster; a profile is
where a single provider's rating, review total and tenure are stated for them
alone, so it is the attributable source — a number scraped off a
"Best doctors for autonomic disorders in Chandler" index is true of many
doctors, not this one.

EVERY PATTERN HERE WAS READ OFF A REAL FETCHED PAGE. The first version of this
module inferred the webmd and vitals profile wordings from those platforms'
LISTING markup, and was wrong about both: a webmd profile never uses the
listing's `[4.5 61 Ratings]` bracket form, and vitals writes its header pair as
a LINK (`[5 9](#rating-overview)`). Run against real pages, that version
returned tenure and nothing else for two of the three platforms. Inferring one
page's markup from another page on the same site is not evidence — if a shape
below is ever extended, extend it from a fetched page.

THE THREE PLATFORMS, and what each profile actually states:

  healthgrades — TWO templates, and only one carries a count:
      A)  4.3 Star Rating
          Based on 6 reviews                                   rating + count
      B)  Likelihood of recommending Dr. Pandey to family and
          friends is 3.7727273 out of 5                        rating ONLY
    Template B states no total ANYWHERE in the fetched text. Its true count (88,
    per the directory listing) appears 28 times in the raw page and ZERO times
    once the inline SVG is stripped — every one of those 28 was a vector path
    coordinate. So a healthgrades parse may legitimately return a rating with
    `review_count is None`, and a future "fix" that invents one is reading a
    logo. Its count comes from the listing parser instead.

  webmd — the header states the pair on consecutive lines, and the FAQ restates
    it in prose. BOTH survive extraction:
        4.5
        (61 Ratings)
        26 Years Experience
        2201 W Fairview St Ste 1, Chandler, AZ, 85224
      ...
        ### What are Dr. Cinthi Pillai's patient ratings?
        Dr. Cinthi Pillai has received 20 ratings on WebMD Care. Patients gave
        Dr. Cinthi Pillai an average rating of 5 out of 5.
    Tenure is stated TWICE and the two DISAGREE — the provider's self-written
    Overview blurb says "over 18 years of experience" where webmd's own header
    chip and FAQ both say 20. Unanchored first-match takes the blurb, so tenure
    is read from the header region and the FAQ ONLY, never from a whole-page
    search.

  vitals — the header pair is a LINK to the ratings anchor:
        [5 9](#rating-overview)         rating 5, over 9 ratings
        14 Years of Experience 2201 W Fairview St Ste 1 Chandler AZ 85224
    On a profile where that line is absent — observed on a CLAIMED profile,
    which renders "Overview based on verified provider data" in place of the
    "Is this you? / Claim your Profile" block — the rating survives only inside
    `## Compare with Similar Doctors`, a table whose other three columns are
    different doctors. That table is read by locating the subject's COLUMN (the
    one whose last row says `Current Profile` where the others say
    `View Profile`), never by position, and the count it yields must EQUAL the
    count in the doctor's own `### 7 ratings & reviews` heading. Two independent
    anchors agreeing is what makes reading a neighbour-bearing table safe; when
    they disagree the pair is dropped rather than publishing a stranger's stars.

WHAT PROFILES DO NOT CARRY, so nobody writes a pattern for it later:
healthgrades' `### Practice` and vitals' `## Office Locations` are EMPTY
headings — the body is JS-rendered and never reaches the payload, exactly like
webmd's `## Ratings & Reviews` section, which is why webmd's numbers had to be
found in its FAQ. Street addresses for healthgrades come from the listing
parser. A pattern that appears to find one there is matching something else.

webmd's `## Locations` section is the measured EXCEPTION (2026-08-06, saved
extract bytes): it arrives with the full office list — practice link, street
line, `CITY,ST,ZIP` line, optional `Tel:` link, and a `Get Directions` link
whose `destination=` parameter carries the complete address URL-encoded. The
same fetch showed a trailing "Show All", so even a full-looking list is
TRUNCATED — absence of an address from the parsed set is never evidence, which
is why the conflict resolver treats set membership as confirm-only. The list
can also arrive EMPTY on some copies of some doctors (Dr. Kumar's fetch:
heading immediately followed by the next section) while other pages of the
same shape carry five entries — per-copy, not per-platform.
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

# Tavily's markdown inlines images as data URIs, and one of them is enormous:
# on a fetched healthgrades profile a single percent-encoded SVG (the Zocdoc
# logo) ran 8,345 of the page's 15,750 characters — 53% — and carried 1,246 of
# its 1,299 bare integers as vector path coordinates. Stripping it first is what
# makes "count the numbers on this page" a meaningful operation at all. Note it
# is NOT base64 despite the name every data URI shares; matching `;base64,`
# finds nothing here.
_DATA_URI = re.compile(r"data:[a-zA-Z0-9.+/-]+(?:;[a-zA-Z0-9=+/-]+)*,[^\s)]{200,}")

# The subject's own name, as the page's H1. All three platforms write it there
# and only there, which makes it a deterministic `page_provider_name` for the
# identity check — better than asking the cheapest model to transcribe it.
_H1 = re.compile(r"^#\s+(?P<name>[^\n#].*?)\s*$", re.M)

# --- healthgrades ---------------------------------------------------------
# Sections that describe OTHER doctors. Both healthgrades rating patterns are
# unanchored first-match, so without this bound a neighbour's "4.9 Star Rating"
# in a promo strip is indistinguishable from the subject's. Nothing bites on the
# pages measured so far — the bound is here because the equivalent hazard on
# vitals (a comparison table listing three other doctors' ratings) is real and
# these two are the same shape.
_HG_NEIGHBOUR_SECTION = re.compile(
    r"^#{1,4}\s*(?:Compare Providers|Compare with|You May Also Like|Similar Providers"
    r"|Explore More Providers|Recommended Reading)\b", re.M | re.I)
_HG_STAR_PAIR = re.compile(
    r"(?P<rating>\d(?:\.\d+)?)\s*Star Rating\s*\(\s*(?P<count>\d+)\s+reviews?\s*\)", re.I)
_HG_BASED_ON = re.compile(r"Based on\s+(?P<count>\d+)\s+reviews?", re.I)
_HG_STAR_ONLY = re.compile(r"(?P<rating>\d(?:\.\d+)?)\s*Star Rating", re.I)
_HG_LIKELIHOOD = re.compile(
    r"Likelihood of recommending[^\n]{0,80}?is\s+(?P<rating>\d+(?:\.\d+)?)\s+out of\s*5", re.I)

# --- webmd ----------------------------------------------------------------
# The header runs above the first level-2 heading (`## Overview`). Everything
# below it is prose the provider wrote about themselves, including the tenure
# figure that disagrees with webmd's own.
_WEBMD_BODY_START = re.compile(r"^##\s+", re.M)
# "4.5" then "(61 Ratings)" — separate lines in the fetched markdown, the same
# line in some renders. `\s{0,4}` spans a blank line between them and nothing
# larger, so an unrelated number further up the page cannot reach the count.
_WEBMD_HEADER_PAIR = re.compile(
    r"(?<![\d.])(?P<rating>\d(?:\.\d+)?)\s{0,4}\(\s*(?P<count>\d+)\s+Ratings?\s*\)", re.I)
_WEBMD_FAQ_RATING = re.compile(
    r"average rating of\s+(?P<rating>\d+(?:\.\d+)?)\s+out of\s*5", re.I)
_WEBMD_FAQ_COUNT = re.compile(
    r"has received\s+(?P<count>\d+)\s+ratings?\s+on\s+WebMD", re.I)
_WEBMD_FAQ_YEARS = re.compile(
    r"has approximately\s+(?P<years>\d+)\s+years?\s+of\s+experience", re.I)

# --- vitals ---------------------------------------------------------------
# `#rating-overview` is the whole guard: it is the anchor the header's rating
# widget links to, and no other bracketed number pair on the page targets it.
_VITALS_HEADER_PAIR = re.compile(
    r"\[\s*(?P<rating>\d(?:\.\d+)?)\s+(?P<count>\d+)\s*\]\(\s*#rating-overview\s*\)", re.I)
# The doctor's OWN ratings heading. Carries the count and never the rating.
_VITALS_COUNT_HEADING = re.compile(
    r"^#{2,4}\s*(?P<count>\d+)\s+ratings?\s*&\s*reviews?\s*$", re.M | re.I)
_VITALS_YEARS_HEADING = re.compile(
    r"^#{2,4}\s*(?P<years>\d+)\s+Years?\s+(?:of\s+)?Experience\s*$", re.M | re.I)
_VITALS_COMPARE_SECTION = re.compile(
    r"^##\s+Compare with Similar Doctors\s*$(?P<body>[\s\S]*?)(?=^##\s|\Z)", re.M | re.I)
_TABLE_ROW = re.compile(r"^\s*\|(?P<cells>.*)\|\s*$", re.M)
_VITALS_CELL_PAIR = re.compile(
    r"^(?P<rating>\d(?:\.\d+)?)\s*\(\s*(?P<count>\d+)\s+Ratings?\s*\)$", re.I)
_CURRENT_PROFILE_CELL = re.compile(r"^Current Profile$", re.I)
# `/neurology/az/chandler` — specialty, state and city as URL path segments
# rather than prose, on every vitals breadcrumb.
_VITALS_BREADCRUMB = re.compile(
    r"vitals\.com/(?P<specialty>[a-z][a-z-]{2,40})/(?P<state>[a-z]{2})/(?P<city>[a-z][a-z-]{1,40})\b",
    re.I)

# --- shared ---------------------------------------------------------------
# "25+ years of experience", "26 Years Experience", "14 Years of Experience"
_YEARS = re.compile(r"(?P<years>\d+)\s*\+?\s*years?\s+(?:of\s+)?exp(?:erience)?\b", re.I)
# healthgrades separates suite from city with a MIDDLE DOT, and webmd puts a
# comma before the ZIP ("Chandler, AZ, 85224"). `utils.geo.parse_location`
# splits on commas and reads the ZIP end-anchored, so both need normalising.
_ADDRESS = re.compile(
    r"(?P<address>\d[^\n\]()#]{4,90}?[·,]\s*[A-Za-z .'-]{2,40},\s*[A-Z]{2},?\s*\d{5})")
# vitals writes the header address with NO commas at all:
#   "2201 W Fairview St Ste 1 Chandler AZ 85224"
# Street and city cannot be separated without them — "123 Main St Phoenix AZ"
# gives no rule that survives both "Main St Phoenix" and "New York". So one
# comma goes in before the state, which is all that matters: it leaves the ZIP
# end-anchored where `utils.geo` reads it, and the resulting "city" is an
# obvious street address that `normalized_place` already refuses. ZIP-precision
# distance is the win; a city guess we cannot justify is not.
_VITALS_ADDRESS = re.compile(
    r"(?P<street>\d[A-Za-z0-9 .'#/-]{6,80}?)\s+(?P<state>[A-Z]{2})\s+(?P<zip>\d{5})(?!\d)")
# The vitals header line runs tenure straight into the address with no
# separator — "14 Years of Experience 2201 W Fairview St Ste 1 Chandler AZ
# 85224" — so the address match, which starts at the first digit it sees,
# begins at the tenure. Stripped by name rather than by trying to find where
# the street "really" starts: the last house-number-looking token in that line
# is the SUITE number, so "take the rightmost" yields "1 Chandler".
_VITALS_LEADING_TENURE = re.compile(r"^\s*\d+\s*\+?\s*years?\s+(?:of\s+)?exp\w*\s+", re.I)
_MIDDLE_DOT = re.compile(r"\s*·\s*")
_STATE_ZIP_COMMA = re.compile(r",\s*([A-Z]{2}),\s*(\d{5})\b")


def strip_data_uris(text: str) -> str:
    """Remove inlined image payloads, which are most of some pages by volume.

    Exported because the same blob wastes the enrichment excerpt's budget: on a
    fetched healthgrades profile the 8,345-char SVG begins at character 651, so
    a 1,200-char head reservation meant to capture the rating header spends 46%
    of itself on a logo. `strip_boilerplate` does not catch it — its nav filters
    only apply to lines under 80 characters and this is one line of 8,345.
    """
    return _DATA_URI.sub(" ", str(text or ""))


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pairs(found: Dict[str, Any], rating: Any = None, count: Any = None) -> None:
    """Record a rating/count without letting a later, weaker source overwrite."""
    if rating is not None and found.get("rating") is None:
        found["rating"] = rating
    if count is not None and found.get("review_count") is None:
        found["review_count"] = count


def _page_name(text: str) -> Optional[str]:
    match = _H1.search(text)
    if not match:
        return None
    name = match.group("name").strip().strip("*").strip()
    return name or None


def _address(text: str) -> Optional[str]:
    match = _ADDRESS.search(text)
    if not match:
        return None
    address = _MIDDLE_DOT.sub(", ", match.group("address")).strip()
    # "Chandler, AZ, 85224" -> "Chandler, AZ 85224": geo's ZIP pattern is
    # end-anchored to the address and a comma before the ZIP breaks it.
    return _STATE_ZIP_COMMA.sub(r", \1 \2", address)


def _table_rows(body: str) -> List[List[str]]:
    """Markdown table body -> rows of stripped cells (separator rows dropped)."""
    rows: List[List[str]] = []
    for match in _TABLE_ROW.finditer(body):
        cells = [c.strip() for c in match.group("cells").split("|")]
        if cells and all(set(c) <= set("-: ") for c in cells if c):
            continue          # the |---|---| separator
        rows.append(cells)
    return rows


def _vitals_compare_pair(text: str) -> Tuple[Optional[float], Optional[int]]:
    """Rating + count from the subject's column of the comparison table.

    The subject is identified STRUCTURALLY, never positionally: their column is
    the one whose last row reads `Current Profile` where every other column
    reads `View Profile`. Position would be the same mistake as the ranking
    fallback that once let a provider collect their neighbour's penalty — the
    table happens to put the subject first on the page measured, and "happens
    to" is not a rule.
    """
    section = _VITALS_COMPARE_SECTION.search(text)
    if not section:
        return None, None
    rows = _table_rows(section.group("body"))

    subject_index = None
    for cells in rows:
        for index, cell in enumerate(cells):
            if _CURRENT_PROFILE_CELL.match(cell):
                subject_index = index
                break
        if subject_index is not None:
            break
    if subject_index is None:
        return None, None

    for cells in rows:
        if subject_index >= len(cells):
            continue
        pair = _VITALS_CELL_PAIR.match(cells[subject_index])
        if pair:
            return _float(pair.group("rating")), _int(pair.group("count"))
    return None, None


_COMPARE_CELL_YEARS = re.compile(r"^(?P<years>\d+)\s+Years?\s+Experience$", re.I)
_NAME_NOISE = {"dr", "md", "do", "phd", "np", "pa", "od", "dds", "dpm"}


def _compare_subject_column(text: str) -> Optional[List[str]]:
    """The subject's own column of a `## Compare with Similar Doctors` table.

    Shared by vitals and webmd — the two platforms run on one backend and
    ship the same comparison widget: a table of the subject plus three
    neighbours whose LAST row reads `Current Profile` under the subject and
    `View Profile` under everyone else. Identified structurally by that
    marker, never by position (the subject happens to sit first on every
    page measured, and "happens to" is not a rule).

    Second guard, added for webmd (2026-09-02): the column's own name cell
    must agree with the page H1. On a fetched page the marker is the only
    thing separating the subject from a neighbour, and a neighbour's stars on
    a card is the failure every identity rule in this repo exists to prevent.
    A page without an H1 degrades to the structural rule alone.
    """
    section = _VITALS_COMPARE_SECTION.search(text)
    if not section:
        return None
    rows = _table_rows(section.group("body"))
    subject_index = None
    for cells in rows:
        for index, cell in enumerate(cells):
            if _CURRENT_PROFILE_CELL.match(cell):
                subject_index = index
                break
        if subject_index is not None:
            break
    if subject_index is None:
        return None
    column = [cells[subject_index] for cells in rows if subject_index < len(cells)]

    page_name = _page_name(text)
    if page_name:
        def tokens(value: str) -> set:
            return {t for t in re.findall(r"[a-z]+", value.lower()) if t not in _NAME_NOISE}
        wanted = tokens(page_name)
        seen = tokens(" ".join(column))
        if wanted and len(wanted & seen) / len(wanted) < 0.5:
            return None
    return column


def _compare_column_pair(column: List[str]) -> Tuple[Optional[float], Optional[int]]:
    for cell in column:
        pair = _VITALS_CELL_PAIR.match(cell)
        if pair:
            return _float(pair.group("rating")), _int(pair.group("count"))
    return None, None


def _compare_column_years(column: List[str]) -> Optional[int]:
    for cell in column:
        years = _COMPARE_CELL_YEARS.match(cell)
        if years:
            return _int(years.group("years"))
    return None


def _parse_healthgrades(text: str) -> Dict[str, Any]:
    neighbour = _HG_NEIGHBOUR_SECTION.search(text)
    own = text[:neighbour.start()] if neighbour else text

    # `rating_pattern` records WHICH template produced the rating — a measured
    # fact that decides what a downstream merge may trust. Template A
    # ("N.N Star Rating") carries "Based on N reviews" in its payload, so a
    # thin fetch losing the total is a RECOVERABLE gap: Tavily's chunks may
    # legitimately complete the pair. Template B ("Likelihood of recommending
    # … is N out of 5") states NO total anywhere in the payload, so a count
    # appearing only in the chunks of a template-B page cannot be this
    # doctor's — it is a neighbour's, off a promo strip whose stars the chunk
    # boundary happened to cut. Without the marker the two cases are
    # indistinguishable at the merge.
    found: Dict[str, Any] = {}
    pair = _HG_STAR_PAIR.search(own)
    if pair:
        _pairs(found, _float(pair.group("rating")), _int(pair.group("count")))
        found["rating_pattern"] = "star"
    else:
        star = _HG_STAR_ONLY.search(own)
        based = _HG_BASED_ON.search(own)
        _pairs(found,
               _float(star.group("rating")) if star else None,
               _int(based.group("count")) if based else None)
        if found.get("rating") is not None:
            found["rating_pattern"] = "star"
        else:
            # Template B: a rating and no total anywhere. By design — see the
            # module docstring before "fixing" the missing count.
            likelihood = _HG_LIKELIHOOD.search(own)
            if likelihood:
                _pairs(found, _float(likelihood.group("rating")))
                found["rating_pattern"] = "likelihood"

    years = _YEARS.search(own)
    if years:
        found["years_experience"] = _int(years.group("years"))
    return found


# --- webmd Locations section ---------------------------------------------
# The one profile section measured to carry a full office LIST (see the module
# docstring). Anchored on the `Get Directions` links: their `destination=`
# parameter is machine-written and URL-decodes to the complete address with
# commas in place ("14520 W GRANITE VALLEY DR STE 120, SUN CITY WEST, AZ
# 85375") — far more robust than stitching the street and `CITY,ST,ZIP` lines,
# which render differently between fetch tiers. Each block's `Tel:` link sits
# BETWEEN the previous block's directions link and its own, which is how a
# phone stays welded to its own office — Dr. Vandian's card showed the Gilbert
# listing address beside the PHOENIX office's phone because address and phone
# were read from different blocks with nothing pairing them.
_WEBMD_LOCATIONS_HEADING = re.compile(r"^##\s+Locations\s*$", re.M)
_H2_HEADING = re.compile(r"^##\s+", re.M)
_GET_DIRECTIONS = re.compile(r"\[Get Directions\]\([^)]*?destination=(?P<dest>[^)&]+)")
_TEL_LINK = re.compile(r"Tel:\s*\[(?P<tel>[^\]]+)\]")


def _webmd_locations(text: str) -> List[Dict[str, Optional[str]]]:
    """Every office block in the `## Locations` section, page order, with its
    own phone. Empty list when the section is absent or arrived empty."""
    heading = _WEBMD_LOCATIONS_HEADING.search(text)
    if not heading:
        return []
    section_end = _H2_HEADING.search(text, heading.end())
    section = text[heading.end():section_end.start() if section_end else len(text)]

    locations: List[Dict[str, Optional[str]]] = []
    previous_end = 0
    for match in _GET_DIRECTIONS.finditer(section):
        address = unquote(match.group("dest")).strip()
        if not address:
            continue
        tel = _TEL_LINK.search(section, previous_end, match.start())
        locations.append({
            "address": address,
            "phone": tel.group("tel").strip() if tel else None,
        })
        previous_end = match.end()
    return locations


def _parse_webmd(text: str) -> Dict[str, Any]:
    body = _WEBMD_BODY_START.search(text)
    header = text[:body.start()] if body else text

    found: Dict[str, Any] = {}
    pair = _WEBMD_HEADER_PAIR.search(header)
    if pair:
        _pairs(found, _float(pair.group("rating")), _int(pair.group("count")))

    # The FAQ restates both, and is the fallback when the header card did not
    # survive extraction. It is authoritative for tenure either way.
    faq_rating = _WEBMD_FAQ_RATING.search(text)
    faq_count = _WEBMD_FAQ_COUNT.search(text)
    _pairs(found,
           _float(faq_rating.group("rating")) if faq_rating else None,
           _int(faq_count.group("count")) if faq_count else None)

    # Header first, FAQ second, and NEVER a whole-page search: the provider's
    # own Overview blurb states a different, lower number ("over 18 years"
    # against webmd's 20) and sits above the FAQ, so first-match takes it.
    years = _YEARS.search(header) or _WEBMD_FAQ_YEARS.search(text)
    if years:
        found["years_experience"] = _int(years.group("years"))

    # THIRD source, weakest, structurally bound: the comparison table. A
    # fetched webmd profile (Dr. Kan Yu, 2026-09-02, 10,511 chars via
    # /extract) carried NO header card and NO FAQ restatement — its
    # `## Ratings & Reviews` section read "No data" — while
    # `## Compare with Similar Doctors` stated `4.5 (151 Ratings)` and
    # `40 Years Experience` in the subject's column, the exact widget the
    # vitals parser already reads. Consulted only when the page stated
    # neither number elsewhere, so it can never overrule the header or FAQ.
    if found.get("rating") is None and found.get("review_count") is None:
        column = _compare_subject_column(text)
        if column:
            table_rating, table_count = _compare_column_pair(column)
            if table_rating is not None and table_count is not None:
                found["rating"] = table_rating
                found["review_count"] = table_count
            if found.get("years_experience") is None:
                table_years = _compare_column_years(column)
                if table_years is not None:
                    found["years_experience"] = table_years

    locations = _webmd_locations(text)
    if locations:
        found["locations"] = locations
    return found


def _parse_vitals(text: str) -> Dict[str, Any]:
    found: Dict[str, Any] = {}

    header = _VITALS_HEADER_PAIR.search(text)
    if header:
        _pairs(found, _float(header.group("rating")), _int(header.group("count")))

    heading_count = _VITALS_COUNT_HEADING.search(text)
    own_count = _int(heading_count.group("count")) if heading_count else None

    if found.get("rating") is None:
        # No header line — read the comparison table, but only if the doctor's
        # own ratings heading vouches for the column we picked.
        table_rating, table_count = _vitals_compare_pair(text)
        if table_rating is not None and own_count is not None and table_count == own_count:
            _pairs(found, table_rating, table_count)
        elif table_rating is not None:
            # The two anchors disagree (or one is missing), so the column may
            # belong to a neighbour. Publishing it would be a stranger's stars
            # on this provider's card.
            pass

    _pairs(found, None, own_count)

    heading_years = _VITALS_YEARS_HEADING.search(text)
    if heading_years:
        found["years_experience"] = _int(heading_years.group("years"))
    else:
        # The header line carries it inline: "14 Years of Experience 2201 W …".
        # Bounded to the text above the first level-2 heading so the comparison
        # table's "27 Years of Experience" (a neighbour's) can never win.
        body = _WEBMD_BODY_START.search(text)
        years = _YEARS.search(text[:body.start()] if body else text)
        if years:
            found["years_experience"] = _int(years.group("years"))

    crumb = _VITALS_BREADCRUMB.search(text)
    if crumb:
        found["breadcrumb_specialty"] = crumb.group("specialty").replace("-", " ").title()
        found["breadcrumb_city"] = (
            f'{crumb.group("city").replace("-", " ").title()}, {crumb.group("state").upper()}')
    return found


def _vitals_address(text: str) -> Optional[str]:
    match = _VITALS_ADDRESS.search(text)
    if not match:
        return None
    street = _VITALS_LEADING_TENURE.sub("", match.group("street")).strip()
    if not street or not street[0].isdigit():
        return None
    return f'{street}, {match.group("state")} {match.group("zip")}'


_PARSERS = (
    ("healthgrades.com", _parse_healthgrades),
    ("webmd.com", _parse_webmd),
    ("vitals.com", _parse_vitals),
)


def parse_profile(url: str, text: str) -> Dict[str, Any]:
    """Rating / review_count / years_experience / name / location from one profile.

    Returns {} when the domain is unrecognised or nothing at all was found.

    An empty result means "the parser could not read this page", NOT "the page
    has no data" — but so does a PARTIAL one, and that is the more common case:
    healthgrades template B legitimately has a rating and no count, and vitals
    keeps the count in a heading and the rating in a link. So the caller must
    fall back to LLM extraction PER FIELD, not per page. A per-page trigger
    ("the parser returned something, skip the model") silently dropped a
    provider's whole rating pair on any page that yielded only tenure.
    """
    lowered = str(url or "").lower()
    parser = next((fn for domain, fn in _PARSERS if domain in lowered), None)
    if parser is None:
        return {}

    text = strip_data_uris(text)
    found: Dict[str, Any] = {k: v for k, v in parser(text).items() if v is not None}

    # healthgrades template B states the UNROUNDED mean — "3.7727273 out of 5".
    # Every platform displays one decimal, the card prints this value directly,
    # and "3.7727273/5 weighted" reads as a bug. Rounded here rather than at the
    # call site so both the score and the display see the same number.
    if isinstance(found.get("rating"), float):
        found["rating"] = round(found["rating"], 1)

    name = _page_name(text)
    if name:
        found["page_provider_name"] = name

    address = _address(text) or (_vitals_address(text) if "vitals.com" in lowered else None)
    if not address:
        # The extract-tier rendering puts street and CITY,ST,ZIP on separate
        # lines, which the one-line pattern cannot span — but the Locations
        # section's first block carries the same address URL-decoded. Page
        # order, not nearest-anything: block 1 is the page's own primacy.
        locations = found.get("locations") or []
        if locations:
            address = locations[0].get("address")
    if address:
        found["location"] = address
    return found
