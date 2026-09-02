"""Deterministic parsers for review-platform DIRECTORY pages.

One directory page names dozens of providers with a rating, a review count, a
tenure and an address. Reading that with an LLM over a 3,000-char anchored
excerpt recovered a handful of them and did it differently on each run: on one
live search, 4 of 8 fetched pages yielded nothing and a provider was marked
"no profile found" while 81,000 characters of her pages sat in memory. The page
text is highly structured and repetitive, which is what a regex is for.

Each platform is parsed by splitting the page into per-provider BLOCKS at the
entry heading, then reading fields out of each block. One mega-regex per
platform was tried first and is much harder to extend — a block that lacks an
address should still yield a rating.

Everything here is measured against real page text. The formats differ per
platform in ways that are not guessable:

    healthgrades  ### [Dr. Hemant Pandey, MD](…/physician/dr-hemant-pandey-xsjwm)
                  Rated 3.8 out of 5 3.8 from 88 ratings Neurology
                  [4045 W Chandler Blvd Bldg F Chandler, AZ 85226 4.1 mi miles away](…)

    webmd         ## [Dr. Paul Ryan Macdonald](…/doctor/paul-ryan-macdonald-…-overview)
                  Neurology
                  [3.0 13 Ratings](…#ratings)  28 Years Exp erience
                  655 S Dobson Rd Ste 103 Bldg A, Chandler, AZ, 85224 2.03 miles

    vitals        ### Dr. Brandon Craig Woods          (heading sometimes bare,
                  Neurology                             sometimes a markdown link)
                  3.5 12 ratings                        (or "[0 ratings](…)" — a
                  18 years exp                           LINKED form carrying only
                  Mesa, AZ9.8 mi                         the count)

An earlier sweep concluded webmd listings state no rating and vitals listings
state only a city-wide average. Both were wrong: the sweep matched only
healthgrades' "Rated X out of 5 … from N ratings" phrasing, so webmd's
"[3.0 13 Ratings]" and vitals' bare "3.5 12 ratings" scored zero. A parser
written per platform is the fix for that class of error, not a wider regex.
"""
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from utils.provider_key import normalize_name_tokens

# Sponsored inventory, and it is not subtle: the same handful of entries came
# back on pages for unrelated cities, and one of them is "Minder Memory
# Center — San Diego, CA — Psychology, Neuropsychology" sitting on a Queen
# Creek, AZ neurology page. webmd and vitals are both Internet Brands
# properties and share the inventory, so the SAME advertiser can appear on two
# platforms — which would satisfy the cross-platform blend's "two platforms
# agree" condition with one paid placement. That is a manufactured
# independence signal, worse than a wrong number.
#
# `## Featured Results` is a literal level-2 heading in the page text, and
# entry headings are level 3 (or level 2 with a link on webmd), so the region
# ends at the next level-2 heading that is NOT itself an entry.
_FEATURED_HEADING = re.compile(r"^##\s+Featured Results\s*$", re.M)
_LEVEL2_HEADING = re.compile(r"^##\s+(?!\[)", re.M)

# The page's own intro states a CITY-WIDE aggregate: "We found **48
# Neurologists** available in **Queen Creek, AZ** with an average **4 stars**
# across **897 reviews**". Bound to a provider that is their city's mean at a
# 897-review weight, which would dominate the Bayesian blend. It sits above
# the featured block, so starting after the featured region skips it too — but
# the guard is explicit because that ordering is the page's choice, not ours.
_CITY_AGGREGATE = re.compile(
    r"We found\s+\*{0,2}\d+[^\n]{0,120}?average[^\n]{0,60}?reviews?\*{0,2}", re.I)

# Credentials that are not a physician. The target is doctors; a listing mixes
# in PAs, nurse practitioners, midwives and psychologists, and the featured
# block is where most of them arrive.
_NON_PHYSICIAN = re.compile(
    r"\b(?:PA-C|PA|APRN|CNM|MSPA|NP|RN|LPN|CRNA|PsyD|PhD|LCSW|LPC|DDS|DMD|"
    r"DPM|DC|OD|RD|LMT|MSN|DNP)\b", re.I)
# A physician says so: a "Dr." prefix, or an MD/DO/MBBS credential.
_PHYSICIAN = re.compile(r"^\s*Dr\.|\b(?:MD|DO|MBBS|MBChB)\b", re.I)

# Trailing "N.NN miles" / "N mi miles away" / "AZ9.8 mi" on an address.
#
# This must be stripped, and not merely ignored, for two independent reasons.
# First, `utils.geo` anchors its ZIP pattern to the END of the address — that
# anchoring exists because an unanchored pattern read a five-digit STREET
# NUMBER as a ZIP and reported a Peoria AZ address as 2058 miles away at "zip"
# precision. Leaving "85224 2.03 miles" in place puts the ZIP mid-string and
# silently degrades every provider to city precision. Second, the mileage is
# measured from the PAGE's city, not the user's: a Queen Creek page states
# "Mesa, AZ9.8 mi" for a doctor whose distance from the searching user is
# something else entirely. A parsed distance from the wrong origin reads as
# measured data, which is the worst kind of wrong.
_TRAILING_MILES = re.compile(
    r"\s*\d+(?:\.\d+)?\s*(?:mi|miles)(?:\s+miles)?(?:\s+away)?\s*$", re.I)
# vitals glues the mileage straight onto the state: "Mesa, AZ9.8 mi".
_GLUED_MILES = re.compile(r"([A-Z]{2})\d+(?:\.\d+)?\s*mi\b")

# The href may be absolute OR host-relative. healthgrades writes page 1 of a
# city directory with absolute profile links and pages 2+ (and the state-wide
# directory) with `/physician/dr-kan-yu-2b5bc` — measured 2026-09-02 on
# `neurology-directory/az-arizona/chandler_2` (20 entries, all relative) and
# `neurology-directory/az-arizona` (584 results, all relative). Requiring the
# scheme made every entry on those pages invisible: 0 rows from a 25 KB body.
# `parse_listing` resolves the relative form against the page URL.
_HG_HEADING = re.compile(
    r"#{2,3}\s*\[(?P<name>[^\]\n]+)\]\((?P<url>(?:https?://[^)\s/]+)?/physician/[^)\s#]+)[^)]*\)")
_WEBMD_HEADING = re.compile(
    r"#{2,3}\s*\[(?P<name>[^\]\n]+)\]\((?P<url>https?://[^)\s]*?/doctor/[^)\s#]+)[^)]*\)")
_VITALS_HEADING = re.compile(
    r"^#{3}\s*(?:\[(?P<lname>[^\]\n]+)\]\((?P<url>[^)\s]*?/doctors/[^)\s#]+)[^)]*\)"
    r"|(?P<bname>(?!\[)[^\n#][^\n]*?))\s*$",
    re.M)

# `Rated 3.8 out of 5 3.8 from 88 ratings` on a city page — and
# `Rated 4.7 out of 54.7from 48 ratings` on the state directory (2026-09-02),
# where the echoed number is glued to both neighbours. The whitespace between
# "5", the echo and "from" is therefore OPTIONAL: requiring it read 20 stated
# pairs on the state page as zero.
_HG_PAIR = re.compile(
    r"Rated\s+(?P<rating>\d+(?:\.\d+)?)\s+out of 5\s*\d+(?:\.\d+)?\s*"
    r"from\s+(?P<count>\d+)\s+ratings?", re.I)
# State-directory entries state specialty and address on their own lines
# rather than in the city page's `ratings <Specialty> [<address>]` tail.
_HG_SPECIALTY_LINE = re.compile(r"^Specialty:\s*(?P<specialty>[^\n]+?)\s*$", re.M)
_HG_ADDRESS_LINK = re.compile(
    r"\[(?P<address>\d[^\]\n]{6,90}?\d{5})\]\((?:https?://[^)\s/]+)?/physician/")
# `We found 81 results within 10 miles for "Neurologists near Chandler, AZ"` —
# the page's own count of a directory that lists 20 per page.
_HG_RESULT_COUNT = re.compile(r"We found\s*(?P<count>\d[\d,]*)\s*results", re.I)
_HG_TAIL = re.compile(
    r"ratings?\s+(?P<specialty>[A-Z][A-Za-z /&'-]{2,40}?)\s*\[(?P<address>[^\]\n]+)\]", re.I)

# "[3.0 13 Ratings](…#ratings)" — rating and count inside the link TEXT.
_WEBMD_PAIR = re.compile(r"\[\s*(?P<rating>\d+(?:\.\d+)?)\s+(?P<count>\d+)\s+Ratings?\s*\]", re.I)
# "28 Years Exp erience" — the space inside the word is real page text, not a
# transcription slip. A pattern requiring "Experience" matches nothing.
_WEBMD_YEARS = re.compile(r"(?P<years>\d+)\s+Years?\s+Exp\s*erience", re.I)
# Ends AT the ZIP, and excludes URL punctuation.
#
# A permissive `[^\n\]]` body matched from a digit inside the ratings link's
# own URL hash ("…0f97dda8f-overview#ratings)  28 Years Exp erience 655 S
# Dobson Rd…") straight through to the ZIP, producing an "address" that began
# mid-URL. Excluding ()#:/ confines the match to prose, and stopping at the ZIP
# means the trailing " 2.03 miles" is never captured in the first place rather
# than captured and then stripped.
_WEBMD_ADDRESS = re.compile(
    r"(?P<address>\d[^\n\]()#:/]{6,90}?,\s*[A-Z]{2},?\s*\d{5})", re.I)
# webmd punctuates as "Chandler, AZ, 85224" — a comma before the ZIP.
# `utils.geo.parse_location` splits on commas, so that extra comma makes the
# state its own field and pushes the ZIP into a fourth one. Normalised to the
# "City, ST ZIP" shape every other source uses.
_STATE_ZIP_COMMA = re.compile(r",\s*([A-Z]{2}),\s*(\d{5})\b")

# vitals states the pair two ways. The plain form carries a rating; the linked
# form carries only a count, and is what a 0-rating provider gets.
_VITALS_PLAIN_PAIR = re.compile(
    r"(?<![\d.])(?P<rating>\d(?:\.\d)?)\s+(?P<count>\d+)\s+ratings?\b", re.I)
_VITALS_LINKED_COUNT = re.compile(r"\[\s*(?P<count>\d+)\s+ratings?\s*\]", re.I)
_VITALS_YEARS = re.compile(r"(?P<years>\d+)\s+years?\s+exp\b", re.I)
# A vitals heading is sometimes a markdown link and sometimes BARE
# ("### Dr. Brandon Craig Woods"), and the bare form yields no profile URL. The
# same URL is often repeated further down the block as a "View Profile" call to
# action, so that is a second place to look — not a new source of data.
_VITALS_VIEW_PROFILE = re.compile(
    r"\[\s*View Profile\s*\]\((?P<url>[^)\s]*?/doctors/[^)\s#]+)[^)]*\)", re.I)
# How much of the NAME must appear in the slug for a recovered link to be
# accepted. Divided by the name's own token count, not the smaller set: a slug
# carries a random id ("brandon-woods-x8k2") that no name will ever match, so
# dividing by the smaller set would penalise every correct pairing.
_SLUG_NAME_AGREEMENT = 0.5
_VITALS_CITY = re.compile(r"^(?P<city>[A-Z][A-Za-z .'-]{2,40},\s*[A-Z]{2})", re.M)

_SPECIALTY_LINE = re.compile(r"^(?P<specialty>[A-Z][A-Za-z /&'-]{2,40})\s*$", re.M)


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


def strip_featured(text: str) -> str:
    """Drop the sponsored block, and the city-wide aggregate above it.

    Returns the input unchanged when there is no featured block — a page
    without ads must not lose its first entries to a defensive slice.
    """
    text = str(text or "")
    match = _FEATURED_HEADING.search(text)
    if not match:
        return _CITY_AGGREGATE.sub(" ", text)
    end = _LEVEL2_HEADING.search(text, match.end())
    remainder = text[end.start():] if end else ""
    return _CITY_AGGREGATE.sub(" ", text[:match.start()] + "\n" + remainder)


def is_physician(name: str) -> bool:
    """A doctor, not a PA / NP / midwife / psychologist / facility.

    "Minder Memory Center" has no credential and no title, so it fails the
    physician test; "Jeannine Thomas, MSPA, PA-C" carries one that disqualifies
    her. Both arrived through the featured block on a neurology search, so this
    is a second net behind `strip_featured` rather than the only one.
    """
    name = str(name or "").strip()
    if not name or _NON_PHYSICIAN.search(name):
        return False
    return bool(_PHYSICIAN.search(name))


def clean_address(raw: str) -> str:
    """Address with the listing's own mileage removed, punctuated for `utils.geo`.

    See `_TRAILING_MILES` for why the mileage must go, and `_STATE_ZIP_COMMA`
    for why "AZ, 85224" becomes "AZ 85224".
    """
    address = _GLUED_MILES.sub(r"\1", str(raw or "").strip())
    address = _TRAILING_MILES.sub("", address).strip()
    address = _STATE_ZIP_COMMA.sub(r", \1 \2", address)
    return re.sub(r"[\s,]+$", "", address)


def _absolute_url(page_url: str, href: Optional[str]) -> Optional[str]:
    """Resolve a listing's profile link against the page it was found on.

    vitals writes its entry links host-relative — "/doctors/ramzy-medaa-clje80"
    — while healthgrades and webmd write them absolute. A relative href is
    usable for comparing two vitals rows against each other and useless for
    anything else: it cannot be fetched, and it carries no domain, so a
    cross-platform identity check keyed on the profile URL would see
    "/doctors/x" and "https://www.healthgrades.com/physician/y" as two strings
    with nothing in common — which is true, but only because one of them was
    never finished.

    `urljoin` leaves an already-absolute URL byte-identical, so this is a no-op
    on the two platforms that do not need it. A missing or unusable page URL
    degrades to returning the href as-is rather than inventing a host.
    """
    if not href:
        return None
    if not page_url:
        return href
    try:
        return urljoin(page_url, href)
    except ValueError:
        return href


def _blocks(text: str, heading: re.Pattern) -> List[Dict[str, Any]]:
    """(name, profile_url, body) per entry, body running to the next heading."""
    found = list(heading.finditer(text))
    out = []
    for index, match in enumerate(found):
        groups = match.groupdict()
        name = (groups.get("name") or groups.get("lname") or groups.get("bname") or "").strip()
        stop = found[index + 1].start() if index + 1 < len(found) else len(text)
        out.append({
            "name": name,
            "profile_url": (groups.get("url") or "").strip() or None,
            "body": text[match.end():stop],
        })
    return out


# webmd states its total twice on one page, differently: "Chandler, AZ has
# **142 Neurologist** results …" (bold markers between the number and the
# word) and, lower down, "85 providers"; vitals states "142 Neurologists in
# Chandler, AZ" and a smaller "85 Neurologists" further down. Measured
# 2026-09-02. The LARGEST stated total plans the pages: over-planning by one
# page costs a fifth of a credit and the URL-first dedupe absorbs any repeat,
# while under-planning silently drops doctors — the asymmetry decides.
_WEBMD_RESULT_COUNTS = (
    re.compile(r"(?P<count>\d[\d,]*)\s+\**[A-Za-z]+\**\s+results\b", re.I),
    re.compile(r"(?P<count>\d[\d,]*)\s+providers\b", re.I),
)
_VITALS_RESULT_COUNT = re.compile(
    r"(?P<count>\d[\d,]*)\s+(?:[A-Z][A-Za-z]+\s+){0,2}[A-Z][a-z]+(?:ists|ians|ers|ors|eons)\b")


def healthgrades_result_count(text: str) -> Optional[int]:
    """The directory's own result total, or None when the page states none."""
    match = _HG_RESULT_COUNT.search(text or "")
    if not match:
        return None
    return _int(match.group("count").replace(",", ""))


def listing_result_count(url: str, text: str) -> Optional[int]:
    """A city listing's stated result total, per platform, or None.

    Drives pagination (utils.platform_urls.listing_page_urls): every platform
    serves ~20-50 entries per page and the rest behind its own page
    parameter, so page 1 alone read 20 of healthgrades' 81, 46 of webmd's
    142 and 45 of vitals' 142 for Chandler neurology (2026-09-02).
    """
    lowered = str(url or "").lower()
    body = text or ""
    if "healthgrades.com" in lowered:
        return healthgrades_result_count(body)
    if "webmd.com" in lowered:
        patterns = _WEBMD_RESULT_COUNTS
    elif "vitals.com" in lowered:
        patterns = (_VITALS_RESULT_COUNT,)
    else:
        return None
    counts = [
        _int(match.group("count").replace(",", ""))
        for pattern in patterns for match in pattern.finditer(body)
    ]
    counts = [c for c in counts if c]
    return max(counts) if counts else None


def _parse_healthgrades(text: str) -> List[Dict[str, Any]]:
    rows = []
    for block in _blocks(text, _HG_HEADING):
        body = block["body"]
        pair = _HG_PAIR.search(body)
        tail = _HG_TAIL.search(body)
        specialty_line = _HG_SPECIALTY_LINE.search(body)
        address_link = _HG_ADDRESS_LINK.search(body)
        # A block with NO pair is still a row. healthgrades renders pages 2+
        # of a city directory as heading + photo only — no rating, no
        # address — and dropping those entries dropped the doctor entirely:
        # Dr. Kan Yu sat on `chandler_2` while his card showed no
        # healthgrades page at all (2026-09-02). A name plus the platform's
        # own profile link is exactly what extract-mode enrichment fetches;
        # the rating then comes off the profile, as it does for every webmd
        # row (49 rows, 0 pairs on the Chandler page).
        rows.append({
            "name": block["name"],
            "profile_url": block["profile_url"],
            "rating": _float(pair.group("rating")) if pair else None,
            "review_count": _int(pair.group("count")) if pair else None,
            "specialty": (
                tail.group("specialty").strip() if tail
                else specialty_line.group("specialty").strip() if specialty_line
                else None
            ),
            "location": (
                clean_address(tail.group("address")) if tail
                else clean_address(address_link.group("address")) if address_link
                else None
            ),
            "years_experience": None,
        })
    return rows


def _parse_webmd(text: str) -> List[Dict[str, Any]]:
    rows = []
    for block in _blocks(text, _WEBMD_HEADING):
        body = block["body"]
        pair = _WEBMD_PAIR.search(body)
        years = _WEBMD_YEARS.search(body)
        specialty = _SPECIALTY_LINE.search(body.strip())
        # webmd runs the tenure straight into the street address on one line
        # ("…#ratings)  28 Years Exp erience 655 S Dobson Rd…"), and the address
        # pattern starts at a digit, so it began at the "28". `geo` still
        # resolved the city because it reads the comma-separated tail — but the
        # string also renders on the card as the provider's address. Blanking
        # the tenure clause first is cheaper than teaching the address pattern
        # which digits to distrust.
        address = _WEBMD_ADDRESS.search(
            _WEBMD_YEARS.sub(" ", body) if years else body
        )
        rows.append({
            "name": block["name"],
            "profile_url": block["profile_url"],
            "rating": _float(pair.group("rating")) if pair else None,
            "review_count": _int(pair.group("count")) if pair else None,
            "specialty": specialty.group("specialty").strip() if specialty else None,
            "location": clean_address(address.group("address")) if address else None,
            "years_experience": _int(years.group("years")) if years else None,
        })
    return rows


def slug_agrees_with_name(url: Optional[str], name: str) -> bool:
    """Does a profile URL's slug carry the provider's name?

    Every platform puts the name in the slug, with a random id appended:

        /physician/dr-hemant-pandey-xsjwm              -> hemant · pandey · …
        /doctors/ramzy-medaa-clje80                    -> ramzy · medaa · …
        /doctor/paul-ryan-macdonald-5612ad17-…-overview -> paul · ryan · macdonald · …

    So a URL can be checked against the name it is about to be attached to,
    which turns "this link was near that heading" or "this page came back for
    that query" from a guess into a checked guess.

    The shared count is divided by the NAME's tokens, not the smaller set,
    because a slug always carries an id no name will ever match — dividing by
    the smaller set would penalise every correct pairing. Tokenizing goes
    through `normalize_name_tokens`, the same helper the enrichment cache keys
    on, so identity asked in different places is answered the same way.
    """
    if not url or not name:
        return False
    name_tokens = normalize_name_tokens(name)
    if not name_tokens:
        return False
    shared = name_tokens & _slug_tokens(url)
    return len(shared) / len(name_tokens) >= _SLUG_NAME_AGREEMENT


# Words a platform puts in a profile slug that are not part of anyone's name.
# Without this, "/doctor/12345-overview" reduces to {"overview"} and would look
# like a slug that names a person who is not ours.
_SLUG_FURNITURE = frozenset({
    "overview", "profile", "profiles", "doctor", "doctors", "physician",
    "physicians", "provider", "providers", "reviews", "ratings", "md", "do",
})


def _slug_tokens(url: Optional[str]) -> set:
    """Name-like tokens in a URL's last path segment.

    The file extension is stripped BEFORE tokenizing. `normalize_name_tokens`
    removes the dot rather than splitting on it, so "Dr_Khan.html" tokenized to
    the single token "khanhtml" — matching nothing, on a URL that plainly
    carries the surname.
    """
    slug = str(url or "").rstrip("/").rsplit("/", 1)[-1].split("?")[0].split("#")[0]
    slug = re.sub(r"\.(?:html?|php|aspx?|jsp)$", "", slug, flags=re.I)
    return normalize_name_tokens(slug.replace("-", " ").replace("_", " "))


def slug_contradicts_name(url: Optional[str], name: str) -> bool:
    """Does a profile URL's slug name a DIFFERENT person than this one?

    The veto half of `slug_agrees_with_name`, and deliberately not its negation.
    Agreement asks "is this URL evidence FOR the pairing" and divides by the
    name's tokens, so a slug carrying only a surname ("/doctors/Dr_Khan")
    scores 0.33 against a three-token name and fails — correctly, as evidence.
    As grounds for THROWING AWAY an observation that would be wrong: sharing a
    surname is not a contradiction.

    So this returns True only on positive disagreement — the slug carries
    name-like tokens and NONE of them is in the name:

        /doctor/luis-tumialan-…      vs  Dr. Frederick Francis Marciano   True
        /doctors/Dr_Khan.html        vs  Dr. Mohammad B. Khan             False
        /doctors/12345               vs  anyone                           False

    An opaque slug cannot contradict anything. A slug only gets a vote when it
    carries at least TWO name-like tokens — a real person-slug has a given name
    and a family name — because one token is exactly the ambiguous case: a bare
    "/doctors/Dr_Khan" agrees with our Khan, and a bare "/doctor/12345-overview"
    agrees with nobody while naming nobody either. Requiring two is what keeps
    an id-only or truncated slug from vetoing a real observation.
    """
    if not url or not name:
        return False
    name_tokens = normalize_name_tokens(name)
    personal = {
        token for token in _slug_tokens(url)
        if len(token) >= 3 and token.isalpha() and token not in _SLUG_FURNITURE
    }
    if not name_tokens or len(personal) < 2:
        return False
    return not (name_tokens & personal)


def _recover_vitals_profile_url(body: str, name: str) -> Optional[str]:
    """A bare heading's profile URL, taken from the block's "View Profile" link.

    Checked against the heading it is about to be attached to, because vitals
    puts the provider's NAME IN THE SLUG:

        /doctors/brandon-woods-x8k2   ->  brandon · woods · x8k2

    A block runs from one heading to the next, so a trailing link belongs to the
    provider above it — but "belongs to the one above it" is exactly the
    reasoning behind the positional verdict fallback that let a provider collect
    its neighbour's penalty. The slug check turns the heuristic into a checked
    one at no cost, and it uses the same tokenizer the enrichment cache keys on,
    so identity asked in two places is answered the same way.

    Guards against grabbing the NEIGHBOUR's link. It does not distinguish two
    adjacent providers who share a surname, which would need the surname
    identified rather than the token set compared.
    """
    match = _VITALS_VIEW_PROFILE.search(body)
    if not match:
        return None
    name_tokens = normalize_name_tokens(name)
    if not name_tokens:
        return None
    if not slug_agrees_with_name(match.group("url"), name):
        return None
    return match.group("url")


def _parse_vitals(text: str) -> List[Dict[str, Any]]:
    rows = []
    for block in _blocks(text, _VITALS_HEADING):
        body = block["body"]
        if not block["profile_url"]:
            block["profile_url"] = _recover_vitals_profile_url(body, block["name"])
        plain = _VITALS_PLAIN_PAIR.search(body)
        linked = _VITALS_LINKED_COUNT.search(body)
        years = _VITALS_YEARS.search(body)
        city = _VITALS_CITY.search(body.strip())
        # The specialty line is load-bearing here, not decoration: a vitals
        # NEUROLOGY directory lists "Dr. Edgardo D Zavala-Alarcon, MD — Plastic
        # Surgery" among the results, and without the label the caller has
        # nothing to filter on.
        specialty = _SPECIALTY_LINE.search(body.strip())
        rating = _float(plain.group("rating")) if plain else None
        # The linked form states a count with NO rating (a provider with zero
        # ratings gets it). Recording the count alone is honest and useless to
        # the blend, which needs pairs — but it keeps the roster entry, and the
        # provider can still be enriched from their profile page.
        count = _int(plain.group("count")) if plain else (
            _int(linked.group("count")) if linked else None)
        rows.append({
            "name": block["name"],
            "profile_url": block["profile_url"],
            "rating": rating,
            "review_count": count,
            "specialty": specialty.group("specialty").strip() if specialty else None,
            "location": clean_address(city.group("city")) if city else None,
            "years_experience": _int(years.group("years")) if years else None,
        })
    return rows


_PARSERS = (
    ("healthgrades.com", _parse_healthgrades),
    ("webmd.com", _parse_webmd),
    ("vitals.com", _parse_vitals),
)


def parse_listing(url: str, text: str) -> List[Dict[str, Any]]:
    """Providers named on one directory page, or [] when the domain is unknown.

    Dispatch is on the DOMAIN, not on `url_page_kind`: that classifier calls
    every vitals city page ("/neurology/az/mesa") "unknown", so keying off it
    would silently parse nothing for a platform whose pages parse fine.

    Never returns a distance. The page states one and it is measured from the
    page's own city; `utils.geo` computes distance from the user's location.
    """
    lowered = str(url or "").lower()
    parser = next((fn for domain, fn in _PARSERS if domain in lowered), None)
    if parser is None:
        return []

    body = strip_featured(text)
    rows = []
    for row in parser(body):
        if not is_physician(row["name"]):
            continue
        row["review_source_url"] = url
        row["profile_url"] = _absolute_url(url, row.get("profile_url"))
        rows.append(row)
    return rows
