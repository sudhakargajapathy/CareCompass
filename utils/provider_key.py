"""Stable identity for a provider — shared by dedup and the enrichment cache.

One normalization serves both callers on purpose. `_name_token_overlap` in the
gatherer decides whether two extracted candidates are the same physician; the
cache key decides whether a provider we just discovered is one we already paid
to enrich. Those are the same question asked at two moments, and answering them
differently is what let "Hussam Seif-Eddeine, MD" and "Dr. Hussam Seif Eddeine,
MD" occupy ranks 1 and 3 of the same result set.
"""

import hashlib
import re
from typing import Optional, Set

from .geo import parse_location

# Honorifics and credentials carry no identity: two sources writing the same
# physician with different suffixes must still collide.
_STRIP_TOKENS = {
    "dr", "doctor", "md", "do", "dds", "dmd", "dpm", "phd", "np", "pa", "pac",
    "rn", "facp", "faan", "facs", "jr", "sr", "ii", "iii", "iv",
}

# Apostrophes and periods are ELIDED, not spaced.
#   - "O'Brien" -> "obrien", matching the common unpunctuated spelling.
#   - "M.D." -> "md", which the strip set then removes. Spacing it instead
#     leaves {m, d} — two junk tokens that no strip list catches and that make
#     "Andrea An, M.D." a different physician from "Andrea An, MD".
# Everything else becomes a space, so a hyphenated surname and its spaced
# spelling tokenize identically.
_ELIDE_RE = re.compile(r"[’'`.]")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Middle initials are deliberately KEPT. Dropping them would collide
# "Andrea M An" with "Andrea B An" — two different physicians sharing a first
# and last name — and a false cache hit attributes one doctor's reviews to
# another. A miss only costs a search; that asymmetry decides it.


def normalize_name_tokens(name: Optional[str]) -> Set[str]:
    """Identity tokens for a provider name.

    Punctuation collapses to whitespace BEFORE splitting. Without that step
    "Seif-Eddeine" is a single token while "Seif Eddeine" is two, their
    intersection is empty, and no overlap threshold can ever match them.
    """
    if not name:
        return set()

    cleaned = _ELIDE_RE.sub("", str(name))
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    tokens = {t for t in _WS_RE.split(cleaned.lower()) if t}
    return tokens - _STRIP_TOKENS


def normalized_name(name: Optional[str]) -> str:
    """Identity tokens as a stable, order-independent string."""
    return " ".join(sorted(normalize_name_tokens(name)))


# Tokens that mark a parsed "city" as actually being a street address. A city
# name contains no digits and no street-type word; `parse_location` splits on
# COMMAS, so an extractor that writes "…Ste 1 Chandler, AZ 85224" — no comma
# before the city — hands back the whole street address as the city.
#
# That is not a rare shape. On 2026-07-28 two live runs of the SAME search
# rendered the same physician's address both ways:
#
#   "2201 W Fairview St Ste 1 Chandler, AZ 85224"  -> "2201 w fairview st ste 1 chandler az"
#   "2201 W Fairview St Ste 1, Chandler, AZ 85224" -> "chandler az"
#
# Two keys, one doctor: 7 of 8 cache misses on a repeat search, and every run
# writes a fresh orphan row under a key no later read will ask for. Same
# failure class `resolve_cache_key` exists to prevent, one layer further down.
_STREET_TYPE_TOKENS = {
    "st", "street", "ave", "avenue", "blvd", "boulevard", "rd", "road",
    "ln", "lane", "dr", "drive", "way", "ct", "court", "pl", "place",
    "pkwy", "parkway", "hwy", "highway", "cir", "circle", "ter", "terrace",
    "ste", "suite", "unit", "apt", "fl", "floor", "bldg", "building", "#",
    "n", "s", "e", "w", "ne", "nw", "se", "sw",
}


def _strip_street_prefix(city: str) -> str:
    """Drop a street address that `parse_location` mistook for a city name.

    Keeps everything AFTER the last digit-run or street-type token, which is
    where the city name sits in "2201 W Fairview St Ste 1 Chandler". Returns the
    input unchanged when nothing survives — a real city we failed to isolate is
    better than an empty key that collides every provider in the pool onto one
    row.
    """
    tokens = city.split()
    last_marker = -1
    for i, token in enumerate(tokens):
        bare = token.strip(".,#").lower()
        if any(ch.isdigit() for ch in token) or bare in _STREET_TYPE_TOKENS:
            last_marker = i
    remainder = tokens[last_marker + 1:]
    return " ".join(remainder) if remainder else city


def normalized_place(location: Optional[str]) -> str:
    """City+state for keying, drawn from free-form location text.

    City rather than street address: a provider's reviews do not change with
    street precision, and the same physician's address is written a dozen ways
    across directory sites. State is included because Springfield is not a
    unique place.

    Falls back to normalized raw text when the location cannot be parsed, which
    keeps the key stable for that provider even though it will not collide with
    a differently-written form of the same place.
    """
    if not location:
        return ""

    parts = parse_location(location)
    city, state = parts.get("city"), parts.get("state")
    if city:
        city = _strip_street_prefix(_WS_RE.sub(" ", city.strip()))
    if city and state:
        return f"{city.lower()} {state.strip().lower()}"
    if city:
        return city.lower()

    cleaned = _PUNCT_RE.sub(" ", str(location).lower())
    return _WS_RE.sub(" ", cleaned).strip()


def provider_cache_key(name: Optional[str], location: Optional[str]) -> str:
    """Deterministic cache key for one physician in one city.

    Deliberately NOT derived from specialty or the search query, so the same
    physician found via a different phrasing still hits the cache.

    Deliberately NOT Python's `hash()`, which is salted per process — the
    previous ID scheme used it and therefore produced a different ID after
    every restart, which is why nothing could ever be looked up.
    """
    basis = f"{normalized_name(name)}|{normalized_place(location)}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


# Where a provider's key is pinned for the duration of one enrichment pass.
# Leading underscore: internal bookkeeping, not provider evidence.
CACHE_KEY_FIELD = "_cache_key"


def pin_cache_key(provider: dict) -> str:
    """Stamp and return a provider's cache key, before enrichment can move it."""
    key = provider_cache_key(provider.get("name"), provider.get("location"))
    provider[CACHE_KEY_FIELD] = key
    return key


def resolve_cache_key(provider: dict) -> str:
    """A provider's cache key, preferring one pinned before enrichment ran.

    Enrichment REWRITES `location` when a profile's street address gains ZIP
    precision the candidate pass lacked. Computing the key at write time then
    yields a different key than the read used — "Phoenix, AZ" going in,
    "Chandler, AZ 85224" coming out — so the row is stored where no future
    search will look for it. The lookup misses forever and a fresh orphan row
    accumulates every run, and it happens precisely to the providers
    enrichment helped most, since only a successful address backfill triggers
    it. Pin the key at the read and reuse it for the write.
    """
    return provider.get(CACHE_KEY_FIELD) or provider_cache_key(
        provider.get("name"), provider.get("location")
    )
