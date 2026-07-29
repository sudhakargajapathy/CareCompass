"""Offline US geography helpers: ZIP/city centroids and haversine distances.

Powers honest location scoring: real user-to-provider distances computed in
code from vendored data — never estimated by an LLM. Data lives in
data/us_zip_coords.csv.gz, derived from the GeoNames postal dataset
(https://www.geonames.org/, CC BY 4.0): one centroid per US ZIP.

Distances are straight-line (great-circle) between centroids — good to a
couple of miles, which is plenty for ranking; they are not driving distances.
"""

import csv
import gzip
import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "us_zip_coords.csv.gz"

# A US ZIP is the LAST element of an address line ("…, Phoenix, AZ 85004"),
# optionally followed by a country tag. The anchor is what makes this a ZIP
# matcher rather than a five-digit matcher: an unanchored \b(\d{5})\b reads the
# STREET NUMBER of "13640 N Plaza Del Rio Blvd, Peoria, AZ" as ZIP 13640 —
# Schenectady, NY. That provider then scored 2058 miles from a Phoenix user
# instead of 12, and because a ZIP hit is our HIGHEST precision the whole stack
# believed it: `resolution_level` said "zip", the scorer took the
# `computed_distance` branch (not `city_estimate`), zeroed the location
# dimension, and the card printed "2058.2 mi" as a measurement. Enrichment
# feeds street addresses straight into here (`_enrich_provider` backfills
# `location` from a profile's stated address), so this is the common path, not
# an edge case. When the anchor finds nothing we fall to city precision, which
# is the honest answer rather than a confident wrong one.
_ZIP_RE = re.compile(
    r"\b((\d{5})(?:-\d{4})?)"
    r"\s*(?:,\s*)?(?:usa|u\.s\.a\.|united states)?[\s.,;)]*$",
    re.IGNORECASE,
)

_EARTH_RADIUS_MILES = 3958.8

_STATE_NAMES_TO_CODES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN",
    "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "puerto rico": "PR", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA",
    "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY",
}
_STATE_CODES = set(_STATE_NAMES_TO_CODES.values())


@lru_cache(maxsize=1)
def _load_data() -> Tuple[
    Dict[str, Tuple[float, float]],
    Dict[Tuple[str, str], Tuple[float, float]],
    Dict[str, str],
    Dict[Tuple[str, str], int],
]:
    """Load once: (zip -> coords, (city,state) -> centroid, zip -> "City, ST",
    (city,state) -> ZIP count). The ZIP count is a population proxy used to
    keep micro-towns out of nearby-city expansion."""
    zip_coords: Dict[str, Tuple[float, float]] = {}
    zip_place: Dict[str, str] = {}
    city_sums: Dict[Tuple[str, str], Tuple[float, float, int]] = {}

    try:
        with gzip.open(_DATA_PATH, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(row for row in f if not row.startswith("#"))
            for row in reader:
                zip_code, city, state = row["zip"], row["city"], row["state"]
                try:
                    lat, lon = float(row["lat"]), float(row["lon"])
                except (TypeError, ValueError):
                    continue
                zip_coords[zip_code] = (lat, lon)
                zip_place[zip_code] = f"{city}, {state}"
                key = (city.lower(), state.upper())
                s_lat, s_lon, n = city_sums.get(key, (0.0, 0.0, 0))
                city_sums[key] = (s_lat + lat, s_lon + lon, n + 1)
    except OSError as e:
        logger.warning("ZIP coordinate data unavailable (%s); distances disabled", e)
        return {}, {}, {}, {}

    city_coords = {
        key: (s_lat / n, s_lon / n) for key, (s_lat, s_lon, n) in city_sums.items()
    }
    city_counts = {key: n for key, (_, _, n) in city_sums.items()}
    logger.info("Loaded %d ZIP centroids, %d cities", len(zip_coords), len(city_coords))
    return zip_coords, city_coords, zip_place, city_counts


def parse_location(text: Optional[str]) -> Dict[str, Optional[str]]:
    """Parse free-form US location text into {city, state, zip}.

    Handles "Phoenix, AZ", "123 Health St, Phoenix, AZ 85004", "85004",
    "Phoenix, Arizona". Missing parts come back as None.
    """
    parts: Dict[str, Optional[str]] = {"city": None, "state": None, "zip": None}
    if not text:
        return parts

    text = str(text)
    zip_match = _ZIP_RE.search(text)
    if zip_match:
        # group(2) is the bare 5 digits; the match span also swallows any "-1234"
        # extension and country tag, none of which belong in the city chunks.
        parts["zip"] = zip_match.group(2)
        text = text[: zip_match.start()] + text[zip_match.end():]

    chunks = [c.strip() for c in text.split(",") if c.strip()]
    if not chunks:
        return parts

    last = chunks[-1]
    if last.upper() in _STATE_CODES:
        parts["state"] = last.upper()
        chunks = chunks[:-1]
    elif last.lower() in _STATE_NAMES_TO_CODES:
        parts["state"] = _STATE_NAMES_TO_CODES[last.lower()]
        chunks = chunks[:-1]

    if chunks:
        # The chunk nearest the state is the city; earlier chunks are street
        parts["city"] = chunks[-1]
    return parts


def strip_zip(text: Optional[str]) -> str:
    """Location text with any ZIP removed — keeps web queries clean."""
    if not text:
        return ""
    stripped = _ZIP_RE.sub("", str(text))
    stripped = re.sub(r"\s{2,}", " ", stripped).strip(" ,")
    return stripped


def city_state_for_zip(text: Optional[str]) -> Optional[str]:
    """Resolve a location containing a ZIP to "City, ST" (None if unknown)."""
    parts = parse_location(text)
    if not parts["zip"]:
        return None
    return _load_data()[2].get(parts["zip"])


def _coords_for(parts: Dict[str, Optional[str]]) -> Optional[Tuple[float, float]]:
    zip_coords, city_coords, _, _ = _load_data()
    if parts["zip"] and parts["zip"] in zip_coords:
        return zip_coords[parts["zip"]]
    if parts["city"] and parts["state"]:
        return city_coords.get((parts["city"].lower(), parts["state"]))
    return None


def _haversine(coords_a: Tuple[float, float], coords_b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, coords_a)
    lat2, lon2 = map(math.radians, coords_b)
    d_lat, d_lon = lat2 - lat1, lon2 - lon1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    return 2 * _EARTH_RADIUS_MILES * math.asin(math.sqrt(a))


def nearby_cities(location: Optional[str], radius_miles: float, limit: int = 2, min_zip_count: int = 2) -> list:
    """Real cities within radius_miles of `location`, nearest first, as "City, ST".

    Powers adaptive ring expansion: when a one-city search returns a pool too
    thin to fill the research budget, discovery rings out to these. (The
    single-cluster trigger this also served was deleted in round 10.) Excludes the home
    city itself and micro-towns (fewer than min_zip_count ZIP rows — keeps
    Sun Lakes out and Gilbert/Mesa/Tempe in). Empty when the home location
    can't be resolved to coordinates.
    """
    home = parse_location(location)
    home_coords = _coords_for(home)
    if home_coords is None:
        return []

    _, city_coords, _, city_counts = _load_data()
    home_key = (home["city"].lower(), home["state"]) if home["city"] and home["state"] else None

    scored = []
    for (city, state), coords in city_coords.items():
        if (city, state) == home_key:
            continue
        if city_counts.get((city, state), 0) < min_zip_count:
            continue
        dist = _haversine(home_coords, coords)
        if 0 < dist <= radius_miles:
            scored.append((dist, f"{city.title()}, {state}"))

    scored.sort(key=lambda t: t[0])
    return [name for _, name in scored[:limit]]


def distance_miles(location_a: Optional[str], location_b: Optional[str]) -> Optional[float]:
    """Great-circle miles between two US locations, or None if unresolvable."""
    coords_a = _coords_for(parse_location(location_a))
    coords_b = _coords_for(parse_location(location_b))
    if coords_a is None or coords_b is None:
        return None
    return round(_haversine(coords_a, coords_b), 1)


def resolution_level(location: Optional[str]) -> Optional[str]:
    """How precisely we can place a location: "zip" (a ZIP we have coords
    for), "city" (only a city+state centroid), or None (unresolvable).

    Mirrors the precedence in `_coords_for`, so a caller can tell a real ZIP
    coordinate from a city-centroid approximation and avoid treating a
    same-city centroid distance (often ~0 mi) as a precise measurement.
    """
    parts = parse_location(location)
    zip_coords, city_coords, _, _ = _load_data()
    if parts["zip"] and parts["zip"] in zip_coords:
        return "zip"
    if parts["city"] and parts["state"] and (parts["city"].lower(), parts["state"]) in city_coords:
        return "city"
    return None


def location_tier(location_a: Optional[str], location_b: Optional[str]) -> str:
    """Fallback textual comparison when no distance can be computed.

    Returns "same_zip" | "same_city" | "same_state" | "different" | "unknown".
    same_zip only matters when the ZIPs match but aren't in the dataset —
    resolvable equal ZIPs already yield a ~0-mile computed distance.
    """
    a = parse_location(location_a)
    b = parse_location(location_b)

    if a["zip"] and b["zip"] and a["zip"] == b["zip"]:
        return "same_zip"

    states_known = a["state"] and b["state"]
    states_conflict = states_known and a["state"] != b["state"]

    if a["city"] and b["city"] and a["city"].lower() == b["city"].lower():
        return "different" if states_conflict else "same_city"
    if states_known:
        return "same_state" if a["state"] == b["state"] else "different"
    return "unknown"
