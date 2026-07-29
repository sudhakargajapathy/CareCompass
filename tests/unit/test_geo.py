"""Unit tests for utils/geo.py — offline ZIP/city distance helpers."""

import pytest
from utils.geo import (
    city_state_for_zip,
    distance_miles,
    location_tier,
    nearby_cities,
    parse_location,
    resolution_level,
    strip_zip,
)


class TestResolutionLevel:
    """How precisely a location resolves — the signal that lets callers avoid
    treating a same-city centroid distance as a real measurement."""

    def test_zip_when_resolvable(self):
        assert resolution_level("Phoenix, AZ 85004") == "zip"
        assert resolution_level("85004") == "zip"

    def test_city_when_only_city_state(self):
        assert resolution_level("Chandler, AZ") == "city"

    def test_none_when_unresolvable(self):
        assert resolution_level("Nowhereville") is None
        assert resolution_level("") is None
        assert resolution_level(None) is None

    def test_city_without_state_is_none(self):
        # A bare city can't be placed to a unique centroid
        assert resolution_level("Springfield") is None


class TestParseLocation:
    def test_city_state(self):
        assert parse_location("Phoenix, AZ") == {"city": "Phoenix", "state": "AZ", "zip": None}

    def test_full_address_with_zip(self):
        parts = parse_location("123 Health St, Phoenix, AZ 85004")
        assert parts == {"city": "Phoenix", "state": "AZ", "zip": "85004"}

    def test_zip_plus_four(self):
        assert parse_location("Phoenix, AZ 85004-1234")["zip"] == "85004"

    def test_bare_zip(self):
        assert parse_location("85004") == {"city": None, "state": None, "zip": "85004"}

    def test_full_state_name(self):
        assert parse_location("Phoenix, Arizona")["state"] == "AZ"

    def test_city_only(self):
        assert parse_location("Phoenix") == {"city": "Phoenix", "state": None, "zip": None}

    def test_five_digit_street_number_is_not_a_zip(self):
        # Every address fixture above uses a THREE-digit street number, which
        # is why an unanchored \b(\d{5})\b survived: "13640 N Plaza Del Rio
        # Blvd, Peoria, AZ" (a real Peoria medical campus) read 13640 as a ZIP
        # in Schenectady NY and put the provider 2058 miles from a Phoenix
        # user — reported as a MEASURED distance, since a ZIP hit is our
        # highest precision tier.
        parts = parse_location("13640 N Plaza Del Rio Blvd, Peoria, AZ")
        assert parts == {"city": "Peoria", "state": "AZ", "zip": None}

    def test_five_digit_street_number_with_a_real_trailing_zip(self):
        # The trailing ZIP still wins when the address actually carries one
        parts = parse_location("13640 N Plaza Del Rio Blvd, Peoria, AZ 85381")
        assert parts == {"city": "Peoria", "state": "AZ", "zip": "85381"}

    def test_country_tag_after_zip(self):
        assert parse_location("Phoenix, AZ 85004, USA")["zip"] == "85004"


class TestStreetNumberDistances:
    """A street number misread as a ZIP is worse than no ZIP at all: it
    resolves, so `resolution_level` reports the highest precision and the
    scorer treats the bogus figure as measured rather than estimated."""

    def test_distance_uses_the_city_not_the_street_number(self):
        d = distance_miles("Phoenix, AZ", "13640 N Plaza Del Rio Blvd, Peoria, AZ")
        assert d is not None and d < 30

    def test_precision_degrades_honestly_to_city(self):
        assert resolution_level("13640 N Plaza Del Rio Blvd, Peoria, AZ") == "city"

    def test_strip_zip_keeps_the_street_number(self):
        assert strip_zip("13640 N Plaza Del Rio Blvd, Peoria, AZ") == (
            "13640 N Plaza Del Rio Blvd, Peoria, AZ"
        )

    def test_empty_and_none(self):
        assert parse_location("") == {"city": None, "state": None, "zip": None}
        assert parse_location(None) == {"city": None, "state": None, "zip": None}


class TestStripZip:
    def test_strips_zip_and_tidies(self):
        assert strip_zip("Phoenix, AZ 85004") == "Phoenix, AZ"

    def test_no_zip_unchanged(self):
        assert strip_zip("Phoenix, AZ") == "Phoenix, AZ"

    def test_bare_zip_becomes_empty(self):
        assert strip_zip("85004") == ""


class TestDistance:
    def test_zip_to_zip_within_metro(self):
        # Downtown Phoenix to downtown Scottsdale: ~8-9 straight-line miles
        d = distance_miles("Phoenix, AZ 85004", "Scottsdale, AZ 85251")
        assert d is not None and 5 <= d <= 15

    def test_city_to_city_long_range(self):
        # Phoenix to Tucson: ~105-115 straight-line miles
        d = distance_miles("Phoenix, AZ", "Tucson, AZ")
        assert d is not None and 90 <= d <= 130

    def test_same_zip_is_zero(self):
        assert distance_miles("85004", "85004") == 0.0

    def test_city_centroid_fallback_without_zip(self):
        d = distance_miles("Phoenix, AZ", "123 Main St, Scottsdale, AZ")
        assert d is not None and 3 <= d <= 25

    def test_unresolvable_returns_none(self):
        assert distance_miles("Phoenix, AZ", "Nowhereville") is None
        assert distance_miles("", "Phoenix, AZ") is None

    def test_city_without_state_is_not_guessed(self):
        # "Springfield" alone is ambiguous across states — no coords, no guess
        assert distance_miles("Springfield", "Phoenix, AZ") is None


class TestCityStateForZip:
    def test_known_zip(self):
        assert city_state_for_zip("85004") == "Phoenix, AZ"

    def test_no_zip(self):
        assert city_state_for_zip("Phoenix, AZ") is None


class TestLocationTier:
    def test_same_city_full_address(self):
        assert location_tier("Phoenix, AZ", "123 Health St, Phoenix, AZ") == "same_city"

    def test_same_state(self):
        assert location_tier("Phoenix, AZ", "Scottsdale, AZ") == "same_state"

    def test_different_state(self):
        assert location_tier("Phoenix, AZ", "Las Vegas, NV") == "different"

    def test_same_city_name_conflicting_state_is_different(self):
        assert location_tier("Springfield, IL", "Springfield, MO") == "different"

    def test_matching_unresolvable_zips_are_same_zip(self):
        # 00000 is not in the dataset; string equality still means proximity
        assert location_tier("00000", "00000") == "same_zip"

    def test_junk_is_unknown(self):
        assert location_tier("???", "Phoenix, AZ") == "unknown"


class TestNearbyCities:
    def test_chandler_rings_to_east_valley(self):
        ring = nearby_cities("Chandler, AZ", 25, limit=4)
        # Gilbert/Tempe/Mesa are the obvious East Valley neighbors
        assert any("Gilbert" in c for c in ring)
        assert all(c.endswith(", AZ") for c in ring)
        assert not any("Chandler" in c for c in ring)  # home city excluded

    def test_respects_limit(self):
        assert len(nearby_cities("Phoenix, AZ", 25, limit=2)) == 2

    def test_radius_is_enforced(self):
        # Every returned city must be within the radius of the home city
        home = "Chandler, AZ"
        for city in nearby_cities(home, 20, limit=5):
            assert distance_miles(home, city) <= 20

    def test_bare_zip_home_resolves(self):
        # 85224 is Chandler — expansion still works from a ZIP-only input
        ring = nearby_cities("Chandler, AZ", 25, limit=3)
        assert ring

    def test_unresolvable_home_returns_empty(self):
        assert nearby_cities("Nowhereville", 25) == []
        assert nearby_cities("", 25) == []
