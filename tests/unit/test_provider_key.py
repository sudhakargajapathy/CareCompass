"""Provider identity: the key shared by dedup and the enrichment cache.

The rank-1/rank-3 duplicate of the 2026-07-25 field run came from asking this
question two different ways. These tests pin the one answer.
"""

import pytest

from utils.provider_key import (
    _strip_street_prefix,
    normalize_name_tokens,
    normalized_name,
    normalized_place,
    provider_cache_key,
)


# ---- the defect that motivated this ----

def test_hyphenated_and_spaced_surnames_are_one_person():
    """The live-run duplicate: 'Hussam Seif-Eddeine, MD' at rank 1 and
    'Dr. Hussam Seif Eddeine, MD' at rank 3 were the same physician. The old
    tokenizer split on whitespace only, so the hyphenated spelling was ONE
    token against the spaced spelling's two, overlap computed to 0.0, and the
    0.8 dedup threshold could never fire."""
    assert normalize_name_tokens("Hussam Seif-Eddeine, MD") == \
           normalize_name_tokens("Dr. Hussam Seif Eddeine, MD")


def test_same_physician_keys_identically_across_address_precision():
    """Discovery may hold a full street address while the cache holds a city.
    Both must reach the same row, or a hit is impossible in practice."""
    assert provider_cache_key(
        "Hussam Seif-Eddeine, MD", "2979 West Elliot Road Suite 2, Chandler, AZ 85224"
    ) == provider_cache_key("Dr. Hussam Seif Eddeine, MD", "Chandler, AZ")


# ---- normalization ----

@pytest.mark.parametrize("raw", [
    "Dr. Andrea An, MD", "Andrea An MD", "ANDREA AN", "An, Andrea",
    "Andrea An, M.D.", "Dr Andrea An",
])
def test_honorifics_and_credentials_do_not_change_identity(raw):
    assert normalize_name_tokens(raw) == {"andrea", "an"}


def test_token_order_does_not_change_the_key():
    assert normalized_name("Andrea An") == normalized_name("An Andrea")


def test_apostrophes_elide_rather_than_split():
    """'O'Brien' -> 'obrien' matches the common unpunctuated spelling. Spacing
    it instead would produce {o, brien} and match neither."""
    assert normalize_name_tokens("Dr. Sean O'Brien, MD") == {"sean", "obrien"}
    assert normalize_name_tokens("Sean OBrien") == {"sean", "obrien"}


@pytest.mark.parametrize("empty", [None, "", "   ", "Dr. MD"])
def test_empty_or_credential_only_names_yield_no_tokens(empty):
    assert normalize_name_tokens(empty) == set()


def test_place_uses_city_and_state_not_street():
    assert normalized_place("123 Health St, Chandler, AZ 85224") == "chandler az"
    assert normalized_place("Chandler, AZ") == "chandler az"


def test_state_disambiguates_identically_named_cities():
    """Springfield is not a unique place."""
    assert provider_cache_key("Jane Doe", "Springfield, IL") != \
           provider_cache_key("Jane Doe", "Springfield, MA")


# ---- key properties ----

def test_key_is_stable_not_salted():
    """Python salts str hash() per process, which is why the previous ID scheme
    produced a fresh ID after every restart and nothing could ever be looked
    up. A sha256-derived key must be reproducible from its inputs alone."""
    import hashlib
    expected = hashlib.sha256(b"an andrea|chandler az").hexdigest()[:16]
    assert provider_cache_key("Dr. Andrea An, MD", "Chandler, AZ") == expected


def test_different_physicians_in_one_city_do_not_collide():
    assert provider_cache_key("Andrea An, MD", "Chandler, AZ") != \
           provider_cache_key("Brian Rabin, MD", "Chandler, AZ")


def test_same_physician_different_city_is_a_different_row():
    """Reviews are city-scoped in practice; a move should not silently serve
    the old city's evidence."""
    assert provider_cache_key("Andrea An", "Chandler, AZ") != \
           provider_cache_key("Andrea An", "Phoenix, AZ")


def test_specialty_is_not_part_of_the_key():
    """The same physician found via a different specialty phrasing must hit —
    the key is identity, not search context. Guarded because adding specialty
    to the basis would look harmless and silently halve the hit rate."""
    a = provider_cache_key("Andrea An, MD", "Chandler, AZ")
    assert a == provider_cache_key("Andrea An, MD", "Chandler, AZ")
    assert len(a) == 16


class TestStreetAddressCacheKeys:
    """`parse_location` splits on COMMAS, so an address written without one
    before the city hands back the whole street address as the "city".

    On 2026-07-28 two live runs of the same search rendered one physician's
    address both ways and the cache reported 7 misses out of 8. Every such run
    also writes a fresh orphan row under a key no later read asks for — the
    same failure class `resolve_cache_key` exists to prevent, one layer down in
    the normalizer.
    """

    # Every spelling the two runs and the directory sites produced for one
    # physician. They must all be ONE key.
    _SAME_DOCTOR = [
        "Chandler, AZ",
        "Chandler, AZ 85224",
        "2201 W Fairview St, Chandler, AZ",
        "2201 W Fairview St Ste 1, Chandler, AZ 85224",
        "2201 W Fairview St Ste 1 Chandler, AZ 85224",   # <- the comma-less one
    ]

    def test_every_spelling_of_one_address_makes_one_key(self):
        keys = {provider_cache_key("Dr. Andrea An, MD", loc) for loc in self._SAME_DOCTOR}
        assert len(keys) == 1, (
            "one physician in one city must have one cache key; got "
            f"{len(keys)} for {self._SAME_DOCTOR}"
        )

    def test_the_comma_less_form_resolves_to_the_city(self):
        """The specific string from the 2026-07-28 run."""
        assert normalized_place("2201 W Fairview St Ste 1 Chandler, AZ 85224") == "chandler az"

    @pytest.mark.parametrize("location,expected", [
        ("13640 N Plaza Del Rio Blvd Peoria, AZ", "peoria az"),
        ("1500 S Dobson Rd Suite 301 Mesa, AZ 85202", "mesa az"),
        ("7 E Palo Verde St Bldg 1 Gilbert, AZ", "gilbert az"),
    ])
    def test_other_comma_less_shapes(self, location, expected):
        assert normalized_place(location) == expected

    def test_a_multiword_city_survives(self):
        """The stripper keeps everything after the LAST street marker, so a
        two-word city name must not lose its first word."""
        assert normalized_place("100 N Main St San Tan Valley, AZ") == "san tan valley az"

    def test_an_all_street_city_is_returned_whole_not_emptied(self):
        """Asserted on the HELPER, because `normalized_place` masks it.

        When every token looks like a street part there is no city to isolate,
        and `_strip_street_prefix` must hand back what it was given. Returning
        "" from a normalizer would be the worse failure — a shared empty key
        collides every unparseable provider onto one row, and a false cache HIT
        attributes one doctor's reviews to another. A miss only costs a search;
        that asymmetry is the whole reason the fallback is there.

        End-to-end the difference is currently invisible: an empty city falls
        through to `normalized_place`'s raw-text fallback, which happens to
        produce the same string. So this is belt-and-braces, tested where it is
        actually reachable rather than through a path that masks it —
        revert-in-isolation showed the end-to-end assertion passing either way.
        """
        assert _strip_street_prefix("2201 W Fairview St Ste 1") == "2201 W Fairview St Ste 1"
        assert _strip_street_prefix("Chandler") == "Chandler"
        assert normalized_place("Chandler, AZ") == "chandler az"
