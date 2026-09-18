"""Constructed city-listing URLs, one per platform per specialty.

A platform's specialty vocabulary is its own, and a slug it does not use is
answered with a marketing page rather than an error — so a wrong slug is
indistinguishable from a city that has no providers. Discovery simply fetches
one page fewer, every run, for that specialty, and nothing says why.

That is what these assertions are for. They pin the slug each platform
actually serves, verified by fetching the URL (`evals/probe_listing_slugs.py`,
swept over the whole allowlist on two reference cities, 2026-09-17). No
network here: the sweep is the measurement, this is the guard that its result
cannot drift back.
"""
import pytest

from utils.platform_urls import (
    discovery_listing_urls,
    healthgrades_listing_url,
    vitals_listing_url,
    webmd_listing_url,
)
from utils.security import InputValidator

CITY, STATE = "Chandler", "AZ"

BUILDERS = {
    "healthgrades": healthgrades_listing_url,
    "webmd": webmd_listing_url,
    "vitals": vitals_listing_url,
}

# Every DEPARTURE from the naive hyphenation, per platform, as verified live.
# Listed as full URLs rather than slug fragments so a change to the path
# template fails here too.
PINNED = [
    # (platform, specialty, expected URL or None)
    ("healthgrades", "family medicine",
     "https://www.healthgrades.com/family-practice-directory/az-arizona/chandler"),
    ("healthgrades", "obstetrics",
     "https://www.healthgrades.com/obstetrics-gynecology-directory/az-arizona/chandler"),
    ("healthgrades", "gynecology",
     "https://www.healthgrades.com/obstetrics-gynecology-directory/az-arizona/chandler"),
    ("healthgrades", "orthopedics",
     "https://www.healthgrades.com/orthopedic-surgery-directory/az-arizona/chandler"),
    # No pathology directory exists at any level on this platform.
    ("healthgrades", "pathology", None),

    ("webmd", "cardiology",
     "https://doctor.webmd.com/providers/specialty/cardiovascular-disease/arizona/chandler"),
    ("webmd", "endocrinology",
     "https://doctor.webmd.com/providers/specialty/endocrinology-diabetes-metabolism/arizona/chandler"),
    ("webmd", "obstetrics",
     "https://doctor.webmd.com/providers/specialty/obstetrics-gynecology/arizona/chandler"),
    ("webmd", "orthopedics",
     "https://doctor.webmd.com/providers/specialty/orthopedic-surgery/arizona/chandler"),
    ("webmd", "general surgery",
     "https://doctor.webmd.com/providers/specialty/surgery/arizona/chandler"),
    ("webmd", "radiology",
     "https://doctor.webmd.com/providers/specialty/diagnostic-radiology/arizona/chandler"),

    ("vitals", "cardiology", "https://www.vitals.com/cardiovascular-disease/az/chandler"),
    ("vitals", "endocrinology",
     "https://www.vitals.com/endocrinology-diabetes-metabolism/az/chandler"),
    ("vitals", "family medicine", "https://www.vitals.com/family-medicine/az/chandler"),
    ("vitals", "general surgery", "https://www.vitals.com/surgery/az/chandler"),
    ("vitals", "otolaryngology",
     "https://www.vitals.com/otolaryngology-head-neck-surgery/az/chandler"),
    ("vitals", "radiology", "https://www.vitals.com/diagnostic-radiology/az/chandler"),
    ("vitals", "obstetrics", "https://www.vitals.com/obstetrics-gynecology/az/chandler"),
    ("vitals", "orthopedics", "https://www.vitals.com/orthopedic-surgery/az/chandler"),
]


class TestVerifiedSpecialtySlugs:
    @pytest.mark.parametrize("platform, specialty, expected", PINNED)
    def test_the_verified_slug_is_the_one_built(self, platform, specialty, expected):
        assert BUILDERS[platform](specialty, CITY, STATE) == expected

    def test_every_allowlisted_specialty_builds_a_url_on_at_least_two_platforms(self):
        """A specialty silently down to one source is a coverage hole that
        shows up as a thin pool, never as an error. Two is the floor because
        the cross-platform blend needs two platforms to engage at all."""
        thin = {
            specialty: discovery_listing_urls(specialty, f"{CITY}, {STATE}")
            for specialty in sorted(InputValidator.ALLOWED_SPECIALTIES)
        }
        assert {s: len(u) for s, u in thin.items() if len(u) < 2} == {}

    def test_the_one_specialty_with_no_directory_everywhere_yields_the_other_two(self):
        urls = discovery_listing_urls("Pathology", f"{CITY}, {STATE}")
        assert len(urls) == 2
        assert not any("healthgrades" in url for url in urls)

    @pytest.mark.parametrize("specialty, naive", [
        ("cardiology", "/cardiology/"),
        ("general surgery", "/general-surgery/"),
        ("radiology", "/radiology/"),
        ("otolaryngology", "/otolaryngology/"),
        ("endocrinology", "/endocrinology/"),
        ("family medicine", "/family-practice/"),
    ])
    def test_the_naive_slug_is_never_what_vitals_receives(self, specialty, naive):
        """Each of these served an identical 1,326-char marketing stub — the
        same bytes in every city, which is exactly why a wrong slug reads as
        "this town has no providers" rather than as a failure. `family
        medicine` was the one that was actively wrong rather than missing: it
        carried a different platform's term."""
        assert naive not in (vitals_listing_url(specialty, CITY, STATE) or "")

    def test_an_unknown_specialty_still_builds_nothing_rather_than_guessing(self):
        assert healthgrades_listing_url("", CITY, STATE) is None
        assert vitals_listing_url("Neurology", CITY, "ZZ") is None
        assert webmd_listing_url("Neurology", "", STATE) is None
