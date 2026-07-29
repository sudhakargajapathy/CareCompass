"""Unit tests for utils/provenance.py — source attribution helpers."""

from utils.provenance import (
    REVIEW_PLATFORM_DOMAINS,
    is_profile_url,
    label_source,
    linkable,
    source_domain,
)


class TestIsProfileUrl:
    """Distinguish a doctor's own profile page from a city/specialty listing —
    the listing names many doctors, so its ratings aren't attributable to one."""

    def test_platform_profile_urls(self):
        assert is_profile_url("https://www.healthgrades.com/physician/dr-jane-doe-abc") is True
        assert is_profile_url("https://doctor.webmd.com/doctor/jane-doe-123") is True
        assert is_profile_url("https://www.vitals.com/doctors/Dr_Jane_Doe.html") is True
        assert is_profile_url("https://www.zocdoc.com/doctor/jane-doe-456") is True
        assert is_profile_url("https://www.ratemds.com/doctor/dr-jane-doe-chandler-az/") is True

    def test_platform_listing_urls_are_not_profiles(self):
        assert is_profile_url("https://doctor.webmd.com/providers/specialty/neurology/arizona/chandler") is False
        assert is_profile_url("https://www.healthgrades.com/neurology-directory/az-arizona/chandler") is False
        assert is_profile_url("https://www.vitals.com/local/neurologist/az/chandler") is False

    def test_non_platform_urls_are_treated_as_profiles(self):
        # The listing/profile distinction only applies to the review platforms
        assert is_profile_url("https://chandlerneuro.com/our-team") is True

    def test_empty(self):
        assert is_profile_url(None) is False
        assert is_profile_url("") is False


class TestListingPageLabel:
    def test_listing_page_is_labeled(self):
        label = label_source("https://doctor.webmd.com/providers/specialty/neurology/arizona/chandler")
        assert label == "doctor.webmd.com — listing page"

    def test_profile_page_is_clean(self):
        assert label_source("https://doctor.webmd.com/doctor/jane-doe-123") == "doctor.webmd.com"

    def test_healthgrades_listing_labeled(self):
        label = label_source("https://www.healthgrades.com/neurology-directory/az-arizona/chandler")
        assert label == "healthgrades.com — listing page"


class TestPlatformRoster:
    def test_roster_membership_is_deliberate(self):
        """The roster is evidence-managed — membership changes must update
        this test so they can't happen as a side effect."""
        assert REVIEW_PLATFORM_DOMAINS == (
            'healthgrades.com', 'vitals.com', 'zocdoc.com', 'webmd.com',
            'ratemds.com',
        )

    def test_known_drops_stay_dropped(self):
        # google/yelp/usnews were each dropped on live evidence (unscrapeable
        # or zero-yield); healthline/sharecare are syndicated healthgrades
        # data under the same owner and must never join.
        for domain in ("google.com", "yelp.com", "health.usnews.com",
                       "healthline.com", "sharecare.com"):
            assert domain not in REVIEW_PLATFORM_DOMAINS


class TestSourceDomain:
    def test_bare_domain(self):
        assert source_domain("https://www.healthgrades.com/physician/x") == "healthgrades.com"

    def test_schemeless(self):
        assert source_domain("vitals.com/doctors/x") == "vitals.com"

    def test_empty(self):
        assert source_domain(None) == ""
        assert source_domain("") == ""


class TestLabelSource:
    def test_independent_platform_is_plain(self):
        assert label_source("https://www.healthgrades.com/physician/x") == "healthgrades.com"

    def test_own_website_flagged_as_practice_site(self):
        label = label_source(
            "https://chandlerneurologyandsleep.com/reviews",
            website="https://www.chandlerneurologyandsleep.com",
        )
        assert label == "chandlerneurologyandsleep.com — practice site"

    def test_testimonial_url_flagged_even_without_website_field(self):
        label = label_source("https://someclinic.com/testimonials.html")
        assert label.endswith("— practice site")

    def test_platform_not_flagged_by_unrelated_website(self):
        label = label_source(
            "https://www.vitals.com/doctors/Dr_Jane_Doe", website="https://someclinic.com"
        )
        assert label == "vitals.com"

    def test_empty_url(self):
        assert label_source(None) == ""


class TestLinkable:
    def test_https_url(self):
        assert linkable("https://healthgrades.com/x") is True

    def test_rejects_schemeless_and_dangerous(self):
        assert linkable("healthgrades.com/x") is False
        assert linkable("https://x.com/a b") is False
        assert linkable("https://x.com/a)b") is False
        assert linkable(None) is False
