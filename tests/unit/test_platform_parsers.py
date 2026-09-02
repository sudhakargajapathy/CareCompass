"""Deterministic parsing of review-platform listing and profile pages.

Every fixture below reproduces markup taken from real fetched page text. That
matters more than usual here, because an earlier survey of these same pages
concluded that webmd listings state no rating and vitals listings state only a
city-wide average. Both were false, and both for the same reason: the survey
matched only healthgrades' "Rated X out of 5 … from N ratings" phrasing, so
webmd's "[3.0 13 Ratings]" and vitals' bare "3.5 12 ratings" scored zero. A
parser per platform, tested against that platform's own words, is the fix for
that class of error — a wider regex is not.
"""
import re

import pytest

from utils.geo import parse_location
from utils.platform_urls import healthgrades_page_urls, listing_page_urls
from utils.listing_parser import (
    healthgrades_result_count,
    listing_result_count,
    clean_address,
    is_physician,
    parse_listing,
    strip_featured,
)
from utils.profile_parser import parse_profile, strip_data_uris

HG_LISTING_URL = "https://www.healthgrades.com/neurology-directory/az-arizona/chandler"
WEBMD_LISTING_URL = "https://doctor.webmd.com/providers/specialty/neurology/arizona/chandler"
VITALS_LISTING_URL = "https://www.vitals.com/neurology/az/queen-creek"

HG_LISTING = """## Featured Results

### [Dr. Sponsored Person, MD](https://www.healthgrades.com/physician/dr-sponsored-aaa)

Rated 5.0 out of 5 5.0 from 1 ratings Neurology [1 Ad St Phoenix, AZ 85001 1 mi miles away](https://www.healthgrades.com/physician/dr-sponsored-aaa#locations)

## All Results

### [Dr. Hemant Pandey, MD](https://www.healthgrades.com/physician/dr-hemant-pandey-xsjwm)

Rated 3.8 out of 5 3.8 from 88 ratings Neurology [4045 W Chandler Blvd Bldg F Chandler, AZ 85226 4.1 mi miles away](https://www.healthgrades.com/physician/dr-hemant-pandey-xsjwm#locations)

    *   Explains conditions well

### [Dr. Yazan Al-Hasan, MD](https://www.healthgrades.com/physician/dr-yazan-al-hasan-xymrhc5)

Rated 5.0 out of 5 5.0 from 4 ratings Neurology [7242 E Osborn Rd Ste 400 Scottsdale, AZ 85251 13 mi miles away](https://www.healthgrades.com/physician/dr-yazan-al-hasan-xymrhc5#locations)
"""

WEBMD_LISTING = """We found 48 Neurologists in Chandler, AZ.

## Filter Results

## Featured Results

## [Dr. Chelsea Weeks](https://doctor.webmd.com/doctor/chelsea-weeks-abc-overview)
Psychology
[0 0 Ratings](https://doctor.webmd.com/doctor/chelsea-weeks-abc-overview#ratings)

## All Results

*   ![Image 21: Dr. Paul Ryan Macdonald - Chandler, AZ - Neurology](https://mdxvitals-res.cloudinary.com/x/photo.jpg) ## [Dr. Paul Ryan Macdonald](https://doctor.webmd.com/doctor/paul-ryan-macdonald-5612ad17-7376-47fc-b70b-ddf0f97dda8f-overview)
Neurology
[3.0 13 Ratings](https://doctor.webmd.com/doctor/paul-ryan-macdonald-5612ad17-7376-47fc-b70b-ddf0f97dda8f-overview#ratings)  28 Years Exp erience 655 S Dobson Rd Ste 103 Bldg A, Chandler, AZ, 85224 2.03 miles Dr. Paul Macdonald is a neurologist....View Profile
*   ## [Jeannine Thomas, MSPA, PA-C](https://doctor.webmd.com/doctor/jeannine-thomas-zzz-overview)
Neurology
[5.0 4 Ratings](https://doctor.webmd.com/doctor/jeannine-thomas-zzz-overview#ratings)  9 Years Exp erience 1 Main St, Chandler, AZ, 85224 0.5 miles
"""

VITALS_LISTING = """We found **48 Neurologists** available in **Queen Creek, AZ** with an average **4 stars** across **897 reviews**. Find the best one.
## Filter Neurologists Results
## Featured Results
![Minder Memory Center - San Diego, CA - Psychology, Neuropsychology](https://img-vitals.lb.wbmdstatic.com/x.jpg) VERIFIED
### [Minder Memory Center](/doctors/minder-memory-center-1xqvw0)
[0 ratings](/doctors/minder-memory-center-1xqvw0#rating-overview)

## All Results
Dr. Brandon Craig Woods - Mesa, AZ - Neurology, Internal Medicine
### Dr. Brandon Craig Woods
Neurology
3.5 12 ratings
18 years exp
Accepting New Patients
Mesa, AZ9.8 mi
### [Dr. Ramzy G Medaa, MD](/doctors/ramzy-medaa-clje80)
Neurology
[0 ratings](/doctors/ramzy-medaa-clje80#rating-overview)
27 years exp
Scottsdale, AZ12.4 mi
### Dr. Edgardo D Zavala-Alarcon, MD
Plastic Surgery
4.7 72 ratings
48 years exp
Chandler, AZ0.8 mi
"""


def _by_name(rows):
    return {r["name"]: r for r in rows}


class TestHealthgradesListing:
    def test_full_entry(self):
        row = _by_name(parse_listing(HG_LISTING_URL, HG_LISTING))["Dr. Hemant Pandey, MD"]

        assert row["rating"] == 3.8
        assert row["review_count"] == 88
        assert row["specialty"] == "Neurology"
        assert row["profile_url"].endswith("/physician/dr-hemant-pandey-xsjwm")
        assert row["location"] == "4045 W Chandler Blvd Bldg F Chandler, AZ 85226"

    def test_the_listings_own_mileage_is_never_kept(self):
        """Two independent reasons, either of which is disqualifying.

        `utils.geo` anchors its ZIP pattern to the END of the address — that
        anchoring exists because an unanchored one read a five-digit STREET
        NUMBER as a ZIP and reported an address 2058 miles away at "zip"
        precision. And the mileage is measured from the PAGE's city, not the
        user's, so a parsed distance would be from the wrong origin entirely."""
        for row in parse_listing(HG_LISTING_URL, HG_LISTING):
            assert not re.search(r"\bmi\b|miles", row["location"] or "")
            assert "distance" not in row
            assert re.search(r"\d{5}$", row["location"]), "geo needs a terminal ZIP"

    def test_every_address_resolves_through_geo(self):
        for row in parse_listing(HG_LISTING_URL, HG_LISTING):
            parsed = parse_location(row["location"])
            assert parsed["zip"], row["location"]
            assert parsed["state"] == "AZ"

    def test_integer_mileage_is_stripped_too(self):
        """"13 mi miles away" — not every entry states a decimal."""
        row = _by_name(parse_listing(HG_LISTING_URL, HG_LISTING))["Dr. Yazan Al-Hasan, MD"]
        assert row["location"] == "7242 E Osborn Rd Ste 400 Scottsdale, AZ 85251"


class TestWebmdListing:
    def test_rating_and_count_come_from_the_link_TEXT(self):
        """"[3.0 13 Ratings](…#ratings)". A survey looking for healthgrades'
        "Rated X out of 5 … from N ratings" scored webmd at zero pairs and
        concluded the platform states no ratings on listings."""
        row = _by_name(parse_listing(WEBMD_LISTING_URL, WEBMD_LISTING))["Dr. Paul Ryan Macdonald"]

        assert row["rating"] == 3.0
        assert row["review_count"] == 13

    def test_the_broken_word_in_Exp_erience_is_real_page_text(self):
        """webmd renders "28 Years Exp erience" with a space inside the word.
        A pattern requiring "Experience" matches nothing at all."""
        row = _by_name(parse_listing(WEBMD_LISTING_URL, WEBMD_LISTING))["Dr. Paul Ryan Macdonald"]
        assert row["years_experience"] == 28

    def test_the_address_does_not_begin_inside_the_ratings_URL(self):
        """The address pattern starts at a digit, and the ratings link's own URL
        hash is full of them: a permissive body matched from
        "…0f97dda8f-overview#ratings)  28 Years Exp erience 655 S Dobson Rd…"
        through to the ZIP, producing an address beginning mid-URL."""
        row = _by_name(parse_listing(WEBMD_LISTING_URL, WEBMD_LISTING))["Dr. Paul Ryan Macdonald"]

        assert row["location"] == "655 S Dobson Rd Ste 103 Bldg A, Chandler, AZ 85224"
        assert "overview" not in row["location"]
        assert "Years" not in row["location"]

    def test_the_comma_before_the_ZIP_is_normalised(self):
        """webmd writes "Chandler, AZ, 85224". `geo.parse_location` splits on
        commas, so the extra one makes the state its own field."""
        row = _by_name(parse_listing(WEBMD_LISTING_URL, WEBMD_LISTING))["Dr. Paul Ryan Macdonald"]
        parsed = parse_location(row["location"])

        assert parsed["city"] == "Chandler"
        assert parsed["state"] == "AZ"
        assert parsed["zip"] == "85224"


class TestVitalsListing:
    def test_the_plain_rating_form(self):
        """"3.5 12 ratings" — rating then count, no brackets, no link. A survey
        requiring brackets matched only the zero-rating ads, which is why every
        vitals provider came back with count 0."""
        row = _by_name(parse_listing(VITALS_LISTING_URL, VITALS_LISTING))["Dr. Brandon Craig Woods"]

        assert row["rating"] == 3.5
        assert row["review_count"] == 12
        assert row["years_experience"] == 18

    def test_the_linked_form_carries_a_count_and_no_rating(self):
        """"[0 ratings](…#rating-overview)" is what a provider with no reviews
        gets. Recording the count alone keeps the roster entry; the blend
        ignores it because a pair needs both."""
        row = _by_name(parse_listing(VITALS_LISTING_URL, VITALS_LISTING))["Dr. Ramzy G Medaa, MD"]

        assert row["rating"] is None
        assert row["review_count"] == 0

    def test_a_relative_profile_link_is_resolved_against_the_page(self):
        """vitals writes its entry links host-relative — "/doctors/ramzy-medaa-clje80"
        — while the other two write them absolute.

        A relative href is usable for comparing two vitals rows against each
        other and useless for anything else: it cannot be fetched, and it
        carries no domain, so a cross-platform identity check keyed on the
        profile URL would see "/doctors/x" and
        "https://www.healthgrades.com/physician/y" as two strings with nothing
        in common — true, but only because one of them was never finished."""
        row = _by_name(parse_listing(VITALS_LISTING_URL, VITALS_LISTING))["Dr. Ramzy G Medaa, MD"]

        assert row["profile_url"] == "https://www.vitals.com/doctors/ramzy-medaa-clje80"

    @pytest.mark.parametrize("url,text", [
        (HG_LISTING_URL, HG_LISTING),
        (WEBMD_LISTING_URL, WEBMD_LISTING),
    ])
    def test_an_already_absolute_link_is_left_byte_identical(self, url, text):
        """Resolution must be a no-op on the two platforms that do not need it —
        a normaliser that rewrites URLs it did not need to touch is its own
        source of drift."""
        for row in parse_listing(url, text):
            assert row["profile_url"].startswith("https://")
            assert "//" not in row["profile_url"][8:], row["profile_url"]

    def test_a_missing_page_url_degrades_rather_than_inventing_a_host(self):
        from utils.listing_parser import _absolute_url
        assert _absolute_url("", "/doctors/x") == "/doctors/x"
        assert _absolute_url(VITALS_LISTING_URL, None) is None

    def test_both_heading_forms_are_recognised(self):
        """vitals writes "### Dr. Brandon Craig Woods" bare and
        "### [Dr. Ramzy G Medaa, MD](/doctors/…)" linked, on the same page."""
        names = _by_name(parse_listing(VITALS_LISTING_URL, VITALS_LISTING))

        assert "Dr. Brandon Craig Woods" in names        # bare
        assert "Dr. Ramzy G Medaa, MD" in names          # linked
        assert names["Dr. Brandon Craig Woods"]["profile_url"] is None
        assert names["Dr. Ramzy G Medaa, MD"]["profile_url"].endswith(
            "/doctors/ramzy-medaa-clje80")

    def test_a_bare_heading_recovers_its_url_from_View_Profile(self):
        """The bare form yields no URL from the heading, but vitals often
        repeats the link further down the block as a call to action. A second
        place to look, not a new source of data."""
        page = (
            "## All Results\n"
            "### Dr. Brandon Craig Woods\nNeurology\n3.5 12 ratings\n18 years exp\n"
            "Mesa, AZ9.8 mi\nprose [View Profile](/doctors/brandon-woods-x8k2)\n"
        )
        row = _by_name(parse_listing(VITALS_LISTING_URL, page))["Dr. Brandon Craig Woods"]

        assert row["profile_url"] == "https://www.vitals.com/doctors/brandon-woods-x8k2"

    def test_the_slug_must_agree_with_the_name_it_is_attached_to(self):
        """A block runs to the next heading, so a trailing link belongs to the
        provider above it — but "belongs to the one above it" is the same
        reasoning behind the positional verdict fallback that let a provider
        collect its neighbour's penalty. vitals puts the name in the slug
        ("/doctors/brandon-woods-x8k2"), so the guess is checkable."""
        from utils.listing_parser import _recover_vitals_profile_url

        assert _recover_vitals_profile_url(
            "x [View Profile](/doctors/brandon-woods-x8k2)", "Dr. Brandon Craig Woods"
        ) == "/doctors/brandon-woods-x8k2"
        assert _recover_vitals_profile_url(
            "x [View Profile](/doctors/charanjit-dhillon-abc)", "Dr. Brandon Craig Woods"
        ) is None

    def test_a_linked_heading_still_wins_over_a_block_level_link(self):
        """Recovery is a FALLBACK. A block whose heading carries the URL must
        not have it replaced by a "View Profile" link that may point elsewhere."""
        page = (
            "## All Results\n"
            "### [Dr. Ramzy G Medaa, MD](/doctors/ramzy-medaa-clje80)\nNeurology\n"
            "3.0 9 ratings\n27 years exp\nMesa, AZ2.0 mi\n"
            "prose [View Profile](/doctors/someone-else-zzz9)\n"
        )
        row = _by_name(parse_listing(VITALS_LISTING_URL, page))["Dr. Ramzy G Medaa, MD"]

        assert row["profile_url"].endswith("/doctors/ramzy-medaa-clje80")
        assert "someone-else" not in row["profile_url"]

    def test_plain_text_View_Profile_recovers_nothing(self):
        """In real vitals text "View Profile" sometimes appears with no link at
        all. This recovers SOME bare-heading blocks, not all — and the real
        coverage rate on a live page is unmeasured."""
        page = (
            "## All Results\n"
            "### Dr. Plain Text Only\nNeurology\n4.0 5 ratings\n10 years exp\n"
            "Mesa, AZ1.0 mi\nNo link here at all. View Profile\n"
        )
        row = _by_name(parse_listing(VITALS_LISTING_URL, page))["Dr. Plain Text Only"]

        assert row["profile_url"] is None

    def test_mileage_glued_to_the_state_is_split_off(self):
        """"Mesa, AZ9.8 mi" — no space between the state and the distance."""
        row = _by_name(parse_listing(VITALS_LISTING_URL, VITALS_LISTING))["Dr. Brandon Craig Woods"]

        assert row["location"] == "Mesa, AZ"
        assert parse_location(row["location"])["city"] == "Mesa"

    def test_specialty_is_captured_so_the_caller_can_filter(self):
        """A vitals NEUROLOGY directory lists a plastic surgeon. Without the
        label there is nothing to filter on, and 4.7 over 72 ratings would
        outrank most of the real neurologists."""
        row = _by_name(parse_listing(VITALS_LISTING_URL, VITALS_LISTING))["Dr. Edgardo D Zavala-Alarcon, MD"]
        assert row["specialty"] == "Plastic Surgery"


class TestSponsoredAndNonPhysicians:
    @pytest.mark.parametrize("url,text,sponsored", [
        (HG_LISTING_URL, HG_LISTING, "Dr. Sponsored Person, MD"),
        (WEBMD_LISTING_URL, WEBMD_LISTING, "Dr. Chelsea Weeks"),
        (VITALS_LISTING_URL, VITALS_LISTING, "Minder Memory Center"),
    ])
    def test_the_featured_block_never_reaches_the_pool(self, url, text, sponsored):
        """Paid placement, and provably not the local roster: the same entries
        came back on pages for unrelated cities, and one of them is
        "Minder Memory Center — San Diego, CA — Psychology" on a Queen Creek AZ
        neurology page.

        The sharper hazard is cross-platform: webmd and vitals are both Internet
        Brands properties and share this inventory, so ONE advertiser appearing
        on both would satisfy the blend's "two platforms agree" condition —
        independence manufactured from a single paid slot."""
        assert sponsored not in _by_name(parse_listing(url, text))

    def test_strip_featured_leaves_a_page_without_ads_alone(self):
        """A defensive slice on every page would eat the first real entries."""
        plain = "## All Results\n### [Dr. Real, MD](https://www.healthgrades.com/physician/dr-real-a)\n"
        assert strip_featured(plain) == plain

    def test_the_city_wide_average_never_becomes_a_providers_rating(self):
        """vitals' intro states "average **4 stars** across **897 reviews**" for
        the whole city. Bound to a provider that is their city's mean at a
        897-review weight, which would dominate the Bayesian blend."""
        rows = parse_listing(VITALS_LISTING_URL, VITALS_LISTING)
        assert not any(r["review_count"] == 897 for r in rows)
        assert "897" not in strip_featured(VITALS_LISTING)

    @pytest.mark.parametrize("name,ok", [
        ("Dr. Hemant Pandey, MD", True),
        ("Dr. Brandon Craig Woods", True),
        ("Jane Smith, DO", True),
        ("Jeannine Thomas, MSPA, PA-C", False),
        ("Jennifer Wright-Bennion, CNM, APRN", False),
        ("Minder Memory Center", False),
        ("Chandler Regional Medical Center", False),
        ("", False),
    ])
    def test_physician_filter(self, name, ok):
        assert is_physician(name) is ok


class TestListingDispatch:
    def test_an_unknown_domain_yields_nothing(self):
        assert parse_listing("https://example.com/doctors", HG_LISTING) == []

    def test_dispatch_is_by_domain_not_by_page_kind(self):
        """`url_page_kind` classifies every vitals city page ("/neurology/az/mesa")
        as "unknown", so keying dispatch off it would silently parse nothing for
        a platform whose pages parse fine."""
        from utils.provenance import url_page_kind
        assert url_page_kind(VITALS_LISTING_URL) == "unknown"
        assert parse_listing(VITALS_LISTING_URL, VITALS_LISTING)

    def test_the_source_url_is_recorded_on_every_row(self):
        for row in parse_listing(HG_LISTING_URL, HG_LISTING):
            assert row["review_source_url"] == HG_LISTING_URL

    def test_empty_and_garbage_input_are_safe(self):
        assert parse_listing(HG_LISTING_URL, "") == []
        assert parse_listing(HG_LISTING_URL, "### not a heading with no link") == []
        assert parse_listing("", HG_LISTING) == []

    def test_clean_address_is_idempotent(self):
        once = clean_address("4045 W Chandler Blvd Chandler, AZ 85226 4.1 mi miles away")
        assert clean_address(once) == once


HG_PROFILE_URL = "https://www.healthgrades.com/physician/dr-hemant-pandey-xsjwm"
WEBMD_PROFILE_URL = "https://doctor.webmd.com/doctor/andrea-an-e085e811-4d70-overview"
VITALS_PROFILE_URL = "https://www.vitals.com/doctors/marianne-de-lima-fkth8r"

HG_PROFILE_TEMPLATE_A = """# Dr. Andrea An, MD
## Neurology| 25+ years of experience
25+ years of experience
4.1 Star Rating
Based on 70 reviews 4.1 Star Rating(70 reviews)
Review Save
### Practice
2201 W Fairview St Ste 1 ·Chandler, AZ 85224
"""

HG_PROFILE_TEMPLATE_B = """# Dr. Hemant Pandey, MD
## Neurology | 25+ years of experience
### Overall Patient Satisfaction
Likelihood of recommending Dr. Pandey to family and friends is 3.7727273 out of 5
| 5 star | 63% | 63% |
## Compare Providers
### You May Also Like
Dr. Someone Else, MD
4.9 Star Rating
Based on 512 reviews
"""

# The one data URI on the fetched Pandey profile: 8,345 of 15,750 characters,
# carrying 1,246 of the page's 1,299 bare integers as path coordinates. Cut down
# here, with a "88" left in it — that is the count the directory states for this
# doctor, and it appears 28 times in the real page, every one inside this blob.
HG_PROFILE_WITH_SVG = """# Dr. Hemant Pandey, MD
![Zocdoc](data:image/svg+xml,%3csvg%20width='122'%20height='52'%20xmlns='http://www.w3.org/2000/svg'%3e%3cpath%20d='M76.4711%2029.0632C75.8808%2029.0632%2075.3148%2028.8265%2074.8973%2028.4053C74.4798%2027.9839%2074.2454%2027.4124%2074.2454%2026.8164C74.2454%2026.2207%2074.4798%2025.6492%2074.8973%2088.2278C75.3148%2024.8066%2075.8808%2024.5699%2076.4711%2024.5699Z'/%3e%3c/svg%3e)
### Overall Patient Satisfaction
Likelihood of recommending Dr. Pandey to family and friends is 3.7727273 out of 5
"""

# Dr. Andrea Hyeyong An's webmd profile header, as fetched. The rating and the
# count are on consecutive lines, and note the tenure: "26 Years Experience",
# webmd's own figure.
WEBMD_PROFILE_HEADER = """# Dr. Andrea Hyeyong An, MD
Is this you?
[Claim your profile](https://doctor.webmd.com/doctor/andrea-an-e085e811-4d70/claim)
Neurology
4.5
(61 Ratings)
Leave a review
[(480) 800-4890](tel:+14808004890)
26 Years Experience
Neurology Associates Neuroscience Center
2201 W Fairview St Ste 1, Chandler, AZ, 85224
[2 other locations](https://doctor.webmd.com/doctor/andrea-an-e085e811-4d70-overview)
## Overview
Dr. Andrea An is a neurologist with over 19 years of experience in outpatient care.
"""

# Dr. Cinthi Pillai's, where the header card did NOT survive extraction:
# "## Ratings & Reviews for Dr. Pillai" is a heading with an empty body, and the
# only statement of the numbers is the FAQ at the foot of the page. The Overview
# blurb states a DIFFERENT tenure from webmd's own.
WEBMD_PROFILE_FAQ_ONLY = """# Dr. Cinthi Pillai, MD
245 5th Ave Fl 3, New York, NY, 10016
## Overview
# **Dr. Cinthi Pillai is a Neuro-ophthalmologist with over 18 years of experience**
## Ratings & Reviews for Dr. Pillai
#### Patients' Perspective
## Frequently Asked Questions
### What are Dr. Cinthi Pillai's patient ratings?
Dr. Cinthi Pillai has received 20 ratings on WebMD Care. Patients gave Dr. Cinthi Pillai an average rating of 5 out of 5.
### How many years of experience does Dr. Cinthi Pillai have?
Dr. Cinthi Pillai has approximately 20 years of experience in the medical field.
"""

# Dr. Marianne De Lima's vitals profile: an UNCLAIMED profile, which renders the
# header pair as a link to the ratings anchor. Address has no commas at all.
VITALS_PROFILE_HEADER = """* [Home](https://www.vitals.com)
* [Find a Neurology Doctor](https://www.vitals.com/neurology)
* [AZ](https://www.vitals.com/neurology/az)
* [Chandler](https://www.vitals.com/neurology/az/chandler)
* **Dr. Marianne L De Lima, MD**
# Dr. Marianne L De Lima, MD
Is this you?  [Claim your Profile](/doctors/marianne-de-lima-fkth8r/claim)
[5 9](#rating-overview)
[(480) 800-4890](tel:4808004890)
14 Years of Experience 2201 W Fairview St Ste 1 Chandler AZ 85224  (3 other locations)
"""

# Dr. Peter Sunenshine's: a CLAIMED profile, which renders "Overview based on
# verified provider data" instead of the claim block and states no rating in his
# own sections at all. `## Quick Facts`, `## Patients' Perspective` and
# `## Dr. Sunenshine's Ratings and Reviews` are all empty headings.
VITALS_PROFILE_COMPARE = """# Dr. Peter Sunenshine
## Summary
Overview based on verified provider data
## Quick Facts
## Dr. Sunenshine's Ratings and Reviews
### 7 ratings & reviews
No data
very thorough, caring
## Certifications, License, & Education
### 28 Years Experience
## Compare with Similar Doctors

|  |  |  |  |
| --- | --- | --- | --- |
| Dr. Peter Sunenshine | Dr. Ramzy G Medaa, MD | Dr. Cinthi Pillai, MD | Jeannine Thomas, MSPA, PA-C |
| Neurology | Neurology | Neurology | Neurology |
| 5     (7 Ratings) | 0     (0 Ratings) | 0     (0 Ratings) | 0     (0 Ratings) |
| 28 Years of Experience | 27 Years of Experience | 20 Years of Experience | 21 Years of Experience |
| Current Profile | [View Profile](/doctors/ramzy-medaa-clje80) | [View Profile](/doctors/cinthi-pillai-wr6r0q) | [View Profile](/doctors/jeannine-thomas-0p0o0x) |
## Similar Doctors
"""

# The same table with the subject moved out of column 0. Nothing on the page
# says the subject comes first; the `Current Profile` marker is what says which
# column is theirs.
VITALS_PROFILE_COMPARE_REORDERED = VITALS_PROFILE_COMPARE.replace(
    "| Dr. Peter Sunenshine | Dr. Ramzy G Medaa, MD |",
    "| Dr. Ramzy G Medaa, MD | Dr. Peter Sunenshine |",
).replace(
    "| 5     (7 Ratings) | 0     (0 Ratings) |",
    "| 0     (0 Ratings) | 5     (7 Ratings) |",
).replace(
    "| Current Profile | [View Profile](/doctors/ramzy-medaa-clje80) |",
    "| [View Profile](/doctors/ramzy-medaa-clje80) | Current Profile |",
)


class TestHealthgradesProfile:
    def test_template_A_yields_a_full_pair(self):
        got = parse_profile("https://www.healthgrades.com/physician/dr-andrea-an-2pfjn",
                            HG_PROFILE_TEMPLATE_A)

        assert got["rating"] == 4.1
        assert got["review_count"] == 70
        assert got["years_experience"] == 25

    def test_template_B_yields_a_rating_and_NO_count_by_design(self):
        """healthgrades has two profile templates and this one states no total
        anywhere in the fetched text. Every occurrence of the true count (88,
        per the directory page) sits inside an inlined SVG, there is no ld+json,
        and the star breakdown is percentages with no denominator.

        A future change that makes this return a count is inventing one."""
        got = parse_profile(HG_PROFILE_URL, HG_PROFILE_TEMPLATE_B)

        assert got["rating"] == 3.8
        assert "review_count" not in got

    def test_a_neighbours_rating_in_a_promo_strip_is_never_read(self):
        """Both healthgrades rating patterns are unanchored first-match, and the
        page carries "## Compare Providers" / "### You May Also Like" blocks
        describing OTHER doctors. Without the bound, template B — which states
        no Star Rating of its own — picks up the neighbour's 4.9/512 and cards
        it as this provider's."""
        got = parse_profile(HG_PROFILE_URL, HG_PROFILE_TEMPLATE_B)

        assert got["rating"] == 3.8
        assert got.get("review_count") is None

    def test_the_unrounded_mean_is_rounded_for_display(self):
        """Template B states "3.7727273 out of 5". The card prints this value
        directly, and "3.7727273/5 weighted" reads as a bug."""
        assert parse_profile(HG_PROFILE_URL, HG_PROFILE_TEMPLATE_B)["rating"] == 3.8

    def test_the_middle_dot_address_is_normalised(self):
        """healthgrades separates suite from city with "·", not a comma. Without
        normalising it `geo.parse_location` makes the whole street line the city."""
        got = parse_profile("https://www.healthgrades.com/physician/dr-andrea-an-2pfjn",
                            HG_PROFILE_TEMPLATE_A)

        assert got["location"] == "2201 W Fairview St Ste 1, Chandler, AZ 85224"
        assert parse_location(got["location"])["city"] == "Chandler"
        assert parse_location(got["location"])["zip"] == "85224"


class TestInlinedImagePayloads:
    def test_a_count_that_exists_only_inside_an_svg_is_not_read(self):
        """The fetched Pandey profile is 53% one percent-encoded SVG by volume,
        and 1,246 of its 1,299 bare integers are path coordinates — including
        every one of the 28 occurrences of "88", his true review count per the
        directory. Reading one as a count would card a logo."""
        got = parse_profile(HG_PROFILE_URL, HG_PROFILE_WITH_SVG)

        assert got["rating"] == 3.8
        assert got.get("review_count") is None

    def test_the_blob_is_removed_rather_than_merely_ignored(self):
        """It is stripped, not skipped, because the same text feeds the excerpt
        that goes to the model: the blob begins at character 651 of the real
        page, so 46% of the 1,200-char head reservation meant to capture the
        rating header was being spent on it."""
        assert "%3csvg" in HG_PROFILE_WITH_SVG
        assert "%3csvg" not in strip_data_uris(HG_PROFILE_WITH_SVG)
        assert "Likelihood of recommending" in strip_data_uris(HG_PROFILE_WITH_SVG)

    def test_strip_boilerplate_alone_does_not_catch_it(self):
        """The reason this needs its own pass: `strip_boilerplate`'s chrome
        filters only apply to lines under 80 characters, and the real blob is
        one line of 8,345."""
        from utils.excerpt import strip_boilerplate

        assert "%3csvg" in strip_boilerplate(HG_PROFILE_WITH_SVG)


class TestWebmdProfile:
    def test_the_header_pair_is_read_from_consecutive_lines(self):
        """The fetched page writes "4.5" and "(61 Ratings)" on separate lines.
        The first version of this parser looked for the LISTING's bracket form
        `[4.5 61 Ratings]`, which a profile never uses — run against a real page
        it returned tenure and nothing else."""
        got = parse_profile(WEBMD_PROFILE_URL, WEBMD_PROFILE_HEADER)

        assert (got["rating"], got["review_count"]) == (4.5, 61)

    def test_the_FAQ_carries_the_pair_when_the_header_card_did_not_render(self):
        """On Dr. Pillai's profile "## Ratings & Reviews" is a heading with an
        empty body — the card is JS-rendered and never reaches the payload. The
        FAQ prose at the foot of the page is the only statement of the numbers
        that survives."""
        got = parse_profile("https://doctor.webmd.com/doctor/cinthi-pillai-x-overview",
                            WEBMD_PROFILE_FAQ_ONLY)

        assert (got["rating"], got["review_count"]) == (5.0, 20)

    def test_tenure_comes_from_webmd_and_never_from_the_provider_s_own_blurb(self):
        """The page states tenure twice and they DISAGREE: webmd's header chip
        and FAQ both say 20, while the Overview blurb the provider wrote says
        "over 18 years". Unanchored first-match takes the blurb, because the
        blurb sits above the FAQ."""
        assert parse_profile("https://doctor.webmd.com/doctor/cinthi-pillai-x-overview",
                             WEBMD_PROFILE_FAQ_ONLY)["years_experience"] == 20

    def test_a_header_tenure_outranks_a_blurb_further_down(self):
        assert parse_profile(WEBMD_PROFILE_URL,
                             WEBMD_PROFILE_HEADER)["years_experience"] == 26

    def test_the_address_survives_a_comma_before_the_ZIP(self):
        """webmd writes "Chandler, AZ, 85224". `geo`'s ZIP pattern is anchored
        to the END of the address and does not admit that comma, so the address
        resolved to city precision — one shared centroid for every provider in
        the city — instead of a measured distance."""
        got = parse_profile(WEBMD_PROFILE_URL, WEBMD_PROFILE_HEADER)

        assert got["location"] == "2201 W Fairview St Ste 1, Chandler, AZ 85224"
        assert parse_location(got["location"])["zip"] == "85224"


class TestVitalsProfile:
    def test_the_header_pair_is_a_link_to_the_ratings_anchor(self):
        """vitals writes it as `[5 9](#rating-overview)` — 5 stars over 9
        ratings. The `#rating-overview` target is the guard: it is what
        separates this from any other bracketed number pair on the page."""
        got = parse_profile(VITALS_PROFILE_URL, VITALS_PROFILE_HEADER)

        assert (got["rating"], got["review_count"]) == (5.0, 9)

    def test_the_header_also_carries_tenure_and_a_comma_free_address(self):
        got = parse_profile(VITALS_PROFILE_URL, VITALS_PROFILE_HEADER)

        assert got["years_experience"] == 14
        assert got["location"] == "2201 W Fairview St Ste 1 Chandler, AZ 85224"
        assert parse_location(got["location"])["zip"] == "85224"

    def test_the_breadcrumb_gives_specialty_and_city_from_path_segments(self):
        got = parse_profile(VITALS_PROFILE_URL, VITALS_PROFILE_HEADER)

        assert got["breadcrumb_specialty"] == "Neurology"
        assert got["breadcrumb_city"] == "Chandler, AZ"

    def test_a_claimed_profile_falls_back_to_the_comparison_table(self):
        """A claimed profile renders "Overview based on verified provider data"
        in place of the claim block, and states no rating in the doctor's own
        sections — `## Quick Facts`, `## Patients' Perspective` and
        `## Dr. Sunenshine's Ratings and Reviews` are all empty headings. The
        only occurrence is the comparison table."""
        got = parse_profile("https://www.vitals.com/doctors/peter-sunenshine-fy3kqb",
                            VITALS_PROFILE_COMPARE)

        assert (got["rating"], got["review_count"]) == (5.0, 7)

    def test_the_subject_s_column_is_found_by_marker_not_by_position(self):
        """The other three columns are different doctors. Nothing says the
        subject comes first — the `Current Profile` cell, where the others say
        `View Profile`, is what identifies their column. Position is the same
        mistake as the ranking fallback that let a provider collect their
        neighbour's penalty."""
        got = parse_profile("https://www.vitals.com/doctors/peter-sunenshine-fy3kqb",
                            VITALS_PROFILE_COMPARE_REORDERED)

        assert (got["rating"], got["review_count"]) == (5.0, 7)

    def test_a_table_that_disagrees_with_the_doctor_s_own_count_is_dropped(self):
        """The cross-check is what makes reading a neighbour-bearing table safe:
        the count in `### N ratings & reviews` is the doctor's own, so if the
        column we picked states a different total we picked the wrong column.
        Publishing it anyway would put a stranger's stars on the card."""
        drifted = VITALS_PROFILE_COMPARE.replace("### 7 ratings & reviews",
                                                 "### 9 ratings & reviews")
        got = parse_profile("https://www.vitals.com/doctors/peter-sunenshine-fy3kqb", drifted)

        assert got.get("rating") is None
        assert got["review_count"] == 9

    def test_tenure_comes_from_the_doctor_s_own_heading_not_the_table_row(self):
        """The table's tenure row is `| 28 | 27 | 20 | 21 |` and the subject is
        not always first. `### 28 Years Experience` is his own heading."""
        got = parse_profile("https://www.vitals.com/doctors/peter-sunenshine-fy3kqb",
                            VITALS_PROFILE_COMPARE_REORDERED)

        assert got["years_experience"] == 28


class TestProfileParserContract:
    def test_the_page_H1_becomes_the_name_the_identity_check_reads(self):
        """All three platforms write the subject's name as the H1 and nowhere
        else in a fixed position. Taking it here makes `page_provider_name`
        deterministic instead of something the cheapest model transcribes."""
        assert parse_profile(VITALS_PROFILE_URL,
                             VITALS_PROFILE_HEADER)["page_provider_name"] == \
            "Dr. Marianne L De Lima, MD"
        assert parse_profile(WEBMD_PROFILE_URL,
                             WEBMD_PROFILE_HEADER)["page_provider_name"] == \
            "Dr. Andrea Hyeyong An, MD"

    def test_an_empty_result_means_UNREADABLE_not_absent(self):
        """{} must route to the LLM fallback. Treating it as "this page has no
        rating" would discard pairs these platforms are known to publish."""
        assert parse_profile("https://doctor.webmd.com/doctor/x-overview", "") == {}
        assert parse_profile("https://example.com/x", "4.5 33 Ratings") == {}

    def test_a_PARTIAL_result_is_the_normal_case_and_not_a_full_answer(self):
        """This is why the caller's fallback has to be per FIELD. Both of these
        are correct, complete reads of their page and both are missing half the
        pair — a per-page trigger ("the parser returned something, skip the
        model") would silently drop the other half."""
        template_b = parse_profile(HG_PROFILE_URL, HG_PROFILE_TEMPLATE_B)
        assert template_b["rating"] and template_b.get("review_count") is None

        no_pair = parse_profile("https://www.vitals.com/doctors/ramzy-medaa-clje80",
                                "# Dr. Ramzy G Medaa, MD\n### 27 Years Experience\n")
        assert no_pair["years_experience"] == 27
        assert no_pair.get("rating") is None


# ---- wiring: parser first, LLM only for what the parsers could not read ----


@pytest.fixture
def gatherer():
    from unittest.mock import MagicMock, patch as _patch
    from agents.data_gatherer import DataGathererAgent
    with _patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
        agent = DataGathererAgent()
        agent.tavily_client = MagicMock()
        agent.anthropic_client = MagicMock()
        return agent


def _page(url, raw):
    return {"title": "t", "url": url, "content": "", "raw_content": raw}


class TestParserFirstWiring:
    """Composed on `_extract_provider_data`, not on `parse_listing`.

    A parser-only test would stay green if the wiring were removed and every
    page went back to the LLM — which is the regression that matters, since the
    whole point is to stop an LLM deciding whether a rating exists.
    """

    def test_a_parsable_page_never_reaches_the_model(self, gatherer):
        from unittest.mock import MagicMock, patch as _patch
        with _patch.object(gatherer, "_extract_page_shard") as shard:
            out = gatherer._extract_provider_data(
                [_page(HG_LISTING_URL, HG_LISTING)], "Neurology", "Chandler, AZ"
            )
        shard.assert_not_called()
        assert {p["name"] for p in out} >= {"Dr. Hemant Pandey, MD", "Dr. Yazan Al-Hasan, MD"}
        assert all(p["extraction_source"] == "listing_parser" for p in out)

    def test_an_unparsable_page_falls_back_to_the_model(self, gatherer):
        """A site redesign must degrade to the previous behaviour, not empty the
        pool — "No providers found" is what the user would see."""
        from unittest.mock import patch as _patch
        page = _page("https://www.healthgrades.com/neurology-directory/az-arizona/x",
                     "nothing this parser recognises at all")
        with _patch.object(gatherer, "_extract_page_shard", return_value=[{"name": "Dr. LLM"}]) as shard:
            out = gatherer._extract_provider_data([page], "Neurology", "Chandler, AZ")
        shard.assert_called_once()
        assert [p["name"] for p in out] == ["Dr. LLM"]

    def test_parsed_and_fallback_results_are_merged(self, gatherer):
        from unittest.mock import patch as _patch
        pages = [
            _page(HG_LISTING_URL, HG_LISTING),
            _page("https://www.healthgrades.com/neurology-directory/az-arizona/y", "unreadable"),
        ]
        with _patch.object(gatherer, "_extract_page_shard", return_value=[{"name": "Dr. LLM"}]):
            names = {p["name"] for p in gatherer._extract_provider_data(
                pages, "Neurology", "Chandler, AZ")}
        assert "Dr. LLM" in names
        assert "Dr. Hemant Pandey, MD" in names

    def test_a_parsed_row_carries_a_review_observation(self, gatherer):
        """The blend, the same-domain collapse and the platform-pair count all
        read `review_observations`. A row setting only rating/review_count would
        score but contribute nothing to cross-platform agreement."""
        from unittest.mock import patch as _patch
        with _patch.object(gatherer, "_extract_page_shard"):
            out = gatherer._extract_provider_data(
                [_page(HG_LISTING_URL, HG_LISTING)], "Neurology", "Chandler, AZ")
        pandey = next(p for p in out if p["name"] == "Dr. Hemant Pandey, MD")

        assert pandey["review_observations"] == [{
            "source_url": "https://www.healthgrades.com/physician/dr-hemant-pandey-xsjwm",
            "read_from_url": HG_LISTING_URL,
            "rating": 3.8, "review_count": 88,
            "page_provider_name": "Dr. Hemant Pandey, MD",
        }]
        assert pandey["profile_url"].endswith("/physician/dr-hemant-pandey-xsjwm")

    def test_the_observation_points_at_the_DOCTOR_not_at_the_index(self, gatherer):
        """A directory row's numbers are PER ROW — "Rated 3.8 out of 5 … from 88
        ratings" sits inside Pandey's block and is his. So "listing page"
        described where we read it, never whom it was about, and recording the
        index URL made a card say "doctor.webmd.com — listing page" and link
        forty doctors deep while the link to the one doctor it IS about sat
        unused in the same parsed row."""
        from unittest.mock import patch as _patch
        from utils.provenance import url_page_kind
        with _patch.object(gatherer, "_extract_page_shard"):
            out = gatherer._extract_provider_data(
                [_page(HG_LISTING_URL, HG_LISTING)], "Neurology", "Chandler, AZ")
        pandey = next(p for p in out if p["name"] == "Dr. Hemant Pandey, MD")
        observation = pandey["review_observations"][0]

        assert url_page_kind(observation["source_url"]) == "profile"
        assert url_page_kind(observation["read_from_url"]) == "listing"

    def test_a_row_with_no_profile_link_keeps_the_index_url(self, gatherer):
        """About half of vitals' rows carry no link. Falling back to the index
        is right — an observation with no URL at all cannot be collapsed per
        domain, counted as a platform, or clicked."""
        from unittest.mock import patch as _patch
        with _patch.object(gatherer, "_extract_page_shard"):
            out = gatherer._extract_provider_data(
                [_page(VITALS_LISTING_URL, VITALS_LISTING)], "Neurology", "Mesa, AZ")
        woods = next(p for p in out if p["name"] == "Dr. Brandon Craig Woods")

        assert woods["review_observations"][0]["source_url"]

    def test_an_off_specialty_row_is_dropped(self, gatherer):
        """A vitals neurology directory lists a plastic surgeon at 4.7/72, which
        would outrank most of the real neurologists."""
        from unittest.mock import patch as _patch
        with _patch.object(gatherer, "_extract_page_shard"):
            out = gatherer._extract_provider_data(
                [_page(VITALS_LISTING_URL, VITALS_LISTING)], "Neurology", "Chandler, AZ")
        assert "Dr. Edgardo D Zavala-Alarcon, MD" not in {p["name"] for p in out}
        assert "Dr. Brandon Craig Woods" in {p["name"] for p in out}

    def test_an_adjacent_specialty_label_is_KEPT(self, gatherer):
        """Portals file one doctor under adjacent labels. Rejecting on a label
        mismatch alone is the failure that made the enrichment query drop its
        specialty term — "Vascular Neurology" is not a different doctor."""
        from agents.data_gatherer import _specialty_is_compatible
        assert _specialty_is_compatible("Vascular Neurology", "Neurology")
        assert _specialty_is_compatible(None, "Neurology")
        assert not _specialty_is_compatible("Plastic Surgery", "Neurology")
        assert not _specialty_is_compatible("Psychology", "Neurology")

    def test_a_different_DISCIPLINE_is_rejected_however_close_the_stem(self, gatherer):
        """A Neurology search returned "Neurological Surgery" at ranks 1 and 4
        of a live run, because the stem rule only ever saw that both words
        begin "neurol". A neurosurgeon is not a neurologist — different
        training, different appointment — and the critic ended up writing the
        filter's job into a ranking caveat instead ("the list blends brain/spine
        surgeons with general neurologists")."""
        from agents.data_gatherer import _specialty_is_compatible
        assert not _specialty_is_compatible("Neurological Surgery", "Neurology")
        assert not _specialty_is_compatible("Neurosurgery", "Neurology")

    def test_the_discipline_rule_is_symmetric(self, gatherer):
        """Searching a surgical specialty must ADMIT surgeons and reject the
        physicians — a one-way rule would just move the complaint."""
        from agents.data_gatherer import _specialty_is_compatible
        assert _specialty_is_compatible("Neurological Surgery", "General Surgery")
        assert not _specialty_is_compatible("Neurology", "General Surgery")

    def test_a_sub_specialty_whose_head_is_not_a_discipline_stays_lenient(self, gatherer):
        """"Sleep Medicine" is a label portals file neurologists under, and its
        head — "medicine" — is deliberately absent from the discipline list, so
        it falls through to the lenient check rather than being rejected as a
        different field."""
        from agents.data_gatherer import _discipline_head
        assert _discipline_head("Sleep Medicine") is None
        assert _discipline_head("Vascular Neurology") == "neurology"
        assert _discipline_head("Neurological Surgery") == "surgery"


# ---- wiring: the profile parser inside the ENRICHMENT pass ----------------


def _llm(payload):
    """An Anthropic response carrying `payload` as its JSON body."""
    from unittest.mock import MagicMock
    import json as _json
    response = MagicMock()
    response.content = [MagicMock(text=_json.dumps(payload))]
    response.stop_reason = "end_turn"
    response.usage = MagicMock(input_tokens=0, output_tokens=0)
    return response


_EMPTY_LLM = {
    "review_summary": "No reviews available", "review_sentiment": "unknown",
    "review_count": None, "rating": None, "review_source_url": None,
    "review_observations": [], "insurance_accepted": [], "insurance_source_url": None,
    "years_experience": None, "phone": None, "address": None,
}


class TestProfileParserWiringInEnrichment:
    """Composed on `_extract_review_data_only`, never on `parse_profile` alone.

    A parser-only test stays green if the call site is deleted and every page
    goes back to the model — which is the regression that matters. `parse_profile`
    shipped built, unit-tested and documented in one round with ZERO call sites,
    and nothing in the suite noticed.
    """

    def test_a_parsed_profile_becomes_an_observation(self, gatherer):
        gatherer.anthropic_client.messages.create.return_value = _llm(_EMPTY_LLM)

        out = gatherer._extract_review_data_only(
            [_page(VITALS_PROFILE_URL, VITALS_PROFILE_HEADER)],
            "Dr. Marianne L De Lima, MD", "Neurology", "Chandler, AZ")

        assert out["review_observations"] == [{
            "source_url": VITALS_PROFILE_URL, "rating": 5.0, "review_count": 9,
            "page_provider_name": "Dr. Marianne L De Lima, MD",
            "extraction_source": "profile_parser",
        }]

    def test_the_parser_overrides_the_model_on_the_same_page(self, gatherer):
        """Both read the same page. The parser's answer is reproducible; a
        3,000-char anchor-window excerpt scored by the cheapest model is not —
        it moved a provider four ranks between two runs of one search."""
        gatherer.anthropic_client.messages.create.return_value = _llm(dict(
            _EMPTY_LLM, review_observations=[{
                "source_url": VITALS_PROFILE_URL, "rating": 4.0, "review_count": 244,
                "page_provider_name": "Dr. Marianne L De Lima, MD"}]))

        out = gatherer._extract_review_data_only(
            [_page(VITALS_PROFILE_URL, VITALS_PROFILE_HEADER)],
            "Dr. Marianne L De Lima, MD", "Neurology", "Chandler, AZ")

        assert len(out["review_observations"]) == 1
        assert (out["review_observations"][0]["rating"],
                out["review_observations"][0]["review_count"]) == (5.0, 9)

    def test_a_page_the_parser_read_owns_BOTH_halves_of_its_pair(self, gatherer):
        """healthgrades template B genuinely states no total — its count comes
        from the listing. Splicing the model's count onto the parser's rating
        manufactures a pair neither source stated, and here the model's number
        is one it read out of an inlined SVG's path coordinates."""
        gatherer.anthropic_client.messages.create.return_value = _llm(dict(
            _EMPTY_LLM, review_observations=[{
                "source_url": HG_PROFILE_URL, "rating": 3.8, "review_count": 88,
                "page_provider_name": "Dr. Hemant Pandey, MD"}]))

        out = gatherer._extract_review_data_only(
            [_page(HG_PROFILE_URL, HG_PROFILE_TEMPLATE_B)],
            "Dr. Hemant Pandey, MD", "Neurology", "Chandler, AZ")

        assert out["review_observations"][0]["rating"] == 3.8
        assert out["review_observations"][0]["review_count"] is None

    def test_a_page_the_parser_cannot_read_keeps_the_model_s_answer(self, gatherer):
        """A markup change must degrade to the previous behaviour, not empty the
        pool. The fallback is per FIELD and per PAGE both — this page's numbers
        survive even though the page beside it parsed cleanly."""
        unreadable = _page("https://www.vitals.com/doctors/someone-else-zz9",
                           "# Dr. Someone Else\nnothing this parser recognises")
        gatherer.anthropic_client.messages.create.return_value = _llm(dict(
            _EMPTY_LLM, review_observations=[{
                "source_url": unreadable["url"], "rating": 4.2, "review_count": 30,
                "page_provider_name": "Dr. Someone Else"}]))

        out = gatherer._extract_review_data_only(
            [unreadable, _page(VITALS_PROFILE_URL, VITALS_PROFILE_HEADER)],
            "Dr. Marianne L De Lima, MD", "Neurology", "Chandler, AZ")

        by_url = {o["source_url"]: o for o in out["review_observations"]}
        assert by_url[unreadable["url"]]["rating"] == 4.2
        assert by_url[VITALS_PROFILE_URL]["rating"] == 5.0

    def test_a_LISTING_page_is_never_read_by_the_PROFILE_parser(self, gatherer):
        """A directory index names forty-eight doctors, and the profile parser
        DOES find something in one — it reads the first entry's address with the
        entry above it's tenure glued on:

            "28 Years Exp erience 655 S Dobson Rd Ste 103 Bldg A, Chandler, AZ, 85224"

        Whoever we were enriching would be given that. The listing has its own
        parser, which splits at the entry heading precisely so a row's fields
        stay with the row; these patterns assume one page, one doctor. Note the
        H1 identity gate cannot save this — a listing has no `# ` heading, so
        there is no name to disagree with."""
        gatherer.anthropic_client.messages.create.return_value = _llm(_EMPTY_LLM)

        out = gatherer._extract_review_data_only(
            [_page(WEBMD_LISTING_URL, WEBMD_LISTING)],
            "Dr. Hemant Pandey, MD", "Neurology", "Chandler, AZ")

        assert out["review_observations"] == []
        assert out["address"] is None
        assert out["years_experience"] is None

    def test_tenure_is_refused_when_the_page_s_H1_is_a_different_doctor(self, gatherer):
        """The enrichment search is name + city and returns OTHER doctors'
        profiles. Tenure and address are page-level facts with no `source_url`
        for the downstream identity check to bind to, so this is the only place
        that check can happen."""
        gatherer.anthropic_client.messages.create.return_value = _llm(_EMPTY_LLM)

        out = gatherer._extract_review_data_only(
            [_page(WEBMD_PROFILE_URL, WEBMD_PROFILE_HEADER)],
            "Dr. Marianne L De Lima, MD", "Neurology", "Chandler, AZ")

        assert out["years_experience"] is None
        assert out["address"] is None
        # ...but the observation still stands: it carries a source_url, so the
        # merge's own identity check is what decides its fate.
        assert out["review_observations"][0]["rating"] == 4.5

    def test_tenure_is_taken_when_the_page_is_this_provider(self, gatherer):
        gatherer.anthropic_client.messages.create.return_value = _llm(_EMPTY_LLM)

        out = gatherer._extract_review_data_only(
            [_page(WEBMD_PROFILE_URL, WEBMD_PROFILE_HEADER)],
            "Dr. Andrea Hyeyong An, MD", "Neurology", "Chandler, AZ")

        assert out["years_experience"] == 26
        assert out["address"] == "2201 W Fairview St Ste 1, Chandler, AZ 85224"

    def test_the_parsers_survive_a_total_model_failure(self, gatherer):
        """The parse runs before the request and outside its try. An API error
        used to cost the provider every number on every page fetched for them —
        pages we already hold in memory and can read without a model."""
        gatherer.anthropic_client.messages.create.side_effect = RuntimeError("api down")

        out = gatherer._extract_review_data_only(
            [_page(VITALS_PROFILE_URL, VITALS_PROFILE_HEADER)],
            "Dr. Marianne L De Lima, MD", "Neurology", "Chandler, AZ")

        assert out["review_observations"][0]["review_count"] == 9
        assert out["years_experience"] == 14

    def test_enrichment_sources_records_which_path_produced_each_row(self, gatherer):
        """A parser gap and an excerpt gap look identical on a finished card and
        have different fixes: one is markup to re-read off a fetched page, the
        other is a window that missed the header."""
        from agents.data_gatherer import _annotate_source_yields

        rows = _annotate_source_yields(
            [{"url": VITALS_PROFILE_URL, "kind": "profile", "raw_chars": 8459},
             {"url": "https://doctor.webmd.com/doctor/x-overview", "kind": "profile",
              "raw_chars": 7013}],
            [{"source_url": VITALS_PROFILE_URL, "rating": 5.0, "review_count": 9,
              "extraction_source": "profile_parser"},
             {"source_url": "https://doctor.webmd.com/doctor/x-overview",
              "rating": 4.2, "review_count": 30}],
        )

        assert rows[0]["yielded"]["via"] == "profile_parser"
        assert rows[1]["yielded"]["via"] == "llm"


# ---- three defects a live run surfaced, 2026-07-31 ------------------------


class TestSearchRadius:
    """A platform's city page is a radius CENTRE, not a filter, so a Chandler
    search returns Gilbert, Mesa and Phoenix entries. Nothing bounded that, and
    distance alone could not: location's realized span across a real pool is
    under one point, so a provider 40 miles out lost a research-budget slot by
    a rounding error rather than being excluded."""

    @staticmethod
    def _p(name, miles):
        return {"name": name, "computed_distance_miles": miles}

    def test_beyond_the_radius_is_dropped(self):
        from agents.data_gatherer import _split_by_radius
        near, far = _split_by_radius(
            [self._p("Dr. Near", 4.5), self._p("Dr. Far", 41.0)], 25)
        assert [p["name"] for p in near] == ["Dr. Near"]
        assert [p["name"] for p in far] == ["Dr. Far"]

    def test_a_NEARER_provider_in_another_city_survives(self):
        """The reason this is a radius and not a city match. On the run that
        prompted it, Gilbert at 4.5 mi was closer to the searched Chandler ZIP
        than two of that run's own Chandler results at 8.0 mi — "same city
        only" would have dropped the closest provider on the page."""
        from agents.data_gatherer import _split_by_radius
        near, far = _split_by_radius(
            [self._p("Dr. Gilbert", 4.5), self._p("Dr. Chandler", 8.0)], 25)
        assert len(near) == 2 and far == []

    def test_an_unknown_distance_never_drops_anyone(self):
        """That is our geocoding coverage, not their location — the same rule
        the scorer follows when it declines to penalise missing data."""
        from agents.data_gatherer import _split_by_radius
        near, far = _split_by_radius(
            [self._p("Dr. Unplaced", None), {"name": "Dr. NoField"}], 25)
        assert len(near) == 2 and far == []

    def test_exactly_on_the_radius_is_kept(self):
        from agents.data_gatherer import _split_by_radius
        near, _ = _split_by_radius([self._p("Dr. Edge", 25.0)], 25)
        assert len(near) == 1


class TestHeadlineAttribution:
    """The headline is the ONE source a card attributes to the doctor, and a
    "Best Neurologists in Chandler" index is not about any one of them."""

    @staticmethod
    def _obs(url, rating, count):
        return {"source_url": url, "rating": rating, "review_count": count}

    PROFILE = "https://doctor.webmd.com/doctor/jonathan-hodgson-abc-overview"
    LISTING = "https://www.healthgrades.com/neurology-directory/az-arizona/chandler"

    def test_a_listing_with_more_reviews_does_not_take_the_headline(self):
        """The exact 2026-07-31 card: "Best single source: healthgrades.com —
        listing page 4.6/5 (33 reviews)" for a provider whose own webmd profile
        stated 5.0 over 22. Volume decided, and volume is the wrong question
        when one of the two sources is not about this person."""
        from agents.data_gatherer import _select_review_observation
        headline, _ = _select_review_observation([
            self._obs(self.LISTING, 4.6, 33),
            self._obs(self.PROFILE, 5.0, 22),
        ])
        assert headline["source_url"] == self.PROFILE

    def test_volume_still_decides_between_two_attributable_sources(self):
        from agents.data_gatherer import _select_review_observation
        other = "https://www.vitals.com/doctors/jonathan-hodgson-x1"
        headline, _ = _select_review_observation([
            self._obs(self.PROFILE, 5.0, 22),
            self._obs(other, 4.8, 61),
        ])
        assert headline["source_url"] == other

    def test_a_listing_still_headlines_when_it_is_all_we_have(self):
        """The listing pair is not discarded — it stays in "Across platforms",
        it feeds the blend, and it counts toward the platform pair count. It
        just loses to an attributable source when one exists."""
        from agents.data_gatherer import _select_review_observation
        headline, observations = _select_review_observation([
            self._obs(self.LISTING, 4.6, 33)])
        assert headline["source_url"] == self.LISTING
        assert len(observations) == 1

    def test_the_listing_pair_survives_alongside_the_headline(self):
        from agents.data_gatherer import _select_review_observation
        _, observations = _select_review_observation([
            self._obs(self.LISTING, 4.6, 33),
            self._obs(self.PROFILE, 5.0, 22),
        ])
        assert {o["source_url"] for o in observations} == {self.LISTING, self.PROFILE}


class TestColleagueIdentityVeto:
    """Two neurosurgeons at neighbouring addresses on one street shared numbers
    on a live run — both cards read "doctor.webmd.com 5.0/5 (45 reviews)" and
    "vitals.com (46 reviews)". The enrichment query is name + city, so a
    colleague at the same practice ranks for it; the page-name check degrades
    to ACCEPT when nothing was transcribed, and nothing was."""

    def test_the_slug_veto_rejects_a_colleague(self, gatherer):
        provider = {"name": "Dr. Frederick Francis Marciano, MD"}
        gatherer._merge_review_data(provider, {
            "review_summary": "No reviews available", "review_sentiment": "unknown",
            "review_observations": [
                {"source_url": "https://doctor.webmd.com/doctor/luis-tumialan-x1-overview",
                 "rating": 5.0, "review_count": 45},
            ],
        })
        assert not provider.get("review_observations")

    def test_the_doctors_own_profile_is_kept(self, gatherer):
        provider = {"name": "Dr. Luis Manuel Tumialan, MD"}
        gatherer._merge_review_data(provider, {
            "review_summary": "No reviews available", "review_sentiment": "unknown",
            "review_observations": [
                {"source_url": "https://doctor.webmd.com/doctor/luis-tumialan-x1-overview",
                 "rating": 5.0, "review_count": 45},
            ],
        })
        assert provider["review_observations"][0]["review_count"] == 45

    def test_an_opaque_slug_cannot_veto_anything(self, gatherer):
        """A slug carrying no personal name is not evidence of a different
        person. Vetoing on it would discard real observations wholesale."""
        from utils.listing_parser import slug_contradicts_name
        assert not slug_contradicts_name(
            "https://doctor.webmd.com/doctor/12345-overview", "Dr. Anybody")
        assert not slug_contradicts_name(
            "https://www.vitals.com/doctors/dr-a", "Dr. Andrea An, MD")

    def test_a_shared_surname_alone_is_not_a_contradiction(self, gatherer):
        """`slug_agrees_with_name` asks "is this evidence FOR the pairing" and
        divides by the name's tokens, so "/doctors/Dr_Khan" scores 0.33 against
        a three-token name and fails. That is the right answer as evidence and
        the wrong one as grounds for throwing an observation away."""
        from utils.listing_parser import slug_agrees_with_name, slug_contradicts_name
        url = "https://www.vitals.com/doctors/Dr_Khan.html"
        assert not slug_agrees_with_name(url, "Dr. Mohammad B. Khan")
        assert not slug_contradicts_name(url, "Dr. Mohammad B. Khan")

    def test_a_file_extension_does_not_eat_the_last_token(self, gatherer):
        """`normalize_name_tokens` strips the dot rather than splitting on it,
        so "Dr_Khan.html" tokenized to the single token "khanhtml" — matching
        nothing, on a URL that plainly carries the surname."""
        from utils.listing_parser import _slug_tokens
        assert "khan" in _slug_tokens("https://www.vitals.com/doctors/Dr_Khan.html")

    def test_a_LISTING_url_is_never_vetoed_by_its_slug(self, gatherer):
        """A directory index names forty doctors and its slug describes the
        PAGE, so it agrees with nobody. healthgrades' real one —
        "/find-a-doctor/arizona/best-doctors-for-headache-in-chandler" — carries
        five personal-looking tokens and none of them is anyone's name, so
        without the profile-only gate every listing pair on the card would be
        vetoed."""
        from utils.listing_parser import slug_contradicts_name
        listing = ("https://www.healthgrades.com/find-a-doctor/arizona/"
                   "best-doctors-for-headache-in-chandler")
        # The slug DOES contradict — which is exactly why the caller must not
        # ask it about a listing.
        assert slug_contradicts_name(listing, "Dr. Frederick Francis Marciano, MD")

        provider = {"name": "Dr. Frederick Francis Marciano, MD"}
        gatherer._merge_review_data(provider, {
            "review_summary": "No reviews available", "review_sentiment": "unknown",
            "review_observations": [
                {"source_url": listing, "rating": 4.7, "review_count": 74},
            ],
        })
        assert provider["review_observations"][0]["review_count"] == 74

    def test_url_furniture_is_not_a_personal_token(self, gatherer):
        """webmd ends every profile slug with "-overview". Counting that as a
        name makes "/doctor/smith-overview" look like a two-name slug, so it
        clears the two-token bar and vetoes any doctor not called Smith — on
        the strength of a word the platform puts on every URL it serves."""
        from utils.listing_parser import _slug_tokens, slug_contradicts_name
        url = "https://doctor.webmd.com/doctor/smith-overview"
        assert "overview" in _slug_tokens(url)          # it IS in the slug...
        assert not slug_contradicts_name(url, "Dr. Jane Doe, MD")   # ...and votes on nothing

    def test_a_single_token_slug_cannot_veto(self, gatherer):
        """One personal token is the ambiguous case — it is equally "surname
        only" and "opaque id that happens to be alphabetic". A real person-slug
        carries a given name and a family name, so two is the bar; at one, an
        id-only slug vetoes a real observation."""
        from utils.listing_parser import slug_contradicts_name
        assert not slug_contradicts_name(
            "https://www.vitals.com/doctors/tumialan", "Dr. Frederick Marciano, MD")
        assert slug_contradicts_name(
            "https://www.vitals.com/doctors/luis-tumialan", "Dr. Frederick Marciano, MD")


class TestRadiusWiring:
    """Driven through `gather_providers`, because a helper-only test leaves the
    call site deletable with the suite still green — and the filter runs inline
    after location evidence is attached, which is exactly the kind of line that
    gets "simplified" away."""

    def test_gather_drops_a_provider_beyond_the_radius(self, gatherer):
        from unittest.mock import patch as _patch

        extracted = [{"name": "Dr. Near"}, {"name": "Dr. Far"}]
        distances = {"Dr. Near": 4.5, "Dr. Far": 41.0}

        def fake_evidence(provider, user_location):
            provider["computed_distance_miles"] = distances[provider["name"]]
            provider["location_match"] = "same_city"
            provider["distance_precision"] = "zip"

        with _patch.object(gatherer, "_search_providers", return_value=[{"url": "u"}]), \
             _patch.object(gatherer, "_extract_provider_data", return_value=extracted), \
             _patch.object(gatherer, "_attach_location_evidence", side_effect=fake_evidence):
            result = gatherer.gather_providers(
                specialty="Neurology", location="Chandler, AZ 85249", enrich=False)

        assert [p["name"] for p in result["providers"]] == ["Dr. Near"]
        assert result["search_metadata"]["radius_dropped"] == 1
        assert result["search_metadata"]["radius_miles"] == gatherer.config.DEFAULT_SEARCH_RADIUS


class TestSameNameDifferentState:
    """healthgrades holds a "Dr. Nicole Simpkins, MD" in San Antonio, TX —
    Clinical Neurophysiology, 5.0 over 12 reviews. The provider on the card is
    Dr. Nicole Alyce Simpkins in Chandler, AZ. Every guard passes it: the names
    overlap, the slug agrees, and the enrichment query returns her because it
    searches the NAME. Nothing looked at where the page said she practises."""

    HG = "https://www.healthgrades.com/physician/dr-nicole-simpkins-abc123"
    SAN_ANTONIO = """# Dr. Nicole Simpkins, MD
## Clinical Neurophysiology | 20+ years of experience
5 Star Rating
Based on 12 reviews
### Practice
2915 W Bitters Rd Ste 201 ·San Antonio, TX 78248
"""

    def test_every_existing_guard_accepts_the_stranger(self):
        """Why this needed a new check rather than a tighter old one."""
        from utils.listing_parser import slug_agrees_with_name, slug_contradicts_name
        ours = "Dr. Nicole Alyce Simpkins, MD"
        assert slug_agrees_with_name(self.HG, ours)
        assert not slug_contradicts_name(self.HG, ours)
        assert parse_profile(self.HG, self.SAN_ANTONIO)["page_provider_name"] == \
            "Dr. Nicole Simpkins, MD"

    def test_a_different_state_rejects_the_page(self, gatherer):
        gatherer.anthropic_client.messages.create.return_value = _llm(_EMPTY_LLM)

        out = gatherer._extract_review_data_only(
            [_page(self.HG, self.SAN_ANTONIO)],
            "Dr. Nicole Alyce Simpkins, MD", "Neurology",
            "2905 W Warner Rd Ste 1, Chandler, AZ 85224")

        assert out["review_observations"] == []
        assert out["address"] is None       # ...and never overwrites her address,
        assert out["years_experience"] is None   # which would re-score the distance

    def test_the_same_state_is_kept(self, gatherer):
        gatherer.anthropic_client.messages.create.return_value = _llm(_EMPTY_LLM)
        in_state = self.SAN_ANTONIO.replace(
            "2915 W Bitters Rd Ste 201 ·San Antonio, TX 78248",
            "2905 W Warner Rd Ste 1 ·Chandler, AZ 85224")

        out = gatherer._extract_review_data_only(
            [_page(self.HG, in_state)], "Dr. Nicole Alyce Simpkins, MD",
            "Neurology", "2905 W Warner Rd Ste 1, Chandler, AZ 85224")

        assert out["review_observations"][0]["review_count"] == 12

    def test_a_page_stating_no_address_is_never_penalised(self, gatherer):
        """Missing data is not disagreement — the rule the scorer follows."""
        gatherer.anthropic_client.messages.create.return_value = _llm(_EMPTY_LLM)
        no_address = "# Dr. Nicole Simpkins, MD\n5 Star Rating\nBased on 12 reviews\n"

        out = gatherer._extract_review_data_only(
            [_page(self.HG, no_address)], "Dr. Nicole Alyce Simpkins, MD",
            "Neurology", "2905 W Warner Rd Ste 1, Chandler, AZ 85224")

        assert out["review_observations"][0]["review_count"] == 12

    def test_an_unknown_provider_location_never_rejects(self, gatherer):
        """Discovery does not always resolve an address. A provider we could not
        place must not lose every platform because of it."""
        gatherer.anthropic_client.messages.create.return_value = _llm(_EMPTY_LLM)

        out = gatherer._extract_review_data_only(
            [_page(self.HG, self.SAN_ANTONIO)],
            "Dr. Nicole Alyce Simpkins, MD", "Neurology", "")

        assert out["review_observations"][0]["review_count"] == 12

    def test_the_helper_only_fires_on_positive_disagreement(self):
        from agents.data_gatherer import _states_disagree
        assert _states_disagree("Chandler, AZ 85224", "San Antonio, TX 78248")
        assert not _states_disagree("Chandler, AZ 85224", "Mesa, AZ 85201")
        assert not _states_disagree("", "Chandler, AZ 85224")
        assert not _states_disagree("Chandler, AZ 85224", "")
        assert not _states_disagree(None, None)


class TestAddressProvenance:
    """Two providers on one live run carried an IDENTICAL street address and an
    identical distance. The parser path was gated on name and state; the path
    the MODEL feeds was not gated at all — and its input includes practice and
    group pages that name several doctors, so one address can land on all of
    them."""

    OURS = "Phoenix, AZ"

    PROFILE = "https://www.healthgrades.com/physician/dr-harvinder-kumar-x1"

    @staticmethod
    def _backfill(gatherer, address, held="Phoenix, AZ",
                  source="https://www.healthgrades.com/physician/dr-harvinder-kumar-x1"):
        """Drive the real backfill branch without the geo dataset.

        `resolution_level` needs `data/us_zip_coords.csv.gz`, which is a Git-LFS
        pointer in a clone that has not pulled — it returns None there, the
        backfill never fires, and a test asserting "the address was refused"
        passes whether or not the guard exists. Stubbed to the answer the real
        dataset gives for these two inputs, so the branch under test actually
        runs."""
        from unittest.mock import patch as _patch
        provider = {"name": "Dr. Harvinder Kumar", "location": held}
        levels = {held: "city", address: "zip"}
        with _patch("agents.data_gatherer.resolution_level",
                    side_effect=lambda loc: levels.get(loc)):
            gatherer._merge_review_data(provider, {
                "review_summary": "No reviews available",
                "review_sentiment": "unknown",
                "review_observations": [], "address": address,
                "address_source_url": source,
            })
        return provider

    def test_the_model_supplied_address_is_state_checked(self, gatherer):
        provider = self._backfill(
            gatherer, "2915 W Bitters Rd Ste 201, San Antonio, TX 78248")

        assert provider["location"] == self.OURS
        assert "location_source" not in provider

    def test_an_in_state_upgrade_still_lands(self, gatherer):
        """The backfill exists to turn "City, ST" into a ZIP-resolvable address,
        and that must keep working — the state check is a veto, not a gate."""
        provider = self._backfill(
            gatherer, "1450 S Dobson Rd Ste B122, Mesa, AZ 85202")

        assert provider["location"] == "1450 S Dobson Rd Ste B122, Mesa, AZ 85202"

    def test_the_backfill_records_where_the_address_came_from(self, gatherer):
        """The question "where did this address come from" had no answer on any
        surface — not the card, not the panel, not the log. It does now, and the
        two paths are named apart because the model's address carries no URL."""
        provider = self._backfill(
            gatherer, "1450 S Dobson Rd Ste B122, Mesa, AZ 85202")

        assert provider["location_source"] == f"enrichment_model:{self.PROFILE}"

    def test_a_parsed_profile_address_names_its_page(self, gatherer):
        url = "https://www.healthgrades.com/physician/dr-harvinder-kumar-x1"
        page = ("# Dr. Harvinder Kumar\n4.9 Star Rating\nBased on 246 reviews\n"
                "### Practice\n1450 S Dobson Rd Ste B122 ·Mesa, AZ 85202\n")
        gatherer.anthropic_client.messages.create.return_value = _llm(_EMPTY_LLM)

        out = gatherer._extract_review_data_only(
            [_page(url, page)], "Dr. Harvinder Kumar", "Neurology", "Mesa, AZ")

        assert out["address_source"] == f"profile_parser:{url}"

    def test_a_discovery_row_names_the_listing_it_came_from(self, gatherer):
        from unittest.mock import patch as _patch
        with _patch.object(gatherer, "_extract_page_shard"):
            out = gatherer._extract_provider_data(
                [_page(HG_LISTING_URL, HG_LISTING)], "Neurology", "Chandler, AZ")
        pandey = next(p for p in out if p["name"] == "Dr. Hemant Pandey, MD")

        assert pandey["location_source"] == f"listing_parser:{HG_LISTING_URL}"


class TestChunkParse:
    """A page returns TWO texts and only one was ever parsed.

    Dr. Hagevik's healthgrades profile came back at 5,756 chars and yielded
    `{"rating": 4.8, "review_count": null, "via": "profile_parser"}` — while the
    page in a browser reads "4.8 Star Rating / Based on 260 reviews". His pair
    was dropped from the blend, and the card said "2 platforms" beside three
    named ones.

    One URL, two renderings — verified 2026-08-05 against saved bytes: the
    API's copy of his page is the LIKELIHOOD template (`Likelihood of
    recommending … is 4.776923 out of 5`, no count in ANY text, byte-identical
    across four search variants and extract at both depths over three days);
    the star form with the 260 is the browser's fresher rendering of the same
    URL. "The chunks carried his 260" was hypothesis, not observation — run
    1's chunks were never saved. So the star fixture below is a SYNTHETIC
    star-template page: the shape his URL parses to once the vendor index
    refreshes, and the shape the chunk-recovery rule exists for — NOT a
    transcript of his currently-fetched bytes, which the template-marker
    tests cover with the real likelihood form."""

    URL = "https://www.healthgrades.com/physician/dr-andre-hagevik-x7qq6"
    # SYNTHETIC star-template thin page — an earlier comment here claimed the
    # page "fetched fresh" reads star, which a screenshot suggested and the
    # saved API bytes refute (likelihood, count-free; see the class
    # docstring). The distinction is load-bearing either way: a likelihood
    # page REFUSES a chunk count by design (its payload carries no total, so
    # a chunk count is a neighbour's), so these recovery tests MUST run on a
    # star body — presenting that body as his fetched bytes was the error,
    # not using it.
    RAW_NO_COUNT = """# Dr. Andre Hagevik, MD
## Other | 25+ years of experience
4.8 Star Rating
Review Save
"""
    CHUNKS_WITH_COUNT = "4.8 Star Rating Based on 260 reviews Dr. Andre Hagevik, MD"

    @staticmethod
    def _result(url, raw, content=""):
        return {"url": url, "raw_content": raw, "content": content}

    def test_the_chunks_supply_a_count_the_full_page_lacked(self):
        from agents.data_gatherer import _parse_both_texts
        got = _parse_both_texts(
            self.URL, self._result(self.URL, self.RAW_NO_COUNT, self.CHUNKS_WITH_COUNT))

        assert (got["rating"], got["review_count"]) == (4.8, 260)

    def test_without_the_chunks_the_count_is_still_absent(self):
        """The behaviour this replaces, pinned so the gain is attributable."""
        from agents.data_gatherer import _parse_both_texts
        got = _parse_both_texts(self.URL, self._result(self.URL, self.RAW_NO_COUNT))

        assert got["rating"] == 4.8
        assert got.get("review_count") is None

    def test_a_count_the_chunks_alone_state_completes_the_pair(self):
        """REVERSED on evidence 2026-08-04, deliberately.

        This asserted that a chunk count is refused unless the chunks RESTATE
        the rating — written as too-tight-on-purpose, with a note that
        loosening should follow a run showing it cost real coverage. It did:
        Dr. Hagevik's healthgrades profile returned a thin body stating
        `4.8 Star Rating` and no total while the chunks for the same url
        carried `Based on 260 reviews`, so the count was dropped and the card
        read "2 platforms" beside three named ones.

        The full page's rating is the subject anchor. What still protects the
        count is a chunk parse that DISAGREES — see the test below, which is
        the half of this rule that was doing the real work.
        """
        from agents.data_gatherer import _parse_both_texts
        got = _parse_both_texts(self.URL, self._result(
            self.URL, self.RAW_NO_COUNT, "Based on 512 reviews"))

        assert got["rating"] == 4.8
        assert got.get("review_count") == 512

    def test_a_count_with_no_rating_on_either_text_is_refused(self):
        """Nothing names the subject, so the number would attach to whoever the
        page happens to be about."""
        from agents.data_gatherer import _parse_both_texts
        got = _parse_both_texts(
            self.URL, self._result(self.URL, "no numbers here", "Based on 512 reviews"))

        assert got.get("review_count") is None

    def test_a_disagreeing_chunk_parse_is_discarded_whole(self):
        from agents.data_gatherer import _parse_both_texts
        got = _parse_both_texts(self.URL, self._result(
            self.URL, self.RAW_NO_COUNT,
            "5.0 Star Rating Based on 512 reviews 40 years of experience"))

        assert got["rating"] == 4.8
        assert got.get("review_count") is None
        # 25 is the FULL PAGE's own figure and survives; 40 was the chunks' and
        # does not. Discarding the chunk parse never discards the primary.
        assert got["years_experience"] == 25

    def test_a_disagreeing_parse_contributes_NOTHING_not_just_no_count(self):
        """The whole parse is discarded, not only the fields that clashed. Here
        the chunks disagree on the rating and are the ONLY source of a tenure —
        under a rule that dropped just the clashing field, a stranger's career
        length would still be merged in."""
        from agents.data_gatherer import _parse_both_texts
        raw = ("# Dr. Andre Hagevik, MD\n### Overall Patient Satisfaction\n"
               "Likelihood of recommending Dr. Hagevik to family and friends is 4.8 out of 5\n")
        got = _parse_both_texts(self.URL, self._result(
            self.URL, raw, "5.0 Star Rating Based on 512 reviews 40 years of experience"))

        assert got["rating"] == 4.8
        assert got.get("years_experience") is None
        assert got.get("review_count") is None

    def test_the_chunks_can_never_replace_the_page_H1(self):
        """`page_provider_name` is what the identity check binds on, and it is
        not one of the numeric fields the disagreement check compares — so
        merging by override rather than by gap-fill would let a name lifted out
        of a neighbouring chunk become the subject of this observation."""
        from agents.data_gatherer import _parse_both_texts
        raw = "# Dr. Andre Hagevik, MD\n4.8 Star Rating\n"
        got = _parse_both_texts(self.URL, self._result(
            self.URL, raw, "# Dr. Someone Else, MD\n4.8 Star Rating Based on 260 reviews"))

        assert got["page_provider_name"] == "Dr. Andre Hagevik, MD"
        assert got["review_count"] == 260      # the count still lands

    def test_the_full_page_parse_always_wins_a_shared_field(self):
        from agents.data_gatherer import _parse_both_texts
        raw = "# Dr. X\n4.1 Star Rating\nBased on 70 reviews\n"
        got = _parse_both_texts(self.URL, self._result(
            self.URL, raw, "4.1 Star Rating Based on 70 reviews"))

        assert (got["rating"], got["review_count"]) == (4.1, 70)

    def test_an_empty_content_field_changes_nothing(self):
        from agents.data_gatherer import _parse_both_texts
        assert _parse_both_texts(self.URL, self._result(self.URL, self.RAW_NO_COUNT, "")) == \
            _parse_both_texts(self.URL, self._result(self.URL, self.RAW_NO_COUNT))

    def test_it_is_wired_into_enrichment(self, gatherer):
        """Composed, because a helper-only test leaves the call site deletable —
        `parse_profile` itself shipped with zero call sites once already."""
        gatherer.anthropic_client.messages.create.return_value = _llm(_EMPTY_LLM)
        page = _page(self.URL, self.RAW_NO_COUNT)
        page["content"] = self.CHUNKS_WITH_COUNT

        out = gatherer._extract_review_data_only(
            [page], "Dr. Andre Hagevik, MD", "Neurology", "Sun City West, AZ")

        assert out["review_observations"][0]["review_count"] == 260


class TestUserChosenRadius:
    """The radius BOUNDS the pool; the "Nearby" weight ORDERS what survives.

    The weight cannot do the bounding job — a location score's realized span
    across a real pool is under one point, so a provider 40 miles out lost a
    research-budget slot by a rounding error rather than being excluded. The
    critic wrote the same finding in its own words on a live run: "providers 8
    miles away and providers under 4 miles are separated by only a small amount
    in the final ordering"."""

    @staticmethod
    def _run(gatherer, radius, distances):
        from unittest.mock import patch as _patch

        def fake_evidence(provider, user_location):
            provider["computed_distance_miles"] = distances[provider["name"]]
            provider["location_match"] = "same_city"
            provider["distance_precision"] = "zip"

        extracted = [{"name": name} for name in distances]
        with _patch.object(gatherer, "_search_providers", return_value=[{"url": "u"}]), \
             _patch.object(gatherer, "_extract_provider_data", return_value=extracted), \
             _patch.object(gatherer, "_attach_location_evidence", side_effect=fake_evidence):
            return gatherer.gather_providers(
                specialty="Neurology", location="Chandler, AZ 85249",
                enrich=False, radius_miles=radius)

    DISTANCES = {"Dr. Close": 2.7, "Dr. Mid": 11.4, "Dr. Far": 41.0}

    def test_ten_miles_keeps_only_the_nearest(self, gatherer):
        result = self._run(gatherer, 10, self.DISTANCES)
        assert [p["name"] for p in result["providers"]] == ["Dr. Close"]

    def test_fifty_miles_keeps_everyone(self, gatherer):
        result = self._run(gatherer, 50, self.DISTANCES)
        assert len(result["providers"]) == 3

    def test_the_chosen_radius_is_recorded_not_the_default(self, gatherer):
        result = self._run(gatherer, 10, self.DISTANCES)
        assert result["search_metadata"]["radius_miles"] == 10
        assert result["search_metadata"]["radius_dropped"] == 2

    def test_no_choice_falls_back_to_the_configured_default(self, gatherer):
        result = self._run(gatherer, None, self.DISTANCES)
        assert result["search_metadata"]["radius_miles"] == \
            gatherer.config.DEFAULT_SEARCH_RADIUS

    def test_the_ring_never_reaches_past_the_chosen_radius(self, gatherer):
        """Reaching further would import cities the radius bound then deletes —
        two searches and an extraction spent on rows that cannot survive."""
        from unittest.mock import patch as _patch
        with _patch("agents.data_gatherer.nearby_cities", return_value=[]) as ring, \
             _patch.object(gatherer, "_search_providers", return_value=[{"url": "u"}]), \
             _patch.object(gatherer, "_extract_provider_data", return_value=[]), \
             _patch.object(gatherer, "_attach_location_evidence"):
            gatherer.gather_providers(specialty="Neurology", location="Chandler, AZ",
                                      enrich=False, radius_miles=10)
        assert ring.call_args.args[1] == 10


class TestRadiusReachesTheGatherer:
    """The two wiring hops between the form and the filter. Each was GREEN under
    a revert until this class existed — the filter had tests, the path to it did
    not, which is the same gap that let `parse_profile` ship with zero call
    sites."""

    @staticmethod
    def _gather_with(preferences):
        from unittest.mock import patch as _patch
        from agents.orchestrator import ProviderMatchingOrchestrator

        with _patch("agents.orchestrator.DataGathererAgent"), \
             _patch("agents.orchestrator.PreferenceScorerAgent"), \
             _patch("agents.orchestrator.CriticValidatorAgent"), \
             _patch("agents.orchestrator.get_vector_store"):
            orchestrator = ProviderMatchingOrchestrator()
        orchestrator.data_gatherer.gather_providers.return_value = {"providers": []}

        orchestrator._gather_provider_data({
            "specialty": "Neurology", "location": "Chandler, AZ",
            "preferences": preferences, "error_messages": [], "execution_log": [],
        })
        return orchestrator.data_gatherer.gather_providers.call_args.kwargs

    def test_the_orchestrator_passes_the_chosen_radius(self):
        assert self._gather_with({"search_radius_miles": 10.0})["radius_miles"] == 10.0

    def test_a_search_with_no_radius_preference_passes_None(self):
        """Every non-UI caller — tests, the direct-agent path in the README —
        omits it, and must keep getting the configured default."""
        assert self._gather_with({})["radius_miles"] is None

    def test_the_form_labels_map_to_miles(self):
        """The control's own contract: what the user READS and what the filter
        RECEIVES must not drift apart.

        Asserted as a property rather than against a literal dict, so rewording
        a chip does not fail the suite while a genuine drift — a "10" chip
        wired to 25.0 — still does."""
        from app import SEARCH_RADIUS_OPTIONS, SEARCH_RADIUS_DEFAULT

        assert SEARCH_RADIUS_DEFAULT in SEARCH_RADIUS_OPTIONS
        for label, miles in SEARCH_RADIUS_OPTIONS.items():
            digits = "".join(c for c in label if c.isdigit())
            assert digits, f"chip {label!r} states no number"
            assert float(digits) == miles, f"chip {label!r} is wired to {miles}"

    def test_the_FORM_puts_the_radius_into_search_params(self):
        """Driven through `render_search_form`, because the constants above pass
        whether or not the dict carries the value — and the dict is what the
        orchestrator reads."""
        from unittest.mock import MagicMock, patch as _patch
        import app as app_module

        class Box(MagicMock):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class Form(MagicMock):
            # Round 29 retired st.form for st.container(border=True) + a
            # plain st.button; the stub mirrors that surface.
            def container(self, *args, **kwargs):
                return Box()

            def columns(self, spec, **kwargs):
                width = spec if isinstance(spec, int) else len(spec)
                return [Box() for _ in range(width)]

            def selectbox(self, label, options, **kwargs):
                # options[0] gives a REAL state and a real city in it — the
                # picker options come from the GeoNames dataset, so the
                # composed location passes the round-28 allowlist by
                # construction, exactly as it does live. The ZIP became a
                # selectbox too (round 29) and its options[0] is "" — no
                # ZIP, the simplest valid submission.
                return options[0]

            def segmented_control(self, label, options, **kwargs):
                # Matched on a prefix, not the exact string: the label has
                # been reworded before (round 30 moved the unit off it onto
                # the chips) and rewording must not silently turn this into
                # a test of the weight controls.
                return options[0] if label.startswith("Search within") else "Medium"

            def button(self, *args, **kwargs):
                return True

        with _patch.object(app_module, "st", Form()):
            params = app_module.render_search_form()

        first_label = next(iter(app_module.SEARCH_RADIUS_OPTIONS))
        assert params["preferences"]["search_radius_miles"] == \
            app_module.SEARCH_RADIUS_OPTIONS[first_label]

    def test_the_where_row_precedes_the_specialty_row(self):
        """Rounds 28–29 layout: WHERE (State / City / ZIP) is the first
        weighted row and what + how far the second — both inside the ONE
        bordered card that replaced st.form (round 29; dependent
        selectboxes and a form's batched state are mutually exclusive, and
        the two-box compromise read as disconnected). Rows bottom-aligned
        so dropdowns and the radius chips sit on one baseline.

        The column SPEC alone is not the layout — the card could create the
        row and then render the radius outside it. Widgets are therefore
        recorded against the column context they were rendered in."""
        from unittest.mock import MagicMock, patch as _patch
        import app as app_module

        specs, stack, placed = [], [], []

        class Box:
            """A plain class, NOT a MagicMock subclass.

            MagicMock configures its magic methods on the INSTANCE at
            construction, which shadows any `__enter__` defined on a subclass —
            so `with column:` would silently run MagicMock's version and this
            stack would stay empty, making every column assertion below vacuous.
            A column is only ever used as a context manager here, so a plain
            class is all it needs to be."""

            def __init__(self, tag=None):
                self._tag = tag

            def __enter__(self):
                stack.append(self._tag)
                return self

            def __exit__(self, *exc):
                stack.pop()
                return False

        class Rec(MagicMock):
            def container(self, *args, **kwargs):
                return Box()

            def subheader(self, *args, **kwargs):
                return None

            def columns(self, spec, **kwargs):
                specs.append((spec, kwargs.get("vertical_alignment")))
                width = spec if isinstance(spec, int) else len(spec)
                row = len(specs) - 1
                return [Box((row, index)) for index in range(width)]

            def selectbox(self, label, options, **kwargs):
                placed.append((stack[-1] if stack else None, label))
                return options[0]

            def segmented_control(self, label, options, **kwargs):
                placed.append((stack[-1] if stack else None, label))
                return options[0] if label.startswith("Search within") else "Medium"

            def button(self, *args, **kwargs):
                return True

        with _patch.object(app_module, "st", Rec()):
            app_module.render_search_form()

        # Equal thirds since round 31: one 3-column grid down the card —
        # State above Nearby, City above Ratings, ZIP above Search-within
        # above Experience. The old [1, 2, 1] gave the city extra room and
        # read as three unrelated widths.
        assert specs[0] == (3, "bottom")          # WHERE
        # [2, 1] since round 30: the 2/3 boundary puts the radius directly
        # above the third weight column (Experience), one visual column
        # line — [3, 2]'s 60% boundary aligned with nothing below it.
        assert specs[1] == ([2, 1], "bottom")     # what + how far

        by_column = {column: label for column, label in placed}
        assert by_column[(0, 0)] == "State"
        assert by_column[(0, 1)] == "City"
        assert by_column[(0, 2)].startswith("ZIP")
        assert by_column[(1, 0)] == "Medical Specialty"
        assert by_column[(1, 1)].startswith("Search within")

    def test_the_default_preserves_todays_behaviour(self):
        """An untouched form must search exactly as far as it did before the
        control existed, or every previous run becomes unreproducible."""
        from app import SEARCH_RADIUS_OPTIONS, SEARCH_RADIUS_DEFAULT
        from utils.config import get_config

        assert SEARCH_RADIUS_OPTIONS[SEARCH_RADIUS_DEFAULT] == \
            float(get_config().DEFAULT_SEARCH_RADIUS)


class TestLocationFalloffFollowsTheRadius:
    """Choosing 10 miles must change how providers RANK, not only who is in the
    pool. Measured on the 2.7–11.4 mile pool two live runs produced, the
    location dimension's weighted contribution span moves 5.80 -> 14.50 when the
    falloff tracks a 10-mile choice instead of sitting at 50."""

    POOL = [(11.4, 4.9, 246, 29), (8.0, 4.9, 169, 27), (2.7, 4.8, 82, 23)]

    @staticmethod
    def _scorer():
        from unittest.mock import MagicMock, patch as _patch
        from agents.preference_scorer import PreferenceScorerAgent
        with _patch.object(PreferenceScorerAgent, "_initialize_client", return_value=None):
            agent = PreferenceScorerAgent()
            agent.openai_client = MagicMock()
            return agent

    def _spans(self, radius):
        scorer = self._scorer()
        providers = [{"name": f"Dr. {i}", "computed_distance_miles": m,
                      "distance_precision": "zip", "location_match": "same_city",
                      "rating": r, "review_count": c, "blended_rating": r,
                      "blended_review_count": c, "blended_platform_count": 3,
                      "years_experience": y}
                     for i, (m, r, c, y) in enumerate(self.POOL)]
        prefs = {"location_weight": 1.5, "rating_weight": 1.5,
                 "experience_weight": 1.5}
        if radius is not None:
            prefs["search_radius_miles"] = radius
        out = scorer._calculate_base_scores(providers, prefs)
        scores = [p["score_breakdown"]["location"]["score"] for p in out]
        return max(scores) - min(scores)

    def test_a_tighter_radius_sharpens_the_distance_gradient(self):
        assert self._spans(10) > self._spans(25) * 2

    def test_a_wider_radius_flattens_it(self):
        assert self._spans(50) < self._spans(25)

    def test_the_default_scores_exactly_as_before(self):
        """An untouched form must produce the same ordering it always did, or
        every earlier run becomes unreproducible."""
        from utils.config import get_config
        assert self._spans(None) == self._spans(float(get_config().DEFAULT_SEARCH_RADIUS))

    def test_a_measured_provider_still_outranks_a_tiered_one(self):
        """The property the tier numbers exist for: an imputation must never
        out-score a measurement. It has to survive the rescaling."""
        scorer = self._scorer()
        providers = [
            {"name": "Dr. Measured", "computed_distance_miles": 4.0,
             "distance_precision": "zip", "location_match": "same_city"},
            {"name": "Dr. Tiered", "location_match": "same_city"},
        ]
        out = scorer._calculate_base_scores(providers, {
            "location_weight": 1.5, "rating_weight": 1.5,
            "experience_weight": 1.5, "search_radius_miles": 10.0})
        by_name = {p["name"]: p["score_breakdown"]["location"]["score"] for p in out}

        assert by_name["Dr. Measured"] > by_name["Dr. Tiered"]


class TestAddressMustBeTraceable:
    """The model reads SIX blocks, and they include group practice pages that
    state one address for several doctors. Two providers came back carrying the
    same street and the same distance.

    The prompt now asks for an `address_source_url` — the same pattern
    `review_source_url` and `insurance_source_url` have always used — and the
    code refuses an address it cannot trace to a page about this doctor. The
    prompt is not the guard; the predicate is."""

    ADDRESS = "1450 S Dobson Rd Ste B122, Mesa, AZ 85202"

    @staticmethod
    def _backfill(gatherer, source, name="Dr. Harvinder Kumar"):
        from unittest.mock import patch as _patch
        provider = {"name": name, "location": "Mesa, AZ"}
        levels = {"Mesa, AZ": "city", TestAddressMustBeTraceable.ADDRESS: "zip"}
        with _patch("agents.data_gatherer.resolution_level",
                    side_effect=lambda loc: levels.get(loc)):
            gatherer._merge_review_data(provider, {
                "review_summary": "No reviews available",
                "review_sentiment": "unknown", "review_observations": [],
                "address": TestAddressMustBeTraceable.ADDRESS,
                "address_source_url": source,
            })
        return provider

    def test_a_platform_profile_qualifies(self, gatherer):
        provider = self._backfill(
            gatherer, "https://www.healthgrades.com/physician/dr-harvinder-kumar-x1")
        assert provider["location"] == self.ADDRESS

    def test_a_practice_page_naming_the_doctor_qualifies(self, gatherer):
        """`url_page_kind` only knows the review platforms' path markers, so a
        practice site's per-doctor page reads as "unknown" — but its slug names
        them, which is the same evidence by another route."""
        provider = self._backfill(
            gatherer, "https://barrowneuro.org/physicians/harvinder-kumar/")
        assert provider["location"] == self.ADDRESS

    def test_a_GROUP_page_is_refused(self, gatherer):
        """The case that produced the defect, and the reason this asks for
        POSITIVE identification rather than reusing the slug VETO.

        `/physicians/`, `/neurology/`, `/locations/mesa/` name nobody, so they
        contradict nobody — a veto-shaped test admits every one of them, and
        they are exactly the pages that state one address for several doctors.
        (`/our-team/` happens to trip the veto, on two words that are not
        anyone's name; that is luck, not the rule working.)"""
        from utils.listing_parser import slug_contradicts_name

        for group in ("https://barrowneuro.org/physicians/",
                      "https://barrowneuro.org/neurology/",
                      "https://banner.com/locations/mesa/"):
            assert not slug_contradicts_name(group, "Dr. Harvinder Kumar"), group
            assert self._backfill(gatherer, group)["location"] == "Mesa, AZ", group

    def test_another_doctors_profile_is_refused(self, gatherer):
        provider = self._backfill(
            gatherer, "https://www.healthgrades.com/physician/dr-andre-hagevik-x7qq6")
        assert provider["location"] == "Mesa, AZ"

    def test_an_address_with_no_source_at_all_is_refused(self, gatherer):
        """A model that omits the field must not have its address trusted by
        default — the failure mode this whole guard exists for is an address
        whose origin nobody recorded."""
        assert self._backfill(gatherer, None)["location"] == "Mesa, AZ"
        assert self._backfill(gatherer, "")["location"] == "Mesa, AZ"

    def test_the_recorded_source_names_the_page(self, gatherer):
        url = "https://www.healthgrades.com/physician/dr-harvinder-kumar-x1"
        assert self._backfill(gatherer, url)["location_source"] == \
            f"enrichment_model:{url}"


class TestAddressSourceIsAsked_ForAndCarried:
    """The guard above refuses an address it cannot trace. That makes the SUPPLY
    side load-bearing in a way nothing would notice: if the prompt stops asking
    for `address_source_url`, or the return shape drops it, every model-supplied
    address is refused forever — silent coverage loss, not an error."""

    PAYLOAD = {
        "review_summary": "No reviews available", "review_sentiment": "unknown",
        "review_observations": [], "insurance_accepted": [],
        "review_count": None, "rating": None, "review_source_url": None,
        "insurance_source_url": None, "years_experience": None, "phone": None,
        "address": "1450 S Dobson Rd Ste B122, Mesa, AZ 85202",
        "address_source_url": "https://www.healthgrades.com/physician/dr-harvinder-kumar-x1",
    }

    def test_the_prompt_asks_for_it_and_the_answer_carries_it(self, gatherer):
        gatherer.anthropic_client.messages.create.return_value = _llm(self.PAYLOAD)

        out = gatherer._extract_review_data_only(
            [_page("https://www.healthgrades.com/physician/dr-harvinder-kumar-x1",
                   "# Dr. Harvinder Kumar\n")],
            "Dr. Harvinder Kumar", "Neurology", "Mesa, AZ")

        prompt = gatherer.anthropic_client.messages.create.call_args\
            .kwargs["messages"][0]["content"]
        assert "address_source_url" in prompt, "the model is never asked for it"
        assert out["address_source_url"] == self.PAYLOAD["address_source_url"], \
            "the field is asked for but dropped on the way back"

    def test_the_prompt_forbids_a_group_page(self, gatherer):
        """The prompt is not the guard — the predicate is — but an instruction
        the model can follow costs nothing and reduces how often the guard has
        to fire."""
        gatherer.anthropic_client.messages.create.return_value = _llm(self.PAYLOAD)
        gatherer._extract_review_data_only(
            [_page("https://x.com/y", "text")], "Dr. X", "Neurology", "Mesa, AZ")

        prompt = gatherer.anthropic_client.messages.create.call_args\
            .kwargs["messages"][0]["content"]
        assert "lists several doctors" in prompt


class TestAParserDoesNotEraseAnotherParsersCount:
    """The duplicate-page guard in `_apply_parsed_profiles` — mechanics only.

    SCOPE CORRECTED the day it shipped: this guard was written against Dr.
    Hagevik's lost healthgrades count, narrated as "the profile parse lands on
    the listing row's observation". It does not — that function receives only
    the enrichment pass's own (model-written) observations; the listing
    observation lives on the provider and carries no `extraction_source`. The
    two parser reads meet in `_merge_review_data`'s union, and the live fix is
    `_fill_corroborated_gaps` there (TestUnionFillsCorroboratedGaps drives it).

    What this class still pins: when two parsed pages canonicalise to one URL
    (http/https, www, trailing-slash duplicates Tavily returns as distinct
    results), the second — possibly thin — parse must not erase what the first
    read, and the model-vs-parser owns-both rule is untouched. The
    `extraction_source` set by hand below is how a real observation looks only
    after an earlier parsed page's write, which is the one way it reaches this
    code carrying the field.
    """

    URL = "https://www.healthgrades.com/physician/dr-andre-hagevik-x7qq6"

    def _apply(self, prior, facts):
        from agents.data_gatherer import _apply_parsed_profiles, DataGathererAgent
        data = {"review_observations": [dict(prior)] if prior else []}
        out = _apply_parsed_profiles(
            data, {self.URL: facts}, "Dr. Andre Hagevik, MD",
            DataGathererAgent._name_token_overlap)
        return out["review_observations"][0]

    def _listing(self, **over):
        row = {"source_url": self.URL, "rating": 4.8, "review_count": 260,
               "extraction_source": "listing_parser"}
        row.update(over)
        return row

    def test_a_silent_profile_page_keeps_the_listings_count(self):
        obs = self._apply(self._listing(), {"rating": 4.8})
        assert obs["review_count"] == 260, \
            "the thin profile page erased a count the listing parser had read"
        assert obs["rating"] == 4.8

    def test_the_profile_still_wins_when_it_states_its_own_count(self):
        """Silence is what's tolerated — never a weaker number."""
        obs = self._apply(self._listing(), {"rating": 4.8, "review_count": 151})
        assert obs["review_count"] == 151

    def test_a_disagreeing_profile_takes_both_fields(self):
        """Different ratings mean a different reading, not a missing half —
        so the older pair goes rather than being spliced onto the new one."""
        obs = self._apply(self._listing(), {"rating": 3.9})
        assert (obs["rating"], obs["review_count"]) == (3.9, None)

    def test_a_model_read_count_is_still_cleared(self):
        """The original exception, intact: the parser owns BOTH numbers against
        the MODEL, or a parsed rating gets spliced onto a model-read count and
        the card states a pair neither source ever did."""
        obs = self._apply(
            {"source_url": self.URL, "rating": 4.1, "review_count": 300},
            {"rating": 4.8})
        assert obs["review_count"] is None


class TestParsedAddressIsCheckedAgainstItsOwnPage:
    """The address guard read a URL only the MODEL ever wrote.

    Both producers write `address` to one key, but `address_source_url` was set
    only on the model path — so a parser-read address was validated against
    whichever page the model had quoted: refused outright when the model
    supplied none, and checked against an unrelated page when it did.
    """

    URL = "https://www.healthgrades.com/physician/dr-harvinder-kumar-x1"

    def test_the_parser_records_the_page_the_address_came_off(self):
        from agents.data_gatherer import _apply_parsed_profiles, DataGathererAgent
        data = _apply_parsed_profiles(
            {"review_observations": []},
            {self.URL: {"location": "1450 S Dobson Rd, Mesa, AZ 85202",
                        "page_provider_name": "Dr. Harvinder Kumar"}},
            "Dr. Harvinder Kumar", DataGathererAgent._name_token_overlap)

        assert data["address"] == "1450 S Dobson Rd, Mesa, AZ 85202"
        assert data["address_source_url"] == self.URL, \
            "the backfill guard has no page to check this address against"

    def test_the_guard_then_accepts_it_with_no_model_address(self, gatherer):
        """End to end: the parser's address must survive the backfill when the
        model supplied no `address_source_url` of its own.

        `resolution_level` is stubbed because it reads the vendored ZIP table,
        which is an LFS pointer in a clone that has not run `git lfs pull` —
        the geo boundary is not what this test is about.
        """
        from unittest.mock import patch as _patch
        provider = {"name": "Dr. Harvinder Kumar", "location": "Mesa, AZ"}
        levels = {"1450 S Dobson Rd, Mesa, AZ 85202": "zip", "Mesa, AZ": "city"}
        with _patch("agents.data_gatherer.resolution_level",
                    side_effect=lambda loc: levels.get(loc, "city")):
            gatherer._merge_review_data(provider, {
                "address": "1450 S Dobson Rd, Mesa, AZ 85202",
                "address_source": f"profile_parser:{self.URL}",
                "address_source_url": self.URL,
                "review_observations": [],
            })
        assert provider["location"] == "1450 S Dobson Rd, Mesa, AZ 85202"
        assert provider["location_source"] == f"profile_parser:{self.URL}"


class TestAChunkCountNeedsARatingOnTheFullPage:
    """Dr. Hagevik's count was in the payload and refused on a technicality.

    His healthgrades profile came back as a thin 5,756-char body stating
    `4.8 Star Rating` and no total, while the chunks for the SAME url carried
    `Based on 260 reviews` — the text a browser shows. The old rule demanded
    the chunks RESTATE the rating before their count could be admitted, so the
    260 was dropped: healthgrades entered the blend with a rating and no
    weight, and the card read "2 platforms" beside three named ones.

    The full page's rating is the subject anchor. Asking a relevance-selected
    excerpt to re-prove what the full page already established is asking the
    weaker text to carry the stronger one's job.
    """

    URL = "https://www.healthgrades.com/physician/dr-andre-hagevik-x7qq6"
    THIN = "# Dr. Andre Hagevik, MD\n## Other\n4.8 Star Rating\nReview Save\n"

    def _parse(self, raw, content):
        from agents.data_gatherer import _parse_both_texts
        return _parse_both_texts(self.URL, {"raw_content": raw, "content": content})

    def test_the_chunks_complete_a_pair_the_thin_page_started(self):
        facts = self._parse(self.THIN, "Dr. Hagevik ... Based on 260 reviews ...")
        assert (facts.get("rating"), facts.get("review_count")) == (4.8, 260)

    def test_a_count_with_no_rating_anywhere_is_still_refused(self):
        """Nothing names the subject, so the number would attach to whoever
        the page happens to be about."""
        assert self._parse("", "Based on 260 reviews") == {}

    def test_chunks_that_disagree_are_discarded_whole(self):
        """The check that actually protects the count: a promo strip carrying
        someone else's total generally carries their stars too."""
        facts = self._parse(self.THIN, "3.1 Star Rating Based on 512 reviews")
        assert facts.get("rating") == 4.8
        assert facts.get("review_count") is None

    def test_the_full_page_pair_is_never_overwritten(self):
        facts = self._parse(
            "# Dr. A\n4.8 Star Rating\nBased on 260 reviews\n", "Based on 512 reviews")
        assert facts["review_count"] == 260


class TestHagevikEndToEnd:
    """The real page text, through the real parser, to a full pair.

    Copied from the fetched `raw_content` — including `## Other` (healthgrades
    files him under that specialty) and the doubled rating line, both of which
    a hand-written fixture would have smoothed away.
    """

    URL = "https://www.healthgrades.com/physician/dr-andre-hagevik-x7qq6"
    RAW = (
        "[![](https://photos.healthgrades.com/img/prov/X/7/Q/X7QQ6_w120h160_v10265.jpg"
        "?name=Dr.%20Andre%20Hagevik%2C%20MD)\n"
        "# Dr. Andre Hagevik, MD\n"
        "## Other\n"
        "4.8 Star Rating\n"
        "Based on 260 reviews 4.8 Star Rating(260 reviews)\n"
        "Review Save\n"
        "Dr. Andre Hagevik, MD is an other provider in Sun City West, AZ.\n"
        "### Practice\n"
        "14520 W Granite Valley Dr Ste 210 ·Sun City West, AZ 85375\n"
    )

    def test_the_fetched_page_yields_the_pair(self):
        from utils.profile_parser import parse_profile
        facts = parse_profile(self.URL, self.RAW)
        assert (facts["rating"], facts["review_count"]) == (4.8, 260)
        assert facts["page_provider_name"] == "Dr. Andre Hagevik, MD"

    def test_the_doubled_rating_line_does_not_double_the_count(self):
        """`4.8 Star Rating` appears twice and `260` three times on this page.
        A count that came back as 260260 or a rating of 4.84 would mean the
        pattern is spanning lines it should not."""
        from utils.profile_parser import parse_profile
        facts = parse_profile(self.URL, self.RAW)
        assert facts["review_count"] == 260 and facts["rating"] == 4.8


class TestUnionFillsCorroboratedGaps:
    """First-URL-wins decided which READ survives, not just which entry.

    The union exists so one page can't count twice, and that part is right.
    But since a parsed listing row's observation points at the doctor's own
    profile url, discovery and enrichment collide on that url BY CONSTRUCTION —
    and skipping the newcomer outright kept whichever pass ran first, richer or
    not. Reproduced with Dr. Hagevik's shapes: discovery `4.8/None` +
    enrichment recovering `4.8/260` kept the None, so a count lost to a thin
    fetch and to the chunk rule was recoverable a third time and still
    discarded.

    NOTE the placement: an earlier fix put a corroboration guard inside
    `_apply_parsed_profiles`, narrating this exact collision. It cannot happen
    there — that function receives only the enrichment pass's own observations,
    and the listing observation lives on the provider. These tests drive
    `_merge_review_data`, the function where the two passes actually meet.
    """

    URL = "https://www.healthgrades.com/physician/dr-andre-hagevik-x7qq6"

    def _merge(self, discovery_obs, enrichment_obs):
        from unittest.mock import patch as _patch
        from agents.data_gatherer import DataGathererAgent
        with _patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
            g = DataGathererAgent()
        provider = {"name": "Dr. Andre Hagevik, MD", "location": "Mesa, AZ",
                    "review_observations": [dict(discovery_obs)]}
        g._merge_review_data(provider, {"review_observations": [dict(enrichment_obs)]})
        return provider

    def _discovery(self, rating=4.8, count=None):
        # The REAL listing-observation shape (`_listing_row_to_provider`):
        # no extraction_source — an earlier guard assumed one was present.
        return {"source_url": self.URL, "read_from_url": "https://hg/usearch",
                "rating": rating, "review_count": count,
                "page_provider_name": "Dr. Andre Hagevik, MD"}

    def test_enrichments_count_completes_discoverys_rating_only_row(self):
        provider = self._merge(
            self._discovery(4.8, None),
            {"source_url": self.URL, "rating": 4.8, "review_count": 260,
             "extraction_source": "profile_parser"})
        pairs = [(o.get("rating"), o.get("review_count"))
                 for o in provider["review_observations"]]
        assert (4.8, 260) in pairs, "the union kept the poorer read"
        assert provider.get("review_count") == 260, \
            "the blend never saw the recovered count"

    def test_a_disagreeing_read_changes_nothing(self):
        """On ANY disagreement the union's existing behaviour — first wins,
        untouched — is preserved. This fill never overwrites and never clears."""
        provider = self._merge(
            self._discovery(4.8, 260),
            {"source_url": self.URL, "rating": 3.9, "review_count": 500})
        pairs = [(o.get("rating"), o.get("review_count"))
                 for o in provider["review_observations"]]
        assert pairs == [(4.8, 260)]

    def test_no_shared_field_fills_nothing(self):
        """Rating-only meeting count-only: nothing establishes the two reads
        agree, and manufacturing a pair from two half-reads is the splice the
        parser rules refuse."""
        provider = self._merge(
            self._discovery(4.8, None),
            {"source_url": self.URL, "rating": None, "review_count": 260})
        pairs = [(o.get("rating"), o.get("review_count"))
                 for o in provider["review_observations"]]
        assert (4.8, 260) not in pairs

    def test_a_new_url_still_appends(self):
        provider = self._merge(
            self._discovery(4.8, 260),
            {"source_url": "https://www.vitals.com/doctors/dr-andre-hagevik",
             "rating": 4.7, "review_count": 109})
        assert len(provider["review_observations"]) == 2


class TestTemplateMarkerGuardsChunkCounts:
    """healthgrades' template names which pages may take a chunk count.

    Template A ("N.N Star Rating") carries "Based on N reviews" in its payload,
    so a chunk count there recovers a thin fetch's loss — the Hagevik case.
    Template B ("Likelihood of recommending … is N out of 5") states NO total
    anywhere in its payload (measured), so a count appearing only in a
    template-B page's chunks cannot be the subject's — it is a neighbour's,
    off a promo strip whose stars the chunk boundary cut. This was the one
    hole the chunk-rule loosening opened, and the marker closes it with zero
    collateral: the two cases were indistinguishable at the merge without it.
    """

    URL = "https://www.healthgrades.com/physician/dr-hemant-pandey-abc12"
    TEMPLATE_A = "# Dr. A\n4.8 Star Rating\nReview Save\n"
    TEMPLATE_B = ("# Dr. Hemant Pandey, MD\n"
                  "Likelihood of recommending Dr. Pandey is 3.7727273 out of 5\n")

    def test_the_parser_names_the_template(self):
        """All THREE rating-producing branches, because the marker is set per
        branch and a revert check caught the pair branch uncovered: a mutation
        removing only ITS marker line stayed green against the star-only text."""
        from utils.profile_parser import parse_profile
        assert parse_profile(self.URL, self.TEMPLATE_A)["rating_pattern"] == "star"
        assert parse_profile(self.URL, self.TEMPLATE_B)["rating_pattern"] == "likelihood"
        paired = parse_profile(self.URL, "# Dr. A\n4.8 Star Rating(260 reviews)\n")
        assert (paired["rating_pattern"], paired["review_count"]) == ("star", 260)

    def test_a_template_b_page_refuses_a_chunk_count(self):
        from agents.data_gatherer import _parse_both_texts
        facts = _parse_both_texts(self.URL, {
            "raw_content": self.TEMPLATE_B,
            "content": "Based on 512 reviews",
        })
        assert facts.get("rating") == 3.8
        assert facts.get("review_count") is None, \
            "a neighbour's bare count landed on a page whose payload has no total"

    def test_a_template_a_page_still_accepts_one(self):
        """The Hagevik recovery must survive this guard."""
        from agents.data_gatherer import _parse_both_texts
        facts = _parse_both_texts(self.URL, {
            "raw_content": self.TEMPLATE_A,
            "content": "Based on 260 reviews",
        })
        assert facts.get("review_count") == 260


class TestExperienceProvenance:
    """Every producer of a ranked number names itself.

    Tenure decides real ranking points (a stated year vs the unknown
    imputation), and it has three producers — a listing row, a parsed profile,
    the model — with no field saying which one wrote the number on any given
    card. The address gained `location_source` for the same reason; this is
    tenure's half.
    """

    URL = "https://www.healthgrades.com/physician/dr-harvinder-kumar-x1"

    def test_a_listing_row_labels_its_tenure(self):
        from agents.data_gatherer import _listing_row_to_provider
        provider = _listing_row_to_provider(
            {"name": "Dr. A", "years_experience": 28,
             "review_source_url": "https://hg/usearch"}, "Neurology")
        assert provider["experience_source"] == "listing_parser:https://hg/usearch"

    def test_a_row_without_tenure_labels_nothing(self):
        from agents.data_gatherer import _listing_row_to_provider
        provider = _listing_row_to_provider(
            {"name": "Dr. A", "review_source_url": "https://hg/usearch"}, "Neurology")
        assert provider["experience_source"] is None

    def test_a_parsed_profile_labels_its_tenure(self):
        from agents.data_gatherer import _apply_parsed_profiles, DataGathererAgent
        data = _apply_parsed_profiles(
            {"review_observations": []},
            {self.URL: {"years_experience": 20,
                        "page_provider_name": "Dr. Harvinder Kumar"}},
            "Dr. Harvinder Kumar", DataGathererAgent._name_token_overlap)
        assert data["experience_source"] == f"profile_parser:{self.URL}"

    def _merge(self, review_data):
        from unittest.mock import patch as _patch
        from agents.data_gatherer import DataGathererAgent
        with _patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
            g = DataGathererAgent()
        provider = {"name": "Dr. Harvinder Kumar", "location": "Mesa, AZ"}
        review_data.setdefault("review_observations", [])
        g._merge_review_data(provider, review_data)
        return provider

    def test_the_merge_carries_the_parsers_label(self):
        provider = self._merge({"years_experience": 20,
                                "experience_source": f"profile_parser:{self.URL}"})
        assert provider["years_experience"] == 20
        assert provider["experience_source"] == f"profile_parser:{self.URL}"

    def test_an_unlabelled_survivor_is_the_models(self):
        """The parser labels its own write, so tenure arriving with no label
        was read by the model off any of six blocks — which is exactly why the
        label matters for triage."""
        provider = self._merge({"years_experience": 20})
        assert provider["experience_source"] == "enrichment_model"


class TestWebmdLocationsSection:
    """webmd's `## Locations` section carries the office LIST — parse all of it.

    Measured 2026-08-06 on saved extract bytes: Dr. Vandian's fetched copy
    lists five CORE Institute offices (each block: practice link, street line,
    `CITY,ST,ZIP` line, optional Tel link, and a Get Directions link whose
    `destination=` param is the full address URL-encoded), then " Show All" —
    so even a full-looking list is TRUNCATED, which is why membership in the
    parsed set is confirm-only evidence. Dr. Kumar's copy of the same page
    shape carries the heading and NOTHING under it. The old parser read one
    address via a one-line pattern that the extract-tier rendering (street and
    city on separate lines) cannot match at all — and paired it with whatever
    phone matched first anywhere, which put the Phoenix office's number beside
    a Gilbert address on a live card.
    """

    URL = "https://doctor.webmd.com/doctor/vardges-vandian-73b1ff9f-overview"
    SCW = "14520 W GRANITE VALLEY DR STE 120, SUN CITY WEST, AZ 85375"
    THOMAS = "9305 W Thomas Rd Ste 305, Phoenix, AZ 85037"
    MESA = "1450 S Dobson Rd Ste B122, Mesa, AZ 85202"

    # Condensed from the saved fetched bytes — block shapes are real.
    PAGE = """# Dr. Vardges Vandian, DO

## Locations

[The Core Institute](https://doctor.webmd.com/practice/the-core-institute-877b88b9)

14520 W GRANITE VALLEY DR STE 120

SUN CITY WEST,AZ,85375

Tel:[(866) 974-2673](tel:+18669742673)

[Get Directions](https://www.google.com/maps/dir/?api=1&destination=14520%20W%20GRANITE%20VALLEY%20DR%20STE%20120%2C%20SUN%20CITY%20WEST%2C%20AZ%2085375)

Mon 8:00 am - 5:00 pm

[Arizona Spine Care Plc](https://doctor.webmd.com/practice/arizona-spine-care-plc-a5c8e054)

The CORE Institute

9305 W Thomas Rd Ste 305

Phoenix,AZ,85037

Tel:[(623) 742-9975](tel:+16237429975)

[Get Directions](https://www.google.com/maps/dir/?api=1&destination=9305%20W%20Thomas%20Rd%20Ste%20305%2C%20Phoenix%2C%20AZ%2085037)

[The Core Institute Mesa](https://doctor.webmd.com/practice/the-core-institute-mesa-45c592b0)

The CORE Institute

1450 S Dobson Rd Ste B122

Mesa,AZ,85202

[Get Directions](https://www.google.com/maps/dir/?api=1&destination=1450%20S%20Dobson%20Rd%20Ste%20B122%2C%20Mesa%2C%20AZ%2085202)

 Show All

## Ratings & Reviews
"""

    def test_every_block_parses_in_page_order_with_its_own_phone(self):
        from utils.profile_parser import parse_profile
        found = parse_profile(self.URL, self.PAGE)

        assert found["locations"] == [
            {"address": self.SCW, "phone": "(866) 974-2673"},
            {"address": self.THOMAS, "phone": "(623) 742-9975"},
            {"address": self.MESA, "phone": None},
        ]

    def test_block_one_supplies_the_page_address_when_the_one_line_form_is_absent(self):
        """Page order is the page's own primacy — never nearest-anything."""
        from utils.profile_parser import parse_profile
        found = parse_profile(self.URL, self.PAGE)

        assert found["location"] == self.SCW

    def test_an_empty_section_yields_no_locations(self):
        """Dr. Kumar's fetched copy: the heading immediately followed by the
        next section. Per-copy, not per-platform — degrade to the old
        single-address behaviour, never invent."""
        from utils.profile_parser import parse_profile
        page = "# Dr. Harvinder Kumar\n\n## Locations\n\n## Ratings & Reviews\n"
        found = parse_profile(self.URL, page)

        assert "locations" not in found

    def test_a_page_without_the_section_is_unchanged(self):
        from utils.profile_parser import parse_profile
        found = parse_profile(self.URL, "# Dr. X\n4.5\n\n(61 Ratings)\n")

        assert "locations" not in found


class TestProfileLocationSet:
    """Every listed office becomes an address candidate under the page's URL.

    One page stays ONE vote in the resolver (distinct-source counting); what
    the members add is REACH — an office near the listing's claimed address
    corroborates that AREA even when no two strings are equal, which is how a
    group-practice doctor's city-listing address stops reading as a
    contradiction of his own profile."""

    URL = "https://doctor.webmd.com/doctor/vardges-vandian-73b1ff9f-overview"

    def test_every_listed_office_becomes_a_candidate_with_its_phone(self):
        from agents.data_gatherer import _apply_parsed_profiles, DataGathererAgent
        facts = {
            "page_provider_name": "Dr. Vardges Vandian, DO",
            "location": "14520 W GRANITE VALLEY DR STE 120, SUN CITY WEST, AZ 85375",
            "locations": [
                {"address": "14520 W GRANITE VALLEY DR STE 120, SUN CITY WEST, AZ 85375",
                 "phone": "(866) 974-2673"},
                {"address": "9305 W Thomas Rd Ste 305, Phoenix, AZ 85037",
                 "phone": "(623) 742-9975"},
            ],
        }
        data = _apply_parsed_profiles(
            {"review_observations": []}, {self.URL: facts},
            "Dr. Vardges Vandian, DO", DataGathererAgent._name_token_overlap)

        candidates = data["address_candidates"]
        by_address = {c["address"]: c for c in candidates}
        assert "9305 W Thomas Rd Ste 305, Phoenix, AZ 85037" in by_address
        member = by_address["9305 W Thomas Rd Ste 305, Phoenix, AZ 85037"]
        assert member["source"] == f"profile_parser:{self.URL}"
        assert member["phone"] == "(623) 742-9975"


class TestPracticeLocationRecord:
    """Every believable address is RECORDED as the provider's location set.

    The predecessor flagged addresses >20 miles apart as a "conflict" to
    adjudicate. Field evidence retired the framing: the doctors who trip
    this are multi-office group specialists whose own profiles list offices
    across the metro, so several far-apart addresses are a confirmed
    practice, not a dispute — and the distance gate made a doctor with
    offices 19 miles apart the indefensible edge case. The record is now
    unconditional on distance; `_select_nearest_trusted_location` chooses
    which office each member sees.
    """

    MESA = "1450 S Dobson Rd Ste B122, Mesa, AZ 85202"
    MERCY = "3420 S Mercy Rd Ste 200, Gilbert, AZ 85297"

    def _merge(self, provider_location, candidates):
        from unittest.mock import patch as _patch
        from agents.data_gatherer import DataGathererAgent
        with _patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
            g = DataGathererAgent()
        provider = {"name": "Dr. Andre Hagevik, MD", "location": provider_location,
                    "location_source": "listing_parser:https://hg/usearch"}
        # No user_location: the record is member-independent; selection is
        # tested separately and skips without a member to measure from.
        g._merge_review_data(provider, {
            "review_observations": [],
            "address_candidates": candidates,
        })
        return provider

    def test_two_addresses_are_recorded_regardless_of_distance(self):
        """The dropped 20-mile gate, pinned: these two offices are ~10 miles
        apart — the OLD flag stayed silent for them, and a member got no
        "confirm the office" note for a doctor whose offices genuinely
        differ. Any second distinct address now makes a location set."""
        provider = self._merge(
            self.MESA,
            [{"address": self.MERCY, "source": "profile_parser:https://hg/x7qq6"}])

        record = provider["address_conflict"]
        assert record["distinct_count"] == 2
        sources = {c["source"] for c in record["addresses"]}
        assert sources == {"listing_parser:https://hg/usearch",
                           "profile_parser:https://hg/x7qq6"}, \
            "the member-facing note and the panel need WHO claimed each address"

    def test_one_address_from_two_sources_is_not_a_location_set(self):
        """An exact confirmation is corroboration for the TRUST check, not a
        second location — one office, however many pages state it, gives the
        member nothing to choose between."""
        provider = self._merge(
            self.MESA,
            [{"address": self.MESA, "source": "profile_parser:https://hg/x7qq6"}])

        assert "address_conflict" not in provider

    def test_a_single_address_stays_silent(self):
        provider = self._merge(self.MESA, [])
        assert "address_conflict" not in provider


class TestNearestTrustedSelection:
    """TRUST, then MIN: the member sees the nearest office we can trust.

    Supersedes the cluster-corroboration vote and its banned
    nearest-to-searcher tie-break. The vote asked "which single address is
    true?" — but these doctors are confirmed multi-office specialists, so
    several are true, and any fixed pick made the answer depend on which
    pages that run's fetch foregrounded: Dr. Vandian, the pool's strongest
    review record, oscillated between rank 5 at his 6.4-mile Gilbert office
    and rank 7 at a 40-mile office across two same-criteria runs. The order
    of the two steps is load-bearing: a bare min() hands the card to the
    single worst datapoint whenever it is near the member, so an address
    must first be trusted — the doctor's own profile lists it, or two
    distinct sources state exactly the same address.
    """

    CHANDLER = "Chandler, AZ 85224"          # the searching member
    GLENDALE_MEMBER = "Glendale, AZ 85304"   # a different member
    MESA = "1450 S Dobson Rd Ste B122, Mesa, AZ 85202"
    PHOENIX = "9321 W Thomas Rd Ste 205, Phoenix, AZ 85037"
    GLENDALE = "5601 W Eugie Ave Ste 100, Glendale, AZ 85304"
    VAL_VISTA = "2680 S Val Vista Dr Ste 146 Bldg 9, Gilbert, AZ 85295"
    SCW = "14520 W Granite Valley Dr Ste 210, Sun City West, AZ 85375"

    LISTING = "listing_parser:https://doctor.webmd.com/providers/specialty/neurology/arizona/tempe"
    HG_PROFILE = "profile_parser:https://www.healthgrades.com/physician/dr-harvinder-kumar-gdsc5"
    WEBMD_PROFILE = "profile_parser:https://doctor.webmd.com/doctor/harvinder-kumar-43351276-overview"

    # Member-to-office distances as the geo layer would measure them;
    # anything not listed resolves to None (an unplaceable address).
    DISTANCES = {
        frozenset({CHANDLER, MESA}): 6.5,
        frozenset({CHANDLER, PHOENIX}): 25.0,
        frozenset({CHANDLER, GLENDALE}): 28.0,
        frozenset({CHANDLER, VAL_VISTA}): 6.4,
        frozenset({CHANDLER, SCW}): 40.3,
        frozenset({GLENDALE_MEMBER, MESA}): 27.0,
        frozenset({GLENDALE_MEMBER, PHOENIX}): 4.0,
        frozenset({GLENDALE_MEMBER, GLENDALE}): 1.0,
    }

    def _merge(self, provider_location, location_source, candidates,
               user_location=CHANDLER, distances=None):
        from unittest.mock import patch as _patch
        from agents.data_gatherer import DataGathererAgent
        with _patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
            g = DataGathererAgent()
        provider = {"name": "Dr. Harvinder Kumar", "location": provider_location,
                    "location_source": location_source}
        pair_map = self.DISTANCES if distances is None else distances

        def lookup(a, b):
            return pair_map.get(frozenset({str(a), str(b)}))

        with _patch("agents.data_gatherer.distance_miles", side_effect=lookup):
            g._merge_review_data(provider, {
                "review_observations": [],
                "address_candidates": candidates,
            }, user_location=user_location)
        return provider

    def _kumar_candidates(self):
        # His own webmd profile lists the whole set — Mesa included — and
        # healthgrades adds Glendale: the live 2026-08-08 record.
        return [
            {"address": self.MESA, "source": self.WEBMD_PROFILE},
            {"address": self.PHOENIX, "source": self.WEBMD_PROFILE},
            {"address": self.GLENDALE, "source": self.HG_PROFILE},
        ]

    def test_the_nearest_trusted_office_wins_for_this_member(self):
        """Kumar, searched from Chandler: Mesa (6.5 mi) is on his own
        profile, so it is trusted AND nearest — the card keeps it by rule,
        not by tie deadlock (the old vote left this exact shape 2-2
        unresolved)."""
        provider = self._merge(self.MESA, self.LISTING, self._kumar_candidates())

        assert provider["location"] == self.MESA
        selection = provider["address_conflict"]["selection"]
        assert selection["resolved"] is True
        assert selection["chosen"] == self.MESA
        assert selection["distance_miles"] == 6.5

    def test_the_selection_is_member_relative(self):
        """The same doctor, searched from Glendale, shows his Glendale
        office — a fact about the member-doctor PAIR, which is exactly why
        the choice re-runs per search and is never cached."""
        provider = self._merge(
            self.MESA, self.LISTING, self._kumar_candidates(),
            user_location=self.GLENDALE_MEMBER)

        assert provider["location"] == self.GLENDALE
        assert provider["address_conflict"]["selection"]["distance_miles"] == 1.0

    def test_vandian_stops_oscillating(self):
        """The observed failure: rank 5 at 6.4-mile Gilbert one run, rank 7
        at a 40-mile office the next, decided by which page the fetch
        foregrounded. Whichever office is CURRENT, the member ends at the
        near one."""
        candidates = [
            {"address": self.VAL_VISTA, "source": self.WEBMD_PROFILE},
            {"address": self.SCW, "source": self.HG_PROFILE},
        ]
        far_first = self._merge(self.SCW, self.HG_PROFILE, candidates)
        near_first = self._merge(self.VAL_VISTA, self.WEBMD_PROFILE, candidates)

        assert far_first["location"] == self.VAL_VISTA
        assert near_first["location"] == self.VAL_VISTA

    def test_a_bogus_near_singleton_cannot_take_the_card(self):
        """The reason TRUST precedes MIN: an uncorroborated listing-only
        address two miles from the member — the shape a same-named doctor
        or a glued listing row produces — must not win by nearness. Without
        the trust gate, every extra page fetched would be another lottery
        ticket for a bogus-but-near address."""
        bogus = "123 W Ray Rd, Chandler, AZ 85224"
        distances = dict(self.DISTANCES)
        distances[frozenset({self.CHANDLER, bogus})] = 2.0
        provider = self._merge(self.PHOENIX, self.WEBMD_PROFILE, [
            {"address": bogus, "source": "listing_parser:https://vitals.com/chandler"},
        ], distances=distances)

        assert provider["location"] == self.PHOENIX
        selection = provider["address_conflict"]["selection"]
        assert selection["trusted_addresses"] == 1

    def test_an_exact_twin_makes_a_weak_address_trusted(self):
        """Two distinct sources stating the SAME address is real
        corroboration even with no profile page among them — and exact is
        the bar on purpose: the retired 20-mile "same area counts" rule let
        two independently wrong addresses vouch for each other."""
        distances = dict(self.DISTANCES)
        provider = self._merge(self.PHOENIX, self.WEBMD_PROFILE, [
            {"address": self.MESA, "source": self.LISTING},
            {"address": self.MESA, "source": "enrichment_model:https://practice.example/locations"},
        ], distances=distances)

        assert provider["location"] == self.MESA, \
            "trusted via the twin, and nearer than the profile office"

    def test_nothing_trusted_keeps_the_current_address(self):
        """Only uncorroborated singletons — the old satellite-vs-stale dead
        end. Nothing moves, and the record stands so the member still gets
        the confirm-the-office note."""
        provider = self._merge(self.MESA, self.LISTING, [
            {"address": self.SCW, "source": "enrichment_model:https://x/somewhere"},
        ])

        assert provider["location"] == self.MESA
        selection = provider["address_conflict"]["selection"]
        assert selection["resolved"] is False
        assert selection["reason"] == "no_trusted_address"

    def test_unresolvable_distances_keep_the_current_address(self):
        """Trusted offices that our geocoding cannot place against the
        member are our coverage gap, not their conflict."""
        provider = self._merge(self.MESA, self.LISTING, self._kumar_candidates(),
                               distances={})

        assert provider["location"] == self.MESA
        selection = provider["address_conflict"]["selection"]
        assert selection["resolved"] is False
        assert selection["reason"] == "no_resolvable_distance"

    def test_the_phone_travels_with_the_chosen_office(self):
        """A phone travels with its own office block or not at all — never
        the old address's number beside the new address."""
        candidates = [
            {"address": self.VAL_VISTA, "source": self.WEBMD_PROFILE,
             "phone": "(480) 555-0146"},
            {"address": self.SCW, "source": self.HG_PROFILE},
        ]
        provider = self._merge(self.SCW, self.HG_PROFILE, candidates)

        assert provider["location"] == self.VAL_VISTA
        assert provider["phone"] == "(480) 555-0146"

    def test_no_member_location_skips_selection(self):
        """Without a member there is no "nearest" — the record stands, the
        location stays, and nothing invents a choice."""
        provider = self._merge(self.MESA, self.LISTING, self._kumar_candidates(),
                               user_location="")

        assert provider["location"] == self.MESA
        assert "selection" not in provider["address_conflict"]


class TestWebmdCompareTableFallback:
    """A fetched webmd profile (Dr. Kan Yu, 2026-09-02, via /extract) carried
    no header card and no FAQ restatement — `## Ratings & Reviews` read
    "No data" — while `## Compare with Similar Doctors` stated the subject's
    `4.5 (151 Ratings)` and `40 Years Experience`. Same widget vitals ships;
    the webmd parser never read it, so the page yielded nothing."""

    URL = "https://doctor.webmd.com/doctor/kan-yu-cc7025da-33bd-df11-a4b4-001f29e3eb44-overview"

    def _page(self, h1="Dr. Kan Yu, MD", subject_col=0):
        names = ["Dr. Kan Yu, MD", "Dr. David Paul Brown, MD", "Dr. Ramzy G Medaa, MD", "Dr. Cinthi Pillai, MD"]
        pairs = ["4.5     (151 Ratings)", "3.5     (7 Ratings)", "5.0     (1 Rating)", "5.0     (20 Ratings)"]
        years = ["40 Years Experience", "39 Years Experience", "27 Years Experience", "20 Years Experience"]
        last = ["[View Profile](https://doctor.webmd.com/doctor/x-overview)"] * 4
        last[subject_col] = "Current Profile"
        if subject_col != 0:  # the subject's name moves with the marker
            names[0], names[subject_col] = names[subject_col], names[0]
            pairs[0], pairs[subject_col] = pairs[subject_col], pairs[0]
            years[0], years[subject_col] = years[subject_col], years[0]
        row = lambda cells: "| " + " | ".join(cells) + " |"
        return "\n".join([
            f"# {h1}", "", "## Overview", "Dr. Kan Yu, MD, is a Neurologist practicing in Gilbert, AZ.", "",
            "## Ratings & Reviews for Dr. Yu", "", "#### Patients’ Perspective", "", "No data", "",
            "## Compare with Similar Doctors", "",
            "|  |  |  |  |", "| --- | --- | --- | --- |",
            row(names), row(["Neurology"] * 4), row(pairs), row(years),
            row(["Gilbert, AZ", "Tempe, AZ", "Scottsdale, AZ", "New York, NY"]), row(last),
            "", "## Specialties", "Neurology",
        ])

    def test_reads_the_subjects_pair_and_tenure_from_the_table(self):
        parsed = parse_profile(self.URL, self._page())
        assert (parsed.get("rating"), parsed.get("review_count")) == (4.5, 151)
        assert parsed.get("years_experience") == 40

    def test_subject_column_is_structural_not_positional(self):
        """The marker, not the first column, names the subject."""
        parsed = parse_profile(self.URL, self._page(subject_col=2))
        assert (parsed.get("rating"), parsed.get("review_count")) == (4.5, 151)

    def test_column_name_must_agree_with_the_page_h1(self):
        """A `Current Profile` under a stranger's name is not this doctor's
        pair — a neighbour's stars on a card is the failure every identity
        rule exists to prevent."""
        parsed = parse_profile(self.URL, self._page(h1="Dr. Someone Else, MD"))
        assert parsed.get("rating") is None
        assert parsed.get("review_count") is None


class TestHealthgradesLaterPagesAndStateDirectory:
    """healthgrades writes page 1 of a city directory with absolute profile
    links and pages 2+ (and the state directory) with host-relative ones —
    `/physician/dr-kan-yu-2b5bc` — which the heading regex rejected outright:
    0 rows from a 25 KB `chandler_2` that lists Dr. Kan Yu. The state page
    also glues its echoed rating to its neighbours (`out of 54.7from 48`)."""

    PAGE2 = "https://www.healthgrades.com/neurology-directory/az-arizona/chandler_2"
    STATE = "https://www.healthgrades.com/neurology-directory/az-arizona"

    def test_page_two_entries_yield_name_and_absolute_profile_url(self):
        text = "\n".join([
            "# 20 Best Neurologists Near Chandler, AZ",
            '## We found 81 results within 10 miles for "Neurologists near Chandler, AZ"',
            "### [Dr. Brandon Woods, MD](/physician/dr-brandon-woods-3mjyy)", "",
            "![](https://dims.healthgrades.com/a.jpg)", "",
            "### [Dr. Kan Yu, MD](/physician/dr-kan-yu-2b5bc)", "",
            "![](https://dims.healthgrades.com/b.jpg)", "",
        ])
        rows = parse_listing(self.PAGE2, text)
        assert [r["name"] for r in rows] == ["Dr. Brandon Woods, MD", "Dr. Kan Yu, MD"]
        assert rows[1]["profile_url"] == "https://www.healthgrades.com/physician/dr-kan-yu-2b5bc"
        # No pair rendered on these pages — the row still exists, rating-less,
        # exactly like every webmd listing row.
        assert rows[1]["rating"] is None and rows[1]["review_count"] is None
        assert healthgrades_result_count(text) == 81

    def test_state_directory_entry_reads_glued_pair_specialty_and_address(self):
        text = "\n".join([
            "# 20 Best Neurologists In Arizona",
            '## We found584 results for "Neurologists in Arizona"',
            "### [Dr. Peter Struck, MD](/physician/dr-peter-struck-3pp3q)", "",
            "Specialty: Neurology", "",
            "Rated 4.7 out of 54.7from 48 ratings•[31 written reviews](/physician/dr-peter-struck-3pp3q#ratings)", "",
            "[7242 E Osborn Rd Ste 400Scottsdale, AZ 85251](/physician/dr-peter-struck-3pp3q#locations)", "",
            "[View Profile](/physician/dr-peter-struck-3pp3q)",
        ])
        rows = parse_listing(self.STATE, text)
        assert len(rows) == 1
        row = rows[0]
        assert (row["rating"], row["review_count"]) == (4.7, 48)
        assert row["specialty"] == "Neurology"
        assert "85251" in (row["location"] or "") and "Scottsdale" in (row["location"] or "")
        assert row["profile_url"] == "https://www.healthgrades.com/physician/dr-peter-struck-3pp3q"
        assert healthgrades_result_count(text) == 584

    def test_page_one_absolute_links_still_parse(self):
        text = "\n".join([
            "### [Dr. Hemant Pandey, MD](https://www.healthgrades.com/physician/dr-hemant-pandey-xsjwm)",
            "Rated 3.8 out of 5 3.8 from 88 ratings Neurology [4045 W Chandler Blvd Bldg F, Chandler, AZ 85226]",
        ])
        rows = parse_listing("https://www.healthgrades.com/neurology-directory/az-arizona/chandler", text)
        assert (rows[0]["rating"], rows[0]["review_count"]) == (3.8, 88)
        assert rows[0]["profile_url"].startswith("https://www.healthgrades.com/physician/")

    @pytest.mark.parametrize("count, expected", [
        (81, ["_2", "_3", "_4", "_5"]),
        (20, []),
        (21, ["_2"]),
        (200, ["_2", "_3", "_4", "_5"]),   # capped at five pages
        (None, []),
    ])
    def test_page_urls_follow_the_stated_count(self, count, expected):
        base = "https://www.healthgrades.com/neurology-directory/az-arizona/chandler"
        assert healthgrades_page_urls(base, count) == [base + s for s in expected]

    def test_a_later_page_never_paginates_again(self):
        assert healthgrades_page_urls(self.PAGE2, 81) == []


class TestListingPaginationAllPlatforms:
    """Every platform paginates its city listing behind its own parameter
    (2026-09-02, read off the pages' own "Page 2" links): healthgrades
    `<city>_N`, webmd `?pagenumber=N`, vitals `?page=N`. Page 1 alone read 20
    of healthgrades' 81, 46 of webmd's 142 and 45 of vitals' 142."""

    HG = "https://www.healthgrades.com/neurology-directory/az-arizona/chandler"
    WM = "https://doctor.webmd.com/providers/specialty/neurology/arizona/chandler"
    VI = "https://www.vitals.com/neurology/az/chandler"

    def test_webmd_total_is_the_largest_stated_number(self):
        text = ("# Best **Neurologists** in **Chandler, AZ** Chandler, AZ has **142 Neurologist** "
                "results with an average of **31 years of experience** and **a total of 2088 reviews**. "
                "Showing 85 providers")
        assert listing_result_count(self.WM, text) == 142

    def test_vitals_total_is_the_largest_stated_number(self):
        text = "# 142 Neurologists in Chandler, AZ\n\n85 Neurologists accepting new patients"
        assert listing_result_count(self.VI, text) == 142

    def test_healthgrades_total_unchanged(self):
        assert listing_result_count(self.HG, 'We found 81 results within 10 miles') == 81

    def test_no_phrase_or_unknown_platform_is_none(self):
        assert listing_result_count(self.WM, "no totals here") is None
        assert listing_result_count("https://example.com/list", "142 Neurologists") is None

    def test_each_platform_uses_its_own_parameter(self):
        """THE TRAP, pinned: webmd's `?page=2` and vitals' `?pagenumber=2` both
        silently return page 1 again (0 new names) — each platform ignores the
        other's parameter and the wrong one looks like success."""
        webmd = listing_page_urls(self.WM, 142)
        vitals = listing_page_urls(self.VI, 142)
        assert webmd == [self.WM + "?pagenumber=2", self.WM + "?pagenumber=3"]
        assert vitals == [self.VI + "?page=2", self.VI + "?page=3"]
        assert all("?page=" not in u for u in webmd)
        assert all("pagenumber" not in u for u in vitals)

    def test_healthgrades_shape_unchanged(self):
        assert listing_page_urls(self.HG, 81) == [self.HG + f"_{n}" for n in (2, 3, 4, 5)]

    @pytest.mark.parametrize("count", [None, 0, 50])
    def test_one_page_or_unknown_total_fetches_nothing(self, count):
        assert listing_page_urls(self.WM, count) == []
        assert listing_page_urls(self.VI, count) == []

    def test_cap_is_five_pages(self):
        assert len(listing_page_urls(self.VI, 100_000)) == 4   # pages 2..5

    def test_a_later_page_never_paginates_again(self):
        assert listing_page_urls(self.WM + "?pagenumber=2", 142) == []
        assert listing_page_urls(self.VI + "?page=3", 142) == []
        assert listing_page_urls(self.HG + "_2", 81) == []

    def test_non_platform_url_fetches_nothing(self):
        assert listing_page_urls("https://example.com/doctors", 500) == []
