"""The 2026-07-29 field run: a fetched profile that produced nothing usable.

Dr. Andrea An's card named `healthgrades.com — listing page` as its best single
source. The instrumentation showed the profile HAD been fetched:

    sources[0]  healthgrades.com/physician/dr-andrea-an-2pfjn  profile  44,138
    sources[1]  doctor.webmd.com/doctor/andrea-an-…-overview   profile  33,730
    sources[2]  vitals.com/doctors/andrea-an-ul3cfp            profile  31,690
    platform_pairs 3 · profile_backed_platforms 2 · headline_kind "listing"

So the page arrived and lost. Not on the tie-break — `_page_rank` would have
resolved a count tie toward the profile, and both pages state 4.1. It lost
because it never produced a competing rating+count PAIR at all.

Two independent causes, both traced to the same Phase 1 word-boundary change:

  1. the surname anchor "An" is the English indefinite article, scoring ~360
     hits on review prose against 0 for the full name, so the priority windows
     scattered and never reached the header
  2. `\\breview\\b` scores ZERO on "reviews" and `\\brating\\b` misses
     "70 patient ratings" — the vocabulary anchors could not match the words a
     review count is actually written with

and one reason it took two rounds of inference to find: `enrichment_sources`
recorded what was FETCHED and not what each page PRODUCED.
"""

import re

import pytest

from agents.data_gatherer import (
    _DOMAIN_ANCHOR_HINTS,
    _FUNCTION_WORD_SURNAMES,
    _annotate_source_yields,
    _surname_anchors,
)
from utils.excerpt import _anchor_pattern


REVIEW_PROSE = (
    "Dr. An is an excellent neurologist. I waited an hour for an appointment. "
    "She ordered an MRI and an EEG, then explained an unusual finding. "
    "An office staffer called back. Overall an outstanding experience. "
) * 40


def _hits(anchor: str, text: str) -> int:
    return len(re.findall(_anchor_pattern(anchor.lower()), text.lower()))


class TestSurnameIsAnArticle:
    def test_the_bare_surname_matches_the_article_hundreds_of_times(self):
        """The measurement that names the bug. This is not a hypothetical
        collision — it is the dominant signal on the page."""
        assert _hits("an", REVIEW_PROSE) > 300

    def test_the_full_name_matches_nothing_in_review_prose(self):
        """Which is why the bare surname was steering the priority windows
        alone: pages write "Dr. An", not "Dr. Andrea An, MD"."""
        assert _hits("Dr. Andrea An, MD", REVIEW_PROSE) == 0

    def test_the_titled_form_excludes_the_article(self):
        """"Dr. An" cannot collide with prose, and it is what the page says."""
        titled = _hits("dr. an", REVIEW_PROSE)
        assert 0 < titled < _hits("an", REVIEW_PROSE) / 2

    def test_a_function_word_surname_drops_its_bare_anchor(self):
        assert _surname_anchors("An") == ["dr. An", "dr An"]

    def test_an_ordinary_surname_keeps_its_bare_anchor(self):
        """The fix must not cost recall for the 99% case — the bare token is
        still the broadest matcher when a page omits the title."""
        assert _surname_anchors("Hodgson") == ["dr. Hodgson", "dr Hodgson", "Hodgson"]

    def test_short_non_word_surnames_keep_their_bare_anchor(self):
        """The rule is "is it a function word", NOT "is it short". Round 15
        lowered the anchor floor to 2 specifically so these would anchor at
        all; a length rule here would undo that for six of the seven."""
        for surname in ("Ho", "Li", "Ng", "Wu", "Yu", "Oh"):
            assert surname in _surname_anchors(surname), surname

    def test_content_word_surnames_are_deliberately_not_excluded(self):
        """Young, Price, Long and Stone are English words AND real surnames,
        but they appear a handful of times per page — the same order as a real
        surname mention. Excluding them would cost recall for no measured
        gain, so the set is function words only."""
        for surname in ("Young", "Price", "Long", "Stone", "Green"):
            assert surname.lower() not in _FUNCTION_WORD_SURNAMES
            assert surname in _surname_anchors(surname)

    def test_an_empty_surname_yields_no_anchors(self):
        assert _surname_anchors("") == []
        assert _surname_anchors(None) == []


class TestPluralAnchors:
    """A review count is written "70 reviews" / "70 patient ratings" on every
    one of the five platforms. Bounded at both ends, the singular anchors
    matched neither."""

    @pytest.mark.parametrize("anchor,text", [
        ("review", "Read all reviews"),
        ("rating", "70 patient ratings"),
        ("review", "1 review"),
        ("rating", "Overall rating 4.1"),
        ("patient rating", "175 patient ratings"),
    ])
    def test_singular_anchors_match_both_forms(self, anchor, text):
        assert _hits(anchor, text) == 1

    @pytest.mark.parametrize("anchor,text", [
        ("review", "the reviewer was thorough"),
        ("rating", "she was rated highly"),
        ("review", "prereview notes"),
    ])
    def test_only_a_trailing_s_is_admitted(self, anchor, text):
        """Not a stemmer. "reviewer" and "rated" are different parts of speech
        and drift away from the number the anchor is aimed at.

        A hyphenated compound is deliberately NOT in this list: "ratings-based"
        does contain the whole word, the hyphen IS a word boundary, and a probe
        asserting otherwise fails against correct behaviour."""
        assert _hits(anchor, text) == 0

    def test_the_word_boundary_still_holds(self):
        """The plural branch must not reopen what bounding closed: "an" must
        still not match inside "management" or "many"."""
        assert _hits("an", "management of many patients, an evaluation") == 1

    def test_an_anchor_ending_in_punctuation_is_untouched(self):
        """"dr." ends in a period, where a trailing boundary asserts a word
        character follows the punctuation and the anchor stops matching."""
        assert _hits("dr.", "Dr. An was thorough") == 1

    def test_healthgrades_hints_reach_the_review_count(self):
        """The hints aimed at tenure and insurance and named nothing that would
        land on the rating — the one number the blend weighs."""
        hints = _DOMAIN_ANCHOR_HINTS["healthgrades.com"]
        assert any("rating" in hint for hint in hints), hints


class TestSourceYields:
    """`enrichment_sources` claimed to separate four failure modes while
    recording only the fetch. `yielded` records the result."""

    SOURCES = [
        {"url": "https://www.healthgrades.com/physician/dr-andrea-an-2pfjn",
         "kind": "profile", "raw_chars": 44138},
        {"url": "https://doctor.webmd.com/doctor/andrea-an-x-overview",
         "kind": "profile", "raw_chars": 33730},
        {"url": "https://www.vitals.com/doctors/andrea-an-ul3cfp",
         "kind": "profile", "raw_chars": 31690},
    ]

    def test_a_full_pair_is_recorded(self):
        out = _annotate_source_yields(self.SOURCES, [
            {"source_url": "https://doctor.webmd.com/doctor/andrea-an-x-overview",
             "rating": 4.5, "review_count": 61},
        ])
        assert out[1]["yielded"] == {"rating": 4.5, "review_count": 61}

    def test_a_page_that_produced_nothing_reads_none(self):
        out = _annotate_source_yields(self.SOURCES, [])
        assert all(row["yielded"] is None for row in out)

    def test_a_rating_without_a_count_is_visible_as_such(self):
        """THE case this key exists for. A rating-only observation loses the
        same-domain collapse on `has_pair` — the FIRST element of the strength
        tuple — so page kind is never consulted and a directory listing takes
        the headline. Indistinguishable from "produced nothing" before."""
        out = _annotate_source_yields(self.SOURCES, [
            {"source_url": "https://www.healthgrades.com/physician/dr-andrea-an-2pfjn",
             "rating": "4.1 out of 5", "review_count": None},
        ])
        assert out[0]["yielded"] == {"rating": 4.1, "review_count": None}

    def test_the_fetch_fields_survive_annotation(self):
        """The new key is additive — `raw_chars` still separates "Tavily
        returned a thin page" from "we under-read a full one"."""
        out = _annotate_source_yields(self.SOURCES, [])
        assert [row["raw_chars"] for row in out] == [44138, 33730, 31690]
        assert [row["kind"] for row in out] == ["profile"] * 3

    def test_url_matching_tolerates_case_and_a_trailing_slash(self):
        out = _annotate_source_yields(
            [{"url": "https://Www.Vitals.com/doctors/andrea-an-ul3cfp/", "kind": "profile"}],
            [{"source_url": "https://www.vitals.com/doctors/andrea-an-ul3cfp",
              "rating": 3.8, "review_count": 44}],
        )
        assert out[0]["yielded"]["review_count"] == 44

    def test_malformed_input_is_skipped_not_fatal(self):
        out = _annotate_source_yields(
            [None, {"url": "https://www.vitals.com/x"}], [None, "junk", {}]
        )
        assert len(out) == 1 and out[0]["yielded"] is None

    def test_no_sources_yields_an_empty_list(self):
        assert _annotate_source_yields(None, [{"source_url": "x", "rating": 4}]) == []


class TestTheWiring:
    """Helper-only tests let the call site be deleted with the suite green.
    These drive the real methods."""

    @pytest.fixture
    def gatherer(self):
        from unittest.mock import MagicMock, patch
        from agents.data_gatherer import DataGathererAgent
        with patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
            agent = DataGathererAgent()
            agent.tavily_client = MagicMock()
            agent.anthropic_client = MagicMock()
            return agent

    def test_the_enrichment_pass_anchors_on_the_titled_surname(self, gatherer):
        """`_surname_anchors` has to reach `build_excerpt`'s priority_anchors.
        Asserts the bare article is ABSENT and the titled form is PRESENT — the
        two halves of the fix, either of which could be dropped alone."""
        from unittest.mock import MagicMock, patch

        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content[0].text = "{}"
        gatherer.anthropic_client.messages.create.return_value = response

        seen = {}

        def spy(text, **kwargs):
            seen.update(kwargs)
            return "excerpt"

        with patch("agents.data_gatherer.build_excerpt", side_effect=spy):
            gatherer._extract_review_data_only(
                [{"url": "https://www.healthgrades.com/physician/dr-andrea-an-2pfjn",
                  "raw_content": "x" * 5000, "content": "snippet"}],
                "Dr. Andrea An, MD", "Neurology", "Chandler, AZ",
            )

        priority = seen["priority_anchors"]
        assert "dr. An" in priority, priority
        assert "An" not in priority, "the bare article must not steer the windows"
        assert "Andrea An" in priority, "a title-free header needs a specific anchor"

    def test_an_ordinary_surname_still_reaches_the_excerpt_bare(self, gatherer):
        from unittest.mock import MagicMock, patch

        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content[0].text = "{}"
        gatherer.anthropic_client.messages.create.return_value = response

        seen = {}
        with patch("agents.data_gatherer.build_excerpt",
                   side_effect=lambda text, **kw: seen.update(kw) or "excerpt"):
            gatherer._extract_review_data_only(
                [{"url": "https://www.vitals.com/doctors/hodgson",
                  "raw_content": "x" * 5000, "content": "snippet"}],
                "Dr. Sarah Hodgson, MD", "Neurology", "Chandler, AZ",
            )

        assert "Hodgson" in seen["priority_anchors"]

    def test_enrichment_sources_carry_what_each_page_produced(self, gatherer):
        """The composed path: search → extract → annotate. Drives the exact
        2026-07-29 shape — a healthgrades profile that yields a rating with no
        count beside a webmd profile that yields a full pair."""
        from unittest.mock import patch

        provider = {"name": "Dr. Andrea An, MD", "location": "Chandler, AZ"}
        results = [
            {"url": "https://www.healthgrades.com/physician/dr-andrea-an-2pfjn",
             "raw_content": "x" * 44138},
            {"url": "https://doctor.webmd.com/doctor/andrea-an-x-overview",
             "raw_content": "x" * 33730},
        ]
        extracted = {"review_observations": [
            {"source_url": "https://www.healthgrades.com/physician/dr-andrea-an-2pfjn",
             "rating": 4.1, "review_count": None},
            {"source_url": "https://doctor.webmd.com/doctor/andrea-an-x-overview",
             "rating": 4.5, "review_count": 61},
        ]}

        with patch.object(gatherer, "_search_providers", return_value=results), \
             patch.object(gatherer, "_extract_review_data_only", return_value=extracted), \
             patch.object(gatherer, "_merge_review_data", lambda *a, **k: None), \
             patch.object(gatherer, "_classify_enrichment", return_value="enriched"):
            gatherer._enrich_one(provider, "Chandler, AZ", "Neurology")

        sources = {row["url"]: row for row in provider["enrichment_sources"]}
        hg = sources["https://www.healthgrades.com/physician/dr-andrea-an-2pfjn"]
        wm = sources["https://doctor.webmd.com/doctor/andrea-an-x-overview"]

        # The whole diagnosis, readable in one panel row instead of two rounds
        # of inference: the profile WAS fetched (44,138 chars), it DID produce
        # a rating, and it produced no count — so it loses the same-domain
        # collapse before page kind is consulted.
        assert hg["raw_chars"] == 44138 and hg["kind"] == "profile"
        assert hg["yielded"] == {"rating": 4.1, "review_count": None}
        assert wm["yielded"] == {"rating": 4.5, "review_count": 61}
