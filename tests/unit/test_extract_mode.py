"""Extract-only Tavily mode (TAVILY_MODE) — the August 2026 overhaul workaround.

Tavily's August 2026 search overhaul degraded /search three measured ways
(2026-09-02, live): relevance collapse on the listing queries, include_domains
leaking off-domain at both depths, and results ranking 0.98+ arriving with
EMPTY raw_content while /extract returned the same pages whole. The Chandler
pool shrank ~100+ -> 38 and 7 of 8 researched providers ended
`no_profile_found` on the deployed Space.

Extract mode stops asking a degraded index to FIND pages we can NAME: it
constructs the platforms' own listing/profile URLs and pulls bodies via
/extract. These tests pin the wiring — constructed URLs, the measured
depth/format pins, the absence of search calls, the loud fallback, and the
knob's reversibility (TAVILY_MODE=search restores the original pipeline; the
suite's other files run under it via the conftest pin).
"""

from unittest.mock import MagicMock

import pytest

from agents.data_gatherer import DataGathererAgent
from utils.config import Config
from utils.cost_tracker import CostTracker
from utils.platform_urls import (
    discovery_listing_urls,
    healthgrades_listing_url,
    vitals_listing_url,
    webmd_listing_url,
)


HG_CHANDLER = "https://www.healthgrades.com/neurology-directory/az-arizona/chandler"
WEBMD_CHANDLER = "https://doctor.webmd.com/providers/specialty/neurology/arizona/chandler"
VITALS_CHANDLER = "https://www.vitals.com/neurology/az/chandler"


def _shaped(url, raw="# Dr. Test Person, MD\nRated 4.0 out of 5 from 10 ratings"):
    return {"url": url, "title": url, "content": "", "raw_content": raw, "score": 1.0}


def _provider(**overrides):
    provider = {
        "name": "Dr. Test Person, MD",
        "specialty": "Neurology",
        "location": "Chandler, AZ",
        "review_observations": [],
        "insurance_accepted": [],
        "review_summary": "No reviews available",
        "review_sentiment": "unknown",
    }
    provider.update(overrides)
    return provider


class TestConstructedUrls:
    def test_chandler_neurology_matches_measured_urls(self):
        """The three URLs the 2026-09-02 probe fetched and parsed (114 rows,
        1 credit) — the builders must reproduce them exactly."""
        assert healthgrades_listing_url("Neurology", "Chandler", "AZ") == HG_CHANDLER
        assert webmd_listing_url("Neurology", "Chandler", "AZ") == WEBMD_CHANDLER
        assert vitals_listing_url("Neurology", "Chandler", "AZ") == VITALS_CHANDLER

    def test_discovery_urls_in_listing_domain_order(self):
        assert discovery_listing_urls("Neurology", "Chandler, AZ") == [
            HG_CHANDLER, WEBMD_CHANDLER, VITALS_CHANDLER,
        ]

    def test_multiword_city_slugs(self):
        urls = discovery_listing_urls("Neurology", "Sun Lakes, AZ")
        assert all(u.endswith("/sun-lakes") for u in urls)

    def test_webmd_specialty_vocabulary_override(self):
        """webmd names the discipline, not the colloquial specialty — observed
        live: cardiology pages live under /cardiovascular-disease/."""
        url = webmd_listing_url("Cardiology", "Chandler", "AZ")
        assert "/cardiovascular-disease/" in url
        assert "/cardiology/" not in url

    def test_specialty_casing_is_insensitive(self):
        assert (
            discovery_listing_urls("NEUROLOGY", "Chandler, AZ")
            == discovery_listing_urls("Neurology", "Chandler, AZ")
        )

    def test_unresolvable_location_yields_nothing(self):
        """No city+state, no URLs — the caller reads [] as "fall back to
        search", never a crash or a half-built URL."""
        assert discovery_listing_urls("Neurology", "Nowhere") == []
        assert discovery_listing_urls("Neurology", "") == []

    def test_unknown_state_yields_nothing(self):
        assert healthgrades_listing_url("Neurology", "Chandler", "ZZ") is None
        assert vitals_listing_url("Neurology", "Chandler", "ZZ") is None


class TestConfigKnob:
    def test_default_is_extract(self, monkeypatch):
        """The default flipped 2026-09-02 — deploys heal on merge, and
        TAVILY_MODE=search is the rollback lever."""
        monkeypatch.delenv("TAVILY_MODE", raising=False)
        assert Config().TAVILY_MODE == "extract"

    def test_env_override_honored(self, monkeypatch):
        monkeypatch.setenv("TAVILY_MODE", "search")
        assert Config().TAVILY_MODE == "search"

    def test_unknown_value_falls_back_to_extract(self, monkeypatch):
        """A typo'd knob must land on the default loudly, not select a
        pipeline by accident of string comparison."""
        monkeypatch.setenv("TAVILY_MODE", "hybrid-something")
        gatherer = DataGathererAgent()
        assert gatherer._tavily_mode() == "extract"


class TestExtractPages:
    def _gatherer(self):
        gatherer = DataGathererAgent()
        gatherer.tavily_client = MagicMock()
        return gatherer

    def test_results_shaped_like_search_results(self):
        gatherer = self._gatherer()
        gatherer.tavily_client.extract.return_value = {
            "results": [{"url": HG_CHANDLER, "raw_content": "body text"}],
            "failed_results": [],
        }
        out = gatherer._extract_pages([HG_CHANDLER])
        assert out == [{
            "url": HG_CHANDLER, "title": HG_CHANDLER, "content": "",
            "raw_content": "body text", "score": 1.0,
        }]

    def test_depth_and_format_are_pinned(self):
        """Measured 2026-09-02: advanced returned byte-identical bodies at 2x
        credits, and format="text" cost vitals its rating pair (it lives in a
        markdown LINK). Pinned explicitly so a vendor default flip cannot
        silently swap the parsers' input dialect — the
        TAVILY_CHUNKS_PER_SOURCE lesson applied in advance."""
        gatherer = self._gatherer()
        gatherer.tavily_client.extract.return_value = {"results": [], "failed_results": []}
        gatherer._extract_pages([HG_CHANDLER])
        kwargs = gatherer.tavily_client.extract.call_args.kwargs
        assert kwargs["extract_depth"] == "basic"
        assert kwargs["format"] == "markdown"

    def test_empty_bodies_are_dropped(self):
        """An empty body downstream could only produce the "returned thin"
        failure that a missing entry states more honestly."""
        gatherer = self._gatherer()
        gatherer.tavily_client.extract.return_value = {
            "results": [
                {"url": HG_CHANDLER, "raw_content": ""},
                {"url": WEBMD_CHANDLER, "raw_content": "real body"},
            ],
            "failed_results": [],
        }
        out = gatherer._extract_pages([HG_CHANDLER, WEBMD_CHANDLER])
        assert [r["url"] for r in out] == [WEBMD_CHANDLER]

    def test_batches_chunk_to_the_url_cap(self):
        gatherer = self._gatherer()
        gatherer.tavily_client.extract.return_value = {"results": [], "failed_results": []}
        urls = [f"https://www.vitals.com/doctors/doc-{i}" for i in range(23)]
        gatherer._extract_pages(urls)
        calls = gatherer.tavily_client.extract.call_args_list
        assert [len(c.kwargs["urls"]) for c in calls] == [20, 3]

    def test_duplicate_urls_fetched_once(self):
        gatherer = self._gatherer()
        gatherer.tavily_client.extract.return_value = {"results": [], "failed_results": []}
        gatherer._extract_pages([HG_CHANDLER, HG_CHANDLER, WEBMD_CHANDLER])
        assert gatherer.tavily_client.extract.call_args.kwargs["urls"] == [
            HG_CHANDLER, WEBMD_CHANDLER,
        ]

    def test_retries_once_on_transient_failure(self, monkeypatch):
        monkeypatch.setattr("agents.data_gatherer.time.sleep", lambda s: None)
        gatherer = self._gatherer()
        gatherer.tavily_client.extract.side_effect = [
            ConnectionError("blip"),
            {"results": [{"url": HG_CHANDLER, "raw_content": "body"}], "failed_results": []},
        ]
        out = gatherer._extract_pages([HG_CHANDLER])
        assert len(out) == 1
        assert gatherer.tavily_client.extract.call_count == 2

    def test_failed_results_do_not_raise(self):
        gatherer = self._gatherer()
        gatherer.tavily_client.extract.return_value = {
            "results": [],
            "failed_results": [{"url": HG_CHANDLER, "error": "404"}],
        }
        assert gatherer._extract_pages([HG_CHANDLER]) == []

    def test_no_client_returns_empty(self):
        gatherer = DataGathererAgent()
        gatherer.tavily_client = None
        assert gatherer._extract_pages([HG_CHANDLER]) == []


class TestExtractCredits:
    """Extract bills per URL — 1 credit per 5 at basic depth, rounded up."""

    @pytest.mark.parametrize("urls, credits", [(1, 1), (5, 1), (6, 2), (20, 4)])
    def test_ceiling_math(self, urls, credits):
        tracker = CostTracker()
        tracker.record_tavily_extract(urls)
        assert tracker.summary()["tavily"]["credits"] == credits

    def test_zero_urls_records_nothing(self):
        tracker = CostTracker()
        tracker.record_tavily_extract(0)
        assert tracker.summary()["tavily"]["searches"] == 0


class TestDiscoveryWiring:
    def _gatherer(self, monkeypatch, mode="extract", min_pool="1"):
        monkeypatch.setenv("TAVILY_MODE", mode)
        # Keep the ring quiet unless a test wants it: one provider under the
        # default MIN_CANDIDATE_POOL of 8 would ring out and double the
        # extract calls being asserted on.
        monkeypatch.setenv("MIN_CANDIDATE_POOL", min_pool)
        gatherer = DataGathererAgent()
        gatherer._extract_pages = MagicMock(return_value=[_shaped(HG_CHANDLER)])
        gatherer._extract_provider_data = MagicMock(return_value=[_provider()])
        gatherer._discover_candidates = MagicMock(return_value=[])
        gatherer._search_providers = MagicMock(return_value=[])
        return gatherer

    def test_extract_mode_fetches_constructed_urls_and_never_searches(self, monkeypatch):
        gatherer = self._gatherer(monkeypatch)
        result = gatherer.gather_providers("Neurology", "Chandler, AZ", enrich=False)

        gatherer._extract_pages.assert_called_once()
        assert gatherer._extract_pages.call_args.args[0] == [
            HG_CHANDLER, WEBMD_CHANDLER, VITALS_CHANDLER,
        ]
        gatherer._discover_candidates.assert_not_called()
        gatherer._search_providers.assert_not_called()

        metadata = result["search_metadata"]
        assert metadata["fetch_mode"] == "extract"
        assert metadata["fetch_mode_fallback"] is False
        assert all(q.startswith("extract: ") for q in metadata["queries"])

    def test_thin_pool_rings_out_with_constructed_urls(self, monkeypatch):
        """The ring inherits the fetch mode: nearby cities' listing pages are
        just as nameable as the home city's."""
        gatherer = self._gatherer(monkeypatch, min_pool="8")
        gatherer.gather_providers("Neurology", "Chandler, AZ", enrich=False)

        assert gatherer._extract_pages.call_count == 2
        ring_urls = gatherer._extract_pages.call_args_list[1].args[0]
        assert ring_urls, "ring pass fetched nothing"
        assert all("neurology" in u for u in ring_urls)
        assert all(u != h for u in ring_urls for h in
                   (HG_CHANDLER, WEBMD_CHANDLER, VITALS_CHANDLER))
        gatherer._discover_candidates.assert_not_called()

    def test_zero_extract_pages_falls_back_to_search_loudly(self, monkeypatch):
        """Constructed slugs missing for a specialty/city must degrade to a
        search run, flagged in metadata — never an empty page by default."""
        gatherer = self._gatherer(monkeypatch)
        gatherer._extract_pages = MagicMock(return_value=[])
        gatherer._discover_candidates = MagicMock(return_value=[_shaped(HG_CHANDLER)])

        result = gatherer.gather_providers("Neurology", "Chandler, AZ", enrich=False)

        gatherer._discover_candidates.assert_called_once()
        metadata = result["search_metadata"]
        assert metadata["fetch_mode"] == "extract"
        assert metadata["fetch_mode_fallback"] is True
        # The panel shows both what was attempted and what ran.
        assert any(q.startswith("extract: ") for q in metadata["queries"])
        assert any(not q.startswith("extract: ") for q in metadata["queries"])

    def test_search_mode_keeps_the_original_pipeline(self, monkeypatch):
        """The rollback lever: TAVILY_MODE=search must run the search-driven
        discovery untouched and never call extract."""
        gatherer = self._gatherer(monkeypatch, mode="search")
        gatherer._discover_candidates = MagicMock(return_value=[_shaped(HG_CHANDLER)])

        result = gatherer.gather_providers("Neurology", "Chandler, AZ", enrich=False)

        gatherer._extract_pages.assert_not_called()
        gatherer._discover_candidates.assert_called_once()
        assert result["search_metadata"]["fetch_mode"] == "search"


class TestEnrichmentWiring:
    HG_PROFILE = "https://www.healthgrades.com/physician/dr-test-person-abc12"
    WEBMD_PROFILE = (
        "https://doctor.webmd.com/doctor/test-person-0000-1111-overview"
    )
    VITALS_PROFILE = "https://www.vitals.com/doctors/Dr_Test_Person.html"
    HG_LISTING = HG_CHANDLER

    def _gatherer(self, monkeypatch, mode="extract"):
        monkeypatch.setenv("TAVILY_MODE", mode)
        gatherer = DataGathererAgent()
        gatherer._search_providers = MagicMock(return_value=[])
        gatherer._extract_review_data_only = MagicMock(return_value={
            "review_observations": [],
            "review_summary": "",
            "review_sentiment": "unknown",
        })
        return gatherer

    def test_known_profile_urls_one_per_platform_profiles_only(self, monkeypatch):
        gatherer = self._gatherer(monkeypatch)
        provider = _provider(
            platform_profile_urls={"healthgrades.com": self.HG_PROFILE},
            profile_url=self.VITALS_PROFILE,
            review_observations=[
                {"source_url": self.WEBMD_PROFILE},
                {"source_url": self.HG_LISTING},              # listing: excluded
                {"source_url": "https://example.com/dr-foo"}, # off-platform: excluded
                {"source_url": self.HG_PROFILE},              # duplicate domain: first wins
            ],
        )
        urls = gatherer._known_profile_urls(provider)
        assert sorted(urls) == sorted([
            self.HG_PROFILE, self.VITALS_PROFILE, self.WEBMD_PROFILE,
        ])

    def test_extract_mode_fetches_known_urls_and_never_searches(self, monkeypatch):
        gatherer = self._gatherer(monkeypatch)
        gatherer._extract_pages = MagicMock(
            return_value=[_shaped(self.HG_PROFILE, raw="profile body")]
        )
        provider = _provider(
            platform_profile_urls={
                "healthgrades.com": self.HG_PROFILE,
                "doctor.webmd.com": self.WEBMD_PROFILE,
            }
        )
        gatherer._enrich_one(provider, "Chandler, AZ", "Neurology", "Chandler, AZ")

        called_with = gatherer._extract_pages.call_args.args[0]
        assert sorted(called_with) == sorted([self.HG_PROFILE, self.WEBMD_PROFILE])
        gatherer._search_providers.assert_not_called()

    def test_no_known_urls_is_an_honest_gap_not_a_search(self, monkeypatch):
        """Extract mode never falls back to the name search — that is the
        broken path. A provider with no attributed URL records
        no_profile_found, the same honest gap the card already explains."""
        gatherer = self._gatherer(monkeypatch)
        gatherer._extract_pages = MagicMock(return_value=[])
        provider = _provider()

        gatherer._enrich_one(provider, "Chandler, AZ", "Neurology", "Chandler, AZ")

        assert provider["enrichment_outcome"] == "no_profile_found"
        gatherer._extract_pages.assert_not_called()
        gatherer._search_providers.assert_not_called()

    def test_search_mode_enrichment_still_searches(self, monkeypatch):
        gatherer = self._gatherer(monkeypatch, mode="search")
        gatherer._extract_pages = MagicMock(return_value=[])
        provider = _provider(
            platform_profile_urls={"healthgrades.com": self.HG_PROFILE}
        )
        gatherer._enrich_one(provider, "Chandler, AZ", "Neurology", "Chandler, AZ")

        gatherer._search_providers.assert_called_once()
        gatherer._extract_pages.assert_not_called()


class TestDedupeKeepsEveryPlatformUrl:
    """The first extract-mode Space run showed every card on vitals +
    healthgrades and webmd on NONE. webmd's listing rows carry a profile URL
    but no rating pair, so they create no observation; the dedupe merge then
    kept the richer row's scalar profile_url and dropped webmd's. Search mode
    had masked this — the name search re-found webmd for free."""

    VITALS = "https://www.vitals.com/doctors/dan-joseph-capampangan-rtzjkh"
    WEBMD = (
        "https://doctor.webmd.com/doctor/dan-joseph-capampangan-"
        "18f93eea-5285-4ff6-ae91-69de813326b7-overview"
    )
    HG_LISTING = "https://www.healthgrades.com/neurology-directory/az-arizona/chandler"

    def _rows(self):
        rich = _provider(
            name="Dr. Dan Capampangan",
            profile_url=self.VITALS,
            rating=4.9, review_count=17,
            review_observations=[{
                "source_url": self.VITALS, "rating": 4.9, "review_count": 17,
                "page_provider_name": "Dr. Dan Capampangan",
            }],
        )
        bare = _provider(
            name="Dr. Dan Joseph Capampangan, MD",
            profile_url=self.WEBMD,
            rating=0, review_count=None,
        )
        return rich, bare

    def test_merge_unions_the_duplicates_platform_url(self, monkeypatch):
        monkeypatch.setenv("TAVILY_MODE", "extract")
        gatherer = DataGathererAgent()
        rich, bare = self._rows()

        merged = gatherer._dedupe_providers([rich, bare])

        assert len(merged) == 1
        urls = merged[0]["platform_profile_urls"]
        assert urls["vitals.com"] == self.VITALS
        assert urls["doctor.webmd.com"] == self.WEBMD

    def test_merge_order_does_not_matter(self, monkeypatch):
        monkeypatch.setenv("TAVILY_MODE", "extract")
        gatherer = DataGathererAgent()
        rich, bare = self._rows()

        merged = gatherer._dedupe_providers([bare, rich])

        assert set(merged[0]["platform_profile_urls"].values()) == {self.VITALS, self.WEBMD}

    def test_listing_urls_never_become_profile_urls(self, monkeypatch):
        monkeypatch.setenv("TAVILY_MODE", "extract")
        gatherer = DataGathererAgent()
        rich, bare = self._rows()
        bare["profile_url"] = self.HG_LISTING  # a row whose only link was the index

        merged = gatherer._dedupe_providers([rich, bare])

        assert self.HG_LISTING not in merged[0].get("platform_profile_urls", {}).values()

    def test_extract_enrichment_then_fetches_webmd_too(self, monkeypatch):
        """Composed: after the merge, extract-mode enrichment names all
        platforms — the exact page the Space run never fetched."""
        monkeypatch.setenv("TAVILY_MODE", "extract")
        gatherer = DataGathererAgent()
        rich, bare = self._rows()
        merged = gatherer._dedupe_providers([rich, bare])[0]

        assert sorted(gatherer._known_profile_urls(merged)) == sorted([self.VITALS, self.WEBMD])


class TestHealthgradesPaginationWiring:
    """healthgrades lists 20 per page; Chandler neurology is 81 results over
    five pages, and page 1's twenty overlapped the webmd/vitals pool by four.
    The page's own "We found N results" decides how many more pages to fetch
    — nothing speculative — and the extra URLs are recorded for the panel."""

    def _gatherer(self, monkeypatch, page_one_raw):
        monkeypatch.setenv("TAVILY_MODE", "extract")
        monkeypatch.setenv("MIN_CANDIDATE_POOL", "1")
        gatherer = DataGathererAgent()
        gatherer._extract_pages = MagicMock(side_effect=[
            [_shaped(HG_CHANDLER, raw=page_one_raw), _shaped(WEBMD_CHANDLER), _shaped(VITALS_CHANDLER)],
            [_shaped(HG_CHANDLER + "_2")],
        ])
        gatherer._extract_provider_data = MagicMock(return_value=[_provider()])
        gatherer._discover_candidates = MagicMock(return_value=[])
        gatherer._search_providers = MagicMock(return_value=[])
        return gatherer

    def test_stated_count_fetches_pages_two_to_five(self, monkeypatch):
        gatherer = self._gatherer(
            monkeypatch, '## We found 81 results within 10 miles for "Neurologists near Chandler, AZ"'
        )
        result = gatherer.gather_providers("Neurology", "Chandler, AZ", enrich=False)

        assert gatherer._extract_pages.call_count == 2
        later = gatherer._extract_pages.call_args_list[1].args[0]
        assert later == [HG_CHANDLER + f"_{n}" for n in (2, 3, 4, 5)]
        queries = result["search_metadata"]["queries"]
        assert f"extract: {HG_CHANDLER}_2" in queries and f"extract: {HG_CHANDLER}_5" in queries
        gatherer._discover_candidates.assert_not_called()

    def test_one_page_of_results_fetches_nothing_more(self, monkeypatch):
        gatherer = self._gatherer(
            monkeypatch, '## We found 12 results within 10 miles for "Neurologists near Chandler, AZ"'
        )
        gatherer.gather_providers("Neurology", "Chandler, AZ", enrich=False)
        assert gatherer._extract_pages.call_count == 1


class TestAllPlatformPaginationWiring:
    """One extra batch carries every platform's later pages, each in its own
    URL shape, planned from each page's own stated total."""

    def test_three_platforms_paginate_in_one_extra_batch(self, monkeypatch):
        monkeypatch.setenv("TAVILY_MODE", "extract")
        monkeypatch.setenv("MIN_CANDIDATE_POOL", "1")
        gatherer = DataGathererAgent()
        gatherer._extract_pages = MagicMock(side_effect=[
            [
                _shaped(HG_CHANDLER, raw='## We found 81 results within 10 miles for "Neurologists near Chandler, AZ"'),
                _shaped(WEBMD_CHANDLER, raw="Chandler, AZ has **142 Neurologist** results with an average of **31 years**"),
                _shaped(VITALS_CHANDLER, raw="# 142 Neurologists in Chandler, AZ"),
            ],
            [],
        ])
        gatherer._extract_provider_data = MagicMock(return_value=[_provider()])
        gatherer._discover_candidates = MagicMock(return_value=[])
        gatherer._search_providers = MagicMock(return_value=[])

        result = gatherer.gather_providers("Neurology", "Chandler, AZ", enrich=False)

        assert gatherer._extract_pages.call_count == 2
        later = gatherer._extract_pages.call_args_list[1].args[0]
        assert later == (
            [HG_CHANDLER + f"_{n}" for n in (2, 3, 4, 5)]
            + [WEBMD_CHANDLER + f"?pagenumber={n}" for n in (2, 3)]
            + [VITALS_CHANDLER + f"?page={n}" for n in (2, 3)]
        )
        queries = result["search_metadata"]["queries"]
        assert f"extract: {VITALS_CHANDLER}?page=3" in queries
