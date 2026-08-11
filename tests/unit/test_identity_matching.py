"""URL-primary identity: the profile link decides, the name vetoes.

A name is a weak key and a strong veto; a platform's own profile link is the
reverse. Using the link as the key and the name as the guard is the right way
round, and it buys one thing name comparison can never establish: two different
paths on the SAME platform prove two different physician records.

The rule that can go wrong is that same one. It is the only step in dedup that
can CREATE a duplicate rather than remove one — if a platform ever publishes
two URLs for a single doctor, the veto splits that person in two. Nobody has
checked whether healthgrades does this, so every contradiction is recorded on
three surfaces rather than trusted silently.
"""
from unittest.mock import MagicMock, patch

import pytest

from agents.data_gatherer import DataGathererAgent
from utils.provenance import canonical_profile_url, urls_contradict

HG = "https://www.healthgrades.com/physician/dr-hemant-pandey-xsjwm"
HG_OTHER = "https://www.healthgrades.com/physician/dr-jane-kim-a1"
VITALS = "https://www.vitals.com/doctors/hemant-kumar-pandey-eeltvn"


@pytest.fixture
def gatherer():
    with patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
        agent = DataGathererAgent()
        agent.tavily_client = MagicMock()
        agent.anthropic_client = MagicMock()
        return agent


class TestCanonicalisation:
    """Comparisons are string equality, so one page must reduce to one string."""

    @pytest.mark.parametrize("variant", [
        HG, HG + "/", HG + "?ref=search", HG + "#locations",
        "https://healthgrades.com/physician/dr-hemant-pandey-xsjwm",
        "healthgrades.com/PHYSICIAN/DR-HEMANT-PANDEY-XSJWM",
    ])
    def test_one_page_reached_six_ways_is_one_key(self, variant):
        assert canonical_profile_url(variant) == canonical_profile_url(HG)

    def test_a_missing_or_unusable_url_has_no_key(self):
        for value in (None, "", "   ", "not a url"):
            assert canonical_profile_url(value) in ("", "not a url")


class TestContradiction:
    def test_same_platform_different_path_is_a_contradiction(self):
        """The valuable verdict, and the one no name comparison can produce."""
        assert urls_contradict(HG, HG_OTHER)

    def test_a_DIFFERENT_platform_is_not_a_contradiction(self):
        """The same doctor has a different URL on every platform. Treating that
        as a conflict would block every cross-platform pairing, and the
        two-platform blend would never engage."""
        assert not urls_contradict(HG, VITALS)

    @pytest.mark.parametrize("a,b", [(HG, None), (None, HG), (None, None), (HG, "")])
    def test_a_MISSING_url_is_not_a_contradiction(self, a, b):
        """Roughly half of one platform's listing rows carry no link. A rule
        requiring one would quietly shrink the pool — URL identity is a
        preference, never a requirement."""
        assert not urls_contradict(a, b)

    def test_the_same_url_spelled_differently_does_not_contradict(self):
        assert not urls_contradict(HG, HG + "?ref=search")


class TestDedupeUsesTheUrl:
    def _p(self, name, url=None, **extra):
        row = {"name": name, "specialty": "Neurology", "profile_url": url}
        row.update(extra)
        return row

    def test_the_url_merges_records_whose_names_would_not(self, gatherer):
        """Pass 1. "H. Pandey" and "Dr. Hemant Kumar Pandey, MD" need not clear
        the 0.8 name threshold when both carry the same profile link."""
        out = gatherer._dedupe_providers([
            self._p("Dr. Hemant Kumar Pandey, MD", HG),
            self._p("H. Pandey", HG + "?ref=search"),
        ])

        assert len(out) == 1

    def test_the_url_SEPARATES_records_whose_names_agree(self, gatherer):
        """Pass 2's veto, and the reason the whole feature is worth building.
        These two overlap 1.0 on name tokens and merge today."""
        out = gatherer._dedupe_providers([
            self._p("Dr. Jane Kim", "https://www.healthgrades.com/physician/dr-jane-kim-a1"),
            self._p("Dr. Jane Kim", "https://www.healthgrades.com/physician/dr-john-kim-b2"),
        ])

        assert len(out) == 2, "a name match must not outvote a same-platform URL conflict"

    def test_cross_platform_records_still_merge(self, gatherer):
        """The case a careless veto would break: one doctor, two platforms, and
        necessarily two different URLs. Merging them is what feeds the blend."""
        out = gatherer._dedupe_providers([
            self._p("Dr. Hemant Pandey", HG),
            self._p("Dr. Hemant Pandey, MD", VITALS),
        ])

        assert len(out) == 1

    def test_a_provider_with_no_url_stays_matchable(self, gatherer):
        """Bare vitals headings yield no link at all; they must behave exactly
        as they did before URLs existed."""
        out = gatherer._dedupe_providers([
            self._p("Dr. Brandon Craig Woods", None),
            self._p("Dr. Brandon Woods", "https://www.vitals.com/doctors/brandon-woods-x8k2"),
        ])

        assert len(out) == 1

    def test_an_exact_url_match_found_LATE_beats_a_name_match_found_EARLY(self, gatherer):
        """Why this is two passes and not one loop.

        The order entries arrive in is arbitrary, so a single loop breaking on
        its first hit lets a weak name match seen FIRST win over an exact URL
        match seen SECOND. Entry 0 is on a different platform, so no veto fires
        and one loop really would merge into it.

        Asserted on the URL, not the surviving name: `_field_richness` picks the
        fuller record as the survivor, so which NAME comes out is a property of
        the merge, not of the matching decision under test."""
        out = gatherer._dedupe_providers([
            self._p("Dr. Pat Morgan", VITALS),      # name-matches entry 2 at 1.0
            self._p("Dr. Chris Lee", HG),           # URL-matches entry 2 exactly
            self._p("Dr. Pat Morgan", HG, rating=4.9),
        ])

        assert len(out) == 2
        by_url = {canonical_profile_url(p["profile_url"]): p for p in out}
        assert by_url[canonical_profile_url(HG)].get("rating") == 4.9, (
            "the exact URL match must win over the earlier name match"
        )
        assert by_url[canonical_profile_url(VITALS)].get("rating") is None


class TestContradictionsAreRecorded:
    def _pair(self, gatherer):
        gatherer._dedupe_providers([
            {"name": "Dr. Jane Kim", "specialty": "Neurology",
             "profile_url": "https://www.healthgrades.com/physician/dr-jane-kim-a1"},
            {"name": "Dr. Jane Kim", "specialty": "Neurology",
             "profile_url": "https://www.healthgrades.com/physician/dr-john-kim-b2"},
        ])
        return gatherer._identity_contradictions

    def test_the_pair_is_recorded_with_both_urls(self, gatherer):
        found = self._pair(gatherer)

        assert len(found) == 1
        assert found[0]["domain"] == "healthgrades.com"
        assert found[0]["names_agreed"] is True
        assert found[0]["name_overlap"] == 1.0
        assert "dr-john-kim-b2" in found[0]["url"]
        assert "dr-jane-kim-a1" in found[0]["other_url"]

    def test_the_ordinary_case_is_NOT_recorded(self, gatherer):
        """Two different doctors holding different URLs on one platform is the
        rule working, not a finding — and recording it does not scale.

        The veto used to run BEFORE the name score and record every firing:
        on a parsed-listing pool of ~100 candidates that was 3,363 pairs of
        obviously-different doctors ("Dr. Medaa" vs "Dr. Pillai", overlap 0)
        — pool-size arithmetic, not discovery — burying the 0-3 decisive rows
        the record exists to surface and rendering a 3,363-row JSON dump in
        the developer panel. The veto is now consulted only when the names
        would merge, so every recorded row is decisive by construction. The
        outcome (kept separate) is unchanged; only the recording moved."""
        out = gatherer._dedupe_providers([
            {"name": "Dr. Alpha One", "specialty": "Neurology",
             "profile_url": "https://www.healthgrades.com/physician/dr-alpha-one-a"},
            {"name": "Dr. Beta Two", "specialty": "Neurology",
             "profile_url": "https://www.healthgrades.com/physician/dr-beta-two-b"},
        ])

        assert len(out) == 2, "still two separate providers"
        assert gatherer._identity_contradictions == []

    def test_the_record_is_per_SEARCH_not_per_agent(self, gatherer):
        """The orchestrator reuses one gatherer, so without a reset a
        contradiction from an earlier search would be reported against this
        one."""
        self._pair(gatherer)
        assert gatherer._identity_contradictions

        gatherer.tavily_client.search.return_value = {"results": []}
        gatherer.gather_providers("Neurology", "Chandler, AZ", enrich=False)

        assert gatherer._identity_contradictions == []


class TestTheThreeSurfaces:
    """Log, developer panel, patient panel — the same routing the judge
    findings use, and for the same reason: a signal about OUR pipeline needs a
    developer surface, and only the part a reader can act on reaches them."""

    def _results(self, contradictions):
        return {"agent_outputs": {"data_gatherer": {
            "search_metadata": {"identity_contradictions": contradictions}
        }}}

    def test_the_helper_reads_search_metadata(self):
        import app as app_module
        rows = [{"name": "Dr. A", "names_agreed": True}]

        assert app_module._identity_contradictions(self._results(rows)) == rows
        assert app_module._identity_contradictions({}) == []
        assert app_module._identity_contradictions(self._results("not a list")) == []

    def test_the_patient_note_fires_ONLY_when_the_names_agreed(self):
        """Two different doctors holding different links is the rule working
        and says nothing a reader could act on. Two entries a reader would
        assume are one person does."""
        import app as app_module

        assert app_module._identity_note([{"names_agreed": False}]) == ""
        assert app_module._identity_note([]) == ""
        note = app_module._identity_note([{"names_agreed": True}])
        assert "1 pair" in note

    def test_the_patient_note_carries_the_count_and_NO_names(self):
        """Same rule as the judge note: naming providers would put two real
        doctors beside a sentence about a possible data fault. The raw pairs
        are already on the developer surface."""
        import app as app_module
        note = app_module._identity_note([
            {"names_agreed": True, "name": "Dr. Jane Kim", "other_name": "Dr. John Kim",
             "url": "https://x/a", "other_url": "https://x/b"},
        ])

        assert "1 pair" in note
        assert "Kim" not in note
        assert "https://" not in note
        assert "profile_url" not in note

    def test_the_panel_actually_renders_the_note(self):
        """Wiring, not just the helper — a helper-only test would stay green if
        `{identity_note}` were deleted from the panel's f-string."""
        import inspect

        import app as app_module
        source = inspect.getsource(app_module.render_validation_insights)

        assert "identity_note = _identity_note(" in source
        assert "{identity_note}" in source


class TestHarvestingProfileUrls:
    """Step 3 — fill the gaps from the search enrichment ALREADY runs.

    Discovery learns only the URL of whichever platform's listing surfaced the
    provider, and on vitals often none at all. The obvious fix is a per-domain
    search per provider; it would buy URLs already paid for. The enrichment
    search is platform-restricted and returns the profiles.
    """

    def _results(self, *urls):
        return [{"url": u, "title": "t", "content": "", "raw_content": ""} for u in urls]

    def test_urls_are_taken_from_the_search_results(self, gatherer):
        provider = {"name": "Dr. Hemant Pandey, MD", "profile_url": HG}
        gatherer._harvest_profile_urls(provider, self._results(
            VITALS,
            "https://doctor.webmd.com/doctor/hemant-kumar-pandey-5612ad17-overview",
            "https://www.healthgrades.com/neurology-directory/az-arizona/chandler",  # listing
        ))

        found = provider["platform_profile_urls"]
        assert set(found) == {"healthgrades.com", "vitals.com", "doctor.webmd.com"}
        assert found["vitals.com"] == VITALS

    def test_a_stranger_profile_that_ranked_for_the_query_is_rejected(self, gatherer):
        """The query is name + city and platform-restricted, so it returns OTHER
        doctors' profiles too. A page coming back for a query is not evidence
        that it is about the person queried."""
        provider = {"name": "Dr. Hemant Pandey, MD", "profile_url": HG}
        gatherer._harvest_profile_urls(provider, self._results(
            "https://www.vitals.com/doctors/charanjit-dhillon-abc",
        ))

        assert "vitals.com" not in provider["platform_profile_urls"]

    def test_a_listing_page_is_not_a_profile_url(self, gatherer):
        provider = {"name": "Dr. Hemant Pandey, MD"}
        gatherer._harvest_profile_urls(provider, self._results(
            "https://www.healthgrades.com/neurology-directory/az-arizona/chandler",
        ))

        assert not provider.get("platform_profile_urls")

    def test_discoverys_url_is_never_overwritten(self, gatherer):
        """Discovery's URL came off a listing entry whose heading carried the
        name — stronger evidence than "this ranked for the query"."""
        provider = {"name": "Dr. Hemant Pandey, MD", "profile_url": HG}
        gatherer._harvest_profile_urls(provider, self._results(
            "https://www.healthgrades.com/physician/dr-hemant-pandey-DIFFERENT",
        ))

        assert provider["platform_profile_urls"]["healthgrades.com"] == HG


class TestObservationIdentityByUrl:
    """Step 4 — an enrichment observation's source_url IS a profile page."""

    def test_an_observation_from_another_doctors_profile_is_rejected(self, gatherer):
        """Unlike the name check, this does not depend on the cheapest model
        transcribing anything — the URL is the vendor's."""
        provider = {
            "name": "Dr. Hemant Pandey, MD",
            "platform_profile_urls": {"healthgrades.com": HG},
        }
        obs = {"source_url": HG_OTHER, "rating": 5.0, "review_count": 40}

        assert not gatherer._observation_is_same_person(obs, provider["name"], provider)

    def test_the_providers_own_profile_is_accepted(self, gatherer):
        provider = {
            "name": "Dr. Hemant Pandey, MD",
            "platform_profile_urls": {"healthgrades.com": HG},
        }
        obs = {"source_url": HG + "?ref=search", "rating": 3.8, "review_count": 88}

        assert gatherer._observation_is_same_person(obs, provider["name"], provider)

    def test_a_platform_whose_url_we_do_not_know_falls_back_to_the_name(self, gatherer):
        """Most observations, until enrichment has run. Behaviour must be
        exactly what it was before URLs existed."""
        provider = {"name": "Dr. Hemant Pandey, MD", "platform_profile_urls": {}}
        stranger = {"source_url": VITALS, "page_provider_name": "Dr. Someone Else"}
        same = {"source_url": VITALS, "page_provider_name": "Dr. Hemant Pandey"}

        assert not gatherer._observation_is_same_person(stranger, provider["name"], provider)
        assert gatherer._observation_is_same_person(same, provider["name"], provider)

    def test_a_LISTING_source_is_not_checked_by_url(self, gatherer):
        """A directory page names forty doctors, so its URL says nothing about
        identity — only a profile URL does."""
        provider = {
            "name": "Dr. Hemant Pandey, MD",
            "platform_profile_urls": {"healthgrades.com": HG},
        }
        obs = {"source_url": "https://www.healthgrades.com/neurology-directory/az-arizona/chandler",
               "rating": 3.8, "review_count": 88}

        assert gatherer._observation_is_same_person(obs, provider["name"], provider)

    def test_the_check_is_optional_so_old_callers_still_work(self, gatherer):
        assert gatherer._observation_is_same_person(
            {"source_url": HG, "page_provider_name": "Dr. Hemant Pandey"},
            "Dr. Hemant Pandey, MD",
        )
