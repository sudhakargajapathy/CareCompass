"""Phase 2: uniform enrichment, and an outcome recorded for every provider.

The tiered budget these replace never actually rationed anything at real pool
sizes — MAX_PROVIDERS_TO_ENRICH is 10 and the 2026-07-25 field run produced a
pool of exactly 10. Rank 6 arrived with no blended rating because his search
found nothing usable, not because he was skipped. Selection was never the
problem; success was, and it was invisible.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.data_gatherer import DataGathererAgent


@pytest.fixture
def gatherer():
    with patch("agents.data_gatherer.TavilyClient"), patch("agents.data_gatherer.Anthropic"):
        return DataGathererAgent()


def _p(name, **extra):
    base = {"name": name, "location": "Chandler, AZ", "specialty": "Neurology"}
    base.update(extra)
    return base


# ---- uniformity ----

def test_every_uncached_provider_is_enriched_regardless_of_coverage(gatherer):
    """The inversion of the old tier behaviour. A provider arriving with two
    platform pairs used to be exempt; now nothing buys an exemption, so the
    scorer never compares an enriched provider against an unexamined one."""
    well_covered = _p("Dr. Covered", review_observations=[
        {"source_url": "https://healthgrades.com/physician/a", "rating": 4.6, "review_count": 33},
        {"source_url": "https://vitals.com/doctors/a", "rating": 4.4, "review_count": 21},
    ], rating=4.6, review_summary="Praised.", review_sentiment="positive")
    bare = _p("Dr. Bare")

    seen = []
    with patch.object(gatherer, "_enrich_one", side_effect=lambda p, *a, **k: seen.append(p["name"])):
        gatherer._enrich_missing_reviews([well_covered, bare], "Chandler, AZ", "Neurology")

    assert sorted(seen) == ["Dr. Bare", "Dr. Covered"]


def test_cached_providers_are_excluded_from_the_live_pass(gatherer):
    """Under the tiers this was incidental — a cached provider happened to hold
    two pairs. With tiers gone it has to be explicit, or the cache would pay
    for a search it just avoided."""
    cached = _p("Dr. Cached", enrichment_outcome="cached")
    fresh = _p("Dr. Fresh")

    seen = []
    with patch.object(gatherer, "_enrich_one", side_effect=lambda p, *a, **k: seen.append(p["name"])):
        gatherer._enrich_missing_reviews([cached, fresh], "Chandler, AZ", "Neurology")

    assert seen == ["Dr. Fresh"]


def test_all_cached_means_no_live_work_at_all(gatherer):
    providers = [_p(f"Dr. C{i}", enrichment_outcome="cached") for i in range(4)]
    seen = []
    with patch.object(gatherer, "_enrich_one", side_effect=lambda p, *a, **k: seen.append(p["name"])):
        gatherer._enrich_missing_reviews(providers, "Chandler, AZ", "Neurology")
    assert seen == []


# ---- the cap is a guard, and it is never silent ----

def test_cap_truncates_but_marks_what_it_dropped(gatherer):
    """A silent cap reads as 'everyone was covered' when they were not."""
    gatherer.config.MAX_PROVIDERS_TO_ENRICH = 3
    providers = [_p(f"Dr. N{i}") for i in range(5)]

    seen = []
    with patch.object(gatherer, "_enrich_one", side_effect=lambda p, *a, **k: seen.append(p["name"])):
        gatherer._enrich_missing_reviews(providers, "Chandler, AZ", "Neurology")

    assert len(seen) == 3
    assert [p["enrichment_outcome"] for p in providers[3:]] == ["over_budget", "over_budget"]


def test_cap_does_not_bind_at_normal_pool_size(gatherer):
    """The observed pool is 10 and the cap is 10 — the guard should not fire in
    ordinary operation, which is precisely why the tiers rationed nothing."""
    gatherer.config.MAX_PROVIDERS_TO_ENRICH = 10
    providers = [_p(f"Dr. N{i}") for i in range(10)]

    seen = []
    with patch.object(gatherer, "_enrich_one", side_effect=lambda p, *a, **k: seen.append(p["name"])):
        gatherer._enrich_missing_reviews(providers, "Chandler, AZ", "Neurology")

    assert len(seen) == 10
    assert not any(p.get("enrichment_outcome") == "over_budget" for p in providers)


# ---- outcome classification ----

def test_usable_observations_classify_as_enriched(gatherer):
    review_data = {
        "review_summary": "Thorough and kind.",
        "review_observations": [
            {"page_provider_name": "Dr. Andrea An", "source_url": "https://healthgrades.com/a",
             "rating": 4.2, "review_count": 175},
        ],
    }
    assert gatherer._classify_enrichment("Dr. Andrea An, MD", review_data) == "enriched"


def test_summary_alone_still_counts_as_enriched(gatherer):
    """A narrative with no numeric pair is still evidence the judge reads."""
    review_data = {"review_summary": "Consistently praised for clear explanations.",
                   "review_observations": []}
    assert gatherer._classify_enrichment("Dr. Andrea An, MD", review_data) == "enriched"


def test_pages_found_but_all_rejected_is_identity_rejected(gatherer):
    """Separated from no_profile_found on purpose: this points at the identity
    guard or the query, whereas an empty result set points at coverage."""
    review_data = {
        "review_summary": "No reviews available",
        "review_observations": [
            {"page_provider_name": "Dr. Someone Else Entirely", "source_url": "https://healthgrades.com/x",
             "rating": 4.9, "review_count": 12},
        ],
    }
    assert gatherer._classify_enrichment("Dr. Andrea An, MD", review_data) == "identity_rejected"


def test_nothing_extracted_is_no_profile_found(gatherer):
    review_data = {"review_summary": "No reviews available", "review_observations": []}
    assert gatherer._classify_enrichment("Dr. Andrea An, MD", review_data) == "no_profile_found"


# ---- outcomes reach the provider ----

def test_empty_search_records_no_profile_found(gatherer):
    p = _p("Dr. Invisible")
    with patch.object(gatherer, "_search_providers", return_value=[]):
        gatherer._enrich_one(p, "Chandler, AZ", "Neurology")
    assert p["enrichment_outcome"] == "no_profile_found"


def test_exception_records_failed(gatherer):
    """A transient failure must be distinguishable from a genuine absence of
    data — they call for different responses."""
    p = _p("Dr. Unlucky")
    with patch.object(gatherer, "_search_providers", side_effect=Exception("tavily down")):
        gatherer._enrich_one(p, "Chandler, AZ", "Neurology")
    assert p["enrichment_outcome"] == "failed"


def test_nameless_provider_records_failed(gatherer):
    p = {"location": "Chandler, AZ"}
    gatherer._enrich_one(p, "Chandler, AZ", "Neurology")
    assert p["enrichment_outcome"] == "failed"


def test_successful_pass_records_enriched(gatherer):
    p = _p("Dr. Andrea An, MD")
    review_data = {
        "review_summary": "Thorough and kind.",
        "review_sentiment": "positive",
        "review_observations": [
            {"page_provider_name": "Dr. Andrea An", "source_url": "https://healthgrades.com/a",
             "rating": 4.2, "review_count": 175},
        ],
    }
    with patch.object(gatherer, "_search_providers", return_value=[{"url": "https://healthgrades.com/a"}]), \
         patch.object(gatherer, "_extract_review_data_only", return_value=review_data):
        gatherer._enrich_one(p, "Chandler, AZ", "Neurology")

    assert p["enrichment_outcome"] == "enriched"
