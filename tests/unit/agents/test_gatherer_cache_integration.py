"""The cache seam inside enrich_providers.

Unit tests on the store prove a payload round-trips. They cannot prove the
gatherer actually SKIPS live work on a hit, recomputes distance, or refrains
from re-stamping an entry it just read — which is where the cost saving, the
correctness risk, and the eviction behaviour all live.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.data_gatherer import DataGathererAgent
from utils.provider_key import provider_cache_key


@pytest.fixture
def gatherer():
    with patch("agents.data_gatherer.TavilyClient"), patch("agents.data_gatherer.Anthropic"):
        agent = DataGathererAgent()
    agent._enrich_one = MagicMock(name="_enrich_one")
    agent._attach_location_evidence = MagicMock(name="_attach_location_evidence")
    return agent


def _cached_payload():
    return {
        "review_summary": "Cached: thorough, some wait complaints.",
        "review_sentiment": "positive",
        "review_observations": [
            {"platform": "healthgrades", "rating": 4.2, "review_count": 175,
             "source_url": "https://healthgrades.com/physician/dr-a"},
            {"platform": "vitals", "rating": 4.0, "review_count": 28,
             "source_url": "https://vitals.com/doctors/dr-a"},
        ],
        "years_experience": 26,
    }


@pytest.fixture
def store():
    s = MagicMock(name="vector_store")
    s.get_cached_providers.return_value = ({}, [])
    s.upsert_enriched_providers.return_value = 0
    return s


def _run(gatherer, store, providers, **kwargs):
    with patch("utils.vector_store.get_vector_store", return_value=store):
        return gatherer.enrich_providers(providers, location="Chandler, AZ 85249", **kwargs)


def test_cache_hit_skips_the_live_enrichment_search(gatherer, store):
    """The entire point: a hit must remove that provider from the live pass.

    Under the old tiered budget this passed incidentally — a cached provider
    carried two platform pairs, so `_needs_second_opinion` said no. Selection
    is now explicit, which is what keeps this true after the tiers were
    deleted."""
    p = {"name": "Dr. Andrea An, MD", "location": "Chandler, AZ", "rating": 4.2}
    key = provider_cache_key(p["name"], p["location"])
    store.get_cached_providers.return_value = ({key: _cached_payload()}, [])

    _run(gatherer, store, [p])

    assert p["review_summary"].startswith("Cached:")
    assert p["enrichment_outcome"] == "cached"
    gatherer._enrich_one.assert_not_called()


def test_cache_miss_still_enriches_live(gatherer, store):
    p = {"name": "Dr. Nobody Cached, MD", "location": "Chandler, AZ"}
    _run(gatherer, store, [p])
    gatherer._enrich_one.assert_called_once()


def test_distance_is_recomputed_for_the_current_user_location(gatherer, store):
    """Distance is never cached. A stored Chandler distance restored into a
    Phoenix search would be wrong AND would read as measured."""
    p = {"name": "Dr. Andrea An, MD", "location": "Chandler, AZ", "rating": 4.2}
    key = provider_cache_key(p["name"], p["location"])
    store.get_cached_providers.return_value = ({key: _cached_payload()}, [])

    _run(gatherer, store, [p])

    gatherer._attach_location_evidence.assert_called_once()
    assert gatherer._attach_location_evidence.call_args[0][1] == "Chandler, AZ 85249"


def test_a_cache_hit_is_not_written_back(gatherer, store):
    """Re-stamping an entry we just read would refresh its timestamp on every
    search, so a single row could live forever without re-verification."""
    p = {"name": "Dr. Andrea An, MD", "location": "Chandler, AZ", "rating": 4.2}
    key = provider_cache_key(p["name"], p["location"])
    store.get_cached_providers.return_value = ({key: _cached_payload()}, [])

    _run(gatherer, store, [p])

    written = store.upsert_enriched_providers.call_args[0][0]
    assert written == []


def test_freshly_enriched_providers_are_written_back(gatherer, store):
    p = {"name": "Dr. New Person, MD", "location": "Chandler, AZ"}
    _run(gatherer, store, [p])

    written = store.upsert_enriched_providers.call_args[0][0]
    assert [w["name"] for w in written] == ["Dr. New Person, MD"]


def test_use_cache_false_neither_reads_nor_writes(gatherer, store):
    """The sidebar's cold-run switch. It must not silently keep writing, or a
    'cold' verification run would still mutate the store it is checking."""
    p = {"name": "Dr. Andrea An, MD", "location": "Chandler, AZ"}
    _run(gatherer, store, [p], use_cache=False)

    store.get_cached_providers.assert_not_called()
    store.upsert_enriched_providers.assert_not_called()
    gatherer._enrich_one.assert_called_once()


def test_cache_failure_degrades_to_a_cold_run(gatherer, store):
    """A broken cache must never fail a search."""
    store.get_cached_providers.side_effect = Exception("chroma exploded")
    p = {"name": "Dr. Andrea An, MD", "location": "Chandler, AZ"}

    _run(gatherer, store, [p])

    gatherer._enrich_one.assert_called_once()


def test_hits_and_misses_reach_the_cost_tracker(gatherer, store):
    """A hit is invisible in a cost table — it shows up only as calls that did
    not happen — so the counters are the only way to verify a warm run."""
    hit = {"name": "Dr. Andrea An, MD", "location": "Chandler, AZ", "rating": 4.2}
    miss = {"name": "Dr. Uncached, MD", "location": "Chandler, AZ"}
    key = provider_cache_key(hit["name"], hit["location"])
    store.get_cached_providers.return_value = ({key: _cached_payload()}, [])

    from utils.cost_tracker import get_cost_tracker
    get_cost_tracker().reset()

    _run(gatherer, store, [hit, miss])

    cache = get_cost_tracker().summary()["cache"]
    assert cache == {"hits": 1, "misses": 1, "lookups": 2}
