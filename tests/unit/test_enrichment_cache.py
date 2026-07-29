"""Enrichment cache: round-trip, TTL, and what must never be stored.

Exercised against a REAL temp-dir ChromaDB rather than a mock. The defect this
feature exists to fix was a storage-layer one — non-deterministic IDs meant no
lookup could ever hit — and a mocked collection would have happily reported
success for the broken scheme too.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from utils.provider_key import provider_cache_key, resolve_cache_key
from utils.vector_store import CACHE_SCHEMA_VERSION, ProviderVectorStore


pytestmark = pytest.mark.real_vector_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A real persistent Chroma collection in a temp dir, embeddings stubbed."""
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_cache")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ENCRYPTION_KEY", "")

    import utils.config
    utils.config._config_instance = None

    with patch("utils.vector_store.OpenAI"):
        s = ProviderVectorStore()

    # One deterministic vector per document — embeddings are irrelevant to a
    # keyed lookup, which is precisely the point of this design.
    s._get_embeddings_batch = lambda texts: [[0.1] * 8 for _ in texts]
    return s


def _provider(name="Dr. Andrea An, MD", location="Chandler, AZ", **extra):
    base = {
        "name": name,
        "location": location,
        "specialty": "Neurology",
        "review_summary": "Praised for thorough explanations; some wait-time complaints.",
        "review_sentiment": "positive",
        "review_observations": [
            {"platform": "healthgrades", "rating": 4.2, "review_count": 175,
             "source_url": "https://healthgrades.com/physician/dr-andrea-an"}
        ],
        "years_experience": 26,
    }
    base.update(extra)
    return base


# ---- round trip ----

def test_enrichment_round_trips_through_the_store(store):
    p = _provider()
    assert store.upsert_enriched_providers([p]) == 1

    fresh, stale = store.get_cached_providers([{"name": p["name"], "location": p["location"]}])
    key = provider_cache_key(p["name"], p["location"])

    assert stale == []
    assert key in fresh
    assert fresh[key]["review_summary"] == p["review_summary"]
    assert fresh[key]["review_observations"][0]["review_count"] == 175
    assert fresh[key]["years_experience"] == 26


def test_lookup_matches_across_name_and_address_spelling(store):
    """Stored from a directory page with a street address; looked up from a
    city-level candidate with the hyphenated spelling. Same physician."""
    store.upsert_enriched_providers([
        _provider(name="Hussam Seif-Eddeine, MD",
                  location="2979 West Elliot Road Suite 2, Chandler, AZ 85224")
    ])

    fresh, _ = store.get_cached_providers([
        {"name": "Dr. Hussam Seif Eddeine, MD", "location": "Chandler, AZ"}
    ])
    assert len(fresh) == 1


def test_upsert_replaces_rather_than_duplicating(store):
    """The old ID scheme appended a new row per search forever."""
    store.upsert_enriched_providers([_provider(review_summary="First pass.")])
    store.upsert_enriched_providers([_provider(review_summary="Second pass.")])

    assert store.get_collection_stats()["total_providers"] == 1
    fresh, _ = store.get_cached_providers([{"name": "Dr. Andrea An, MD", "location": "Chandler, AZ"}])
    assert list(fresh.values())[0]["review_summary"] == "Second pass."


def test_miss_returns_nothing_for_an_unknown_provider(store):
    store.upsert_enriched_providers([_provider()])
    fresh, stale = store.get_cached_providers([{"name": "Dr. Nobody", "location": "Chandler, AZ"}])
    assert fresh == {} and stale == []


# ---- freshness ----

def _age_stored_entry(store, key, seconds_old):
    """Rewrite one row's timestamp to simulate the passage of time."""
    row = store.collection.get(where={"provider_key": {"$in": [key]}}, include=["metadatas", "documents"])
    meta = dict(row["metadatas"][0])
    meta["enriched_at_epoch"] = time.time() - seconds_old
    store.collection.upsert(ids=[key], documents=row["documents"], metadatas=[meta],
                            embeddings=[[0.1] * 8])


@pytest.mark.parametrize("age_days,expect_hit", [
    (6.95, True),    # just inside a 7-day TTL
    (7.05, False),   # just outside
])
def test_ttl_boundary(store, age_days, expect_hit):
    p = _provider()
    store.upsert_enriched_providers([p])
    key = provider_cache_key(p["name"], p["location"])
    _age_stored_entry(store, key, age_days * 86400)

    fresh, stale = store.get_cached_providers(
        [{"name": p["name"], "location": p["location"]}], ttl_days=7
    )
    assert (key in fresh) is expect_hit
    assert (key in stale) is not expect_hit


def test_ttl_zero_disables_reuse_without_deleting_data(store):
    p = _provider()
    store.upsert_enriched_providers([p])
    fresh, _ = store.get_cached_providers(
        [{"name": p["name"], "location": p["location"]}], ttl_days=0
    )
    assert fresh == {}
    assert store.get_collection_stats()["total_providers"] == 1


def test_incompatible_schema_version_is_treated_as_stale(store):
    """A record shape we no longer understand must not be served."""
    p = _provider()
    store.upsert_enriched_providers([p])
    key = provider_cache_key(p["name"], p["location"])

    row = store.collection.get(where={"provider_key": {"$in": [key]}}, include=["metadatas", "documents"])
    meta = dict(row["metadatas"][0])
    meta["schema_version"] = str(int(CACHE_SCHEMA_VERSION) - 1)
    store.collection.upsert(ids=[key], documents=row["documents"], metadatas=[meta],
                            embeddings=[[0.1] * 8])

    fresh, stale = store.get_cached_providers([{"name": p["name"], "location": p["location"]}])
    assert fresh == {} and key in stale


# ---- what must never be cached ----

@pytest.mark.parametrize("field", [
    "computed_distance_miles", "location_match", "location_evidence",
    "final_score", "base_score", "refined_score", "rank",
])
def test_search_specific_fields_are_never_stored(store, field):
    """Distance depends on the USER's location. A cached Chandler distance
    restored into a Phoenix search would be wrong AND would read as measured
    rather than imputed. Scores depend on per-search weights."""
    p = _provider(**{field: 4.7})
    store.upsert_enriched_providers([p])

    fresh, _ = store.get_cached_providers([{"name": p["name"], "location": p["location"]}])
    assert field not in list(fresh.values())[0]


def test_provider_with_no_enrichment_is_not_stored(store):
    """Storing an empty payload would let a failed lookup masquerade as a fresh
    hit for the whole TTL — strictly worse than a miss."""
    bare = {"name": "Dr. Nothing Found", "location": "Chandler, AZ", "specialty": "Neurology"}
    assert store.upsert_enriched_providers([bare]) == 0
    assert store.get_collection_stats()["total_providers"] == 0


# ---- degradation ----

def test_unreadable_cache_degrades_to_a_cold_run(store):
    """A cache that cannot be read is a cold start, never an error."""
    store.collection = Mock()
    store.collection.get.side_effect = Exception("chroma is unhappy")

    fresh, stale = store.get_cached_providers([{"name": "Dr. A", "location": "Chandler, AZ"}])
    assert fresh == {} and stale == []


def test_unwritable_cache_does_not_fail_the_search(store):
    store.collection = Mock()
    store.collection.upsert.side_effect = Exception("disk full")
    assert store.upsert_enriched_providers([_provider()]) == 0


def test_a_cache_hit_unions_observations_instead_of_replacing_them(monkeypatch):
    """A hit must save a SEARCH, not delete evidence the search already found.

    `provider.update(payload)` replaced `review_observations` wholesale, so a
    cache hit discarded whatever THIS run's discovery pass had extracted. That
    is the fill-if-empty mistake round 4 fixed on the cold path — keeping only
    one side halved platform coverage and starved the blend, which needs two
    platforms to produce anything at all.

    It is reachable whenever the stored row predates a platform today's
    discovery found: the provider ends up scored on fewer platforms than the
    run actually gathered.
    """
    from agents.data_gatherer import DataGathererAgent

    with patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
        gatherer = DataGathererAgent()
    gatherer.tavily_client = MagicMock()
    gatherer.anthropic_client = MagicMock()

    # discovery found vitals THIS run; the stored row only knows healthgrades
    provider = {
        "name": "Dr. Andrea An, MD",
        "location": "Chandler, AZ",
        "review_observations": [
            {"source_url": "https://www.vitals.com/doctors/andrea-an",
             "rating": 3.8, "review_count": 44},
        ],
    }
    cached_payload = {
        "review_observations": [
            {"source_url": "https://www.healthgrades.com/physician/dr-andrea-an-2pfjn",
             "rating": 4.1, "review_count": 70},
        ],
        "review_summary": "Detailed feedback.",
        "review_sentiment": "positive",
    }

    store = MagicMock()
    store.get_cached_providers.return_value = ({resolve_cache_key(provider): cached_payload}, [])
    monkeypatch.setattr("utils.vector_store.get_vector_store", lambda: store)

    gatherer._apply_cached_enrichment([provider])

    urls = {o["source_url"] for o in provider["review_observations"]}
    assert len(urls) == 2, f"both platforms must survive the hit, got {urls}"
    assert provider["blended_platform_count"] == 2, "and the blend must see both"
