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


# ---- inventory (the Data Gatherer panel's cache catalogue) ----

class TestInventory:
    """`inventory()` — plaintext catalogue for the cache-miss diagnosis.

    The open defect it services: a repeat search reused 1 of an expected ~5,
    and diagnosing WHY required diffing two runs' coverage panels by hand.
    The inventory shows the stored rows (name, basis, age, flags) so a drifted
    key or an expired row is visible on one screen. Metadata only — a test
    asserting decryption happened here would be asserting a defect.
    """

    def test_rows_come_back_newest_first_with_age_and_expiry(self, store, monkeypatch):
        import utils.vector_store as vs

        real_time = vs.time.time
        # An 8-day-old row (past the 7-day TTL) written first...
        monkeypatch.setattr(vs.time, "time", lambda: real_time() - 8 * 86400)
        store.upsert_enriched_providers(
            [_provider(name="Dr. Old Timer", location="Mesa, AZ")]
        )
        # ...then a fresh row, written now.
        monkeypatch.setattr(vs.time, "time", real_time)
        store.upsert_enriched_providers([_provider()])

        inv = store.inventory()

        assert inv["total"] == 2
        assert [r["name"] for r in inv["rows"]] == [
            "Dr. Andrea An, MD", "Dr. Old Timer",
        ], "newest first — the reader is diagnosing the LAST run, not history"

        fresh, old = inv["rows"]
        assert fresh["expired"] is False
        assert old["expired"] is True, "8 days > the 7-day TTL"
        assert old["age_days"] == pytest.approx(8.0, abs=0.1)
        assert fresh["stored_basis"] == "an andrea|chandler az"
        # No pin drift on this path: the key was minted from the same fields
        # the row stores.
        assert fresh["key_matches_stored_fields"] is True
        assert "raw_data_encrypted" not in fresh, "metadata catalogue, not payload"

    def test_limit_caps_rows_but_total_reports_the_store(self, store):
        store.upsert_enriched_providers([
            _provider(name="Dr. A One", location="Chandler, AZ"),
            _provider(name="Dr. B Two", location="Chandler, AZ"),
            _provider(name="Dr. C Three", location="Chandler, AZ"),
        ])

        inv = store.inventory(limit=1)

        assert inv["total"] == 3
        assert len(inv["rows"]) == 1

    def test_pinned_key_row_reports_the_mismatch_without_calling_it_corrupt(self, store):
        """A ZIP-backfilled row's key legitimately hashes the PRE-enrichment
        location (the pin is the fix for the orphan-row bug), so the stored
        name+location no longer reproduce the key. The inventory must SAY so
        — `key_matches_stored_fields` False — because that row is exactly the
        shape a cache-miss investigation needs to see."""
        from utils.provider_key import CACHE_KEY_FIELD, pin_cache_key

        p = _provider(location="Phoenix, AZ")
        pin_cache_key(p)  # pinned while discovery only knew the city
        p["location"] = "2201 W Fairview St Ste 1, Chandler, AZ 85224"  # enrichment rewrote
        store.upsert_enriched_providers([p])

        inv = store.inventory()

        assert inv["total"] == 1
        row = inv["rows"][0]
        assert row["provider_key"] == p[CACHE_KEY_FIELD]
        assert row["key_matches_stored_fields"] is False

    def test_schema_mismatch_rows_are_shown_and_flagged(self, store):
        """A row a read would skip on CACHE_SCHEMA_VERSION still appears —
        hiding it would make a schema-caused miss look like key drift."""
        store.collection.upsert(
            ids=["feedfacefeedface"],
            documents=["legacy row"],
            metadatas=[{
                "provider_key": "feedfacefeedface",
                "name": "Dr. Legacy Row",
                "specialty": "Neurology",
                "location": "Chandler, AZ",
                "schema_version": "1",
                "enriched_at_epoch": time.time(),
                "enriched_at_iso": "2026-08-01T00:00:00Z",
            }],
            embeddings=[[0.1] * 8],
        )

        inv = store.inventory()

        assert inv["total"] == 1
        assert inv["rows"][0]["schema_mismatch"] is True

    def test_unreadable_store_reports_instead_of_raising(self, store):
        """A diagnostic must never take down the panel it diagnoses."""
        store.collection = Mock()
        store.collection.get.side_effect = RuntimeError("chroma unavailable")

        inv = store.inventory()

        assert inv == {"total": 0, "rows": [], "error": "chroma unavailable"}


def test_two_hits_with_live_observations_both_restore(monkeypatch):
    """The three-day cache outage in one test: the union branch's scratch list
    was named `fresh`, SHADOWING the hits dict the loop iterates against. The
    first hit that carried live discovery observations rebound the name; the
    next iteration's `fresh.get(key)` raised AttributeError on a list; the
    outer except — built for store outages — logged "Cache read failed,
    continuing cold" and every provider after that point silently missed,
    re-enriched, and re-billed, at most ONE such hit surviving per search
    (live: always exactly Vandian, three runs straight, while the store
    passed every durability test).

    The seam is two CONSECUTIVE hits that both carry live observations —
    a shape no single-provider test can walk, which is why three store-level
    reproductions passed while every real run failed."""
    from agents.data_gatherer import DataGathererAgent

    with patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
        gatherer = DataGathererAgent()
    gatherer.tavily_client = MagicMock()
    gatherer.anthropic_client = MagicMock()

    providers = [
        {
            "name": name,
            "location": city,
            "review_observations": [
                {"source_url": f"https://www.vitals.com/doctors/{slug}",
                 "rating": 4.0, "review_count": 20},
            ],
        }
        for name, city, slug in [
            ("Dr. Vardges Vandian, DO", "Gilbert, AZ", "vandian"),
            ("Dr. Nicole Alyce Simpkins, MD", "Chandler, AZ", "simpkins"),
        ]
    ]
    payloads = {
        resolve_cache_key(p): {
            "review_observations": [
                {"source_url": f"https://www.healthgrades.com/physician/{i}",
                 "rating": 4.5, "review_count": 100 + i},
            ],
            "review_summary": "Stored summary.",
            "review_sentiment": "positive",
        }
        for i, p in enumerate(providers)
    }

    store = MagicMock()
    store.get_cached_providers.return_value = (payloads, [])
    monkeypatch.setattr("utils.vector_store.get_vector_store", lambda: store)

    hits = gatherer._apply_cached_enrichment(providers)

    assert hits == 2, "the SECOND hit is the one the shadowed name lost"
    for p in providers:
        assert p["enrichment_outcome"] == "cached", p["name"]
        urls = {o["source_url"] for o in p["review_observations"]}
        assert len(urls) == 2, f"union must keep both platforms for {p['name']}"


def test_failure_outcomes_are_never_cached(monkeypatch):
    """Dr. Raja, 2026-08-08: `outcome: no_profile_found` yet a freshly stamped
    store row. The write filter was `!= "cached"`, and the store's
    substantive-payload guard predates discovery emitting listing-parsed
    observations — so a provider whose own profile was never found carried
    enough discovery "substance" to be cached anyway. Next run the hit
    relabels him `cached`, which passes the recommendation gate the failure
    had correctly failed, and his profile search is not retried for the
    whole TTL. Only `enriched` may be written: failures retry every run,
    cache hits keep their timestamp, over_budget rows never masquerade as
    researched."""
    from agents.data_gatherer import DataGathererAgent

    with patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
        gatherer = DataGathererAgent()

    providers = [
        {"name": "Dr. Kept, MD", "enrichment_outcome": "enriched",
         "review_observations": [{"rating": 4.5, "review_count": 10,
                                  "source_url": "https://vitals.com/doctors/kept"}]},
        {"name": "Dr. Roshan Raja, DO", "enrichment_outcome": "no_profile_found",
         "review_observations": [{"rating": 3.9, "review_count": 12,
                                  "source_url": "https://doctor.webmd.com/listing"}]},
        {"name": "Dr. Hit, MD", "enrichment_outcome": "cached"},
        {"name": "Dr. Deferred, MD", "enrichment_outcome": "over_budget",
         "review_observations": [{"rating": 4.0, "review_count": 8,
                                  "source_url": "https://vitals.com/doctors/deferred"}]},
        {"name": "Dr. Wrong Person", "enrichment_outcome": "identity_rejected"},
        {"name": "Dr. Errored", "enrichment_outcome": "failed"},
    ]

    store = MagicMock()
    store.upsert_enriched_providers.return_value = 1
    monkeypatch.setattr("utils.vector_store.get_vector_store", lambda: store)

    gatherer._store_enrichment(providers)

    written = store.upsert_enriched_providers.call_args.args[0]
    assert [p["name"] for p in written] == ["Dr. Kept, MD"]


def test_undecryptable_rows_are_counted_and_named(store, caplog):
    """A rotated or unset ENCRYPTION_KEY used to kill the whole cache
    INVISIBLY: decrypt returned None, the row joined neither `fresh` nor
    `stale`, nothing logged — and the plaintext inventory is structurally
    blind to it because it never decrypts. The one cache failure with no
    surface anywhere. Now it is counted, keyed into `stale` (so the caller's
    log shows the rows ignored rather than nonexistent), and the warning
    names the likely cause."""
    import logging

    from utils.encryption import DataEncryption

    p = _provider()
    store.upsert_enriched_providers([p])

    # Simulate the key rotating between the write and the read.
    store.encryptor = DataEncryption.__new__(DataEncryption)
    from cryptography.fernet import Fernet
    store.encryptor.cipher = Fernet(Fernet.generate_key())

    with caplog.at_level(logging.WARNING, logger="utils.vector_store"):
        fresh, stale = store.get_cached_providers(
            [{"name": p["name"], "location": p["location"]}]
        )

    assert fresh == {}
    assert stale == [provider_cache_key(p["name"], p["location"])]
    assert any("undecryptable" in r.message and "ENCRYPTION_KEY" in r.message
               for r in caplog.records)


def test_every_read_logs_keys_against_store_rows(store, caplog):
    """Fix D: one INFO line per read — keys asked vs rows in the store vs
    fresh vs stale. Three days of debugging reconstructed exactly these
    numbers from screenshots of downstream panels; the line separates an
    empty store from key drift from rejected rows at a glance."""
    import logging

    store.upsert_enriched_providers([_provider()])

    with caplog.at_level(logging.INFO, logger="utils.vector_store"):
        store.get_cached_providers([
            {"name": "Dr. Andrea An, MD", "location": "Chandler, AZ"},
            {"name": "Dr. Nobody Stored, MD", "location": "Mesa, AZ"},
        ])

    read_lines = [r.message for r in caplog.records if r.message.startswith("Cache read:")]
    assert read_lines, "the read must announce itself"
    assert "2 key(s) against 1 stored row(s)" in read_lines[-1]
    assert "1 fresh" in read_lines[-1]
