"""Regression guards for the second review round.

Every test here drives the exact input that reproduced the defect. Several of
these bugs survived the existing suite because its fixtures were synthetic —
a three-digit street number, a provider dict without the placeholders the
extractor always writes, a validation entry whose name was already normalized.
Where that was the cause, the fixture below uses the real shape instead.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.critic_validator import _verdict_class, is_judge_concern
from agents.data_gatherer import DataGathererAgent, _blended_platform_rating
from agents.preference_scorer import _same_provider_name
from utils.excerpt import strip_boilerplate
from utils.provider_key import pin_cache_key, provider_cache_key, resolve_cache_key
from utils.vector_store import ProviderVectorStore


def _gatherer():
    with patch("agents.data_gatherer.TavilyClient"), patch("agents.data_gatherer.Anthropic"):
        return DataGathererAgent()


# --------------------------------------------------------------------------
# A cache hit must reproduce a cold run's score.
# --------------------------------------------------------------------------

TWO_PLATFORM_OBS = [
    {"platform": "healthgrades.com", "rating": 4.2, "review_count": 175,
     "source_url": "https://www.healthgrades.com/physician/dr-a-1"},
    {"platform": "vitals.com", "rating": 4.0, "review_count": 28,
     "source_url": "https://www.vitals.com/doctors/dr-a"},
]


def _hit_provider(payload):
    """Run one provider through enrich_providers against a stubbed cache hit."""
    g = _gatherer()
    provider = {"name": "Dr. Andrea An, MD", "location": "Chandler, AZ 85224",
                "specialty": "Neurology"}
    store = MagicMock()
    store.get_cached_providers.return_value = ({"k": payload}, [])
    store.upsert_enriched_providers.return_value = 1
    with patch("utils.vector_store.get_vector_store", lambda: store), \
         patch("agents.data_gatherer.resolve_cache_key", lambda p: "k"), \
         patch("agents.data_gatherer.pin_cache_key", lambda p: "k"):
        return g.enrich_providers([provider], "Chandler, AZ 85249", specialty="Neurology")[0]


def test_cache_hit_rebuilds_the_cross_platform_blend():
    """The blend is DERIVED from review_observations and is not cached, so a
    hit that restores the observations but not the blend scored the provider
    as though no platform evidence existed."""
    p = _hit_provider({"review_observations": TWO_PLATFORM_OBS,
                       "review_summary": "Warm and thorough.",
                       "review_sentiment": "positive"})
    assert p["enrichment_outcome"] == "cached"
    assert p["blended_rating"] == 4.2
    assert p["blended_review_count"] == 203
    assert p["blended_platform_count"] == 2


def test_cache_hit_matches_a_cold_merge_exactly():
    """Warm must equal cold. A cache that changes the ranking is worse than
    no cache — the stated acceptance bar for the whole feature."""
    warm = _hit_provider({"review_observations": TWO_PLATFORM_OBS,
                          "review_summary": "Warm.", "review_sentiment": "positive"})
    cold = {"name": "Dr. Andrea An, MD", "location": "Chandler, AZ 85224"}
    _gatherer()._merge_review_data(
        cold, {"review_observations": TWO_PLATFORM_OBS, "review_summary": "Warm.",
               "review_sentiment": "positive"})
    for field in ("blended_rating", "blended_review_count", "blended_platform_count",
                  "rating", "review_count"):
        assert warm.get(field) == cold.get(field), field


def test_cache_hit_applies_the_platform_headline():
    p = _hit_provider({"review_observations": TWO_PLATFORM_OBS,
                       "review_summary": "Warm.", "review_sentiment": "positive"})
    assert p["rating"] == 4.2
    assert p["review_count"] == 175


# --------------------------------------------------------------------------
# The cache key must survive enrichment rewriting `location`.
# --------------------------------------------------------------------------

def test_cache_key_survives_an_address_backfill():
    """Enrichment replaces a city with a street address when that gains ZIP
    precision. Recomputing the key at write time then stored the row where no
    later read would look — a permanent miss plus an orphan row per run, for
    exactly the providers enrichment helped most."""
    p = {"name": "Dr. Andrea An, MD", "location": "Phoenix, AZ"}
    pinned = pin_cache_key(p)
    p["location"] = "1234 W Frye Rd, Chandler, AZ 85224"   # what enrichment does
    assert resolve_cache_key(p) == pinned


def test_unpinned_provider_still_resolves_a_key():
    """Backwards compatibility: no pin (cache disabled) must not raise."""
    p = {"name": "Dr. B", "location": "Mesa, AZ"}
    assert resolve_cache_key(p) == provider_cache_key("Dr. B", "Mesa, AZ")


# --------------------------------------------------------------------------
# The extractor's placeholders are not evidence.
# --------------------------------------------------------------------------

def test_a_provider_whose_enrichment_found_nothing_is_not_cached():
    """`_extract_provider_data` sets review_summary="No reviews available" and
    review_sentiment="unknown" on EVERY provider, and both are substantive
    fields — so the emptiness test admitted them and the guard never fired for
    the case it exists to catch. The old fixture omitted both keys."""
    quiet = {"name": "Dr. Quiet", "location": "Phoenix, AZ", "specialty": "Neurology",
             "review_summary": "No reviews available", "review_sentiment": "unknown"}
    assert ProviderVectorStore.cacheable_payload(quiet) == {}


def test_a_provider_with_real_evidence_is_still_cached():
    real = {"name": "Dr. Real", "location": "Phoenix, AZ",
            "review_summary": "Patients praise her thoroughness.",
            "review_sentiment": "positive"}
    assert ProviderVectorStore.cacheable_payload(real)


def test_placeholder_never_reaches_the_stored_payload():
    """Otherwise a cached "No reviews available" overwrites a real summary a
    later candidate pass found."""
    mixed = {"name": "Dr. Mixed", "location": "Phoenix, AZ",
             "review_observations": TWO_PLATFORM_OBS,
             "review_summary": "No reviews available", "review_sentiment": "unknown"}
    payload = ProviderVectorStore.cacheable_payload(mixed)
    assert "review_observations" in payload
    assert "review_summary" not in payload


# --------------------------------------------------------------------------
# is_judge_concern — both directions.
# --------------------------------------------------------------------------

MIXED_VERDICTS = [
    "Scoring matches the evidence for review_substance, but practical_access at 10 "
    "ignores the summary line about 90-minute waits.",
    "No correction needed for red_flags, but practical_access should have been lowered.",
]

NULL_ANSWERS = ["None", "None.", "N/A", "-", "...", "No adjustments.",
                "No adjustments needed.", "No changes required.", "nothing"]

CLEAN_PASSES = [
    "Judge scoring matches the evidence.",
    "No correction needed.",
    "Judge did not misread anything here; the neutral score is warranted. No correction needed.",
    "Judge scoring matches evidence; practical_access at 14 fairly reflects "
    "mixed-but-mostly-positive scheduling feedback.",
]


@pytest.mark.parametrize("verdict", MIXED_VERDICTS)
def test_a_mixed_verdict_is_kept(verdict):
    """The strong-pass test used to run first, so a verdict opening with an
    all-clear and then reporting a real problem was dropped from all three
    destinations — the under-reporting the predicate must never do."""
    assert is_judge_concern(verdict) is True


@pytest.mark.parametrize("answer", NULL_ANSWERS)
def test_a_null_answer_is_not_a_concern(answer):
    """The prompt asks for "". A model writing "None" complies in spirit, but
    neither string matched a pass pattern, so default-to-concern reproduced
    the false patient-facing count through the guard rather than around it."""
    assert is_judge_concern(answer) is False


@pytest.mark.parametrize("verdict", CLEAN_PASSES)
def test_a_clean_pass_is_still_dropped(verdict):
    """Including the hyphenated-compound case: a bare \\bbut\\b matched inside
    "mixed-but-mostly-positive" and turned a real live PASS into a finding."""
    assert is_judge_concern(verdict) is False


def test_ten_null_answers_produce_a_silent_panel():
    """The 2026-07-25 shape: every provider returning a null answer must count
    zero, not ten."""
    assert sum(is_judge_concern(a) for a in ["None"] * 10) == 0


# --------------------------------------------------------------------------
# Critic verdict classification.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    ("approved", "approved"),
    ("conditional", "conditional"),
    ("rejected", "rejected"),
    ("conditional approval", "conditional"),
    ("approved with conditions", "conditional"),
    ("not approved", "other"),
    ("unapproved", "other"),
    ("disapproved", "other"),
    ("not rejected", "other"),
])
def test_verdict_classification(status, expected):
    """Bare substring tests misread their own negations: "conditional
    approval" escaped the -8 entirely and scored as a clean approval, while
    "not rejected" — an explicit clearing — took the full -15."""
    assert _verdict_class(status) == expected


# --------------------------------------------------------------------------
# Excerpting must not delete the data extraction exists to find.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "4.9 (127)",
    "Rating: 4.8/5 (127)",
    "4.0 (31 reviews)",
    "4.9 out of 5",
    "480-555-1234",
    "Dr. An has 26 years of experience.",
    "30+ years in practice",
    "[Dr. Andrea An, MD](https://www.healthgrades.com/physician/dr-andrea-an-3xyz9)",
])
def test_signal_lines_survive_boilerplate_stripping(line):
    """A profile header states its numbers compactly, which reads as
    "symbol-heavy"; a directory entry is a single markdown link, which read as
    a nav row. Both filters were deleting the highest-value lines on the page
    before the extractor ever saw them."""
    assert line in strip_boilerplate(line)


@pytest.mark.parametrize("line", [
    "[Find a Doctor](https://x.com/find) | [Locations](https://x.com/l) | [Jobs](https://x.com/j)",
    "• • • • •",
    "----------",
])
def test_real_chrome_is_still_dropped(line):
    assert line not in strip_boilerplate(line)


# --------------------------------------------------------------------------
# Dedup must union evidence, not pick a winner.
# --------------------------------------------------------------------------

def test_dedupe_unions_platform_observations():
    """Appearing on two directory pages is the normal case. Fill-if-empty
    discarded the duplicate's pair whenever the survivor had one, halving
    platform coverage before enrichment ran and starving the blend, which
    needs two."""
    a = {"name": "Dr. Pritish Pawar", "location": "Chandler, AZ", "specialty": "Neurology",
         "review_observations": [{"platform": "healthgrades.com", "rating": 4.0,
                                  "review_count": 31,
                                  "source_url": "https://www.healthgrades.com/physician/dr-p"}],
         "insurance_accepted": ["Aetna"]}
    b = {"name": "Pritish Pawar, MD", "location": "Chandler, AZ", "specialty": "Neurology",
         "review_observations": [{"platform": "vitals.com", "rating": 3.5,
                                  "review_count": 16,
                                  "source_url": "https://www.vitals.com/doctors/dr-p"}],
         "insurance_accepted": ["Cigna", "Aetna"]}

    out = _gatherer()._dedupe_providers([a, b])
    assert len(out) == 1
    platforms = {o["platform"] for o in out[0]["review_observations"]}
    assert platforms == {"healthgrades.com", "vitals.com"}
    assert _blended_platform_rating(out[0]["review_observations"])["platforms"] == 2


def test_dedupe_unions_insurance_without_duplicating():
    a = {"name": "Dr. P", "location": "Chandler, AZ", "insurance_accepted": ["Aetna"]}
    b = {"name": "Dr. P, MD", "location": "Chandler, AZ",
         "insurance_accepted": ["Cigna", "Aetna"]}
    out = _gatherer()._dedupe_providers([a, b])
    assert sorted(out[0]["insurance_accepted"]) == ["Aetna", "Cigna"]


# --------------------------------------------------------------------------
# The identity guard must not be bypassed by the fallback.
# --------------------------------------------------------------------------

def test_a_rejected_stranger_does_not_reach_the_provider_via_fallback():
    """When every observation is rejected as a different physician, the
    top-level rating/count/URL are the model's reading of those SAME pages —
    so falling back to them re-admitted the stranger's numbers and profile
    link, and the provider was then reported as cleanly `enriched`."""
    provider = {"name": "Kavita Sharma", "location": "Chandler, AZ"}
    review_data = {
        "review_observations": [{
            "platform": "healthgrades.com", "rating": 4.9, "review_count": 212,
            "page_provider_name": "Anil Kumar",
            "source_url": "https://www.healthgrades.com/physician/dr-anil-kumar-xyz",
        }],
        "rating": 4.9, "review_count": 212,
        "review_source_url": "https://www.healthgrades.com/physician/dr-anil-kumar-xyz",
        "review_summary": "Excellent bedside manner.",
    }
    _gatherer()._merge_review_data(provider, review_data)

    assert provider.get("rating") in (None, 0, 0.0)
    assert provider.get("review_count") in (None, 0)
    assert "anil-kumar" not in str(provider.get("review_source_url") or "")


def test_the_fallback_still_works_when_identity_was_never_challenged():
    """The gate must not disable the fallback for the ordinary no-observations
    case it exists to serve."""
    provider = {"name": "Dr. Solo", "location": "Chandler, AZ"}
    _gatherer()._merge_review_data(provider, {
        "review_observations": [],
        "rating": 4.5, "review_count": 20,
        "review_source_url": "https://www.vitals.com/doctors/dr-solo",
        "review_summary": "Consistently praised.",
    })
    assert provider["rating"] == 4.5


# --------------------------------------------------------------------------
# The judge must not bind its answer to the wrong provider.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("claimed,actual,same", [
    ("Dr. Andrea An, M.D.", "Andrea An MD", True),
    ("Dr. Hussam Seif Eddeine", "Hussam Seif-Eddeine, MD", True),
    ("Dr. Hemant Pandey", "Dr. Hemant Kumar Pandey, MD", True),
    ("Andrea An", "Hussam Seif-Eddeine", False),
    ("", "Andrea An", True),
])
def test_judge_name_cross_check(claimed, actual, same):
    """The judge's input is SHUFFLED, so a mis-echoed index silently attaches
    one doctor's rubric, citations and patient-facing sentences to another —
    and every value still passes the range check."""
    assert _same_provider_name(claimed, actual) is same


def _judge_response(payload):
    """A stubbed OpenAI chat completion carrying `payload` as its JSON body."""
    import json as _json
    message = MagicMock()
    message.content = _json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    return response


def _score_with_judge(providers, judge_payload):
    from agents.preference_scorer import PreferenceScorerAgent
    with patch("agents.preference_scorer.OpenAI") as client:
        scorer = PreferenceScorerAgent()
        scorer.openai_client.chat.completions.create.return_value = _judge_response(
            judge_payload
        )
        return scorer._generate_ai_rankings(providers, {})


def test_a_misbound_judge_entry_does_not_score_the_wrong_provider():
    """The parser must REFUSE an entry whose echoed name is not the name at
    the index it claims. Without this, a shuffled-order mis-echo silently
    moved one doctor's rubric and patient-facing reasoning onto another, and
    nothing downstream could detect it."""
    providers = [
        {"name": "Dr. Andrea An, MD", "base_score": 80.0},
        {"name": "Dr. Hussam Seif-Eddeine, MD", "base_score": 78.0},
    ]
    # Index 0 is Andrea An, but the entry names the OTHER provider.
    scored = _score_with_judge(providers, [{
        "provider_index": 0,
        "provider_name": "Dr. Hussam Seif-Eddeine, MD",
        "scores": {"review_substance": 45, "red_flags": 28, "practical_access": 18},
        "evidence": {}, "ai_score": 91,
        "reasoning": "Patients consistently praise the wait times.",
        "strengths": [], "concerns": [],
    }])
    assert scored[0].get("ai_score") != 91.0, (
        "Andrea An took the rubric written about a different provider"
    )
    assert "wait times" not in str(scored[0].get("ai_reasoning") or "")


def test_a_correctly_bound_judge_entry_is_applied():
    """The guard must not reject a correct answer over spelling."""
    providers = [{"name": "Dr. Andrea An, MD", "base_score": 80.0}]
    scored = _score_with_judge(providers, [{
        "provider_index": 0,
        "provider_name": "Andrea An M.D.",          # same person, different spelling
        "scores": {"review_substance": 45, "red_flags": 28, "practical_access": 18},
        "evidence": {}, "ai_score": 91,
        "reasoning": "Thorough and well reviewed.",
        "strengths": [], "concerns": [],
    }])
    assert scored[0]["ai_score"] == 91.0


def test_a_duplicated_index_is_ignored():
    """A response that is not a permutation means the model numbered by
    position; the second claim on an index must not overwrite the first."""
    providers = [
        {"name": "Dr. Andrea An, MD", "base_score": 80.0},
        {"name": "Dr. Hussam Seif-Eddeine, MD", "base_score": 78.0},
    ]
    scored = _score_with_judge(providers, [
        {"provider_index": 0, "provider_name": "Dr. Andrea An, MD",
         "scores": {"review_substance": 40, "red_flags": 25, "practical_access": 15},
         "evidence": {}, "reasoning": "", "strengths": [], "concerns": []},
        {"provider_index": 0, "provider_name": "Dr. Andrea An, MD",
         "scores": {"review_substance": 10, "red_flags": 5, "practical_access": 2},
         "evidence": {}, "reasoning": "", "strengths": [], "concerns": []},
    ])
    assert scored[0]["ai_score"] == 80.0     # the first claim, not the second


def test_the_judge_payload_labels_providers_with_the_key_it_asks_for():
    """The input said `index` while the output format said `provider_index`.
    A model returning its position in the shuffled array — the natural
    reading — mis-bound every provider."""
    import inspect
    from agents import preference_scorer

    source = inspect.getsource(preference_scorer)
    assert '"provider_index": i,' in source
    assert '"index": i,' not in source
