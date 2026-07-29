"""Regression guards for the six defects found in the PR #34 live run.

The judge ones matter most: the entire 30% of the score vanished from every
card, and the run's stderr is gone, so both candidate causes are pinned here
rather than the one that happened to fire.
"""

import inspect
import json
from unittest.mock import MagicMock, patch

import pytest

import app as app_module


def _code_only(source: str) -> str:
    """Source with `#` comment lines removed.

    These assertions are about what the code DOES; the comments explaining the
    defect naturally quote the very names being asserted absent.
    """
    return "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
from agents.preference_scorer import (
    _judge_token_budget,
    _matches_other_provider,
    _parse_ranking_response,
    _salvage_json_objects,
)
from utils.provenance import label_source, url_page_kind


# --------------------------------------------------------------------------
# #1 — the judge produced nothing for any provider
# --------------------------------------------------------------------------

POOL = [
    {"name": "Dr. Andrea An, MD", "base_score": 80.0},
    {"name": "Dr. Hussam Seif-Eddeine, MD", "base_score": 78.0},
]

PLACEHOLDER = "copy the name from that provider's record verbatim"


def _entry(index, name, substance=45):
    return {
        "provider_index": index,
        "provider_name": name,
        "scores": {"review_substance": substance, "red_flags": 28, "practical_access": 18},
        "evidence": {}, "reasoning": "Fine.", "strengths": [], "concerns": [],
    }


def _judge(providers, body, finish_reason="stop"):
    """Drive _generate_ai_rankings against a stubbed judge response."""
    from agents.preference_scorer import PreferenceScorerAgent

    message = MagicMock(); message.content = body
    choice = MagicMock(); choice.message = message; choice.finish_reason = finish_reason
    response = MagicMock(); response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

    with patch("agents.preference_scorer.OpenAI"):
        scorer = PreferenceScorerAgent()
        scorer.openai_client.chat.completions.create.return_value = response
        return scorer._generate_ai_rankings([dict(p) for p in providers], {})


def test_a_placeholder_echo_no_longer_costs_every_provider_its_rubric():
    """The output_format's example value was the self-describing string
    "copy the name from that provider's record verbatim". A model echoing it
    matched nobody, so the guard dropped every entry and the whole judge
    section disappeared from every card."""
    scored = _judge(POOL, json.dumps([_entry(0, PLACEHOLDER), _entry(1, PLACEHOLDER)]))
    assert scored[0]["ai_score"] == 91.0
    assert scored[1]["ai_score"] == 91.0


def test_the_prompt_no_longer_ships_a_self_describing_example():
    source = inspect.getsource(
        __import__("agents.preference_scorer", fromlist=["x"])
    )
    assert PLACEHOLDER not in source, "the placeholder example invites a literal echo"


def test_a_name_naming_a_different_pool_member_is_still_dropped():
    """The guard's actual purpose. Index 0 is An; the entry claims Seif-Eddeine."""
    scored = _judge(POOL, json.dumps([_entry(0, "Dr. Hussam Seif-Eddeine, MD")]))
    assert scored[0]["ai_score"] == 50.0          # neutral fallback, not 91
    assert "ai_rubric" not in scored[0]


def test_a_name_matching_nobody_keeps_the_entry():
    """A name that belongs to no one in the pool is a formatting slip, not a
    mis-binding — dropping it loses a paid answer for no safety gain."""
    scored = _judge(POOL, json.dumps([_entry(0, "Somebody Entirely Unrelated")]))
    assert scored[0]["ai_score"] == 91.0


def test_matches_other_provider_requires_a_positive_match():
    assert _matches_other_provider("Dr. Hussam Seif-Eddeine, MD", POOL, 0) is True
    assert _matches_other_provider("Somebody Unrelated", POOL, 0) is False
    assert _matches_other_provider(PLACEHOLDER, POOL, 0) is False
    assert _matches_other_provider("", POOL, 0) is False


def test_a_truncated_array_still_yields_the_entries_that_arrived():
    """The scorer was the only agent with no JSON repair path, so one clipped
    entry discarded the whole paid call."""
    body = json.dumps([_entry(0, "Dr. Andrea An, MD"),
                       _entry(1, "Dr. Hussam Seif-Eddeine, MD")])
    scored = _judge(POOL, body[:-40], finish_reason="length")
    assert scored[0]["ai_score"] == 91.0
    assert scored[1]["ai_score"] == 50.0


def test_salvage_extracts_complete_objects_only():
    objects = _salvage_json_objects('[{"a": 1}, {"b": {"c": 2}}, {"d": ')
    assert objects == [{"a": 1}, {"b": {"c": 2}}]


def test_salvage_is_not_confused_by_braces_inside_strings():
    assert _salvage_json_objects('[{"q": "a } b {"}]') == [{"q": "a } b {"}]


def test_a_stringified_index_is_coerced_not_discarded():
    entry = _entry(0, "Dr. Andrea An, MD")
    entry["provider_index"] = "0"
    assert _judge(POOL, json.dumps([entry]))[0]["ai_score"] == 91.0


@pytest.mark.parametrize("body,expected", [
    ('{"rankings": [{"provider_index": 0}]}', 1),   # wrapped in an object
    ('```JSON\n[{"provider_index": 0}]\n```', 1),   # uppercase fence
    ('Here you go: [{"provider_index": 0}]', 1),    # prose preamble
    ('[]', 0),
])
def test_parse_tolerates_real_model_output_shapes(body, expected):
    assert len(_parse_ranking_response(body)) == expected


def test_the_token_budget_scales_with_the_pool():
    """A flat 4000 was fine for 10 providers and truncated at 16 — and on a
    reasoning model the reasoning trace comes out of the same allowance."""
    assert _judge_token_budget(10) < _judge_token_budget(16) < _judge_token_budget(20)
    assert _judge_token_budget(3) >= 4000          # never below the old floor
    assert _judge_token_budget(500) <= 16000       # bounded


# --------------------------------------------------------------------------
# #2 — insurance lists with no source
# --------------------------------------------------------------------------

def _gatherer():
    from agents.data_gatherer import DataGathererAgent
    with patch("agents.data_gatherer.TavilyClient"), patch("agents.data_gatherer.Anthropic"):
        return DataGathererAgent()


def test_enrichment_backfills_the_source_even_when_the_list_already_exists():
    """The URL write was nested inside the "list is empty" guard, so a
    known-good source was discarded and the card named eight payers with no
    attribution."""
    provider = {"name": "Dr. Kept", "insurance_accepted": ["Aetna PPO"]}
    _gatherer()._merge_review_data(provider, {
        "review_summary": "Fine.", "review_sentiment": "positive",
        "insurance_accepted": ["Cigna"],
        "insurance_source_url": "https://www.healthgrades.com/physician/dr-kept",
    })
    assert provider["insurance_accepted"] == ["Aetna PPO"]     # list not clobbered
    assert provider["insurance_source_url"].endswith("dr-kept")


def test_an_existing_source_is_never_downgraded():
    provider = {"name": "Dr. Sourced", "insurance_accepted": ["Aetna"],
                "insurance_source_url": "https://www.vitals.com/doctors/sourced"}
    _gatherer()._merge_review_data(provider, {
        "review_summary": "Fine.", "review_sentiment": "positive",
        "insurance_accepted": ["Cigna"], "insurance_source_url": "https://other.example",
    })
    assert "vitals.com" in provider["insurance_source_url"]


def test_the_discovery_prompt_asks_for_an_insurance_source():
    """It asked for review_source_url but not this one, so every
    discovery-sourced plan list was structurally unattributable."""
    from agents.data_gatherer import DataGathererAgent
    # `_extract_page_shard`, not `_extract_provider_data`: the prompt and the
    # cleaning loop moved there when discovery extraction was split across
    # concurrent calls, and reading the outer method quietly asserted against
    # a page-selection function that contains neither half.
    source = inspect.getsource(DataGathererAgent._extract_page_shard)
    # Both halves, asserted separately — the prompt must ASK for the field and
    # the cleaning loop must CARRY it. A bare substring check passed with
    # either one missing, because both mention the same key.
    assert "- insurance_source_url:" in source, "discovery prompt does not request it"
    assert 'cleaned_provider["insurance_source_url"]' in source, "not carried through"


# --------------------------------------------------------------------------
# #3 — profile-vs-listing classification (this one moves scores)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://www.vitals.com/doctors/marianne-de-lima-md",
    "https://www.vitals.com/doctors/hodgson",
    "https://www.ratemds.com/doctor-ratings/dr-jane-doe-chandler-az/",
    "https://www.healthgrades.com/physician/dr-andrea-an-3xyz9",
    "https://doctor.webmd.com/doctor/marianne-de-lima-md-abc",
])
def test_real_profile_shapes_are_recognised(url):
    """vitals required the slug to BEGIN with "dr", and ratemds' real
    `/doctor-ratings/` matched neither marker — so no ratemds URL could ever
    be a profile, while its probation exit criterion is "a clean profile-based
    pair"."""
    assert url_page_kind(url) == "profile"
    assert "listing page" not in label_source(url)


@pytest.mark.parametrize("url", [
    "https://www.healthgrades.com/usearch?what=Neurology&where=Chandler",
    "https://www.healthgrades.com/neurology-directory/az-arizona/chandler",
    "https://www.vitals.com/search?q=neurologist",
    "https://www.healthgrades.com/",
])
def test_real_listing_shapes_are_still_labelled(url):
    assert url_page_kind(url) == "listing"
    assert "listing page" in label_source(url)


def test_an_unrecognised_platform_shape_is_unknown_not_a_listing():
    """A false warning is worse than none: it teaches the eye to skip the
    warning that is true."""
    url = "https://www.healthgrades.com/group-practice/chandler-neurology-xyz"
    assert url_page_kind(url) == "unknown"
    assert "listing page" not in label_source(url)


def test_unknown_outranks_a_confirmed_listing_in_tie_breaks():
    """Under the old boolean both were False, so the tie fell to first-seen —
    and discovery (city listings) always runs before enrichment (profiles)."""
    from agents.data_gatherer import _page_rank
    profile = _page_rank("https://www.vitals.com/doctors/de-lima")
    unknown = _page_rank("https://www.healthgrades.com/group-practice/xyz")
    listing = _page_rank("https://www.healthgrades.com/usearch?what=x")
    assert profile > unknown > listing


def test_a_profile_beats_a_listing_for_the_same_platform_pair():
    """This is the scoring path: at equal counts the source page decides which
    rating enters the blend."""
    from agents.data_gatherer import _platform_rating_pairs
    pairs = _platform_rating_pairs([
        {"platform": "healthgrades.com", "rating": 4.6, "review_count": 23,
         "source_url": "https://www.healthgrades.com/usearch?what=Neurology"},
        {"platform": "healthgrades.com", "rating": 3.1, "review_count": 23,
         "source_url": "https://www.healthgrades.com/physician/dr-x-123"},
    ])
    assert len(pairs) == 1
    assert pairs[0]["rating"] == 3.1          # the profile, not the directory


# --------------------------------------------------------------------------
# #4 / #5 — re-running a search, and the progress panel
# --------------------------------------------------------------------------

def test_an_identical_resubmit_is_no_longer_suppressed():
    """`render_search_form` returns None unless submitted, so a truthy result
    already means "the user clicked". Gating that on a params-changed check
    made a repeat run impossible — which is exactly the cold-vs-warm cache
    comparison the sidebar advertises."""
    source = _code_only(inspect.getsource(app_module.main))
    assert "search_params_changed" not in source
    assert not hasattr(app_module, "search_params_changed"), "dead helper still present"
    # The reset must be UNCONDITIONAL on a submitted form. Asserting only the
    # old helper's absence was not a guard: any other condition wrapped around
    # the reset re-creates the same silent no-op.
    normalized = "\n".join(line.rstrip() for line in source.splitlines())
    assert (
        "    if search_params:\n"
        "        st.session_state.search_executed = False"
    ) in normalized, "a submitted form must always re-arm the search"


def test_completion_does_not_rerun_over_the_status_collapse():
    """`status.update()` only enqueues; RerunException clears the unflushed
    queue, so the panel stayed expanded reading "agents are working...". The
    tell was that the exception path, which has no rerun, collapsed fine."""
    source = _code_only(inspect.getsource(app_module.main))
    marker = 'label=f"Search complete'
    assert marker in source, "completion branch not found — test needs re-anchoring"
    after_completion = source.split(marker, 1)[1].split("except Exception", 1)[0]
    assert "st.rerun()" not in after_completion


# --------------------------------------------------------------------------
# #6 — "Other providers considered" rows
# --------------------------------------------------------------------------

def _render_rows(rows, monkeypatch):
    captured = []
    monkeypatch.setattr(app_module.st, "markdown", lambda t, **k: captured.append(str(t)))
    monkeypatch.setattr(app_module.st, "caption", lambda t, **k: captured.append(str(t)))
    monkeypatch.setattr(app_module.st, "expander", lambda *a, **k: __import__(
        "contextlib").nullcontext())
    app_module.render_other_providers(rows)
    return "\n".join(captured)


def test_a_city_centroid_row_says_so(monkeypatch):
    """The precision honesty round 7 shipped for the top cards never reached
    this projection, so a shared centroid read as a measurement."""
    out = _render_rows([{
        "rank": 6, "name": "Dr. A", "computed_distance_miles": 2.7,
        "distance_precision": "city", "refined_score": 63,
    }], monkeypatch)
    assert "(city-level)" in out


def test_a_zip_measured_row_carries_no_qualifier(monkeypatch):
    out = _render_rows([{
        "rank": 6, "name": "Dr. B", "computed_distance_miles": 2.7,
        "distance_precision": "zip", "refined_score": 63,
    }], monkeypatch)
    assert "~2.7 mi" in out and "city-level" not in out


def test_an_unrated_row_says_why(monkeypatch):
    """"We looked and found nothing" and "we never looked" are different
    facts that produced identical rows."""
    out = _render_rows([{
        "rank": 9, "name": "Dr. C", "enrichment_outcome": "no_profile_found",
        "refined_score": 44,
    }], monkeypatch)
    assert "no reviews found" in out

    out = _render_rows([{
        "rank": 9, "name": "Dr. D", "enrichment_outcome": "over_budget",
        "refined_score": 44,
    }], monkeypatch)
    assert "pool limit" in out


def test_the_orchestrator_projects_the_fields_the_rows_need():
    from agents import orchestrator
    source = inspect.getsource(orchestrator.ProviderMatchingOrchestrator._finalize_results)
    # Match the exact key form. A bare `"distance_precision" in source` also
    # matched a renamed `distance_precision_REMOVED`, so the test survived the
    # field being dropped.
    for field in ("distance_precision", "enrichment_outcome", "critic_status"):
        assert f'"{field}":' in source, (
            f"{field} missing from the other-providers projection"
        )


# ---- Round 12: the expander's two groups ----


def _grouped_rows(others):
    import app as app_module
    out = []
    with patch.object(app_module.st, "markdown", side_effect=lambda t, **k: out.append(str(t))), \
         patch.object(app_module.st, "caption", side_effect=lambda t, **k: out.append(str(t))), \
         patch.object(app_module.st, "expander"):
        app_module.render_other_providers(others)
    return "\n".join(out)


_RESEARCHED = {"rank": 6, "name": "Dr. Researched", "researched": True,
               "withheld_reason": None, "withheld_label": "",
               "blended_rating": 3.9, "blended_review_count": 65,
               "blended_platform_count": 3, "computed_distance_miles": 1.9,
               "refined_score": 60.0, "enrichment_outcome": "enriched"}
_NO_DATA = {"rank": 7, "name": "Dr. NoData", "researched": True,
            "withheld_reason": "no_profile_found",
            "withheld_label": "researched, but no reviews were found",
            "computed_distance_miles": 3.1, "refined_score": 61.0,
            "enrichment_outcome": "no_profile_found"}
_OUR_FAULT = {"rank": 8, "name": "Dr. Unscored", "researched": True,
              "withheld_reason": "not_critiqued",
              "withheld_label": "our independent review did not complete for this provider",
              "computed_distance_miles": 2.2, "refined_score": 62.0,
              "enrichment_outcome": "enriched"}
_UNRESEARCHED = {"rank": 9, "name": "Dr. Unresearched", "researched": False,
                 "withheld_reason": "over_budget",
                 "withheld_label": "not researched — outside this search's research budget",
                 "computed_distance_miles": 5.8, "distance_precision": "city",
                 "refined_score": 64.0, "enrichment_outcome": "over_budget"}


def test_the_expander_splits_by_recommendability():
    """Three states, three headings. A provider we searched but couldn't fully
    assess is NOT the same as one we never looked at, and neither is the same as
    one that simply ranked sixth."""
    markup = _grouped_rows([_RESEARCHED, _OUR_FAULT, _NO_DATA, _UNRESEARCHED])

    assert "Ranked below the top 5" in markup
    assert "Researched, but not recommendable" in markup
    assert "Found but not researched" in markup
    # The never-researched caption states its reason and its score caveat
    assert "research budget" in markup
    assert "provisional" in markup
    assert "not comparable" in markup


def test_a_pipeline_failure_says_so_on_the_row():
    """"Unrated — no reviews found" already covers the coverage cases, but a
    provider whose data we HELD and whose scoring WE failed to finish would
    otherwise render with no explanation at all."""
    markup = _grouped_rows([_OUR_FAULT])
    assert "independent review did not complete" in markup


def test_a_coverage_row_does_not_state_its_reason_twice():
    """The rating branch already prints "Unrated — no reviews found"; adding the
    withheld label too would say the same thing twice on one line."""
    markup = _grouped_rows([_NO_DATA])
    assert markup.count("no reviews") == 1


def test_the_group_caption_names_the_configured_budget():
    """A hardcoded 10 would silently lie the moment MAX_PROVIDERS_TO_ENRICH moved."""
    from utils.config import get_config
    markup = _grouped_rows([_RESEARCHED, _UNRESEARCHED])
    assert f"top {get_config().MAX_PROVIDERS_TO_ENRICH}" in markup


def test_a_single_group_gets_no_heading():
    """Headings only earn their space when there is a distinction to draw."""
    markup = _grouped_rows([_RESEARCHED])
    assert "Ranked below the top 5" not in markup
    assert "Found but not researched" not in markup
    assert "Dr. Researched" in markup


def test_rows_missing_the_reason_are_treated_as_recommendable():
    """Backward compatibility: an older payload without `withheld_reason` must
    not silently land every provider in a 'we couldn't assess them' group."""
    legacy = {k: v for k, v in _RESEARCHED.items() if k != "withheld_reason"}
    markup = _grouped_rows([legacy])
    assert "Found but not researched" not in markup
    assert "not recommendable" not in markup
    assert "Dr. Researched" in markup


# ---- Round 13: the developer surface names who was withheld ----


def _withheld_detail(withheld, others):
    import app as app_module
    out = []
    with patch.object(app_module.st, "markdown", side_effect=lambda t, **k: out.append(str(t))), \
         patch.object(app_module.st, "caption", side_effect=lambda t, **k: out.append(str(t))):
        app_module._render_withheld_detail(withheld, others)
    return "\n".join(out)


def test_the_developer_surface_names_providers_and_stages():
    """Inverse of the patient panel: names ARE shown and stage vocabulary IS
    allowed, because this surface exists to make our own failures actionable."""
    markup = _withheld_detail(
        {"total": 2, "no_data": 1, "pipeline_failures": 1},
        [_OUR_FAULT, _NO_DATA],
    )
    assert "Dr. Unscored" in markup and "Dr. NoData" in markup
    assert "independent review did not complete" in markup
    assert "no reviews were found" in markup


def test_our_own_failures_are_listed_first_and_marked():
    """A stage we paid for and didn't get is the actionable entry; a gap in what
    the web holds is not. Ordering by actionability, not by rank."""
    markup = _withheld_detail(
        {"total": 3, "no_data": 2, "pipeline_failures": 1},
        [_NO_DATA, _UNRESEARCHED, _OUR_FAULT],
    )
    assert markup.index("Dr. Unscored") < markup.index("Dr. NoData")
    assert "⚠️" in markup
    assert "the missing step is ours" in markup


def test_no_pipeline_failures_says_so_rather_than_implying_them():
    """With only coverage gaps, the caption must not leave a developer hunting
    for a bug we didn't have."""
    markup = _withheld_detail({"total": 1, "no_data": 1, "pipeline_failures": 0}, [_NO_DATA])
    assert "not pipeline" in markup
    assert "⚠️" not in markup


def test_nothing_renders_when_nothing_was_withheld():
    assert _withheld_detail({}, []) == ""
    assert _withheld_detail({"total": 0}, []) == ""


def test_the_detail_always_says_the_providers_are_still_listed():
    """Withheld is not deleted. Both branches of the caption must say so."""
    for withheld, rows in (
        ({"total": 1, "no_data": 1, "pipeline_failures": 0}, [_NO_DATA]),
        ({"total": 1, "no_data": 0, "pipeline_failures": 1}, [_OUR_FAULT]),
    ):
        assert "Other providers considered" in _withheld_detail(withheld, rows)
