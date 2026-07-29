"""Unit tests for the DataGathererAgent."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from unittest.mock import MagicMock, patch
from agents.data_gatherer import (
    DataGathererAgent,
    _DISCOVERY_EXCERPT_BUDGET,
    _DISCOVERY_EXCERPT_WINDOWS,
    _DISCOVERY_MAX_BLOCKS,
    _ENRICHMENT_EXCERPT_BUDGET,
    _ENRICHMENT_HEAD_CHARS,
    _is_review_platform_url,
)
from utils.excerpt import SUMMARY_MAX_CHARS, build_excerpt
from tests.fixtures.mock_agent_responses import (
    MOCK_TAVILY_SEARCH_RESPONSE,
    MOCK_CLAUDE_EXTRACTION_RESPONSE,
    MOCK_GATHER_PROVIDERS_RESULT
)

@pytest.fixture
def data_gatherer():
    """Fixture to create a DataGathererAgent with mocked clients."""
    with patch.object(DataGathererAgent, '_initialize_clients', return_value=None):
        agent = DataGathererAgent()
        agent.tavily_client = MagicMock()
        agent.anthropic_client = MagicMock()
        return agent

def test_build_search_query(data_gatherer: DataGathererAgent):
    """Discovery query covers specialty/location/reviews — but never the payer,
    which would act as an implicit filter at the recall stage."""
    query = data_gatherer._build_search_query(
        specialty="Cardiology",
        location="New York, NY",
        insurance="Aetna"
    )
    assert "Cardiology" in query
    assert "New York, NY" in query
    assert "Aetna" not in query
    assert "reviews" in query

@patch('agents.data_gatherer.DataGathererAgent._search_providers')
def test_gather_providers_success(mock_search, data_gatherer: DataGathererAgent):
    """Test the main gather_providers method for a successful run."""

    mock_search.return_value = MOCK_TAVILY_SEARCH_RESPONSE['results']

    # Mock the extraction method
    with patch.object(data_gatherer, '_extract_provider_data', return_value=MOCK_GATHER_PROVIDERS_RESULT['providers']) as mock_extract:

        result = data_gatherer.gather_providers(
            specialty="Neurology",
            location="Phoenix, AZ"
        )

        assert result['status'] == 'success'
        assert len(result['providers']) == 2
        assert result['providers'][0]['name'] == "Dr. Emily Carter"
        # Multi-query discovery: several phrasings, all reading page bodies
        assert mock_search.called
        assert mock_extract.called
        assert mock_search.call_args.kwargs.get('include_raw_content') is True

def test_extract_provider_data_success(data_gatherer: DataGathererAgent):
    """Test the _extract_provider_data method for successful extraction."""
    
    # Mock the Anthropic client's response
    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_CLAUDE_EXTRACTION_RESPONSE
    data_gatherer.anthropic_client.messages.create.return_value = mock_response
    
    providers = data_gatherer._extract_provider_data(
        search_results=MOCK_TAVILY_SEARCH_RESPONSE['results'],
        specialty="Neurology",
        location="Phoenix, AZ"
    )
    
    assert len(providers) == 2
    assert providers[0]['name'] == "Dr. Emily Carter"
    assert providers[1]['name'] == "Dr. Ben Adams"
    assert providers[0]['rating'] == 4.8
    assert "BCBS" in providers[0]['insurance_accepted']

def test_search_providers_api_failure(data_gatherer: DataGathererAgent):
    """A persistent Tavily failure yields [] after the retry, not a crash."""

    data_gatherer.tavily_client.search.side_effect = Exception("API Error")

    with patch("agents.data_gatherer.time.sleep"):
        results = data_gatherer._search_providers(query="test query")

    assert results == []
    assert data_gatherer.tavily_client.search.call_count == 2

def test_search_providers_retries_once_on_transient_failure(data_gatherer: DataGathererAgent):
    """One network blip must not kill the workflow — the retry recovers."""

    data_gatherer.tavily_client.search.side_effect = [
        Exception("timeout"),
        MOCK_TAVILY_SEARCH_RESPONSE,
    ]

    with patch("agents.data_gatherer.time.sleep") as mock_sleep:
        results = data_gatherer._search_providers(query="test query")

    assert results == MOCK_TAVILY_SEARCH_RESPONSE["results"]
    assert data_gatherer.tavily_client.search.call_count == 2
    mock_sleep.assert_called_once()

def test_extract_provider_data_json_error(data_gatherer: DataGathererAgent):
    """Test _extract_provider_data with a malformed JSON response from Claude."""

    mock_response = MagicMock()
    mock_response.content[0].text = "This is not valid JSON"
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    providers = data_gatherer._extract_provider_data(
        search_results=MOCK_TAVILY_SEARCH_RESPONSE['results'],
        specialty="Neurology",
        location="Phoenix, AZ"
    )

    assert providers == []

def test_review_count_is_never_fabricated(data_gatherer: DataGathererAgent):
    """A rating without a stated review count stays None — no invented 25/50."""

    mock_response = MagicMock()
    mock_response.content[0].text = (
        '[{"name": "Dr. No Count", "specialty": "Neurology", "location": "Phoenix, AZ",'
        ' "rating": 4.5, "source_url": "https://www.google.com/maps/place/xyz"}]'
    )
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    providers = data_gatherer._extract_provider_data(
        search_results=MOCK_TAVILY_SEARCH_RESPONSE['results'],
        specialty="Neurology",
        location="Phoenix, AZ"
    )

    assert providers[0]['rating'] == 4.5
    assert providers[0]['review_count'] is None

def test_gather_records_insurance_in_metadata_only(data_gatherer: DataGathererAgent):
    """The payer never filters, labels, or scores — it rides in the search
    metadata for the audit trail; scraped insurance lists stay untouched
    neutral info (coverage questions belong to the FHIR network check)."""

    extracted = [
        {"name": "Dr. Match", "insurance_accepted": ["Aetna PPO"], "rating": 4.5,
         "review_summary": "Great", "review_sentiment": "positive"},
        {"name": "Dr. Other", "insurance_accepted": ["Cigna"], "rating": 4.2,
         "review_summary": "Fine", "review_sentiment": "positive"},
    ]
    with patch.object(data_gatherer, '_search_providers', return_value=MOCK_TAVILY_SEARCH_RESPONSE['results']), \
         patch.object(data_gatherer, '_extract_provider_data', return_value=extracted), \
         patch.object(data_gatherer, '_enrich_missing_reviews', side_effect=lambda p, loc, spec="": p):
        result = data_gatherer.gather_providers(
            specialty="Neurology", location="Phoenix, AZ", insurance="Aetna"
        )

    providers = result['providers']
    assert len(providers) == 2                      # nobody was deleted
    assert all("insurance_match" not in p for p in providers)
    assert result["search_metadata"]["insurance"] == "Aetna"

def test_gather_providers_requests_raw_content_for_candidates(data_gatherer: DataGathererAgent):
    """The candidate search reads page bodies — directory-style results name providers only there."""

    with patch.object(data_gatherer, '_search_providers', return_value=MOCK_TAVILY_SEARCH_RESPONSE['results']) as mock_search, \
         patch.object(data_gatherer, '_extract_provider_data', return_value=MOCK_GATHER_PROVIDERS_RESULT['providers']), \
         patch.object(data_gatherer, '_enrich_missing_reviews', side_effect=lambda p, loc, kw="", spec="": p):
        result = data_gatherer.gather_providers(specialty="Neurology", location="Phoenix, AZ")

    assert result['status'] == 'success'
    assert mock_search.call_args.kwargs.get('include_raw_content') is True

def test_gather_providers_no_results_when_extraction_empty(data_gatherer: DataGathererAgent):
    """Search results without extractable providers yield benign no_results, not a crash."""

    with patch.object(data_gatherer, '_search_providers', return_value=MOCK_TAVILY_SEARCH_RESPONSE['results']), \
         patch.object(data_gatherer, '_extract_provider_data', return_value=[]), \
         patch.object(data_gatherer, '_enrich_missing_reviews') as mock_enrich:
        result = data_gatherer.gather_providers(specialty="Neurology", location="Phoenix, AZ")

    assert result['status'] == 'no_results'
    assert result['providers'] == []
    mock_enrich.assert_not_called()

def test_extract_provider_data_includes_raw_content_excerpt(data_gatherer: DataGathererAgent):
    """Candidate extraction feeds capped page text to Claude, like the review pass."""

    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_CLAUDE_EXTRACTION_RESPONSE
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    results = [{
        "title": "Top 10 Neurologists in Phoenix", "url": "https://example.com/best",
        "content": "snippet", "raw_content": "P" * (_DISCOVERY_EXCERPT_BUDGET + 1000),
    }]
    providers = data_gatherer._extract_provider_data(results, "Neurology", "Phoenix, AZ")

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs['messages'][0]['content']
    assert "Full page text (excerpt)" in prompt
    assert "P" * _DISCOVERY_EXCERPT_BUDGET in prompt
    assert "P" * (_DISCOVERY_EXCERPT_BUDGET + 1) not in prompt
    assert len(providers) == 2


def _listicle_page(count: int = 15) -> str:
    """A directory page shaped like the ones discovery actually reads: `count`
    providers, each ~940 chars, spread over ~14K chars."""
    return "\n\n".join(
        f"{i}. Dr. Provider{i} MD is a neurologist in Chandler AZ. "
        + ("Patients rate this practice highly for care and communication. " * 14)
        for i in range(1, count + 1)
    )


def _names_in(excerpt: str, count: int = 15) -> set:
    return {f"Provider{i}" for i in range(1, count + 1) if f"Provider{i}" in excerpt}


def test_discovery_reads_wider_than_enrichment():
    """The two passes read different documents and must not share a budget.

    Enrichment reads ONE provider's profile, where 2000/3 is ample. Discovery
    reads directories spreading 10-20 providers over 10-20K chars."""
    assert _DISCOVERY_EXCERPT_BUDGET > 2000          # enrichment's budget
    assert _DISCOVERY_EXCERPT_WINDOWS > 3            # enrichment's window count


def test_a_long_directory_page_yields_most_of_its_entries():
    """The live failure: a Chandler search read ~2 names off each listicle, the
    home pool came in under MIN_CANDIDATE_POOL, and the ring expanded to cities
    the user never asked for — paying for a second search AND a second
    extraction to recover names already on pages we had."""
    page = _listicle_page()
    anchors = ["neurologist", "dr.", "patients"]

    wide = _names_in(build_excerpt(
        page, anchors=anchors,
        budget=_DISCOVERY_EXCERPT_BUDGET, max_windows=_DISCOVERY_EXCERPT_WINDOWS,
    ))
    narrow = _names_in(build_excerpt(page, anchors=anchors, budget=2000))

    assert len(narrow) <= 3, "the old settings were not the bottleneck this test assumes"
    assert len(wide) >= 6, f"only {len(wide)} of 15 entries reached the extractor"
    assert len(wide) >= 3 * len(narrow), (
        f"widening barely moved recall ({len(narrow)} -> {len(wide)}) — if this fails, "
        f"the thin pool has another cause and a bigger budget will not fix it"
    )


def test_more_windows_at_a_fixed_budget_buys_nothing():
    """Guards the reasoning behind the constants, which is easy to get backwards.

    It is the BUDGET that recovers names, not the window count: splitting a
    fixed budget into more windows shrinks each below one entry's length and
    recovers no additional providers. Measured 6000/8 -> 7 names and
    6000/12 -> 7 names, while 8000/12 -> 9. Anyone tempted to 'tune' recall by
    raising only the window count should see this fail."""
    page, anchors = _listicle_page(), ["neurologist", "dr.", "patients"]

    at_8 = _names_in(build_excerpt(page, anchors=anchors, budget=6000, max_windows=8))
    at_12 = _names_in(build_excerpt(page, anchors=anchors, budget=6000, max_windows=12))
    bigger_budget = _names_in(build_excerpt(page, anchors=anchors, budget=8000, max_windows=12))

    assert len(at_12) <= len(at_8), "more windows at a fixed budget should not help"
    assert len(bigger_budget) > len(at_8), "a bigger budget is what recovers names"

def test_enrichment_reads_full_pages(data_gatherer: DataGathererAgent):
    """The per-provider pass requests raw page content (snippets alone
    structurally under-read reviews)."""

    provider = {"name": "Dr. Quiet", "review_summary": "No reviews available",
                "review_sentiment": "unknown"}
    with patch.object(data_gatherer, '_search_providers', return_value=[]) as mock_search:
        data_gatherer._enrich_one(provider, "Phoenix, AZ")

    args, kwargs = mock_search.call_args
    assert kwargs.get('include_raw_content') is True
    query = args[0] if args else kwargs.get('query')
    assert "Dr. Quiet reviews Phoenix, AZ" in query

def test_review_extraction_includes_truncated_raw_content(data_gatherer: DataGathererAgent):
    """raw_content is fed to the extraction, capped at the enrichment budget.

    Reads `_ENRICHMENT_EXCERPT_BUDGET` rather than a literal. The budget was a
    bare `2000` at the call site and this test hardcoded the same number, so
    the pair agreed with each other while nothing checked the number was right
    — and it stayed 2000 through the round-12 sweep that raised discovery's.
    """
    budget = _ENRICHMENT_EXCERPT_BUDGET

    mock_response = MagicMock()
    mock_response.content[0].text = (
        '{"review_summary": "Substantive feedback about bedside manner.",'
        ' "review_sentiment": "positive", "review_count": 42, "rating": 4.6}'
    )
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    results = [{
        "title": "Reviews", "url": "https://example.com",
        "content": "snippet", "raw_content": "R" * (budget + 1000),
    }]
    review_data = data_gatherer._extract_review_data_only(results, "Dr. Quiet")

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs['messages'][0]['content']
    assert "Full page text (excerpt)" in prompt
    assert "R" * budget in prompt
    assert "R" * (budget + 1) not in prompt
    assert review_data['review_count'] == 42


def test_dedupe_merges_same_provider_across_pages(data_gatherer: DataGathererAgent):
    """'Dr. Pritish Pawar' and 'Pritish Pawar, MD' are one person, not two candidates."""

    providers = [
        {"name": "Dr. Pritish Pawar", "rating": 0.0, "review_count": None,
         "review_summary": "No reviews available", "review_sentiment": "unknown"},
        {"name": "Pritish Pawar, MD", "rating": 4.7, "review_count": 61,
         "review_summary": "Caring and thorough.", "review_sentiment": "positive"},
    ]
    deduped = data_gatherer._dedupe_providers(providers)

    assert len(deduped) == 1
    assert deduped[0]["rating"] == 4.7
    assert deduped[0]["review_count"] == 61
    assert deduped[0]["review_summary"] == "Caring and thorough."

def test_dedupe_keeps_distinct_same_surname_providers(data_gatherer: DataGathererAgent):
    """A shared surname alone (overlap 0.5) must not merge two real people."""

    providers = [{"name": "John Ortega"}, {"name": "Maria Ortega"}]
    assert len(data_gatherer._dedupe_providers(providers)) == 2

def test_gather_skips_enrichment_when_disabled(data_gatherer: DataGathererAgent):
    """enrich=False defers review enrichment to the orchestrator's scoring step."""

    with patch.object(data_gatherer, "_search_providers", return_value=MOCK_TAVILY_SEARCH_RESPONSE["results"]), \
         patch.object(data_gatherer, "_extract_provider_data", return_value=[dict(p) for p in MOCK_GATHER_PROVIDERS_RESULT["providers"]]), \
         patch.object(data_gatherer, "_enrich_missing_reviews") as mock_enrich:
        result = data_gatherer.gather_providers(specialty="Neurology", location="Phoenix, AZ", enrich=False)

    assert result["status"] == "success"
    mock_enrich.assert_not_called()

def test_gather_attaches_code_computed_location_evidence(data_gatherer: DataGathererAgent):
    """Distance evidence comes from utils/geo.py in code — never from the LLM."""

    extracted = [
        {"name": "Dr. Near", "location": "123 Health St, Phoenix, AZ 85004"},
        {"name": "Dr. Vague", "location": "Somewhere"},
    ]
    with patch.object(data_gatherer, "_search_providers", return_value=MOCK_TAVILY_SEARCH_RESPONSE["results"]), \
         patch.object(data_gatherer, "_extract_provider_data", return_value=extracted):
        result = data_gatherer.gather_providers(
            specialty="Neurology", location="Phoenix, AZ 85004", enrich=False
        )

    near, vague = result["providers"]
    assert near["computed_distance_miles"] == 0.0
    assert near["location_match"] == "same_zip"
    assert vague["computed_distance_miles"] is None
    assert vague["location_match"] == "unknown"

def test_zip_only_location_builds_city_query(data_gatherer: DataGathererAgent):
    """A bare-ZIP input resolves to City, ST for the web query — no raw ZIP noise."""

    with patch.object(data_gatherer, "_search_providers", return_value=[]) as mock_search:
        result = data_gatherer.gather_providers(specialty="Neurology", location="85004")

    query = mock_search.call_args.args[0]
    assert "Phoenix, AZ" in query
    assert "85004" not in query
    assert result["status"] == "no_results"

def test_extraction_prompt_never_estimates_distance(data_gatherer: DataGathererAgent):
    """The old same-city/same-metro guessing instructions are gone."""

    mock_response = MagicMock()
    mock_response.content[0].text = "[]"
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    data_gatherer._extract_provider_data(
        MOCK_TAVILY_SEARCH_RESPONSE["results"], "Neurology", "Phoenix, AZ"
    )

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "NEVER estimate" in prompt
    assert "estimate 0-5 miles" not in prompt
    assert "same metro area" not in prompt

def test_enrich_query_is_name_and_city_without_specialty(data_gatherer: DataGathererAgent):
    """The specialty is an IDENTITY signal, not a RETRIEVAL one.

    Asserting "Neurology" in the query made healthgrades' Neurology directory
    pages outrank the target's own Sleep-Medicine-labeled profile, which then
    fell under the relevance floor — a doctor with real healthgrades reviews
    came back single-sourced. ZIPs still never reach a web query.
    """
    provider = {"name": "Dr. Quiet", "review_summary": "No reviews available",
                "review_sentiment": "unknown"}
    with patch.object(data_gatherer, "_search_providers", return_value=[]) as mock_search:
        data_gatherer.enrich_providers(
            [provider], "Phoenix, AZ 85004", specialty="Neurology"
        )

    query = mock_search.call_args.args[0]
    assert query == "Dr. Quiet reviews Phoenix, AZ"
    assert "Neurology" not in query
    assert "85004" not in query


def test_enrich_search_gives_every_platform_a_chance(data_gatherer: DataGathererAgent):
    """Five domains contesting five slots is how a cross-platform doctor
    came back single-sourced. Ask for more slots than there are platforms."""

    provider = {"name": "Dr. Quiet", "review_summary": "No reviews available",
                "review_sentiment": "unknown"}
    with patch.object(data_gatherer, "_search_providers", return_value=[]) as mock_search:
        data_gatherer.enrich_providers([provider], "Phoenix, AZ", specialty="Neurology")

    from agents.data_gatherer import _REVIEW_PLATFORM_DOMAINS

    kwargs = mock_search.call_args.kwargs
    assert kwargs["max_results"] > len(_REVIEW_PLATFORM_DOMAINS)
    assert kwargs["search_depth"] == "advanced"
    assert set(kwargs["include_domains"]) == set(_REVIEW_PLATFORM_DOMAINS)

def test_review_extraction_prompt_has_identity_guard(data_gatherer: DataGathererAgent):
    """Identity matches on the PERSON: unrelated fields are rejected, but a
    differing portal specialty label alone (Neurology vs Sleep Medicine for
    the same doctor) must never be — that over-rejection cost Dr. Khan his
    real Healthgrades data and left his own practice site as the source."""

    mock_response = MagicMock()
    mock_response.content[0].text = (
        '{"review_summary": "No reviews available", "review_sentiment": "unknown",'
        ' "review_count": null, "rating": null}'
    )
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    data_gatherer._extract_review_data_only(
        [{"title": "t", "url": "u", "content": "c"}], "Dr. Ortega", "Neurology",
        "Chandler, AZ 85224",
    )

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "IDENTITY CHECK" in prompt
    assert "a Neurology provider" in prompt
    assert "practicing in/near Chandler, AZ 85224" in prompt   # known location anchors identity
    assert "Sleep Medicine" in prompt                          # adjacent-label example present
    assert "NEVER grounds for rejection" in prompt
    assert "a different specialty or city" not in prompt        # old over-strict phrasing gone

def test_review_prompt_source_quality_rules(data_gatherer: DataGathererAgent):
    """Rating/count must come from independent platforms with stated totals —
    never the practice's own site, never a count of snippets read."""

    mock_response = MagicMock()
    mock_response.content[0].text = "{}"
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    data_gatherer._extract_review_data_only(
        [{"title": "t", "url": "u", "content": "c"}], "Dr. Ortega", "Neurology"
    )

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "SOURCE QUALITY RULES" in prompt
    assert "NEVER supply rating or review_count" in prompt      # own-site ban
    assert "LARGEST stated count" in prompt                     # multi-portal pick
    assert "NEVER the number of review snippets" in prompt      # no snippet counting

def test_enrichment_blocks_order_platforms_first(data_gatherer: DataGathererAgent):
    """The practice's own site goes LAST in the model's input, after portals."""

    mock_response = MagicMock()
    mock_response.content[0].text = "{}"
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    results = [
        {"title": "own", "url": "https://chandlerneurologyandsleep.com/testimonials", "content": "own-site"},
        {"title": "hg", "url": "https://www.healthgrades.com/physician/dr-khan", "content": "hg-profile"},
        {"title": "vitals", "url": "https://www.vitals.com/doctors/khan", "content": "vitals-profile"},
    ]
    data_gatherer._extract_review_data_only(results, "Dr. Khan", "Neurology")

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert prompt.index("healthgrades.com") < prompt.index("chandlerneurologyandsleep.com")
    assert prompt.index("vitals.com/doctors") < prompt.index("chandlerneurologyandsleep.com")

def test_enrich_one_passes_provider_location(data_gatherer: DataGathererAgent):
    """The provider's known city reaches the identity check."""

    provider = {"name": "Dr. Khan", "location": "3195 S Price Rd, Chandler, AZ 85248",
                "review_summary": "No reviews available", "review_sentiment": "unknown"}
    with patch.object(data_gatherer, "_search_providers", return_value=[{"url": "u"}]), \
         patch.object(data_gatherer, "_extract_review_data_only", return_value={}) as mock_extract:
        data_gatherer._enrich_one(provider, "Chandler, AZ", "Neurology")

    args = mock_extract.call_args.args
    assert args[3] == "3195 S Price Rd, Chandler, AZ 85248"

def test_candidate_extraction_preserves_review_source_url(data_gatherer: DataGathererAgent):
    """The page a rating came from survives cleaning — provenance for the card."""

    mock_response = MagicMock()
    mock_response.content[0].text = (
        '[{"name": "Dr. Sourced", "specialty": "Neurology", "location": "Phoenix, AZ",'
        ' "rating": 4.5, "review_count": 61,'
        ' "review_source_url": "https://www.healthgrades.com/physician/dr-sourced"}]'
    )
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    providers = data_gatherer._extract_provider_data(
        MOCK_TAVILY_SEARCH_RESPONSE["results"], "Neurology", "Phoenix, AZ"
    )

    assert providers[0]["review_source_url"] == "https://www.healthgrades.com/physician/dr-sourced"
    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "review_source_url" in prompt

def test_enrichment_extracts_insurance_and_source(data_gatherer: DataGathererAgent):
    """Enrichment reads directory profiles — it must pull insurance lists and cite pages."""

    mock_response = MagicMock()
    mock_response.content[0].text = (
        '{"review_summary": "Kind and thorough.", "review_sentiment": "positive",'
        ' "review_count": 61, "rating": 4.5,'
        ' "review_source_url": "https://healthgrades.com/x",'
        ' "insurance_accepted": ["UnitedHealth", "Aetna"],'
        ' "insurance_source_url": "https://healthgrades.com/x#insurance"}'
    )
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    review_data = data_gatherer._extract_review_data_only(
        [{"title": "t", "url": "https://healthgrades.com/x", "content": "c"}], "Dr. Sourced", "Neurology"
    )

    assert review_data["review_source_url"] == "https://healthgrades.com/x"
    assert review_data["insurance_accepted"] == ["UnitedHealth", "Aetna"]
    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "insurance_accepted" in prompt
    assert "review_source_url" in prompt

def test_merge_backfills_insurance_and_source_without_overwriting(data_gatherer: DataGathererAgent):
    """Enriched insurance fills gaps only; candidate-pass evidence is never clobbered."""

    empty = {"name": "Dr. Empty", "insurance_accepted": []}
    data_gatherer._merge_review_data(empty, {
        "review_summary": "Great care.", "review_sentiment": "positive",
        "review_count": 10, "rating": 4.0,
        "review_source_url": "https://vitals.com/dr-empty",
        "insurance_accepted": ["Cigna"], "insurance_source_url": "https://vitals.com/dr-empty",
    })
    assert empty["insurance_accepted"] == ["Cigna"]
    assert empty["insurance_source_url"] == "https://vitals.com/dr-empty"
    assert empty["review_source_url"] == "https://vitals.com/dr-empty"

    kept = {"name": "Dr. Kept", "insurance_accepted": ["Aetna PPO"]}
    data_gatherer._merge_review_data(kept, {
        "review_summary": "Fine.", "review_sentiment": "positive",
        "review_count": 5, "rating": 4.0,
        "review_source_url": None,
        "insurance_accepted": ["Cigna"], "insurance_source_url": "https://x.com",
    })
    assert kept["insurance_accepted"] == ["Aetna PPO"]
    # The LIST is not overwritten — the candidate pass already found one — but
    # the SOURCE is still adopted. It used to be nested inside the list guard,
    # so a known-good URL was discarded and the card rendered eight named
    # payers attributed to nothing.
    assert kept["insurance_source_url"] == "https://x.com"

def test_candidate_queries_span_phrasings(data_gatherer: DataGathererAgent):
    """Discovery fans out several phrasings of one city — never the ZIP.
    Exactly one spec is domain-restricted to the review platforms (and deep):
    platform ratings must land at extraction time, not enrichment."""
    from agents.data_gatherer import _REVIEW_PLATFORM_DOMAINS

    specs = data_gatherer._candidate_queries("Neurology", "Chandler, AZ")
    queries = [s["query"] for s in specs]
    assert len(queries) == 3
    assert len(set(queries)) == 3  # all distinct
    assert all("Chandler, AZ" in q for q in queries)
    assert any("best" in q.lower() for q in queries)    # listicle variant

    platform_specs = [s for s in specs if s.get("include_domains")]
    assert len(platform_specs) == 1
    assert set(platform_specs[0]["include_domains"]) == set(_REVIEW_PLATFORM_DOMAINS)
    assert platform_specs[0]["search_depth"] == "advanced"
    # Domain names no longer ride in the query text — the restriction targets
    assert "healthgrades" not in platform_specs[0]["query"]

def test_discover_candidates_merges_and_dedupes_by_url(data_gatherer: DataGathererAgent):
    """Parallel queries are merged with per-URL dedup; each query is issued."""
    pages = {
        "qA": [{"url": "https://a.com", "title": "A"}, {"url": "https://shared.com", "title": "S"}],
        "qB": [{"url": "https://shared.com", "title": "S"}, {"url": "https://b.com", "title": "B"}],
    }
    with patch.object(data_gatherer, "_search_providers", side_effect=lambda q, **k: pages[q]):
        merged = data_gatherer._discover_candidates(["qA", "qB"], max_results=20)

    urls = sorted(r["url"] for r in merged)
    assert urls == ["https://a.com", "https://b.com", "https://shared.com"]  # shared collapsed

def test_discover_candidates_interleaves_queries_round_robin(data_gatherer: DataGathererAgent):
    """Every query contributes to the HEAD of the merged list — downstream
    extraction caps ([:20]/[:18]) must sample all phrasings, not just query 1."""
    pages = {
        "q1": [{"url": f"https://q1-{i}.com"} for i in range(3)],
        "q2": [{"url": f"https://q2-{i}.com"} for i in range(3)],
        "q3": [{"url": f"https://q3-{i}.com"} for i in range(3)],
    }
    with patch.object(data_gatherer, "_search_providers", side_effect=lambda q, **k: pages[q]):
        merged = data_gatherer._discover_candidates(["q1", "q2", "q3"], max_results=20)

    head = [r["url"] for r in merged[:3]]
    assert head == ["https://q1-0.com", "https://q2-0.com", "https://q3-0.com"]
    assert [r["url"] for r in merged[3:6]] == ["https://q1-1.com", "https://q2-1.com", "https://q3-1.com"]

def test_discovery_platform_query_targets_domains(data_gatherer: DataGathererAgent):
    """The platform spec's include_domains + advanced depth must reach the
    search layer for exactly one of the three discovery calls — the other
    phrasings stay unrestricted at the global depth."""
    from agents.data_gatherer import _REVIEW_PLATFORM_DOMAINS

    calls = []

    def record(query, **kwargs):
        calls.append((query, kwargs))
        return []

    specs = data_gatherer._candidate_queries("Neurology", "Chandler, AZ")
    with patch.object(data_gatherer, "_search_providers", side_effect=record):
        data_gatherer._discover_candidates(specs, max_results=20)

    assert len(calls) == 3
    restricted = [(q, k) for q, k in calls if k.get("include_domains")]
    assert len(restricted) == 1
    _, kwargs = restricted[0]
    assert set(kwargs["include_domains"]) == set(_REVIEW_PLATFORM_DOMAINS)
    assert kwargs["search_depth"] == "advanced"
    for _, kwargs in calls:
        if not kwargs.get("include_domains"):
            assert kwargs.get("search_depth") is None  # global knob governs

def test_search_depth_override_reaches_tavily_and_cost_tracker(data_gatherer: DataGathererAgent):
    """A per-call depth override must drive BOTH the Tavily request and the
    cost record — recording the config depth would misprice advanced calls
    (they cost 2 credits, not 1)."""
    data_gatherer.config.TAVILY_SEARCH_DEPTH = "basic"
    data_gatherer.tavily_client.search.return_value = {"results": []}
    tracker = MagicMock()

    with patch("agents.data_gatherer.get_cost_tracker", return_value=tracker):
        data_gatherer._search_providers("q", search_depth="advanced")
        assert data_gatherer.tavily_client.search.call_args.kwargs["search_depth"] == "advanced"
        assert tracker.record_tavily.call_args.kwargs["depth"] == "advanced"

        data_gatherer._search_providers("q")
        assert data_gatherer.tavily_client.search.call_args.kwargs["search_depth"] == "basic"
        assert tracker.record_tavily.call_args.kwargs["depth"] == "basic"

def test_ring_expansion_fires_when_pool_thin(data_gatherer: DataGathererAgent):
    """A thin home pool triggers nearby-city expansion and merges the results."""
    home = [{"name": "Dr. Home", "location": "Chandler, AZ 85224"}]
    ring = [{"name": "Dr. Ring", "location": "Gilbert, AZ 85234"}]

    with patch.object(data_gatherer, "_search_providers", return_value=[{"url": "u", "title": "t"}]), \
         patch.object(data_gatherer, "_extract_provider_data", side_effect=[home, ring]), \
         patch("agents.data_gatherer.nearby_cities", return_value=["Gilbert, AZ"]) as mock_nearby:
        result = data_gatherer.gather_providers(
            specialty="Neurology", location="Chandler, AZ", enrich=False
        )

    mock_nearby.assert_called_once()
    names = {p["name"] for p in result["providers"]}
    assert names == {"Dr. Home", "Dr. Ring"}
    assert result["search_metadata"]["query_count"] == 4  # 3 home + 1 ring
    # Recorded where it is KNOWN — the UI must not have to infer it by
    # hardcoding the home-phrasing count.
    assert result["search_metadata"]["ring_expanded"] is True

def test_ring_expansion_rescues_empty_home_pool(data_gatherer: DataGathererAgent):
    """Zero extractable providers at home — the thinnest pool — must still
    ring out, not short-circuit to no_results."""
    ring = [{"name": "Dr. Ring Only", "location": "Gilbert, AZ 85234"}]

    with patch.object(data_gatherer, "_search_providers", return_value=[{"url": "u", "title": "t"}]), \
         patch.object(data_gatherer, "_extract_provider_data", side_effect=[[], ring]), \
         patch("agents.data_gatherer.nearby_cities", return_value=["Gilbert, AZ"]) as mock_nearby:
        result = data_gatherer.gather_providers(
            specialty="Neurology", location="Sun Lakes, AZ", enrich=False
        )

    mock_nearby.assert_called_once()
    assert result["status"] == "success"
    assert [p["name"] for p in result["providers"]] == ["Dr. Ring Only"]

def test_ring_expansion_skipped_when_pool_is_rich(data_gatherer: DataGathererAgent):
    """A large, geographically-spread pool needs no expansion — no ring cost."""
    rich = [
        {"name": f"Dr. {i}", "location": f"City{i}, AZ 8500{i}"} for i in range(12)
    ]
    with patch.object(data_gatherer, "_search_providers", return_value=[{"url": "u", "title": "t"}]), \
         patch.object(data_gatherer, "_extract_provider_data", return_value=rich), \
         patch("agents.data_gatherer.nearby_cities") as mock_nearby:
        result = data_gatherer.gather_providers(
            specialty="Neurology", location="Phoenix, AZ", enrich=False
        )

    mock_nearby.assert_not_called()
    assert result["search_metadata"]["query_count"] == 3  # home phrasings only
    assert result["search_metadata"]["ring_expanded"] is False


def test_a_healthy_single_city_pool_does_not_ring_out(data_gatherer: DataGathererAgent):
    """The case the deleted clustering trigger used to catch — and the reason
    it had to go.

    A Chandler search returning twelve Chandler providers is the IDEAL outcome,
    not a defect. The old `distinct locations < 3` clause fired on it, spending
    two extra searches and an extraction to import providers from a city the
    user didn't ask for, who then compete for the enrichment and judging budget.
    Keyed on city alone the metric would fire on EVERY single-city search; keyed
    on ZIP-else-city it measured our address-parsing coverage instead of
    geography. No unit repairs it, so the trigger is gone."""
    single_city = [
        {"name": f"Dr. {i}", "location": "Chandler, AZ"} for i in range(12)
    ]
    with patch.object(data_gatherer, "_search_providers", return_value=[{"url": "u", "title": "t"}]), \
         patch.object(data_gatherer, "_extract_provider_data", return_value=single_city), \
         patch("agents.data_gatherer.nearby_cities") as mock_nearby:
        result = data_gatherer.gather_providers(
            specialty="Neurology", location="Chandler, AZ", enrich=False
        )

    mock_nearby.assert_not_called()
    assert result["search_metadata"]["query_count"] == 3
    assert result["search_metadata"]["ring_expanded"] is False
    assert len(result["providers"]) == 12


def test_single_query_mode_is_the_escape_hatch(data_gatherer: DataGathererAgent):
    """MULTI_QUERY_ENABLED=False restores exactly one discovery search."""
    data_gatherer.config.MULTI_QUERY_ENABLED = False
    with patch.object(data_gatherer, "_search_providers", return_value=[{"url": "u"}]) as mock_search, \
         patch.object(data_gatherer, "_extract_provider_data", return_value=[{"name": "Dr. Solo", "location": "Chandler, AZ"}]):
        result = data_gatherer.gather_providers(
            specialty="Neurology", location="Chandler, AZ", enrich=False
        )

    mock_search.assert_called_once()
    assert result["search_metadata"]["query_count"] == 1

def test_candidate_excerpt_skips_boilerplate_head(data_gatherer: DataGathererAgent):
    """The prompt carries anchor-centered page content, not the nav chrome a
    blind head-truncation used to capture."""

    mock_response = MagicMock()
    mock_response.content[0].text = "[]"
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    filler = "generic wellness marketing text without key terms " * 70   # ~3.4K chars
    deep_content = "Dr. Deep Body is a Neurology specialist with excellent patient reviews. "
    results = [{
        "title": "Directory", "url": "https://example.com/dir",
        "content": "snippet", "score": 0.9,
        "raw_content": "Accept all cookies\nSign in | Register\n" + filler + deep_content,
    }]

    data_gatherer._extract_provider_data(results, "Neurology", "Phoenix, AZ")

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Dr. Deep Body" in prompt          # deep content reached the LLM
    assert "Accept all cookies" not in prompt  # chrome stripped

def test_enrichment_excerpt_anchors_on_provider_name(data_gatherer: DataGathererAgent):
    """Enrichment windows center on THIS provider's name, not another doctor
    who happens to open the page."""

    mock_response = MagicMock()
    mock_response.content[0].text = '{"review_summary": "No reviews available", "review_sentiment": "unknown", "review_count": null, "rating": null}'
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    other = "Dr. Wrong Person, a cardiology physician, has many reviews here. " * 40  # ~2.6K
    target = "Dr. Ortega earns consistent praise in neurology patient reviews for clarity. "
    results = [{
        "title": "t", "url": "https://vitals.com/x", "content": "c", "score": 0.8,
        "raw_content": other + target,
    }]

    data_gatherer._extract_review_data_only(results, "Dr. Maria Ortega", "Neurology")

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Dr. Ortega earns consistent praise" in prompt

def test_enrichment_reaches_a_profile_header_rating(data_gatherer: DataGathererAgent):
    """WIRING guard, not a helper test.

    `build_excerpt(include_head=...)` can be correct while nothing asks for it:
    reverting the call site's `include_head=` left the whole suite green. Same
    reason the Responsible-AI panel has a wiring test — a helper test alone
    would let the argument be deleted and nobody would know until a live run
    quietly lost every profile-header rating again."""

    mock_response = MagicMock()
    mock_response.content[0].text = '{"review_summary": "No reviews available", "review_sentiment": "unknown", "review_count": null, "rating": null}'
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    # The stated pair appears ONCE, at the top; the name-heavy comment body and
    # the vocabulary-dense percentage table below both out-rank it on density.
    # The figures are deliberately NOT the ones the prompt's own rules quote as
    # examples — the first draft of this test asserted "3.4 out of 5 (23
    # ratings)" and passed against the instruction text while the excerpt had
    # none of it.
    header = "Dr. Ellen Kuniyoshi, MD. 2.9 out of 5 (147 ratings). 28 years of experience. "
    body = "Dr. Kuniyoshi was thorough. Kuniyoshi listened. review rating patient " * 40
    table = "5 star 48% 1 star 39% ratings reviews review " * 30
    results = [{
        "title": "t", "url": "https://healthgrades.com/physician/dr-ellen-kuniyoshi",
        "content": "c", "score": 0.9, "raw_content": header + body + table,
    }]

    data_gatherer._extract_review_data_only(results, "Dr. Ellen Kuniyoshi", "Neurology")

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    excerpt = prompt[prompt.index("Full page text (excerpt):"):]
    assert "2.9 out of 5 (147 ratings)" in excerpt
    assert "28 years of experience" in excerpt


def test_truncated_extraction_is_logged_not_swallowed(data_gatherer: DataGathererAgent, caplog):
    """Same class as the judge's unchecked `finish_reason` (round 9): a response
    cut at max_tokens fails to parse, the handler returns nothing, and the
    provider becomes indistinguishable from one the web had no data for. The
    gatherer was never swept for it."""
    mock_response = MagicMock()
    mock_response.stop_reason = "max_tokens"
    mock_response.content[0].text = '{"review_summary": "Patients praise her thoroughness'  # cut mid-string
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    results = [{"title": "t", "url": "https://healthgrades.com/p/x", "content": "c", "score": 0.9}]
    with caplog.at_level(logging.WARNING):
        data_gatherer._extract_review_data_only(results, "Dr. Ellen Kuniyoshi", "Neurology")

    assert any("max_tokens" in record.getMessage() for record in caplog.records), \
        "truncation must announce itself"

    # And the budget must stay above the early two-block figure it outgrew.
    assert data_gatherer.anthropic_client.messages.create.call_args.kwargs["max_tokens"] >= 1500


def test_non_platform_pages_do_not_reserve_the_head(data_gatherer: DataGathererAgent):
    """The head window is spent only where a header carries the rating. On a
    practice site it is a nav bar, and the anchors should keep all three."""
    assert _is_review_platform_url("https://healthgrades.com/physician/dr-x") is True
    assert _is_review_platform_url("https://vitals.com/doctors/dr-x") is True
    assert _is_review_platform_url("https://chandlerneurology.com/about") is False
    assert _is_review_platform_url(None) is False


def test_low_relevance_results_dropped_by_floor_not_sorted(data_gatherer: DataGathererAgent):
    """Score is a floor, never a sort key: junk (0.1) is dropped; order of the
    survivors is unchanged; scoreless results are kept."""

    mock_response = MagicMock()
    mock_response.content[0].text = "[]"
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    results = [
        {"title": "first", "url": "https://a.com", "content": "keep-A", "score": 0.4},
        {"title": "junk", "url": "https://junk.com", "content": "drop-me", "score": 0.1},
        {"title": "third", "url": "https://b.com", "content": "keep-B"},  # no score
        {"title": "hot", "url": "https://c.com", "content": "keep-C", "score": 0.95},
    ]
    data_gatherer._extract_provider_data(results, "Neurology", "Phoenix, AZ")

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "drop-me" not in prompt
    assert "keep-A" in prompt and "keep-B" in prompt and "keep-C" in prompt
    # order preserved (no re-sort by score: 0.95 must NOT jump ahead of 0.4)
    assert prompt.index("keep-A") < prompt.index("keep-C")

def test_all_low_scores_falls_back_to_unfiltered(data_gatherer: DataGathererAgent):
    """If the floor would empty the list, keep everything rather than extract from nothing."""

    mock_response = MagicMock()
    mock_response.content[0].text = "[]"
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    results = [{"title": "only", "url": "https://x.com", "content": "still-here", "score": 0.05}]
    data_gatherer._extract_provider_data(results, "Neurology", "Phoenix, AZ")

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "still-here" in prompt

def test_parse_rating_normalizes_every_stated_format():
    """Pages phrase ratings many ways; the parser reads them all — and a review
    COUNT can never masquerade as a rating."""
    from agents.data_gatherer import _parse_rating

    assert _parse_rating(4.7) == 4.7
    assert _parse_rating("4.7") == 4.7
    assert _parse_rating("4.7/5") == 4.7
    assert _parse_rating("1.2 / 5") == 1.2
    assert _parse_rating("4 stars") == 4.0
    assert _parse_rating("4.5 out of 5") == 4.5
    assert _parse_rating("Excellent (5/5)") == 5.0
    assert _parse_rating("271") is None      # out of range — that's a count
    assert _parse_rating(0) is None
    assert _parse_rating("no rating") is None
    assert _parse_rating(None) is None

def test_select_review_observation_prefers_credible_pairs():
    """The Khan case: a 1-review 1.0 pair must NOT headline over a 5/5 whose
    count didn't scrape; among credible pairs the largest count wins."""
    from agents.data_gatherer import _select_review_observation

    # Real shape from the Khan debug: rating-only claims spanning 1.2..5.0 —
    # any single headline would mislead, so selection DECLINES (the card shows
    # "Across platforms" instead)
    khan = [
        {"source_url": "https://www.zocdoc.com/x", "rating": "1.2/5", "review_count": None},
        {"source_url": "https://medicalnewstoday.com/x", "rating": 1.0, "review_count": 1},
        {"source_url": "https://www.ratemds.com/x", "rating": 5.0, "review_count": None},
    ]
    headline, normalized = _select_review_observation(khan)
    assert headline is None                  # disagreement guard
    assert len(normalized) == 3
    assert normalized[0]["rating"] == 1.2    # string form normalized

    # Agreeing rating-only claims: first (platform-order) wins
    agreeing = [
        {"source_url": "https://www.vitals.com/x", "rating": 4.5, "review_count": None},
        {"source_url": "https://webmd.com/x", "rating": 4.0, "review_count": None},
    ]
    headline, _ = _select_review_observation(agreeing)
    assert headline["source_url"] == "https://www.vitals.com/x"

    # A credible pair beats everything (even amid disagreement); largest count wins
    with_pairs = khan + [
        {"source_url": "https://healthgrades.com/x", "rating": 4.0, "review_count": 31},
        {"source_url": "https://webmd.com/x", "rating": 5.0, "review_count": 271},
    ]
    headline, _ = _select_review_observation(with_pairs)
    assert headline["review_count"] == 271
    assert headline["source_url"] == "https://webmd.com/x"

    # Nothing usable -> None
    headline, normalized = _select_review_observation([{"source_url": "u"}, "junk", None])
    assert headline is None and normalized == []

def test_merge_declines_headline_on_conflicting_observations(data_gatherer: DataGathererAgent):
    """When platforms wildly disagree, the provider stays unrated — the model's
    own single pick (one of the conflicting numbers) must NOT sneak back in."""

    provider = {"name": "Dr. Khan", "rating": 0.0, "review_count": None}
    data_gatherer._merge_review_data(provider, {
        "review_summary": "Praised.", "review_sentiment": "positive",
        "review_count": None, "rating": "1.2/5",
        "review_source_url": "https://www.zocdoc.com/x",
        "review_observations": [
            {"source_url": "https://www.zocdoc.com/x", "rating": 1.2, "review_count": None},
            {"source_url": "https://www.ratemds.com/x", "rating": 5.0, "review_count": None},
        ],
    })

    assert not provider.get("rating")                       # declined, still unrated
    assert len(provider["review_observations"]) == 2        # but the disagreement is visible

def test_merge_uses_observations_and_stores_them(data_gatherer: DataGathererAgent):
    """The code-side pick drives the headline; all observations ride on the provider."""

    provider = {"name": "Dr. Khan", "rating": 0.0, "review_count": None}
    data_gatherer._merge_review_data(provider, {
        "review_summary": "Praised for attentiveness.", "review_sentiment": "positive",
        "review_count": None, "rating": "1.2/5",             # model's own (bad) pick
        "review_source_url": "https://www.zocdoc.com/x",
        "review_observations": [
            {"source_url": "https://www.ratemds.com/x", "rating": 5.0, "review_count": None},
            {"source_url": "https://www.healthgrades.com/x", "rating": 4.0, "review_count": 31},
        ],
    })

    assert provider["rating"] == 4.0                          # credible pair beat rating-only
    assert provider["review_count"] == 31
    assert provider["review_source_url"] == "https://www.healthgrades.com/x"
    assert len(provider["review_observations"]) == 2

def test_merge_fallback_parses_string_ratings(data_gatherer: DataGathererAgent):
    """Without observations, the single-pair fallback no longer chokes on '4.7/5'."""

    provider = {"name": "Dr. Solo", "rating": 0.0}
    data_gatherer._merge_review_data(provider, {
        "review_summary": "Fine.", "review_sentiment": "positive",
        "review_count": 12, "rating": "4.7/5",
        "review_source_url": "https://vitals.com/x",
    })
    assert provider["rating"] == 4.7
    assert provider["review_count"] == 12

def test_review_prompt_requests_observations(data_gatherer: DataGathererAgent):
    """The model transcribes per-platform observations; derivation is banned."""

    mock_response = MagicMock()
    mock_response.content[0].text = "{}"
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    data_gatherer._extract_review_data_only(
        [{"title": "t", "url": "u", "content": "c"}], "Dr. Khan", "Neurology"
    )

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "review_observations" in prompt
    assert "NEVER derive a rating or count from star-percentage distributions" in prompt
    assert "Bare JSON numbers only" in prompt

def test_platform_observations_outrank_hospital_site_pairs():
    """Field bug F2: Banner Health's own 4.7 (486 surveys) must not
    out-headline healthgrades' 2.8 (16 reviews) — employer sites have a
    marketing incentive. Platform class wins; banner stays in the caption."""
    from agents.data_gatherer import _select_review_observation

    hashmi = [
        {"source_url": "https://doctors.bannerhealth.com/x", "rating": 4.7, "review_count": 486},
        {"source_url": "https://www.healthgrades.com/x", "rating": 2.8, "review_count": 16},
        {"source_url": "https://www.ratemds.com/x", "rating": None, "review_count": 48},
    ]
    headline, normalized = _select_review_observation(hashmi)
    assert headline["source_url"] == "https://www.healthgrades.com/x"
    assert headline["rating"] == 2.8
    assert len(normalized) == 3   # banner still visible in Across-platforms

def test_non_platform_observations_headline_only_by_forfeit():
    from agents.data_gatherer import _select_review_observation

    # No platform observation at all -> a non-platform pair may headline
    only_banner = [{"source_url": "https://doctors.bannerhealth.com/x", "rating": 4.7, "review_count": 486}]
    headline, _ = _select_review_observation(only_banner)
    assert headline["review_count"] == 486

    # Platform observations that DECLINE (disagreement) are not overridden
    conflicted = [
        {"source_url": "https://www.zocdoc.com/x", "rating": 1.2, "review_count": None},
        {"source_url": "https://www.ratemds.com/x", "rating": 5.0, "review_count": None},
        {"source_url": "https://doctors.bannerhealth.com/x", "rating": 4.7, "review_count": 486},
    ]
    headline, _ = _select_review_observation(conflicted)
    assert headline is None       # decline stands; self-published can't win by forfeit

def test_enrichment_is_one_platform_restricted_advanced_search(data_gatherer: DataGathererAgent):
    """The per-provider pass is ONE include_domains-restricted advanced
    search: rating/count may only come from the platforms anyway, so the
    credits buy exactly those result slots (no open query for SEO spam to
    crowd, no conditional rescue search)."""

    results = [
        {"url": "https://www.zocdoc.com/doctor/yu", "title": "zocdoc"},
        {"url": "https://www.healthgrades.com/physician/yu", "title": "hg"},
    ]
    provider = {"name": "Dr. Kan Yu", "location": "Gilbert, AZ",
                "review_summary": "No reviews available", "review_sentiment": "unknown"}

    with patch.object(data_gatherer, "_search_providers", return_value=results) as mock_search, \
         patch.object(data_gatherer, "_extract_review_data_only", return_value={}) as mock_extract:
        data_gatherer._enrich_one(provider, "Chandler, AZ", "Neurology")

    from agents.data_gatherer import _REVIEW_PLATFORM_DOMAINS
    assert mock_search.call_count == 1
    kwargs = mock_search.call_args.kwargs
    assert set(kwargs["include_domains"]) == set(_REVIEW_PLATFORM_DOMAINS)
    assert kwargs["search_depth"] == "advanced"
    assert kwargs["include_raw_content"] is True
    assert mock_extract.call_args.args[0] == results
    # And the query used the provider's OWN city, not the search city
    assert "Gilbert, AZ" in mock_search.call_args.args[0]

def test_credible_platform_pair_supersedes_stale_candidate_count(data_gatherer: DataGathererAgent):
    """Field wart: candidate pass scraped '(1 review)' from arbitrary text;
    healthgrades states 4.0 (15). The platform pair must supersede — but a
    NON-platform pair still respects backfill-only."""

    provider = {"name": "Dr. Lockwood", "rating": 4.0, "review_count": 1}
    data_gatherer._merge_review_data(provider, {
        "review_summary": "Praised.", "review_sentiment": "positive",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [
            {"source_url": "https://www.healthgrades.com/x", "rating": 4.0, "review_count": 15},
        ],
    })
    assert provider["review_count"] == 15    # platform pair overrode the stale 1

    banner_only = {"name": "Dr. Hospital", "rating": 3.9, "review_count": 4}
    data_gatherer._merge_review_data(banner_only, {
        "review_summary": "Praised.", "review_sentiment": "positive",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [
            {"source_url": "https://doctors.bannerhealth.com/x", "rating": 4.9, "review_count": 400},
        ],
    })
    assert banner_only["rating"] == 3.9      # non-platform pair: backfill-only
    assert banner_only["review_count"] == 4

def test_merge_stores_blended_fields(data_gatherer: DataGathererAgent):
    """≥2 platform pairs -> count-weighted blend fields for the SCORE, while
    the headline (largest credible pair) stays the display attribution."""
    provider = {"name": "Dr. Qureshi"}
    data_gatherer._merge_review_data(provider, {
        "review_summary": "Mixed.", "review_sentiment": "mixed",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [
            {"source_url": "https://www.vitals.com/x", "rating": 3.5, "review_count": 16},
            {"source_url": "https://www.healthgrades.com/x", "rating": 2.1, "review_count": 13},
            {"source_url": "https://doctor.webmd.com/x", "rating": 3.5, "review_count": 1},
        ],
    })
    # (3.5*16 + 2.1*13 + 3.5*1) / 30 = 2.89 -> 2.9
    assert provider["blended_rating"] == 2.9
    assert provider["blended_review_count"] == 30
    assert provider["blended_platform_count"] == 3
    # Headline semantics unchanged: vitals is the largest credible pair
    assert provider["rating"] == 3.5
    assert provider["review_count"] == 16

def test_blend_requires_two_pairs_and_agreement(data_gatherer: DataGathererAgent):
    """One pair: nothing to blend. Extreme pair disagreement: the blend
    declines (averaging a 1.2 against a 5.0 manufactures a number nobody
    reported) even though a headline pair still shows."""
    one_pair = {"name": "Dr. One"}
    data_gatherer._merge_review_data(one_pair, {
        "review_summary": "Praised.", "review_sentiment": "positive",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [
            {"source_url": "https://www.healthgrades.com/x", "rating": 4.0, "review_count": 15},
        ],
    })
    assert "blended_rating" not in one_pair

    split = {"name": "Dr. Split"}
    data_gatherer._merge_review_data(split, {
        "review_summary": "Polarized.", "review_sentiment": "mixed",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [
            {"source_url": "https://www.vitals.com/x", "rating": 5.0, "review_count": 40},
            {"source_url": "https://www.zocdoc.com/x", "rating": 1.2, "review_count": 10},
        ],
    })
    assert "blended_rating" not in split
    assert split["rating"] == 5.0            # headline: largest credible pair still shows

def test_enrichment_backfills_years_experience(data_gatherer: DataGathererAgent):
    """Platform profiles state tenure; unscraped years cost real ranking
    points (neutral 45 vs up to 100) — backfill when missing, never clobber."""
    missing = {"name": "Dr. De Lima", "years_experience": None}
    data_gatherer._merge_review_data(missing, {
        "review_summary": "No reviews available", "review_sentiment": "unknown",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [], "years_experience": 20,
    })
    assert missing["years_experience"] == 20

    present = {"name": "Dr. Qureshi", "years_experience": 26}
    data_gatherer._merge_review_data(present, {
        "review_summary": "No reviews available", "review_sentiment": "unknown",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [], "years_experience": 10,
    })
    assert present["years_experience"] == 26  # never clobbered

    junk = {"name": "Dr. Junk", "years_experience": None}
    data_gatherer._merge_review_data(junk, {
        "review_summary": "No reviews available", "review_sentiment": "unknown",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [], "years_experience": "decades of care",
    })
    assert junk["years_experience"] is None   # unparseable stays missing

def test_enrichment_gives_single_pair_leaders_second_opinion(data_gatherer: DataGathererAgent):
    """One platform pair is one platform's opinion, not settled numbers.
    City-level discovery is a healthgrades monoculture (only it publishes
    per-city directory pages that rank), so a leader who 'arrives rated'
    must still get the name-query pass that reaches vitals/webmd — that
    second pair is what the cross-platform blend needs to fire."""
    single_pair_leader = {
        "name": "Dr. Hodgson", "rating": 4.6, "review_count": 33,
        "review_summary": "Praised for unrushed visits.", "review_sentiment": "positive",
        "review_source_url": "https://www.healthgrades.com/physician/dr-hodgson",
        "review_observations": [
            {"source_url": "https://www.healthgrades.com/physician/dr-hodgson",
             "rating": 4.6, "review_count": 33},
        ],
    }
    enriched_names = []

    def fake_enrich(provider, location, kw="", spec=""):
        enriched_names.append(provider["name"])
        return provider

    with patch.object(data_gatherer, "_enrich_one", side_effect=fake_enrich):
        data_gatherer._enrich_missing_reviews(
            [single_pair_leader], "Chandler, AZ", "Neurology"
        )

    assert enriched_names == ["Dr. Hodgson"]

def _pool_provider(name: str, pairs: int) -> dict:
    """Fixture provider holding `pairs` platform rating+count pairs."""
    domains = ["healthgrades.com/physician", "vitals.com/doctors"]
    obs = [
        {"source_url": f"https://www.{domains[i]}/{name}",
         "rating": 4.0 + i * 0.2, "review_count": 20 + i}
        for i in range(pairs)
    ]
    provider = {
        "name": name,
        "rating": 4.0 if pairs else None,
        "review_count": 20 if pairs else None,
        "review_summary": "Praised." if pairs else "No reviews available",
        "review_sentiment": "positive" if pairs else "unknown",
        "review_observations": obs,
    }
    return provider

def test_same_domain_duplicate_never_poses_as_second_opinion(data_gatherer: DataGathererAgent):
    """The Khan field case: portals list one doctor under two URLs of the
    SAME platform (healthgrades' Neurology and Sleep Medicine paths, both
    4.0/31). One platform is one voice — one stored observation, no blend,
    and the headline is just that platform's pair."""
    provider = {"name": "Dr. Khan"}
    data_gatherer._merge_review_data(provider, {
        "review_summary": "Praised.", "review_sentiment": "positive",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [
            {"source_url": "https://www.healthgrades.com/physician/dr-khan-neurology",
             "rating": 4.0, "review_count": 31},
            {"source_url": "https://www.healthgrades.com/physician/dr-khan-sleep",
             "rating": 4.0, "review_count": 31},
        ],
    })

    assert len(provider["review_observations"]) == 1   # one card line, not two
    assert "blended_rating" not in provider            # nothing to blend from one voice
    assert provider["rating"] == 4.0
    assert provider["review_count"] == 31

def test_trigger_fires_despite_same_domain_duplicate_pairs(data_gatherer: DataGathererAgent):
    """Two same-domain pairs must never read as two platform opinions —
    live, Khan skipped enrichment on exactly this, so the webmd/vitals
    pairs that exist for him were never fetched."""
    khan = {
        "name": "Dr. Khan", "rating": 4.0, "review_count": 31,
        "review_summary": "Praised.", "review_sentiment": "positive",
        "review_observations": [
            {"source_url": "https://www.healthgrades.com/a", "rating": 4.0, "review_count": 31},
            {"source_url": "https://www.healthgrades.com/b", "rating": 4.0, "review_count": 31},
        ],
    }
    enriched_names = []

    def fake_enrich(provider, location, spec="", user_location=""):
        enriched_names.append(provider["name"])
        return provider

    with patch.object(data_gatherer, "_enrich_one", side_effect=fake_enrich):
        data_gatherer._enrich_missing_reviews([khan], "Chandler, AZ", "Neurology")

    assert enriched_names == ["Dr. Khan"]

def test_platform_pairs_keep_largest_count_per_domain():
    from agents.data_gatherer import _platform_rating_pairs

    pairs = _platform_rating_pairs([
        {"source_url": "https://www.healthgrades.com/a", "rating": 4.0, "review_count": 12},
        {"source_url": "https://www.healthgrades.com/b", "rating": 4.2, "review_count": 31},
        {"source_url": "https://www.vitals.com/x", "rating": 4.4, "review_count": 20},
    ])

    assert len(pairs) == 2                              # one voice per domain
    healthgrades = next(p for p in pairs if "healthgrades" in p["source_url"])
    assert healthgrades["review_count"] == 31           # largest stated count wins

def test_platform_pairs_prefer_profile_over_listing_at_equal_count():
    """At equal review counts, the provider's own profile page wins over a
    directory listing — so the pair's recorded URL is the attributable link."""
    from agents.data_gatherer import _platform_rating_pairs

    pairs = _platform_rating_pairs([
        {"source_url": "https://doctor.webmd.com/providers/specialty/neurology/arizona/chandler",
         "rating": 5.0, "review_count": 42},
        {"source_url": "https://doctor.webmd.com/doctor/nicole-simpkins-123",
         "rating": 5.0, "review_count": 42},
    ])

    assert len(pairs) == 1
    assert "/doctor/" in pairs[0]["source_url"]          # profile beat the listing

def test_listing_only_pairs_are_recorded_as_unbacked(data_gatherer: DataGathererAgent):
    """Simpkins case: two platform pairs, but BOTH on directory listing pages.

    `pair_count` is 2, so every coverage measure keyed on pair count calls this
    provider well-sourced — while neither number is attributable to him. A
    "Best Neurologists in Chandler" index states figures for many doctors.

    Round 3 caught this with `_has_profile_source`, which the 2026-07-25
    enrichment-uniformity phase deleted alongside the tier predicates it sat
    among. It was not one of them — the tiers rationed WHO got enriched, this
    asked whether what came back was attributable — and its test survived the
    deletion asserting only that enrichment fires, which uniform enrichment
    makes true for every provider unconditionally. It passed vacuously for
    three rounds. Round 9 wrote that down in a doc nobody reads.

    Re-pointed at the outcome instead of the trigger: fires for any provider, so
    the assertion has to be about what the pass produced.
    """
    provider = {
        "name": "Dr. Simpkins", "rating": 5.0, "review_count": 42,
        "review_summary": "Praised.", "review_sentiment": "positive",
        "review_observations": [
            {"source_url": "https://doctor.webmd.com/providers/specialty/neurology/arizona/chandler",
             "rating": 5.0, "review_count": 42},
            {"source_url": "https://www.vitals.com/local/neurologist/az/chandler",
             "rating": 4.5, "review_count": 40},
        ],
    }

    data_gatherer._rederive_from_observations(provider)

    assert provider["platform_pair_count"] == 2      # two platforms agree...
    assert provider["profile_backed_platforms"] == 0  # ...and neither is his page


def test_profile_backed_pairs_counts_only_confirmed_profiles(
    data_gatherer: DataGathererAgent,
):
    """A profile URL counts, a directory index does not, and an UNRECOGNISED
    shape does not either.

    `unknown` is deliberately asymmetric with `_page_rank`, which ranks it ABOVE
    a confirmed listing so an unrecognised-but-real profile can win a tie-break.
    This measure answers "did we demonstrably reach a real profile", and an
    unidentified URL cannot demonstrate it — counting it would report coverage
    we cannot show.
    """
    provider = {
        "name": "Dr. Mixed",
        "review_observations": [
            # confirmed profile
            {"source_url": "https://www.healthgrades.com/physician/dr-a-b-12345",
             "rating": 4.6, "review_count": 23},
            # confirmed directory index
            {"source_url": "https://www.vitals.com/search?q=neurology",
             "rating": 4.0, "review_count": 30},
            # review platform, shape we have no pattern for
            {"source_url": "https://www.ratemds.com/some-new-layout/dr-a-b/",
             "rating": 5.0, "review_count": 8},
        ],
    }

    data_gatherer._rederive_from_observations(provider)

    assert provider["platform_pair_count"] == 3
    assert provider["profile_backed_platforms"] == 1


def test_warm_cache_reports_the_same_coverage_as_cold(data_gatherer: DataGathererAgent):
    """Warm must reproduce cold exactly — the cache's acceptance bar.

    These counts derive from `review_observations`, which IS cached, so
    deriving them on only the cold path would make a hit report different
    coverage than the run that populated it. That is the shape of the bug where
    a hit restored the observations but nothing re-derived the blend, and the
    provider scored as though it had no platform evidence at all.
    """
    observations = [
        {"source_url": "https://www.healthgrades.com/physician/dr-c-d-99999",
         "rating": 4.2, "review_count": 40},
        {"source_url": "https://www.vitals.com/doctors/dr-c-d",
         "rating": 4.4, "review_count": 12},
    ]

    cold = {"name": "Dr. Warm"}
    data_gatherer._merge_review_data(cold, {
        "review_summary": "Detailed feedback.",
        "review_sentiment": "positive",
        "review_observations": [dict(o) for o in observations],
    })

    warm = {"name": "Dr. Warm", "review_observations": [dict(o) for o in observations]}
    data_gatherer._rederive_from_observations(warm)

    assert warm["platform_pair_count"] == cold["platform_pair_count"] == 2
    assert warm["profile_backed_platforms"] == cold["profile_backed_platforms"] == 2

class TestLocationEvidenceArtifact:
    """The city-centroid artifact: a provider located only to city precision,
    in the user's city, must NOT get a fake ~0-mile bullseye that out-scores a
    provider with a real ZIP a few miles out (more data scoring worse)."""

    def test_city_only_same_city_nulls_fake_zero(self, data_gatherer: DataGathererAgent):
        provider = {"name": "X", "location": "Chandler, AZ"}
        data_gatherer._attach_location_evidence(provider, "Chandler, AZ")
        assert provider["computed_distance_miles"] is None   # artifact nulled
        assert provider["location_match"] == "same_city"     # honest tier fallback

    def test_zip_precise_provider_keeps_distance(self, data_gatherer: DataGathererAgent):
        provider = {"name": "Y", "location": "Chandler, AZ 85224"}
        data_gatherer._attach_location_evidence(provider, "Chandler, AZ")
        assert isinstance(provider["computed_distance_miles"], (int, float))

    def test_intercity_centroid_distance_kept(self, data_gatherer: DataGathererAgent):
        provider = {"name": "Z", "location": "Tucson, AZ"}
        data_gatherer._attach_location_evidence(provider, "Phoenix, AZ")
        assert provider["computed_distance_miles"] and provider["computed_distance_miles"] > 50

def test_enrichment_backfills_address_and_never_clobbers(data_gatherer: DataGathererAgent):
    """A ZIP-resolvable address upgrades a city-only location; a location we
    can already place to a ZIP is never overwritten."""
    vague = {"name": "Dr. Vague", "location": "Chandler, AZ"}
    data_gatherer._merge_review_data(vague, {
        "review_summary": "No reviews available", "review_sentiment": "unknown",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [], "address": "1234 W Frye Rd, Chandler, AZ 85224",
    })
    assert "85224" in vague["location"]                  # gained ZIP precision

    precise = {"name": "Dr. Precise", "location": "Chandler, AZ 85224"}
    data_gatherer._merge_review_data(precise, {
        "review_summary": "No reviews available", "review_sentiment": "unknown",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [], "address": "9 E Main St, Chandler, AZ 85225",
    })
    assert precise["location"] == "Chandler, AZ 85224"   # not clobbered

def test_enrichment_backfills_phone(data_gatherer: DataGathererAgent):
    """Platform profiles show the office phone prominently; directory-sourced
    candidates arrive without one. Backfill when missing, never clobber,
    and the model's 'N/A' habit never lands on a card."""
    missing = {"name": "Dr. Quiet", "phone": ""}
    data_gatherer._merge_review_data(missing, {
        "review_summary": "No reviews available", "review_sentiment": "unknown",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [], "phone": "480-722-0239",
    })
    assert missing["phone"] == "480-722-0239"

    kept = {"name": "Dr. Kept", "phone": "480-111-2222"}
    data_gatherer._merge_review_data(kept, {
        "review_summary": "No reviews available", "review_sentiment": "unknown",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [], "phone": "480-999-8888",
    })
    assert kept["phone"] == "480-111-2222"

    na = {"name": "Dr. NA"}
    data_gatherer._merge_review_data(na, {
        "review_summary": "No reviews available", "review_sentiment": "unknown",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [], "phone": "N/A",
    })
    assert not na.get("phone")

def test_merge_unions_observations_across_passes(data_gatherer: DataGathererAgent):
    """Second-opinion enrichment ADDS pairs next to what the candidate pass
    found; a name-query result set that doesn't re-include the original
    healthgrades page must not erase its numbers. The blend then hears all
    platforms and the headline ladder re-picks over the union."""
    provider = {
        "name": "Dr. Hodgson", "rating": 4.6, "review_count": 33,
        "review_summary": "Praised.", "review_sentiment": "positive",
        "review_source_url": "https://www.healthgrades.com/physician/dr-hodgson",
        "review_observations": [
            {"source_url": "https://www.healthgrades.com/physician/dr-hodgson",
             "rating": 4.6, "review_count": 33},
        ],
    }
    data_gatherer._merge_review_data(provider, {
        "review_summary": "Consistently praised across platforms.",
        "review_sentiment": "positive",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [
            {"source_url": "https://doctor.webmd.com/doctor/hodgson", "rating": 4.2, "review_count": 25},
            {"source_url": "https://www.vitals.com/doctors/hodgson", "rating": 4.4, "review_count": 40},
            # Same URL as the existing observation, different numbers: the
            # first-seen observation wins (stability across re-reads)
            {"source_url": "https://www.healthgrades.com/physician/dr-hodgson",
             "rating": 4.0, "review_count": 10},
        ],
    })
    assert len(provider["review_observations"]) == 3
    hg = next(o for o in provider["review_observations"]
              if "healthgrades" in o["source_url"])
    assert hg["rating"] == 4.6 and hg["review_count"] == 33
    # (4.6*33 + 4.2*25 + 4.4*40) / 98 = 4.416 -> 4.4 over all three platforms
    assert provider["blended_rating"] == 4.4
    assert provider["blended_review_count"] == 98
    assert provider["blended_platform_count"] == 3
    # Headline re-picked over the union: vitals is now the largest credible pair
    assert provider["rating"] == 4.4
    assert provider["review_count"] == 40
    assert "vitals.com" in provider["review_source_url"]

def test_stale_blend_cleared_when_disagreement_grows(data_gatherer: DataGathererAgent):
    """A pair added by enrichment can push the set past the disagreement
    span; the blend computed before that pair arrived is stale and must not
    keep feeding the score."""
    provider = {
        "name": "Dr. Split",
        "review_observations": [
            {"source_url": "https://www.healthgrades.com/x", "rating": 4.6, "review_count": 33},
            {"source_url": "https://doctor.webmd.com/x", "rating": 4.2, "review_count": 25},
        ],
        "blended_rating": 4.4, "blended_review_count": 58, "blended_platform_count": 2,
    }
    data_gatherer._merge_review_data(provider, {
        "review_summary": "Polarized.", "review_sentiment": "mixed",
        "review_count": None, "rating": None, "review_source_url": None,
        "review_observations": [
            {"source_url": "https://www.vitals.com/x", "rating": 1.0, "review_count": 50},
        ],
    })
    assert len(provider["review_observations"]) == 3
    assert "blended_rating" not in provider
    assert "blended_review_count" not in provider
    assert "blended_platform_count" not in provider

def test_domain_anchor_hints_extend_excerpt_anchors():
    """Platform URLs aim the excerpt window at score-feeding sections
    (years -> experience, insurance -> payer evidence); unknown domains get
    the base anchors untouched."""
    from agents.data_gatherer import _anchors_for

    base = ["neurology", "review"]
    hg = _anchors_for("https://www.healthgrades.com/physician/x", base)
    assert hg[:2] == base
    assert "years of experience" in hg and "insurance accepted" in hg

    zoc = _anchors_for("https://www.zocdoc.com/doctor/x", base)
    assert "in-network" in zoc

    other = _anchors_for("https://www.example.com/page", base)
    assert other == base
    assert base == ["neurology", "review"]    # input list never mutated


class TestEnrichmentRecall:
    """Round-4 recall work: a doctor with real reviews on three platforms came
    back single-sourced. Four independent defects had to line up, so each gets
    its own guard here."""

    # ---- the dead predicate: observations never left the discovery pass ----

    def test_discovery_pass_emits_observations(self, data_gatherer: DataGathererAgent):
        """review_observations used to be written ONLY by enrichment, so the
        tier-1 predicate read zero pairs for every provider and 'needs a second
        opinion' was unconditionally true — budget spent by rank, not by need."""
        mock_response = MagicMock()
        mock_response.content[0].text = json.dumps([{
            "name": "Dr. Discovered", "specialty": "Neurology",
            "location": "Chandler, AZ", "rating": 4.6, "review_count": 33,
            "review_summary": "Praised widely.", "review_sentiment": "positive",
            "review_source_url": "https://www.healthgrades.com/physician/dr-discovered",
            "review_observations": [
                {"source_url": "https://www.healthgrades.com/physician/dr-discovered",
                 "rating": 4.6, "review_count": 33},
                {"source_url": "https://www.vitals.com/doctors/Dr_Discovered.html",
                 "rating": 4.4, "review_count": 21},
            ],
        }])
        data_gatherer.anthropic_client.messages.create.return_value = mock_response

        providers = data_gatherer._extract_provider_data(
            MOCK_TAVILY_SEARCH_RESPONSE["results"], "Neurology", "Chandler, AZ"
        )

        observations = providers[0]["review_observations"]
        assert len(observations) == 2
        assert {o["source_url"] for o in observations} == {
            "https://www.healthgrades.com/physician/dr-discovered",
            "https://www.vitals.com/doctors/Dr_Discovered.html",
        }

    def test_discovery_prompt_asks_for_observations(self, data_gatherer: DataGathererAgent):
        mock_response = MagicMock()
        mock_response.content[0].text = "[]"
        data_gatherer.anthropic_client.messages.create.return_value = mock_response

        data_gatherer._extract_provider_data(
            MOCK_TAVILY_SEARCH_RESPONSE["results"], "Neurology", "Chandler, AZ"
        )

        prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "review_observations" in prompt
        assert "transcribe stated values only" in prompt

    def test_discovery_still_emits_observations_and_everyone_is_enriched(
        self, data_gatherer: DataGathererAgent
    ):
        """Round 4's end-to-end guard, kept: extraction is run for real rather
        than hand-building the observations production never produced.

        The assertion inverted in Phase 2. This used to prove a well-covered
        provider was SKIPPED by the tiered budget; those tiers are gone, because
        at real pool sizes they rationed nothing while rank 6 still arrived
        without a blended rating — enrichment had run and found nothing usable.
        Coverage no longer buys an exemption; what a provider gets instead is a
        recorded outcome."""
        mock_response = MagicMock()
        mock_response.content[0].text = json.dumps([{
            "name": "Dr. Covered", "specialty": "Neurology",
            "location": "Chandler, AZ", "rating": 4.6, "review_count": 33,
            "review_summary": "Praised widely.", "review_sentiment": "positive",
            "review_observations": [
                {"source_url": "https://www.healthgrades.com/physician/dr-covered",
                 "rating": 4.6, "review_count": 33},
                {"source_url": "https://www.vitals.com/doctors/Dr_Covered.html",
                 "rating": 4.4, "review_count": 21},
            ],
        }])
        data_gatherer.anthropic_client.messages.create.return_value = mock_response
        providers = data_gatherer._extract_provider_data(
            MOCK_TAVILY_SEARCH_RESPONSE["results"], "Neurology", "Chandler, AZ"
        )

        enriched = []
        with patch.object(data_gatherer, "_enrich_one",
                          side_effect=lambda p, *a, **k: enriched.append(p["name"])):
            data_gatherer._enrich_missing_reviews(providers, "Chandler, AZ", "Neurology")

        # Round-4 guarantee, still load-bearing: the CANDIDATE pass emits
        # review_observations. Until it did, every provider read zero pairs.
        assert providers[0]["review_observations"], "discovery must emit observations"
        assert len(providers[0]["review_observations"]) == 2

        # Phase-2 contract: coverage no longer exempts anyone.
        assert enriched == ["Dr. Covered"]

    def test_single_pair_provider_from_discovery_still_enriched(
        self, data_gatherer: DataGathererAgent
    ):
        mock_response = MagicMock()
        mock_response.content[0].text = json.dumps([{
            "name": "Dr. Thin", "specialty": "Neurology",
            "location": "Chandler, AZ", "rating": 4.0, "review_count": 28,
            "review_summary": "Kind and thorough.", "review_sentiment": "positive",
            "review_observations": [
                {"source_url": "https://www.vitals.com/doctors/Dr_Thin.html",
                 "rating": 4.0, "review_count": 28},
            ],
        }])
        data_gatherer.anthropic_client.messages.create.return_value = mock_response
        providers = data_gatherer._extract_provider_data(
            MOCK_TAVILY_SEARCH_RESPONSE["results"], "Neurology", "Chandler, AZ"
        )

        enriched = []
        with patch.object(data_gatherer, "_enrich_one",
                          side_effect=lambda p, *a, **k: enriched.append(p["name"])):
            data_gatherer._enrich_missing_reviews(providers, "Chandler, AZ", "Neurology")

        assert enriched == ["Dr. Thin"]     # one platform is not a second opinion

    # ---- block selection: one slot per platform before any domain repeats ----

    def test_every_platform_gets_a_block_before_any_repeats(
        self, data_gatherer: DataGathererAgent
    ):
        """A chatty domain used to take every block: four healthgrades pages
        crowded out the vitals and webmd profiles that ARE the second opinion."""
        results = [
            {"title": f"HG {i}", "url": f"https://www.healthgrades.com/physician/p{i}",
             "content": "c", "score": 0.9} for i in range(4)
        ] + [
            {"title": "V", "url": "https://www.vitals.com/doctors/Dr_X.html",
             "content": "c", "score": 0.5},
            {"title": "W", "url": "https://doctor.webmd.com/doctor/dr-x",
             "content": "c", "score": 0.4},
        ]
        mock_response = MagicMock()
        mock_response.content[0].text = (
            '{"review_summary": "No reviews available", "review_sentiment": "unknown",'
            ' "review_count": null, "rating": null}'
        )
        data_gatherer.anthropic_client.messages.create.return_value = mock_response

        data_gatherer._extract_review_data_only(results, "Dr. X", "Neurology", "Chandler, AZ")

        prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "vitals.com/doctors/Dr_X.html" in prompt
        assert "doctor.webmd.com/doctor/dr-x" in prompt

    # ---- identity guard: the backstop for a specialty-free query ----

    def test_observation_from_a_different_person_is_rejected(
        self, data_gatherer: DataGathererAgent
    ):
        provider = {"name": "Dr. Mohammad B. Khan"}
        data_gatherer._merge_review_data(provider, {
            "review_summary": "No reviews available", "review_sentiment": "unknown",
            "review_observations": [
                {"source_url": "https://www.vitals.com/doctors/Dr_Whitfield.html",
                 "page_provider_name": "Dr. Sarah Whitfield",
                 "rating": 4.9, "review_count": 120},
            ],
        })
        assert not provider.get("review_observations")

    def test_adjacent_specialty_label_same_person_is_kept(
        self, data_gatherer: DataGathererAgent
    ):
        """The Khan regression: healthgrades files him under Sleep Medicine
        while the search said Neurology. Same person — must survive."""
        provider = {"name": "Dr. Mohammad B. Khan"}
        data_gatherer._merge_review_data(provider, {
            "review_summary": "No reviews available", "review_sentiment": "unknown",
            "review_observations": [
                {"source_url": "https://www.healthgrades.com/physician/dr-mohammad-khan",
                 "page_provider_name": "Mohammad Khan, MD",
                 "rating": 4.0, "review_count": 31},
            ],
        })
        assert len(provider["review_observations"]) == 1
        assert provider["review_observations"][0]["review_count"] == 31

    def test_missing_page_name_degrades_to_previous_behavior(
        self, data_gatherer: DataGathererAgent
    ):
        """A model that skips the field must not cost us every observation."""
        provider = {"name": "Dr. Mohammad B. Khan"}
        data_gatherer._merge_review_data(provider, {
            "review_summary": "No reviews available", "review_sentiment": "unknown",
            "review_observations": [
                {"source_url": "https://www.vitals.com/doctors/Dr_Khan.html",
                 "rating": 4.0, "review_count": 28},
            ],
        })
        assert len(provider["review_observations"]) == 1

    def test_guard_never_drops_observations_already_held(
        self, data_gatherer: DataGathererAgent
    ):
        """Discovery-pass observations predate the guard and are not re-judged."""
        provider = {
            "name": "Dr. Mohammad B. Khan",
            "review_observations": [
                {"source_url": "https://www.vitals.com/doctors/Dr_Khan.html",
                 "rating": 4.0, "review_count": 28},
            ],
        }
        data_gatherer._merge_review_data(provider, {
            "review_summary": "No reviews available", "review_sentiment": "unknown",
            "review_observations": [
                {"source_url": "https://www.example.com/other",
                 "page_provider_name": "Someone Entirely Different",
                 "rating": 1.0, "review_count": 99},
            ],
        })
        urls = {o["source_url"] for o in provider["review_observations"]}
        assert urls == {"https://www.vitals.com/doctors/Dr_Khan.html"}

    # ---- prose-stated counts still make a pair ----

    def test_count_written_as_prose_still_counts(self, data_gatherer: DataGathererAgent):
        """int("31 reviews") raised and the count was silently dropped, which
        demoted the observation to rating-only — no pair, no blend."""
        provider = {"name": "Dr. Prose"}
        data_gatherer._merge_review_data(provider, {
            "review_summary": "No reviews available", "review_sentiment": "unknown",
            "review_observations": [
                {"source_url": "https://www.healthgrades.com/physician/dr-prose",
                 "rating": 4.0, "review_count": "31 reviews"},
                {"source_url": "https://www.vitals.com/doctors/Dr_Prose.html",
                 "rating": 4.2, "review_count": "(28)"},
            ],
        })
        counts = sorted(o["review_count"] for o in provider["review_observations"])
        assert counts == [28, 31]
        assert provider["blended_platform_count"] == 2      # a real second opinion

    def test_thousands_separator_survives_the_parse(self, data_gatherer: DataGathererAgent):
        """A plain \\d+ stops at the comma: "1,234 reviews" read as 1.

        The busiest, best-evidenced providers were the ones this hit, and it
        does not fail loudly — a count of 1 shrinks a 4.8 rating to a Bayesian
        3.62 instead of 4.79 and marks the confidence "low".
        """
        provider = {"name": "Dr. Popular"}
        data_gatherer._merge_review_data(provider, {
            "review_summary": "No reviews available", "review_sentiment": "unknown",
            "review_observations": [
                {"source_url": "https://www.healthgrades.com/physician/dr-popular",
                 "rating": 4.8, "review_count": "1,234 reviews"},
                {"source_url": "https://www.vitals.com/doctors/Dr_Popular.html",
                 "rating": 4.7, "review_count": "(12,000)"},
            ],
        })
        counts = sorted(o["review_count"] for o in provider["review_observations"])
        assert counts == [1234, 12000]
        assert provider["blended_review_count"] == 13234

    # ---- end-to-end: enrichment ADDS to what discovery found ----

    def test_enrichment_adds_a_second_platform_to_the_discovery_pair(
        self, data_gatherer: DataGathererAgent
    ):
        provider = {
            "name": "Dr. Mohammad B. Khan",
            "review_observations": [
                {"source_url": "https://www.vitals.com/doctors/Dr_Khan.html",
                 "rating": 4.0, "review_count": 28},
            ],
        }
        data_gatherer._merge_review_data(provider, {
            "review_summary": "Kind, thorough, listens carefully.",
            "review_sentiment": "positive",
            "review_observations": [
                {"source_url": "https://www.healthgrades.com/physician/dr-mohammad-khan",
                 "page_provider_name": "Dr. Mohammad B Khan, MD",
                 "rating": 4.0, "review_count": 31},
            ],
        })
        assert len(provider["review_observations"]) == 2
        assert provider["blended_platform_count"] == 2
        assert provider["blended_review_count"] == 59       # 28 + 31, both platforms


class TestReviewSummaryCarriesForward:
    """Observations UNION across passes but the narrative was clobbered by a
    pass that never saw the earlier one — so the paragraph described a single
    search's pages while the ratings above it spanned several platforms."""

    def _run(self, data_gatherer: DataGathererAgent, prior: str) -> str:
        mock_response = MagicMock()
        mock_response.content[0].text = (
            '{"review_summary": "No reviews available", "review_sentiment": "unknown",'
            ' "review_count": null, "rating": null}'
        )
        data_gatherer.anthropic_client.messages.create.return_value = mock_response
        data_gatherer._extract_review_data_only(
            [{"title": "t", "url": "https://www.vitals.com/doctors/x", "content": "c"}],
            "Dr. X", "Neurology", "Chandler, AZ", prior_summary=prior,
        )
        return data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]

    def test_prior_summary_reaches_the_prompt(self, data_gatherer: DataGathererAgent):
        prompt = self._run(data_gatherer, "Healthgrades patients praise his patience.")
        assert "PREVIOUSLY GATHERED PATIENT FEEDBACK" in prompt
        assert "Healthgrades patients praise his patience." in prompt
        assert "integrate it, do not discard it" in prompt

    def test_placeholder_prior_is_not_carried(self, data_gatherer: DataGathererAgent):
        prompt = self._run(data_gatherer, "No reviews available")
        assert "PREVIOUSLY GATHERED" not in prompt

    def test_absent_prior_is_not_carried(self, data_gatherer: DataGathererAgent):
        assert "PREVIOUSLY GATHERED" not in self._run(data_gatherer, "")

    def test_prompt_states_the_summary_replaces_the_earlier_one(
        self, data_gatherer: DataGathererAgent
    ):
        prompt = self._run(data_gatherer, "Earlier feedback.")
        assert "anything you leave out is lost" in prompt

    def test_enrich_one_passes_the_held_summary(self, data_gatherer: DataGathererAgent):
        provider = {"name": "Dr. Held", "location": "Chandler, AZ",
                    "review_summary": "Vitals reviewers call him thorough."}
        with patch.object(data_gatherer, "_search_providers",
                          return_value=[{"title": "t", "url": "u", "content": "c"}]), \
             patch.object(data_gatherer, "_extract_review_data_only",
                          return_value={}) as mock_extract, \
             patch.object(data_gatherer, "_merge_review_data"):
            data_gatherer._enrich_one(provider, "Chandler, AZ", "Neurology")

        assert mock_extract.call_args.kwargs["prior_summary"] == (
            "Vitals reviewers call him thorough."
        )


def test_a_bare_surname_does_not_absorb_a_full_name(data_gatherer: DataGathererAgent):
    """`_name_token_overlap` divides by the SMALLER token set, so every subset
    scored a perfect 1.0 and a nameless directory row swallowed a real
    physician. Reachable in normal operation: extraction rule 3 asks for thin
    entries by design and directory pages print "Dr. Kim" with no given name."""
    assert data_gatherer._name_token_overlap("Dr. Kim", "Dr. Jane Kim") == 0.0
    assert data_gatherer._name_token_overlap("Dr. Jane Kim", "Dr. Kim") == 0.0
    # Token sets are unordered, so a first name could collapse into a surname
    assert data_gatherer._name_token_overlap("Dr. Kim", "Kim Nguyen") == 0.0


def test_two_full_names_are_unaffected_by_the_guard(data_gatherer: DataGathererAgent):
    """The guard must not change the case dedupe exists for."""
    # Same person, different spellings — still merges
    assert data_gatherer._name_token_overlap(
        "Dr. Hussam Seif-Eddeine, MD", "Hussam Seif Eddeine"
    ) >= 0.8
    # Different people sharing a surname — still separate
    assert data_gatherer._name_token_overlap("Dr. David Kim", "Dr. Jane Kim") == 0.5
    assert data_gatherer._name_token_overlap("Andrea M An", "Andrea B An") < 0.8
    # Two bare surnames may still merge — nothing distinguishes them
    assert data_gatherer._name_token_overlap("Dr. Kim", "Kim") == 1.0


def test_profile_pages_cannot_evict_every_directory_page(data_gatherer: DataGathererAgent):
    """The worst case for hoisting: the domain-restricted discovery query fills
    the list with review-platform PROFILE pages, which name one physician each.
    Hoisting them ahead of everything else pushed the many-name directory pages
    past the block cut, leaving the pass that fills the candidate pool reading
    only one-name pages."""
    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_CLAUDE_EXTRACTION_RESPONSE
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    results = [
        {"title": f"Dr. Solo {i}", "url": f"https://www.healthgrades.com/physician/dr-{i}",
         "content": "one physician", "raw_content": "profile"}
        for i in range(_DISCOVERY_MAX_BLOCKS + 1)
    ] + [{"title": "Top 15 Neurologists in Chandler",
          "url": "https://directory.example.com/best-neurologists",
          "content": "fifteen physicians", "raw_content": "listicle"}]

    data_gatherer._extract_provider_data(results, "Neurology", "Chandler, AZ")
    # EVERY call's prompt, not `call_args` (the last one). Extraction now runs
    # in concurrent shards, so the last call holds only the tail of the page
    # list — reading it alone reported "every directory page was evicted" for a
    # directory page sitting in the first shard's prompt.
    prompts = [
        call.kwargs["messages"][0]["content"]
        for call in data_gatherer.anthropic_client.messages.create.call_args_list
    ]
    assert prompts, "no extraction call was made"

    carrying = [p for p in prompts if "directory.example.com" in p]
    assert carrying, "every many-name page was evicted by profile pages"
    # Priority is unchanged — review-heavy still leads, within the shard that
    # holds the directory page. Sharding is CONTIGUOUS precisely so this stays
    # true: the page list alternates (review-platform, everything-else) with
    # period 2, so dealing it round-robin would put every directory page in one
    # call and every profile page in the other.
    prompt = carrying[0]
    assert prompt.index("healthgrades.com") < prompt.index("directory.example.com")


class TestEnrichmentConcurrencyAndSources:
    """The enrichment stage's wall clock, and what it records about its own
    retrieval. Both come from the 2026-07-28 field run."""

    def test_worker_count_follows_the_configured_width(
        self, data_gatherer: DataGathererAgent
    ):
        """A hardcoded 4 workers ran the standard budget of 10 in THREE
        sequential waves — the last half empty and still costing a full wave of
        ~18s (Tavily search + Haiku extraction per provider). That is ~54s of a
        145s run, spent waiting rather than working."""
        candidates = [{"name": f"Dr. {i}"} for i in range(10)]
        data_gatherer.config.ENRICHMENT_MAX_WORKERS = 8

        seen = {}
        real_pool = ThreadPoolExecutor

        def spy(*args, **kwargs):
            seen["max_workers"] = kwargs.get("max_workers")
            return real_pool(*args, **kwargs)

        with patch.object(data_gatherer, "_enrich_one", lambda *a, **k: None), \
             patch("agents.data_gatherer.ThreadPoolExecutor", side_effect=spy):
            data_gatherer._enrich_missing_reviews(candidates, "Chandler, AZ", "Neurology")

        assert seen["max_workers"] == 8, "must fan out to the configured width"

    def test_width_never_exceeds_the_number_of_providers(
        self, data_gatherer: DataGathererAgent
    ):
        """Seven providers must ask for seven workers, not the configured eight.

        The THREAD-COUNT outcome does not depend on this: `ThreadPoolExecutor`
        spawns lazily, so `max_workers=8` over 7 tasks peaks at 7 threads
        anyway (measured). What depends on it is the number we REPORT — the
        wave-count log line prints `workers`, so dropping the `min()` would
        claim eight workers for seven that exist. Same doctrine as the rubric's
        absent-notes: a line may only assert what the code observed.

        An earlier version of this test was deleted during round 14's
        revert-in-isolation pass as guarding a tautology of `min()`. That was
        wrong — `min()` is only a tautology while it is written, and
        `workers = self.config.ENRICHMENT_MAX_WORKERS` is a plausible
        simplification for someone who reads the clamp as redundant.
        """
        data_gatherer.config.ENRICHMENT_MAX_WORKERS = 8
        seen = {}
        real_pool = ThreadPoolExecutor

        def spy(*args, **kwargs):
            seen["max_workers"] = kwargs.get("max_workers")
            return real_pool(*args, **kwargs)

        with patch.object(data_gatherer, "_enrich_one", lambda *a, **k: None), \
             patch("agents.data_gatherer.ThreadPoolExecutor", side_effect=spy):
            data_gatherer._enrich_missing_reviews(
                [{"name": f"Dr. {i}"} for i in range(7)], "Chandler, AZ", "Neurology"
            )

        assert seen["max_workers"] == 7

    def test_a_zero_width_setting_does_not_take_the_search_down(
        self, data_gatherer: DataGathererAgent
    ):
        """`ENRICHMENT_MAX_WORKERS=0` in a .env — or a blank one read as 0 —
        must not end the search with a raw traceback.

        `ThreadPoolExecutor(max_workers=0)` raises ValueError, and the wave-count
        log line beside it divides by the same number. This is the failure mode
        already recorded for ten other unguarded env integers
        (`DEFAULT_SEARCH_RADIUS=0` divides by zero); a new knob should not add an
        eleventh. The `min()` half of the same expression is guarded above."""
        data_gatherer.config.ENRICHMENT_MAX_WORKERS = 0
        seen = {}
        real_pool = ThreadPoolExecutor

        def spy(*args, **kwargs):
            seen["max_workers"] = kwargs.get("max_workers")
            return real_pool(*args, **kwargs)

        with patch.object(data_gatherer, "_enrich_one", lambda *a, **k: None), \
             patch("agents.data_gatherer.ThreadPoolExecutor", side_effect=spy):
            data_gatherer._enrich_missing_reviews(
                [{"name": "A"}, {"name": "B"}], "Chandler, AZ", ""
            )

        assert seen["max_workers"] == 1, "a zero width must floor to serial, not crash"

    def test_enrichment_records_the_pages_its_search_reached(
        self, data_gatherer: DataGathererAgent
    ):
        """Three different failures produce the same finished card: the
        platform's profile was never returned, it was returned but yielded no
        observation, or it yielded one that lost the same-domain collapse to a
        directory listing.

        On 2026-07-28 two providers carded "healthgrades.com — listing page" as
        their best single source and nothing in the run said which. Recording
        what the search actually reached is what separates them."""
        provider = {"name": "Dr. De Lima", "location": "Chandler, AZ"}
        results = [
            {"url": "https://www.healthgrades.com/usearch?what=Neurology"},
            {"url": "https://www.vitals.com/doctors/dr-de-lima"},
        ]

        with patch.object(data_gatherer, "_search_providers", return_value=results), \
             patch.object(data_gatherer, "_extract_review_data_only", return_value={}), \
             patch.object(data_gatherer, "_merge_review_data", lambda *a, **k: None), \
             patch.object(data_gatherer, "_classify_enrichment", return_value="enriched"):
            data_gatherer._enrich_one(provider, "Chandler, AZ", "Neurology")

        # `yielded` is None here because the patched extractor returned no
        # observations — "we fetched this page and nothing in this pass named
        # it", which is exactly the state the key exists to make visible.
        assert provider["enrichment_sources"] == [
            {"url": "https://www.healthgrades.com/usearch?what=Neurology",
             "kind": "listing", "raw_chars": 0, "yielded": None},
            {"url": "https://www.vitals.com/doctors/dr-de-lima",
             "kind": "profile", "raw_chars": 0, "yielded": None},
        ]

    def test_cache_hit_records_no_sources_rather_than_an_empty_list(
        self, data_gatherer: DataGathererAgent
    ):
        """A cache hit runs no search. An empty list would read as "we looked
        and found nothing" — the exact conflation `enrichment_outcome` exists to
        prevent."""
        cached = {"name": "Dr. Cached", "enrichment_outcome": "cached"}
        live = {"name": "Dr. Live"}

        with patch.object(data_gatherer, "_search_providers", return_value=[]):
            data_gatherer._enrich_missing_reviews([cached, live], "Chandler, AZ", "")

        assert "enrichment_sources" not in cached
        assert live["enrichment_sources"] == []
        assert live["enrichment_outcome"] == "no_profile_found"


class TestRingProvenance:
    """`ring_expanded` said the ring FIRED; nothing said what it bought.

    The decision it blocks is whether MIN_CANDIDATE_POOL should
    drop below the research budget so the ring stops firing on nearly every
    search. That turns on whether ring providers reach cards — which no field
    recorded, so the only way to guess was to read the card list and wonder
    which names looked out-of-town."""

    def test_a_provider_found_by_both_passes_counts_as_home(
        self, data_gatherer: DataGathererAgent
    ):
        """The one case where the survivor's own value is the wrong answer.

        `_dedupe_providers` picks the survivor by FIELD RICHNESS, not by which
        pass found them, so a fuller ring-sourced entry outlives the home entry
        for the same doctor — and the merged provider would then be counted as
        something the ring bought. It was discoverable without the ring; that is
        the entire question this field answers.

        Driven with the ring entry deliberately RICHER so it wins survivorship.
        """
        home = {"name": "Dr. Jane Kim", "discovery_source": "home"}
        ring = {
            "name": "Dr. Jane Kim, MD", "discovery_source": "ring",
            "location": "Gilbert, AZ 85234", "phone": "480-555-0100",
            "rating": 4.6, "review_count": 31, "years_experience": 12,
            "review_summary": "Thorough and unhurried.",
        }

        merged = data_gatherer._dedupe_providers([home, ring])

        assert len(merged) == 1
        assert merged[0]["discovery_source"] == "home", (
            "the ring did not buy this provider — the home pass already had them"
        )
        # ...and the richer entry still won on everything else
        assert merged[0]["review_count"] == 31

    def test_a_provider_only_the_ring_found_stays_ring(
        self, data_gatherer: DataGathererAgent
    ):
        """The counterpart: without this the field would be a constant."""
        home = {"name": "Dr. Jane Kim", "discovery_source": "home"}
        ring = {"name": "Dr. Omar Haddad", "discovery_source": "ring"}

        merged = data_gatherer._dedupe_providers([home, ring])
        sources = {p["name"]: p["discovery_source"] for p in merged}

        assert sources == {"Dr. Jane Kim": "home", "Dr. Omar Haddad": "ring"}

    def test_ring_added_counts_the_net_new_not_the_extracted(
        self, data_gatherer: DataGathererAgent
    ):
        """`ring_added` must survive dedupe. Counting what the ring EXTRACTED
        would report a purchase for every doctor both passes happened to find —
        overstating the ring's value precisely when it is least useful, i.e.
        when it re-finds the home city's own doctors in an adjacent town's
        directory pages."""
        home_pool = [{"name": "Dr. Jane Kim"}, {"name": "Dr. Ann Patel"}]
        ring_pool = [{"name": "Dr. Jane Kim, MD"}, {"name": "Dr. Omar Haddad"}]
        extractions = [home_pool, ring_pool]

        with patch.object(data_gatherer, "_discover_candidates",
                          return_value=[{"url": "https://example.com/a", "content": "x"}]), \
             patch.object(data_gatherer, "_extract_provider_data",
                          side_effect=lambda *a, **k: extractions.pop(0)), \
             patch.object(data_gatherer, "_attach_location_evidence", lambda *a, **k: None), \
             patch("agents.data_gatherer.nearby_cities", return_value=["Gilbert, AZ"]):
            data_gatherer.config.MIN_CANDIDATE_POOL = 10   # force the ring
            result = data_gatherer.gather_providers("Neurology", "Chandler, AZ", enrich=False)

        assert result["search_metadata"]["ring_expanded"] is True
        # Two ring extractions, but one was a doctor the home pass already had.
        assert result["search_metadata"]["ring_added"] == 1

        # The per-provider tag through the REAL call path, not just the count.
        # `test_a_provider_only_the_ring_found_stays_ring` drives
        # `_dedupe_providers` directly, so it stays green if the tagging is
        # deleted from `gather_providers` — the helper is guarded and the wiring
        # is not. Caught by revert-in-isolation.
        tags = {p["name"]: p.get("discovery_source") for p in result["providers"]}
        assert tags == {
            "Dr. Jane Kim": "home",       # both passes found her — not a purchase
            "Dr. Ann Patel": "home",
            "Dr. Omar Haddad": "ring",    # only the ring found him
        }

    def test_no_ring_means_every_provider_is_home(
        self, data_gatherer: DataGathererAgent
    ):
        """A pool that never rang out must report zero bought, not a missing
        key the UI would render as a blank."""
        with patch.object(data_gatherer, "_discover_candidates",
                          return_value=[{"url": "https://example.com/a", "content": "x"}]), \
             patch.object(data_gatherer, "_extract_provider_data",
                          return_value=[{"name": "Dr. Jane Kim"}]), \
             patch.object(data_gatherer, "_attach_location_evidence", lambda *a, **k: None):
            data_gatherer.config.MIN_CANDIDATE_POOL = 0    # never fires
            result = data_gatherer.gather_providers("Neurology", "Chandler, AZ", enrich=False)

        assert result["search_metadata"]["ring_expanded"] is False
        assert result["search_metadata"]["ring_added"] == 0
        assert result["providers"][0]["discovery_source"] == "home"


def test_the_shipped_enrichment_excerpt_config_reaches_a_rating_header(
    data_gatherer: DataGathererAgent,
):
    """The CONSTANTS as shipped, not the mechanism in isolation.

    `tests/unit/test_excerpt.py` proves a sized head reservation works; this
    proves the three values wired into `_extract_review_data_only` actually
    deliver it. Without this, `_ENRICHMENT_EXCERPT_WINDOWS` could go back to 3
    or `_ENRICHMENT_HEAD_CHARS` could be dropped from the call and every
    excerpt test would stay green — the helper guarded, the wiring not, which
    revert-in-isolation caught twice in round 14.

    Driven through the real extraction path so the block assembly, the
    platform round-robin and `include_head`'s per-domain gate all participate.

    WEBMD deliberately, not healthgrades. `_DOMAIN_ANCHOR_HINTS` gives
    healthgrades "years of experience", which a profile states one line below
    its rating — so a density window lands on the header by luck and the test
    passes with the reservation removed. webmd's hints are "conditions
    treated" / "procedures", nowhere near the header, and webmd is exactly
    where the 2026-07-28 runs disagreed ("4.5/5" with no count, then
    "4.5/5 (61 reviews)"). Caught by revert-in-isolation.
    """
    from tests.unit.test_excerpt import _HEADER_FACT, _profile_page

    mock_response = MagicMock()
    mock_response.content[0].text = '{"review_summary": "x", "review_sentiment": "positive"}'
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    results = [{
        "title": "Dr. Andrea An, MD - Neurology",
        # a REVIEW PLATFORM url — include_head is gated on that
        "url": "https://doctor.webmd.com/doctor/andrea-an-e085e811-overview",
        "content": "snippet",
        "raw_content": _profile_page(),
    }]
    data_gatherer._extract_review_data_only(results, "Dr. Andrea An, MD")

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert _HEADER_FACT in prompt, (
        "the shipped budget/windows/head_chars must put a profile's rating "
        "header in front of the extractor"
    )


def test_enrichment_sources_record_how_much_text_each_page_gave(
    data_gatherer: DataGathererAgent,
):
    """`build_excerpt` takes min(budget, available), so "Tavily returned a thin
    page" and "we under-read a full one" produce the same empty result. Two
    live runs of the same search disagreed about a provider's review count and
    nothing recorded which had happened."""
    provider = {"name": "Dr. Andrea An, MD", "location": "Chandler, AZ"}
    results = [
        {"url": "https://www.healthgrades.com/physician/dr-andrea-an-2pfjn",
         "raw_content": "x" * 18000},
        {"url": "https://www.vitals.com/doctors/andrea-an", "raw_content": "y" * 400},
    ]

    with patch.object(data_gatherer, "_search_providers", return_value=results), \
         patch.object(data_gatherer, "_extract_review_data_only", return_value={}), \
         patch.object(data_gatherer, "_merge_review_data", lambda *a, **k: None), \
         patch.object(data_gatherer, "_classify_enrichment", return_value="enriched"):
        data_gatherer._enrich_one(provider, "Chandler, AZ", "Neurology")

    assert [s["raw_chars"] for s in provider["enrichment_sources"]] == [18000, 400]


def test_each_platform_leads_with_its_profile_not_its_top_ranked_page(
    data_gatherer: DataGathererAgent,
):
    """The round-robin gives every platform one slot before any gets a second.
    "Best page" has to mean the doctor's PROFILE, not whatever Tavily ranked.

    On 2026-07-28 healthgrades returned a "best doctors for headache in
    chandler" DIRECTORY above `/physician/dr-andrea-an-2pfjn`, so the directory
    took healthgrades' lead slot and the profile was deferred. `_page_rank`
    existed and was consulted in two downstream tie-breaks, but never in
    selection — so the profile-over-listing preference only ever applied to
    whatever extraction had already produced.

    Driven with the listing FIRST, which is the order the live run returned.
    """
    mock_response = MagicMock()
    mock_response.content[0].text = '{"review_summary": "x", "review_sentiment": "positive"}'
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    results = [
        {"title": "Best doctors for headache in Chandler",
         "url": "https://www.healthgrades.com/find-a-doctor/arizona/best-doctors-for-headache-in-chandler",
         "content": "LISTING-MARKER", "raw_content": "directory of many doctors"},
        {"title": "Dr. Andrea An, MD",
         "url": "https://www.healthgrades.com/physician/dr-andrea-an-2pfjn",
         "content": "PROFILE-MARKER", "raw_content": "her own profile page"},
    ]
    data_gatherer._extract_review_data_only(results, "Dr. Andrea An, MD")

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert prompt.index("PROFILE-MARKER") < prompt.index("LISTING-MARKER"), (
        "the profile must lead its domain's slot, whatever Tavily ranked first"
    )


def test_relevance_still_orders_pages_of_the_same_kind(
    data_gatherer: DataGathererAgent,
):
    """The sort is by page KIND only. Within a kind, Tavily's order is still
    the best signal available and must survive — a sort that reshuffled equals
    would be throwing away relevance for nothing."""
    mock_response = MagicMock()
    mock_response.content[0].text = '{"review_summary": "x", "review_sentiment": "positive"}'
    data_gatherer.anthropic_client.messages.create.return_value = mock_response

    results = [
        {"title": "A", "url": "https://www.vitals.com/doctors/andrea-an-first",
         "content": "FIRST", "raw_content": "a"},
        {"title": "B", "url": "https://www.vitals.com/doctors/andrea-an-second",
         "content": "SECOND", "raw_content": "b"},
    ]
    data_gatherer._extract_review_data_only(results, "Dr. Andrea An, MD")

    prompt = data_gatherer.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert prompt.index("FIRST") < prompt.index("SECOND")
