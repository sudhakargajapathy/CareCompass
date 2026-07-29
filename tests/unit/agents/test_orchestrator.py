"""Unit tests for the ProviderMatchingOrchestrator."""

import time

import pytest
from unittest.mock import MagicMock, patch
from agents.orchestrator import ProviderMatchingOrchestrator, WorkflowState
from tests.fixtures.mock_agent_responses import (
    MOCK_GATHER_PROVIDERS_RESULT,
    MOCK_SCORED_PROVIDERS_RESULT,
    MOCK_VALIDATION_RESULT,
)

@pytest.fixture
def orchestrator():
    """Fixture to create a ProviderMatchingOrchestrator with mocked agents."""
    with patch('agents.orchestrator.DataGathererAgent') as mock_data_gatherer, \
         patch('agents.orchestrator.PreferenceScorerAgent') as mock_scorer, \
         patch('agents.orchestrator.CriticValidatorAgent') as mock_validator, \
         patch('agents.orchestrator.get_vector_store') as mock_vector_store:
        
        orchestrator = ProviderMatchingOrchestrator()
        
        # Configure mock agents
        orchestrator.data_gatherer.gather_providers.return_value = MOCK_GATHER_PROVIDERS_RESULT
        orchestrator.preference_scorer.score_providers.return_value = MOCK_SCORED_PROVIDERS_RESULT
        orchestrator.critic_validator.validate_rankings.return_value = MOCK_VALIDATION_RESULT
        
        yield orchestrator

def test_orchestrator_initialization(orchestrator: ProviderMatchingOrchestrator):
    """Test that the orchestrator and its workflow are initialized correctly."""
    assert orchestrator.workflow is not None
    assert orchestrator.data_gatherer is not None
    assert orchestrator.preference_scorer is not None
    assert orchestrator.critic_validator is not None

def test_execute_workflow_success(orchestrator: ProviderMatchingOrchestrator):
    """Test a full, successful execution of the workflow."""
    
    result = orchestrator.execute_workflow(
        specialty="Neurology",
        location="Phoenix, AZ"
    )
    
    assert result['success']
    assert len(result['final_recommendations']) > 0
    assert result['workflow_summary']['total_providers_found'] == 2
    
    # Check that agents were called
    orchestrator.data_gatherer.gather_providers.assert_called_once()
    orchestrator.preference_scorer.score_providers.assert_called_once()
    orchestrator.critic_validator.validate_rankings.assert_called_once()

def test_finalize_results_exposes_other_providers(orchestrator: ProviderMatchingOrchestrator):
    """Ranks 6+ land in workflow_summary['other_providers'] (refined order,
    compact shape) so the UI can show 'Other providers considered'."""
    ranked = [
        _recommendable(f"Dr. {i}", 90 - i, rating=4.0, specialty="Neurology")
        for i in range(7)
    ]
    out = _finalize(ranked, orchestrator)

    others = out["workflow_summary"]["other_providers"]
    assert len(out["final_recommendations"]) == 5        # top 5 shortlisted
    assert [o["rank"] for o in others] == [6, 7]         # ranks 6+ only
    assert others[0]["name"] == "Dr. 5"                  # 6th provider (0-indexed)

def test_execute_workflow_data_gathering_fails(orchestrator: ProviderMatchingOrchestrator):
    """Test workflow failure when the data gathering step fails."""
    
    # Simulate a failure in the data gatherer
    orchestrator.data_gatherer.gather_providers.return_value = {
        "status": "no_results",
        "providers": []
    }
    
    result = orchestrator.execute_workflow(
        specialty="Obscure Specialty",
        location="Remote Location"
    )
    
    assert not result['success']
    assert "data gathering" in result['error_messages'][0].lower()
    
    # Check that subsequent agents were not called
    orchestrator.preference_scorer.score_providers.assert_not_called()
    orchestrator.critic_validator.validate_rankings.assert_not_called()

def test_check_data_gathering_success(orchestrator: ProviderMatchingOrchestrator):
    """Test the conditional edge logic for data gathering."""
    
    # Success case
    state = {"gathered_data": {"status": "success"}}
    assert orchestrator._check_data_gathering_success(state) == "success"
    
    # Error case
    state = {"gathered_data": {"status": "error"}}
    assert orchestrator._check_data_gathering_success(state) == "error"

def test_score_node_enriches_top_core_candidates_before_judging(orchestrator: ProviderMatchingOrchestrator):
    """ONE cut, pinned before enrichment, honoured by enrichment AND the judge.

    The judge used to receive the whole pool while only the top slice had been
    researched, so it scored providers on an empty record — grading our coverage
    and reporting it as their quality."""

    providers = [{"name": f"P{i}"} for i in range(12)]
    state = {
        "gathered_data": {"providers": providers},
        "preferences": {"rating_weight": 0.5},
        "location": "Phoenix, AZ",
        "specialty": "Neurology",
        "current_step": "",
        "error_messages": [],
        "execution_log": [],
    }
    core_order = list(reversed(providers))
    budget = orchestrator.config.MAX_PROVIDERS_TO_ENRICH

    call_sequence = []
    orchestrator.preference_scorer.score_core.side_effect = \
        lambda p, prefs: (call_sequence.append("core"), core_order)[1]
    orchestrator.data_gatherer.enrich_providers.side_effect = \
        lambda *a, **k: call_sequence.append("enrich")
    orchestrator.preference_scorer.score_providers.side_effect = \
        lambda **k: (call_sequence.append("judge"), MOCK_SCORED_PROVIDERS_RESULT)[1]

    result_state = orchestrator._score_providers(state)

    assert call_sequence == ["core", "enrich", "judge"]
    args, kwargs = orchestrator.data_gatherer.enrich_providers.call_args
    assert args[0] == core_order[:budget]
    assert kwargs["location"] == "Phoenix, AZ"
    assert kwargs["specialty"] == "Neurology"
    assert result_state["scored_providers"]["status"] == "success"

    # The judge is handed the SAME cut, positionally.
    judge_kwargs = orchestrator.preference_scorer.score_providers.call_args.kwargs
    assert judge_kwargs["judge_count"] == budget
    assert judge_kwargs["providers"][:budget] == core_order[:budget]
    assert len(judge_kwargs["providers"]) == len(providers)

    # And the providers past it say so, rather than looking un-attempted.
    assert all(p.get("enrichment_outcome") == "over_budget" for p in core_order[budget:])
    assert not any(p.get("enrichment_outcome") == "over_budget" for p in core_order[:budget])


def test_enrichment_is_its_own_timeline_step(orchestrator: ProviderMatchingOrchestrator):
    """Review enrichment runs INSIDE the scoring node but is DataGathererAgent
    work — a Tavily search plus a Haiku extraction per provider.

    On 2026-07-28 the timeline read "Preference Scoring — 70.1s", the longest
    step in a 145s run, while the scorer's own share was ~15s of it: the other
    ~54s was enrichment. The banner TEXT had been corrected for this same
    confusion in an earlier round; the clock had not, so the panel still named
    the wrong agent as the bottleneck."""
    providers = [{"name": f"P{i}"} for i in range(3)]
    state = {
        "gathered_data": {"providers": providers},
        "preferences": {"rating_weight": 0.5},
        "location": "Phoenix, AZ",
        "specialty": "Neurology",
        "current_step": "",
        "error_messages": [],
        "execution_log": [],
    }
    orchestrator.preference_scorer.score_core.side_effect = lambda p, prefs: list(p)
    orchestrator.data_gatherer.enrich_providers.side_effect = lambda *a, **k: time.sleep(0.05)

    orchestrator._score_providers(state)

    steps = [(e["step"], e["status"]) for e in state["execution_log"]]
    assert ("enrich_reviews", "started") in steps
    assert ("enrich_reviews", "completed") in steps

    enrich_elapsed = next(
        e["details"]["elapsed_s"] for e in state["execution_log"]
        if e["step"] == "enrich_reviews" and e["status"] == "completed"
    )
    assert enrich_elapsed >= 0.05


def test_scoring_step_does_not_double_count_enrichment(
    orchestrator: ProviderMatchingOrchestrator,
):
    """The rows must sum to the run. Reporting enrichment on its own row while
    the scoring row still contains it would present 145s of work as ~200s and
    leave the scorer looking like the bottleneck anyway."""
    providers = [{"name": "P0"}]
    state = {
        "gathered_data": {"providers": providers},
        "preferences": {"rating_weight": 0.5},
        "location": "Phoenix, AZ",
        "specialty": "Neurology",
        "current_step": "",
        "error_messages": [],
        "execution_log": [],
    }
    orchestrator.preference_scorer.score_core.side_effect = lambda p, prefs: list(p)
    orchestrator.data_gatherer.enrich_providers.side_effect = lambda *a, **k: time.sleep(0.2)

    orchestrator._score_providers(state)

    by_step = {
        e["step"]: e["details"]["elapsed_s"]
        for e in state["execution_log"] if e["status"] == "completed"
    }
    assert by_step["enrich_reviews"] >= 0.2
    # The scoring row is what is LEFT after enrichment — its own work only.
    assert by_step["score_providers"] < 0.2


def test_critic_audits_exactly_the_judged_set(orchestrator: ProviderMatchingOrchestrator):
    """A provider the judge never saw has no rubric to check and no researched
    evidence to argue from — a verdict on them reviews our data gap, at Opus
    prices. Narrows the register's broader 'validates every ranked provider'."""

    judged = [{"name": "Judged1"}, {"name": "Judged2"}]
    skipped = [{"name": "Skipped", "ai_judged": False}]
    state = {
        "scored_providers": {"ranked_providers": judged + skipped},
        "preferences": {"rating_weight": 0.5},
        "current_step": "",
        "error_messages": [],
        "execution_log": [],
    }
    orchestrator.critic_validator.validate_rankings.return_value = {"status": "success"}

    orchestrator._validate_rankings(state)

    sent = orchestrator.critic_validator.validate_rankings.call_args.kwargs["ranked_providers"]
    assert sent == judged
    assert all(p.get("ai_judged") is not False for p in sent)

def test_payer_never_reaches_judge_or_critic(orchestrator: ProviderMatchingOrchestrator):
    """Insurance is verification-only (the sidebar FHIR check): even when the
    workflow is invoked with a payer, it stays in state/metadata and never
    rides into the preferences the judge or critic score against."""

    providers = [{"name": "P0"}]
    state = {
        "gathered_data": {"providers": providers},
        "preferences": {"rating_weight": 0.5},
        "insurance": "UnitedHealth",
        "location": "Chandler, AZ",
        "specialty": "Neurology",
        "current_step": "",
        "error_messages": [],
        "execution_log": [],
    }
    orchestrator.preference_scorer.score_core.return_value = providers
    orchestrator.preference_scorer.score_providers.return_value = MOCK_SCORED_PROVIDERS_RESULT

    state = orchestrator._score_providers(state)
    judge_prefs = orchestrator.preference_scorer.score_providers.call_args.kwargs["preferences"]
    assert "insurance" not in judge_prefs

    state["scored_providers"] = MOCK_SCORED_PROVIDERS_RESULT
    orchestrator.critic_validator.validate_rankings.return_value = {"status": "success"}
    orchestrator._validate_rankings(state)
    critic_prefs = orchestrator.critic_validator.validate_rankings.call_args.kwargs["preferences"]
    assert "insurance" not in critic_prefs


# ---- Round 12: the researched/unresearched boundary ----


def _recommendable(name: str, score: float, **overrides):
    """A provider for whom all three stages completed — data found, judge
    scored, critic reviewed. Anything less is not a recommendation.

    `critic_review` is NOT set here: `refine_rankings` writes it from the
    validation results on every pass, so a fixture value would be overwritten.
    `_finalize` supplies a matching verdict instead, which is what production
    does — and it means "the critic returned an entry for this provider" is
    tested through the real binding rather than asserted into place."""
    provider = {
        "name": name, "final_score": score, "refined_score": score,
        "enrichment_outcome": "enriched",
        "ai_rubric": {"review_substance": 44.0, "red_flags": 27.0, "practical_access": 12.0},
    }
    provider.update(overrides)
    return provider


def _pool(researched: int, unresearched: int):
    """Descending scores, with the UNRESEARCHED ones scoring HIGHEST — the
    inversion the live run produced, since only researched providers can be
    docked by the critic."""
    return [
        {"name": f"Unresearched {i}", "final_score": 90.0 - i,
         "refined_score": 90.0 - i, "enrichment_outcome": "over_budget"}
        for i in range(unresearched)
    ] + [
        _recommendable(f"Researched {i}", 70.0 - i) for i in range(researched)
    ]


def _finalize(providers, orchestrator=None, critiqued=None):
    """Run finalization. `critiqued` names the providers the critic returned an
    entry for; the default mirrors production, where the critic audits exactly
    the providers the judge scored."""
    orchestrator = orchestrator or ProviderMatchingOrchestrator()
    if critiqued is None:
        critiqued = [p["name"] for p in providers if p.get("ai_rubric")]
    validations = [
        {"rank": i + 1, "provider_name": name, "validation_status": "approved",
         "red_flags": [], "confidence_in_recommendation": "high"}
        for i, name in enumerate(critiqued)
    ]
    state = {
        "workflow_id": "wf-test",
        "gathered_data": {"providers": providers, "status": "success"},
        "scored_providers": {"ranked_providers": providers, "status": "success"},
        # Production shape: `refine_rankings` unwraps a nested
        # "validation_results" key when present, so putting the verdicts at the
        # top level alongside an empty one silently discards them.
        "validation_results": {
            "validation_results": {
                "top_provider_validation": {"top_provider_validations": validations},
            },
            "status": "success",
        },
        "execution_log": [],
        "error_messages": [],
        "preferences": {},
    }
    return orchestrator._finalize_results(state)


def test_an_unresearched_provider_never_reaches_the_shortlist(orchestrator: ProviderMatchingOrchestrator):
    """A provider we never looked at cannot be a recommendation.

    Their score is built ENTIRELY from imputations — the rating prior, the
    unknown-tenure constant, and a city centroid shared by everyone in the
    city — and it carries no critic penalty, because only a provider the critic
    saw can be docked. On 2026-07-27 that put four such providers at an
    identical 64 above three researched ones at 60/58/57, where the whole gap
    was the critic's -8. Here they score 90+, well above every researched one,
    and must still be excluded."""
    out = _finalize(_pool(researched=8, unresearched=4), orchestrator)

    shortlisted = [r["provider"]["name"] for r in out["final_recommendations"]]
    assert len(shortlisted) == 5
    assert all(n.startswith("Researched") for n in shortlisted), shortlisted


def test_the_shortlist_never_exceeds_the_researched_set(orchestrator: ProviderMatchingOrchestrator):
    """There is no "fill from unresearched" path, and none is reachable.

    `over_budget` is set only on core_ranked[budget:], so len(researched) is
    min(pool, budget) and unresearched is non-empty only when pool > budget.
    A short researched set therefore implies budget < 5, and a pool under 5 has
    nobody to fill FROM. Where it IS reachable — an operator setting the budget
    to 3 — padding would put providers we explicitly declined to research onto
    recommendation cards.

    An earlier version of this test constructed researched=3 with unresearched=6,
    a state the pipeline cannot produce."""
    out = _finalize(_pool(researched=3, unresearched=0), orchestrator)

    shortlisted = [r["provider"]["name"] for r in out["final_recommendations"]]
    assert shortlisted == ["Researched 0", "Researched 1", "Researched 2"]
    assert out["workflow_summary"]["other_providers"] == []


def test_a_sub_five_research_budget_shows_only_what_was_researched(
    orchestrator: ProviderMatchingOrchestrator,
):
    """The one configuration that can produce fewer than 5 researched providers
    alongside unresearched ones. Three researched means three cards; the other
    six stay in the expander's second group where their provisional scores are
    labelled as such."""
    out = _finalize(_pool(researched=3, unresearched=6), orchestrator)

    shortlisted = [r["provider"]["name"] for r in out["final_recommendations"]]
    assert len(shortlisted) == 3
    assert all(n.startswith("Researched") for n in shortlisted)
    others = out["workflow_summary"]["other_providers"]
    assert len(others) == 6
    assert all(o["researched"] is False for o in others)


def test_the_expander_carries_the_research_flag_and_researched_come_first(orchestrator: ProviderMatchingOrchestrator):
    out = _finalize(_pool(researched=8, unresearched=4), orchestrator)
    others = out["workflow_summary"]["other_providers"]

    flags = [o["researched"] for o in others]
    assert flags == sorted(flags, reverse=True), "researched entries must lead"
    assert sum(1 for f in flags if not f) == 4
    # Ranks continue from the shortlist without a gap or a repeat
    assert [o["rank"] for o in others] == list(range(6, 6 + len(others)))


def test_every_shortlisted_provider_is_absent_from_the_expander(orchestrator: ProviderMatchingOrchestrator):
    out = _finalize(_pool(researched=8, unresearched=4), orchestrator)
    others = out["workflow_summary"]["other_providers"]
    shortlisted = {r["provider"]["name"] for r in out["final_recommendations"]}
    assert shortlisted.isdisjoint({o["name"] for o in others})


def test_over_budget_and_unjudged_select_the_same_providers(
    orchestrator: ProviderMatchingOrchestrator,
):
    """The shortlist filter keys on `enrichment_outcome`, but the REASON a
    provider has no rubric and no critic verdict is `ai_judged is False`. Both
    derive from the same pinned budget, so they coincide — and nothing asserted
    it. If they ever diverge, a provider no model ever scored could reach the
    cards through the enrichment_outcome filter while carrying an empty rubric.

    Guarding the equality is cheaper than guarding every consequence of it."""
    providers = [
        {"name": f"P{i}", "base_score": 90.0 - i, "location": "Chandler, AZ"}
        for i in range(14)
    ]
    budget = orchestrator.config.MAX_PROVIDERS_TO_ENRICH

    core_ranked = providers[:]                 # already in core-score order
    for provider in core_ranked[budget:]:
        provider["enrichment_outcome"] = "over_budget"
    for provider in core_ranked[budget:]:
        provider["ai_judged"] = False

    over_budget = {p["name"] for p in providers
                   if p.get("enrichment_outcome") == "over_budget"}
    unjudged = {p["name"] for p in providers if p.get("ai_judged") is False}

    assert over_budget == unjudged
    assert len(over_budget) == len(providers) - budget


@pytest.mark.parametrize("outcome", ["no_profile_found", "identity_rejected", "failed"])
def test_a_provider_whose_details_we_could_not_find_is_not_recommended(
    orchestrator: ProviderMatchingOrchestrator, outcome,
):
    """Round 12 let these onto cards on the grounds that they were judged and
    critic-reviewed. They are still excluded now: a recommendation asserts we
    found this provider's details, and for these three we did not — the search
    came back empty, the pages we found named someone else, or the lookup
    errored. The card would render a blank reviews section under a match ring."""
    providers = [
        {"name": "No details", "final_score": 70.0, "refined_score": 70.0,
         "enrichment_outcome": outcome, "ai_rubric": {"review_substance": 24.0}},
        _recommendable("Fully researched", 55.0),
    ]
    out = _finalize(providers, orchestrator)

    shortlisted = [r["provider"]["name"] for r in out["final_recommendations"]]
    assert shortlisted == ["Fully researched"], "a lower-scoring but complete provider wins"
    others = out["workflow_summary"]["other_providers"]
    assert others[0]["withheld_reason"] == outcome
    assert others[0]["withheld_label"]


def test_our_own_stage_failing_withholds_the_provider_and_is_named_as_ours(
    orchestrator: ProviderMatchingOrchestrator,
):
    """A provider whose data we DID find, but whom our judge or critic never
    scored. Withheld — the card cannot show a rubric or a verdict — but the
    reason is a fault in OUR pipeline, and it is counted separately so the
    Responsible-AI panel can say so rather than implying the provider was
    unverifiable."""
    providers = [
        _recommendable("Judge dropped them", 90.0, ai_rubric={}),
        _recommendable("Critic dropped them", 89.0),
        _recommendable("Complete", 50.0),
    ]
    # The critic returned no entry for "Critic dropped them"
    out = _finalize(providers, orchestrator,
                    critiqued=["Judge dropped them", "Complete"])

    assert [r["provider"]["name"] for r in out["final_recommendations"]] == ["Complete"]

    withheld = out["workflow_summary"]["withheld"]
    assert withheld["pipeline_failures"] == 2
    assert sorted(withheld["pipeline_failure_names"]) == ["Critic dropped them", "Judge dropped them"]
    assert withheld["no_data"] == 0 and withheld["not_researched"] == 0


def test_ai_judged_is_not_the_judged_test(orchestrator: ProviderMatchingOrchestrator):
    """`ai_judged` is set False only on providers deferred past the budget. One
    that WAS sent to the judge but whose entry the judge omitted keeps it unset
    and receives ai_score 50.0 from a setdefault — with an EMPTY rubric. Gating
    on `ai_judged is not False` would put that blank card on the page."""
    dropped = _recommendable("Judge omitted them", 95.0, ai_rubric={}, ai_score=50.0)
    assert dropped.get("ai_judged") is not False        # the misleading signal

    out = _finalize([dropped, _recommendable("Complete", 40.0)], orchestrator)
    assert [r["provider"]["name"] for r in out["final_recommendations"]] == ["Complete"]


def test_the_panel_footer_claim_is_enforced_by_the_gate(
    orchestrator: ProviderMatchingOrchestrator,
):
    """The Responsible-AI footer states "every provider we recommend has been
    reviewed independently by a second AI". It previously said "every provider
    we RESEARCHED", which round 13 made false on its own page: `not_critiqued`
    counts a researched provider the critic never reviewed, and that count can
    render four lines above the footer. The claim now holds by construction —
    this test is what keeps it that way."""
    providers = [
        _recommendable("Reviewed", 60.0),
        _recommendable("Critic never returned", 99.0),
    ]
    out = _finalize(providers, orchestrator, critiqued=["Reviewed"])

    for recommendation in out["final_recommendations"]:
        assert recommendation["provider"].get("critic_review"), (
            f"{recommendation['provider']['name']} is recommended without an "
            f"independent review — the panel footer would be false"
        )
    assert [r["provider"]["name"] for r in out["final_recommendations"]] == ["Reviewed"]


def test_ring_contribution_counts_how_far_ring_providers_got(
    orchestrator: ProviderMatchingOrchestrator,
):
    """`ring_expanded` is a boolean — it says the ring FIRED, never what it
    bought, so the only way to judge whether the extra searches earn their cost
    was to read the card list and guess which names looked out-of-town.

    Candidates added is the wrong number on its own. The ring's real cost is
    that it FILLS the research budget, so every provider it adds also consumes
    an enrichment search, a slot in the judge prompt and an Opus verdict. What
    settles the threshold decision is how far those providers actually get: added but
    never shortlisted is spend for nothing, while routinely shortlisted means
    the ring is carrying the results and MIN_CANDIDATE_POOL should stay.

    Driven so the three counts are all DIFFERENT — equal counts would pass
    against a helper that returned the same number three times."""
    providers = (
        # ranks above the shortlist cut, so it reaches a card
        [_recommendable("Ring Top", 80.0, discovery_source="ring")]
        + [_recommendable(f"Home {i}", 70.0 - i, discovery_source="home")
           for i in range(6)]
        # researched but ranks below the top 5
        + [_recommendable("Ring Middle", 60.0, discovery_source="ring")]
        # never researched: past the budget, reached no model
        + [{"name": "Ring Deferred", "final_score": 50.0, "refined_score": 50.0,
            "enrichment_outcome": "over_budget", "discovery_source": "ring"}]
    )

    out = _finalize(providers, orchestrator)
    ring = out["workflow_summary"]["ring_contribution"]

    assert ring == {"added": 3, "researched": 2, "shortlisted": 1}


def test_ring_contribution_is_zero_rather_than_absent_on_a_home_only_run(
    orchestrator: ProviderMatchingOrchestrator,
):
    """A run that never rang out must report zeros, not a missing key the
    caption would render as a blank."""
    out = _finalize(
        [_recommendable(f"Home {i}", 70.0 - i, discovery_source="home") for i in range(6)],
        orchestrator,
    )

    assert out["workflow_summary"]["ring_contribution"] == {
        "added": 0, "researched": 0, "shortlisted": 0
    }
