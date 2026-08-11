"""Unit tests for the CriticValidatorAgent."""

import logging
import pytest
from unittest.mock import MagicMock, patch
import json
from agents.critic_validator import (
    _VALIDATION_MAX_TOKENS,
    _validation_token_budget,
    CriticValidatorAgent,
    refine_rankings,
)
from tests.fixtures.mock_agent_responses import (
    MOCK_RANKED_PROVIDERS,
    MOCK_BIAS_ANALYSIS_RESPONSE,
    MOCK_VALIDATION_RESPONSE
)

@pytest.fixture
def critic_validator():
    """Fixture to create a CriticValidatorAgent with a mocked Anthropic client."""
    with patch.object(CriticValidatorAgent, '_initialize_client', return_value=None):
        agent = CriticValidatorAgent()
        agent.anthropic_client = MagicMock()
        return agent

def test_analyze_ranking_bias_success(critic_validator: CriticValidatorAgent):
    """Test the _analyze_ranking_bias method for successful analysis."""
    
    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_BIAS_ANALYSIS_RESPONSE
    critic_validator.anthropic_client.messages.create.return_value = mock_response

    analysis = critic_validator._analyze_ranking_bias(MOCK_RANKED_PROVIDERS, {})

    assert "bias_assessment" in analysis
    assert analysis["bias_assessment"]["severity"] == "low"

def test_bias_prompt_grounded_in_mechanics_no_phantom_factors(critic_validator: CriticValidatorAgent):
    """Live, the bias analysis flagged 'insurance excluded by weight 0.0' —
    a false positive seeded by the sanitizer's legacy insurance_priority
    default — and worried a small-sample 5.0 outranks large review bases,
    blind to the Bayesian volume shrinkage. The prompt now states the real
    mechanics, the payload carries counts + blend volume, and no phantom
    ranking factor appears."""
    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_BIAS_ANALYSIS_RESPONSE
    critic_validator.anthropic_client.messages.create.return_value = mock_response

    providers = [{
        "name": "Dr. Blended", "rating": 5.0, "review_count": 23,
        "final_score": 88, "ai_reasoning": "Strong.",
        "blended_rating": 5.0, "blended_review_count": 66, "blended_platform_count": 3,
        "score_breakdown": {
            "rating": {"adjusted_rating": 4.32},
            "location": {"basis": "same_city", "value": "same_city"},
        },
    }]
    critic_validator._analyze_ranking_bias(providers, {"rating_weight": 0.45})

    prompt = critic_validator.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Bayesian-shrunk" in prompt
    assert "verification-only BY DESIGN" in prompt
    assert "Never flag its absence from the scoring weights" in prompt
    assert "insurance_priority" not in prompt          # phantom factor is gone
    assert '"notes"' not in prompt                     # ditto the empty notes field
    assert '"blended_platform_count": 3' in prompt     # evidence volume reaches the critic
    assert '"review_count": 23' in prompt
    # Round 3: the critic sees REAL location evidence + the shrunk rating, so
    # it can't invent "missing-distance leniency" from a phantom "N/A".
    assert '"location_evidence"' in prompt
    assert "same-city tier fallback" in prompt         # honest imputation label, not "N/A"
    assert '"adjusted_rating": 4.32' in prompt
    assert '"distance": "N/A"' not in prompt            # the phantom field is gone
    assert "adjusted_rating IS the star value actually scored" in prompt
    assert "already-penalized imputation" in prompt

def test_alternative_rankings_feature_removed(critic_validator: CriticValidatorAgent):
    """The alternative-ranking-perspectives feature was removed (field-test
    round 3: no user value, potential confusion). The generator method and the
    output key must both be gone."""
    assert not hasattr(critic_validator, "_generate_alternative_rankings")

def test_validate_top_recommendations_success(critic_validator: CriticValidatorAgent):
    """Test the _validate_top_recommendations method."""

    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_VALIDATION_RESPONSE
    critic_validator.anthropic_client.messages.create.return_value = mock_response

    validation = critic_validator._validate_top_recommendations(MOCK_RANKED_PROVIDERS)

    assert "top_provider_validations" in validation
    assert len(validation["top_provider_validations"]) == 1
    assert validation["top_provider_validations"][0]["validation_status"] == "approved"

def test_validation_covers_all_ranked_with_explicit_ranks(critic_validator: CriticValidatorAgent):
    """Deep validation audits EVERY ranked provider — a partial audit is a
    flat tax on the audited (live: uniform red flags on the validated top-8
    promoted the never-audited pre-#9 to #1). Ranks are explicit positional
    values (final_rank doesn't exist yet at validation time)."""
    providers = [
        {"name": f"Dr. Number {i}", "final_score": 90 - i, "rating": 4.0,
         "review_count": 10, "review_summary": "Fine.", "review_sentiment": "positive"}
        for i in range(1, 11)
    ]
    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_VALIDATION_RESPONSE
    critic_validator.anthropic_client.messages.create.return_value = mock_response

    critic_validator._validate_top_recommendations(providers)

    # Deep validation runs in concurrent shards, so "every provider is
    # audited" is now a claim about the UNION of the calls. Reading
    # `call_args` alone reports whichever shard happened to finish last —
    # which is how this test first failed after the split, with all ten
    # providers demonstrably sent.
    sent = []
    for call in critic_validator.anthropic_client.messages.create.call_args_list:
        prompt = call.kwargs["messages"][0]["content"]
        sent.extend(json.loads(
            prompt.split("TOP PROVIDERS TO VALIDATE:\n")[1].split("\n\nEach provider")[0]
        ))

    assert sorted(e["name"] for e in sent) == sorted(f"Dr. Number {i}" for i in range(1, 11))
    # Ranks are GLOBAL and survive sharding — `_generate_final_recommendations`
    # looks providers up by rank, so a shard renumbering 1..5 would attach the
    # wrong verdicts to the top three cards.
    assert sorted(e["rank"] for e in sent) == list(range(1, 11))
    # No provider may be audited twice: a duplicated record is a second verdict
    # for the same doctor, and `refine_rankings` claims each entry once.
    assert len(sent) == 10

def test_validation_prompt_carries_verdict_rubric(critic_validator: CriticValidatorAgent):
    """Unanchored labels produce rubber stamps — live, all 14 providers got
    'conditional · 2 flags · low confidence', a uniform ~-20 that cancels out
    and transmits zero ranking information. The rubric gives every output
    field entry criteria (evidence-cited rejects, banned generic caveats,
    approve-is-expected, missing-data-is-not-a-flag, evidence-anchored
    confidence) and the payload carries the blend fields the confidence
    bands key off."""
    providers = [
        {"name": "Dr. Blended", "final_score": 88, "rating": 4.6, "review_count": 33,
         "review_summary": "Praised.", "review_sentiment": "positive",
         "blended_rating": 4.4, "blended_review_count": 98, "blended_platform_count": 3,
         "review_observations": [
             {"source_url": "https://www.healthgrades.com/x", "rating": 4.6, "review_count": 33},
             {"source_url": "https://www.vitals.com/x", "rating": 4.2, "review_count": 65},
         ]},
        {"name": "Dr. Single", "final_score": 82, "rating": 5.0, "review_count": 14,
         "review_summary": "Glowing.", "review_sentiment": "positive"},
    ]
    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_VALIDATION_RESPONSE
    critic_validator.anthropic_client.messages.create.return_value = mock_response

    critic_validator._validate_top_recommendations(providers)

    prompt = critic_validator.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    # Rubric anchors present
    assert "VERDICT RUBRIC" in prompt
    assert "DIFFERENTIATION CHECK" in prompt
    assert "Missing data is NEVER a red flag" in prompt
    assert 'the EXPECTED verdict for a provider whose evidence is consistent' in prompt
    assert "blended_platform_count >= 2" in prompt
    # The flatness driver is gone
    assert "Be extremely critical" not in prompt
    # Payload grounds the confidence bands in real evidence volume, and
    # carries each platform's own numbers so disagreement can be named
    sent = json.loads(prompt.split("TOP PROVIDERS TO VALIDATE:\n")[1].split("\n\nEach provider")[0])
    assert sent[0]["blended_platform_count"] == 3
    assert sent[0]["blended_rating"] == 4.4
    assert sent[0]["platform_observations"] == [
        "healthgrades.com 4.6/5 (33 reviews)",
        "vitals.com 4.2/5 (65 reviews)",
    ]
    assert sent[1]["blended_platform_count"] == 0
    assert sent[1]["platform_observations"] == []

def test_refine_matches_a_loosely_echoed_name():
    """Models echo names loosely ('Dr. Hemant Pandey' for 'Dr. Hemant Kumar
    Pandey, MD'). Matching is by NAME TOKENS, so the entry still binds — the
    positional rank fallback that used to cover this case is gone."""
    ranked = [
        {"name": "Dr. Hemant Kumar Pandey, MD", "final_score": 80.0},
        {"name": "Dr. Marianne De Lima, MD", "final_score": 75.0},
    ]
    validation_results = {
        "top_provider_validation": {"top_provider_validations": [{
            "rank": 1,
            "provider_name": "Dr. Hemant Pandey",   # middle name dropped
            "validation_status": "approved_with_concerns",
            "red_flags": ["billing complaints"],
            "confidence_in_recommendation": "medium",
            "validation_notes": "Solid but check billing.",
        }]},
        "alternative_rankings": [],
    }

    refined, _summary = refine_rankings(ranked, validation_results)

    pandey = next(p for p in refined if "Pandey" in p["name"])
    assert pandey["critic_review"] is not None
    assert pandey["critic_review"]["red_flags"] == ["billing complaints"]
    # And it bound to the right person
    de_lima = next(p for p in refined if "De Lima" in p["name"])
    assert de_lima["critic_review"] is None


def test_a_verdict_never_binds_by_position():
    """Round 10's research budget split the critic's rank space from the
    scorer's: the critic numbers its entries over the providers it AUDITED,
    while refinement walks the whole pool. Each unjudged provider sorting above
    a judged one shifted the two apart by one, so the old positional fallback
    handed a provider its neighbour's verdict — and an UNRESEARCHED provider,
    which the critic never saw, could collect a -8 that belonged to someone
    else. It only escaped notice because all four unjudged providers happened
    to sort last on 2026-07-27."""
    ranked = [
        {"name": "Dr. Alpha", "final_score": 90.0},
        {"name": "Dr. Bravo", "final_score": 85.0, "ai_judged": False},
        {"name": "Dr. Chen", "final_score": 80.0},
    ]
    validation_results = {
        "top_provider_validation": {"top_provider_validations": [
            # The critic audited Alpha and Chen; its rank 2 IS Chen. A garbled
            # name is what used to send this down the positional path.
            {"rank": 1, "provider_name": "Dr. Alpha",
             "validation_status": "approved", "red_flags": []},
            {"rank": 2, "provider_name": "!!! unparseable !!!",
             "validation_status": "rejected", "red_flags": ["fabricated credentials"]},
        ]},
        "alternative_rankings": [],
    }

    refined, _summary = refine_rankings(ranked, validation_results)
    by_name = {p["name"]: p for p in refined}

    # Bravo sits at full-position 2 and would have inherited the rank-2 verdict
    assert by_name["Dr. Bravo"]["critic_review"] is None
    assert by_name["Dr. Bravo"]["refined_score"] == 85.0
    # Chen keeps its own score rather than a stranger's -15
    assert by_name["Dr. Chen"]["critic_review"] is None
    assert by_name["Dr. Chen"]["refined_score"] == 80.0


def test_one_verdict_cannot_be_claimed_twice():
    """Two providers sharing a surname must not both absorb one verdict."""
    ranked = [
        {"name": "Dr. David Kim", "final_score": 80.0},
        {"name": "Dr. Jane Kim", "final_score": 75.0},
    ]
    validation_results = {
        "top_provider_validation": {"top_provider_validations": [{
            "rank": 1, "provider_name": "Dr. Jane Kim",
            "validation_status": "rejected", "red_flags": ["malpractice"],
        }]},
        "alternative_rankings": [],
    }

    refined, _summary = refine_rankings(ranked, validation_results)
    reviewed = [p["name"] for p in refined if p["critic_review"] is not None]
    assert reviewed == ["Dr. Jane Kim"]


@pytest.mark.parametrize(
    "critic_name,provider_name",
    [
        ("Dr. Hussam Seif Eddeine", "Hussam Seif-Eddeine, MD"),
        ("Dr. Andrea An, M.D.", "Andrea An, MD"),
        ("Kumar Sannapaneni, MD", "Dr. Kumar Sannapaneni"),
        ("Dr. Jane O'Brien DO", "Jane OBrien"),
    ],
)
def test_refine_matches_by_name_across_credential_drift(critic_name, provider_name):
    """The NAME index must actually match — the rank fallback above is the
    safety net, not the mechanism.

    The previous local normalizer stripped only "dr." and "dr ", so every one
    of these pairs missed ("hussam seif eddeine md" vs "hussam seif eddeine")
    and the name index was dead in practice. Only the rank-fallback direction
    was ever tested, so nothing caught it.
    """
    ranked = [{"name": provider_name, "final_score": 80.0}]
    validation_results = {
        "top_provider_validation": {"top_provider_validations": [{
            "rank": 99,                      # deliberately unusable
            "provider_name": critic_name,
            "validation_status": "approved",
            "red_flags": [],
            "confidence_in_recommendation": "high",
            "validation_notes": "Matched by name, not by rank.",
        }]},
    }

    refined, _summary = refine_rankings(ranked, validation_results)

    assert refined[0]["critic_review"] is not None
    assert refined[0]["critic_review"]["notes"] == "Matched by name, not by rank."


def test_refine_does_not_misattribute_verdicts_when_critic_reorders():
    """A verdict must never land on the wrong provider.

    With the name index dead, EVERY entry fell through to positional rank
    matching — so a critic that returns its entries in a different order than
    the scorer's put the rejection and its red flag on the innocent provider:
    Andrea An dropped 80 -> 61 and the actually-rejected provider kept the top
    spot.
    """
    ranked = [
        {"name": "Andrea An, MD", "final_score": 80.0},
        {"name": "Hussam Seif-Eddeine, MD", "final_score": 78.0},
    ]
    validation_results = {
        "top_provider_validation": {"top_provider_validations": [
            {"rank": 1, "provider_name": "Dr. Hussam Seif Eddeine",
             "validation_status": "rejected", "red_flags": ["billing complaints"],
             "confidence_in_recommendation": "low"},
            {"rank": 2, "provider_name": "Dr. Andrea An, M.D.",
             "validation_status": "approved", "red_flags": [],
             "confidence_in_recommendation": "high"},
        ]},
    }

    refined, _summary = refine_rankings(ranked, validation_results)
    by_name = {p["name"]: p for p in refined}

    assert by_name["Hussam Seif-Eddeine, MD"]["refinement_adjustment"] < 0
    assert by_name["Andrea An, MD"]["refinement_adjustment"] >= 0
    assert by_name["Andrea An, MD"]["critic_review"]["red_flags"] == []


def test_critic_calls_use_configured_model(critic_validator: CriticValidatorAgent):
    """All three analysis calls read CRITIC_MODEL (default Opus 4.8 —
    briefly Opus 5, 2026-08-07 to 2026-08-09, reverted on measured
    latency: ~35-40% slower per call at the identical $5/$25 price, so
    the flip bought reasoning depth the demo pays for in visible
    seconds); the JSON-repair utility stays on Haiku regardless."""
    assert critic_validator.config.CRITIC_MODEL == "claude-opus-4-8"

    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_BIAS_ANALYSIS_RESPONSE
    critic_validator.anthropic_client.messages.create.return_value = mock_response
    critic_validator._analyze_ranking_bias(MOCK_RANKED_PROVIDERS, {})
    assert (critic_validator.anthropic_client.messages.create
            .call_args.kwargs["model"] == "claude-opus-4-8")

    critic_validator.config.CRITIC_MODEL = "claude-sonnet-5"
    mock_response.content[0].text = MOCK_VALIDATION_RESPONSE
    critic_validator._validate_top_recommendations(MOCK_RANKED_PROVIDERS)
    assert (critic_validator.anthropic_client.messages.create
            .call_args.kwargs["model"] == "claude-sonnet-5")

@patch('agents.critic_validator.CriticValidatorAgent._analyze_ranking_bias')
@patch('agents.critic_validator.CriticValidatorAgent._validate_top_recommendations')
def test_validate_rankings_main_method(mock_validate, mock_bias, critic_validator: CriticValidatorAgent):
    """Test the main validate_rankings method (two analyses, no alternatives)."""

    mock_bias.return_value = json.loads(MOCK_BIAS_ANALYSIS_RESPONSE)
    mock_validate.return_value = json.loads(MOCK_VALIDATION_RESPONSE)

    result = critic_validator.validate_rankings(MOCK_RANKED_PROVIDERS, {})

    assert result['status'] == 'success'
    assert 'bias_analysis' in result['validation_results']
    assert 'top_provider_validation' in result['validation_results']
    # The removed feature must not reappear in the output shape
    assert 'alternative_rankings' not in result['validation_results']
    mock_bias.assert_called_once()
    mock_validate.assert_called_once()

def test_validate_rankings_no_providers(critic_validator: CriticValidatorAgent):
    """Test that validate_rankings handles an empty list of providers."""

    result = critic_validator.validate_rankings([], {})

    assert result['status'] == 'no_providers'


class TestRefineRankings:
    """The critique feedback loop: deterministic re-ranking, no LLM calls."""

    @staticmethod
    def _providers():
        return [
            {"name": "Dr. Alpha", "final_score": 90.0},
            {"name": "Dr. Beta", "final_score": 88.0},
            {"name": "Dr. Gamma", "final_score": 80.0},
        ]

    def test_red_flags_demote_the_top_provider(self):
        validation = {
            "validation_results": {
                "top_provider_validation": {
                    "top_provider_validations": [
                        {
                            "provider_name": "Dr. Alpha",
                            "rank": 1,
                            "validation_status": "caution",
                            "red_flags": ["negative review pattern", "sparse data"],
                            "confidence_in_recommendation": "low",
                        },
                        {
                            "provider_name": "Dr. Beta",
                            "rank": 2,
                            "validation_status": "approved",
                            "red_flags": [],
                            "confidence_in_recommendation": "high",
                        },
                    ]
                },
                "alternative_rankings": [],
            }
        }

        refined, summary = refine_rankings(self._providers(), validation)

        # Alpha: 90 - 8 (caution) - 8 (2 flags) - 4 (low confidence) = 70,
        # which drops it below both Beta (88 + 2 = 90) and untouched Gamma (80)
        assert [p["name"] for p in refined] == ["Dr. Beta", "Dr. Gamma", "Dr. Alpha"]
        assert summary["applied"] is True
        assert {(m["from"], m["to"]) for m in summary["moves"]} == {(1, 3), (2, 1), (3, 2)}
        assert refined[0]["final_rank"] == 1
        alpha = refined[2]
        assert alpha["pre_refinement_rank"] == 1
        assert alpha["refinement_reasons"]  # demotion is explained

    def test_clean_validation_is_a_no_op(self):
        validation = {
            "validation_results": {
                "top_provider_validation": {
                    "top_provider_validations": [
                        {"provider_name": "Dr. Alpha", "rank": 1, "validation_status": "approved",
                         "red_flags": [], "confidence_in_recommendation": "medium"},
                    ]
                },
                "alternative_rankings": [],
            }
        }

        refined, summary = refine_rankings(self._providers(), validation)

        assert [p["name"] for p in refined] == ["Dr. Alpha", "Dr. Beta", "Dr. Gamma"]
        assert summary["applied"] is False
        assert summary["moves"] == []

    def test_empty_or_missing_validation_is_safe(self):
        refined, summary = refine_rankings(self._providers(), {})
        assert [p["name"] for p in refined] == ["Dr. Alpha", "Dr. Beta", "Dr. Gamma"]
        assert summary["applied"] is False

        refined_empty, summary_empty = refine_rankings([], {"validation_results": {}})
        assert refined_empty == [] and summary_empty["applied"] is False

    def test_name_matching_tolerates_dr_prefix_and_case(self):
        validation = {
            "validation_results": {
                "top_provider_validation": {
                    "top_provider_validations": [
                        {"provider_name": "ALPHA", "rank": 1, "validation_status": "rejected",
                         "red_flags": [], "confidence_in_recommendation": "medium"},
                    ]
                },
                "alternative_rankings": [],
            }
        }

        refined, _ = refine_rankings(self._providers(), validation)
        alpha = next(p for p in refined if p["name"] == "Dr. Alpha")
        assert alpha["refinement_adjustment"] == -15.0

    def test_scores_clamped_and_inputs_not_mutated(self):
        providers = [{"name": "Dr. Alpha", "final_score": 3.0}]
        validation = {
            "validation_results": {
                "top_provider_validation": {
                    "top_provider_validations": [
                        {"provider_name": "Dr. Alpha", "rank": 1, "validation_status": "rejected",
                         "red_flags": ["a", "b", "c", "d"], "confidence_in_recommendation": "low"},
                    ]
                },
            }
        }

        refined, _ = refine_rankings(providers, validation)

        assert refined[0]["refined_score"] == 0.0  # clamped at the floor
        assert "refined_score" not in providers[0]  # originals untouched


def test_refine_attaches_critic_review_text():
    """The critic's own words ride on each validated provider for the UI."""

    providers = [
        {"name": "Dr. Alpha", "final_score": 90.0},
        {"name": "Dr. Beta", "final_score": 80.0},
    ]
    validation = {
        "validation_results": {
            "top_provider_validation": {
                "top_provider_validations": [
                    {
                        "provider_name": "Dr. Alpha",
                        "rank": 1,
                        "validation_status": "approved",
                        "confidence_in_recommendation": "medium",
                        "validation_notes": "Strong reviews but thin insurance evidence.",
                        "red_flags": ["single-source rating"],
                        "patient_considerations": "Verify coverage before booking.",
                    }
                ]
            },
            "alternative_rankings": [],
        }
    }

    refined, _ = refine_rankings(providers, validation)

    alpha = next(p for p in refined if p["name"] == "Dr. Alpha")
    assert alpha["critic_review"]["notes"] == "Strong reviews but thin insurance evidence."
    assert alpha["critic_review"]["red_flags"] == ["single-source rating"]
    assert alpha["critic_review"]["considerations"] == "Verify coverage before booking."
    assert alpha["critic_review"]["status"] == "approved"

    beta = next(p for p in refined if p["name"] == "Dr. Beta")
    assert beta["critic_review"] is None


class TestJsonRepair:
    """Escalating JSON repair: mechanical fixes first, one Haiku call as last resort."""

    def test_valid_json_needs_no_repair(self, critic_validator: CriticValidatorAgent):
        result = critic_validator._parse_json_with_repair('{"a": 1}', "test")
        assert result == {"a": 1}
        critic_validator.anthropic_client.messages.create.assert_not_called()

    def test_mechanical_repair_handles_trailing_commas_and_newlines(self, critic_validator: CriticValidatorAgent):
        broken = '{"notes": "line one\nline two", "flags": ["a", "b",],}'
        result = critic_validator._parse_json_with_repair(broken, "test")
        assert result["flags"] == ["a", "b"]
        assert "line one" in result["notes"]
        critic_validator.anthropic_client.messages.create.assert_not_called()

    def test_llm_repair_fixes_unescaped_quotes(self, critic_validator: CriticValidatorAgent):
        # The classic Sonnet failure: an unescaped inner double quote
        broken = '{"assessment": "reviews say "great doctor" repeatedly"}'
        fixed = MagicMock()
        fixed.content[0].text = '{"assessment": "reviews say \\"great doctor\\" repeatedly"}'
        critic_validator.anthropic_client.messages.create.return_value = fixed

        result = critic_validator._parse_json_with_repair(broken, "test")

        assert result == {"assessment": 'reviews say "great doctor" repeatedly'}
        call_kwargs = critic_validator.anthropic_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5"
        assert "fix syntax only" in call_kwargs["messages"][0]["content"]

    def test_unrecoverable_returns_none(self, critic_validator: CriticValidatorAgent):
        junk = MagicMock()
        junk.content[0].text = "still not json"
        critic_validator.anthropic_client.messages.create.return_value = junk

        result = critic_validator._parse_json_with_repair('{"broken": "x" "y"}', "test")

        assert result is None

    def test_bias_analysis_recovers_via_repair_chain(self, critic_validator: CriticValidatorAgent):
        """A malformed Sonnet response no longer drops bias findings."""

        sonnet_broken = MagicMock()
        sonnet_broken.content[0].text = '{"bias_assessment": {"detected_biases": ["scored "neutral" wrongly"]}}'
        haiku_fixed = MagicMock()
        haiku_fixed.content[0].text = '{"bias_assessment": {"detected_biases": ["scored \\"neutral\\" wrongly"]}}'
        critic_validator.anthropic_client.messages.create.side_effect = [sonnet_broken, haiku_fixed]

        analysis = critic_validator._analyze_ranking_bias(MOCK_RANKED_PROVIDERS, {})

        assert analysis["bias_assessment"]["detected_biases"] == ['scored "neutral" wrongly']
        assert critic_validator.anthropic_client.messages.create.call_count == 2


class TestJudgeConsistencyHandoff:
    """The critic audits the judge's scoring, not just the provider.

    Round 5: the payload slot that was supposed to carry the judge's
    confidence held `ai_confidence` — a key NO agent has ever written. Every
    provider on every search arrived with the literal default 50, so the
    checklist line asking whether confidence tracked quality was auditing a
    constant, and the critic never saw the judge's per-criterion scores at
    all. It was structurally unable to catch a judge scoring practical_access
    "no evidence" beside a summary describing long wait times.
    """

    @staticmethod
    def _sent(critic_validator):
        prompt = critic_validator.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
        return json.loads(
            prompt.split("TOP PROVIDERS TO VALIDATE:\n")[1].split("\n\nEach provider")[0]
        ), prompt

    def test_critic_gets_the_judges_real_rubric_not_a_constant(self, critic_validator):
        provider = {
            "name": "Dr. Rabin", "final_score": 90, "review_summary": "Long wait times.",
            "ai_rubric": {"review_substance": 46, "red_flags": 27, "practical_access": 10},
            "ai_evidence": {"practical_access": "no evidence"},
        }
        mock_response = MagicMock()
        mock_response.content[0].text = MOCK_VALIDATION_RESPONSE
        critic_validator.anthropic_client.messages.create.return_value = mock_response

        critic_validator._validate_top_recommendations([provider])
        sent, prompt = self._sent(critic_validator)

        assert sent[0]["ai_rubric"]["practical_access"] == 10
        assert sent[0]["ai_evidence"]["practical_access"] == "no evidence"
        # The phantom field is gone — a hardcoded 50 masquerading as a signal
        assert "ai_confidence" not in sent[0]
        assert "ai_confidence" not in prompt
        assert "JUDGE CONSISTENCY" in prompt

    def test_prompt_routes_judge_faults_away_from_the_provider_score(self, critic_validator):
        """A judge mistake in `red_flags` would cost the PROVIDER 4 points via
        refine_rankings — demoting someone for an error they did not cause."""
        mock_response = MagicMock()
        mock_response.content[0].text = MOCK_VALIDATION_RESPONSE
        critic_validator.anthropic_client.messages.create.return_value = mock_response

        critic_validator._validate_top_recommendations([{"name": "Dr. X", "final_score": 80}])
        _, prompt = self._sent(critic_validator)

        assert "A mistake by the upstream judge is NEVER a red flag" in prompt
        assert "recommendation_adjustments (free text; affects NO score" in prompt

    def test_judge_findings_are_logged_and_never_move_the_score(self, caplog):
        """The finding has to land somewhere. It is a fault in our pipeline,
        not information a patient needs — so it goes to the log, and the
        provider's score is untouched."""
        validation = {
            "validation_results": {
                "top_provider_validation": {
                    "top_provider_validations": [{
                        "provider_name": "Dr. Alpha",
                        "rank": 1,
                        "validation_status": "approved",
                        "red_flags": [],
                        "confidence_in_recommendation": "high",
                        "recommendation_adjustments": (
                            "practical_access scored 10/20 'no evidence' though the "
                            "summary describes long wait times."
                        ),
                    }]
                }
            }
        }
        with caplog.at_level(logging.WARNING, logger="agents.critic_validator"):
            refined, _ = refine_rankings([{"name": "Dr. Alpha", "final_score": 90.0}], validation)

        alpha = refined[0]
        assert "long wait times" in alpha["critic_review"]["judge_findings"]
        # approved + no flags + high confidence = +2 and nothing else; the
        # judge finding itself contributes zero
        assert alpha["refinement_adjustment"] == 2.0
        assert "judge/evidence inconsistency" in caplog.text

    def test_clean_judge_scoring_logs_nothing(self, caplog):
        validation = {
            "validation_results": {
                "top_provider_validation": {
                    "top_provider_validations": [{
                        "provider_name": "Dr. Alpha", "rank": 1,
                        "validation_status": "approved", "red_flags": [],
                        "confidence_in_recommendation": "high",
                        "recommendation_adjustments": "",
                    }]
                }
            }
        }
        with caplog.at_level(logging.WARNING, logger="agents.critic_validator"):
            refined, _ = refine_rankings([{"name": "Dr. Alpha", "final_score": 90.0}], validation)

        assert refined[0]["critic_review"]["judge_findings"] == ""
        assert "judge/evidence inconsistency" not in caplog.text


def test_clean_run_produces_no_user_guidance(critic_validator: CriticValidatorAgent):
    """`user_guidance` renders under "What this ranking doesn't capture",
    alongside real gaps the critic identified in OUR ranking.

    It used to open with "Review detailed provider information beyond just
    rankings" — appended unconditionally, from no model output. Round 7 wired
    the field to that panel; wiring it revealed the first element was filler,
    and filler under that heading reads as a gap we found but can't articulate.
    Same pattern as the two hardcoded `key_findings` strings deleted in round 7,
    whose gravestone comment sits eleven lines above where this one lived."""
    recommendations = critic_validator._generate_final_recommendations(
        [{"name": "Dr. Alpha", "final_score": 88}],
        {"bias_assessment": {"severity": "low", "detected_biases": []}},
        {"overall_ranking_validity": {"confidence": "high"}},
    )
    assert recommendations["user_guidance"] == []
    assert recommendations["key_findings"] == []


def test_low_confidence_guidance_is_earned(critic_validator: CriticValidatorAgent):
    """The one surviving entry is conditional, so it says something true when
    it appears."""
    recommendations = critic_validator._generate_final_recommendations(
        [{"name": "Dr. Alpha", "final_score": 88}],
        {"bias_assessment": {"severity": "low", "detected_biases": []}},
        {"overall_ranking_validity": {"confidence": "low"}},
    )
    assert recommendations["user_guidance"] == [
        "Exercise additional caution in provider selection"
    ]


class TestValidationTokenBudget:
    """A flat ceiling on a pool that became a knob — the same defect the judge
    hit one agent over, and it was not re-derived here when it was fixed there.

    Round 13 raised the blast radius: an unrecoverable critic response leaves
    every provider `not_critiqued`, which correctly EMPTIES the shortlist. The
    right failure with the wrong log is still a bug — an operator sees zero
    recommendations and no cause.
    """

    def test_the_budget_scales_with_the_pool(self):
        assert _validation_token_budget(16) > _validation_token_budget(8)

    def test_a_small_pool_still_gets_the_json_envelope(self):
        """The floor exists because the fixed part of the response — the
        overall_ranking_validity object — does not shrink with the pool."""
        assert _validation_token_budget(0) >= 4000
        assert _validation_token_budget(1) >= 4000

    def test_the_budget_is_bounded(self):
        assert _validation_token_budget(10_000) == _VALIDATION_MAX_TOKENS

    def test_the_scaled_budget_reaches_the_api_call(self, critic_validator):
        """The WIRING, not the helper.

        Revert-in-isolation caught this: replacing the call site with a flat
        `budget = 6500` left every test above green, because they all exercise
        `_validation_token_budget` directly. A scaling function nothing calls
        is the same defect as no scaling function.
        """
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content[0].text = '{"top_provider_validations": []}'
        critic_validator.anthropic_client.messages.create.return_value = response

        seen = {}
        # 24, not 12: under three shards a pool of 12 deals shards of 4,
        # whose budget sits on the same floor as a pool of 2 — the scaling
        # only becomes visible once a SHARD is big enough to clear the floor.
        for count in (2, 24):
            critic_validator.anthropic_client.messages.create.reset_mock()
            critic_validator._validate_top_recommendations(
                [{"name": f"Dr. {i}", "final_score": 80.0} for i in range(count)]
            )
            calls = critic_validator.anthropic_client.messages.create.call_args_list
            seen[count] = [c.kwargs["max_tokens"] for c in calls]

        assert max(seen[24]) > max(seen[2]), f"max_tokens must scale with the pool, got {seen}"
        # The budget scales to what ONE CALL has to return, which after the
        # Phase 2 split is a shard. Asserting against the whole pool's size
        # would demand a ceiling twice as large as the response it bounds —
        # and would pass just as well if the split silently stopped happening.
        assert seen[2] == [_validation_token_budget(2)], "a small pool stays one call"
        assert seen[24] == [_validation_token_budget(8)] * 3, \
            "24 providers deal three shards of eight, each budgeted per shard"

    def test_truncation_is_logged_with_its_named_cause(self, critic_validator, caplog):
        """Not "JSON parse failed" — the number that has to change, and what it
        costs. This is the exact incident the judge already handles
        (`finish_reason == "length"`); the critic had no equivalent."""
        response = MagicMock()
        response.stop_reason = "max_tokens"
        response.content[0].text = '{"top_provider_validations": [{"provider_name": "Dr. A"'
        critic_validator.anthropic_client.messages.create.return_value = response

        with caplog.at_level(logging.ERROR):
            critic_validator._validate_top_recommendations(
                [{"name": "Dr. A", "final_score": 80.0}]
            )

        assert any("TRUNCATED" in r.message for r in caplog.records), caplog.text
        assert any("EMPTIES the shortlist" in r.message for r in caplog.records)

    def test_a_clean_response_logs_no_truncation(self, critic_validator, caplog):
        """A permanent warning trains the eye to skip the one that matters."""
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content[0].text = '{"top_provider_validations": []}'
        critic_validator.anthropic_client.messages.create.return_value = response

        with caplog.at_level(logging.ERROR):
            critic_validator._validate_top_recommendations(
                [{"name": "Dr. A", "final_score": 80.0}]
            )

        assert not any("TRUNCATED" in r.message for r in caplog.records)


def test_the_bias_payload_uses_the_research_budget_not_a_parallel_cap(critic_validator):
    """`ranked_providers[:10]` was a SECOND hardcoded budget, and it had
    already diverged: the shipped `MAX_PROVIDERS_TO_ENRICH` is 8, so the bias
    analyst was reasoning over two providers the rest of the pipeline never
    researched."""
    critic_validator.config.MAX_PROVIDERS_TO_ENRICH = 3
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content[0].text = '{"bias_assessment": {"severity": "low", "detected_biases": []}}'
    critic_validator.anthropic_client.messages.create.return_value = response

    providers = [{"name": f"Dr. {i}", "final_score": 90.0 - i} for i in range(8)]
    critic_validator._analyze_ranking_bias(providers, {})

    prompt = critic_validator.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Dr. 2" in prompt
    assert "Dr. 3" not in prompt, "the payload must stop at the research budget"


class TestBiasOutputContract:
    """The bias call's output contract is the validation stage's latency lever.

    First instrumented run (2026-08-05): bias 35.37s == the stage total, both
    deep shards (~18s) finished and waiting — and with thinking disabled that
    time is output generation. The prompt requested eleven fields; SIX were
    parsed and read by nothing (validity_concerns x3, blind_spots.impact,
    blind_spots.recommendations, overall_assessment — checked consumer by
    consumer), the patient-facing explanation is discarded by the panel
    unless a bias was flagged, and the live output stated the same two facts
    three times. The contract now requests only consumed fields, caps them
    at what the consumers render, and tells the model a clean run earns one
    sentence, not an essay.
    """

    def _prompt(self, critic_validator):
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [MagicMock(text=MOCK_BIAS_ANALYSIS_RESPONSE)]
        critic_validator.anthropic_client.messages.create.return_value = response
        critic_validator._analyze_ranking_bias(MOCK_RANKED_PROVIDERS, {})
        return critic_validator.anthropic_client.messages.create.call_args.kwargs

    def test_no_dead_fields_are_requested(self, critic_validator):
        """Every schema key the prompt asks for must have a consumer; a field
        nobody reads is pure generation latency on the stage's long pole."""
        prompt = self._prompt(critic_validator)["messages"][0]["content"]

        for dead in ("validity_concerns", "ranking_issues", "misleading_aspects",
                     "overall_assessment", '"impact"', '"recommendations"',
                     "RANKING VALIDITY"):
            assert dead not in prompt, f"dead field {dead!r} is being requested again"
        for live in ("detected_biases", "severity", "explanation",
                     "technical_explanation", "missing_factors"):
            assert live in prompt

    def test_the_caps_and_the_clean_case_contract_are_stated(self, critic_validator):
        """The caps mirror the consumers (the panel renders at most 4 bias
        bullets and 3 missing factors), and a clean run must not fund an
        essay the panel will discard — app gates the explanation on a bias
        actually being flagged."""
        prompt = self._prompt(critic_validator)["messages"][0]["content"]

        assert "at most 4 entries" in prompt
        assert "at most 3 sentences" in prompt
        assert "at most 3 entries" in prompt
        assert "Do not manufacture findings" in prompt
        assert "must NOT restate the detected_biases entries" in prompt

    def test_clean_runs_still_earn_a_patient_read(self, critic_validator):
        """Round 30 (owner call): the panel renders the explanation on EVERY
        run — the critic's independent read is portfolio-grade prose ("the
        top two are separated by a very small margin — both are strong
        choices") and hiding it on clean runs hid the validator's best
        moments. The old clean-case rule forced ONE throwaway sentence
        because the panel used to discard it; the prompt must now tell the
        model the sentence budget is worth spending — while lists stay
        empty and findings stay unmanufactured."""
        prompt = self._prompt(critic_validator)["messages"][0]["content"]

        assert '"explanation" is NOT discarded on clean runs' in prompt
        assert "how close the top ranks are" in prompt
        # The old rule must not linger beside the new one.
        assert "write ONE short sentence for each explanation field" not in prompt

    def test_the_severity_vocabulary_is_pinned(self, critic_validator):
        """This run wrote "moderate" — not in the low/medium/high set that
        three surfaces string-match. It degraded gracefully by luck (the
        biases-exist branches fired), but a clean run with severity
        "moderate" would SUPPRESS the callout the model meant to raise. An
        output field downstream code string-matches is a contract, and a
        contract the prompt doesn't state is one the model improvises on."""
        prompt = self._prompt(critic_validator)["messages"][0]["content"]

        assert 'EXACTLY one of "low", "medium", or "high"' in prompt

    def test_missing_data_imputations_are_stated_as_verified_equivalents(self, critic_validator):
        """The 2026-07-28 run's panel told patients "Doctors with no listed
        years of experience are treated as fairly new (a low, fixed
        experience score)" — false since the imputation sweeps: unknown
        tenure scores exactly what a VERIFIED 10-year career scores, and an
        unknown rating scores the 3.5-star Bayesian prior itself. The
        mechanics block stated neither equivalence, so the model invented a
        mechanism — the same failure class as the causality claims the
        weighted_contribution rule closed. The equivalences are now stated
        and the "fairly new" framing is named as forbidden."""
        prompt = self._prompt(critic_validator)["messages"][0]["content"]

        assert "NEUTRAL VERIFIED-EQUIVALENT" in prompt
        assert "3.5-star" in prompt
        assert "VERIFIED 10-year career" in prompt
        assert '"fairly new"' in prompt

    def test_the_severity_ladder_is_anchored(self, critic_validator):
        """Pinning the vocabulary settled WHICH words; nothing said WHEN. Two
        consecutive live runs then scored "medium" for findings the same
        output's arithmetic register called the chosen weights working as
        chosen — an un-anchored rung is not reproducible, the same defect the
        judge's practical_access band had before it was tiled. Each rung now
        states its entry criterion: low = advisory/preference-dependent (a
        chosen weight doing its job is not a bias), medium = could mislead a
        reader who skips the panel, high = the arithmetic shows distortion."""
        prompt = self._prompt(critic_validator)["messages"][0]["content"]

        assert "A chosen weight doing its job is NOT a bias" in prompt
        assert "could mislead a reader who never opens this panel" in prompt
        assert "arithmetic shows real distortion" in prompt

    def test_detected_biases_admits_only_distortions(self, critic_validator):
        """The severity ladder fixed the WORD; nothing fixed the LIST. Both
        2026-08-09 live runs filed working-as-designed observations
        (Bayesian shrinkage doing its job, chosen weights realized as
        chosen) as detected_biases, so the panel announced "2 potential
        biases flagged" on runs whose own technical register concluded no
        factor distorted anything — the tile contradicting the prose
        beneath it. The list now has an entry criterion of its own:
        distortions only; system-working-as-designed observations go to
        the explanation or user_guidance."""
        prompt = self._prompt(critic_validator)["messages"][0]["content"]

        assert '"detected_biases" lists ONLY distortions' in prompt
        assert "is NOT a detected bias and must NOT appear in that list" in prompt
        assert "put it in the explanation or user_guidance instead" in prompt

    def test_the_patient_register_is_gated_by_the_arithmetic(self, critic_validator):
        """On 2026-08-06 the two registers disagreed and the PATIENT got the
        wrong one: the plain-language bullet said the #1 doctor "edges ahead
        almost entirely because of how near they are", while the technical
        register — the only one forced to cite weighted_contribution — read
        rating +0.54 / location +0.44 / experience -0.36 and concluded "the
        decisive factor is NOT distance". The no-jargon rule had become a
        no-arithmetic rule. Plain language is now a TRANSLATION step only
        survivors of the arithmetic reach; refuted candidates are dropped,
        not translated."""
        prompt = self._prompt(critic_validator)["messages"][0]["content"]

        assert 'Derive "technical_explanation" FIRST' in prompt
        assert "never translated into plain language" in prompt

    def test_the_ceiling_matches_the_capped_contract(self, critic_validator):
        """4500 tokens sized the old eleven-field essay; the capped contract
        needs ~700 and gets ~3x headroom, not the old ceiling back."""
        assert self._prompt(critic_validator)["max_tokens"] == 2000

    def test_truncation_is_named_not_silent(self, critic_validator, caplog):
        """Round-9 lesson, applied here when the ceiling dropped: a response
        cut by max_tokens fails the parse and falls back, which without the
        log reads as a model failure — the wrong bug to hunt."""
        import logging
        response = MagicMock()
        response.stop_reason = "max_tokens"
        response.content = [MagicMock(text='{"bias_assessment": {"sever')]
        critic_validator.anthropic_client.messages.create.return_value = response

        with caplog.at_level(logging.WARNING, logger="agents.critic_validator"):
            critic_validator._analyze_ranking_bias(MOCK_RANKED_PROVIDERS, {})

        assert any("ceiling" in r.message for r in caplog.records)


class TestCriticSendsNoSamplingParams:
    """The critic must send NO `temperature` — the live API rejects it.

    Round 23 shipped `temperature=0` on both verdict-affecting calls to stop
    sampled verdict flips (-8/+2 in refine_rankings). The live API returned
    400 "`temperature` is deprecated for this model" on EVERY call — Opus 4.8
    is in the same no-temperature family as the reasoning judge — so both
    deep shards died, every provider fell to `not_critiqued`, and the live
    shortlist rendered EMPTY while the suite stayed green: mocks are blind
    to wire-level param validation. These tests pin the ABSENCE, so the next
    determinism attempt cannot re-ship the outage; critic sampling is part
    of the honest stability floor. Re-verified live 2026-08-07 against the
    new Opus 5 default: temperature draws the identical 400 there, so the
    pins remain both correct and necessary.
    """

    def test_the_bias_call_sends_no_temperature(self, critic_validator):
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [MagicMock(text=MOCK_BIAS_ANALYSIS_RESPONSE)]
        critic_validator.anthropic_client.messages.create.return_value = response

        critic_validator._analyze_ranking_bias(MOCK_RANKED_PROVIDERS, {})

        assert "temperature" not in (critic_validator.anthropic_client
                                     .messages.create.call_args.kwargs)

    def test_the_deep_validation_sends_no_temperature(self, critic_validator):
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [MagicMock(text=MOCK_VALIDATION_RESPONSE)]
        critic_validator.anthropic_client.messages.create.return_value = response

        critic_validator._validate_top_recommendations(MOCK_RANKED_PROVIDERS)

        assert "temperature" not in (critic_validator.anthropic_client
                                     .messages.create.call_args.kwargs)


class TestValidationCallTimings:
    """Per-call wall times for the validation stage's concurrent Opus calls.

    The stage total is max() of three concurrent calls (bias + two deep
    shards), so the timeline's 35.7s cannot say WHICH call is the long pole —
    and "shard the deep validation 3 ways?" turns exactly on that: if the
    bias call is the pole, more deep shards buy ~nothing, because bias
    reasons about the ORDERING and cannot be split. These timings are the
    measurement that decision reads.
    """

    def test_sharded_deep_validation_times_each_shard(self, critic_validator):
        """A pool of 4 still deals TWO shards, not `_VALIDATION_SHARDS` (3):
        the ceil(pool/3) cap exists because three shards over four providers
        is 2/1/1, and a single-record shard is exactly what the split floor
        exists to prevent — "find the real differences between these
        providers" has no meaning for a call holding one."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=MOCK_VALIDATION_RESPONSE)]
        mock_response.stop_reason = "end_turn"
        critic_validator.anthropic_client.messages.create.return_value = mock_response

        pool = [dict(p, name=f"{p['name']} {i}") for i, p in
                enumerate(MOCK_RANKED_PROVIDERS * 2)]
        result = critic_validator._validate_top_recommendations(pool)

        timings = result["call_timings"]
        assert [t["call"] for t in timings] == ["deep_shard_1_of_2", "deep_shard_2_of_2"]
        assert all(isinstance(t["seconds"], float) and t["seconds"] >= 0 for t in timings)
        assert [t["providers"] for t in timings] == [2, 2]

    def test_a_budget_pool_deals_three_shards(self, critic_validator):
        """Taken on the measured reading the timing instrumentation existed
        for (2026-08-06: deep shards 18.88s/16.02s over 4+4 while the trimmed
        bias call sat at 12.61s — deep is the stage's pole). Eight providers
        deal 3/3/2: one thin shard is the accepted cost; the next run's
        eyeball check is whether the 2-provider shard's verdicts read flatter
        than the others."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=MOCK_VALIDATION_RESPONSE)]
        mock_response.stop_reason = "end_turn"
        critic_validator.anthropic_client.messages.create.return_value = mock_response

        pool = [dict(p, name=f"{p['name']} {i}") for i, p in
                enumerate(MOCK_RANKED_PROVIDERS * 4)]
        result = critic_validator._validate_top_recommendations(pool)

        timings = result["call_timings"]
        assert [t["call"] for t in timings] == [
            "deep_shard_1_of_3", "deep_shard_2_of_3", "deep_shard_3_of_3"]
        assert sorted(t["providers"] for t in timings) == [2, 3, 3]

    def test_a_pool_of_six_deals_threes_not_all_twos(self, critic_validator):
        """min(3, ceil(6/3)) == 2, so six providers deal 3/3 — three shards
        would make EVERY shard a 2-provider shard, the thin-material failure
        in every call at once rather than in the one accepted straggler."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=MOCK_VALIDATION_RESPONSE)]
        mock_response.stop_reason = "end_turn"
        critic_validator.anthropic_client.messages.create.return_value = mock_response

        pool = [dict(p, name=f"{p['name']} {i}") for i, p in
                enumerate(MOCK_RANKED_PROVIDERS * 3)]
        result = critic_validator._validate_top_recommendations(pool)

        timings = result["call_timings"]
        assert [t["call"] for t in timings] == ["deep_shard_1_of_2", "deep_shard_2_of_2"]
        assert [t["providers"] for t in timings] == [3, 3]

    def test_unsharded_deep_validation_reports_one_timing(self, critic_validator):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=MOCK_VALIDATION_RESPONSE)]
        mock_response.stop_reason = "end_turn"
        critic_validator.anthropic_client.messages.create.return_value = mock_response

        result = critic_validator._validate_top_recommendations(MOCK_RANKED_PROVIDERS)

        assert [t["call"] for t in result["call_timings"]] == ["deep_1_of_1"]
        assert result["call_timings"][0]["providers"] == len(MOCK_RANKED_PROVIDERS)

    @patch('agents.critic_validator.CriticValidatorAgent._analyze_ranking_bias')
    @patch('agents.critic_validator.CriticValidatorAgent._validate_top_recommendations')
    def test_metadata_merges_bias_and_deep_timings(self, mock_validate, mock_bias,
                                                   critic_validator):
        """The stage's timings live in ONE place — validation_metadata — and
        are POPPED off the deep result, not copied: top_provider_validation
        flows to the panel and the recommendation builder, which read provider
        entries and must not grow a second shape to skip."""
        mock_bias.return_value = json.loads(MOCK_BIAS_ANALYSIS_RESPONSE)
        deep = json.loads(MOCK_VALIDATION_RESPONSE)
        deep["call_timings"] = [{"call": "deep_shard_1_of_2", "seconds": 1.23, "providers": 4}]
        mock_validate.return_value = deep

        result = critic_validator.validate_rankings(MOCK_RANKED_PROVIDERS, {})

        timings = result["validation_metadata"]["call_timings"]
        assert timings[0]["call"] == "bias"
        assert isinstance(timings[0]["seconds"], float)
        assert timings[1:] == [{"call": "deep_shard_1_of_2", "seconds": 1.23, "providers": 4}]
        assert "call_timings" not in result["validation_results"]["top_provider_validation"]


def test_per_provider_fields_must_be_self_contained(critic_validator: CriticValidatorAgent):
    """No cross-provider comparatives in per-provider output — every shard.

    The 2026-08-07 run (the first on Opus 5) carded Dr. Hagevik's
    For-patients line with "about 6.4 mi from the search ZIP, the farthest
    of the three" — "the three" being his VALIDATION SHARD, a grouping no
    reader can see: the page shows five cards, and shard composition is an
    implementation artifact. The rubric already forbade SCORING providers
    against each other; nothing forbade the notes from referencing
    shard-mates, and the richer model was the first to write that shape.
    The rule must reach every shard, because any shard can leak."""
    providers = [
        {"name": f"Dr. Number {i}", "final_score": 90 - i, "rating": 4.0,
         "review_count": 10, "review_summary": "Fine.", "review_sentiment": "positive"}
        for i in range(1, 11)
    ]
    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_VALIDATION_RESPONSE
    critic_validator.anthropic_client.messages.create.return_value = mock_response

    critic_validator._validate_top_recommendations(providers)

    calls = critic_validator.anthropic_client.messages.create.call_args_list
    assert len(calls) > 1, "the pool must actually shard, or the per-shard claim is untested"
    for call in calls:
        prompt = call.kwargs["messages"][0]["content"]
        assert "must be SELF-CONTAINED about that provider alone" in prompt
        assert '"the farthest of the three"' in prompt
        assert "write conclusions about one provider at a time" in prompt


class TestMergeCarriesTheCollapseReason:
    """Round 32 (2026-08-11): all four critic calls failed on "credit balance
    is too low to access the Anthropic API". Each deep shard's fallback had
    recorded the exception text in its summary, but the all-failed merge
    discarded every copy and returned the bare "Validation could not be
    completed" — so the one surface that could have named the cause above a
    zero-card page said nothing, and the diagnosis needed container logs.
    """

    _CREDIT_ERROR = (
        "Validation error: Error code: 400 - credit balance is too low to "
        "access the Anthropic API"
    )

    def _failed_shard(self, text=None):
        return {
            "top_provider_validations": [],
            "overall_ranking_validity": {
                "status": "error",
                "confidence": "low",
                "summary": text or self._CREDIT_ERROR,
                "improvement_suggestions": [],
            },
        }

    def test_all_failed_merge_names_the_cause(self):
        from agents.critic_validator import _merge_validation_shards

        merged = _merge_validation_shards(
            [self._failed_shard(), self._failed_shard(), self._failed_shard()]
        )
        validity = merged["overall_ranking_validity"]
        assert validity["status"] == "error"
        assert validity["confidence"] == "low"
        assert validity["summary"].startswith("Validation could not be completed")
        assert "credit balance is too low" in validity["summary"]

    def test_a_healthy_merge_still_ignores_failed_shard_prose(self):
        """When any shard produced verdicts, failed shards stay excluded from
        the confidence vote AND their error prose stays out of the summary —
        the collapse note is for the all-failed case only."""
        from agents.critic_validator import _merge_validation_shards

        healthy = {
            "top_provider_validations": [{"rank": 1, "provider_name": "Dr. A"}],
            "overall_ranking_validity": {
                "status": "validated",
                "confidence": "high",
                "summary": "Ranking holds.",
                "improvement_suggestions": [],
            },
        }
        merged = _merge_validation_shards([healthy, self._failed_shard()])
        validity = merged["overall_ranking_validity"]
        assert validity["summary"] == "Ranking holds."
        assert "credit balance" not in validity["summary"]
        assert validity["confidence"] == "high"
