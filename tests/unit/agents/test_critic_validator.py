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
    """All three analysis calls read CRITIC_MODEL (default Opus 4.8 — the
    validator is the deepest-reasoning role and its output reorders the
    final list); the JSON-repair utility stays on Haiku regardless."""
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
    """A flat ceiling on a pool that became a knob — DESIGN §10.17, one agent
    over from where it was first learned.

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
        for count in (2, 12):
            critic_validator.anthropic_client.messages.create.reset_mock()
            critic_validator._validate_top_recommendations(
                [{"name": f"Dr. {i}", "final_score": 80.0} for i in range(count)]
            )
            calls = critic_validator.anthropic_client.messages.create.call_args_list
            seen[count] = [c.kwargs["max_tokens"] for c in calls]

        assert max(seen[12]) > max(seen[2]), f"max_tokens must scale with the pool, got {seen}"
        # The budget scales to what ONE CALL has to return, which after the
        # Phase 2 split is a shard. Asserting against the whole pool's size
        # would demand a ceiling twice as large as the response it bounds —
        # and would pass just as well if the split silently stopped happening.
        assert seen[2] == [_validation_token_budget(2)], "a small pool stays one call"
        assert seen[12] == [_validation_token_budget(6), _validation_token_budget(6)]

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
