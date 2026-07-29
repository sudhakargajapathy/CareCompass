"""Round-6 panel fixes, critic side: the pass-verdict filter and the payload gap.

The strings in `LIVE_PASS_VERDICTS` are verbatim from the 2026-07-25 run's
audit log. Fixtures invented by hand are what let this bug ship: every existing
fixture was *finding-shaped*, so the helper was tested for extraction,
emptiness, pluralisation and jargon — never for whether the text asserts a
problem.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.critic_validator import (
    CriticValidatorAgent,
    _score_contributions,
    is_judge_concern,
)


# Verbatim from logs/audit.log, 2026-07-25. Nine of the ten entries the panel
# counted as "inconsistencies" were the critic confirming the judge was RIGHT.
LIVE_PASS_VERDICTS = [
    "Judge parked practical_access at 10 (neutral) citing 'appointments are not "
    "rushed' — that is a valid mild access signal, so the neutral score is "
    "acceptable here since no scheduling data exists. No correction needed.",
    "Judge correctly scored practical_access low (5) reflecting wait-time "
    "complaints; scoring matches evidence.",
    "Judge scoring matches evidence; practical_access at 14 fairly reflects "
    "mixed-but-mostly-positive scheduling feedback.",
    "Judge correctly scored practical_access at 8 reflecting the split "
    "scheduling reports; scoring matches evidence.",
    "Judge scoring is consistent with the summary; practical_access at 17 "
    "fairly credits telehealth availability.",
    "Judge scoring matches evidence; the lower Vitals score is consistent with "
    "the mixed reports cited.",
    "Judge correctly scored practical_access very low (3) reflecting the "
    "3-month wait and billing/scheduling issues; scoring matches evidence.",
    "Judge scoring is consistent; review_substance at 25 appropriately "
    "reflects the divided feedback.",
    "Judge correctly scored practical_access at 3 reflecting phone-access and "
    "authorization problems; scoring matches evidence.",
]

REAL_CONCERNS = [
    'practical_access scored 10/20 "no evidence" though the summary describes long wait times.',
    "The ai_evidence snippet for red_flags does not appear anywhere in the review_summary.",
    "practical_access does not match the evidence; the summary describes a 3-month wait.",
    "Judge parked practical_access in its neutral band while the summary plainly "
    "names scheduling failures.",
    "ai_reasoning complains the summary lacks access detail, despite it describing "
    "phone-access problems.",
    "The judge cited a snippet with no basis in the summary.",
    "red_flags scored 30/30 yet the summary contradicts that with repeated billing disputes.",
]


# ---- Change 0: the pass-verdict filter ----

@pytest.mark.parametrize("verdict", LIVE_PASS_VERDICTS)
def test_real_pass_verdicts_are_not_concerns(verdict):
    assert is_judge_concern(verdict) is False


@pytest.mark.parametrize("finding", REAL_CONCERNS)
def test_real_findings_survive_the_filter(finding):
    """Under-reporting is the dangerous direction: it silently discards the
    signal the whole mechanism exists to surface."""
    assert is_judge_concern(finding) is True


def test_the_live_run_would_now_report_zero():
    """The panel said "10 inconsistencies were found". True count: zero."""
    assert sum(is_judge_concern(v) for v in LIVE_PASS_VERDICTS) == 0


def test_a_negated_confirmation_is_a_concern():
    """A real finding can quote the very language a confirmation uses, so
    matching pass phrases alone would discard it."""
    assert is_judge_concern("practical_access does not match the evidence.") is True


def test_an_explicit_all_clear_outranks_problem_vocabulary():
    """A stated verdict beats inferred tone. This entry trips the CONCERN
    pattern ("did not") while explicitly declaring no fault, which is the shape
    of the live entry that first defeated the filter."""
    assert is_judge_concern(
        "Judge did not misread anything here; the neutral score is warranted. "
        "No correction needed."
    ) is False


def test_the_live_parked_entry_is_still_a_pass():
    assert is_judge_concern(
        "Judge parked practical_access at 10 (neutral) citing 'appointments are "
        "not rushed' — acceptable. No correction needed."
    ) is False


def test_unrecognized_phrasing_defaults_to_concern():
    """Better to show a developer one extra line than to drop a real finding."""
    assert is_judge_concern("Something about the rubric looks off here.") is True


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_empty_is_never_a_concern(empty):
    assert is_judge_concern(empty) is False


# ---- Change 1: score_contributions reaches the bias payload ----

def _scored_provider(name, rating_s, exp_s, loc_s, w=(0.3636, 0.3636, 0.2727)):
    return {
        "name": name,
        "score_breakdown": {
            "rating": {"score": rating_s, "weight": w[0]},
            "experience": {"score": exp_s, "weight": w[1]},
            "location": {"score": loc_s, "weight": w[2]},
        },
    }


def test_contributions_reproduce_the_hand_verified_arithmetic():
    """Dr. An's +3.29 margin over Dr. Capampangan: experience +4.36 and
    location +3.00 outweigh rating −4.07. The critic told patients the opposite."""
    an = _score_contributions(_scored_provider("An", 83.2, 92.0, 94.6))
    cap = _score_contributions(_scored_provider("Cap", 94.4, 80.0, 83.6))

    delta = {d: round(an[d]["weighted_contribution"] - cap[d]["weighted_contribution"], 2)
             for d in an}
    assert delta["rating"] < 0, "An LOSES the rating dimension"
    assert delta["experience"] > 0 and delta["location"] > 0
    assert round(sum(delta.values()), 1) == 3.3


def test_contributions_are_absent_when_the_provider_was_never_scored():
    assert _score_contributions({}) == {}
    assert _score_contributions({"score_breakdown": "not a dict"}) == {}


@pytest.fixture
def critic():
    with patch("agents.critic_validator.Anthropic"):
        return CriticValidatorAgent()


def _bias_prompt(critic, providers):
    response = MagicMock()
    response.content[0].text = json.dumps({
        "bias_assessment": {"detected_biases": [], "severity": "low",
                            "explanation": "x", "technical_explanation": "y"}
    })
    critic.anthropic_client.messages.create.return_value = response
    critic._analyze_ranking_bias(providers, {"rating_weight": 0.36})
    return critic.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]


def test_bias_payload_carries_score_contributions(critic):
    """The correctness fix. Without this the analyst sees inputs and a total,
    and has to guess which dimension drove the ordering.

    Asserts on the computed VALUES, not the key name: "score_contributions"
    also appears in the SCORING MECHANICS prose, so its presence in the prompt
    proves nothing about the payload.
    """
    prompt = _bias_prompt(critic, [_scored_provider("Dr. An", 83.2, 92.0, 94.6)])

    ranking = json.loads(prompt.split("CURRENT TOP RANKINGS:")[1]
                               .split("SCORING MECHANICS")[0].strip())
    contributions = ranking[0]["score_contributions"]

    assert set(contributions) == {"rating", "experience", "location"}
    assert contributions["experience"]["weighted_contribution"] == 33.45
    assert contributions["rating"]["weighted_contribution"] == 30.25


def test_bias_prompt_forbids_inferring_causality_from_inputs(critic):
    prompt = _bias_prompt(critic, [_scored_provider("Dr. An", 83.2, 92.0, 94.6)])
    assert "MUST cite weighted_contribution" in prompt
    # And the specific false claim the run produced is pre-empted by name.
    assert "amplified beyond its nominal value" in prompt


# ---- Change 2: two registers, one finding ----

def test_bias_prompt_demands_a_patient_register_and_a_technical_one(critic):
    prompt = _bias_prompt(critic, [_scored_provider("Dr. An", 83.2, 92.0, 94.6)])
    assert "technical_explanation" in prompt
    assert "shown DIRECTLY TO PATIENTS" in prompt
    # The vocabulary the judge has been forbidden from using for rounds, now
    # forbidden here too — it is what leaked onto the panel.
    for banned in ("adjusted_rating", "ai_reasoning", "snake_case"):
        assert banned in prompt


# ---- Change 3: no more hardcoded key findings ----

def test_key_findings_is_no_longer_populated(critic):
    """Both entries were literal strings restating the tiles above them."""
    recs = critic._generate_final_recommendations(
        [{"name": "Dr. A"}],
        {"bias_assessment": {"detected_biases": ["something"], "severity": "medium"}},
        {"overall_ranking_validity": {"status": "validated"}},
    )
    assert recs["key_findings"] == []


# ---- the wrapper feeding critic_review["judge_findings"] and the WARNING log ----
#
# Distinct from is_judge_concern: this is the path that decides what a
# developer sees in the log and what refine_rankings carries per provider.
# It went untested until a revert-in-isolation run showed the predicate could
# be removed from it with every other test still green.

def test_wrapper_keeps_a_real_finding():
    from agents.critic_validator import _judge_finding_or_empty
    assert _judge_finding_or_empty(REAL_CONCERNS[0], "Dr. A") == REAL_CONCERNS[0]


@pytest.mark.parametrize("verdict", LIVE_PASS_VERDICTS)
def test_wrapper_drops_pass_verdicts(verdict):
    from agents.critic_validator import _judge_finding_or_empty
    assert _judge_finding_or_empty(verdict, "Dr. A") == ""


def test_wrapper_logs_what_it_drops(caplog):
    """Filtering must be visible: a prompt regression that starts producing
    concerns in unrecognized phrasing should show up in the log, not silently
    shrink the count."""
    from agents.critic_validator import _judge_finding_or_empty
    with caplog.at_level("INFO"):
        _judge_finding_or_empty(LIVE_PASS_VERDICTS[1], "Dr. Rabin")
    assert "Dropped a judge PASS verdict" in caplog.text
    assert "Dr. Rabin" in caplog.text


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_wrapper_passes_empty_through(empty):
    from agents.critic_validator import _judge_finding_or_empty
    assert _judge_finding_or_empty(empty, "Dr. A") == ""
