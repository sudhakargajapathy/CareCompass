"""The judge and the critic must review the SAME evidence.

Field test #5 exposed the failure this file exists to prevent. The scorer sent
the judge `review_summary[:400]` — a silent hard cut — while the critic was
sent the same field in full. Measured on the two live providers, the judge saw
55% of each summary, and the cut landed mid-clause at "However, practice-level
compla".

The consequences were not subtle, and none of them looked like a truncation bug
from the outside:

  * the judge scored practical_access in its NEUTRAL band ("no details on
    scheduling, wait times, or office responsiveness") for a provider whose
    full summary named long waits and scheduling problems;
  * a second provider scored 18/20 on access, blind to the "administrative
    challenges with scheduling and staff communication" sentence that had been
    cut off — the inflated case, which looks correct on the card;
  * the judge told the patient, in copy rendered verbatim on the card, that
    "the supplied summary begins to mention practice-level complaints without
    providing their details" — the truncation describing itself.

The direction is systematic, not random: the gatherer prompt asks for "most
praised aspects, common complaints, and overall experience themes" IN THAT
ORDER, so a head-cut always keeps the praise and discards the caveats. That
biases red_flags and practical_access — 50 of the judge's 100 rubric points,
and precisely the two criteria whose evidence lives in the tail.

A critic that reads different text than the agent it audits is not an
independent check. These tests compare the two payloads directly.
"""

import json

import pytest
from unittest.mock import MagicMock, patch

from agents.critic_validator import CriticValidatorAgent
from agents.preference_scorer import JUDGE_RUBRIC, PreferenceScorerAgent

# The real Dr. Rabin summary from the field test — 732 chars, praise first,
# caveats last. Any bound that trims THIS is the bug, not a safeguard.
LIVE_SUMMARY = (
    "Patients consistently praise Dr. Rabin as exceptionally caring, thorough, and "
    "compassionate, with many noting his ability to truly listen to concerns and explain "
    "conditions in detail. Reviewers highlight his intelligence, determination to "
    "investigate root causes, and respect for patient input, with several describing him "
    "as the best neurologist they have encountered. However, practice-level complaints "
    "emerge regarding scheduling difficulties, long wait times, and administrative issues "
    "(including an office policy restricting patient access to MRI results), though these "
    "are attributed to staff and office management rather than Dr. Rabin himself. One "
    "reviewer noted concerns about punctuality policies affecting access to care."
)

# The sentence the old [:400] cut discarded — the entire reason the two agents
# reached different conclusions about the same provider.
CUT_TAIL = "long wait times"

PROVIDER = {
    "name": "Dr. Brian Rabin, MD",
    "specialty": "Neurology",
    "location": "Chandler, AZ 85224",
    "rating": 4.7,
    "review_count": 30,
    "review_summary": LIVE_SUMMARY,
    "review_sentiment": "positive",
    "review_source_url": "https://www.vitals.com/doctors/Dr_Brian_Rabin.html",
    "blended_rating": 4.8,
    "blended_review_count": 73,
    "blended_platform_count": 3,
    "years_experience": 22,
    "final_score": 90,
    "ai_reasoning": "Independent reviews consistently describe caring, thorough visits.",
    "ai_rubric": {"review_substance": 46, "red_flags": 27, "practical_access": 10},
    "ai_evidence": {"review_substance": "Patients consistently praise Dr. Rabin"},
    "ai_strengths": ["Thorough and compassionate care"],
    "ai_concerns": ["Practice-level complaints"],
    "review_observations": [
        {"source_url": "https://healthgrades.com/x", "rating": 4.6, "review_count": 19},
    ],
}


def _payload_after(prompt: str, marker: str) -> list:
    """The provider array a prompt embeds after `marker`.

    Decoded with raw_decode rather than a regex: both prompts also contain an
    illustrative output-format array further down, and a greedy match spans
    straight through the provider data into it.
    """
    start = prompt.index(marker) + len(marker)
    start = prompt.index("[", start)
    payload, _ = json.JSONDecoder().raw_decode(prompt[start:])
    return payload


def _judge_payload_from(prompt: str) -> list:
    return _payload_after(prompt, "<providers_data>")


def _critic_payload_from(prompt: str) -> list:
    return _payload_after(prompt, "TOP PROVIDERS TO VALIDATE:")


@pytest.fixture
def judge_payload():
    with patch.object(PreferenceScorerAgent, "_initialize_client", return_value=None):
        agent = PreferenceScorerAgent()
        agent.openai_client = MagicMock()
        # Judge output is irrelevant here; we assert on what went IN.
        agent.openai_client.chat.completions.create.return_value.choices[0].message.content = "[]"
        agent._generate_ai_rankings([dict(PROVIDER)], {})
        prompt = agent.openai_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    return _judge_payload_from(prompt)[0]


@pytest.fixture
def critic_payload():
    with patch.object(CriticValidatorAgent, "_initialize_client", return_value=None):
        agent = CriticValidatorAgent()
        agent.anthropic_client = MagicMock()
        agent.anthropic_client.messages.create.return_value.content[0].text = "{}"
        agent._validate_top_recommendations([dict(PROVIDER)])
        prompt = agent.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    return _critic_payload_from(prompt)[0]


def test_judge_receives_the_whole_summary(judge_payload):
    """The regression guard that did not exist: nothing in the suite asserted
    on the [:400] cap, so it survived four field-test rounds."""
    assert judge_payload["review_summary"] == LIVE_SUMMARY
    assert CUT_TAIL in judge_payload["review_summary"]


def test_critic_receives_the_whole_summary(critic_payload):
    assert critic_payload["review_summary"] == LIVE_SUMMARY


def test_both_agents_are_handed_byte_identical_summaries(judge_payload, critic_payload):
    """The contract itself. Either side may change its bound — but only by
    changing the shared one, which moves both."""
    assert judge_payload["review_summary"] == critic_payload["review_summary"]


def test_missing_summary_reads_the_same_to_both():
    """The judge used to default to "" while the critic got "No reviews
    available". The judge's rubric has an explicit neutral band for "no review
    text available"; an empty string is a weaker signal for that band than the
    sentence saying so, so the two disagreed on absent evidence too."""
    bare = {"name": "Dr. Unreviewed", "rating": 0}

    with patch.object(PreferenceScorerAgent, "_initialize_client", return_value=None):
        scorer = PreferenceScorerAgent()
        scorer.openai_client = MagicMock()
        scorer.openai_client.chat.completions.create.return_value.choices[0].message.content = "[]"
        scorer._generate_ai_rankings([dict(bare)], {})
        judge = _judge_payload_from(
            scorer.openai_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        )[0]

    with patch.object(CriticValidatorAgent, "_initialize_client", return_value=None):
        critic = CriticValidatorAgent()
        critic.anthropic_client = MagicMock()
        critic.anthropic_client.messages.create.return_value.content[0].text = "{}"
        critic._validate_top_recommendations([dict(bare)])
        critic_seen = _critic_payload_from(
            critic.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
        )[0]

    assert judge["review_summary"] == critic_seen["review_summary"] == "No reviews available"


def test_both_agents_see_the_same_review_provenance(judge_payload, critic_payload):
    """The judge has a source-credibility rule keyed on review_source (cap the
    top band when the text is self-published marketing). The critic could not
    check that verdict without the same field."""
    assert judge_payload["review_source"] == critic_payload["review_source"] == "vitals.com"


def test_neither_agent_receives_insurance(judge_payload, critic_payload):
    """Insurance is verification-only — the sidebar FHIR network check.

    The judge was denied it deliberately from the start; the critic was not, and
    the asymmetry was documented in prose rather than tested. Live, it produced
    a card that contradicted itself inside one sentence: the critic's
    patient_considerations ASSERTED coverage ("Located 6.5 mi away; accepts
    Aetna, Cigna, Humana, Medicare") and our own copy immediately re-presented
    the same scraped list as unverified with "verify coverage with the provider
    directly".

    Nothing in the critic's checklist, rubric or verdict criteria ever read the
    field — it was load-bearing for nothing and asserted for everything."""
    for payload in (judge_payload, critic_payload):
        assert "insurance" not in payload
        assert "insurance_accepted" not in payload


def test_the_critic_prompt_forbids_coverage_claims():
    """Belt and braces: the field is gone AND the instruction is explicit.

    The prompt-only half of this pattern is exactly what `is_judge_concern` had
    to be built to backstop — the critic wrote PASS verdicts into a field the
    prompt asked it to leave empty, on 9 of 10 providers."""
    with patch.object(CriticValidatorAgent, '_initialize_client', return_value=None):
        agent = CriticValidatorAgent()
        agent.anthropic_client = MagicMock()
        response = MagicMock()
        response.content[0].text = "{}"
        agent.anthropic_client.messages.create.return_value = response
        agent._validate_top_recommendations(
            [{"name": "Dr. Alpha", "insurance_accepted": ["Aetna", "Cigna"]}]
        )

    prompt = agent.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "NO insurance or plan data" in prompt
    assert "Aetna" not in prompt, "the payer list must not ride in via the payload"


# ---- parity of the STANDARD, not just the evidence ----

@pytest.fixture
def judge_prompt():
    with patch.object(PreferenceScorerAgent, "_initialize_client", return_value=None):
        agent = PreferenceScorerAgent()
        agent.openai_client = MagicMock()
        agent.openai_client.chat.completions.create.return_value.choices[0].message.content = "[]"
        agent._generate_ai_rankings([dict(PROVIDER)], {})
        return agent.openai_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]


@pytest.fixture
def critic_prompt():
    with patch.object(CriticValidatorAgent, "_initialize_client", return_value=None):
        agent = CriticValidatorAgent()
        agent.anthropic_client = MagicMock()
        agent.anthropic_client.messages.create.return_value.content[0].text = "{}"
        agent._validate_top_recommendations([dict(PROVIDER)])
        return agent.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]


def test_the_critic_audits_against_the_judges_actual_rubric(judge_prompt, critic_prompt):
    """Payload parity applied to the STANDARD rather than the evidence.

    The critic audits the judge's per-criterion scores but was given only the
    criterion names and maxima plus the phrase "neutral band" — so it audited
    against a standard it had to infer. On 2026-07-28 it published this to the
    patient as a real inconsistency:

        "practical_access at 2.0 ... is not a neutral-band error, but red_flags
         at 18.0 is generous given the summary cites both incomplete workups
         and REPEATED DELAYS"

    The judge had scored the delays under practical_access and correctly not
    charged them again under red_flags — which is what the rubric's own routing
    rule requires. The critic was asking for the double-charge, because it had
    never been shown the rule.

    Asserts the same TEXT reaches both. A paraphrase in the critic's prompt is
    exactly the drift this prevents.
    """
    assert JUDGE_RUBRIC in judge_prompt
    assert JUDGE_RUBRIC in critic_prompt, (
        "the critic must receive the judge's rubric verbatim, not a summary of it"
    )


def test_the_critic_is_told_the_routing_rules_bind_its_audit(critic_prompt):
    """Handing over the rubric is necessary and not sufficient. Without being
    told the rules constrain what counts as an ERROR, the critic reads them as
    background and still calls a compliant score wrong — which is the shape of
    the finding that shipped."""
    assert "A score is only an error if the RUBRIC ABOVE says so" in critic_prompt
    assert "must NOT also be charged to red_flags" in critic_prompt
