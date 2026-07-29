"""Unit tests for the PreferenceScorerAgent."""

import re

import pytest
from unittest.mock import MagicMock, patch
from agents.preference_scorer import (
    EXPERIENCE_KNEE_RATE,
    EXPERIENCE_POST_KNEE_RATE,
    EXPERIENCE_CAP,
    EXPERIENCE_UNKNOWN_SCORE,
    EXPERIENCE_UNKNOWN_EQUIV_YEARS,
    RATING_PRIOR,
    RATING_UNKNOWN_SCORE,
    PreferenceScorerAgent,
    _EVIDENCE_MAX_CHARS,
    _clip_evidence,
    _core_rank_order,
    calculate_experience_score,
    calculate_rating_score_with_confidence,
    interpret_rating_status,
)
from tests.fixtures.mock_agent_responses import MOCK_GATHER_PROVIDERS_RESULT, MOCK_OPENAI_RESPONSE

@pytest.fixture
def preference_scorer():
    """Fixture to create a PreferenceScorerAgent with a mocked OpenAI client."""
    with patch.object(PreferenceScorerAgent, '_initialize_client', return_value=None):
        agent = PreferenceScorerAgent()
        agent.openai_client = MagicMock()
        return agent

def test_calculate_rating_score_with_confidence():
    """Test the rating score calculation logic."""
    # Test high confidence
    result = calculate_rating_score_with_confidence(rating=4.5, review_count=100)
    assert result['score'] > 80
    assert result['confidence'] == 'high'

    # Test low confidence
    result = calculate_rating_score_with_confidence(rating=4.5, review_count=3)
    assert result['confidence'] == 'low'

    # Unrated is imputed at the prior, not zeroed
    result = calculate_rating_score_with_confidence(rating=0, review_count=0)
    assert result['score'] == RATING_UNKNOWN_SCORE
    assert result['confidence'] == 'no_rating'


def test_unrated_is_not_scored_below_a_measured_bad_provider():
    """The old imputation, 40, equalled a MEASURED 2.0 stars — below the 2.5
    `interpret_rating_status` itself calls 'poor_quality'. A provider nobody had
    looked at scored worse than one measured as bad, and that score then decided
    who got enriched, which is what produces ratings in the first place."""
    unrated = interpret_rating_status(0, 0)["base_score"]

    # Every rating the function is willing to call poor must score below it.
    assert unrated > interpret_rating_status(2.0, 10)["base_score"]
    assert unrated > interpret_rating_status(2.4, 10)["base_score"]

    # And the specific number that used to collide with it.
    assert interpret_rating_status(2.0, 10)["base_score"] == 40.0
    assert unrated != 40.0


def test_measured_rating_above_the_prior_still_wins():
    """The imputation must not reward absence: supplying real data remains the
    way to score well, exactly as tenure and the location tiers work."""
    unrated = interpret_rating_status(0, 0)["base_score"]
    assert interpret_rating_status(4.5, 100)["base_score"] > unrated
    assert interpret_rating_status(3.6, 100)["base_score"] > unrated


def test_the_stated_rating_equivalence_is_true():
    """The constant and its documented meaning must not drift apart."""
    assert RATING_UNKNOWN_SCORE == (RATING_PRIOR / 5.0) * 100
    assert interpret_rating_status(0, 0)["base_score"] == RATING_UNKNOWN_SCORE
    # The prior the imputation claims to be is the one the shrink actually uses.
    heavily_shrunk = calculate_rating_score_with_confidence(rating=5.0, review_count=0)
    assert heavily_shrunk["adjusted_rating"] == RATING_PRIOR

@pytest.mark.parametrize(
    "years,expected",
    [(0, 55.0), (5, 62.5), (10, 70.0), (15, 77.5), (20, 80.0), (30, 85.0), (45, 85.0)],
)
def test_experience_curve_rewards_long_careers(years, expected):
    """Round 16 flattened this ramp (floor 40 -> 55, rates 2.0/1.0 -> 1.5/0.5).
    It still rewards a long career; it no longer out-swings measured patient
    ratings ~3x at equal weight — see the constants for the three runs of field
    evidence and the span arithmetic."""
    result = calculate_experience_score(years)
    assert result['score'] == expected
    assert result['data_quality'] == 'known'


def test_experience_no_longer_out_swings_measured_ratings():
    """The defect, stated as the arithmetic that produced it.

    Equal WEIGHTS did not give equal INFLUENCE: across a realistic pool the
    rating dimension realizes ~8.5 points (4.0-4.5 stars, doubly compressed by
    Bayesian shrinkage and by real doctors clustering high) while experience
    realized 25 (10-30 years, an unshrunk ramp on one scraped integer). The
    critic named it in three consecutive live runs.

    Asserts the SPAN RATIO, not the endpoints, because that is the property
    that was wrong. Deliberately not parity — a 30-year career is genuinely
    more informative than half a star, and closing the gap entirely would need
    a prior over career length nobody has measured."""
    exp_span = calculate_experience_score(30)["score"] - calculate_experience_score(10)["score"]
    rating_span = (
        calculate_rating_score_with_confidence(4.5, 60)["score"]
        - calculate_rating_score_with_confidence(4.0, 60)["score"]
    )

    assert exp_span == 15.0, "10-30 years must span ~15 points, not the old 25"
    assert exp_span / rating_span < 2.0, (
        f"experience still carries {exp_span / rating_span:.1f}x rating's leverage"
    )

def test_experience_missing_is_neutral_not_zero():
    for value in (None, "unknown", "15 years"):
        result = calculate_experience_score(value)
        assert result['score'] == EXPERIENCE_UNKNOWN_SCORE
        assert result['data_quality'] == 'missing'
        assert result['warning']


def test_unknown_tenure_is_not_scored_as_a_novice():
    """The old imputation, 45, equalled a MEASURED 2.5 years — so a provider
    whose tenure our extractor missed was ranked as more junior than almost any
    practising physician. Years come from excerpt anchors hitting a profile
    header, so absence reflects our extraction coverage, not their career."""
    unknown = calculate_experience_score(None)["score"]
    assert unknown == calculate_experience_score(EXPERIENCE_UNKNOWN_EQUIV_YEARS)["score"]
    assert unknown > calculate_experience_score(2.5)["score"]
    assert unknown > calculate_experience_score(5)["score"]


def test_measured_tenure_above_the_equivalence_still_wins():
    """The imputation is the pessimistic edge, not the median: supplying real
    data must remain the way to score well, exactly as the location tiers work
    (same_city 82 == a verified ~9 mi)."""
    unknown = calculate_experience_score(None)["score"]
    assert calculate_experience_score(11)["score"] > unknown
    assert calculate_experience_score(14)["score"] > unknown
    assert calculate_experience_score(26)["score"] > unknown


def test_the_stated_equivalence_is_true():
    """The constant and its documented meaning must not drift apart.

    Asserted against the CURVE, not a hardcoded formula: round 10 reshaped the
    curve and a formula-shaped assertion would have had to be edited in the same
    commit, which is exactly when it stops being a guard."""
    assert EXPERIENCE_UNKNOWN_SCORE == calculate_experience_score(
        EXPERIENCE_UNKNOWN_EQUIV_YEARS
    )["score"]


def test_tenure_cannot_out_score_measured_reviews():
    """Experience has no shrinkage while every rating is pulled toward the
    prior, so an UNVERIFIED career length used to out-swing measured patient
    experience — 96 on tenure against 73 on rating for the top provider of the
    2026-07-25 run, which the critic's bias analysis flagged twice.

    The cap must sit below what a strong measured rating reaches. Under the old
    ceiling of 100 it did not, and no rating could ever catch a long career."""
    best_possible_tenure = calculate_experience_score(50)["score"]
    strong_measured_rating = calculate_rating_score_with_confidence(4.75, 100)["score"]
    assert best_possible_tenure == EXPERIENCE_CAP
    assert best_possible_tenure < strong_measured_rating


def test_the_curve_flattens_where_the_evidence_stops_discriminating():
    """A scraped profile header cannot support 30 points of difference between a
    20-year and a 35-year physician. Past the knee a year is worth strictly
    less — a third, since round 16 flattened the ramp to 1.5/0.5.

    The knee's MEANING is "the marginal year stops discriminating", so that is
    what is asserted: the rates match their constants, and the post-knee rate
    is strictly smaller. An earlier version of this test asserted `below ==
    above * 2`, which was a property of the OLD 2.0/1.0 rates dressed up as
    the general rule — it failed the moment the rates moved, correctly."""
    below_knee = calculate_experience_score(11)["score"] - calculate_experience_score(10)["score"]
    above_knee = calculate_experience_score(21)["score"] - calculate_experience_score(20)["score"]
    assert below_knee == pytest.approx(EXPERIENCE_KNEE_RATE)
    assert above_knee == pytest.approx(EXPERIENCE_POST_KNEE_RATE)
    assert above_knee < below_knee, "past the knee a year must be worth less"

    # The top-end swing keeps shrinking: 15->30 was 30 points at round 7,
    # 15 at round 14, 7.5 now.
    assert calculate_experience_score(30)["score"] - calculate_experience_score(15)["score"] == 7.5

def test_core_rank_order_breaks_ties_deterministically():
    """Exact ties are ordinary — the 2026-07-25 pool had 53/53 and 50/50
    adjacent — and this ranking decides who gets the enrichment budget. Under
    `sorted`'s input-order fallback an identical search could research a
    different set depending on what order discovery happened to return."""
    alpha = {"name": "Dr. Alpha", "location": "Chandler, AZ", "base_score": 53.0}
    beta = {"name": "Dr. Beta", "location": "Chandler, AZ", "base_score": 53.0}

    forward = [[alpha, beta][i]["name"] for i in _core_rank_order([alpha, beta])]
    reverse = [[beta, alpha][i]["name"] for i in _core_rank_order([beta, alpha])]
    assert forward == reverse

    # Score still dominates the tie-break.
    beta_better = dict(beta, base_score=53.1)
    order = _core_rank_order([alpha, beta_better])
    assert [alpha, beta_better][order[0]]["name"] == "Dr. Beta"


def test_core_ranking_is_stable_across_shuffled_input(preference_scorer: PreferenceScorerAgent):
    """End-to-end version of the above, through the real scoring path."""
    providers = [
        {"name": "Dr. Alpha", "location": "Chandler, AZ", "rating": 0, "review_count": 0},
        {"name": "Dr. Beta", "location": "Chandler, AZ", "rating": 0, "review_count": 0},
        {"name": "Dr. Gamma", "location": "Chandler, AZ", "rating": 0, "review_count": 0},
    ]
    prefs = {"rating_weight": 0.34, "location_weight": 0.33, "experience_weight": 0.33}

    baseline = [p["name"] for p in preference_scorer.score_core(providers, prefs)]
    for shuffled in ([providers[2], providers[0], providers[1]],
                     [providers[1], providers[2], providers[0]]):
        assert [p["name"] for p in preference_scorer.score_core(shuffled, prefs)] == baseline


def test_judge_count_pins_the_researched_set(preference_scorer: PreferenceScorerAgent):
    """The judge must receive exactly the providers enrichment was spent on.

    Positional rather than re-derived: enrichment backfills ratings and moves
    core scores, so a set recomputed at judging time would not be the set that
    was actually researched."""
    providers = [{"name": f"P{i}", "location": "Chandler, AZ"} for i in range(5)]
    seen = {}

    def capture(scored, prefs):
        seen["names"] = [p["name"] for p in scored]
        return scored

    with patch.object(preference_scorer, "_generate_ai_rankings", side_effect=capture):
        result = preference_scorer.score_providers(providers, {"rating_weight": 1.0}, judge_count=3)

    assert seen["names"] == ["P0", "P1", "P2"]

    ranked = {p["name"]: p for p in result["ranked_providers"]}
    assert len(ranked) == 5, "providers past the cut are still scored and listed"
    assert all(ranked[f"P{i}"].get("ai_judged") is not False for i in range(3))
    assert all(ranked[f"P{i}"]["ai_judged"] is False for i in (3, 4))


def test_judge_count_none_judges_everyone(preference_scorer: PreferenceScorerAgent):
    """The parameter is opt-in: direct callers and tests keep the old behavior."""
    providers = [{"name": f"P{i}", "location": "Chandler, AZ"} for i in range(4)]
    seen = {}

    def capture(scored, prefs):
        seen["count"] = len(scored)
        return scored

    with patch.object(preference_scorer, "_generate_ai_rankings", side_effect=capture):
        result = preference_scorer.score_providers(providers, {"rating_weight": 1.0})

    assert seen["count"] == 4
    assert not any(p.get("ai_judged") is False for p in result["ranked_providers"])


def test_calculate_base_scores(preference_scorer: PreferenceScorerAgent):
    """Base scores use rating/location/experience — insurance is not scored."""
    providers = MOCK_GATHER_PROVIDERS_RESULT['providers']
    preferences = {"rating_weight": 0.5, "location_weight": 0.3, "experience_weight": 0.2}

    scored_providers = preference_scorer._calculate_base_scores(providers, preferences)

    assert len(scored_providers) == 2
    assert "base_score" in scored_providers[0]
    assert scored_providers[0]['base_score'] > 0
    breakdown = scored_providers[0]['score_breakdown']
    assert set(breakdown.keys()) == {"rating", "location", "experience"}
    assert "insurance" not in breakdown
    # Weighted core of 0-100 components stays within 0-100
    assert 0 <= scored_providers[0]['base_score'] <= 100

def test_rating_uses_cross_platform_blend(preference_scorer: PreferenceScorerAgent):
    """The score hears ALL platforms: when the gatherer stored a
    count-weighted blend, it replaces the headline as the rating input
    (basis recorded) — a 3.5 vitals headline with a 2.1 healthgrades beside
    it must not score as a 3.5 doctor."""
    provider = {
        "name": "Dr. Blend", "location": "Chandler, AZ",
        "rating": 3.5, "review_count": 16,                 # headline (vitals)
        "blended_rating": 2.9, "blended_review_count": 30,
        "blended_platform_count": 3,
    }
    scored = preference_scorer._calculate_base_scores(
        [provider],
        {"rating_weight": 1.0, "location_weight": 0.0, "experience_weight": 0.0},
    )
    rating_bd = scored[0]["score_breakdown"]["rating"]
    assert rating_bd["basis"] == "cross_platform_blend"
    assert rating_bd["value"] == 2.9
    assert rating_bd["review_count"] == 30
    assert rating_bd["platforms"] == 3
    # Bayesian over the blend: (10*3.5 + 30*2.9)/40 = 3.05 -> 61/100
    assert rating_bd["score"] == pytest.approx(61.0, abs=0.5)

def test_rating_headline_when_no_blend(preference_scorer: PreferenceScorerAgent):
    """No blend fields -> exactly today's headline math, basis 'headline'."""
    provider = {"name": "Dr. Single", "rating": 4.0, "review_count": 20}
    scored = preference_scorer._calculate_base_scores(
        [provider],
        {"rating_weight": 1.0, "location_weight": 0.0, "experience_weight": 0.0},
    )
    rating_bd = scored[0]["score_breakdown"]["rating"]
    assert rating_bd["basis"] == "headline"
    assert "platforms" not in rating_bd
    assert rating_bd["value"] == 4.0

@patch('agents.preference_scorer.PreferenceScorerAgent._generate_ai_rankings')
def test_score_providers_success(mock_ai_rankings, preference_scorer: PreferenceScorerAgent):
    """Test the main score_providers method for a successful run."""

    # Let AI ranking return providers in the same order
    mock_ai_rankings.side_effect = lambda providers, prefs: providers

    providers = MOCK_GATHER_PROVIDERS_RESULT['providers']
    preferences = {"rating_weight": 0.5, "location_weight": 0.3, "experience_weight": 0.2}

    result = preference_scorer.score_providers(providers, preferences)

    assert result['status'] == 'success'
    assert len(result['ranked_providers']) == 2
    assert "final_score" in result['ranked_providers'][0]
    assert result['ranked_providers'][0]['final_rank'] == 1
    # Composite of two 0-100 parts is a true percentage
    assert all(0 <= p['final_score'] <= 100 for p in result['ranked_providers'])

def test_generate_ai_rankings_rubric(preference_scorer: PreferenceScorerAgent):
    """The rubric judge yields clamped subscores summed into ai_score."""

    mock_response = MagicMock()
    mock_response.choices[0].message.content = MOCK_OPENAI_RESPONSE
    preference_scorer.openai_client.chat.completions.create.return_value = mock_response

    providers = preference_scorer._calculate_base_scores(MOCK_GATHER_PROVIDERS_RESULT['providers'], {})

    ranked_providers = preference_scorer._generate_ai_rankings(providers, {})

    assert len(ranked_providers) == 2
    first = ranked_providers[0]
    assert first['ai_score'] == 84.0  # 42 + 27 + 15
    assert first['ai_rubric'] == {
        "review_substance": 42, "red_flags": 27, "practical_access": 15
    }
    assert "listening skills" in first['ai_evidence']['review_substance']
    assert "Excellent ratings" in first['ai_strengths']
    assert ranked_providers[1]['ai_score'] == 69.0


def test_clip_evidence_leaves_ordinary_citations_alone():
    quote = 'one patient called him "a very compassionate and skilled doctor"'
    assert _clip_evidence(quote) == quote


def test_clip_evidence_breaks_on_a_word_boundary():
    """A blind slice ended a live card mid-word at `who "take` — never again."""
    long_quote = "Patients consistently praise this physician " * 20
    clipped = _clip_evidence(long_quote)

    assert len(clipped) <= _EVIDENCE_MAX_CHARS
    assert clipped.endswith(" …")
    # The character before the ellipsis closes a whole word, not a fragment
    assert clipped[:-2].rstrip().split()[-1] in long_quote.split()

def test_generate_ai_rankings_skipped_provider_scores_neutral(preference_scorer: PreferenceScorerAgent):
    """A provider the judge omits gets the neutral 50, never a punitive score."""

    mock_response = MagicMock()
    # Judge only returns provider 0; subscores exceed caps to test clamping
    mock_response.choices[0].message.content = (
        '[{"provider_index": 0, '
        '"scores": {"review_substance": 60, "red_flags": 35, "practical_access": 25}, '
        '"reasoning": "r", "strengths": [], "concerns": []}]'
    )
    preference_scorer.openai_client.chat.completions.create.return_value = mock_response

    providers = preference_scorer._calculate_base_scores(MOCK_GATHER_PROVIDERS_RESULT['providers'], {})
    ranked_providers = preference_scorer._generate_ai_rankings(providers, {})

    # Clamped to criterion caps: 50 + 30 + 20 = 100
    assert ranked_providers[0]['ai_score'] == 100.0
    assert ranked_providers[1]['ai_score'] == 50.0

@patch('agents.preference_scorer.PreferenceScorerAgent._calculate_base_scores')
@patch('agents.preference_scorer.PreferenceScorerAgent._generate_ai_rankings')
def test_composite_gives_judge_a_true_30_points(mock_ai, mock_base, preference_scorer):
    """Equal base scores + ai_scores 0 and 100 => exactly 30 final points apart."""
    providers = [
        {"name": "A", "base_score": 80.0, "ai_score": 100.0},
        {"name": "B", "base_score": 80.0, "ai_score": 0.0},
    ]
    mock_base.side_effect = lambda p, prefs: providers
    mock_ai.side_effect = lambda p, prefs: providers

    result = preference_scorer.score_providers([{"name": "A"}, {"name": "B"}], {})
    scores = {p['name']: p['final_score'] for p in result['ranked_providers']}
    assert scores['A'] - scores['B'] == 30.0
    assert result['ranked_providers'][0]['name'] == 'A'

def test_score_providers_no_providers(preference_scorer: PreferenceScorerAgent):
    """Test that score_providers handles an empty list of providers."""

    result = preference_scorer.score_providers([], {})

    assert result['status'] == 'no_providers'
    assert len(result['ranked_providers']) == 0

def test_location_score_prefers_computed_distance(preference_scorer: PreferenceScorerAgent):
    """Precedence: code-computed haversine beats a page-stated distance and tiers."""

    providers = [{
        "name": "Dr. A", "rating": 4.0, "review_count": 50,
        "computed_distance_miles": 5.0, "distance": 40.0, "location_match": "same_city",
    }]
    scored = preference_scorer._calculate_base_scores(providers, {"location_weight": 0.4})

    loc = scored[0]["score_breakdown"]["location"]
    assert loc["basis"] == "computed_distance"
    assert loc["score"] == 90.0  # 100 - 5/50*100 at the default 25-mile radius
    assert loc["data_quality"] == "complete"

def test_location_score_falls_back_to_stated_then_tiers(preference_scorer: PreferenceScorerAgent):
    """Without a computed distance: page-stated distance, then honest tiers."""

    prefs = {"location_weight": 0.4}

    stated = preference_scorer._calculate_base_scores(
        [{"name": "B", "distance": 10.0}], prefs
    )[0]["score_breakdown"]["location"]
    assert stated["basis"] == "stated_distance"
    assert stated["score"] == 80.0

    for tier, expected in (("same_zip", 90), ("same_city", 82), ("same_state", 55), ("different", 25)):
        loc = preference_scorer._calculate_base_scores(
            [{"name": "C", "location_match": tier}], prefs
        )[0]["score_breakdown"]["location"]
        assert loc["basis"] == tier
        assert loc["score"] == expected
        assert loc["data_quality"] == "derived"

def test_location_score_missing_path_unchanged(preference_scorer: PreferenceScorerAgent):
    """No distance, no tier: the weight-sensitive missing-data penalty applies."""

    loc = preference_scorer._calculate_base_scores(
        [{"name": "D", "location_match": "unknown"}], {"location_weight": 0.4}
    )[0]["score_breakdown"]["location"]
    assert loc["basis"] == "missing"
    assert loc["score"] == 35  # medium-priority missing data: 0.7 * 50

def test_location_falloff_uses_configured_radius(preference_scorer: PreferenceScorerAgent, monkeypatch):
    """DEFAULT_SEARCH_RADIUS drives the distance falloff (zero at 2x radius)."""

    monkeypatch.setattr(preference_scorer.config, "DEFAULT_SEARCH_RADIUS", 10)
    loc = preference_scorer._calculate_base_scores(
        [{"name": "E", "computed_distance_miles": 10.0}], {"location_weight": 0.4}
    )[0]["score_breakdown"]["location"]
    assert loc["score"] == 50.0  # 100 - 10/(2*10)*100

def test_verified_distance_never_loses_to_same_city_tier(preference_scorer: PreferenceScorerAgent):
    """The load-bearing invariant behind the tier recalibration: a MEASURED
    in-radius distance must score at least as high as the same_city tier
    imputation. Otherwise a provider we located precisely could rank below one
    we only guessed at — the 'more data scores worse' inversion the round-3
    distance fix removes."""
    prefs = {"location_weight": 0.4}
    measured = preference_scorer._calculate_base_scores(
        [{"name": "M", "computed_distance_miles": 5.0}], prefs
    )[0]["score_breakdown"]["location"]["score"]
    tier = preference_scorer._calculate_base_scores(
        [{"name": "T", "location_match": "same_city"}], prefs
    )[0]["score_breakdown"]["location"]["score"]
    assert measured >= tier          # 90 (verified ~5 mi) >= 82 (same_city tier)

def test_score_core_orders_originals_without_llm(preference_scorer: PreferenceScorerAgent):
    """score_core ranks the ORIGINAL dicts (for in-place enrichment), no LLM call."""

    providers = [
        {"name": "Weak", "rating": 2.0, "review_count": 50},
        {"name": "Strong", "rating": 5.0, "review_count": 200},
    ]
    ranked = preference_scorer.score_core(providers, {"rating_weight": 1.0})

    assert [p["name"] for p in ranked] == ["Strong", "Weak"]
    assert ranked[0] is providers[1]
    preference_scorer.openai_client.chat.completions.create.assert_not_called()
    assert preference_scorer.score_core([], {}) == []

def test_judge_prompt_carries_three_band_rubric_and_plain_language(preference_scorer: PreferenceScorerAgent):
    """The evidence-only rubric: three bands with neutral floors, no dead
    user-input criteria, sources visible, patient-facing language rule."""

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "[]"
    preference_scorer.openai_client.chat.completions.create.return_value = mock_response

    providers = [{
        "name": "Dr. Cited", "rating": 4.5, "review_count": 61,
        "review_summary": "Patients praise her thoroughness.",
        "review_source_url": "https://www.healthgrades.com/physician/dr-cited",
    }]
    preference_scorer._generate_ai_rankings(providers, {})

    call_kwargs = preference_scorer.openai_client.chat.completions.create.call_args.kwargs
    prompt = str(call_kwargs.get("messages"))
    assert "healthgrades.com" in prompt              # source domain reached the judge
    assert "review_substance (0-50)" in prompt
    assert "red_flags (0-30)" in prompt
    assert "practical_access (0-20)" in prompt
    # Absence must never be scored as a problem — both neutral floors present
    assert prompt.count("ever score absence below this band") == 2
    # Dead user-input criteria are gone from the prompt entirely
    assert "requirements_fit" not in prompt
    assert "insurance_access" not in prompt
    assert "sum of the three criterion scores" in prompt
    assert "NEVER use internal criterion names" in prompt
    # Location is scored once, in the user-weighted deterministic core —
    # the access band must never double-count proximity
    assert "NEVER score it here" in prompt


# ---- Round 11: the rubric's bands must tile their range ----


def _rubric_bands(preference_scorer: PreferenceScorerAgent) -> dict:
    """Parse the <rubric> block out of the prompt the judge actually receives.

    Parsed from the live prompt string rather than a parallel table, because the
    failure mode is someone editing the prompt text directly. A table would
    agree with itself while the model saw something else."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "[]"
    preference_scorer.openai_client.chat.completions.create.return_value = mock_response
    preference_scorer._generate_ai_rankings([{"name": "Dr. X"}], {})

    prompt = str(
        preference_scorer.openai_client.chat.completions.create.call_args.kwargs["messages"]
    )
    block = prompt.split("<rubric>")[1].split("</rubric>")[0]

    bands, current = {}, None
    for line in block.replace("\\n", "\n").splitlines():
        stripped = line.strip()
        criterion = re.match(r"\d+\.\s+(\w+)\s+\(0-(\d+)\)", stripped)
        if criterion:
            current = criterion.group(1)
            bands[current] = {"max": int(criterion.group(2)), "ranges": []}
            continue
        span = re.match(r"(\d+)-(\d+):", stripped)
        if span and current:
            bands[current]["ranges"].append((int(span.group(1)), int(span.group(2))))
    return bands


@pytest.mark.parametrize("criterion", ["review_substance", "red_flags", "practical_access"])
def test_every_score_lands_on_exactly_one_band(preference_scorer: PreferenceScorerAgent, criterion):
    """A gap between anchors is where the model improvises.

    Rabin's practical_access had no band for "two isolated access mentions":
    0-6 said REPEATED (scoring there would defame him), 8-11 said "no signals
    either way" (false — the summary named wait times), and 12-16 did not exist.
    He was parked at the neutral 10, which the scoring rules then required him
    to label "no evidence" — while the same sentence was quoted under red_flags
    on the same card. The 2026-07-25 run scored the same provider 5 for the same
    class of evidence; an un-anchored region is not reproducible."""
    spec = _rubric_bands(preference_scorer)[criterion]

    for score in range(0, spec["max"] + 1):
        covering = [(lo, hi) for lo, hi in spec["ranges"] if lo <= score <= hi]
        assert len(covering) == 1, (
            f"{criterion} score {score} matches {len(covering)} bands {covering} — "
            f"a gap is where the judge improvises, an overlap is ambiguous"
        )


@pytest.mark.parametrize("criterion,expected_low", [
    ("review_substance", 0), ("red_flags", 0), ("practical_access", 0),
])
def test_bands_start_at_zero_and_reach_the_cap(preference_scorer: PreferenceScorerAgent,
                                               criterion, expected_low):
    spec = _rubric_bands(preference_scorer)[criterion]
    assert min(lo for lo, _ in spec["ranges"]) == expected_low
    assert max(hi for _, hi in spec["ranges"]) == spec["max"]


@pytest.mark.parametrize("score,must_contain", [
    (3, "repeated access complaints"),        # De Lima, Lockwood — 3-month wait, phone/auth
    (5, "isolated access friction"),          # Rabin on 2026-07-25 — wait-time complaints
    (10, "no access signals either way"),     # genuine silence
    (12, "MIXED"),                            # An 2026-07-28 — timely appts AND phone trouble
    (14, "mild or incidental access"),        # De Lima 2026-07-28 — "spend time with patients"
    (17, "concrete access positives"),        # Pandey — telehealth availability
])
def test_observed_scores_land_on_an_anchor_that_describes_them(
    preference_scorer: PreferenceScorerAgent, score, must_contain
):
    """Calibrated against real scores from `logs/audit.log` and the critic's own
    recorded description of each. Round 11 added 5 and 14; round 14 added 12.

    The 12 row is the one the 2026-07-28 run demanded. Dr. An's summary cited
    "timely appointments" AND "difficulty reaching the office via phone", and no
    band described evidence pointing both ways — so she was parked at neutral,
    whose text says there are no signals either way. Two-sided feedback is the
    ordinary case, not an edge case.

    The 10 and 14 rows are why the mixed band sits at 12-13 rather than
    straddling neutral: placing it there keeps every anchor round 11 calibrated
    on the band that described it."""
    spec = _rubric_bands(preference_scorer)["practical_access"]
    band = next((lo, hi) for lo, hi in spec["ranges"] if lo <= score <= hi)

    prompt = str(
        preference_scorer.openai_client.chat.completions.create.call_args.kwargs["messages"]
    )
    rubric_text = prompt.split("<rubric>")[1].split("</rubric>")[0].replace("\\n", "\n")
    anchor_line = f"{band[0]}-{band[1]}:"
    start = rubric_text.index(anchor_line)
    # Read to the next anchor so a multi-line band description is captured whole
    following = re.search(r"\n\s+\d+-\d+:", rubric_text[start + len(anchor_line):])
    description = rubric_text[start:start + len(anchor_line) + (following.start() if following else 400)]

    assert must_contain in description, (
        f"practical_access {score} lands on {band} whose text does not describe it: "
        f"{description!r}"
    )


def _practical_access_band_text(preference_scorer: PreferenceScorerAgent, score: int) -> str:
    """The full description of the practical_access band containing `score`.

    Scoped to the practical_access SECTION before matching: a bare "8-" search
    over the whole rubric hits review_substance's "28-40" first.
    """
    spec = _rubric_bands(preference_scorer)["practical_access"]
    low, high = next((lo, hi) for lo, hi in spec["ranges"] if lo <= score <= hi)

    prompt = str(
        preference_scorer.openai_client.chat.completions.create.call_args.kwargs["messages"]
    )
    rubric_text = prompt.split("<rubric>")[1].split("</rubric>")[0].replace("\\n", "\n")
    section = rubric_text[rubric_text.index("3. practical_access"):]

    anchor = f"{low}-{high}:"
    start = section.index(anchor)
    following = re.search(r"\n\s+\d+-\d+:", section[start + len(anchor):])
    return section[start:start + len(anchor) + (following.start() if following else 400)]


def test_neutral_access_band_cannot_be_reached_with_a_citation(
    preference_scorer: PreferenceScorerAgent,
):
    """Dr. De Lima, 2026-07-28: practical_access 10/20 citing "willingness to
    spend time with patients" — a mild access positive, quoted verbatim, scored
    in the band whose own text reads "no access signals either way".

    Round 11 made the bands tile the range numerically; this is the semantic
    hole that survived. The neutral band must now disqualify itself out loud, so
    a judge holding a citation cannot land there and then be required by the
    scoring rules to describe what it just quoted as absent.
    """
    neutral = _practical_access_band_text(preference_scorer, 10)

    assert "no access signals either way" in neutral   # still the neutral band
    assert "you are NOT in this band" in neutral, (
        "the neutral band must state that a citation disqualifies it; without "
        f"that a cited positive lands here again: {neutral!r}"
    )


def test_mixed_access_evidence_has_a_band_of_its_own(
    preference_scorer: PreferenceScorerAgent,
):
    """Dr. An, 2026-07-28: practical_access 10/20 citing "timely appointments"
    AND "difficulty reaching the office via phone".

    The bands are a single positive→negative ladder, so evidence pointing both
    ways had no rung and the neutral band absorbed it — the same 10 De Lima got
    for having only a mild positive. Two providers on one number for opposite
    reasons is the tell that neutral had become the improvisation slot.

    The mixed band must exist, must be distinct from neutral, and must tell the
    judge to quote both sides — a band that admitted mixed evidence but still
    took one citation would reproduce the contradiction on the card.
    """
    mixed = _practical_access_band_text(preference_scorer, 12)
    neutral_low = 8

    assert "MIXED" in mixed
    assert "positives AND" in mixed, f"the band must name two-sided evidence: {mixed!r}"
    assert "BOTH" in mixed, f"the band must require quoting both sides: {mixed!r}"

    spec = _rubric_bands(preference_scorer)["practical_access"]
    mixed_band = next((lo, hi) for lo, hi in spec["ranges"] if lo <= 12 <= hi)
    assert mixed_band[0] != neutral_low, "mixed must not BE the neutral band"


def test_evidence_instruction_is_not_tied_to_the_band(preference_scorer: PreferenceScorerAgent):
    """"Write 'no evidence' where a neutral band applies" MANUFACTURED the
    contradiction: landing in neutral for any reason — including the missing
    band above — required the judge to deny evidence it had just quoted
    elsewhere. Whether evidence exists and which band applies are separate
    questions."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "[]"
    preference_scorer.openai_client.chat.completions.create.return_value = mock_response
    preference_scorer._generate_ai_rankings([{"name": "Dr. X"}], {})
    prompt = str(
        preference_scorer.openai_client.chat.completions.create.call_args.kwargs["messages"]
    )

    assert 'write "no evidence" where a neutral band applies' not in prompt
    assert "never merely because you landed" in prompt
    assert "must not deny it under another" in prompt


def test_access_complaints_are_partitioned_out_of_red_flags(preference_scorer: PreferenceScorerAgent):
    """One wait-time sentence used to dock red_flags AND leave practical_access
    neutral — one signal charged across 50 of the judge's 100 points. Mirrors
    the two routing rules the prompt already carried."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "[]"
    preference_scorer.openai_client.chat.completions.create.return_value = mock_response
    preference_scorer._generate_ai_rankings([{"name": "Dr. X"}], {})
    prompt = str(
        preference_scorer.openai_client.chat.completions.create.call_args.kwargs["messages"]
    )

    assert "belong to practical_access — do NOT" in prompt or \
           "belong to practical_access \\u2014 do NOT" in prompt
    assert "charged twice" in prompt
