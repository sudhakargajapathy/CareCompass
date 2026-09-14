"""Metamorphic invariants over the deterministic core (observability P5).

No model, no network, no recorded run: each test perturbs an input in a
direction whose effect on the score is DOCUMENTED — more reviews at a
rating above the prior can only help; moving a provider farther can only
hurt; an unknown value scores exactly its stated imputation; a larger
radius never drops someone a smaller one kept; deduping twice is deduping
once; a blend sits between its inputs — and asserts the direction. A
rubric can drift without anyone noticing; an invariant that flips is a
red PR. Where a documented equivalence turns out false, the test is the
finding, not the thing to loosen.
"""

import random

import pytest

from agents.data_gatherer import DataGathererAgent, _blended_platform_rating, _split_by_radius
from agents.preference_scorer import (
    EXPERIENCE_CAP,
    EXPERIENCE_UNKNOWN_SCORE,
    RATING_PRIOR,
    RATING_UNKNOWN_SCORE,
    PreferenceScorerAgent,
    calculate_experience_score,
    calculate_rating_score_with_confidence,
    plausible_experience_years,
)

WEIGHTS = {"rating_weight": 0.3, "location_weight": 0.4, "experience_weight": 0.3, "search_radius_miles": 25}


# Function-scoped ON PURPOSE. A module-scoped agent is built before the
# function-scoped autouse fixture that pins placeholder API keys, so on a
# runner with no keys the constructor raised "OpenAI API key not found" —
# CI run 34742249056 errored 9 of these while the sandbox, whose
# environment carries real keys, passed them all. Construction is cheap
# (no network); the scope is not worth the ordering trap.
@pytest.fixture
def scorer():
    return PreferenceScorerAgent()


@pytest.fixture
def gatherer():
    return DataGathererAgent()


def core(scorer, provider, prefs=WEIGHTS):
    scored = scorer._calculate_base_scores([dict(provider)], dict(prefs))[0]
    total = scored.get("base_score", scored.get("score"))
    assert total is not None, sorted(scored)
    return float(total), scored.get("score_breakdown") or {}


def dimension(breakdown, name):
    entry = breakdown.get(name) or {}
    return float(entry.get("score", entry.get("value", 0)))


# --------------------------------------------------------------------------
# Rating
# --------------------------------------------------------------------------

class TestRatingInvariants:
    @pytest.mark.parametrize("count", [1, 5, 25, 200])
    def test_higher_rating_never_scores_lower_at_a_fixed_count(self, count):
        scores = [calculate_rating_score_with_confidence(r / 10, count)["score"] for r in range(10, 51)]
        assert all(b >= a for a, b in zip(scores, scores[1:]))

    def test_reviews_pull_toward_the_measured_rating_away_from_the_prior(self):
        above = [calculate_rating_score_with_confidence(4.8, n)["score"] for n in (1, 3, 10, 50, 500)]
        below = [calculate_rating_score_with_confidence(2.0, n)["score"] for n in (1, 3, 10, 50, 500)]
        assert all(b >= a for a, b in zip(above, above[1:])), "a 4.8 with more reviews can only rise"
        assert all(b <= a for a, b in zip(below, below[1:])), "a 2.0 with more reviews can only fall"
        at_prior = [calculate_rating_score_with_confidence(RATING_PRIOR, n)["score"] for n in (1, 50, 500)]
        assert max(at_prior) - min(at_prior) < 1e-6, "at the prior itself, reviews change nothing"

    def test_unknown_rating_is_the_prior_not_a_bad_grade(self):
        assert RATING_UNKNOWN_SCORE == pytest.approx(RATING_PRIOR / 5 * 100)
        assert RATING_UNKNOWN_SCORE > calculate_rating_score_with_confidence(2.4, 40)["score"], (
            "a provider nobody looked at must not score below one measured as bad")

    def test_dropping_reviews_moves_toward_the_prior_in_the_documented_direction(self, scorer):
        strong = {"name": "S", "rating": 4.8, "review_count": 200, "computed_distance_miles": 5.0, "years_experience": 12}
        with_reviews, _ = core(scorer, strong)
        without, _ = core(scorer, {**strong, "rating": None, "review_count": None})
        assert without <= with_reviews, "dropping a strong provider's reviews must not raise their score"
        weak = {**strong, "rating": 1.8, "review_count": 40}
        weak_with, _ = core(scorer, weak)
        weak_without, _ = core(scorer, {**weak, "rating": None, "review_count": None})
        assert weak_without >= weak_with, "the imputation is the prior: dropping a bad record cannot lower the score"


# --------------------------------------------------------------------------
# Experience
# --------------------------------------------------------------------------

class TestExperienceInvariants:
    def test_monotone_and_capped(self):
        scores = [calculate_experience_score(y)["score"] for y in range(0, 60)]
        assert all(b >= a for a, b in zip(scores, scores[1:]))
        assert max(scores) <= EXPERIENCE_CAP and scores[-1] == EXPERIENCE_CAP

    def test_unknown_tenure_scores_exactly_the_documented_equivalence(self):
        """EXPERIENCE_UNKNOWN_SCORE is documented as 'a verified 10 years'."""
        assert calculate_experience_score(None)["score"] == EXPERIENCE_UNKNOWN_SCORE
        assert calculate_experience_score(10)["score"] == pytest.approx(EXPERIENCE_UNKNOWN_SCORE)
        assert calculate_experience_score("15 years")["score"] == EXPERIENCE_UNKNOWN_SCORE, "an extraction gap is unknown"

    def test_implausible_tenure_is_dropped_not_capped(self):
        assert plausible_experience_years(2026) is None and plausible_experience_years(-1) is None
        assert plausible_experience_years(70) == 70 and plausible_experience_years(71) is None
        assert calculate_experience_score(2026)["score"] == EXPERIENCE_UNKNOWN_SCORE, (
            "a scraped YEAR once clamped to the cap and won 'Most experienced'")


# --------------------------------------------------------------------------
# Location
# --------------------------------------------------------------------------

class TestLocationInvariants:
    def test_farther_never_scores_higher(self, scorer):
        base = {"name": "L", "rating": 4.2, "review_count": 30, "years_experience": 10}
        totals = [core(scorer, {**base, "computed_distance_miles": d})[0] for d in (1, 4, 9, 15, 30, 45)]
        assert all(b <= a for a, b in zip(totals, totals[1:]))

    def test_a_measurement_beats_the_tier_it_would_fall_into(self, scorer):
        base = {"name": "L", "rating": 4.2, "review_count": 30, "years_experience": 10}
        measured, _ = core(scorer, {**base, "computed_distance_miles": 4.0})
        tiered, _ = core(scorer, {**base, "location_match": "same_city"})
        assert measured >= tiered, "same_city is a pessimistic-edge imputation (~9 mi); a measured 4 mi must not lose to it"

    def test_city_precision_carries_its_margin(self, scorer):
        base = {"name": "L", "rating": 4.2, "review_count": 30, "years_experience": 10, "computed_distance_miles": 6.0}
        zip_precise, _ = core(scorer, {**base, "distance_precision": "zip"})
        city_estimate, _ = core(scorer, {**base, "distance_precision": "city"})
        assert zip_precise > city_estimate

    def test_the_chosen_radius_shapes_the_falloff(self, scorer):
        base = {"name": "L", "rating": 4.2, "review_count": 30, "years_experience": 10, "computed_distance_miles": 8.0}
        at_10, _ = core(scorer, base, {**WEIGHTS, "search_radius_miles": 10})
        at_50, _ = core(scorer, base, {**WEIGHTS, "search_radius_miles": 50})
        assert at_50 > at_10, "8 miles is far at radius 10 and near at radius 50"


# --------------------------------------------------------------------------
# Ranking as a whole
# --------------------------------------------------------------------------

def pool(seed=7, n=12):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append({
            "name": f"Dr. P{i}", "rating": round(rng.uniform(2.5, 5.0), 1), "review_count": rng.randint(1, 300),
            "years_experience": rng.randint(1, 35), "computed_distance_miles": round(rng.uniform(0.5, 45), 1),
        })
    return out


class TestRankingInvariants:
    def test_scaling_every_weight_leaves_the_order_unchanged(self, scorer):
        order_a = [p["name"] for p in scorer.score_core(pool(), {"rating_weight": 0.3, "location_weight": 0.4, "experience_weight": 0.3})]
        order_b = [p["name"] for p in scorer.score_core(pool(), {"rating_weight": 0.6, "location_weight": 0.8, "experience_weight": 0.6})]
        assert order_a == order_b, "weights are normalized to 1.0"

    def test_input_order_does_not_change_the_ranking(self, scorer):
        providers = pool()
        shuffled = list(providers)
        random.Random(3).shuffle(shuffled)
        assert [p["name"] for p in scorer.score_core(providers, WEIGHTS)] == [p["name"] for p in scorer.score_core(shuffled, WEIGHTS)]

    def test_improving_one_provider_never_drops_them(self, scorer):
        providers = pool()
        before = [p["name"] for p in scorer.score_core([dict(p) for p in providers], WEIGHTS)].index("Dr. P5")
        better = [dict(p) for p in providers]
        target = next(p for p in better if p["name"] == "Dr. P5")
        target.update(rating=5.0, review_count=400, computed_distance_miles=0.5, years_experience=30)
        after = [p["name"] for p in scorer.score_core(better, WEIGHTS)].index("Dr. P5")
        assert after <= before


# --------------------------------------------------------------------------
# Discovery-side invariants
# --------------------------------------------------------------------------

class TestRadiusAndDedupeInvariants:
    def test_a_larger_radius_never_drops_someone_a_smaller_one_kept(self):
        providers = pool() + [{"name": "Dr. Unknown", "computed_distance_miles": None}, {"name": "Dr. Nowhere"}]
        kept = {}
        for radius in (5, 10, 25, 50):
            near, far = _split_by_radius(providers, radius)
            kept[radius] = {p["name"] for p in near}
            assert {p["name"] for p in far} | kept[radius] == {p["name"] for p in providers}
            assert "Dr. Unknown" in kept[radius] and "Dr. Nowhere" in kept[radius], "an unknown distance never drops anyone"
        assert kept[5] <= kept[10] <= kept[25] <= kept[50]

    def test_dedupe_is_idempotent_and_order_insensitive_in_what_survives(self, gatherer):
        raw = [
            {"name": "Dr. Pritish Pawar", "profile_url": "https://www.healthgrades.com/physician/dr-pritish-pawar-1", "review_observations": [{"source_url": "https://www.healthgrades.com/physician/dr-pritish-pawar-1", "rating": 4.5, "review_count": 20}]},
            {"name": "Pritish Pawar, MD", "profile_url": "https://www.healthgrades.com/physician/dr-pritish-pawar-1", "years_experience": 12},
            {"name": "Dr. Jane Kim", "profile_url": "https://www.vitals.com/doctors/jane-kim-a1"},
            {"name": "Dr. J Kim", "profile_url": "https://www.vitals.com/doctors/j-kim-b2"},
            {"name": "Dr. Andrea An", "profile_url": "https://doctor.webmd.com/doctor/andrea-an-overview"},
            {"name": "Dr. Andrea Hyeyong An, MD"},
        ]
        once = gatherer._dedupe_providers([dict(p) for p in raw])
        twice = gatherer._dedupe_providers([dict(p) for p in once])
        assert len(once) == len(twice) and {p["name"] for p in once} == {p["name"] for p in twice}
        assert len(once) == 4, "Pawar merges by URL, An by name, the two Kims stay apart (different vitals profiles)"
        shuffled = list(raw)
        random.Random(11).shuffle(shuffled)
        again = gatherer._dedupe_providers([dict(p) for p in shuffled])
        assert len(again) == len(once)
        assert {p.get("profile_url") for p in again} == {p.get("profile_url") for p in once}


class TestBlendInvariants:
    def obs(self, *pairs):
        domains = ("https://www.healthgrades.com/physician/x", "https://doctor.webmd.com/doctor/x-overview", "https://www.vitals.com/doctors/x")
        return [{"source_url": domains[i], "rating": r, "review_count": c} for i, (r, c) in enumerate(pairs)]

    def test_blend_sits_between_its_inputs_and_needs_two_platforms(self):
        assert _blended_platform_rating(self.obs((4.5, 20))) is None
        blend = _blended_platform_rating(self.obs((3.5, 16), (2.1, 13)))
        assert 2.1 <= blend["rating"] <= 3.5 and blend["review_count"] == 29 and blend["platforms"] == 2
        assert blend["spread"] == pytest.approx(1.4)
        equal = _blended_platform_rating(self.obs((4.0, 5), (4.0, 500), (4.0, 1)))
        assert equal["rating"] == 4.0 and equal["spread"] == 0

    def test_volume_on_the_higher_platform_never_lowers_the_blend(self):
        before = _blended_platform_rating(self.obs((5.0, 10), (1.2, 3)))["rating"]
        after = _blended_platform_rating(self.obs((5.0, 200), (1.2, 3)))["rating"]
        assert after >= before and after == pytest.approx(4.9, abs=0.05)

    def test_a_rating_without_a_count_carries_no_weight(self):
        with_bare = self.obs((4.0, 10), (2.0, 10)) + [{"source_url": "https://www.vitals.com/doctors/y", "rating": 1.0, "review_count": None}]
        assert _blended_platform_rating(with_bare)["rating"] == _blended_platform_rating(self.obs((4.0, 10), (2.0, 10)))["rating"]
