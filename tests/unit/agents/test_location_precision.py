"""City-centroid distances are estimates, and must not pose as measurements.

The 2026-07-25 field run showed nearly the whole pool sharing an identical
"8.0 mi (computed straight-line)". That was not a scoring artifact — it is what
one city centroid looks like when ten providers resolve to it. The defect was
presenting it as ten measurements.

The fix does NOT manufacture differentiation between providers in one city: at
city precision we genuinely cannot tell them apart, and inventing a spread
would be the same overstatement in the other direction.
"""

from unittest.mock import patch

import pytest

from agents.preference_scorer import (
    CITY_CENTROID_MARGIN_MILES,
    PreferenceScorerAgent,
)


@pytest.fixture
def scorer():
    with patch("agents.preference_scorer.OpenAI"):
        return PreferenceScorerAgent()


WEIGHTS = {"rating_weight": 0.34, "location_weight": 0.33, "experience_weight": 0.33}


def _location_part(scorer, provider):
    """The location slice of the scorer's own breakdown.

    Uses `_calculate_base_scores` rather than `score_core`: the latter returns
    the ORIGINAL provider dicts (so enrichment can mutate them) and therefore
    carries no breakdown.
    """
    scored = scorer._calculate_base_scores(
        [{**provider, "name": "Dr. X", "rating": 4.0,
          "review_count": 20, "years_experience": 12}], WEIGHTS)
    return scored[0]["score_breakdown"]["location"]


def test_zip_measured_distance_keeps_the_measured_basis(scorer):
    part = _location_part(scorer, {
        "computed_distance_miles": 8.0, "distance_precision": "zip",
        "location_match": "same_city",
    })
    assert part["basis"] == "computed_distance"
    assert part["data_quality"] == "complete"


def test_city_estimate_is_labelled_as_an_estimate(scorer):
    part = _location_part(scorer, {
        "computed_distance_miles": 8.0, "distance_precision": "city",
        "location_match": "same_city",
    })
    assert part["basis"] == "city_estimate"
    assert part["data_quality"] == "derived"
    # The reported value stays the true centroid distance — only the SCORE
    # carries the uncertainty margin, so the UI never shows an inflated number.
    assert part["value"] == 8.0


def test_a_measured_provider_always_beats_an_estimated_one_at_the_same_distance(scorer):
    """The location tiers' governing rule, extended to centroid distances:
    an imputation must never out-score a measurement."""
    measured = _location_part(scorer, {
        "computed_distance_miles": 8.0, "distance_precision": "zip",
        "location_match": "same_city"})
    estimated = _location_part(scorer, {
        "computed_distance_miles": 8.0, "distance_precision": "city",
        "location_match": "same_city"})
    assert measured["score"] > estimated["score"]


def test_the_margin_is_exactly_the_documented_uncertainty(scorer):
    """Score difference must equal the margin's worth of falloff, so the
    constant and its effect cannot drift apart."""
    falloff = 2 * scorer.config.DEFAULT_SEARCH_RADIUS
    measured = _location_part(scorer, {
        "computed_distance_miles": 8.0, "distance_precision": "zip",
        "location_match": "same_city"})["score"]
    estimated = _location_part(scorer, {
        "computed_distance_miles": 8.0, "distance_precision": "city",
        "location_match": "same_city"})["score"]

    expected_gap = CITY_CENTROID_MARGIN_MILES / falloff * 100
    assert measured - estimated == pytest.approx(expected_gap, abs=0.05)


def test_a_genuinely_closer_estimate_still_beats_a_far_measurement(scorer):
    """The margin is proportionate, not punitive: real distance signal survives
    across cities, which is the differentiation we actually have."""
    near_estimate = _location_part(scorer, {
        "computed_distance_miles": 4.0, "distance_precision": "city",
        "location_match": "same_city"})
    far_measured = _location_part(scorer, {
        "computed_distance_miles": 20.0, "distance_precision": "zip",
        "location_match": "same_city"})
    assert near_estimate["score"] > far_measured["score"]


def test_providers_sharing_a_centroid_still_tie(scorer):
    """Deliberate. At city precision we cannot tell them apart, and inventing a
    spread would overstate precision exactly as the old label did."""
    a = _location_part(scorer, {"computed_distance_miles": 8.0,
                                "distance_precision": "city", "location_match": "same_city"})
    b = _location_part(scorer, {"computed_distance_miles": 8.0,
                                "distance_precision": "city", "location_match": "same_city"})
    assert a["score"] == b["score"]


def test_missing_precision_is_treated_as_measured(scorer):
    """Backward compatibility: a cached or pre-upgrade provider without the
    field must not be silently penalized."""
    part = _location_part(scorer, {
        "computed_distance_miles": 8.0, "location_match": "same_city"})
    assert part["basis"] == "computed_distance"


# ---- the gatherer records the precision ----

def _gatherer():
    from agents.data_gatherer import DataGathererAgent
    with patch("agents.data_gatherer.TavilyClient"), patch("agents.data_gatherer.Anthropic"):
        return DataGathererAgent()


def test_gatherer_records_zip_precision_for_a_real_zip():
    provider = {"name": "Dr. A", "location": "Chandler, AZ 85224"}
    _gatherer()._attach_location_evidence(provider, "Chandler, AZ 85249")
    assert provider["distance_precision"] == "zip"


def test_gatherer_records_city_precision_for_a_city_only_address():
    """Gilbert has no ZIP in the text, so its coordinate is a centroid — the
    shape that produced the whole pool's identical 8.0 mi."""
    provider = {"name": "Dr. B", "location": "Gilbert, AZ"}
    _gatherer()._attach_location_evidence(provider, "Chandler, AZ 85249")
    assert provider["computed_distance_miles"] is not None
    assert provider["distance_precision"] == "city"


def test_precision_is_none_when_no_distance_could_be_computed():
    provider = {"name": "Dr. C", "location": "somewhere unparseable"}
    _gatherer()._attach_location_evidence(provider, "Chandler, AZ 85249")
    assert provider["computed_distance_miles"] is None
    assert provider["distance_precision"] is None


def test_same_city_centroid_artifact_still_falls_to_the_tier():
    """The pre-existing rule survives: a city-precision provider in the USER's
    own city would score a fake ~0-mile bullseye, so it gets no distance at
    all rather than an estimate."""
    provider = {"name": "Dr. D", "location": "Chandler, AZ"}
    _gatherer()._attach_location_evidence(provider, "Chandler, AZ 85249")
    assert provider["computed_distance_miles"] is None
    assert provider["location_match"] == "same_city"
