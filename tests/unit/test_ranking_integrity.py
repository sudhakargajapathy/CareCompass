"""Defects where the RANKING misrepresented the evidence behind it.

All three were visible on one live run (2026-07-29, Neurology / Chandler AZ):

  A1  a card read "2026 yrs experience" — a YEAR scraped into the tenure field,
      which then clamped to the MAXIMUM experience score on a dimension carrying
      36% of the core weight, and won the pool's "Most experienced" chip.

  A2  five providers who were NEVER SEARCHED sat at ranks 4-8, above one with
      4.8 stars over 102 reviews at rank 9 and one with 4.2 over 175 at rank 13.

  A3  three surfaces disagreed about the same provider's rank on the same page.

They share a shape: the number shown was not the number the evidence supported.
"""
from unittest.mock import MagicMock, patch

import pytest

from agents.critic_validator import refine_rankings
from agents.preference_scorer import (
    EXPERIENCE_CAP,
    EXPERIENCE_MAX_PLAUSIBLE_YEARS,
    EXPERIENCE_UNKNOWN_SCORE,
    PreferenceScorerAgent,
    calculate_experience_score,
    plausible_experience_years,
)


@pytest.fixture
def scorer():
    with patch.object(PreferenceScorerAgent, "_initialize_client", return_value=None):
        agent = PreferenceScorerAgent()
        agent.openai_client = MagicMock()
        return agent


class TestYearShapedTenure:
    """A1 — the observed value was 2026."""

    def test_a_year_does_not_earn_the_best_experience_score(self):
        """The exact failing input. `calculate_experience_score` rejected only
        `years < 0`, so 2026 ran up the ramp and clamped to EXPERIENCE_CAP —
        the worst data in the pool scoring highest on 36% of the core weight."""
        result = calculate_experience_score(2026)

        assert result["score"] != EXPERIENCE_CAP
        assert result["score"] == EXPERIENCE_UNKNOWN_SCORE
        assert result["years"] is None
        assert result["data_quality"] == "implausible"

    def test_a_scraped_review_count_is_also_rejected(self):
        """The other shape seen in this field: a review total (244 — vitals
        states "average 4 stars across 244 reviews") landing in the tenure
        field. Nothing distinguishes it from a year except plausibility."""
        assert calculate_experience_score(244)["score"] == EXPERIENCE_UNKNOWN_SCORE

    def test_the_warning_names_the_value_it_discarded(self):
        """The warning renders to the patient as a data caveat. "Not found" would
        be a lie about a value we did find and chose not to trust."""
        warning = calculate_experience_score(2026)["warning"]

        assert "2026" in warning
        assert "not found" not in warning.lower()

    @pytest.mark.parametrize(
        "years,plausible",
        [(0, True), (20, True), (64, True),
         (EXPERIENCE_MAX_PLAUSIBLE_YEARS, True),
         (EXPERIENCE_MAX_PLAUSIBLE_YEARS + 1, False),
         (2026, False), (-5, False)],
    )
    def test_the_boundary_admits_every_real_career(self, years, plausible):
        """70 is chosen to admit an MD at ~26 still practising at ~90 (~64 yrs).
        The ramp is flat past the knee anyway, so nothing above ~30 moves the
        score — this bounds the FIELD, it does not cap the ramp."""
        assert (plausible_experience_years(years) is not None) is plausible

    def test_a_real_career_is_completely_untouched(self):
        """Guards against a clamp that "fixes" 2026 by also reshaping 20."""
        result = calculate_experience_score(20)

        assert result["years"] == 20
        assert result["data_quality"] == "known"
        assert result["warning"] is None
        assert result["score"] == 80.0

    def test_unparseable_tenure_stays_MISSING_not_implausible(self):
        """"15 years" is an extraction gap, not an impossible career. The first
        version of this split on "was there any value at all", which labelled a
        parse failure implausible and told the patient 15 years was not a
        possible career length. Split on whether it parsed to a NUMBER."""
        for value in (None, "unknown", "15 years", ""):
            assert calculate_experience_score(value)["data_quality"] == "missing"


class TestImplausibleTenureLeavesTheCard:
    """A1, the half a score-only fix misses.

    Composed on `_calculate_base_scores`, not on `calculate_experience_score`:
    the card renders `years_experience` straight off the provider dict
    (`app.py` "N yrs experience") and the pool's "Most experienced" chip ranks
    on the same field. A neutral SCORE still left "2026 yrs experience" printed
    and still awarded that chip to the provider with the worst data.
    """

    def _provider(self, years):
        return {
            "name": "Dr. Yeeshu Arora, MD", "specialty": "Neurology",
            "location": "Chandler, AZ 85224", "years_experience": years,
            "rating": 4.8, "review_count": 102,
        }

    def test_the_value_is_dropped_from_the_scored_provider(self, scorer):
        scored = scorer._calculate_base_scores(
            [self._provider(2026)], {"rating_weight": 0.36, "location_weight": 0.27,
                                     "experience_weight": 0.36}
        )

        assert scored[0]["years_experience"] is None, (
            "the card and the Most-experienced chip both read this field"
        )

    def test_a_plausible_value_survives_scoring(self, scorer):
        scored = scorer._calculate_base_scores(
            [self._provider(26)], {"rating_weight": 0.36, "location_weight": 0.27,
                                   "experience_weight": 0.36}
        )

        assert scored[0]["years_experience"] == 26

    def test_the_discarded_value_is_still_reported(self, scorer):
        """Dropping it from the card must not make it disappear silently — the
        caveat is how a reader learns the tenure was rejected rather than absent."""
        scored = scorer._calculate_base_scores(
            [self._provider(2026)], {"rating_weight": 0.36, "location_weight": 0.27,
                                     "experience_weight": 0.36}
        )

        assert any("2026" in w for w in scored[0]["data_warnings"])


def _validation(name, status="conditional", confidence="high", red_flags=()):
    return {
        "validation_results": {
            "top_provider_validation": {
                "top_provider_validations": [{
                    "provider_name": name,
                    "validation_status": status,
                    "confidence_in_recommendation": confidence,
                    "red_flags": list(red_flags),
                    "validation_notes": "n",
                    "patient_considerations": "c",
                }]
            }
        }
    }


class TestUnresearchedCannotOutrankResearched:
    """A2 — the 2026-07-29 ordering, reproduced.

    Five providers who reached no model sat at ranks 4-8, above one with
    4.8 stars over 102 reviews and one with 4.2 over 175. The critic's
    adjustments reach only providers it SAW, and their range is wider than any
    scoring dimension's realized span, so being researched cost ~7 points and
    being unresearched cost nothing.
    """

    def _pool(self):
        return [
            {"name": "Dr. Marianne De Lima, MD", "final_score": 68.0,
             "enrichment_outcome": "over_budget"},
            {"name": "Dr. Brian Rabin, MD", "final_score": 67.0,
             "enrichment_outcome": "over_budget"},
            {"name": "Dr. Andrea An, MD", "final_score": 64.0,
             "enrichment_outcome": "enriched"},
        ]

    def test_a_researched_provider_wins_despite_a_lower_raw_score(self):
        """An is docked -8 for a 'conditional' verdict, landing at 56 against two
        untouched 68/67s. Before the partition she ranked 3rd of 3."""
        refined, _ = refine_rankings(self._pool(), _validation("Dr. Andrea An, MD"))

        assert [p["name"] for p in refined][0] == "Dr. Andrea An, MD"
        assert refined[0]["refined_score"] < refined[1]["refined_score"], (
            "the researched provider is FIRST while scoring LOWEST — that is the "
            "point: the two groups are not on one scale"
        )

    def test_every_researched_provider_precedes_every_unresearched_one(self):
        refined, _ = refine_rankings(self._pool(), _validation("Dr. Andrea An, MD"))
        outcomes = [p.get("enrichment_outcome") for p in refined]

        assert outcomes.index("enriched") < min(
            i for i, o in enumerate(outcomes) if o == "over_budget"
        )

    def test_score_order_still_holds_inside_each_group(self):
        """The partition must not scramble the scorer's work — only group it."""
        refined, _ = refine_rankings(self._pool(), {})
        deferred = [p for p in refined if p["enrichment_outcome"] == "over_budget"]

        assert [p["name"] for p in deferred] == [
            "Dr. Marianne De Lima, MD", "Dr. Brian Rabin, MD"
        ]

    def test_the_move_carries_a_reason_naming_the_cause(self):
        """The panel would otherwise report these as "moved as others were
        re-scored", crediting them with critic feedback they never received."""
        refined, _ = refine_rankings(self._pool(), {})
        deferred = next(p for p in refined if p["enrichment_outcome"] == "over_budget")

        assert any("not comparable" in r for r in deferred["refinement_reasons"])

    def test_the_partition_is_not_counted_as_a_critic_finding(self):
        """`adjusted_count` renders as "the validator's findings changed N
        recommendation(s)". Being deprioritised for lack of research is not a
        finding the validator made."""
        _, summary = refine_rankings(self._pool(), {})

        assert summary["adjusted_count"] == 0
