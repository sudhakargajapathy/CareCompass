"""Round 16 — the three fixes taken before the portfolio wrap.

All three are things a viewer of the demo can see:

  1. the Responsible-AI panel reported "the validator's findings changed 8
     recommendation(s)" when exactly one provider had been docked
  2. experience out-swung measured patient ratings ~3x at equal weight — the
     critic named it, unprompted, in three consecutive live runs
  3. `red_flags` was the only judge criterion with no rung for "no evidence",
     so a provider nobody had reviewed collected the TOP band by default
"""

import re

import pytest

from agents.critic_validator import refine_rankings
from agents.preference_scorer import JUDGE_RUBRIC


def _validation(**over):
    entry = {
        "provider_name": "Dr. Alpha",
        "rank": 1,
        "validation_status": "approved",
        "confidence_in_recommendation": "high",
        "red_flags": [],
    }
    entry.update(over)
    return entry


def _refine(providers, entries):
    return refine_rankings(providers, {"validation_results": {
        "top_provider_validation": {"top_provider_validations": entries}
    }})


class TestFindingsAreNotScoreDeltas:
    """`adjusted_count` fed the panel sentence "the validator's findings changed
    N recommendation(s)". It counted any non-zero `refinement_adjustment`,
    including the +2 every clean provider gets for high confidence — so N was
    the pool size."""

    def test_a_pool_wide_confidence_bonus_is_not_a_finding(self):
        """THE 2026-07-29 defect. Eight providers, eight +2s, one real finding:
        the panel said 8."""
        providers = [{"name": f"Dr. P{i}", "final_score": 90 - i} for i in range(8)]
        entries = [
            _validation(provider_name=f"Dr. P{i}", rank=i + 1,
                        confidence_in_recommendation="high")
            for i in range(8)
        ]
        entries[3]["validation_status"] = "conditional"

        refined, summary = _refine(providers, entries)

        assert summary["adjusted_count"] == 1, (
            "only the 'conditional' verdict is a finding; +2 for high "
            "confidence is the EXPECTED verdict for a clean record"
        )
        # The scores still move — this changes the COUNT, not the arithmetic.
        assert all(p["refinement_adjustment"] for p in refined)

    def test_low_confidence_is_a_finding(self):
        """The rubric reserves "low" for a provider with no independent platform
        evidence at all, or directly conflicting evidence. That IS something the
        critic found, unlike its opposite."""
        providers = [{"name": "Dr. Alpha", "final_score": 90}]
        _, summary = _refine(providers, [
            _validation(confidence_in_recommendation="low"),
        ])
        assert summary["adjusted_count"] == 1

    def test_red_flags_are_findings(self):
        providers = [{"name": "Dr. Alpha", "final_score": 90}]
        _, summary = _refine(providers, [
            _validation(red_flags=["healthgrades 2.1/13 contradicts vitals 3.5/16"]),
        ])
        assert summary["adjusted_count"] == 1

    def test_a_rejection_is_a_finding(self):
        providers = [{"name": "Dr. Alpha", "final_score": 90}]
        _, summary = _refine(providers, [
            _validation(validation_status="rejected"),
        ])
        assert summary["adjusted_count"] == 1

    def test_findings_are_counted_once_per_provider_not_per_reason(self):
        """The panel sentence counts RECOMMENDATIONS, not reasons — a provider
        with a conditional verdict AND two red flags is one changed
        recommendation."""
        providers = [{"name": "Dr. Alpha", "final_score": 90}]
        refined, summary = _refine(providers, [
            _validation(validation_status="conditional", red_flags=["a", "b"]),
        ])
        assert summary["adjusted_count"] == 1
        assert refined[0]["refinement_findings"] == 2, (
            "the per-provider tally still counts each distinct finding"
        )

    def test_a_clean_pool_reports_zero(self):
        """A permanent non-zero row would train the eye to skip the row that
        matters — the same reason the ring and judge-note rows are conditional."""
        providers = [{"name": f"Dr. P{i}", "final_score": 90 - i} for i in range(5)]
        _, summary = _refine(providers, [
            _validation(provider_name=f"Dr. P{i}", rank=i + 1) for i in range(5)
        ])
        assert summary["adjusted_count"] == 0

    def test_an_unmatched_provider_contributes_nothing(self):
        providers = [{"name": "Dr. Unaudited", "final_score": 90}]
        _, summary = _refine(providers, [_validation(validation_status="rejected")])
        assert summary["adjusted_count"] == 0


class TestRedFlagsAbsenceBand:
    """`review_substance` has "no review text available — neutral" and
    `practical_access` has "no access signals either way". `red_flags` had
    neither, so "we found nothing bad" and "there was nothing to look at"
    shared the 25-30 band — and the second is an assertion we cannot support."""

    @staticmethod
    def _bands(criterion: str):
        """Every (lo, hi) band under one numbered criterion of the live rubric."""
        block = JUDGE_RUBRIC.split(f"{criterion} (0-")[1]
        block = re.split(r"\n\d+\. ", block)[0]
        return [(int(lo), int(hi)) for lo, hi in re.findall(r"^\s*(\d+)-(\d+):", block, re.M)]

    def test_red_flags_has_an_absence_band(self):
        bands = self._bands("red_flags")
        assert len(bands) == 4, f"expected 4 bands after the split, got {bands}"

    def test_the_absence_band_sits_below_verified_clean(self):
        """Absence must not claim the top band — that band asserts we looked and
        found nothing, which needs evidence we could have found something in."""
        bands = dict.fromkeys(self._bands("red_flags"))
        assert (25, 27) in bands and (28, 30) in bands

    def test_absence_is_still_never_penalized(self):
        """The standing doctrine, and the reason this band sits at 25-27 rather
        than the ~18-22 the wrap-up plan proposed: `review_substance` already
        carries the cost of a thin record, and charging it twice is the
        double-penalty the rubric forbids elsewhere."""
        lo, _ = min(band for band in self._bands("red_flags") if band[0] >= 25)
        assert lo >= 25, "the absence band must stay in the upper third of 0-30"

    def test_every_red_flag_score_lands_on_exactly_one_band(self):
        """The tiling invariant, restated for the re-banded criterion. A score
        no anchor describes is one the model improvises (§10.25)."""
        bands = self._bands("red_flags")
        for score in range(31):
            hits = [b for b in bands if b[0] <= score <= b[1]]
            assert len(hits) == 1, f"{score}/30 lands on {len(hits)} bands: {hits}"

    def test_the_top_band_demands_substantive_evidence(self):
        """Mirrors `review_substance`'s source-credibility cap: the best score
        is a finding, not a default."""
        assert "SUBSTANTIVE EVIDENCE" in JUDGE_RUBRIC
        assert "requires evidence you could have found a problem in" in JUDGE_RUBRIC

    @pytest.mark.parametrize("criterion", ["review_substance", "red_flags", "practical_access"])
    def test_all_three_criteria_now_name_absence(self, criterion):
        """The half-done story F2 was raised about: two criteria told the
        patient what absence means and the third silently rewarded it."""
        block = JUDGE_RUBRIC.split(f"{criterion} (0-")[1]
        block = re.split(r"\n\d+\. ", block)[0].lower()
        assert "no " in block and ("absence" in block or "either way" in block or
                                   "no review text" in block), block[:200]
