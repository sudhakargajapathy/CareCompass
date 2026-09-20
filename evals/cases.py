"""The canary cases — the searches the monitors run, and each case's floors.

Four cases, chosen for what each exercises rather than for coverage: the
incident case daily, and three weekly cases that each touch a known edge
(a vitals slug gap, a 700-result directory against the 5-page cap, a thin
town that fires the ring). Floors are P2 thresholds on the discovery pool;
the Chandler floor (60) is half the measured baseline (120, 2026-09-02 —
the outage took it to 38). The three weekly floors are CONSERVATIVE GUESSES
until their first runs measure them; a guess that never fires teaches
nothing, and one that fires weekly is re-cut from the observed value.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

PLATFORMS = ("hg", "wm", "vi")
PLATFORM_LABELS = {"hg": "healthgrades", "wm": "webmd", "vi": "vitals"}


@dataclass(frozen=True)
class CanaryCase:
    case_id: str
    specialty: str
    location: str
    schedule: str           # "daily" | "weekly"
    why: str
    pool_floor: int         # P2 when the discovery pool falls below it
    tier_b: bool = False    # also run the full pipeline weekly (P3)
    # Platforms whose listing rows must be non-zero. Phoenix cardiology is
    # KNOWN to hit a vitals stub (the slug gap), so vitals is not expected
    # there — a permanent alert is noise, and the stub is tracked in the
    # slug-coverage item, not by a canary.
    platforms_expected: Tuple[str, ...] = PLATFORMS


CASES: Tuple[CanaryCase, ...] = (
    CanaryCase(
        "chandler-neurology", "Neurology", "Chandler, AZ", "daily",
        "the incident case; baselines pool 120, coverage 71/87/91, 3 credits",
        pool_floor=60, tier_b=True,
    ),
    CanaryCase(
        "phoenix-cardiology", "Cardiology", "Phoenix, AZ", "weekly",
        "exercises the known vitals stub cell — drift and the slug gap",
        pool_floor=30, platforms_expected=("hg", "wm"),
    ),
    CanaryCase(
        "gilbert-family-medicine", "Family Medicine", "Gilbert, AZ", "weekly",
        "healthgrades 722-result directory — exercises the 5-page cap",
        pool_floor=60,
    ),
    # A case is a SEARCH, so its location must survive the same gate a
    # member's does. This case shipped as "Sun Lakes, AZ" and could never
    # run: a real place, but one the vendored GeoNames dataset does not
    # name (its ZIPs file under Chandler), so the location allowlist
    # refused it and every run returned `invalid_input` — 0 pages, 0
    # credits, reported correctly as a P1. It went unseen because the case
    # is WEEKLY and the schedules run in one repository only: it first
    # executed six days after arming, and the canary's own tests mock the
    # gatherer, which is precisely the seam the production gate sits
    # behind. `TestCases` now runs every case through
    # `validate_search_params`. Gold Canyon is the same market shape the
    # case was written for — an affluent retirement community at the far
    # edge of the metro, one ZIP row, real neighbours to ring out to
    # (Apache Junction 8.8 mi, San Tan Valley 14.0, Gilbert 23.3) — and
    # the dataset names it.
    #
    # NO platform is expected to serve rows. In a thin market a platform
    # that answers the town with nothing is a fact about the TOWN, not
    # drift — it is the pipeline's own second ring trigger — so demanding
    # three non-zero platforms here would alert every week for the exact
    # condition this case exists to watch. The pool floor still guards the
    # aggregate (all three blank is a pool of zero); fill this tuple with
    # whatever the first runs actually serve.
    CanaryCase(
        "gold-canyon-neurology", "Neurology", "Gold Canyon, AZ", "weekly",
        "thin home pool at the metro edge — exercises the ring",
        pool_floor=4, platforms_expected=(),
    ),
)


def cases_for(schedule: str) -> Tuple[CanaryCase, ...]:
    """daily → the incident case; weekly → the other three; all → every case."""
    if schedule == "all":
        return CASES
    if schedule not in ("daily", "weekly"):
        raise ValueError(f"unknown schedule {schedule!r}: daily | weekly | all")
    return tuple(c for c in CASES if c.schedule == schedule)


def case_by_id(case_id: str) -> Optional[CanaryCase]:
    return next((c for c in CASES if c.case_id == case_id), None)


def tier_b_cases() -> Tuple[CanaryCase, ...]:
    """The cases that also run the full pipeline (weekly, cold, ~$0.55 each).

    A case's `schedule` is its Tier A cadence; Tier B has one cadence for
    every case that opts in, so the selection is by the flag, not the
    schedule — Chandler is daily for fetch and weekly for the pipeline.
    """
    return tuple(c for c in CASES if c.tier_b)
