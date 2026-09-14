"""The golden set cross-checks Tier B: what the LIVE pipeline read off a
key page, graded against the key, with no extra fetch.

The weekly freshness grade (`evals/golden_check.py`) fetches each key page
on its own and asks whether the parsers still read it. It says nothing
about what the PIPELINE read — the two could disagree with both green: a
Tier B run researches the same doctors through the same URLs, and the
numbers it attached to a golden page can be graded against the key by
matching the observation's URL. What these pin: a page is matched by
CANONICAL URL (the vendor's copy of a profile URL carries a query string
the key's does not); the tolerance is the freshness grader's and a
likelihood-template copy counts (same URL, other template, same page); a
COUNT is never graded (listing rows, star pages and likelihood pages state
three different totals for one healthy doctor); a page the pipeline
attached to a differently named provider is `identity` — the stranger's-
page failure class — and `disagree` / `identity` become P2 findings scoped
to Tier B while `unread` (the pipeline knew the URL and read no rating) is
coverage, reported and never alerted; a key for another case grades
nothing; a crash in the cross-check never costs the run its measurement.
"""

import json
from datetime import datetime, timezone

import pytest

from evals import canary_pipeline, golden, thresholds, watch
from evals.cases import case_by_id, tier_b_cases
from evals.thresholds import checks_for_tier, evaluate_pipeline
from tests.unit.test_observability_p3 import FakeOrchestrator

CHANDLER = case_by_id("chandler-neurology")
PHOENIX = case_by_id("phoenix-cardiology")
NOW = datetime(2026, 9, 13, 13, 0, tzinfo=timezone.utc)

HG_URL = "https://www.healthgrades.com/physician/dr-hemant-pandey-xsjwm"
WM_URL = "https://doctor.webmd.com/doctor/cinthi-pillai-5612ad17-overview"
VI_URL = "https://www.vitals.com/doctors/ramzy-medaa-clje80"


def key(case_id="chandler-neurology"):
    return {
        "case_id": case_id, "as_of": "2026-09-13",
        "providers": [
            {"provider_id": "hemant-pandey", "stated_name": "Dr. Hemant Pandey, MD", "pages": [
                {"platform": "healthgrades", "url": HG_URL, "rating": 3.8, "review_count": 88, "rating_pattern": "star"},
                {"platform": "healthgrades", "url": HG_URL, "rating": 3.77, "review_count": None, "rating_pattern": "likelihood"},
            ]},
            {"provider_id": "cinthi-pillai", "stated_name": "Dr. Cinthi Pillai, MD", "pages": [
                {"platform": "webmd", "url": WM_URL, "rating": 4.5, "review_count": 61, "rating_pattern": None},
                {"platform": "vitals", "url": VI_URL, "rating": 4.0, "review_count": 9, "rating_pattern": None},
            ]},
        ],
    }


def obs(url, rating, count, via="profile_parser"):
    return {"source_url": url, "rating": rating, "review_count": count, "extraction_source": via}


def provider(name, outcome="enriched", observations=(), urls=None):
    return {"name": name, "enrichment_outcome": outcome, "review_observations": list(observations),
            "platform_profile_urls": dict(urls or {})}


PANDEY_AGREES = provider("Dr. Hemant Pandey, MD", observations=[obs(HG_URL + "?ref=search", 3.8, 200)])
PILLAI_DISAGREES = provider("Dr. Cinthi Pillai, MD", observations=[obs(WM_URL, 4.9, 61)],
                            urls={"doctor.webmd.com": WM_URL, "vitals.com": VI_URL})
STRANGER = provider("Dr. Nicole Simpkins, MD", observations=[obs(HG_URL, 3.8, 88)])


class TestCrossCheck:
    def test_statuses_by_canonical_url_with_counts_never_graded(self):
        providers = [
            # Reached no model: graded nothing, even carrying a golden URL. Placed
            # FIRST so that, were the outcome gate missing, its 1.0 would claim
            # vitals as a disagreement and Pillai's copy would never be seen.
            provider("Dr. Somebody Else", outcome="over_budget", observations=[obs(VI_URL, 1.0, 1)]),
            PANDEY_AGREES,      # query string on the vendor's copy; count 200 vs the key's 88 — counts are not graded
            PILLAI_DISAGREES,   # webmd read 4.9 against 4.5; vitals known to the pipeline but read nothing this run
        ]
        out = golden.cross_check_pipeline(key(), providers)
        assert (out["matched"], out["key_pages"]) == (3, 3)
        assert {k: out[k] for k in golden.CROSS_STATUSES} == {"agree": 1, "disagree": 1, "identity": 0, "unread": 1}
        by_url = {p["url"]: p for p in out["pages"]}
        assert by_url[HG_URL]["status"] == "agree"
        assert by_url[HG_URL]["observed"] == {"rating": 3.8, "review_count": 200, "via": "profile_parser"}
        assert [e["rating_pattern"] for e in by_url[HG_URL]["key"]] == ["star", "likelihood"]
        assert by_url[WM_URL]["status"] == "disagree" and "pipeline rating 4.9 vs key [4.5]" in by_url[WM_URL]["notes"]
        assert (by_url[WM_URL]["provider_id"], by_url[WM_URL]["platform"]) == ("cinthi-pillai", "webmd")
        assert by_url[VI_URL]["status"] == "unread" and by_url[VI_URL]["observed"] is None
        assert by_url[VI_URL]["notes"] == ["URL known to the pipeline; no observation read from it this run"]

    def test_the_tolerance_is_the_freshness_graders_and_either_template_copy_counts(self):
        pandey = lambda rating: [provider("Dr. Hemant Pandey, MD", observations=[obs(HG_URL, rating, 10)])]
        assert golden.cross_check_pipeline(key(), pandey(3.4))["disagree"] == 1, "0.37 from the nearer copy"
        assert golden.cross_check_pipeline(key(), pandey(4.05))["agree"] == 1, "within 0.3 of the star copy"
        assert golden.cross_check_pipeline(key(), pandey(4.05), tolerance=0.1)["disagree"] == 1
        # 3.49 is 0.31 from the star copy's 3.8 and 0.28 from the likelihood copy's
        # 3.77 — the same URL served as the other template is still this page.
        assert golden.cross_check_pipeline(key(), pandey(3.49))["agree"] == 1

    def test_identity_is_a_golden_url_on_a_differently_named_provider(self):
        out = golden.cross_check_pipeline(key(), [STRANGER])
        page = out["pages"][0]
        assert out["identity"] == 1 and page["status"] == "identity" and page["observed"] is None
        assert page["provider_id"] == "hemant-pandey"
        assert page["notes"] == ["the pipeline attached this page to a provider whose name does not match the key"]
        # One grade per URL: the first provider carrying it is the one graded, so a
        # stranger's claim is not laundered by the right doctor appearing later.
        out = golden.cross_check_pipeline(key(), [STRANGER, PANDEY_AGREES])
        assert (out["matched"], out["identity"], out["agree"]) == (1, 1, 0)

    def test_rating_less_observations_empty_keys_and_junk_never_raise(self):
        out = golden.cross_check_pipeline(key(), [provider("Dr. Hemant Pandey, MD", observations=[obs(HG_URL, None, 88)])])
        assert out["unread"] == 1 and out["pages"][0]["notes"] == ["observation carries no rating"]
        assert golden.cross_check_pipeline({"providers": []}, [PANDEY_AGREES]) == {
            "matched": 0, "key_pages": 0, "agree": 0, "disagree": 0, "identity": 0, "unread": 0, "pages": []}
        junk = [None, {"name": "x"}, provider("Dr. Hemant Pandey, MD", observations=["junk", {"rating": 5}])]
        assert golden.cross_check_pipeline(key(), junk)["matched"] == 0


class GoldenOrchestrator(FakeOrchestrator):
    """The P3 fake, plus a ranked pool the way the scorer node leaves it."""

    def __init__(self, ranked, where="preference_scorer"):
        super().__init__()
        self.ranked, self.where = ranked, where

    def execute_workflow(self, *args, **kwargs):
        result = super().execute_workflow(*args, **kwargs)
        if self.where == "preference_scorer":
            result["agent_outputs"]["preference_scorer"] = {"ranked_providers": self.ranked}
        else:
            result["agent_outputs"]["data_gatherer"] = {**result["agent_outputs"]["data_gatherer"], "providers": self.ranked}
        return result


class TestTierBWiring:
    def test_findings_are_p2_for_disagree_and_identity_only_and_scoped_to_tier_b(self, tmp_path):
        entry = canary_pipeline.run_case(CHANDLER, orchestrator_factory=FakeOrchestrator, key_path=tmp_path / "none.json")
        cross = golden.cross_check_pipeline(key(), [STRANGER, PILLAI_DISAGREES, PANDEY_AGREES])
        found = evaluate_pipeline(entry["record"], CHANDLER, entry["vendor_errors"], entry["status"], entry["fetch_stats"], cross)
        assert [(f.check, f.severity, f.case_id, f.detail) for f in found] == [
            ("golden_pipeline_mismatch", "P2", "chandler-neurology", "hemant-pandey/healthgrades"),
            ("golden_pipeline_mismatch", "P2", "chandler-neurology", "cinthi-pillai/webmd"),
        ]
        assert found[0].observed.startswith("identity: ") and "pipeline rating 4.9 vs key [4.5]" in found[1].observed
        assert found[0].title == "[P2] Pipeline read disagrees with the golden key — chandler-neurology (hemant-pandey/healthgrades)"
        assert found[1].threshold == "rating within 0.3 of a saved copy, same person"
        assert "Tier B trace" in found[0].first_step
        assert "golden_pipeline_mismatch" in checks_for_tier("B") and "golden_pipeline_mismatch" not in checks_for_tier("A")
        assert "golden_pipeline_mismatch" not in thresholds.GOLDEN_CHECKS, "the weekly freshness grade is the other check family"
        unread_only = {"pages": [{"status": "unread", "provider_id": "a", "platform": "vitals"}, {"status": "agree"}]}
        assert evaluate_pipeline(entry["record"], CHANDLER, [], "success", entry["fetch_stats"], unread_only) == []
        assert evaluate_pipeline(entry["record"], CHANDLER, [], "success", entry["fetch_stats"], None) == []

    def test_run_case_grades_the_pipelines_read_when_the_key_is_for_the_case(self, tmp_path):
        key_path = tmp_path / "key.json"
        golden.save_key(key(), key_path)
        ranked = [PANDEY_AGREES, PILLAI_DISAGREES]
        entry = canary_pipeline.run_case(CHANDLER, orchestrator_factory=lambda: GoldenOrchestrator(ranked), key_path=key_path)
        cross = entry["golden_cross_check"]
        assert (cross["matched"], cross["key_pages"], cross["agree"], cross["disagree"], cross["unread"]) == (3, 3, 1, 1, 1)
        lines = canary_pipeline.describe(entry, CHANDLER)
        assert "    golden cross-check: 3 of 3 key pages matched · agree 1 · disagree 1 · identity 0 · unread 1" in lines
        assert any(line.startswith("    [P2] Pipeline read disagrees with the golden key — chandler-neurology (cinthi-pillai/webmd): disagree: ")
                   for line in lines)
        # The run's own measurement is untouched by the grade.
        assert entry["status"] == "success" and entry["record"]["shortlist_size"] == 5 and entry["record"]["pool_raw"] == 117
        # A failed scoring node leaves no ranked pool: the gatherer's providers are graded instead.
        entry = canary_pipeline.run_case(CHANDLER, orchestrator_factory=lambda: GoldenOrchestrator(ranked, where="data_gatherer"), key_path=key_path)
        assert entry["golden_cross_check"]["matched"] == 3
        # Phoenix has no key: nothing graded, no line, no finding; no key file at all: the same.
        entry = canary_pipeline.run_case(PHOENIX, orchestrator_factory=lambda: GoldenOrchestrator(ranked), key_path=key_path)
        assert entry["golden_cross_check"] is None
        assert not any("golden cross-check" in line for line in canary_pipeline.describe(entry, PHOENIX))
        entry = canary_pipeline.run_case(CHANDLER, orchestrator_factory=lambda: GoldenOrchestrator(ranked), key_path=tmp_path / "missing.json")
        assert entry["golden_cross_check"] is None

    def test_a_broken_cross_check_never_costs_the_run_its_measurement(self, tmp_path, monkeypatch):
        key_path = tmp_path / "key.json"
        golden.save_key(key(), key_path)

        def boom(*_a, **_k):
            raise RuntimeError("grader died")

        monkeypatch.setattr(golden, "cross_check_pipeline", boom)
        entry = canary_pipeline.run_case(CHANDLER, orchestrator_factory=lambda: GoldenOrchestrator([PANDEY_AGREES]), key_path=key_path)
        assert entry["golden_cross_check"] is None and entry["status"] == "success" and entry["record"]["pool_raw"] == 117

    def test_watcher_judges_the_cross_check_and_the_summary_names_it(self, tmp_path, capsys):
        key_path = tmp_path / "key.json"
        golden.save_key(key(), key_path)
        entry = canary_pipeline.run_case(CHANDLER, orchestrator_factory=lambda: GoldenOrchestrator([PANDEY_AGREES, PILLAI_DISAGREES]), key_path=key_path)
        payload = {"tier": "B", "source": "canary", "ts": NOW.isoformat(), "cases": [entry]}
        path = tmp_path / "b.json"
        path.write_text(json.dumps(payload, default=str))
        assert watch.main(["--tier", "B", "--input", str(path), "--no-issues"], {}) == 1
        out = capsys.readouterr().out
        assert "[P2] Pipeline read disagrees with the golden key — chandler-neurology (cinthi-pillai/webmd)" in out
        assert "chandler-neurology: golden cross-check — 3 of 3 key pages matched, agree 1, disagree 1, identity 0, unread 1" in out
        findings, evaluated = watch.evaluate_results("B", tier_b_cases(), json.loads(path.read_text()))
        assert [f.check for f in findings] == ["golden_pipeline_mismatch"]
        assert "golden_pipeline_mismatch" in evaluated["chandler-neurology"], "a recovered mismatch issue closes on the next Tier B run"
        # An all-agree run: exit 0, the note still rendered.
        entry = canary_pipeline.run_case(CHANDLER, orchestrator_factory=lambda: GoldenOrchestrator([PANDEY_AGREES]), key_path=key_path)
        path.write_text(json.dumps({**payload, "cases": [entry]}, default=str))
        assert watch.main(["--tier", "B", "--input", str(path), "--no-issues"], {}) == 0
        assert "golden cross-check — 1 of 3 key pages matched, agree 1, disagree 0" in capsys.readouterr().out
