"""Observability P4: the golden set — minimizer, grader, findings, watcher and
workflow wiring — on synthetic pages shaped like the parsers' docstrings.

The grader's job is to say WHICH moved, the world or the template, and to
grade identity before either; the minimizer's job is to keep exactly the
lines the parsers read and nothing a reviewer wrote. Both are pinned here
on synthetic pages; `test_golden_fixtures.py` pins the real ones.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evals import golden, golden_check, refresh_fixtures, summary, thresholds, watch
from evals.thresholds import evaluate_golden

REPO = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 9, 13, 13, 0, tzinfo=timezone.utc)

HG_URL = "https://www.healthgrades.com/physician/dr-hemant-pandey-xsjwm"
WM_URL = "https://doctor.webmd.com/doctor/cinthi-pillai-5612ad17-overview"
VI_URL = "https://www.vitals.com/doctors/ramzy-medaa-clje80"

REVIEW = ("I saw Dr. Pillai last spring and my experience was excellent; she listened to me, "
          "we discussed my options, and our follow-up was quick. Five star rating from me.")

WEBMD_PAGE = f"""# Dr. Cinthi Pillai, MD
Neurology
4.5
(61 Ratings)
26 Years Experience
2201 W Fairview St Ste 1, Chandler, AZ, 85224
## Overview
Dr. Pillai has over 18 years of experience treating patients in the Chandler area and enjoys hiking.
## Ratings & Reviews
{REVIEW}
Reviewer Jane D. said the office was clean and the staff friendly and she would come back.
### What are Dr. Cinthi Pillai's patient ratings?
Dr. Cinthi Pillai has received 61 ratings on WebMD Care. Patients gave Dr. Cinthi Pillai an average rating of 4.5 out of 5.
### How many years of experience does Dr. Cinthi Pillai have?
Dr. Cinthi Pillai has approximately 26 years of experience.
## Locations
Chandler Neurology
2201 W Fairview St Ste 1
Chandler, AZ, 85224
Tel: [(480) 555-0100](tel:4805550100)
[Get Directions](https://www.google.com/maps/dir/?api=1&destination=2201%20W%20Fairview%20St%20Ste%201%2C%20Chandler%2C%20AZ%2085224)
"""

HG_PAGE_B = """# Dr. Hemant Pandey, MD
Neurology · Chandler, AZ
Likelihood of recommending Dr. Pandey to family and friends is 3.7727273 out of 5
25+ years of experience
2905 W Warner Rd Ste 20 · Chandler, AZ, 85224
## Patient Reviews
I would not recommend this office to anyone; my appointment was cancelled twice and we waited an hour.
## Compare Providers
4.9 Star Rating
Based on 120 reviews
"""

HG_PAGE_A = HG_PAGE_B.replace(
    "Likelihood of recommending Dr. Pandey to family and friends is 3.7727273 out of 5",
    "3.8 Star Rating\nBased on 88 reviews")


def hg_entry(**over):
    entry = {"platform": "healthgrades", "url": HG_URL, "rating": 3.8, "review_count": None, "years_experience": 25,
             "page_provider_name": "Dr. Hemant Pandey, MD", "location": "2905 W Warner Rd Ste 20, Chandler, AZ 85224",
             "rating_pattern": "likelihood", "content_sha256": "old"}
    entry.update(over)
    return entry


PANDEY = {"provider_id": "hemant-pandey", "stated_name": "Dr. Hemant Pandey, MD"}
PILLAI = {"provider_id": "cinthi-pillai", "stated_name": "Dr. Cinthi Pillai, MD"}


class TestMinimizer:
    def test_keeps_the_parser_regions_and_drops_every_review_line(self):
        mini = golden.minimize(WM_URL, WEBMD_PAGE)
        assert golden.minimized_agrees(WM_URL, WEBMD_PAGE, mini)
        fields = golden.parse_fields(WM_URL, mini)
        assert (fields["rating"], fields["review_count"], fields["years_experience"]) == (4.5, 61, 26)
        assert fields["page_provider_name"] == "Dr. Cinthi Pillai, MD"
        assert "I saw Dr. Pillai" not in mini and "Jane D." not in mini
        assert "(61 Ratings)" in mini and "\n4.5\n" in mini, "the two-line header pair survives"
        assert "destination=" in mini
        mini_b = golden.minimize(HG_URL, HG_PAGE_B)
        assert "would not recommend" not in mini_b and "Likelihood of recommending" in mini_b
        assert golden.parse_fields(HG_URL, mini_b) == golden.parse_fields(HG_URL, HG_PAGE_B)

    def test_first_person_and_long_lines_are_refused_even_when_they_name_a_rating(self):
        page = "# Dr. X\n" + "Loved the ratings this office has, I told my friends. " * 6 + "\n5 Star Rating\n"
        mini = golden.minimize(HG_URL, page)
        assert "Loved" not in mini and "5 Star Rating" in mini
        assert golden.minimize(HG_URL, "") == ""

    def test_provider_id_and_platform(self):
        assert golden.provider_id("Dr. Hemant Pandey, MD") == "hemant-pandey"
        assert golden.provider_id("Dr. Seif-Eddeine, DO") == "seif-eddeine"
        assert golden.platform_of(WM_URL) == "webmd" and golden.platform_of("https://zocdoc.com/x") is None
        assert golden.name_overlap("Dr. J Kim", "Dr. Jane Kim") == pytest.approx(0.5)


class TestGrader:
    def test_identity_is_graded_first(self):
        stranger = WEBMD_PAGE.replace("Dr. Cinthi Pillai, MD", "Dr. Nicole Simpkins, MD")
        grade = golden.grade_page(PILLAI, {"platform": "webmd", "url": WM_URL, "rating": 4.5, "review_count": 61}, stranger)
        assert grade.status == "identity" and "Simpkins" in grade.notes[0]
        slug = golden.grade_page(PANDEY, hg_entry(url="https://www.healthgrades.com/physician/dr-nicole-simpkins-abcde"), HG_PAGE_B)
        assert slug.status == "identity" and "slug" in slug.notes[0]
        assert golden.identity_agrees(HG_URL, "Dr. Hemant Pandey, MD", "Dr. Hemant Pandey, MD")
        assert not golden.identity_agrees(HG_URL, "Dr. Hemant Pandey, MD", "Dr. Nicole Simpkins, MD")
        assert golden.identity_agrees(HG_URL, "Dr. Hemant Pandey, MD", None), "no page name degrades to the slug check"

    def test_world_vs_template_vs_review(self):
        same = golden.grade_page(PANDEY, hg_entry(), HG_PAGE_B)
        assert same.status == "ok" and same.changed_bytes is True and same.expected["rating"] == 3.8
        grown = golden.grade_page(PILLAI, {"platform": "webmd", "url": WM_URL, "rating": 4.5, "review_count": 61, "years_experience": 26},
                                  WEBMD_PAGE.replace("(61 Ratings)", "(70 Ratings)").replace("received 61", "received 70").replace("4.5", "4.6"))
        assert grown.status == "world_drift" and grown.notes == ["review_count 61 → 70", "rating 4.5 → 4.6"]
        down = golden.grade_page(PILLAI, {"platform": "webmd", "url": WM_URL, "rating": 4.5, "review_count": 61},
                                 WEBMD_PAGE.replace("(61 Ratings)", "(50 Ratings)").replace("received 61", "received 50"))
        assert down.status == "needs_review" and "went DOWN" in down.notes[0]
        jump = golden.grade_page(PILLAI, {"platform": "webmd", "url": WM_URL, "rating": 4.5, "review_count": 61},
                                 WEBMD_PAGE.replace("4.5", "4.0"))
        assert jump.status == "needs_review" and "rating moved 4.5 → 4.0" in jump.notes
        tenure = golden.grade_page(PILLAI, {"platform": "webmd", "url": WM_URL, "rating": 4.5, "review_count": 61, "years_experience": 26},
                                   WEBMD_PAGE.replace("26 Years", "27 Years").replace("approximately 26", "approximately 27"))
        assert tenure.status == "needs_review" and "years_experience 26 → 27" in tenure.notes
        blind = WEBMD_PAGE.replace("(61 Ratings)", "").replace("received 61 ratings", "received sixty-one ratings").replace("rating of 4.5 out of 5", "rating of four and a half")
        drift = golden.grade_page(PILLAI, {"platform": "webmd", "url": WM_URL, "rating": 4.5, "review_count": 61}, blind,
                                  fixture_text=golden.minimize(WM_URL, WEBMD_PAGE))
        assert drift.status == "template_drift" and "saved copy still parses" in drift.notes[0]
        regression = golden.grade_page(PILLAI, {"platform": "webmd", "url": WM_URL, "rating": 4.5, "review_count": 61}, blind, fixture_text="# Dr. Cinthi Pillai, MD\n")
        assert regression.status == "template_drift" and "parser regression" in regression.notes[0]
        template = golden.grade_page(PANDEY, hg_entry(), HG_PAGE_A)
        assert template.status == "world_drift" and any("template likelihood → star" in n for n in template.notes)
        assert any("review_count none → 88" in n for n in template.notes)

    def test_grade_url_matches_the_live_template_to_its_saved_copy(self):
        """healthgrades serves A or B per fetch: 7 of 15 key pages flipped
        likelihood → star within twenty minutes of the build. A URL holds one
        copy per template, and the live parse is graded against ITS copy."""
        b_entry = hg_entry()
        a_entry = hg_entry(rating=3.8, review_count=88, rating_pattern="star", content_sha256="a")
        fixtures = {"likelihood": golden.minimize(HG_URL, HG_PAGE_B), "star": golden.minimize(HG_URL, HG_PAGE_A)}
        assert golden.grade_url(PANDEY, [b_entry, a_entry], HG_PAGE_A, fixtures).status == "ok"
        assert golden.grade_url(PANDEY, [b_entry, a_entry], HG_PAGE_B, fixtures).status == "ok"
        grown = golden.grade_url(PANDEY, [b_entry, a_entry], HG_PAGE_A.replace("Based on 88", "Based on 90"), fixtures)
        assert grown.status == "world_drift" and grown.notes == ["review_count 88 → 90"] and grown.expected["rating_pattern"] == "star"
        # A template the key has never seen is a proposal to add a copy, not drift in the numbers.
        new = golden.grade_url(PANDEY, [b_entry], HG_PAGE_A, {"likelihood": fixtures["likelihood"]})
        assert new.status == "world_drift" and new.notes == ["new template 'star' (key holds likelihood): add a copy"]
        # Identity and emptiness still come first, whatever the template.
        assert golden.grade_url(PANDEY, [b_entry], "", fixtures).status == "missing"
        assert golden.grade_url(PANDEY, [b_entry], HG_PAGE_A.replace("# Dr. Hemant Pandey, MD", "# Dr. Nicole Simpkins, MD"), fixtures).status == "identity"
        # Platforms without templates: the single entry is the copy.
        wm = {"platform": "webmd", "url": WM_URL, "rating": 4.5, "review_count": 61}
        assert golden.grade_url(PILLAI, [wm], WEBMD_PAGE, {None: golden.minimize(WM_URL, WEBMD_PAGE)}).status == "ok"
        assert golden.fixture_path("healthgrades", "hemant-pandey", pattern="star").name == "hemant-pandey.star.md"
        assert golden.fixture_path("webmd", "cinthi-pillai").name == "cinthi-pillai.md"

    def test_missing_and_address_flag(self):
        missing = golden.grade_page(PANDEY, hg_entry(), "")
        assert missing.status == "missing" and missing.changed_bytes is None
        assert golden.grade_page(PANDEY, hg_entry(), "x" * 50).status == "missing"
        moved = golden.grade_page(PANDEY, hg_entry(), HG_PAGE_B.replace("2905 W Warner Rd Ste 20", "1000 N Alma School Rd"))
        assert moved.status == "world_drift" and "flagged, never failed" in moved.notes[0]
        counts = golden.summarize_grades([missing, moved])
        assert counts["missing"] == 1 and counts["world_drift"] == 1 and counts["ok"] == 0

    def test_page_entry_records_the_parse_and_the_agreement(self):
        entry = golden.page_entry(WM_URL, WEBMD_PAGE, NOW)
        assert (entry["platform"], entry["rating"], entry["review_count"], entry["years_experience"]) == ("webmd", 4.5, 61, 26)
        assert entry["minimized_agrees"] is True and entry["as_of"] == "2026-09-13"
        assert entry["content_sha256"] == golden.content_hash(WEBMD_PAGE) and entry["minimized_chars"] < entry["fetched_chars"]


class TestGoldenFindings:
    def test_statuses_map_to_checks_with_the_page_as_detail(self):
        payload = {"pages": [
            {"status": "ok", "provider_id": "a", "platform": "webmd"},
            {"status": "world_drift", "provider_id": "a", "platform": "vitals", "notes": ["review_count 1 → 2"]},
            {"status": "identity", "provider_id": "b", "platform": "webmd", "notes": ["page names 'X'"]},
            {"status": "template_drift", "provider_id": "c", "platform": "healthgrades", "notes": ["reads nothing"], "expected": {"rating": 4.0, "review_count": 3}},
            {"status": "missing", "provider_id": "d", "platform": "vitals", "notes": ["empty"]},
            {"status": "needs_review", "provider_id": "e", "platform": "webmd", "notes": ["review_count went DOWN 9 → 8"], "expected": {"rating": 4.5, "review_count": 9}},
        ]}
        found = evaluate_golden(payload)
        assert [(f.check, f.severity, f.detail) for f in found] == [
            ("golden_identity_failure", "P2", "b/webmd"), ("template_drift", "P2", "c/healthgrades"),
            ("golden_page_missing", "P3", "d/vitals"), ("golden_needs_review", "P3", "e/webmd"),
        ]
        assert all(f.case_id == "golden-set" for f in found)
        assert found[1].threshold == "rating 4.0 / count 3" and found[3].title == "[P3] Golden set needs review — golden-set (e/webmd)"
        assert evaluate_golden(None) == [] and evaluate_golden({"pages": [{"status": "ok"}]}) == []
        assert thresholds.GOLDEN_CHECKS <= thresholds.ALL_CHECKS and set(thresholds.CHECK_TITLES) == thresholds.ALL_CHECKS


class TestWatcherAndSummaryWiring:
    def test_golden_file_is_expected_when_asked_for(self, tmp_path):
        assert watch.evaluate_golden_file(None) == ([], {})
        findings, scope = watch.evaluate_golden_file(tmp_path / "none.json")
        assert [(f.check, f.case_id, f.observed) for f in findings] == [("canary_did_not_run", "golden-set", "no golden results file")]
        assert scope == {"golden-set": {"canary_did_not_run"}}
        path = tmp_path / "golden.json"
        path.write_text(json.dumps({"pages": [{"status": "identity", "provider_id": "b", "platform": "webmd", "notes": ["x"]}]}))
        findings, scope = watch.evaluate_golden_file(path)
        assert [f.check for f in findings] == ["golden_identity_failure"]
        assert scope == {"golden-set": set(thresholds.GOLDEN_CHECKS) | {"canary_did_not_run"}}

    def test_main_renders_the_golden_section(self, tmp_path, capsys):
        golden_path = tmp_path / "golden.json"
        golden_path.write_text(json.dumps({"key_as_of": "2026-09-13", "counts": {"ok": 40, "world_drift": 2, "template_drift": 1}, "pages": [
            {"status": "template_drift", "provider_id": "c", "platform": "healthgrades", "notes": ["reads nothing"], "expected": {}},
            {"status": "world_drift", "provider_id": "a", "platform": "vitals", "notes": ["review_count 1 → 2"]},
        ]}))
        payload = {"tier": "A", "schedule": "weekly", "cases": []}
        results = tmp_path / "a.json"
        results.write_text(json.dumps(payload))
        code = watch.main(["--tier", "A", "--schedule", "weekly", "--input", str(results), "--no-issues", "--golden", str(golden_path)], {})
        assert code == 1
        out = capsys.readouterr().out
        assert "### Golden set" in out and "| template_drift | c/healthgrades | reads nothing |" in out
        assert "python -m evals.refresh_fixtures --refresh a" in out and "Key as of 2026-09-13" in out
        assert "[P2] Template drift — golden-set (c/healthgrades)" in out

    def test_weekly_workflow_grades_the_set_and_watches_the_report(self):
        import yaml

        text = (REPO / ".github" / "workflows" / "canary-tier-a.yml").read_text(encoding="utf-8")
        steps = yaml.safe_load(text)["jobs"]["fetch-canary"]["steps"]
        plan = next(s for s in steps if s.get("id") == "plan")
        assert "--reports-dir reports" in plan["run"] and "--golden evals/out/golden.json" in plan["run"]
        assert 'SCHEDULE" = "weekly"' in plan["run"]
        grade = next(s for s in steps if "evals.golden_check" in str(s.get("run", "")))
        assert grade["if"] == "always() && steps.plan.outputs.golden == 'true'"
        watcher = next(s for s in steps if "evals.watch" in str(s.get("run", "")))
        assert "steps.plan.outputs.watch_args" in watcher["run"]


class FakeAgent:
    """Discovery + extract, canned; records the URLs it was asked to fetch."""

    def __init__(self, bodies):
        self.bodies = bodies
        self.fetched = []

    def gather_providers(self, specialty, location, insurance=None, enrich=True, radius_miles=None):
        return {"providers": [
            {"name": "Dr. Cinthi Pillai, MD", "specialty": "Neurology",
             "platform_profile_urls": {"doctor.webmd.com": WM_URL, "www.healthgrades.com": HG_URL, "www.vitals.com": VI_URL}},
            {"name": "Dr. Hemant Pandey, MD", "specialty": "Neurology", "platform_profile_urls": {"www.healthgrades.com": HG_URL}},
            {"name": "Dr. Lonely Single", "specialty": "Neurology", "platform_profile_urls": {}},
        ], "search_metadata": {}, "status": "success"}

    def _extract_pages(self, urls, purpose="", stage="discovery"):
        self.fetched.append(list(urls))
        return [{"url": u, "raw_content": self.bodies.get(u, ""), "title": u, "content": "", "score": 1.0} for u in urls if self.bodies.get(u)]


class TestBuildRefreshInspectAndCheck:
    def test_build_writes_key_fixtures_and_private_bodies(self, tmp_path):
        agent = FakeAgent({WM_URL: WEBMD_PAGE, HG_URL: HG_PAGE_B, VI_URL: ""})
        key = refresh_fixtures.build(
            "chandler-neurology", n=15, min_platforms=1, key_path=tmp_path / "key.json",
            fixture_dir=tmp_path / "fixtures", bodies_dir=tmp_path / "bodies", agent_factory=lambda: agent, now=NOW,
        )
        assert [p["provider_id"] for p in key["providers"]] == ["cinthi-pillai", "hemant-pandey"], "most platforms first, then name; the URL-less provider is skipped"
        pillai = key["providers"][0]
        assert [pg["platform"] for pg in pillai["pages"]] == ["webmd"], (
            "an empty vitals body is not an answer, and the healthgrades URL serves PANDEY's page — identity is gated at build")
        assert (tmp_path / "fixtures" / "webmd" / "cinthi-pillai.md").exists() and (tmp_path / "bodies" / "webmd" / "cinthi-pillai.md").read_text() == WEBMD_PAGE
        assert (tmp_path / "fixtures" / "healthgrades" / "hemant-pandey.likelihood.md").exists()
        assert "I saw Dr. Pillai" not in (tmp_path / "fixtures" / "webmd" / "cinthi-pillai.md").read_text()
        assert key["as_of"] == "2026-09-13" and key["case_id"] == "chandler-neurology"
        assert golden.load_key(tmp_path / "key.json")["providers"][1]["pages"][0]["rating_pattern"] == "likelihood"
        with pytest.raises(ValueError):
            refresh_fixtures.build("nowhere", agent_factory=lambda: agent, key_path=tmp_path / "k2.json")

    def test_refresh_rewrites_one_provider_and_inspect_prints_the_read(self, tmp_path):
        agent = FakeAgent({WM_URL: WEBMD_PAGE, HG_URL: HG_PAGE_B})
        paths = dict(key_path=tmp_path / "key.json", fixture_dir=tmp_path / "fixtures", bodies_dir=tmp_path / "bodies")
        refresh_fixtures.build("chandler-neurology", min_platforms=1, agent_factory=lambda: agent, now=NOW, **paths)
        agent.bodies[WM_URL] = WEBMD_PAGE.replace("(61 Ratings)", "(70 Ratings)").replace("received 61", "received 70")
        provider = refresh_fixtures.refresh("cinthi-pillai", agent_factory=lambda: agent, now=NOW, **paths)
        assert provider["pages"][0]["review_count"] == 70 and agent.fetched[-1] == [WM_URL]
        assert golden.load_key(tmp_path / "key.json")["providers"][0]["pages"][0]["review_count"] == 70
        assert refresh_fixtures.refresh("nobody", agent_factory=lambda: agent, **paths) is None
        # Pandey's healthgrades URL now serves template A: the refresh ADDS the star copy beside the likelihood one.
        agent.bodies[HG_URL] = HG_PAGE_A
        pandey = refresh_fixtures.refresh("hemant-pandey", agent_factory=lambda: agent, now=NOW, **paths)
        assert [(pg["rating_pattern"], pg["review_count"]) for pg in pandey["pages"]] == [("likelihood", None), ("star", 88)]
        assert (tmp_path / "fixtures" / "healthgrades" / "hemant-pandey.star.md").exists()
        # An empty fetch keeps what the key had.
        agent.bodies[HG_URL] = ""
        assert len(refresh_fixtures.refresh("hemant-pandey", agent_factory=lambda: agent, now=NOW, **paths)["pages"]) == 2
        lines = refresh_fixtures.inspect(WM_URL, agent_factory=lambda: agent)
        assert lines[0] == WM_URL and any("rating: 4.5" in l for l in lines) and any("agrees with full parse: True" in l for l in lines)
        assert refresh_fixtures.inspect("https://www.vitals.com/doctors/none", agent_factory=lambda: agent) == ["https://www.vitals.com/doctors/none: no body returned"]

    def test_check_grades_every_key_page_and_writes_bodies_privately(self, tmp_path):
        agent = FakeAgent({WM_URL: WEBMD_PAGE, HG_URL: HG_PAGE_B})
        paths = dict(key_path=tmp_path / "key.json", fixture_dir=tmp_path / "fixtures", bodies_dir=tmp_path / "bodies")
        refresh_fixtures.build("chandler-neurology", min_platforms=1, agent_factory=lambda: agent, now=NOW, **paths)
        live = {WM_URL: WEBMD_PAGE.replace("(61 Ratings)", "(65 Ratings)").replace("received 61", "received 65"), HG_URL: ""}
        payload = golden_check.run(
            key_path=paths["key_path"], fixture_dir=paths["fixture_dir"], bodies_dir=tmp_path / "check_bodies",
            out_path=tmp_path / "golden.json", fetcher=lambda urls: {u: live.get(u, "") for u in urls}, now=NOW,
        )
        assert payload["counts"]["world_drift"] == 1 and payload["counts"]["missing"] == 1 and payload["tier"] == "golden"
        assert (tmp_path / "check_bodies" / "webmd" / "cinthi-pillai.md").exists() and not (tmp_path / "check_bodies" / "healthgrades").exists()
        written = json.loads((tmp_path / "golden.json").read_text())
        assert written["case_id"] == "chandler-neurology" and len(written["pages"]) == 2
        lines = golden_check.describe(written)
        assert lines[0].startswith("golden set (chandler-neurology, key as of 2026-09-13)") and any("missing" in l for l in lines[1:])
        assert golden_check.run(key_path=tmp_path / "absent.json", fetcher=lambda urls: {}) is None
        assert [f.check for f in evaluate_golden(written)] == ["golden_page_missing"]
        # Re-grading from saved bodies needs no fetch: the runbook's artifact diff.
        fetch = golden_check.bodies_fetcher(tmp_path / "check_bodies", golden.load_key(paths["key_path"]))
        assert set(fetch([WM_URL, HG_URL])) == {WM_URL}
        again = golden_check.run(key_path=paths["key_path"], fixture_dir=paths["fixture_dir"], bodies_dir=tmp_path / "again",
                                 out_path=tmp_path / "again.json", fetcher=fetch, now=NOW)
        assert again["counts"]["world_drift"] == 1 and again["counts"]["missing"] == 1
