"""A page the vendor serves stripped is asked for once more, past its stored copy.

2026-11-22: both canaries went red on a commit that had been green the day
before. Tavily /extract was answering specific URLs from a stored copy that
had lost the page's substance — healthgrades and webmd city listings with no
provider headings or profile links, webmd and vitals profiles without their
rating — and the bodies were NOT empty, so the empty-body count read zero,
the listing parser read 0 rows from two of three platforms, and every
researched provider carried vitals data only. The alerts said "platform
listing rows zero" and "template drift", whose first step is to rewrite a
parser; nothing was wrong with a parser. The same request with
`use_cache=False` returned every page whole (9 of 9 URLs, three rounds).

These tests pin, with the shapes those bodies actually had:

  * the recognition — a listing that states its total and names nobody, a
    profile with neither rating nor count — and that a WHOLE page of each
    kind is left alone;
  * the wiring in the real fetch primitive: only the stripped URLs are asked
    for again, that second request carries `use_cache=False`, the first one
    does not, the restored body replaces the stripped one where every
    consumer reads it, and a healthy batch costs exactly one call;
  * the refusals — a body that reads the same the second time is kept and
    NOT counted as restored (a doctor nobody has rated), and a refetch that
    raises costs nothing but itself;
  * the monitor — a restored page is its own low-severity finding per
    platform, in the fetch canary, the pipeline canary and the golden check.

The bodies are synthetic reductions of the fetched pages: the frame lines
the parsers and the result-count patterns read, and no review prose.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agents.data_gatherer import DataGathererAgent
from evals import golden, golden_check, refresh_fixtures, thresholds
from evals.cases import case_by_id
from evals.thresholds import checks_for_tier, evaluate_fetch, evaluate_golden, evaluate_pipeline
from tests.helpers.fake_langfuse import FakeLangfuse, by_name, install_fake_tracing
from utils import tracing
from utils.cost_tracker import get_cost_tracker
from utils.degraded_fetch import LISTING_WITHOUT_ROWS, PROFILE_WITHOUT_PAIR, degraded_reason
from utils.listing_parser import parse_listing
from utils.profile_parser import parse_profile

CHANDLER = case_by_id("chandler-neurology")
NOW = datetime(2026, 11, 23, 13, 0, tzinfo=timezone.utc)

HG_LISTING = "https://www.healthgrades.com/neurology-directory/az-arizona/chandler"
WM_LISTING = "https://doctor.webmd.com/providers/specialty/neurology/arizona/chandler"
VI_LISTING = "https://www.vitals.com/neurology/az/chandler"
VI_PROFILE = "https://www.vitals.com/doctors/hemant-kumar-pandey-eeltvn"
WM_PROFILE = "https://doctor.webmd.com/doctor/dan-mirza-b8329684-1fab-4281-b365-f58711b6f0d3-overview"

# healthgrades, as served stripped: the result total and the per-card
# attribute bullets, and not one provider heading.
HG_LISTING_STRIPPED = (
    '# 20 Best Neurologists Near Chandler, AZ\n\n'
    '## We found 80 results for "Neurologists near Chandler, AZ"\n\n'
    '- Listened/answered questions\n- Explains conditions well\n\n<!--THE END-->\n\n- Virtual Visits\n'
)
HG_LISTING_WHOLE = HG_LISTING_STRIPPED + (
    '\n### [Dr. Peter Struck, MD](/physician/dr-peter-struck-3pp3q)\n\n'
    '![](https://photos.example/3pp3q.jpg)\n\n'
    '### [Dr. Anthony Mosley, MD](/physician/dr-anthony-mosley-29h87)\n'
)

# webmd, as served stripped: the total, the filter labels, and each card's
# address — with the heading that names the doctor and links the profile gone.
WM_LISTING_STRIPPED = (
    '# Best **Neurologists** in **Chandler, AZ**\n\n'
    'Chandler, AZ has **146 Neurologist** results with an average of **31 years of experience**.\n\n'
    'Sort By\n\nRatings\n\nInsurance\n\n'
    '- VERIFIED\n  \n  Neurology\n  \n  655 S Dobson Rd Ste 103 Bldg A, Chandler, AZ, 85224 2.03 miles ...View Profile\n'
)
WM_LISTING_WHOLE = WM_LISTING_STRIPPED + (
    '\n## [Dr. Paul Ryan Macdonald](https://doctor.webmd.com/doctor/paul-ryan-macdonald-0a1b-overview)\n'
    'Neurology\n'
    '[3.0 13 Ratings](https://doctor.webmd.com/doctor/paul-ryan-macdonald-0a1b-overview#ratings)  28 Years Exp erience\n'
    '655 S Dobson Rd Ste 103 Bldg A, Chandler, AZ, 85224 2.03 miles\n'
)

# vitals profile, as served stripped: the compare table without its rating
# row and without the links that mark the subject's column.
VI_PROFILE_STRIPPED = (
    '|  |  |  |  |\n|---|---|---|---|\n'
    '| Dr. Hemant Kumar Pandey | Dr. Shan Patel, PsyD | Jeannine Thomas, MSPA, PA-C | Dr. Cinthi Pillai, MD |\n'
    '| Neurology | Neurology | Neurology | Neurology |\n'
    '| 30 Years of Experience | 13 Years of Experience | 21 Years of Experience | 20 Years of Experience |\n'
    '| Chandler , AZ | Tampa , FL | San Antonio , TX | New York , NY |\n'
    '| Current Profile | View Profile | View Profile | View Profile |\n'
)
VI_LISTING_WHOLE = (
    '## 142 Neurologists near Chandler, AZ\n\n'
    '### Dr. Brandon Craig Woods\nNeurology\n3.5 12 ratings\n18 years exp\nMesa, AZ9.8 mi\n'
)
# vitals' Gilbert family-medicine listing as the 2026-11-23 canary received
# it (5,476 chars): the filter panel, an explainer and an FAQ. No entry and
# no "N Family Physicians Found" line.
VI_FRAME_LISTING = "https://www.vitals.com/family-medicine/az/gilbert"
VI_LISTING_FRAME_ONLY = (
    "## Filter Family Physicians Results\n\n"
    "## What is a Family Physician and How Can They Help You?\n\n"
    "A family physician provides continuous, comprehensive healthcare.\n\n"
    "## Frequently Asked Questions\n\n"
    "## Ensuring Data Transparency and Trust: A Guide to Vitals Doctor Listings and Reviews\n"
)
VI_PROFILE_WHOLE = (golden.FIXTURE_DIR / "vitals" / "hemant-pandey.md").read_text(encoding="utf-8")


def _gatherer(*responses):
    gatherer = DataGathererAgent()
    gatherer.tavily_client = MagicMock()
    gatherer.tavily_client.extract.side_effect = [
        {"results": [{"url": url, "raw_content": body} for url, body in response.items()], "failed_results": []}
        if isinstance(response, dict) else response
        for response in responses
    ]
    return gatherer


def _calls(gatherer):
    return gatherer.tavily_client.extract.call_args_list


class TestRecognition:
    def test_fixture_bodies_are_what_they_claim(self):
        """A reduction that stopped reproducing the failure would test nothing."""
        assert parse_listing(HG_LISTING, HG_LISTING_STRIPPED) == [] and len(parse_listing(HG_LISTING, HG_LISTING_WHOLE)) == 2
        assert parse_listing(WM_LISTING, WM_LISTING_STRIPPED) == [] and len(parse_listing(WM_LISTING, WM_LISTING_WHOLE)) == 1
        assert len(parse_listing(VI_LISTING, VI_LISTING_WHOLE)) == 1
        whole = parse_profile(VI_PROFILE, VI_PROFILE_WHOLE)
        assert (whole.get("rating"), whole.get("review_count")) == (4.2, 30)
        stripped = parse_profile(VI_PROFILE, VI_PROFILE_STRIPPED)
        assert stripped.get("rating") is None and stripped.get("review_count") is None

    @pytest.mark.parametrize("url, body", [(HG_LISTING, HG_LISTING_STRIPPED), (WM_LISTING, WM_LISTING_STRIPPED)])
    def test_a_listing_that_states_a_total_and_names_nobody_is_stripped(self, url, body):
        assert degraded_reason(url, body) == LISTING_WITHOUT_ROWS

    def test_a_profile_with_neither_rating_nor_count_is_stripped(self):
        assert degraded_reason(VI_PROFILE, VI_PROFILE_STRIPPED) == PROFILE_WITHOUT_PAIR

    @pytest.mark.parametrize("url, body", [
        (HG_LISTING, HG_LISTING_WHOLE), (WM_LISTING, WM_LISTING_WHOLE), (VI_PROFILE, VI_PROFILE_WHOLE),
    ])
    def test_a_whole_page_is_left_alone(self, url, body):
        assert degraded_reason(url, body) is None

    def test_a_vitals_listing_cut_to_its_frame_is_stripped_even_with_no_total(self):
        """The first public canary run on the refetch still reported "vitals:
        0 rows" for gilbert-family-medicine: the stored copy had lost its
        entries AND its "1,552 Family Physicians Found" line, so the
        states-a-total rule passed it. The same request past the cache
        returned 56 rows. The filter panel heading is what marks the body as
        a listing that lost its results."""
        from utils.listing_parser import listing_result_count
        assert listing_result_count(VI_FRAME_LISTING, VI_LISTING_FRAME_ONLY) is None
        assert parse_listing(VI_FRAME_LISTING, VI_LISTING_FRAME_ONLY) == []
        assert degraded_reason(VI_FRAME_LISTING, VI_LISTING_FRAME_ONLY) == LISTING_WITHOUT_ROWS
        # The panel on a page that still has its rows is a healthy listing.
        assert degraded_reason(VI_LISTING, "## Filter Neurologists Results\n\n" + VI_LISTING_WHOLE) is None
        # Another platform's page is not judged by vitals' frame.
        assert degraded_reason(HG_LISTING, VI_LISTING_FRAME_ONLY) is None

    def test_the_frame_only_listing_is_refetched_and_its_rows_reach_the_caller(self):
        gatherer = _gatherer({VI_FRAME_LISTING: VI_LISTING_FRAME_ONLY}, {VI_FRAME_LISTING: VI_LISTING_WHOLE})
        pages = gatherer._extract_pages([VI_FRAME_LISTING], purpose="discovery")
        assert [p["raw_content"] for p in pages] == [VI_LISTING_WHOLE]
        assert _calls(gatherer)[1].kwargs["use_cache"] is False
        assert len(parse_listing(VI_FRAME_LISTING, pages[0]["raw_content"])) == 1

    def test_a_listing_page_that_states_no_total_is_not_judged(self):
        """A wrong slug is answered with a marketing page that states no
        total: that is the slug probe's finding, and asking for it again
        past the cache would return the same page at a credit a search."""
        assert degraded_reason(VI_LISTING, "# Popular specialties\n\nFind a doctor near you.\n") is None

    def test_pages_without_a_parser_and_empty_bodies_are_not_judged(self):
        assert degraded_reason("https://www.barrowneuro.org/our-team/", "A practice page with no numbers.") is None
        assert degraded_reason(HG_LISTING, "") is None and degraded_reason(None, HG_LISTING_STRIPPED) is None


class TestRefetchWiring:
    def test_stripped_pages_are_refetched_past_the_cache_and_replaced(self):
        """The live shape: one batch, two listings and a profile stripped,
        one listing whole. Only the three are asked for again."""
        gatherer = _gatherer(
            {HG_LISTING: HG_LISTING_STRIPPED, WM_LISTING: WM_LISTING_STRIPPED,
             VI_PROFILE: VI_PROFILE_STRIPPED, VI_LISTING: VI_LISTING_WHOLE},
            {HG_LISTING: HG_LISTING_WHOLE, WM_LISTING: WM_LISTING_WHOLE, VI_PROFILE: VI_PROFILE_WHOLE},
        )
        pages = gatherer._extract_pages([HG_LISTING, WM_LISTING, VI_PROFILE, VI_LISTING], purpose="discovery")
        first, second = _calls(gatherer)
        assert "use_cache" not in first.kwargs, "a healthy fetch must stay on the fast path"
        assert second.kwargs["use_cache"] is False
        assert second.kwargs["urls"] == [HG_LISTING, WM_LISTING, VI_PROFILE]
        assert (second.kwargs["extract_depth"], second.kwargs["format"]) == ("basic", "markdown")
        by_url = {p["url"]: p["raw_content"] for p in pages}
        assert [p["url"] for p in pages] == [HG_LISTING, WM_LISTING, VI_PROFILE, VI_LISTING], "order and URLs unchanged"
        # Composed: what the consumers of these pages now read.
        assert len(parse_listing(HG_LISTING, by_url[HG_LISTING])) == 2
        assert parse_listing(WM_LISTING, by_url[WM_LISTING])[0]["rating"] == 3.0
        assert parse_profile(VI_PROFILE, by_url[VI_PROFILE])["review_count"] == 30

    def test_counts_reach_the_fetch_stats_per_stage_and_platform(self):
        gatherer = _gatherer(
            {HG_LISTING: HG_LISTING_STRIPPED, WM_LISTING: WM_LISTING_STRIPPED},
            {HG_LISTING: HG_LISTING_WHOLE, WM_LISTING: WM_LISTING_STRIPPED},
        )
        gatherer._extract_pages([HG_LISTING, WM_LISTING], stage="discovery")
        discovery = gatherer.fetch_stats()["fetch"]["discovery"]
        assert (discovery["degraded"], discovery["recovered"]) == (2, 1)
        assert discovery["by_domain"]["healthgrades.com"] == {"fetched": 1, "empty": 0, "degraded": 1, "recovered": 1}
        assert discovery["by_domain"]["doctor.webmd.com"] == {"fetched": 1, "empty": 0, "degraded": 1}
        assert (discovery["planned"], discovery["fetched"]) == (2, 2), "the refetch is not a second planned page"
        assert gatherer.fetch_stats()["fetch"]["enrichment"]["degraded"] == 0

    def test_a_healthy_batch_costs_exactly_one_call(self):
        gatherer = _gatherer({HG_LISTING: HG_LISTING_WHOLE, VI_PROFILE: VI_PROFILE_WHOLE})
        gatherer._extract_pages([HG_LISTING, VI_PROFILE])
        assert len(_calls(gatherer)) == 1
        discovery = gatherer.fetch_stats()["fetch"]["discovery"]
        assert (discovery["degraded"], discovery["recovered"]) == (0, 0)
        assert discovery["by_domain"]["healthgrades.com"] == {"fetched": 1, "empty": 0}

    def test_a_body_that_reads_the_same_twice_is_kept_and_not_counted_restored(self):
        """A doctor nobody has rated states no pair on the live page either.
        Asked once, never looped, and never reported as a recovery."""
        gatherer = _gatherer({VI_PROFILE: VI_PROFILE_STRIPPED}, {VI_PROFILE: VI_PROFILE_STRIPPED + "\nNo data\n"})
        pages = gatherer._extract_pages([VI_PROFILE], stage="enrichment")
        assert len(_calls(gatherer)) == 2
        assert pages[0]["raw_content"] == VI_PROFILE_STRIPPED
        enrichment = gatherer.fetch_stats()["fetch"]["enrichment"]
        assert (enrichment["degraded"], enrichment["recovered"]) == (1, 0)

    @pytest.mark.parametrize("second", [
        RuntimeError("vendor rejected an unknown parameter"),
        {"results": [], "failed_results": [{"url": WM_LISTING, "error": "timeout"}]},
        {"results": [{"url": WM_LISTING, "raw_content": ""}], "failed_results": []},
    ])
    def test_a_failed_refetch_costs_nothing_but_itself(self, second):
        """`use_cache` is undocumented: the day the vendor rejects it, a
        search must behave exactly as it did before this refetch existed."""
        gatherer = _gatherer({WM_LISTING: WM_LISTING_STRIPPED}, second)
        pages = gatherer._extract_pages([WM_LISTING])
        assert [(p["url"], p["raw_content"]) for p in pages] == [(WM_LISTING, WM_LISTING_STRIPPED)]
        assert len(_calls(gatherer)) == 2, "one attempt, no retry"
        assert gatherer.fetch_stats()["fetch"]["discovery"]["recovered"] == 0

    def test_the_refetch_is_billed_and_traced_as_its_own_extract_call(self, monkeypatch):
        install_fake_tracing(monkeypatch)
        try:
            gatherer = _gatherer({HG_LISTING: HG_LISTING_STRIPPED}, {HG_LISTING: HG_LISTING_WHOLE})
            with tracing.start_run(run_id="run-refetch"):
                gatherer._extract_pages([HG_LISTING], purpose="discovery", stage="discovery")
            first, second = by_name(FakeLangfuse.instances[0], "tavily.extract")
            assert "refetch" not in first.merged["input"]
            assert second.merged["input"] == {
                "url_count": 1, "purpose": "discovery", "stage": "discovery", "attempt": 1, "refetch": "cache_bypass",
            }
            out = second.merged["output"]
            assert (out["refetched"], out["recovered"], out["credits"]) == (1, 1, 1)
            assert out["pages"] == [{"url": HG_LISTING, "chars": len(HG_LISTING_WHOLE), "sha256": tracing.sha256_text(HG_LISTING_WHOLE)[:16]}]
            assert "Peter Struck" not in str(second.merged), "page text must never reach the trace"
            assert get_cost_tracker().summary()["tavily"]["credits_by_stage"] == {"discovery": 2}
        finally:
            tracing.reset_client()


def _stats(**by_stage):
    return {"fetch": {
        stage: {"planned": 4, "fetched": 4, "empty": 0, "failed": 0, "by_domain": domains}
        for stage, domains in by_stage.items()
    }}


def _record(**over):
    record = {
        "pages_planned": 11, "pages_fetched": 10, "empty_bodies": 0,
        "rows_hg": 73, "rows_wm": 114, "rows_vi": 105, "pool_raw": 117,
        "fallback_fired": False, "ring_fired": False,
    }
    record.update(over)
    return record


class TestMonitor:
    def test_a_restored_page_is_one_low_severity_finding_per_platform(self):
        stats = _stats(discovery={
            "healthgrades.com": {"fetched": 3, "empty": 0, "degraded": 3, "recovered": 3},
            "doctor.webmd.com": {"fetched": 3, "empty": 0, "degraded": 1, "recovered": 1},
            "vitals.com": {"fetched": 3, "empty": 0},
        })
        found = evaluate_fetch(_record(), CHANDLER, stats)
        assert [(f.check, f.severity, f.detail) for f in found] == [
            ("degraded_fetch_recovered", "P3", "doctor.webmd.com"),
            ("degraded_fetch_recovered", "P3", "healthgrades.com"),
        ]
        assert found[1].observed == "healthgrades.com: 3 stripped page(s) restored by a cache-bypass refetch"
        assert "use_cache" in thresholds.FIRST_STEPS["degraded_fetch_recovered"]

    def test_a_page_the_refetch_did_not_restore_is_left_to_the_existing_checks(self):
        """Stripped and still unreadable is a real parser or vendor failure;
        the row and drift checks already speak for it, and a "restored"
        finding there would say the opposite of what happened."""
        stats = _stats(discovery={"doctor.webmd.com": {"fetched": 3, "empty": 0, "degraded": 3}})
        found = evaluate_fetch(_record(rows_wm=0), CHANDLER, stats)
        assert [(f.check, f.severity) for f in found] == [("platform_rows_zero", "P2")]
        assert evaluate_fetch(_record(), CHANDLER, _stats(discovery={"vitals.com": {"fetched": 3, "empty": 0}})) == []

    def test_the_pipeline_canary_counts_profiles_restored_during_enrichment(self):
        record = _record(
            shortlist_size=5, validation_collapsed=False, n_no_profile_found=1, n_enriched=7,
            coverage_hg=5, coverage_wm=6, coverage_vi=6, judge_truncated=False,
            critic_shards_failed=0, cost_usd=0.56, latency_s=70.0,
        )
        stats = _stats(
            discovery={"doctor.webmd.com": {"fetched": 3, "empty": 0, "degraded": 1, "recovered": 1}},
            enrichment={"doctor.webmd.com": {"fetched": 8, "empty": 0, "degraded": 2, "recovered": 2},
                        "vitals.com": {"fetched": 8, "empty": 0, "degraded": 1, "recovered": 1}},
        )
        found = evaluate_pipeline(record, CHANDLER, [], "success", stats)
        assert [(f.check, f.detail, f.observed.split(":")[1].split()[0]) for f in found] == [
            ("degraded_fetch_recovered", "doctor.webmd.com", "3"),
            ("degraded_fetch_recovered", "vitals.com", "1"),
        ]

    def test_each_tier_that_can_raise_it_can_also_close_it(self):
        assert "degraded_fetch_recovered" in checks_for_tier("A") & checks_for_tier("B") & thresholds.GOLDEN_CHECKS

    def test_the_golden_check_reports_restored_pages_beside_its_grades(self, tmp_path):
        """The four golden profiles that read as template drift on 11-22 now
        grade ok — and the run still says the vendor served them stripped."""
        whole = (golden.FIXTURE_DIR / "vitals" / "hemant-pandey.md").read_text(encoding="utf-8")
        gatherer = _gatherer({VI_PROFILE: whole}, {VI_PROFILE: VI_PROFILE_STRIPPED}, {VI_PROFILE: whole})
        paths = dict(key_path=tmp_path / "key.json", fixture_dir=tmp_path / "fixtures", bodies_dir=tmp_path / "bodies")
        gatherer.gather_providers = lambda *args, **kwargs: {"status": "success", "providers": [{
            "name": "Dr. Hemant Kumar Pandey", "specialty": "Neurology",
            "platform_profile_urls": {"www.vitals.com": VI_PROFILE},
        }]}
        refresh_fixtures.build("chandler-neurology", min_platforms=1, agent_factory=lambda: gatherer, now=NOW, **paths)
        gatherer._reset_run_stats()

        def fetch(urls):
            return {p["url"]: p["raw_content"] for p in gatherer._extract_pages(urls, purpose="golden set")}
        fetch.stats = gatherer.fetch_stats

        payload = golden_check.run(
            key_path=paths["key_path"], fixture_dir=paths["fixture_dir"], bodies_dir=tmp_path / "check_bodies",
            out_path=tmp_path / "golden.json", fetcher=fetch, now=NOW,
        )
        assert (payload["counts"]["ok"], payload["counts"]["template_drift"]) == (1, 0)
        written = json.loads((tmp_path / "golden.json").read_text())
        assert written["fetch"] == {"degraded": 1, "recovered": 1, "by_domain": {"vitals.com": {"degraded": 1, "recovered": 1}}}
        assert [(f.check, f.severity, f.case_id, f.detail) for f in evaluate_golden(written)] == [
            ("degraded_fetch_recovered", "P3", thresholds.GOLDEN_CASE_ID, "vitals.com"),
        ]
        assert any("arrived stripped" in line and "restored 1" in line for line in golden_check.describe(written))

    def test_a_golden_run_with_nothing_stripped_writes_no_fetch_block(self, tmp_path):
        """A healthy week's file is byte-for-byte what it was before: the
        block appears only when there is something to report, and a fetcher
        that replays saved bodies has no stats to ask for."""
        def fetch(urls):
            return {}
        fetch.stats = lambda: {"fetch": {"discovery": {"degraded": 0, "recovered": 0, "by_domain": {}}}}
        assert golden_check._refetch_counts(fetch) is None
        assert golden_check._refetch_counts(lambda urls: {}) is None

        def broken(urls):
            return {}
        broken.stats = MagicMock(side_effect=RuntimeError("no stats"))
        assert golden_check._refetch_counts(broken) is None
        assert evaluate_golden({"pages": []}) == []

    def test_the_production_golden_fetcher_carries_the_gatherers_counts(self, monkeypatch):
        """The wiring the weekly run actually uses: without `fetch.stats` on
        the default fetcher the golden file never gains its fetch block and
        a restored page goes unreported, with every other test still green."""
        gatherer = _gatherer({VI_PROFILE: VI_PROFILE_STRIPPED}, {VI_PROFILE: VI_PROFILE_WHOLE})
        monkeypatch.setattr("agents.data_gatherer.create_data_gatherer", lambda: gatherer)
        fetch = golden_check._default_fetcher()
        assert fetch([VI_PROFILE]) == {VI_PROFILE: VI_PROFILE_WHOLE}
        assert golden_check._refetch_counts(fetch) == {
            "degraded": 1, "recovered": 1, "by_domain": {"vitals.com": {"degraded": 1, "recovered": 1}},
        }


RESTORED = thresholds.Finding(
    "degraded_fetch_recovered", "P3", "chandler-neurology",
    "healthgrades.com: 2 stripped page(s) restored by a cache-bypass refetch", "0 stripped pages",
    detail="healthgrades.com",
)
HG_ZERO = thresholds.Finding("platform_rows_zero", "P2", "chandler-neurology", "healthgrades: 0 rows", "≥ 1 row", detail="healthgrades")


class _Issues:
    """An in-memory Issues API behind the alerts transport seam."""

    def __init__(self):
        self.issues, self.comments, self.patches = [], [], []

    def __call__(self, method, path, body, params):
        if "/labels/" in path:
            return 200, {}
        if path.endswith("/issues") and method == "GET":
            return 200, [i for i in self.issues if i["state"] == "open"] if params["page"] == 1 else []
        if path.endswith("/issues") and method == "POST":
            issue = {"number": 100 + len(self.issues), "state": "open", **body}
            self.issues.append(issue)
            return 201, issue
        if path.endswith("/comments"):
            self.comments.append((int(path.split("/")[-2]), body["body"]))
            return 201, {}
        if method == "PATCH":
            for issue in self.issues:
                if issue["number"] == int(path.split("/")[-1]):
                    issue.update({k: body[k] for k in ("state", "body") if k in body})
                    self.patches.append((issue["number"], sorted(body)))
            return 200, {}
        raise AssertionError(f"unexpected call {method} {path}")


class TestRestoredPageIsAdvisory:
    """A restored page is reported and tracked, and never fails the run.

    The first live runs with the refetch in place read 258 rows from 10 of 10
    pages and exited 1 anyway, because the watcher failed on ANY finding and
    a restored page is one. With the vendor's stripped copies in place for
    days, that is a red run, a failure email and an issue comment every
    morning for something nobody here can fix, on runs whose numbers are
    whole. These pin the softer treatment and, as importantly, its limit.
    """

    def _tier_a_file(self, tmp_path, record=None):
        from tests.unit.test_canary_tier_a import entry_for, payload_for
        entry = entry_for(CHANDLER, record)
        entry["fetch_stats"]["fetch"]["discovery"].update(
            degraded=2, recovered=2, by_domain={"healthgrades.com": {"degraded": 2, "recovered": 2}})
        path = tmp_path / "canary_fetch.json"
        path.write_text(json.dumps(payload_for([entry])))
        return path

    def test_a_run_whose_only_finding_is_a_restored_page_exits_green(self, tmp_path, capsys):
        from evals import watch
        path = self._tier_a_file(tmp_path)
        assert watch.main(["--tier", "A", "--schedule", "daily", "--input", str(path), "--no-issues"], {}) == 0
        out = capsys.readouterr().out
        assert "Status: passing with 1 advisory finding(s)" in out
        assert "| P3 (advisory) | Stripped page restored by refetch | chandler-neurology |" in out, (
            "green must not mean silent: the finding and its first step stay on the run page")
        assert "Send the vendor the URLs" in out

    def test_a_real_breach_beside_a_restored_page_still_fails_the_run(self, tmp_path, capsys):
        """The limit: advisory findings are subtracted from the exit code,
        never the other way round. The same file with the pool under its
        floor is red, and the headline names the failing severity only."""
        from evals import watch
        from tests.unit.test_canary_tier_a import baseline_record
        path = self._tier_a_file(tmp_path, baseline_record(pool_raw=38))
        assert watch.main(["--tier", "A", "--schedule", "daily", "--input", str(path), "--no-issues"], {}) == 1
        out = capsys.readouterr().out
        assert "Status: P2 — 1 finding(s)" in out and "Plus 1 advisory finding(s) that do not fail the run." in out

    def test_only_the_restored_page_check_is_advisory(self):
        """The set is one check wide on purpose. The checks that fire when
        the refetch STOPS restoring pages must keep failing the run, or the
        softening would hide the outage it was built around."""
        assert thresholds.ADVISORY_CHECKS == frozenset({"degraded_fetch_recovered"})
        assert RESTORED.advisory and not HG_ZERO.advisory
        assert thresholds.failing_findings([RESTORED, HG_ZERO]) == [HG_ZERO]
        for check in ("platform_rows_zero", "template_drift", "pool_size_low", "canary_did_not_run"):
            assert check in thresholds.ALL_CHECKS and check not in thresholds.ADVISORY_CHECKS

    def _restored(self, case_id, domain, count=2):
        return thresholds.Finding(
            "degraded_fetch_recovered", "P3", case_id,
            f"{domain}: {count} stripped page(s) restored by a cache-bypass refetch", "0 stripped pages", detail=domain)

    def test_every_restored_page_shares_one_issue_that_is_rewritten_in_place(self):
        """The first public run opened TWELVE issues for one vendor condition
        (four cases and the golden set, up to three platforms each) and the
        one finding that needed a person sat in the middle of them. One
        condition is one issue: opened once, its table rewritten on every
        run, never commented on, closed once."""
        from evals import alerts
        gh = _Issues()
        client = alerts.GitHubIssues("tok", "o/r", transport=gh)
        weekly = {c: checks_for_tier("A") for c in ("chandler-neurology", "phoenix-cardiology")}
        found = [self._restored("chandler-neurology", "healthgrades.com", 4),
                 self._restored("chandler-neurology", "doctor.webmd.com"),
                 self._restored("phoenix-cardiology", "healthgrades.com"), HG_ZERO]
        first = alerts.sync_issues(client, found, weekly, "https://gh/run/1", now=NOW)
        assert sorted(i["title"] for i in gh.issues) == sorted([
            "[P3] Stripped page restored by refetch — all-cases", HG_ZERO.title]), "three restored findings, one issue"
        assert first.opened.count("[P3] Stripped page restored by refetch — all-cases") == 1
        rollup = next(i for i in gh.issues if "all-cases" in i["title"])
        for cell in ("| chandler-neurology | healthgrades.com | healthgrades.com: 4 stripped page(s)",
                     "| chandler-neurology | doctor.webmd.com |", "| phoenix-cardiology | healthgrades.com |",
                     "It is advisory: it does not fail the run", "Send the vendor the URLs"):
            assert cell in rollup["body"]

        # The DAILY run evaluates Chandler only: webmd came back clean there,
        # healthgrades did not. Phoenix was not looked at and must keep its row.
        later = datetime(2026, 11, 24, 13, 0, tzinfo=timezone.utc)
        daily = {"chandler-neurology": checks_for_tier("A")}
        second = alerts.sync_issues(client, [self._restored("chandler-neurology", "healthgrades.com", 1), HG_ZERO],
                                    daily, "https://gh/run/2", now=later)
        assert second.opened == [] and second.updated == [rollup["title"]] and second.commented == [HG_ZERO.title]
        assert len(gh.issues) == 2 and gh.patches == [(rollup["number"], ["body"])]
        assert [n for n, _ in gh.comments] == [next(i["number"] for i in gh.issues if i is not rollup)], (
            "the failing finding is commented on; the rolled-up issue never is while it is open")
        body = rollup["body"]
        assert "healthgrades.com: 1 stripped page(s)" in body and "| 2026-11-24 13:00 UTC |" in body
        assert "| chandler-neurology | doctor.webmd.com |" not in body, "an evaluated case that came back clean loses its row"
        assert "| phoenix-cardiology | healthgrades.com |" in body and "| 2026-11-23 13:00 UTC |" in body, (
            "a case this run did not evaluate keeps its row and its own last-seen time")
        assert "- Opened: 2026-11-23 13:00 UTC" in body and "- Updated: 2026-11-24 13:00 UTC" in body
        assert "Latest run: https://gh/run/2" in body

        # Chandler clean on the next daily run: Phoenix still holds the issue open.
        third = alerts.sync_issues(client, [], daily, "https://gh/run/3", now=later)
        assert rollup["state"] == "open" and "chandler-neurology" not in rollup["body"].split("First step")[0]
        assert third.closed == [HG_ZERO.title]
        # The run that evaluates Phoenix and finds it clean closes it, once.
        fourth = alerts.sync_issues(client, [], weekly, "https://gh/run/4", now=later)
        assert rollup["state"] == "closed" and fourth.closed == [rollup["title"]]
        assert gh.comments[-1] == (rollup["number"], alerts.recovery_comment("https://gh/run/4", later))

    def test_the_older_per_case_issues_are_folded_into_the_one_issue(self):
        """What the public repository holds today: one advisory issue per
        case and platform, opened before the roll-up, including a pair of
        duplicates from two runs started the same second. They are closed
        into the single issue whatever this run evaluated — a Gilbert issue
        must not wait for Sunday to disappear."""
        from evals import alerts
        gh = _Issues()
        old = [self._restored("chandler-neurology", "healthgrades.com"),
               self._restored("chandler-neurology", "healthgrades.com"),
               self._restored("gilbert-family-medicine", "doctor.webmd.com")]
        for number, finding in enumerate(old, start=28):
            gh.issues.append({"number": number, "state": "open", "title": finding.title, "labels": ["canary"],
                              "body": alerts.issue_body(finding, None, None, NOW)})
        gh.issues.append({"number": 34, "state": "open", "title": HG_ZERO.title, "labels": ["canary"],
                          "body": alerts.issue_body(HG_ZERO, None, None, NOW)})
        client = alerts.GitHubIssues("tok", "o/r", transport=gh)
        result = alerts.sync_issues(client, [self._restored("chandler-neurology", "healthgrades.com"), HG_ZERO],
                                    {"chandler-neurology": checks_for_tier("A")}, "https://gh/run/9", now=NOW)
        still_open = sorted(i["title"] for i in gh.issues if i["state"] == "open")
        assert still_open == sorted(["[P3] Stripped page restored by refetch — all-cases", HG_ZERO.title])
        assert len(result.folded) == 3 and "folded into it 3" in result.describe()
        rollup = next(i for i in gh.issues if "all-cases" in i["title"])
        assert (28, f"This finding is now tracked in #{rollup['number']}, which is updated in place on every run.") in gh.comments
        assert result.untouched == 0, "the folded issues are not also counted as left open"

    def test_a_second_roll_up_is_folded_and_an_unreadable_one_costs_only_its_rows(self):
        from evals import alerts
        gh = _Issues()
        finding = alerts.rollup_finding("degraded_fetch_recovered", {})
        for number, body in ((40, "table lost in a hand edit\n" + alerts.marker(finding)),
                             (41, alerts.rollup_body("degraded_fetch_recovered", {"x|y": {"observed": "o", "seen": "s"}}, "then", None, NOW))):
            gh.issues.append({"number": number, "state": "open", "title": finding.title, "labels": ["canary"], "body": body})
        client = alerts.GitHubIssues("tok", "o/r", transport=gh)
        result = alerts.sync_issues(client, [self._restored("chandler-neurology", "vitals.com")],
                                    {"chandler-neurology": checks_for_tier("A")}, None, now=NOW)
        assert [i["state"] for i in gh.issues] == ["open", "closed"] and result.errors == []
        assert "| chandler-neurology | vitals.com |" in gh.issues[0]["body"]
        assert alerts.rollup_entries(gh.issues[0]) == {
            "chandler-neurology|vitals.com": {"observed": self._restored("chandler-neurology", "vitals.com").observed,
                                              "seen": "2026-11-23 13:00 UTC"}}
