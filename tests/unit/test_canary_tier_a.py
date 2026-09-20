"""Observability P2: the Tier A fetch canary, its thresholds, watcher and alerts.

The September 2026 Tavily degradation went unseen for three weeks while
every number that would have caught it in a day was computed and thrown
away. These tests pin the monitor that now keeps them: thresholds fire at
the MEASURED breach and not at the baseline (a floor that fires on a
healthy day trains the reader to ignore it), one outage is one finding
rather than a cascade of consequences, the watcher treats a canary that
did not run as the P1 it is, issues open on breach / comment on repeat /
close only for checks the run actually evaluated, the canary runner
records a crashed case and keeps going, vendor exceptions are captured
with their CLASS (log scraping cannot see it), and the workflow file keeps
the shape the alert path depends on.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evals import alerts, canary_fetch, summary, thresholds, vendor_watch, watch
from evals.cases import CASES, case_by_id, cases_for
from evals.thresholds import (
    Finding,
    checks_for_tier,
    evaluate_fetch,
    evaluate_pipeline,
    finding_from_dict,
    worst_severity,
)
from tests.helpers.fake_langfuse import FakeLangfuse, by_name, install_fake_tracing
from utils.security import validate_search_params

REPO = Path(__file__).resolve().parents[2]
CHANDLER = case_by_id("chandler-neurology")
PHOENIX = case_by_id("phoenix-cardiology")
NOW = datetime(2026, 9, 13, 13, 0, tzinfo=timezone.utc)


def baseline_record(**over):
    """The 2026-09-13 live Chandler measurement (pool 117, rows 73/114/105, 3 credits)."""
    record = {
        "pages_planned": 11, "pages_fetched": 10, "empty_bodies": 0, "pages_failed": 1,
        "rows_hg": 73, "rows_wm": 114, "rows_vi": 105, "pool_raw": 117,
        "fallback_fired": False, "ring_fired": False, "tavily_credits": 3,
        "cost_usd": 0.024, "latency_s": 7.5,
    }
    record.update(over)
    return record


def healthy_stats():
    return {
        "fetch": {"discovery": {"planned": 11, "fetched": 10, "empty": 0, "failed": 1, "by_domain": {
            "healthgrades.com": {"fetched": 4, "empty": 0},
            "doctor.webmd.com": {"fetched": 3, "empty": 0},
            "vitals.com": {"fetched": 3, "empty": 0},
        }}},
        "listing_rows": {"healthgrades.com": 73, "doctor.webmd.com": 114, "vitals.com": 105},
        "specialty_rows_dropped": 39,
    }


def pipeline_record(**over):
    record = baseline_record(
        shortlist_size=5, validation_collapsed=False, n_no_profile_found=1, n_enriched=7,
        coverage_hg=5, coverage_wm=6, coverage_vi=6, judge_truncated=False,
        critic_shards_failed=0, cost_usd=0.56, latency_s=70.0,
    )
    record.update(over)
    return record


def checks(findings):
    return [f.check for f in findings]


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

class TestCases:
    def test_schedules_partition_the_cases(self):
        assert [c.case_id for c in cases_for("daily")] == ["chandler-neurology"]
        assert len(cases_for("weekly")) == 3 and len(cases_for("all")) == len(CASES) == 4
        assert len({c.case_id for c in CASES}) == 4
        with pytest.raises(ValueError):
            cases_for("hourly")

    def test_phoenix_does_not_expect_vitals(self):
        """The known vitals stub cell must not be a permanent alert."""
        assert "vi" not in PHOENIX.platforms_expected and CHANDLER.platforms_expected == ("hg", "wm", "vi")
        assert all(c.pool_floor > 0 for c in CASES)
        assert case_by_id("nowhere") is None

    def test_every_case_is_a_search_the_gatherer_would_accept(self):
        """A case the input gate refuses can never measure anything.

        `sun-lakes-neurology` shipped with "Sun Lakes, AZ" — a real place the
        vendored GeoNames dataset does not name — so `gather_providers`
        returned `invalid_input` before one page was fetched, and the weekly
        run spent a P1 telling us about our own case list. Nothing offline
        said so: every runner test in this file mocks the gatherer, and the
        gate lives behind exactly that seam. `validate_search_params` is what
        `gather_providers` runs on entry; a case list is an input to it.

        The canonical check is the second half: the dataset echoes its own
        spelling, so a case whose location merely RESOLVES (a stray ZIP, a
        casing variant) would have the canary searching a string the reports
        and issue titles do not name.
        """
        for case in CASES:
            result = validate_search_params(case.specialty, case.location)
            assert result["is_valid"], f"{case.case_id}: {[e for e in result['errors'] if e]}"
            assert result["location"] == case.location, (
                f"{case.case_id}: {case.location!r} is not the dataset's own "
                f"spelling of itself ({result['location']!r})"
            )
            assert result["specialty"] == case.specialty, (
                f"{case.case_id}: {case.specialty!r} is not the allowlist's "
                f"spelling ({result['specialty']!r})"
            )


# --------------------------------------------------------------------------
# Thresholds — Tier A
# --------------------------------------------------------------------------

class TestFetchThresholds:
    def test_baseline_is_clean(self):
        assert evaluate_fetch(baseline_record(), CHANDLER, healthy_stats(), [], "success") == []

    def test_pool_floor_fires_at_the_outage_value_not_the_baseline(self):
        """Aug 12 the pool fell 120 → 38; the floor of 60 catches it on day one."""
        found = evaluate_fetch(baseline_record(pool_raw=38), CHANDLER, healthy_stats(), [], "success")
        assert [(f.check, f.severity, f.observed, f.threshold) for f in found] == [("pool_size_low", "P2", "38", "≥ 60")]
        assert evaluate_fetch(baseline_record(pool_raw=60), CHANDLER)== []

    def test_platform_rows_zero_respects_the_case_exemption(self):
        chandler = evaluate_fetch(baseline_record(rows_vi=0), CHANDLER, healthy_stats(), [], "success")
        assert [(f.check, f.detail) for f in chandler] == [("platform_rows_zero", "vitals")]
        assert chandler[0].title == "[P2] Platform listing rows zero — chandler-neurology (vitals)"
        phoenix = evaluate_fetch(baseline_record(rows_vi=0), PHOENIX, healthy_stats(), [], "success")
        assert phoenix == []

    def test_fallback_and_empty_bodies(self):
        assert checks(evaluate_fetch(baseline_record(fallback_fired=True), CHANDLER)) == ["fetch_mode_fallback"]
        stats = healthy_stats()
        stats["fetch"]["discovery"]["by_domain"]["vitals.com"] = {"fetched": 2, "empty": 3}
        found = evaluate_fetch(baseline_record(), CHANDLER, stats, [], "success")
        assert [(f.check, f.detail, f.observed) for f in found] == [("empty_body_rate_high", "vitals.com", "vitals.com: 3/5 empty")]
        stats["fetch"]["discovery"]["by_domain"]["vitals.com"] = {"fetched": 0, "empty": 1}
        assert evaluate_fetch(baseline_record(), CHANDLER, stats) == [], "one page is not a rate"

    def test_nothing_fetched_is_one_p1_not_a_cascade(self):
        record = baseline_record(pages_fetched=0, pool_raw=0, rows_hg=0, rows_wm=0, rows_vi=0, fallback_fired=True)
        found = evaluate_fetch(record, CHANDLER, healthy_stats(), [], "no_results")
        assert [(f.check, f.severity, f.observed) for f in found] == [("no_pages_fetched", "P1", "0 of 11 pages")]

    def test_fatal_vendor_error_explains_the_failed_gather(self):
        errors = [{"vendor": "tavily", "error": "UsageLimitExceededError", "message": "Usage limit exceeded", "kind": "fatal"}]
        found = evaluate_fetch({}, CHANDLER, None, errors, "error")
        assert [(f.check, f.severity) for f in found] == [("vendor_error", "P1")]
        assert "tavily: UsageLimitExceededError — Usage limit exceeded" in found[0].observed

    def test_fatal_errors_from_two_vendors_are_one_finding(self):
        errors = [
            {"vendor": "tavily", "error": "InvalidAPIKeyError", "message": "", "kind": "fatal"},
            {"vendor": "tavily", "error": "InvalidAPIKeyError", "message": "", "kind": "fatal"},
            {"vendor": "anthropic", "error": "AuthenticationError", "message": "invalid x-api-key", "kind": "fatal"},
        ]
        found = evaluate_fetch(baseline_record(), CHANDLER, None, errors, "success")
        assert len(found) == 1 and found[0].observed == "tavily: InvalidAPIKeyError; anthropic: AuthenticationError — invalid x-api-key"

    def test_transient_errors_count_only_when_nothing_came_back(self):
        errors = [{"vendor": "tavily", "error": "TimeoutError", "message": "timed out", "kind": "transient"}]
        assert evaluate_fetch(baseline_record(), CHANDLER, None, errors, "success") == []
        found = evaluate_fetch({"pages_planned": 11, "pages_fetched": 0}, CHANDLER, None, errors, "no_results")
        assert checks(found) == ["vendor_error"] and "1 transient error(s) and nothing fetched" in found[0].observed

    def test_gather_failed_without_a_vendor_explanation(self):
        for status in ("error", "invalid_input", "crashed"):
            found = evaluate_fetch({}, CHANDLER, None, [], status)
            assert [(f.check, f.severity, f.observed) for f in found] == [("gather_failed", "P1", f"status={status}")]
        # A pool of zero is a measurement, not a failed gather.
        found = evaluate_fetch(baseline_record(pool_raw=0), CHANDLER, healthy_stats(), [], "no_results")
        assert checks(found) == ["pool_size_low", ]

    def test_finding_identity_title_and_roundtrip(self):
        f = Finding("pool_size_low", "P2", "chandler-neurology", "38", "≥ 60")
        assert f.title == "[P2] Pool size low — chandler-neurology" and f.identity == ("pool_size_low", "chandler-neurology", "")
        assert f.first_step.startswith("Open the trace's discovery step")
        assert finding_from_dict(f.as_dict()) == f
        assert f.as_dict()["title"] == f.title


# --------------------------------------------------------------------------
# Thresholds — Tier B
# --------------------------------------------------------------------------

class TestPipelineThresholds:
    def test_clean_pipeline_record(self):
        assert evaluate_pipeline(pipeline_record(), CHANDLER, [], "success") == []

    def test_p1_conditions(self):
        assert checks(evaluate_pipeline(pipeline_record(shortlist_size=0), CHANDLER)) == ["no_shortlist"]
        assert checks(evaluate_pipeline(pipeline_record(validation_collapsed=True), CHANDLER)) == ["validation_collapsed"]
        assert all(f.severity == "P1" for f in evaluate_pipeline(pipeline_record(shortlist_size=0, validation_collapsed=True), CHANDLER))

    def test_p2_conditions(self):
        assert checks(evaluate_pipeline(pipeline_record(n_no_profile_found=3), CHANDLER)) == ["no_profile_found_high"]
        assert evaluate_pipeline(pipeline_record(n_no_profile_found=2), CHANDLER) == []
        found = evaluate_pipeline(pipeline_record(coverage_wm=0), CHANDLER)
        assert [(f.check, f.detail) for f in found] == [("platform_coverage_zero", "webmd")]
        assert evaluate_pipeline(pipeline_record(coverage_vi=0), PHOENIX) == []
        assert evaluate_pipeline(pipeline_record(n_enriched=None, coverage_wm=0), CHANDLER) == [], "unmeasured never fires"
        assert checks(evaluate_pipeline(pipeline_record(judge_truncated=True), CHANDLER)) == ["judge_truncated"]
        assert checks(evaluate_pipeline(pipeline_record(critic_shards_failed=2), CHANDLER)) == ["critic_shard_failed"]

    def test_p3_bands(self):
        for cost in (1.20, 0.10):
            found = evaluate_pipeline(pipeline_record(cost_usd=cost), CHANDLER)
            assert [(f.check, f.severity) for f in found] == [("cost_out_of_band", "P3")]
        assert evaluate_pipeline(pipeline_record(cost_usd=0.90), CHANDLER) == []
        found = evaluate_pipeline(pipeline_record(latency_s=200), CHANDLER)
        assert [(f.check, f.severity, f.observed) for f in found] == [("latency_high", "P3", "200 s")]

    def test_failed_pipeline_short_circuits_the_model_checks(self):
        """A workflow that did not complete is one P1 with its own first step, not a
        failed gather plus an empty shortlist plus a missing platform."""
        found = evaluate_pipeline(pipeline_record(shortlist_size=0, coverage_wm=0), CHANDLER, [], "error")
        assert [(f.check, f.severity, f.observed) for f in found] == [("pipeline_failed", "P1", "status=error")]

    def test_severity_order_and_tier_scopes(self):
        findings = [Finding("latency_high", "P3", "c", "", ""), Finding("no_shortlist", "P1", "c", "", "")]
        assert worst_severity(findings) == "P1" and worst_severity([]) is None
        assert "no_shortlist" not in checks_for_tier("A") and "canary_did_not_run" in checks_for_tier("A")
        assert checks_for_tier("B") > checks_for_tier("A")
        assert set(thresholds.CHECK_TITLES) == set(thresholds.FIRST_STEPS) == thresholds.ALL_CHECKS
        assert thresholds.ALL_CHECKS == checks_for_tier("B") | thresholds.REPORT_CHECKS | thresholds.GOLDEN_CHECKS
        with pytest.raises(ValueError):
            checks_for_tier("C")


# --------------------------------------------------------------------------
# Vendor watch
# --------------------------------------------------------------------------

class TestVendorWatch:
    def test_classification_by_class_then_message(self):
        from tavily.errors import UsageLimitExceededError

        class BadRequestError(Exception):
            pass

        class RateLimitError(Exception):
            pass

        assert vendor_watch.classify(UsageLimitExceededError("x")) == "fatal"
        assert vendor_watch.classify(BadRequestError("Your credit balance is too low to access the Anthropic API")) == "fatal"
        assert vendor_watch.classify(RateLimitError("insufficient_quota")) == "fatal"
        assert vendor_watch.classify(RateLimitError("rate limited, retry after 3s")) == "transient"
        assert vendor_watch.classify(TimeoutError("read timed out")) == "transient"

    def test_redaction_scrubs_configured_secrets_and_token_shapes(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "supersecretvalue123")
        text = "key supersecretvalue123 rejected; also sk-ant-abcdefghijklmnop and tvly-zzzzzzzzzzzz"
        out = vendor_watch.redact(text)
        assert "supersecretvalue123" not in out and "sk-ant-" not in out and "tvly-" not in out
        assert out.count("***") == 3
        long = vendor_watch.redact("x" * 500)
        assert len(long) == vendor_watch.MESSAGE_CHARS + 1 and long.endswith("…")

    def test_proxy_records_and_reraises_including_nested_resources(self):
        from tavily.errors import UsageLimitExceededError

        class Tavily:
            api_key = "tvly-not-a-secret-shape-1234"

            def extract(self, urls, **kw):
                raise UsageLimitExceededError("Usage limit exceeded")

        class Messages:
            def create(self, **kw):
                raise TimeoutError("read timed out")

        class Anthropic:
            def __init__(self):
                self.messages = Messages()

        class Agent:
            def __init__(self):
                self.tavily_client = Tavily()
                self.anthropic_client = Anthropic()

        agent, w = Agent(), vendor_watch.VendorWatch()
        w.wrap_gatherer(agent)
        assert agent.tavily_client and agent.tavily_client.api_key == "tvly-not-a-secret-shape-1234"
        with pytest.raises(UsageLimitExceededError):
            agent.tavily_client.extract(urls=["u"], extract_depth="basic")
        with pytest.raises(TimeoutError):
            agent.anthropic_client.messages.create(model="m")
        assert [(e["vendor"], e["error"], e["kind"]) for e in w.errors] == [
            ("tavily", "UsageLimitExceededError", "fatal"), ("anthropic", "TimeoutError", "transient"),
        ]
        assert len(w.fatal) == 1 and len(w.transient) == 1

    def test_wrap_skips_missing_clients_and_passes_setattr_through(self):
        class Agent:
            tavily_client = None
            anthropic_client = None

        agent, w = Agent(), vendor_watch.VendorWatch()
        w.wrap_gatherer(agent)
        assert agent.tavily_client is None and w.errors == []

        class Client:
            flag = 1

        proxy = w.wrap(Client(), "tavily")
        proxy.flag = 2
        assert proxy.flag == 2 and "recording tavily" in repr(proxy)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def payload_for(entries, tier="A", schedule="daily"):
    return {"tier": tier, "schedule": schedule, "ts": NOW.isoformat(), "cases": entries}


def entry_for(case, record=None, status="success", trace="https://lf.example/t/1", **extra):
    entry = {
        "case_id": case.case_id, "status": status, "trace_url": trace,
        "record": baseline_record() if record is None else record,
        "fetch_stats": healthy_stats(), "vendor_errors": [], "error": None, "elapsed_s": 7.5,
    }
    entry.update(extra)
    return entry


class TestSummary:
    def test_breach_summary_leads_with_severity_then_first_steps_then_cases(self):
        findings = [Finding("pool_size_low", "P2", "chandler-neurology", "38", "≥ 60")]
        text = summary.render_summary("A", "daily", payload_for([entry_for(CHANDLER, baseline_record(pool_raw=38))]),
                                      findings, (CHANDLER,), "https://gh/run/1", ["issue sync skipped"], NOW)
        assert text.startswith("## Tier A fetch canary — daily · 2026-09-13 13:00 UTC")
        assert "**Status: P2 — 1 finding(s)**" in text and "Run: https://gh/run/1" in text
        assert "| P2 | Pool size low | chandler-neurology | 38 | ≥ 60 |" in text
        assert "- **[P2] Pool size low — chandler-neurology** — Open the trace's discovery step" in text
        assert "| chandler-neurology | success | 38 | 73 / 114 / 105 | 10 / 11 | 0 | no | no | 3 | $0.024 | 8 s | [trace](https://lf.example/t/1) |" in text
        assert text.index("### Findings") < text.index("### Cases") < text.index("### Notes")
        assert "- issue sync skipped" in text

    def test_clean_summary_and_notes_from_the_payload(self):
        entry = entry_for(CHANDLER, vendor_errors=[{"vendor": "tavily", "error": "TimeoutError", "message": "t", "kind": "transient"}])
        text = summary.render_summary("A", "daily", payload_for([entry]), [], (CHANDLER,), generated_at=NOW)
        assert "**Status: clean** — 1 of 1 scheduled case(s) evaluated" in text
        assert "### Findings" not in text
        assert "- chandler-neurology: 1 transient vendor error(s) (tavily TimeoutError)" in text
        crashed = entry_for(CHANDLER, record={}, status="crashed", trace=None, error="RuntimeError: boom")
        text = summary.render_summary("A", "daily", payload_for([crashed]), [], (CHANDLER,), generated_at=NOW)
        assert "| chandler-neurology | crashed | — | — / — / — | — / — | — | — | — | — | — | — | — |" in text
        assert "- chandler-neurology: RuntimeError: boom" in text

    def test_step_summary_appends_only_when_a_target_exists(self, tmp_path, monkeypatch):
        target = tmp_path / "summary.md"
        target.write_text("existing\n")
        assert summary.write_step_summary("## new", str(target)) is True
        assert target.read_text() == "existing\n## new\n"
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        assert summary.write_step_summary("## new") is False


# --------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------

class FakeGitHub:
    """An in-memory Issues API behind the transport seam."""

    def __init__(self, open_issues=(), label_exists=True, fail=()):
        self.issues = [dict(i) for i in open_issues]
        self.labels = {"canary"} if label_exists else set()
        self.comments = []
        self.calls = []
        self.fail = set(fail)  # method names that return HTTP 500
        self._next = 100 + len(self.issues)

    def __call__(self, method, path, json, params):
        self.calls.append((method, path))
        if path.endswith("/issues") and method == "GET":
            if "list" in self.fail:
                return 500, None
            wanted = params["labels"]
            page = [i for i in self.issues if i["state"] == "open" and wanted in i["labels"]]
            return 200, page if params["page"] == 1 else []
        if path.endswith("/issues") and method == "POST":
            if "create" in self.fail:
                return 500, None
            self._next += 1
            issue = {"number": self._next, "title": json["title"], "body": json["body"], "labels": json["labels"], "state": "open"}
            self.issues.append(issue)
            return 201, issue
        if path.endswith("/comments"):
            number = int(path.split("/")[-2])
            self.comments.append((number, json["body"]))
            return 201, {}
        if method == "PATCH":
            number = int(path.split("/")[-1])
            for issue in self.issues:
                if issue["number"] == number:
                    issue["state"] = json["state"]
            return 200, {}
        if "/labels/" in path and method == "GET":
            return (200, {}) if path.split("/")[-1] in self.labels else (404, None)
        if path.endswith("/labels") and method == "POST":
            self.labels.add(json["name"])
            return 201, {}
        raise AssertionError(f"unexpected call {method} {path}")

    def open_titles(self):
        return sorted(i["title"] for i in self.issues if i["state"] == "open")


def open_issue(number, finding):
    return {"number": number, "title": finding.title, "body": alerts.issue_body(finding, None, None, NOW),
            "labels": ["canary"], "state": "open"}


POOL_LOW = Finding("pool_size_low", "P2", "chandler-neurology", "38", "≥ 60")
PHOENIX_LOW = Finding("pool_size_low", "P2", "phoenix-cardiology", "12", "≥ 30")
NO_SHORTLIST = Finding("no_shortlist", "P1", "chandler-neurology", "0 recommendations", "≥ 1")


class TestAlerts:
    def test_breach_opens_one_issue_then_comments_on_repeat(self):
        gh = FakeGitHub()
        client = alerts.GitHubIssues("tok", "o/r", transport=gh)
        first = alerts.sync_issues(client, [POOL_LOW], {"chandler-neurology": checks_for_tier("A")},
                                   "https://gh/run/1", {"chandler-neurology": "https://lf/t1"}, now=NOW)
        assert first.opened == [POOL_LOW.title] and gh.open_titles() == ["[P2] Pool size low — chandler-neurology"]
        body = gh.issues[0]["body"]
        assert alerts.marker(POOL_LOW) in body and "Observed: 38" in body and "Trace: https://lf/t1" in body
        assert "First step: Open the trace's discovery step" in body
        second = alerts.sync_issues(client, [POOL_LOW], {"chandler-neurology": checks_for_tier("A")}, "https://gh/run/2", now=NOW)
        assert second.commented == [POOL_LOW.title] and second.opened == [] and len(gh.issues) == 1
        assert gh.comments[-1][1].startswith("Still breached at 2026-09-13 13:00 UTC: observed 38 (threshold ≥ 60)")

    def test_recovery_closes_only_what_this_run_evaluated(self):
        gh = FakeGitHub(open_issues=[open_issue(1, POOL_LOW), open_issue(2, PHOENIX_LOW), open_issue(3, NO_SHORTLIST)])
        client = alerts.GitHubIssues("tok", "o/r", transport=gh)
        result = alerts.sync_issues(client, [], {"chandler-neurology": checks_for_tier("A")}, "https://gh/run/3", now=NOW)
        assert result.closed == [POOL_LOW.title] and result.untouched == 2
        assert gh.open_titles() == sorted([PHOENIX_LOW.title, NO_SHORTLIST.title]), (
            "the daily run did not evaluate Phoenix, and a fetch-only run did not evaluate no_shortlist"
        )
        assert gh.comments == [(1, alerts.recovery_comment("https://gh/run/3", NOW))]

    def test_label_is_created_when_missing_and_identity_parses(self):
        gh = FakeGitHub(label_exists=False)
        client = alerts.GitHubIssues("tok", "o/r", transport=gh)
        assert client.ensure_label() is True and client.ensure_label() is False
        with_detail = Finding("platform_rows_zero", "P2", "chandler-neurology", "vitals: 0 rows", "≥ 1 row", detail="vitals")
        assert alerts.identity_of({"body": alerts.issue_body(with_detail, None, None)}) == with_detail.identity
        assert alerts.identity_of({"body": alerts.issue_body(POOL_LOW, None, None)}) == POOL_LOW.identity
        assert alerts.identity_of({"body": "no marker"}) is None

    def test_listing_failure_aborts_but_a_create_failure_does_not(self):
        client = alerts.GitHubIssues("tok", "o/r", transport=FakeGitHub(fail=("list",)))
        with pytest.raises(alerts.GitHubError):
            alerts.sync_issues(client, [POOL_LOW], {}, now=NOW)
        gh = FakeGitHub(open_issues=[open_issue(1, PHOENIX_LOW)], fail=("create",))
        client = alerts.GitHubIssues("tok", "o/r", transport=gh)
        result = alerts.sync_issues(client, [POOL_LOW], {"phoenix-cardiology": checks_for_tier("A")}, now=NOW)
        assert result.errors and result.opened == [] and result.closed == [PHOENIX_LOW.title]
        assert "opened 0, commented 0, closed 1" in result.describe() and "errors 1" in result.describe()

    def test_two_findings_with_one_identity_make_one_issue(self):
        gh = FakeGitHub()
        client = alerts.GitHubIssues("tok", "o/r", transport=gh)
        alerts.sync_issues(client, [POOL_LOW, POOL_LOW], {}, now=NOW)
        assert len(gh.issues) == 1


# --------------------------------------------------------------------------
# Watcher
# --------------------------------------------------------------------------

class TestWatch:
    def test_missing_case_is_a_p1_and_present_cases_get_the_full_scope(self):
        payload = payload_for([entry_for(PHOENIX)], schedule="weekly")
        findings, evaluated = watch.evaluate_results("A", cases_for("weekly"), payload)
        assert sorted((f.check, f.case_id) for f in findings) == [
            ("canary_did_not_run", "gilbert-family-medicine"), ("canary_did_not_run", "gold-canyon-neurology"),
        ]
        assert all(f.severity == "P1" and f.observed == "no result recorded" for f in findings)
        assert evaluated["phoenix-cardiology"] == set(checks_for_tier("A"))
        assert evaluated["gilbert-family-medicine"] == {"canary_did_not_run"}
        findings, _ = watch.evaluate_results("A", (CHANDLER,), None)
        assert findings[0].observed == "no results file"

    def test_exit_codes_and_step_summary(self, tmp_path, capsys):
        env = {"GITHUB_STEP_SUMMARY": str(tmp_path / "step.md")}
        assert watch.main(["--tier", "A", "--schedule", "daily", "--input", str(tmp_path / "none.json"), "--no-issues"], env) == 1
        assert "Canary did not run" in capsys.readouterr().out
        clean = tmp_path / "clean.json"
        clean.write_text(json.dumps(payload_for([entry_for(CHANDLER)])))
        assert watch.main(["--tier", "A", "--schedule", "daily", "--input", str(clean), "--no-issues"], env) == 0
        breach = tmp_path / "breach.json"
        breach.write_text(json.dumps(payload_for([entry_for(CHANDLER, baseline_record(pool_raw=38))])))
        assert watch.main(["--tier", "A", "--schedule", "daily", "--input", str(breach), "--no-issues"], env) == 1
        text = (tmp_path / "step.md").read_text()
        assert text.count("## Tier A fetch canary") == 3 and "issue sync skipped (--no-issues)" in text
        assert watch.main(["--case", "nowhere", "--no-issues"], env) == 2

    def test_issue_sync_is_wired_with_the_evaluated_scope(self, tmp_path, monkeypatch, capsys):
        seen = {}

        class StubClient:
            def __init__(self, token, repo):
                seen["client"] = (token, repo)

        def stub_sync(client, findings, evaluated, run_url, trace_urls):
            seen.update(findings=findings, evaluated=evaluated, run_url=run_url, trace_urls=trace_urls)
            return alerts.SyncResult(opened=[f.title for f in findings])

        monkeypatch.setattr(watch, "GitHubIssues", StubClient)
        monkeypatch.setattr(watch, "sync_issues", stub_sync)
        path = tmp_path / "r.json"
        path.write_text(json.dumps(payload_for([entry_for(CHANDLER, baseline_record(pool_raw=38))])))
        env = {"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "o/r", "GITHUB_SERVER_URL": "https://gh", "GITHUB_RUN_ID": "9"}
        assert watch.main(["--tier", "A", "--input", str(path)], env) == 1
        assert seen["client"] == ("t", "o/r") and seen["run_url"] == "https://gh/o/r/actions/runs/9"
        assert seen["evaluated"] == {"chandler-neurology": set(checks_for_tier("A"))}
        assert seen["trace_urls"] == {"chandler-neurology": "https://lf.example/t/1"}
        assert "issues: opened 1" in capsys.readouterr().out
        # No token → the sync is skipped, not attempted.
        assert watch.main(["--tier", "A", "--input", str(path)], {}) == 1
        assert "issue sync skipped (GITHUB_TOKEN" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Canary runner
# --------------------------------------------------------------------------

class HealthyTavily:
    def extract(self, **kw):
        return {"results": []}


class FakeGatherer:
    """Returns a canned extract-mode discovery; bills 3 credits like the real one."""

    def __init__(self, status="success", raise_in_gather=None, tavily=None):
        self.status = status
        self.raise_in_gather = raise_in_gather
        self.tavily_client = tavily or HealthyTavily()
        self.anthropic_client = object()
        self.calls = []

    def gather_providers(self, specialty, location, insurance=None, enrich=True, radius_miles=None):
        from utils.cost_tracker import get_cost_tracker

        self.calls.append((specialty, location, enrich, radius_miles))
        if self.raise_in_gather:
            raise self.raise_in_gather
        get_cost_tracker().record_tavily_extract(11, agent="data_gatherer", stage="discovery")
        try:
            self.tavily_client.extract(urls=["u"])
        except Exception:
            return {"providers": [], "search_metadata": {"total_found": 0}, "status": "error", "message": "Error gathering provider data."}
        providers = [{"name": f"Dr. {i}", "specialty": specialty} for i in range(117)]
        return {
            "providers": providers,
            "search_metadata": {
                "total_found": 117, "radius_dropped": 1, "radius_miles": radius_miles,
                "fetch_mode": "extract", "fetch_mode_fallback": False, "ring_expanded": False, "ring_added": 0,
            },
            "status": self.status,
            "message": "Found 117 Neurology providers",
        }

    def fetch_stats(self):
        return healthy_stats()


class TestCanaryRunner:
    def test_a_traced_case_records_the_yield_and_leaves_the_plan_shaped_trace(self, tmp_path, monkeypatch):
        captured = install_fake_tracing(monkeypatch, env="ci")
        gatherer = FakeGatherer()
        payload = canary_fetch.run("daily", out_path=tmp_path / "out.json", agent_factory=lambda: gatherer)
        assert gatherer.calls == [("Neurology", "Chandler, AZ", False, 25.0)]
        entry = payload["cases"][0]
        record = entry["record"]
        assert entry["status"] == "success" and entry["error"] is None and entry["vendor_errors"] == []
        assert record["pool_raw"] == 118 and record["pool_after_radius"] == 117 and record["radius_dropped"] == 1
        assert (record["rows_hg"], record["rows_wm"], record["rows_vi"]) == (73, 114, 105)
        assert record["pages_planned"] == 11 and record["pages_fetched"] == 10 and record["tavily_credits"] == 3
        assert record["credits_discovery"] == 3 and record["cost_usd"] == pytest.approx(0.024)
        assert record["source"] == "canary" and record["case_id"] == "chandler-neurology"
        assert record["extra"]["tier"] == "A" and record["extra"]["specialty_rows_dropped"] == 39
        assert entry["run_id"].startswith("canary-chandler-neurology-") and entry["trace_url"].endswith(entry["run_id"])
        # The trace: root tagged as a canary, the gather step under it, scores attached.
        client = FakeLangfuse.instances[-1]
        assert {"source:canary", "tier:A", "case:chandler-neurology"} <= set(captured["tags"])
        assert any(t.startswith("mode:") for t in captured["tags"])
        assert captured["metadata"]["source"] == "canary" and captured["metadata"]["case_id"] == "chandler-neurology"
        step = by_name(client, "step.gather_data")[0]
        assert step.ended and step.merged["output"]["status"] == "success"
        scores = {s["name"]: s["value"] for s in client.scores}
        assert scores["pool_raw"] == 118 and scores["rows_wm"] == 114 and scores["pages_fetched"] == 10
        assert by_name(client, "carecompass.search")[0].ended
        written = json.loads((tmp_path / "out.json").read_text())
        assert written["tier"] == "A" and written["cases"][0]["record"]["pool_raw"] == 118

    def test_an_untraced_case_still_writes(self, tmp_path, monkeypatch):
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        from utils import tracing

        tracing.reset_client()
        payload = canary_fetch.run("daily", out_path=tmp_path / "o.json", agent_factory=FakeGatherer)
        assert payload["cases"][0]["trace_url"] is None and payload["cases"][0]["record"]["pool_raw"] == 118

    def test_a_crashed_case_is_recorded_and_the_next_case_still_runs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-secretsecretsecret")
        gatherers = iter([
            FakeGatherer(raise_in_gather=RuntimeError("boom with tvly-secretsecretsecret")),
            FakeGatherer(), FakeGatherer(),
        ])
        payload = canary_fetch.run("weekly", out_path=tmp_path / "o.json", agent_factory=lambda: next(gatherers))
        statuses = [(c["case_id"], c["status"]) for c in payload["cases"]]
        assert statuses == [("phoenix-cardiology", "crashed"), ("gilbert-family-medicine", "success"), ("gold-canyon-neurology", "success")]
        crashed = payload["cases"][0]
        assert crashed["error"] == "RuntimeError: boom with ***" and crashed["record"]["pool_raw"] is None
        assert crashed["record"]["extra"]["errors"] == 1
        assert [f.check for f in evaluate_fetch(crashed["record"], PHOENIX, crashed["fetch_stats"], [], crashed["status"])] == ["gather_failed"]

    def test_a_fatal_vendor_error_is_captured_with_its_class(self, tmp_path):
        from tavily.errors import UsageLimitExceededError

        class Tavily:
            def extract(self, **kw):
                raise UsageLimitExceededError("Usage limit exceeded")

        payload = canary_fetch.run("daily", out_path=tmp_path / "o.json", agent_factory=lambda: FakeGatherer(tavily=Tavily()))
        entry = payload["cases"][0]
        assert entry["status"] == "error"
        assert [(e["vendor"], e["error"], e["kind"]) for e in entry["vendor_errors"]] == [("tavily", "UsageLimitExceededError", "fatal")]
        found = evaluate_fetch(entry["record"], CHANDLER, entry["fetch_stats"], entry["vendor_errors"], entry["status"])
        assert [(f.check, f.severity) for f in found] == [("vendor_error", "P1")]

    def test_main_selects_cases_and_always_exits_zero(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(canary_fetch, "_default_factory", FakeGatherer)
        assert canary_fetch.main(["--case", "nowhere"]) == 2
        assert canary_fetch.main(["--case", "gold-canyon-neurology", "--out", str(tmp_path / "o.json")]) == 0
        out = capsys.readouterr().out
        assert "gold-canyon-neurology" in out and "chandler" not in out and "pool 118" in out
        assert canary_fetch.main(["--schedule", "daily", "--out", str(tmp_path / "o.json"), "--radius", "10"]) == 0
        assert json.loads((tmp_path / "o.json").read_text())["cases"][0]["record"]["radius_miles"] == 10.0


# --------------------------------------------------------------------------
# Workflow file
# --------------------------------------------------------------------------

class TestTierAWorkflow:
    def test_workflow_keeps_the_shape_the_alert_path_depends_on(self):
        import yaml

        path = REPO / ".github" / "workflows" / "canary-tier-a.yml"
        text = path.read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
        on = doc.get("on", doc.get(True))
        assert doc["name"].startswith("P2 drift watch")
        crons = [c["cron"].split() for c in on["schedule"]]
        # One daily cron and one Sunday cron — by FIELD, not literal: the owner
        # shifts minutes in the web editor to re-attribute failure mail.
        assert len(crons) == 2 and all(len(c) == 5 for c in crons)
        assert [c[4] for c in crons] == ["*", "0"]
        assert set(on["workflow_dispatch"]["inputs"]) == {"schedule", "case"}
        assert doc["permissions"] == {"contents": "read", "issues": "write"}
        job = doc["jobs"]["fetch-canary"]
        for secret in ("TAVILY_API_KEY", "APP_ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                       "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            assert job["env"][secret] == "${{ secrets.%s }}" % secret
        # The base URL is a repository VARIABLE, never a secret: GitHub masks
        # every occurrence of a secret's value in logs and step summaries, so
        # with the host stored as a secret the summary's trace links rendered
        # as "***/project/<id>/traces/<id>" and the browser resolved them
        # relative to the run page — a 404 on github.com (owner, 2026-09-14).
        # Referencing the secret anywhere re-registers the mask, so the ban
        # is on the reference, not just this env line.
        assert job["env"]["LANGFUSE_BASE_URL"] == "${{ vars.LANGFUSE_BASE_URL }}"
        assert "secrets.LANGFUSE_BASE_URL" not in text
        # Schedules are opt-in per repository (the file ships to the private
        # dev repo AND the public one; only one should spend on a cron):
        # scheduled triggers run only where the SCHEDULED_RUNS variable is
        # "true", and a manual dispatch always runs. On the JOB, so a skip is
        # a grey "skipped" with no steps, no artifact, no failure mail.
        assert job["if"] == "github.event_name != 'schedule' || vars.SCHEDULED_RUNS == 'true'"
        assert job["env"]["ENV"] == "ci" and job["env"]["TAVILY_MODE"] == "extract"
        steps = job["steps"]
        checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout"))
        assert checkout["with"]["lfs"] is True
        canary = next(s for s in steps if "evals.canary_fetch" in str(s.get("run", "")))
        watcher = next(s for s in steps if "evals.watch" in str(s.get("run", "")))
        assert watcher["if"] == "always()"
        assert watcher["env"]["GITHUB_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
        assert "steps.plan.outputs.args" in canary["run"] and "steps.plan.outputs.watch_args" in watcher["run"]
        upload = next(s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact"))
        assert upload["if"] == "always()" and upload["with"]["path"] == "evals/out/" and upload["with"]["retention-days"] == 90
        plan = next(s for s in steps if s.get("id") == "plan")
        assert 'WATCH_ARGS="--tier A $ARGS' in plan["run"], "the watcher judges the same schedule the canary ran"
        assert "github.event.inputs" not in plan["run"], "dispatch inputs reach the shell via env, never inline"
        assert "github.event.inputs.case" in plan["env"]["INPUT_CASE"]
        # Weekly is detected by the Sunday day-of-week field, never by a literal
        # cron string that an owner's minute shift would silently break.
        assert 'case "$CRON" in' in plan["run"] and '*" 0") SCHEDULE=weekly' in plan["run"]
        assert "30 13" not in plan["run"]

    def test_measurements_directory_is_ignored(self):
        assert "evals/out/" in (REPO / ".gitignore").read_text().splitlines()
