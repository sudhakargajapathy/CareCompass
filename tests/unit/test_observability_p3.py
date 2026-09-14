"""Observability P3: the Tier B pipeline canary, the weekly report and its
exported rows, the freshness watchdog, and the two workflows.

What these pin: the pipeline canary measures the full workflow through the
orchestrator's own trace and never raises; a workflow that did not
complete is one `pipeline_failed` finding, not a cascade; every vendor
client on all three agents is wrapped; the weekly report's rows come from
exactly two places on a trace (the run record in metadata, the scores) and
nothing else — the file is PUBLIC; rows dedupe by run_id so re-running a
week is safe; the freshness stamp turns a silently disabled cron into a P3;
and the workflow files keep the shape the alert path depends on.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import canary_pipeline, thresholds, vendor_watch, watch, weekly_report
from evals.cases import case_by_id, tier_b_cases
from evals.thresholds import Finding, checks_for_tier, evaluate_pipeline, evaluate_report_freshness
from utils.run_record import RUN_RECORD_FIELDS, build_run_record

REPO = Path(__file__).resolve().parents[2]
CHANDLER = case_by_id("chandler-neurology")
PHOENIX = case_by_id("phoenix-cardiology")
NOW = datetime(2026, 9, 13, 13, 0, tzinfo=timezone.utc)


def minimal_state(**over):
    state = {
        "run_id": "run-1", "run_source": "canary", "case_id": "chandler-neurology",
        "specialty": "Neurology", "location": "Chandler, AZ", "preferences": {"search_radius_miles": 25},
        "gathered_data": {"providers": [], "search_metadata": {"total_found": 117, "radius_dropped": 0,
                          "fetch_mode": "extract", "fetch_mode_fallback": False, "ring_expanded": False,
                          "fetch_stats": {"fetch": {"discovery": {"planned": 11, "fetched": 10, "empty": 0, "failed": 1, "by_domain": {}}},
                                          "listing_rows": {"healthgrades.com": 73, "doctor.webmd.com": 114, "vitals.com": 105}}}},
        "scored_providers": {}, "validation_results": {}, "final_recommendations": [{"name": "x"}] * 5,
        "execution_log": [], "error_messages": [],
        "workflow_summary": {"cost_summary": {"total_usd": 0.56, "elapsed_s": 70.0, "tavily": {"credits": 5, "credits_by_stage": {"discovery": 3, "enrichment": 2}}, "llm": {"input_tokens": 1000, "output_tokens": 200}}},
    }
    state.update(over)
    return state


class FakeOrchestrator:
    """Returns a packaged result the way execute_workflow does."""

    def __init__(self, shape="success"):
        self.shape = shape
        self.calls = []
        self.data_gatherer = SimpleNamespace(tavily_client=object(), anthropic_client=object())
        self.preference_scorer = SimpleNamespace(openai_client=object())
        self.critic_validator = SimpleNamespace(anthropic_client=object())

    def execute_workflow(self, specialty, location, insurance=None, preferences=None, use_cache=True, source="user", case_id=None):
        self.calls.append({"specialty": specialty, "location": location, "preferences": preferences,
                           "use_cache": use_cache, "source": source, "case_id": case_id})
        if self.shape == "failure":
            return {"success": False, "final_recommendations": [], "workflow_summary": {"workflow_failed": True, "error": "node exploded sk-ant-abcdefghijklmnop"},
                    "cost_summary": {}, "execution_log": [], "agent_outputs": {}, "error_messages": ["node exploded"], "run_id": "run-f", "trace_url": None}
        state = minimal_state(run_source=source, case_id=case_id)
        record = build_run_record(state)
        if self.shape == "no_shortlist":
            record["shortlist_size"] = 0
        return {
            "success": True, "final_recommendations": state["final_recommendations"],
            "workflow_summary": {**state["workflow_summary"], "run_record": record},
            "cost_summary": state["workflow_summary"]["cost_summary"], "execution_log": [],
            "agent_outputs": {"data_gatherer": state["gathered_data"], "preference_scorer": {}, "critic_validator": {}},
            "error_messages": [], "run_id": "run-ok", "trace_url": "https://lf.example/t/run-ok",
        }


# --------------------------------------------------------------------------
# Selection and thresholds
# --------------------------------------------------------------------------

class TestTierBSelectionAndThresholds:
    def test_tier_b_is_by_flag_not_schedule(self):
        assert [c.case_id for c in tier_b_cases()] == ["chandler-neurology"]
        assert [c.case_id for c in watch.expected_cases("B", "weekly", None)] == ["chandler-neurology"]
        assert len(watch.expected_cases("A", "weekly", None)) == 3
        assert watch.expected_cases("B", "weekly", "nowhere") is None
        assert [c.case_id for c in watch.expected_cases("B", "weekly", "phoenix-cardiology")] == ["phoenix-cardiology"]

    def test_pipeline_failed_is_its_own_p1_unless_a_vendor_explains_it(self):
        found = evaluate_pipeline({}, CHANDLER, [], "crashed")
        assert [(f.check, f.severity) for f in found] == [("pipeline_failed", "P1")]
        assert "did not complete" in found[0].first_step
        fatal = [{"vendor": "anthropic", "error": "BadRequestError", "message": "credit balance is too low", "kind": "fatal"}]
        assert [f.check for f in evaluate_pipeline({}, CHANDLER, fatal, "error")] == ["vendor_error"]
        assert "pipeline_failed" in checks_for_tier("B") and "pipeline_failed" not in checks_for_tier("A")

    def test_pipeline_uses_fetch_stats_for_the_empty_body_rate(self):
        record = {"pages_planned": 11, "pages_fetched": 10, "pool_raw": 117, "rows_hg": 1, "rows_wm": 1, "rows_vi": 1,
                  "shortlist_size": 5, "n_enriched": 7, "coverage_hg": 1, "coverage_wm": 1, "coverage_vi": 1}
        stats = {"fetch": {"discovery": {"by_domain": {"vitals.com": {"fetched": 1, "empty": 4}}}}}
        found = evaluate_pipeline(record, CHANDLER, [], "success", stats)
        assert [(f.check, f.detail) for f in found] == [("empty_body_rate_high", "vitals.com")]

    def test_report_freshness(self):
        assert evaluate_report_freshness(None, NOW) == [] and evaluate_report_freshness({}, NOW) == []
        fresh = {"generated_at": (NOW - timedelta(days=6)).isoformat(), "week": "2026-W36"}
        assert evaluate_report_freshness(fresh, NOW) == []
        stale = {"generated_at": (NOW - timedelta(days=9, hours=1)).isoformat(), "week": "2026-W35"}
        found = evaluate_report_freshness(stale, NOW)
        assert [(f.check, f.severity, f.case_id, f.observed) for f in found] == [
            ("report_stale", "P3", "weekly-report", "9 days old (week 2026-W35)")
        ]
        assert found[0].title == "[P3] Weekly report stale — weekly-report"
        naive = {"generated_at": (NOW - timedelta(days=20)).replace(tzinfo=None).isoformat() + "Z"}
        assert [f.check for f in evaluate_report_freshness(naive, NOW)] == ["report_stale"]
        assert evaluate_report_freshness({"generated_at": "garbage"}, NOW)[0].observed == "unreadable generated_at"


# --------------------------------------------------------------------------
# Vendor watch over the orchestrator
# --------------------------------------------------------------------------

class TestVendorWatchOrchestrator:
    def test_every_agent_client_is_wrapped_and_missing_ones_skipped(self):
        class Completions:
            def create(self, **kw):
                raise RuntimeError("insufficient_quota")

        orch = FakeOrchestrator()
        orch.preference_scorer.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        orch.critic_validator.anthropic_client = None
        w = vendor_watch.VendorWatch()
        w.wrap_orchestrator(orch)
        assert isinstance(orch.data_gatherer.tavily_client, vendor_watch._Recording)
        assert isinstance(orch.preference_scorer.openai_client, vendor_watch._Recording)
        assert orch.critic_validator.anthropic_client is None
        with pytest.raises(RuntimeError):
            orch.preference_scorer.openai_client.chat.completions.create(model="m")
        assert [(e["vendor"], e["kind"]) for e in w.errors] == [("openai", "fatal")]
        w.wrap_orchestrator(SimpleNamespace())  # no agents at all: nothing to wrap, nothing raised


# --------------------------------------------------------------------------
# The pipeline canary
# --------------------------------------------------------------------------

class TestPipelineCanary:
    def test_success_uses_the_orchestrators_record_and_trace(self, tmp_path):
        orch = FakeOrchestrator()
        payload = canary_pipeline.run("canary", out_path=tmp_path / "b.json", orchestrator_factory=lambda: orch)
        call = orch.calls[0]
        assert call["source"] == "canary" and call["case_id"] == "chandler-neurology" and call["use_cache"] is False
        assert call["preferences"] == {"search_radius_miles": 25.0} and call["location"] == "Chandler, AZ"
        entry = payload["cases"][0]
        assert entry["status"] == "success" and entry["error"] is None and entry["tier"] == "B"
        assert entry["run_id"] == "run-ok" and entry["trace_url"] == "https://lf.example/t/run-ok"
        record = entry["record"]
        assert record["pool_raw"] == 117 and record["shortlist_size"] == 5 and record["cost_usd"] == 0.56
        assert record["extra"]["tier"] == "B" and record["extra"]["vendor_errors"] == 0
        assert entry["fetch_stats"]["listing_rows"]["doctor.webmd.com"] == 114
        assert evaluate_pipeline(record, CHANDLER, entry["vendor_errors"], entry["status"], entry["fetch_stats"]) == []
        written = json.loads((tmp_path / "b.json").read_text())
        assert written["tier"] == "B" and written["source"] == "canary" and len(written["cases"]) == 1

    def test_a_failed_workflow_gets_a_record_of_nones_and_a_p1(self, tmp_path):
        payload = canary_pipeline.run("smoke", out_path=tmp_path / "b.json", orchestrator_factory=lambda: FakeOrchestrator("failure"))
        entry = payload["cases"][0]
        assert entry["status"] == "error" and entry["source"] == "smoke"
        assert entry["error"] == "node exploded ***" and entry["record"]["pool_raw"] is None
        assert entry["record"]["extra"]["workflow_failed"] is True and entry["record"]["source"] == "smoke"
        assert entry["record"]["extra"]["pipeline"] is True
        found = evaluate_pipeline(entry["record"], CHANDLER, entry["vendor_errors"], entry["status"])
        assert [(f.check, f.severity) for f in found] == [("pipeline_failed", "P1")]

    def test_a_crash_is_recorded_and_the_zero_shortlist_shape_is_a_finding(self, tmp_path):
        def boom():
            raise RuntimeError("factory died")

        payload = canary_pipeline.run("canary", case_id="phoenix-cardiology", out_path=tmp_path / "b.json", orchestrator_factory=boom)
        entry = payload["cases"][0]
        assert entry["case_id"] == "phoenix-cardiology" and entry["status"] == "crashed"
        assert entry["error"] == "RuntimeError: factory died"
        payload = canary_pipeline.run("canary", out_path=tmp_path / "b.json", orchestrator_factory=lambda: FakeOrchestrator("no_shortlist"))
        entry = payload["cases"][0]
        assert [f.check for f in evaluate_pipeline(entry["record"], CHANDLER, [], entry["status"])] == ["no_shortlist"]

    def test_main_validates_source_and_case(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(canary_pipeline, "_default_factory", FakeOrchestrator)
        with pytest.raises(ValueError):
            canary_pipeline.run("nope")
        assert canary_pipeline.main(["--case", "nowhere"]) == 2
        assert canary_pipeline.main(["--source", "smoke", "--case", "chandler-neurology", "--out", str(tmp_path / "b.json")]) == 0
        out = capsys.readouterr().out
        assert "Tier B pipeline canary — smoke: chandler-neurology" in out and "shortlist 5" in out
        assert json.loads((tmp_path / "b.json").read_text())["source"] == "smoke"


# --------------------------------------------------------------------------
# The watcher on Tier B and the freshness stamp
# --------------------------------------------------------------------------

def pipeline_payload(shape="success"):
    entry = canary_pipeline.run_case(CHANDLER, orchestrator_factory=lambda: FakeOrchestrator(shape))
    return {"tier": "B", "source": "canary", "ts": NOW.isoformat(), "cases": [entry]}


class TestWatchTierB:
    def test_tier_b_expects_the_tier_b_cases_and_judges_the_pipeline_record(self, tmp_path, capsys):
        path = tmp_path / "b.json"
        path.write_text(json.dumps(pipeline_payload(), default=str))
        assert watch.main(["--tier", "B", "--input", str(path), "--no-issues"], {}) == 0
        assert "Tier B pipeline canary" in capsys.readouterr().out
        path.write_text(json.dumps(pipeline_payload("no_shortlist"), default=str))
        assert watch.main(["--tier", "B", "--input", str(path), "--no-issues"], {}) == 1
        assert "No shortlist" in capsys.readouterr().out
        # A payload carrying only Phoenix leaves Chandler unmeasured → P1.
        payload = pipeline_payload()
        payload["cases"][0]["case_id"] = "phoenix-cardiology"
        path.write_text(json.dumps(payload, default=str))
        findings, evaluated = watch.evaluate_results("B", tier_b_cases(), json.loads(path.read_text()))
        assert [(f.check, f.case_id) for f in findings] == [("canary_did_not_run", "chandler-neurology")]
        assert evaluated == {"chandler-neurology": {"canary_did_not_run"}}

    def test_report_stamp_is_evaluated_only_when_present(self, tmp_path):
        assert watch.evaluate_report(None) == ([], {})
        assert watch.evaluate_report(tmp_path) == ([], {})
        (tmp_path / "latest.json").write_text(json.dumps({"generated_at": (NOW - timedelta(days=30)).isoformat(), "week": "2026-W32"}))
        findings, scope = watch.evaluate_report(tmp_path)
        assert [f.check for f in findings] == ["report_stale"] and scope == {"weekly-report": {"report_stale"}}
        (tmp_path / "latest.json").write_text("{not json")
        findings, _ = watch.evaluate_report(tmp_path)
        assert findings[0].observed == "unreadable generated_at"

    def test_main_wires_the_reports_dir(self, tmp_path, capsys):
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "latest.json").write_text(json.dumps({"generated_at": "2026-08-01T00:00:00+00:00", "week": "2026-W30"}))
        payload = {"tier": "A", "schedule": "daily", "cases": [{"case_id": "chandler-neurology", "status": "success",
                   "record": {"pages_planned": 11, "pages_fetched": 10, "pool_raw": 117, "rows_hg": 1, "rows_wm": 1, "rows_vi": 1},
                   "fetch_stats": {}, "vendor_errors": []}]}
        path = tmp_path / "a.json"
        path.write_text(json.dumps(payload))
        assert watch.main(["--tier", "A", "--input", str(path), "--no-issues", "--reports-dir", str(reports)], {}) == 1
        assert "Weekly report stale" in capsys.readouterr().out
        assert watch.main(["--tier", "A", "--input", str(path), "--no-issues"], {}) == 0


# --------------------------------------------------------------------------
# The weekly report
# --------------------------------------------------------------------------

def fake_trace(trace_id, ts, source="canary", case_id="chandler-neurology", tier="A", with_record=True, extra_metadata=None,
               environment="ci"):
    record = {
        "run_id": f"run-{trace_id}", "ts": ts.isoformat(timespec="seconds"), "source": source, "case_id": case_id,
        "git_sha": "abc1234", "schema_version": 1, "tavily_mode": "extract", "gatherer_model": "claude-haiku-4-5",
        "judge_model": "gpt-5.6-terra", "critic_model": "claude-opus-4-8", "budget": 8, "radius_miles": 25,
        "specialty": "Neurology", "city": "Chandler", "state": "AZ", "zip_present": False, "fetch_mode": "extract",
        "pairs_hist": {"2": 3}, "extra": {"tier": tier, "pipeline": tier == "B", "workflow_failed": False},
    }
    metadata = {"run_record": record, "resourceAttributes": {"service.name": "x"}, "scope": {"name": "langfuse"},
                "provider_name": "Dr. Leak"}  # a key that must never reach a row
    if not with_record:
        metadata = {"scope": {}}
    metadata.update(extra_metadata or {})
    return SimpleNamespace(id=trace_id, timestamp=ts, tags=[f"source:{source}", f"case:{case_id}", f"tier:{tier}"],
                           environment=environment, metadata=metadata, name="carecompass.search")


def fake_scores(trace_id, **values):
    out = []
    for name, value in values.items():
        data_type = "BOOLEAN" if isinstance(value, bool) else "NUMERIC"
        out.append(SimpleNamespace(name=name, value=float(value), data_type=data_type, trace_id=trace_id))
    out.append(SimpleNamespace(name="not_a_field", value=1.0, data_type="NUMERIC", trace_id=trace_id))
    return out


class FakeApi:
    def __init__(self, traces, scores):
        self._traces, self._scores = traces, scores
        self.calls = []
        self.trace = SimpleNamespace(list=self._list)
        self.scores = SimpleNamespace(get_many=self._get_many)

    def _list(self, *, page, limit, name, from_timestamp, to_timestamp, environment=None):
        # Recorded but NOT applied: the fake ignores the filter on purpose, so
        # the exporter's own client-side check has to do the dropping.
        self.calls.append(("trace.list", page, name) + ((environment,) if environment is not None else ()))
        assert name == "carecompass.search"
        window = [t for t in self._traces if from_timestamp <= t.timestamp < to_timestamp]
        size = 2  # force pagination
        chunk = window[(page - 1) * size: page * size]
        total_pages = max(1, -(-len(window) // size))
        return SimpleNamespace(data=chunk, meta=SimpleNamespace(total_pages=total_pages))

    def _get_many(self, *, page, limit, trace_id):
        self.calls.append(("scores.get_many", page, trace_id))
        return SimpleNamespace(data=self._scores.get(trace_id, []), meta=SimpleNamespace(total_pages=1))


def fake_client(traces, scores):
    return SimpleNamespace(api=FakeApi(traces, scores))


class TestWeeklyReport:
    def test_week_bounds(self):
        start, end, label = weekly_report.week_bounds(NOW, 1)
        assert (start, end, label) == (datetime(2026, 8, 31, tzinfo=timezone.utc), datetime(2026, 9, 7, tzinfo=timezone.utc), "2026-W36")
        start, end, label = weekly_report.week_bounds(NOW, 0)
        assert (start.date().isoformat(), end.date().isoformat(), label) == ("2026-09-07", "2026-09-14", "2026-W37")

    def test_row_comes_from_the_record_and_the_scores_only(self):
        trace = fake_trace("t1", NOW)
        scores = fake_scores("t1", pool_raw=117, rows_wm=114, fallback_fired=False, cost_usd=0.024)
        row = weekly_report.row_from_trace(trace, scores)
        assert tuple(row.keys()) == weekly_report.ROW_FIELDS
        assert set(weekly_report.ROW_FIELDS) == set(RUN_RECORD_FIELDS) | {"trace_id", "environment"}
        assert row["pool_raw"] == 117.0 and row["rows_wm"] == 114.0 and row["fallback_fired"] is False
        assert row["source"] == "canary" and row["case_id"] == "chandler-neurology" and row["extra"]["pipeline"] is False
        assert row["trace_id"] == "t1" and row["environment"] == "ci" and row["run_id"] == "run-t1"
        assert "provider_name" not in row and "resourceAttributes" not in row and "not_a_field" not in row
        assert row["n_enriched"] is None, "a measurement no score carried stays None"
        # Verify traces and traces without a record are not runs.
        assert weekly_report.row_from_trace(fake_trace("t2", NOW, source="verify"), []) is None
        assert weekly_report.row_from_trace(fake_trace("t3", NOW, with_record=False), []) is None
        # Source and run_id fall back to the tag and the trace id.
        bare = fake_trace("t4", NOW)
        bare.metadata["run_record"] = {"extra": {}}
        row = weekly_report.row_from_trace(bare, [])
        assert row["source"] == "canary" and row["run_id"] == "trace:t4" and row["ts"] == NOW.isoformat(timespec="seconds")

    def test_fetch_rows_paginates_and_skips_verify(self):
        base = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
        traces = [fake_trace(f"t{i}", base + timedelta(hours=i)) for i in range(5)]
        traces[2] = fake_trace("t2", base + timedelta(hours=2), source="verify")
        traces.append(fake_trace("old", base - timedelta(days=30)))
        scores = {t.id: fake_scores(t.id, pool_raw=100 + i) for i, t in enumerate(traces)}
        client = fake_client(traces, scores)
        start, end, _ = weekly_report.week_bounds(NOW, 0)
        rows = weekly_report.fetch_rows(client, start, end)
        assert [r["trace_id"] for r in rows] == ["t0", "t1", "t3", "t4"]
        assert [c for c in client.api.calls if c[0] == "trace.list"] == [("trace.list", 1, "carecompass.search"), ("trace.list", 2, "carecompass.search"), ("trace.list", 3, "carecompass.search")]

    def test_only_runs_that_spent_or_measured_are_exported(self):
        """Seven of the fourteen leaked test traces were mocked FAILURES —
        `workflow_failed` admitted them; spend-or-measure does not."""
        assert weekly_report.is_run({"cost_usd": 0.024}) and weekly_report.is_run({"tavily_credits": 3})
        assert weekly_report.is_run({"pool_raw": 0}) and weekly_report.is_run({"pages_planned": 11})
        assert not weekly_report.is_run({"cost_usd": 0, "extra": {"workflow_failed": True, "errors": 1}})
        assert not weekly_report.is_run({"cost_usd": None, "pool_raw": None, "shortlist_size": 0})
        trace = fake_trace("mock", NOW, source="user")
        assert weekly_report.row_from_trace(trace, []) is not None
        client = fake_client([trace], {})
        start, end, _ = weekly_report.week_bounds(NOW, 0)
        assert weekly_report.fetch_rows(client, start, end) == []
        # Pipeline vs fetch: the record states it; older rows fall back to the judge.
        assert weekly_report.is_pipeline({"extra": {"pipeline": True}}) and not weekly_report.is_pipeline({"extra": {"pipeline": False}, "shortlist_size": 5})
        assert weekly_report.is_pipeline({"extra": {}, "judge_applied": 8}) and not weekly_report.is_pipeline({"extra": {}, "shortlist_size": 0})

    def test_rows_file_appends_without_duplicates(self, tmp_path):
        path = tmp_path / "rows.jsonl"
        rows = [{"run_id": "a", "pool_raw": 1}, {"run_id": "b", "pool_raw": 2}]
        assert weekly_report.append_rows(path, rows) == (2, 0)
        assert weekly_report.append_rows(path, rows + [{"run_id": "c", "pool_raw": 3}]) == (1, 2)
        path.write_text(path.read_text() + "not json\n")
        assert [r["run_id"] for r in weekly_report.read_rows(path)] == ["a", "b", "c"]

    def test_render_report_tables(self):
        a = weekly_report.row_from_trace(fake_trace("t1", NOW), fake_scores("t1", pool_raw=117, rows_hg=73, rows_wm=114, rows_vi=105, pages_fetched=10, pages_planned=11, empty_bodies=0, fallback_fired=False, ring_fired=False, tavily_credits=3, cost_usd=0.024, latency_s=7.5))
        b = weekly_report.row_from_trace(fake_trace("t2", NOW, tier="B"), fake_scores("t2", pool_raw=118, n_enriched=7, n_no_profile_found=1, coverage_hg=5, coverage_wm=6, coverage_vi=6, shortlist_size=5, judge_applied=8, critic_shards_failed=0, cost_usd=0.56, latency_s=70))
        start, end, label = weekly_report.week_bounds(NOW, 0)
        text = weekly_report.render_report(label, start, end, [a, b], NOW)
        assert text.startswith("# CareCompass weekly report — 2026-W37")
        assert "| canary | ci | 2 | $0.292 | 39 s | 118 | 5 |" in text
        assert "| chandler-neurology | 117 | ≥ 60 | 73 / 114 / 105 | 10 / 11 | 0 | no | no | 3 |" in text
        assert "| chandler-neurology | canary | 118 | 7 | 1 | 5 / 6 / 6 | 5 | 8 | 0 | $0.560 | 70 s |" in text
        empty = weekly_report.render_report(label, start, end, [], NOW)
        assert "No Tier A canary runs this week." in empty and "No pipeline canary or smoke runs this week." in empty

    def test_generate_writes_rows_report_and_stamp_idempotently(self, tmp_path):
        base = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
        traces = [fake_trace("t1", base), fake_trace("t2", base + timedelta(hours=1), tier="B")]
        scores = {"t1": fake_scores("t1", pool_raw=117), "t2": fake_scores("t2", shortlist_size=5, cost_usd=0.5)}
        client = fake_client(traces, scores)
        dry = weekly_report.generate(client, tmp_path / "reports", weeks_back=0, now=NOW, dry_run=True)
        assert dry["rows"] == 2 and not (tmp_path / "reports").exists()
        result = weekly_report.generate(client, tmp_path / "reports", weeks_back=0, now=NOW)
        assert (result["added"], result["skipped"], result["total"], result["week"]) == (2, 0, 2, "2026-W37")
        assert (tmp_path / "reports" / "2026-W37.md").read_text().startswith("# CareCompass weekly report — 2026-W37")
        stamp = json.loads((tmp_path / "reports" / "latest.json").read_text())
        assert stamp["week"] == "2026-W37" and stamp["rows_total"] == 2 and stamp["generated_at"].startswith("2026-09-13T13:00")
        again = weekly_report.generate(client, tmp_path / "reports", weeks_back=0, now=NOW)
        assert (again["added"], again["skipped"], again["total"]) == (0, 2, 2)
        lines = (tmp_path / "reports" / "run_records.jsonl").read_text().splitlines()
        assert len(lines) == 2 and set(json.loads(lines[0])) == set(weekly_report.ROW_FIELDS)

    def test_main_exit_codes(self, tmp_path, monkeypatch, capsys):
        assert weekly_report.main(["--out-dir", str(tmp_path)], env={}) == 2
        keys = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk", "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com"}

        def broken(env):
            def fail(**kw):
                raise ConnectionError("api down")
            return SimpleNamespace(api=SimpleNamespace(trace=SimpleNamespace(list=fail), scores=None))

        monkeypatch.setattr(weekly_report, "_client", broken)
        assert weekly_report.main(["--out-dir", str(tmp_path)], env=keys) == 1
        assert "export failed — ConnectionError" in capsys.readouterr().err
        # main() has no --now, so "this week" is the WALL CLOCK's week: the
        # trace must be stamped now, not at the fixed NOW — stamped 2026-09-13
        # (ISO week 37) it went red at midnight on the 14th (week 38), with
        # zero rows pulled, on every CI run from that day.
        live = fake_trace("t1", datetime.now(timezone.utc))
        monkeypatch.setattr(weekly_report, "_client", lambda env: fake_client([live], {"t1": fake_scores("t1", pool_raw=1)}))
        assert weekly_report.main(["--out-dir", str(tmp_path / "r"), "--weeks-back", "0"], env=keys) == 0
        assert "1 run(s) pulled, 1 row(s) added" in capsys.readouterr().out


    def test_the_client_strips_whitespace_from_the_pasted_values(self, monkeypatch):
        """The public repo's LANGFUSE_BASE_URL variable was pasted with a
        trailing space (visible in the first armed run's log). The canaries
        survived because utils.tracing strips; this client passed the value
        raw, and a host ending in a space fails every API call — the first
        Monday report would have gone red."""
        import sys
        import types

        seen = {}

        class FakeLangfuse:
            def __init__(self, **kw):
                seen.update(kw)

        module = types.ModuleType("langfuse")
        module.Langfuse = FakeLangfuse
        monkeypatch.setitem(sys.modules, "langfuse", module)
        weekly_report._client({"LANGFUSE_PUBLIC_KEY": " pk ", "LANGFUSE_SECRET_KEY": "sk\n", "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com "})
        assert seen["base_url"] == "https://us.cloud.langfuse.com"
        assert seen["public_key"] == "pk" and seen["secret_key"] == "sk" and seen["tracing_enabled"] is False

    def test_environment_filter_is_sent_to_the_api_and_enforced_on_the_rows(self):
        """Two Spaces write to ONE Langfuse project. The public repo's report
        must carry only the public Space's traffic, so the exporter filters
        by environment label — server-side, and again on every row: the
        fake API here ignores the filter, and the development row must
        still not reach the public rows."""
        base = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
        demo = fake_trace("d1", base, source="user", case_id=None, environment="demo")
        dev = fake_trace("v1", base + timedelta(hours=1), source="user", case_id=None, environment="development")
        ci = fake_trace("c1", base + timedelta(hours=2), environment="ci")
        client = fake_client([demo, dev, ci], {t.id: fake_scores(t.id, pool_raw=100) for t in (demo, dev, ci)})
        start, end, label = weekly_report.week_bounds(NOW, 0)
        rows = weekly_report.fetch_rows(client, start, end, environments=["demo", "ci"])
        assert [r["trace_id"] for r in rows] == ["d1", "c1"]
        assert client.api.calls[0] == ("trace.list", 1, "carecompass.search", ["ci", "demo"]), "the API is asked to filter"
        assert [r["trace_id"] for r in weekly_report.fetch_rows(client, start, end)] == ["d1", "v1", "c1"], "no filter = all"
        assert weekly_report.parse_environments(" Demo, ci ,,demo") == ["demo", "ci"]
        assert weekly_report.parse_environments(None) == [] and weekly_report.parse_environments("  ") == []

    def test_the_filter_reaches_the_rows_file_the_stamp_the_header_and_main(self, tmp_path, monkeypatch, capsys):
        base = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
        traces = [fake_trace("d1", base, source="user", case_id=None, environment="demo"),
                  fake_trace("v1", base, source="user", case_id=None, environment="development")]
        client = fake_client(traces, {"d1": fake_scores("d1", pool_raw=1), "v1": fake_scores("v1", pool_raw=2)})
        result = weekly_report.generate(client, tmp_path / "r", weeks_back=0, now=NOW, environments=["demo"])
        assert result["rows"] == 1 and result["environments"] == ["demo"]
        assert "Traces filtered to environment(s) demo." in result["report"]
        assert json.loads((tmp_path / "r" / "latest.json").read_text())["environments"] == ["demo"]
        assert [r["environment"] for r in weekly_report.read_rows(tmp_path / "r" / "run_records.jsonl")] == ["demo"]
        unfiltered = weekly_report.generate(client, tmp_path / "all", weeks_back=0, now=NOW)
        assert unfiltered["rows"] == 2 and "All environments." in unfiltered["report"]
        assert json.loads((tmp_path / "all" / "latest.json").read_text())["environments"] == []
        # main: REPORT_ENVIRONMENTS is the workflow's knob, --environments wins
        # over it, and an empty value means all. Traces are stamped NOW: main
        # has no --now, so "this week" is the wall clock's.
        keys = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk", "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com"}
        live = [fake_trace("d2", datetime.now(timezone.utc), source="user", case_id=None, environment="demo"),
                fake_trace("v2", datetime.now(timezone.utc), source="user", case_id=None, environment="development")]
        monkeypatch.setattr(weekly_report, "_client", lambda env: fake_client(live, {"d2": fake_scores("d2", pool_raw=1), "v2": fake_scores("v2", pool_raw=1)}))
        assert weekly_report.main(["--out-dir", str(tmp_path / "m1"), "--weeks-back", "0"], env={**keys, "REPORT_ENVIRONMENTS": "demo"}) == 0
        assert [r["environment"] for r in weekly_report.read_rows(tmp_path / "m1" / "run_records.jsonl")] == ["demo"]
        assert "[Traces filtered to environment(s) demo.]" in capsys.readouterr().out
        assert weekly_report.main(["--out-dir", str(tmp_path / "m2"), "--weeks-back", "0", "--environments", "development"],
                                  env={**keys, "REPORT_ENVIRONMENTS": "demo"}) == 0
        assert [r["environment"] for r in weekly_report.read_rows(tmp_path / "m2" / "run_records.jsonl")] == ["development"]
        assert weekly_report.main(["--out-dir", str(tmp_path / "m3"), "--weeks-back", "0"], env={**keys, "REPORT_ENVIRONMENTS": ""}) == 0
        assert len(weekly_report.read_rows(tmp_path / "m3" / "run_records.jsonl")) == 2


# --------------------------------------------------------------------------
# Workflow files
# --------------------------------------------------------------------------

class TestP3Workflows:
    def test_tier_b_workflow(self):
        import yaml

        text = (REPO / ".github" / "workflows" / "canary-tier-b.yml").read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
        on = doc.get("on", doc.get(True))
        assert doc["name"].startswith("P1 vendor health")
        crons = [c["cron"].split() for c in on["schedule"]]
        assert len(crons) == 1 and crons[0][4] == "0", "one Sunday cron; minutes are the owner's to shift"
        assert set(on["workflow_dispatch"]["inputs"]) == {"mode", "case"}
        assert doc["permissions"] == {"contents": "read", "issues": "write"}
        job = doc["jobs"]["pipeline-canary"]
        for secret in ("TAVILY_API_KEY", "APP_ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            assert job["env"][secret] == "${{ secrets.%s }}" % secret
        # A variable, not a secret: a secret's value is masked in the step
        # summary, and the trace links rendered as "***/project/…" (404).
        assert job["env"]["LANGFUSE_BASE_URL"] == "${{ vars.LANGFUSE_BASE_URL }}" and "secrets.LANGFUSE_BASE_URL" not in text
        assert job["if"] == "github.event_name != 'schedule' || vars.SCHEDULED_RUNS == 'true'", "schedules are opt-in per repo"
        assert job["env"]["PROVIDER_CACHE_TTL_DAYS"] == "0"
        steps = job["steps"]
        watcher = next(s for s in steps if "evals.watch" in str(s.get("run", "")))
        assert watcher["if"] == "always()" and "--tier B" in watcher["run"]
        plan = next(s for s in steps if s.get("id") == "plan")
        assert "github.event.inputs" not in plan["run"] and "chandler-neurology" in plan["run"]
        upload = next(s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact"))
        assert upload["if"] == "always()" and upload["with"]["retention-days"] == 90

    def test_weekly_report_workflow_and_readme(self):
        import yaml

        text = (REPO / ".github" / "workflows" / "weekly-report.yml").read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
        on = doc.get("on", doc.get(True))
        assert doc["name"].startswith("P3 cost & freshness")
        crons = [c["cron"].split() for c in on["schedule"]]
        assert len(crons) == 1 and crons[0][4] == "1", "one Monday cron; minutes are the owner's to shift"
        assert doc["permissions"] == {"contents": "write"}
        job = doc["jobs"]["report"]
        for secret in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            assert job["env"][secret] == "${{ secrets.%s }}" % secret
        assert job["env"]["LANGFUSE_BASE_URL"] == "${{ vars.LANGFUSE_BASE_URL }}" and "secrets.LANGFUSE_BASE_URL" not in text
        assert job["if"] == "github.event_name != 'schedule' || vars.SCHEDULED_RUNS == 'true'", "schedules are opt-in per repo"
        # Each repo exports only its own Space's environment labels from the
        # shared Langfuse project — a repository variable, empty = all.
        assert job["env"]["REPORT_ENVIRONMENTS"] == "${{ vars.REPORT_ENVIRONMENTS }}"
        assert "secrets.TAVILY_API_KEY" not in text and "secrets.OPENAI_API_KEY" not in text, "the report reads traces; it runs no search"
        runs = [s["run"] for s in job["steps"] if "run" in s]
        assert any("evals.weekly_report" in r for r in runs)
        commit = next(r for r in runs if "git push" in r)
        assert "git diff --cached --quiet" in commit and "git add reports/" in commit
        readme = (REPO / "reports" / "README.md").read_text(encoding="utf-8")
        assert "run_records.jsonl" in readme and "latest.json" in readme
