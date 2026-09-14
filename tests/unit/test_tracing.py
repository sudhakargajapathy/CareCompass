"""Observability P1: tracing is a no-op without keys, and a correct tree with them.

The failures these pin:

  * A developer with no Langfuse account — or the offline unit suite, or a
    Space deployed without the keys — must run with ZERO tracing side
    effects: no client constructed, no network, no warnings per call. The
    first test makes the SDK constructor itself raise if reached.
  * Keys WITHOUT a base URL would send every trace to the SDK's default (EU)
    host, invisible on the US dashboard. That combination disables tracing
    with a warning instead of defaulting.
  * Enrichment, judge and critic calls run on ThreadPoolExecutor workers, and
    the SDK's context does not follow threads: without an explicit parent
    rule every worker's spans become orphan traces. The tree test runs real
    threads through the real helpers against a fake client that records
    parent ids, and asserts the shape the plan draws.
  * "One seam, two sinks": the generation helper writes the SAME cost to the
    cost tracker and the span, so the cost card and the trace cannot show
    two dollar figures for one call.
  * Bodies stay out: a page's text never reaches a span — only its length
    and hash — and the mask hook caps any string that slips through.
"""

import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.helpers.fake_langfuse import FakeLangfuse, by_name as _by_name, install_fake_tracing
from utils import tracing
from utils.cost_tracker import get_cost_tracker
from utils.run_record import DESCRIPTOR_FIELDS, RUN_RECORD_FIELDS, STRUCTURED_FIELDS, build_run_record, measurement_fields

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def traced(monkeypatch):
    """Tracing ON against the fake client: keys + base URL present."""
    captured = install_fake_tracing(monkeypatch)
    yield captured
    tracing.reset_client()


@pytest.fixture
def untraced(monkeypatch):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    tracing.reset_client()
    yield
    tracing.reset_client()


class TestNoopWithoutKeys:
    def test_no_client_is_ever_constructed_without_keys(self, untraced, monkeypatch):
        module = types.ModuleType("langfuse")

        class Forbidden:
            def __init__(self, **kw):
                raise AssertionError("Langfuse() constructed with no keys configured")

        module.Langfuse = Forbidden
        monkeypatch.setitem(__import__("sys").modules, "langfuse", module)
        assert tracing.get_client() is None
        with tracing.start_run(run_id="abc") as run:
            assert run is None
            with tracing.generation("x", model="m", agent="a") as gen:
                in_t, out_t = gen.finish(None)
            with tracing.tool("t") as t:
                t.finish(output={"ok": 1})
            with tracing.span("s") as s:
                s.finish(output={})
        assert (in_t, out_t) == (0, 0)

    def test_keys_without_base_url_disable_tracing_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
        module = types.ModuleType("langfuse")
        module.Langfuse = FakeLangfuse
        monkeypatch.setitem(__import__("sys").modules, "langfuse", module)
        FakeLangfuse.instances.clear()
        tracing.reset_client()
        with caplog.at_level("WARNING"):
            assert tracing.get_client() is None
        assert FakeLangfuse.instances == []
        assert any("LANGFUSE_BASE_URL" in r.getMessage() for r in caplog.records)
        tracing.reset_client()

    def test_generation_still_records_cost_when_untraced(self, untraced):
        # The seam is the ONLY cost writer now; tracing off must not mean
        # the cost card goes blank.
        get_cost_tracker().reset()
        response = MagicMock()
        response.usage.input_tokens, response.usage.output_tokens = 1000, 500
        with tracing.generation("judge.shard", model="claude-haiku-4-5", agent="data_gatherer") as gen:
            assert gen.finish(response) == (1000, 500)
        summary = get_cost_tracker().summary()
        assert summary["llm"]["calls"] == 1
        assert summary["llm"]["cost_usd"] == pytest.approx((1000 * 1.0 + 500 * 5.0) / 1e6)


class TestEnvironmentLabel:
    @pytest.mark.parametrize("raw, expected", [
        ("production", "production"), ("CI", "ci"), (" development ", "development"),
        ("", "development"), (None, "development"), ("Prod Env!", "development"),
        ("langfuse-x", "development"),
    ])
    def test_env_derives_the_label_and_falls_back_to_development(self, raw, expected):
        assert tracing.environment_label(raw) == expected

    def test_client_is_built_with_env_label_base_url_and_mask(self, traced):
        client = tracing.get_client()
        assert client.kw["environment"] == "ci"
        assert client.kw["base_url"] == "https://us.cloud.langfuse.com"
        assert client.kw["mask"] is tracing.mask


class TestBodiesStayOut:
    def test_preview_and_mask_cap_long_strings(self):
        long = "x" * (tracing.PREVIEW_CHARS + 50)
        p = tracing.preview(long)
        assert p["chars"] == len(long) and len(p["preview"]) < len(long) and p["preview"].endswith("[truncated]")
        masked = tracing.mask(data={"a": long, "b": [long, 3], "c": "short"})
        assert masked["a"].endswith("[truncated]") and masked["b"][0].endswith("[truncated]") and masked["c"] == "short"

    def test_page_digest_carries_length_and_hash_never_text(self):
        body = "PATIENT REVIEW TEXT " * 100
        digest = tracing.page_digest([{"url": "https://www.vitals.com/doctors/x", "raw_content": body}])
        assert digest == [{"url": "https://www.vitals.com/doctors/x", "chars": len(body), "sha256": tracing.sha256_text(body)[:16]}]
        assert "PATIENT" not in str(digest)



class TestSpanTree:
    def test_root_is_seeded_by_run_id_and_carries_descriptors(self, traced):
        with tracing.start_run(run_id="run-1", input={"specialty": "Neurology"}, metadata={"source": "canary", "case_id": None}, tags=["source:canary"]) as run:
            assert run.trace_id == "trace-run-1"
        client = FakeLangfuse.instances[0]
        root = client.spans[0]
        assert root.name == tracing.ROOT_NAME and root.as_type == "agent"
        assert root.trace_context == {"trace_id": "trace-run-1"}
        assert traced["tags"] == ["source:canary"] and traced["metadata"] == {"source": "canary"}

    def test_steps_nest_and_worker_threads_attach_to_the_open_step(self, traced):
        with tracing.start_run(run_id="run-2") as run:
            run.step_started("gather_data", {"agent": "DataGathererAgent"})
            with tracing.generation("discovery.extract", model="claude-haiku-4-5", agent="data_gatherer", prompt="p") as gen:
                gen.finish(None)
            run.step_finished("gather_data", "completed", {"providers_found": 3})

            run.step_started("score_providers", {})
            run.step_started("enrich_reviews", {})

            gate = threading.Barrier(3)  # forces three DISTINCT threads to be live at once

            def worker(i):
                gate.wait(timeout=5)
                with tracing.span("enrichment.provider", input={"provider": f"Dr {i}"}) as unit:
                    with tracing.tool("tavily.extract", input={"url_count": 1}) as call:
                        call.finish(output={"fetched": 1})
                    with tracing.generation("enrichment.extract", model="claude-haiku-4-5", agent="data_gatherer") as gen:
                        gen.finish(None)
                    unit.finish(output={"outcome": "enriched"})

            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(worker, range(3)))
            run.step_finished("enrich_reviews", "completed", {})

            def judge_shard(_):
                with tracing.generation("judge.shard", model="gpt-5.6-terra", agent="preference_scorer") as gen:
                    gen.finish(None)

            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(judge_shard, range(2)))
            run.step_finished("score_providers", "completed", {})
            run.finish(record={}, output={"ok": True})

        client = FakeLangfuse.instances[0]
        root = client.spans[0]
        step = {s.name: s for s in client.spans if s.name.startswith("step.")}
        assert step["step.gather_data"].parent_id == root.id
        assert step["step.score_providers"].parent_id == root.id
        assert step["step.enrich_reviews"].parent_id == step["step.score_providers"].id
        assert _by_name(client, "discovery.extract")[0].parent_id == step["step.gather_data"].id

        units = _by_name(client, "enrichment.provider")
        assert len(units) == 3 and {u.parent_id for u in units} == {step["step.enrich_reviews"].id}
        assert len({u.thread for u in units}) == 3, "the workers really ran on three threads"
        for unit in units:
            children = [s for s in client.spans if s.parent_id == unit.id]
            assert sorted(c.name for c in children) == ["enrichment.extract", "tavily.extract"]
            assert all(c.thread == unit.thread for c in children)
        judges = _by_name(client, "judge.shard")
        assert len(judges) == 2 and {j.parent_id for j in judges} == {step["step.score_providers"].id}
        assert all(s.ended for s in client.spans)
        assert client.flushes >= 1

    def test_a_step_that_never_reported_completion_is_unwound_by_the_next_close(self, traced):
        with tracing.start_run(run_id="run-3") as run:
            run.step_started("score_providers", {})
            run.step_started("enrich_reviews", {})
            run.step_finished("score_providers", "failed", {"error": "boom"})
            assert run.current_parent() is run.root
        client = FakeLangfuse.instances[0]
        assert all(s.ended for s in client.spans)
        failed = [s for s in client.spans if s.name == "step.score_providers"][0]
        assert failed.merged["level"] == "ERROR" and failed.merged["status_message"] == "boom"

    def test_exception_inside_a_generation_marks_the_span_and_propagates(self, traced):
        with tracing.start_run(run_id="run-4"):
            with pytest.raises(RuntimeError):
                with tracing.generation("critic.deep", model="claude-opus-4-8", agent="critic_validator"):
                    raise RuntimeError("credit balance is too low")
        client = FakeLangfuse.instances[0]
        gen = _by_name(client, "critic.deep")[0]
        assert gen.ended and gen.merged["level"] == "ERROR" and "credit balance" in gen.merged["status_message"]


class TestOneSeamTwoSinks:
    def test_generation_writes_the_same_cost_to_tracker_and_span(self, traced):
        response = MagicMock()
        response.usage.input_tokens, response.usage.output_tokens = 2000, 400
        with tracing.start_run(run_id="run-5"):
            with tracing.generation("critic.bias", model="claude-opus-4-8", agent="critic_validator", prompt="P" * 10, params={"max_tokens": 2000}) as gen:
                gen.finish(response, output_text="{}", stop_reason="end_turn")
        client = FakeLangfuse.instances[0]
        span = _by_name(client, "critic.bias")[0]
        expected = (2000 * 5.0 + 400 * 25.0) / 1e6
        assert span.merged["usage_details"] == {"input": 2000, "output": 400, "total": 2400}
        assert span.merged["cost_details"]["total"] == pytest.approx(expected, abs=1e-6)
        assert span.merged["model"] == "claude-opus-4-8"
        assert span.merged["input"]["chars"] == 10 and "preview" in span.merged["input"]
        assert span.merged["metadata"] == {"stop_reason": "end_turn"}
        tracker = get_cost_tracker().summary()
        assert tracker["llm"]["cost_usd"] == pytest.approx(expected, abs=1e-6)
        assert tracker["llm"]["by_agent"]["critic_validator"] == pytest.approx(expected, abs=1e-6)


class TestRunRecordOnTheTrace:
    def test_measurements_become_scores_and_the_rest_becomes_metadata(self, traced):
        record = {f: None for f in RUN_RECORD_FIELDS}
        record.update({
            "run_id": "run-6", "source": "canary", "schema_version": 1, "pool_raw": 120,
            "fallback_fired": False, "cost_usd": 0.55, "stage_timings": {"gather_data": 12.0},
            "extra": {"errors": 0}, "judge_model": "gpt-5.6-terra",
        })
        with tracing.start_run(run_id="run-6") as run:
            run.finish(record=record, output={"recommendations": 5})
        client = FakeLangfuse.instances[0]
        scores = {s["name"]: s for s in client.scores}
        assert scores["pool_raw"] == {"name": "pool_raw", "value": 120.0, "trace_id": "trace-run-6", "data_type": "NUMERIC"}
        assert scores["fallback_fired"]["data_type"] == "BOOLEAN" and scores["fallback_fired"]["value"] == 0
        assert scores["cost_usd"]["value"] == pytest.approx(0.55)
        assert set(scores) <= set(measurement_fields()), "descriptors and structures are never scores"
        assert "judge_model" not in scores and "stage_timings" not in scores
        meta = client.spans[0].merged["metadata"]["run_record"]
        assert meta["judge_model"] == "gpt-5.6-terra" and meta["stage_timings"] == {"gather_data": 12.0}
        assert meta["source"] == "canary" and "pool_raw" not in meta
        assert client.spans[0].ended and client.flushes >= 1

    def test_every_field_is_classified_so_none_silently_drops_off_the_trace(self):
        assert set(measurement_fields()) | DESCRIPTOR_FIELDS | STRUCTURED_FIELDS == set(RUN_RECORD_FIELDS)


class TestBuildRunRecord:
    def _refined(self, state):
        """What refine_rankings hands finalize: copies of the ranked pool with
        the critic's verdict on each — the ONLY place critic_review lives."""
        reviews = [
            {"status": "approved", "judge_findings": ""},
            {"status": "conditional", "judge_findings": "citation off-topic"},
            None,
        ]
        out = []
        for provider, review in zip(state["scored_providers"]["ranked_providers"], reviews):
            copy = dict(provider)
            if review is not None:
                copy["critic_review"] = review
            out.append(copy)
        return out

    def _state(self):
        return {
            "run_id": "r", "run_source": "canary", "case_id": "chandler-neurology",
            "specialty": "Neurology", "location": "Chandler, AZ 85224",
            "preferences": {"search_radius_miles": 10},
            "gathered_data": {"status": "success", "providers": [{}] * 5, "search_metadata": {
                "fetch_mode": "extract", "fetch_mode_fallback": False, "total_found": 5,
                "radius_dropped": 2, "ring_expanded": False, "ring_added": 0,
                "identity_contradictions": [{"a": 1}],
                "fetch_stats": {
                    "fetch": {
                        "discovery": {"planned": 9, "fetched": 8, "empty": 1, "failed": 0, "by_domain": {}},
                        "enrichment": {"planned": 4, "fetched": 3, "empty": 1, "failed": 0,
                                       "by_domain": {"healthgrades.com": {"fetched": 1, "empty": 1}, "vitals.com": {"fetched": 2, "empty": 0}}},
                    },
                    "listing_rows": {"healthgrades.com": 20, "doctor.webmd.com": 46, "vitals.com": 45},
                    "specialty_rows_dropped": 3,
                },
            }},
            "scored_providers": {"scoring_metadata": {"judge_shards": {"total": 2, "ok": 2, "truncated": 0}}, "ranked_providers": [
                {"enrichment_outcome": "enriched", "platform_pair_count": 3, "profile_backed_platforms": 3, "ai_judged": True,
                 "ai_rubric": {"a": 1}, "ai_evidence": {"review_substance": "quote", "practical_access": "no evidence"},
                 "platform_profile_urls": {"healthgrades.com": "u1", "doctor.webmd.com": "u2", "vitals.com": "u3"},
                 # `via` rides inside `yielded`, the shape the gatherer writes.
                 "enrichment_sources": [{"url": "u1", "yielded": {"via": "profile_parser"}}, {"url": "u2", "yielded": {"via": "llm"}}]},
                {"enrichment_outcome": "no_profile_found", "platform_pair_count": 1, "profile_backed_platforms": 0, "ai_judged": True,
                 "ai_rubric": {"a": 1}, "ai_evidence": {"review_substance": "no evidence"},
                 "platform_profile_urls": {"vitals.com": "u3"}, "enrichment_sources": []},
                {"enrichment_outcome": "over_budget"},
            ]},
            "validation_results": {"validation_metadata": {"call_timings": [{"call": "bias"}, {"call": "deep_1", "failed": True}, {"call": "deep_2", "failed": False}]}},
            "final_recommendations": [{}],
            "workflow_summary": {"withheld": {"no_data": 1, "pipeline_failures": 0, "not_researched": 1},
                                 "cost_summary": {"total_usd": 0.41, "elapsed_s": 58.2,
                                                  "tavily": {"credits": 7, "credits_by_stage": {"discovery": 3, "enrichment": 4}},
                                                  "llm": {"input_tokens": 90000, "output_tokens": 9000}}},
            "execution_log": [
                {"step": "gather_data", "status": "completed", "details": {"elapsed_s": 12.5}},
                {"step": "validate_rankings", "status": "completed", "details": {"elapsed_s": 20.0, "validation_collapsed": False}},
            ],
            "error_messages": [],
        }

    def test_fields_come_from_the_finished_state(self):
        config = types.SimpleNamespace(TAVILY_MODE="extract", GATHERER_MODEL="claude-haiku-4-5", JUDGE_MODEL="gpt-5.6-terra",
                                       CRITIC_MODEL="claude-opus-4-8", MAX_PROVIDERS_TO_ENRICH=8, DEFAULT_SEARCH_RADIUS=25)
        state = self._state()
        r = build_run_record(state, config, providers=self._refined(state))
        assert set(r) == set(RUN_RECORD_FIELDS)
        assert (r["source"], r["case_id"], r["city"], r["state"], r["zip_present"], r["radius_miles"]) == ("canary", "chandler-neurology", "Chandler", "AZ", True, 10)
        assert (r["pages_planned"], r["pages_fetched"], r["empty_bodies"], r["pages_failed"]) == (9, 8, 1, 0)
        assert (r["rows_hg"], r["rows_wm"], r["rows_vi"]) == (20, 46, 45)
        assert (r["pool_raw"], r["pool_after_specialty"], r["pool_after_radius"], r["radius_dropped"]) == (7, 7, 5, 2)
        assert (r["n_enriched"], r["n_no_profile_found"], r["n_over_budget"], r["n_cached"]) == (1, 1, 1, 0)
        assert r["pairs_hist"] == {"1": 1, "3": 1} and r["profile_backed_hist"] == {"0": 1, "3": 1}
        assert (r["coverage_hg"], r["coverage_wm"], r["coverage_vi"], r["all_three"]) == (1, 1, 2, 1)
        assert (r["via_parser"], r["via_llm"]) == (1, 1)
        assert r["empty_body_rate"] == {"hg": 0.5, "vi": 0.0}
        assert (r["credits_discovery"], r["credits_enrichment"], r["tavily_credits"]) == (3, 4, 7)
        assert (r["judge_shards_ok"], r["judge_truncated"], r["judge_applied"]) == (2, False, 2)
        assert r["neutral_band_rate"] == pytest.approx(2 / 3, abs=1e-4) and r["evidence_present_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert (r["critic_shards_failed"], r["validation_collapsed"], r["verdict_hist"]) == (1, False, {"approved": 1, "conditional": 1})
        assert (r["judge_findings"], r["conditional_rate"], r["identity_contradictions"]) == (1, 0.5, 1)
        assert (r["shortlist_size"], r["withheld_coverage"], r["withheld_ours"], r["withheld_budget"]) == (1, 1, 0, 1)
        assert (r["cost_usd"], r["tokens_in"], r["tokens_out"], r["latency_s"]) == (0.41, 90000, 9000, 58.2)
        assert r["stage_timings"] == {"gather_data": 12.5, "validate_rankings": 20.0}
        assert r["extra"]["specialty_rows_dropped"] == 3

    def test_without_the_refined_pool_the_critic_fields_are_honestly_none(self):
        """critic_review lives on the copies refine_rankings returns, never on
        scored_providers.ranked_providers — the first live Tier B canary
        recorded verdict_hist None beside a critic log full of verdicts
        because the builder read the pre-refinement list."""
        config = types.SimpleNamespace(TAVILY_MODE="extract", GATHERER_MODEL="m", JUDGE_MODEL="j",
                                       CRITIC_MODEL="c", MAX_PROVIDERS_TO_ENRICH=8, DEFAULT_SEARCH_RADIUS=25)
        r = build_run_record(self._state(), config)
        assert (r["verdict_hist"], r["judge_findings"], r["conditional_rate"]) == (None, None, None)
        assert (r["n_enriched"], r["judge_applied"], r["via_parser"], r["via_llm"]) == (1, 2, 1, 1)

    def test_unknowns_are_none_not_zero(self):
        r = build_run_record({"specialty": "Neurology", "location": "Chandler, AZ"}, types.SimpleNamespace())
        for field in ("pages_fetched", "n_enriched", "judge_applied", "critic_shards_failed", "cost_usd", "coverage_hg", "credits_discovery"):
            assert r[field] is None, field
        assert r["shortlist_size"] == 0 and r["n_over_budget"] == 0


class TestCiWorkflow:
    def test_workflow_runs_the_offline_suite_with_lfs_and_no_secrets(self):
        import yaml

        path = REPO / ".github" / "workflows" / "ci.yml"
        text = path.read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
        assert "pull_request" in doc[True] if True in doc else "pull_request" in doc["on"]
        steps = doc["jobs"]["unit-suite"]["steps"]
        checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout"))
        assert checkout["with"]["lfs"] is True, "geo data is LFS-tracked; without it the suite goes red"
        run_lines = [s["run"] for s in steps if "run" in s]
        assert any("pytest" in r and "--no-cov" in r for r in run_lines)
        assert "secrets." not in text, "the PR suite must stay offline and secret-free"
        # The SCHEDULED_RUNS opt-in gates the three CRON workflows only. CI
        # runs on every push and PR in EVERY repo, flag or no flag — a gate
        # here would let a repo without the variable merge red code green.
        assert "if" not in doc["jobs"]["unit-suite"] and "SCHEDULED_RUNS" not in text
