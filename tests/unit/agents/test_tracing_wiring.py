"""Observability P1: the seams are WIRED, not just available.

A helper-only test lets the wiring be deleted with the suite still green, so
these drive the real agents against a fake Langfuse client and assert what
lands on the tree: a Tavily extract batch becomes a tool span carrying page
counts and hashes (never text) and feeds the fetch accounting; a provider's
enrichment runs inside its own span; a whole orchestrated run yields a root
seeded by the run id, one span per workflow step, the run record as scores,
and a trace URL in the result. A source guard pins that no model call records
cost anywhere but the seam — the "one seam, two sinks" rule.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.data_gatherer import DataGathererAgent
from agents.orchestrator import ProviderMatchingOrchestrator
from tests.fixtures.mock_agent_responses import (
    MOCK_GATHER_PROVIDERS_RESULT,
    MOCK_SCORED_PROVIDERS_RESULT,
    MOCK_VALIDATION_RESULT,
)
from tests.helpers.fake_langfuse import FakeLangfuse, by_name, install_fake_tracing
from utils import tracing
from utils.cost_tracker import get_cost_tracker
from utils.run_record import RUN_RECORD_FIELDS, measurement_fields

REPO = Path(__file__).resolve().parents[3]
HG = "https://www.healthgrades.com/physician/dr-kan-yu-2b5bc"
VI = "https://www.vitals.com/doctors/dr-kan-yu"


@pytest.fixture
def traced(monkeypatch):
    captured = install_fake_tracing(monkeypatch)
    yield captured
    tracing.reset_client()


def _gatherer():
    gatherer = DataGathererAgent()
    gatherer.tavily_client = MagicMock()
    return gatherer


class TestGathererFetchSpans:
    def test_extract_batch_is_a_tool_span_with_digests_and_feeds_fetch_stats(self, traced):
        gatherer = _gatherer()
        body = "PRIVATE-LOOKING REVIEW PROSE " * 50
        gatherer.tavily_client.extract.return_value = {
            "results": [{"url": HG, "raw_content": body}, {"url": VI, "raw_content": ""}],
            "failed_results": [{"url": "https://doctor.webmd.com/x", "error": "404"}],
        }
        with tracing.start_run(run_id="run-x"):
            pages = gatherer._extract_pages([HG, VI, "https://doctor.webmd.com/x"], purpose="enrichment: Dr. Kan Yu", stage="enrichment")
        assert [p["url"] for p in pages] == [HG]
        client = FakeLangfuse.instances[0]
        call = by_name(client, "tavily.extract")[0]
        assert call.as_type == "tool"
        assert call.merged["input"] == {"url_count": 3, "purpose": "enrichment: Dr. Kan Yu", "stage": "enrichment", "attempt": 1}
        out = call.merged["output"]
        assert (out["fetched"], out["empty_bodies"], out["failed"], out["credits"]) == (1, 1, 1, 1)
        # Credits are money: the first live trace summed model spans only and
        # disagreed with the cost card by exactly the Tavily line.
        assert call.merged["cost_details"] == {"total": pytest.approx(1 * 0.008)}
        # Langfuse keeps cost on generation-like observations only (a live
        # probe, 2026-09-13), so the dollar figure a reader can see on a tool
        # span is the one in its metadata.
        assert call.merged["metadata"]["cost_usd"] == pytest.approx(0.008)
        assert out["pages"] == [{"url": HG, "chars": len(body), "sha256": tracing.sha256_text(body)[:16]}]
        assert "REVIEW PROSE" not in str(call.merged), "page text must never reach the trace"

        stats = gatherer.fetch_stats()
        enrichment = stats["fetch"]["enrichment"]
        assert (enrichment["planned"], enrichment["fetched"], enrichment["empty"], enrichment["failed"]) == (3, 1, 1, 1)
        assert enrichment["by_domain"] == {"healthgrades.com": {"fetched": 1, "empty": 0}, "vitals.com": {"fetched": 0, "empty": 1}}
        assert stats["fetch"]["discovery"]["planned"] == 0
        assert get_cost_tracker().summary()["tavily"]["credits_by_stage"] == {"enrichment": 1}

    def test_search_call_is_a_tool_span_and_counts_empty_bodies(self, traced):
        gatherer = _gatherer()
        gatherer.tavily_client.search.return_value = {"results": [
            {"url": HG, "raw_content": "x" * 10}, {"url": VI, "raw_content": ""},
        ]}
        with tracing.start_run(run_id="run-y"):
            results = gatherer._search_providers("neurologist chandler", max_results=5, include_domains=["healthgrades.com"], search_depth="basic")
        assert len(results) == 2
        client = FakeLangfuse.instances[0]
        call = by_name(client, "tavily.search")[0]
        assert call.merged["input"]["query"] == "neurologist chandler" and call.merged["input"]["stage"] == "discovery"
        assert call.merged["output"]["results"] == 2 and call.merged["output"]["empty_bodies"] == 1
        assert call.merged["cost_details"]["total"] == pytest.approx(0.008)
        assert gatherer.fetch_stats()["fetch"]["discovery"]["empty"] == 1
        assert get_cost_tracker().summary()["tavily"]["credits_by_stage"] == {"discovery": 1}

    def test_a_provider_is_enriched_inside_its_own_span(self, traced, monkeypatch):
        gatherer = _gatherer()

        def fake_untraced(provider, location, specialty="", user_location=""):
            # The body runs on whatever thread the worker owns; its own trace
            # calls must nest under the provider span, so open one here.
            with tracing.tool("tavily.extract", input={"url_count": 1}) as call:
                call.finish(output={"fetched": 1})
            provider["enrichment_outcome"] = "enriched"
            provider["platform_pair_count"] = 2

        monkeypatch.setattr(gatherer, "_enrich_one_untraced", fake_untraced)
        provider = {"name": "Dr. Kan Yu", "platform_profile_urls": {"healthgrades.com": HG}}
        with tracing.start_run(run_id="run-z"):
            gatherer._enrich_one(provider, "Chandler, AZ", "Neurology")
        client = FakeLangfuse.instances[0]
        unit = by_name(client, "enrichment.provider")[0]
        assert unit.merged["input"] == {"provider": "Dr. Kan Yu", "known_profile_urls": 1}
        assert unit.merged["output"]["outcome"] == "enriched" and unit.merged["output"]["platform_pairs"] == 2
        assert by_name(client, "tavily.extract")[0].parent_id == unit.id


@pytest.fixture
def orchestrator():
    with patch("agents.orchestrator.DataGathererAgent"), patch("agents.orchestrator.PreferenceScorerAgent"), \
         patch("agents.orchestrator.CriticValidatorAgent"), patch("agents.orchestrator.get_vector_store"):
        orch = ProviderMatchingOrchestrator()
        orch.data_gatherer.gather_providers.return_value = MOCK_GATHER_PROVIDERS_RESULT
        orch.preference_scorer.score_providers.return_value = MOCK_SCORED_PROVIDERS_RESULT
        orch.critic_validator.validate_rankings.return_value = MOCK_VALIDATION_RESULT
        # A MagicMock gatherer answers fetch_stats() with a MagicMock; the
        # orchestrator must ignore anything that is not a dict.
        yield orch


class TestOrchestratedRunTrace:
    def test_a_run_is_one_trace_with_steps_scores_and_a_url(self, traced, orchestrator):
        result = orchestrator.execute_workflow(specialty="Neurology", location="Phoenix, AZ", source="canary", case_id="phoenix-neurology")
        assert result["success"]
        assert result["run_id"] and result["trace_url"] == f"https://lf.example/trace/trace-{result['run_id']}"

        client = FakeLangfuse.instances[0]
        root = client.spans[0]
        assert root.name == "carecompass.search" and root.trace_context == {"trace_id": f"trace-{result['run_id']}"}
        assert root.merged["input"]["specialty"] == "Neurology"
        assert traced["metadata"]["source"] == "canary" and traced["metadata"]["case_id"] == "phoenix-neurology"
        assert "source:canary" in traced["tags"] and "case:phoenix-neurology" in traced["tags"]

        steps = [s.name for s in client.spans if s.name.startswith("step.")]
        for expected in ("step.initialize", "step.gather_data", "step.score_providers", "step.enrich_reviews", "step.validate_rankings", "step.finalize_results"):
            assert expected in steps, expected
        step = {s.name: s for s in client.spans if s.name.startswith("step.")}
        assert step["step.enrich_reviews"].parent_id == step["step.score_providers"].id
        assert all(s.ended for s in client.spans)

        record = result["workflow_summary"]["run_record"]
        assert set(record) == set(RUN_RECORD_FIELDS)
        assert record["run_id"] == result["run_id"] and record["source"] == "canary"
        # The orchestrator states that the pipeline ran, and hands finalize's
        # REFINED pool to the builder so the critic's verdicts are counted
        # (the first live Tier B canary recorded verdict_hist None beside a
        # critic log full of verdicts — the builder had the unrefined list).
        assert record["extra"]["pipeline"] is True
        assert record["verdict_hist"], "critic verdicts must reach the record through the refined pool"
        scored = {s["name"] for s in client.scores}
        assert "shortlist_size" in scored and scored <= set(measurement_fields())
        assert root.merged["metadata"]["run_record"]["source"] == "canary"
        assert root.merged["output"]["recommendations"] == len(result["final_recommendations"])
        assert client.flushes >= 1

    def test_workflow_id_is_the_run_id_prefix_so_log_and_trace_name_one_run(self, traced, orchestrator):
        result = orchestrator.execute_workflow(specialty="Neurology", location="Phoenix, AZ")
        assert result["workflow_summary"]["workflow_id"] == result["run_id"][:8]
        assert result["workflow_summary"]["run_record"]["source"] == "user"

    def test_untraced_run_is_unchanged_except_for_the_new_keys(self, monkeypatch, orchestrator):
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        tracing.reset_client()
        result = orchestrator.execute_workflow(specialty="Neurology", location="Phoenix, AZ")
        assert result["success"] and result["trace_url"] is None and result["run_id"]
        assert set(result["workflow_summary"]["run_record"]) == set(RUN_RECORD_FIELDS)
        tracing.reset_client()


class TestOneSeamGuard:
    def test_no_model_call_records_cost_outside_the_seam(self):
        # Every messages.create / completions.create / embeddings.create in
        # the agents sits inside `tracing.generation(...)`, and nothing else
        # calls record_llm: a second writer is a second dollar figure.
        offenders = []
        for path in list((REPO / "agents").glob("*.py")) + [REPO / "utils" / "vector_store.py"]:
            text = path.read_text(encoding="utf-8")
            if "record_llm(" in text:
                offenders.append(f"{path.name}: record_llm")
            for match in re.finditer(r"\.(messages|chat\.completions|embeddings)\.create\(", text):
                window = text[max(0, match.start() - 600):match.start()]
                if "tracing.generation(" not in window:
                    offenders.append(f"{path.name}: create() at offset {match.start()} is outside a generation seam")
        assert offenders == []
