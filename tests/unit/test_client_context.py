"""The search criteria and the visitor's coarse whereabouts on the run
record — and what is deliberately NOT captured.

The owner asked whether the weekly report could say what each live search
asked for and where it came from, "or IP address". What was built: the
three weights join the record beside the inputs it already carried
(specialty, city, state, ZIP-present, radius) as `extra.weights`; the
visitor's country / region (from the headers the hosting proxy adds),
timezone and locale (from the browser) join as `extra.client`; and the IP
address is stored NOWHERE — not on the trace, not in the record, not in a
log. A salted 16-hex pseudonym of it rides on the TRACE's metadata alone,
and only when `VISITOR_HASH_SALT` is set, so distinct visitors can be
counted on the private trace store; the record, and therefore the public
rows in `reports/`, never carry it. This is a healthcare-adjacent app
whose run records ship to a public repository, and nothing the monitoring
needs requires an address.

What these pin: the capture reads coarse facts and never the address;
no salt means no pseudonym; the capture returns {} rather than raising
when there is no request (a worker thread, this test process); the
orchestrator puts the facts and the pseudonym on the trace and ONLY the
public subset on the record; the app captures on the script thread and
hands the worker the result; the weekly report renders the criteria and
the coarse origin; and the only module in the codebase that reads the
address is the one that hashes it.
"""

import inspect
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents.orchestrator import ProviderMatchingOrchestrator
from evals import weekly_report
from tests.fixtures.mock_agent_responses import (
    MOCK_GATHER_PROVIDERS_RESULT,
    MOCK_SCORED_PROVIDERS_RESULT,
    MOCK_VALIDATION_RESULT,
)
from tests.helpers.fake_langfuse import FakeLangfuse, install_fake_tracing
from tests.unit.test_observability_p3 import fake_scores, fake_trace, minimal_state
from utils import tracing
from utils.client_context import PUBLIC_KEYS, capture_client_context, public_subset, visitor_id
from utils.config import Config
from utils.run_record import build_run_record

REPO = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 9, 13, 13, 0, tzinfo=timezone.utc)
IP = "203.0.113.7"
CLIENT = {"country": "US", "region": "AZ", "timezone": "America/Phoenix", "locale": "en-US"}
WEIGHTS = {"rating_weight": 0.5, "location_weight": 0.3, "experience_weight": 0.2}


def ctx(**over):
    base = dict(headers={"CF-IPCountry": "us", "cf-region-code": "AZ"}, timezone="America/Phoenix",
                locale="en-US", ip_address=IP)
    base.update(over)
    return SimpleNamespace(**base)


class TestCapture:
    def test_coarse_facts_only_and_the_address_never_appears(self):
        out = capture_client_context(ctx(), env={"VISITOR_HASH_SALT": "pepper"})
        assert out == {**CLIENT, "visitor": visitor_id(IP, "pepper")}
        assert re.fullmatch(r"[0-9a-f]{16}", out["visitor"])
        assert IP not in json.dumps(out) and "ip" not in out

    def test_no_salt_means_no_pseudonym_and_the_pseudonym_is_salted(self):
        out = capture_client_context(ctx(), env={})
        assert "visitor" not in out and out["country"] == "US"
        assert visitor_id(IP, None) is None and visitor_id(None, "pepper") is None and visitor_id("", "pepper") is None
        assert visitor_id(IP, "pepper") != visitor_id(IP, "other")
        assert visitor_id(IP, "pepper") != visitor_id("203.0.113.8", "pepper")
        assert visitor_id(IP, "pepper") == visitor_id(IP, "pepper"), "stable, so distinct visitors can be counted"

    def test_other_hosts_headers_and_case_insensitive_names(self):
        out = capture_client_context(ctx(headers={"X-Vercel-IP-Country": "de", "x-vercel-ip-country-region": "BE"}), env={})
        assert out == {"country": "DE", "region": "BE", "timezone": "America/Phoenix", "locale": "en-US"}
        assert capture_client_context(ctx(headers={"CloudFront-Viewer-Country": "ca"}), env={})["country"] == "CA"

        class OnlyGet:  # a headers object that is not a mapping but answers .get
            def get(self, name):
                return {"cf-ipcountry": "gb"}.get(name)

        assert capture_client_context(ctx(headers=OnlyGet()), env={})["country"] == "GB"

    def test_nothing_known_is_an_empty_dict_and_a_failure_never_raises(self):
        assert capture_client_context(SimpleNamespace(headers={}, timezone=None, locale=None, ip_address=None), env={}) == {}

        class Exploding:
            @property
            def headers(self):
                raise RuntimeError("no request")

        assert capture_client_context(Exploding(), env={"VISITOR_HASH_SALT": "x"}) == {}
        # This process runs no Streamlit script: the real context has no request.
        assert capture_client_context() == {}
        assert capture_client_context(ctx(timezone="x" * 100), env={})["timezone"] == "x" * 40

    def test_the_public_subset_drops_the_pseudonym(self):
        assert public_subset({**CLIENT, "visitor": "deadbeefdeadbeef"}) == CLIENT
        assert public_subset({"visitor": "deadbeefdeadbeef"}) is None
        assert public_subset(None) is None and public_subset({}) is None and public_subset("US") is None
        assert "visitor" not in PUBLIC_KEYS and "ip" not in PUBLIC_KEYS


class TestRecord:
    def test_weights_and_the_public_facts_join_extra_and_the_pseudonym_never_does(self):
        state = minimal_state(preferences={**WEIGHTS, "search_radius_miles": 25},
                              client={**CLIENT, "visitor": "deadbeefdeadbeef"})
        record = build_run_record(state)
        assert record["extra"]["weights"] == {"rating": 0.5, "location": 0.3, "experience": 0.2}
        assert record["extra"]["client"] == CLIENT and record["radius_miles"] == 25
        text = json.dumps(record)
        assert "deadbeefdeadbeef" not in text and "visitor" not in text
        # Unknown is None, never {}: a row without weights is an old row, not a search with none.
        bare = build_run_record(minimal_state(preferences={"search_radius_miles": 25}, client={}))
        assert bare["extra"]["weights"] is None and bare["extra"]["client"] is None
        assert build_run_record(minimal_state(preferences={"rating_weight": True}))["extra"]["weights"] is None
        assert build_run_record(minimal_state(preferences={"rating_weight": 0.33333}))["extra"]["weights"] == {"rating": 0.333}


@pytest.fixture
def orchestrator():
    with patch("agents.orchestrator.DataGathererAgent"), patch("agents.orchestrator.PreferenceScorerAgent"), \
         patch("agents.orchestrator.CriticValidatorAgent"), patch("agents.orchestrator.get_vector_store"):
        orch = ProviderMatchingOrchestrator()
        orch.data_gatherer.gather_providers.return_value = MOCK_GATHER_PROVIDERS_RESULT
        orch.preference_scorer.score_providers.return_value = MOCK_SCORED_PROVIDERS_RESULT
        orch.critic_validator.validate_rankings.return_value = MOCK_VALIDATION_RESULT
        yield orch


@pytest.fixture
def traced(monkeypatch):
    captured = install_fake_tracing(monkeypatch)
    yield captured
    tracing.reset_client()


class TestTraceAndOrchestrator:
    def test_the_trace_carries_the_facts_and_the_pseudonym_the_record_only_the_public_facts(self, traced, orchestrator):
        result = orchestrator.execute_workflow(specialty="Neurology", location="Phoenix, AZ", preferences=dict(WEIGHTS),
                                               client={**CLIENT, "visitor": "deadbeefdeadbeef"})
        assert result["success"]
        assert traced["metadata"]["client_country"] == "US" and traced["metadata"]["client_region"] == "AZ"
        assert traced["metadata"]["client_timezone"] == "America/Phoenix" and traced["metadata"]["client_locale"] == "en-US"
        assert traced["metadata"]["visitor"] == "deadbeefdeadbeef" and "country:US" in traced["tags"]
        assert not any("ip" in k.split("_") for k in traced["metadata"]), "no address, under any key"
        record = result["workflow_summary"]["run_record"]
        assert record["extra"]["client"] == CLIENT
        assert record["extra"]["weights"] == {"rating": 0.5, "location": 0.3, "experience": 0.2}
        root = FakeLangfuse.instances[0].spans[0]
        assert "deadbeefdeadbeef" not in json.dumps(root.merged["metadata"]["run_record"])

    def test_without_client_facts_the_trace_and_the_record_are_as_before(self, traced, orchestrator):
        result = orchestrator.execute_workflow(specialty="Neurology", location="Phoenix, AZ")
        assert not any(k.startswith("client_") or k == "visitor" for k in traced["metadata"])
        assert not any(t.startswith("country:") for t in traced["tags"])
        assert result["workflow_summary"]["run_record"]["extra"]["client"] is None

    def test_the_streaming_entry_point_takes_the_same_argument(self, traced, orchestrator):
        result = orchestrator.execute_workflow_streaming(specialty="Neurology", location="Phoenix, AZ",
                                                         progress_callback=lambda _u: None, client={"country": "US"})
        assert result["success"] and traced["metadata"]["client_country"] == "US"
        assert result["workflow_summary"]["run_record"]["extra"]["client"] == {"country": "US"}


class TestAppAndPrivacyGuards:
    def test_the_app_captures_on_the_script_thread_and_hands_the_worker_the_result(self):
        import app as app_module

        source = inspect.getsource(app_module._start_search_job)
        assert "capture_client_context()" in source, "captured before the worker is submitted — st.context is request-bound"
        assert "client=client" in source
        assert source.index("capture_client_context()") < source.index("pool.submit(")

    def test_only_the_context_module_reads_the_address_and_it_stores_none(self):
        readers = []
        for path in [REPO / "app.py"] + [p for d in ("agents", "utils", "evals", "fhir") for p in (REPO / d).rglob("*.py")]:
            if "ip_address" in path.read_text(encoding="utf-8"):
                readers.append(path.relative_to(REPO).as_posix())
        assert readers == ["utils/client_context.py"]
        source = (REPO / "utils" / "client_context.py").read_text(encoding="utf-8")
        assert "logger" not in source and "logging" not in source, "an address must not reach a log line either"

    def test_the_salt_is_an_optional_env_knob_documented_as_such(self, monkeypatch):
        monkeypatch.delenv("VISITOR_HASH_SALT", raising=False)
        assert Config().VISITOR_HASH_SALT is None
        monkeypatch.setenv("VISITOR_HASH_SALT", "pepper")
        assert Config().VISITOR_HASH_SALT == "pepper"
        text = (REPO / ".env.example").read_text(encoding="utf-8")
        assert re.search(r"^# VISITOR_HASH_SALT=", text, re.M) and "raw IP is never stored" in text


class TestWeeklyReportLiveSearches:
    def test_the_table_states_criteria_and_coarse_origin_never_the_pseudonym(self):
        trace = fake_trace("u1", NOW, source="user", case_id=None, tier="B",
                           extra_metadata={"visitor": "deadbeefdeadbeef", "client_country": "US"})
        trace.metadata["run_record"]["extra"].update({"weights": {"rating": 0.5, "location": 0.3, "experience": 0.2}, "client": CLIENT})
        row = weekly_report.row_from_trace(trace, fake_scores("u1", pool_raw=117, shortlist_size=5, cost_usd=0.56, latency_s=70))
        assert row["extra"]["client"] == CLIENT and "deadbeef" not in json.dumps(row)
        start, end, label = weekly_report.week_bounds(NOW, 0)
        text = weekly_report.render_report(label, start, end, [row], NOW)
        assert "## Live searches (source: user)" in text
        assert "| 2026-09-13 13:00 | Neurology | Chandler, AZ | no | 25 mi | 0.50 / 0.30 / 0.20 | US / AZ | 117 | 5 | $0.560 | 70 s |" in text
        assert "deadbeef" not in text and "never an address" in text
        # A row from before this change renders dashes; a week without user searches says so.
        old = weekly_report.row_from_trace(fake_trace("u2", NOW, source="user", case_id=None), fake_scores("u2", pool_raw=50, cost_usd=0.4))
        text = weekly_report.render_report(label, start, end, [old], NOW)
        assert "| Chandler, AZ | no | 25 mi | — / — / — | — | 50 | — | $0.400 | — |" in text
        assert "No live (user) searches recorded this week." in weekly_report.render_report(label, start, end, [], NOW)
        # Canary rows are not live searches.
        canary = weekly_report.row_from_trace(fake_trace("c1", NOW), fake_scores("c1", pool_raw=117))
        assert "No live (user) searches recorded this week." in weekly_report.render_report(label, start, end, [canary], NOW)
