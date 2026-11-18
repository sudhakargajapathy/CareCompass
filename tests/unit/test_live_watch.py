"""The live-app watch: Tavily errors and latency in the Space's own traffic.

Every other monitor measures a fresh process, so the November 2026 live
searches (100-120 s, against ~40 s in CI) were invisible to all of them for
two weeks while Langfuse held 34 first-attempt `tavily.extract` errors, every
one in the `demo` environment. These pin that the watch reads the live
environment, counts what it must, publishes nothing a visitor typed, and
that the watcher judges and scopes it.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import live_watch, thresholds, watch
from evals.thresholds import LIVE_CASE_ID, evaluate_live, live_scope
from utils.tracing import ROOT_NAME

REPO = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 17, 13, 0, tzinfo=timezone.utc)


def _trace(trace_id, env, source="user", specialty="Cardiology", location="Chandler, AZ"):
    return SimpleNamespace(
        id=trace_id, environment=env, tags=[f"source:{source}"], timestamp=NOW - timedelta(hours=2),
        metadata={"run_record": {"source": source, "run_id": trace_id,
                                 "specialty": specialty, "location": location}},
    )


def _score(name, value):
    return SimpleNamespace(name=name, value=value, data_type="NUMERIC")


def _obs(trace_id, env, attempt, message, name="tavily.extract"):
    raw = json.dumps({"url_count": 4, "purpose": "Dr. Somebody profiles", "stage": "enrichment",
                      "attempt": attempt})
    return SimpleNamespace(trace_id=trace_id, environment=env, input=raw, status_message=message, name=name)


class FakeClient:
    """The two v4 read APIs the watch uses, with call recording.

    Observations API v2 serves both reads: the runs (each run's ROOT
    observation, by the root's name, which `weekly_report.fetch_rows` turns
    into rows) and the Tavily ERROR observations. Scores come from Scores
    API v3.
    """

    def __init__(self, traces, scores, observations, pages=1):
        self.calls = []
        obs_ns = SimpleNamespace(get_many=self._observations)
        scores_ns = SimpleNamespace(get_many_v3=self._scores)
        self.api = SimpleNamespace(observations=obs_ns, scores_v3=scores_ns)
        self._traces, self._score_map, self._obs, self._pages = traces, scores, observations, pages

    def _scores(self, trace_id, **kw):
        return SimpleNamespace(data=self._score_map.get(trace_id, []), meta=SimpleNamespace(cursor=None))

    def _roots(self, **kw):
        self.calls.append(("runs", kw))
        roots = [SimpleNamespace(trace_id=t.id, start_time=t.timestamp, tags=t.tags,
                                 environment=t.environment, metadata=t.metadata) for t in self._traces]
        return SimpleNamespace(data=roots, meta=SimpleNamespace(cursor=None))

    def _observations(self, **kw):
        if kw["name"] == ROOT_NAME:
            return self._roots(**kw)
        self.calls.append(("observations", kw))
        items = [o for o in self._obs if o.name == kw["name"]]
        if self._pages == 1:
            return SimpleNamespace(data=items, meta=SimpleNamespace(cursor=None))
        # Two pages, joined by a cursor.
        half = len(items) // 2
        if kw.get("cursor") is None:
            return SimpleNamespace(data=items[:half], meta=SimpleNamespace(cursor="next"))
        return SimpleNamespace(data=items[half:], meta=SimpleNamespace(cursor=None))


def _client(pages=1):
    traces = [
        _trace("t-slow", "demo"),
        _trace("t-fast", "demo"),
        _trace("t-canary", "demo", source="canary"),   # a canary under the live label
        _trace("t-ci", "ci"),                           # the server ignored the filter
    ]
    scores = {
        "t-slow": [_score("latency_s", 117.0), _score("cost_usd", 0.55)],
        "t-fast": [_score("latency_s", 41.0), _score("cost_usd", 0.50)],
        "t-canary": [_score("latency_s", 300.0), _score("cost_usd", 0.50)],
        "t-ci": [_score("latency_s", 400.0), _score("cost_usd", 0.50)],
    }
    observations = [
        _obs("t-slow", "demo", 1, "Request timed out after 30 seconds."),
        _obs("t-slow", "demo", 1, "SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred"),
        _obs("t-slow", "demo", 2, "UsageLimitExceededError: rate limited"),
        _obs("t-fast", "demo", 1, "Request timed out after 15 seconds.", name="tavily.search"),
        _obs("t-ci", "ci", 1, "Request timed out after 30 seconds."),
    ]
    return FakeClient(traces, scores, observations, pages=pages)


def test_measure_counts_the_live_environment_only():
    client = _client()
    payload = live_watch.measure(client, ["demo"], now=NOW)

    # Visitor runs in the live environment only: the canary under the live
    # label and the CI trace the server failed to filter are both excluded.
    assert payload["runs"] == 2
    assert payload["latency_s"] == {"p50": 41.0, "p95": 117.0, "max": 117.0, "measured": 2}
    tavily = payload["tavily"]
    assert tavily["first_attempt_errors"] == 3
    assert tavily["retry_errors"] == 1
    assert tavily["runs_affected"] == 2
    assert tavily["by_kind"] == {"timeout": 2, "connection": 1, "other": 1}
    # Both tool names were asked for, server-filtered to the live label and
    # to ERROR, over the window.
    asked = [kw for kind, kw in client.calls if kind == "observations"]
    assert {kw["name"] for kw in asked} == {"tavily.extract", "tavily.search"}
    assert all(kw["level"] == "ERROR" and kw["environment"] == ["demo"] for kw in asked)
    assert all(kw["to_start_time"] - kw["from_start_time"] == timedelta(hours=live_watch.WINDOW_HOURS)
               for kw in asked)


def test_the_payload_carries_counts_and_nothing_a_visitor_typed():
    """Visitor searches are never published, and the artifact this lands in
    is readable by anyone who can read the repository."""
    text = json.dumps(live_watch.measure(_client(), ["demo"], now=NOW))
    for leaked in ("Cardiology", "Chandler", "t-slow", "t-fast", "Somebody", "rate limited"):
        assert leaked not in text


def test_the_observation_cursor_is_followed():
    payload = live_watch.measure(_client(pages=2), ["demo"], now=NOW)
    assert payload["tavily"]["first_attempt_errors"] == 3


@pytest.mark.parametrize("message,kind", [
    ("Request timed out after 30 seconds.", "timeout"),
    ("HTTPSConnectionPool: Read timed out", "timeout"),
    ("SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred'))", "connection"),
    ("ConnectionError: Connection reset by peer", "connection"),
    ("BadRequestError: invalid url", "other"),
    (None, "other"),
])
def test_error_kinds(message, kind):
    assert live_watch.error_kind(message) == kind


def test_unset_environments_measure_nothing(tmp_path, capsys):
    """Unset must not mean every environment: that would count the canaries'
    fresh processes as the live app and vouch for it."""
    out = tmp_path / "live.json"
    assert live_watch.main(["--out", str(out)], env={}) == 0
    assert not out.exists()
    assert "nothing measured" in capsys.readouterr().out


def test_a_failed_pull_is_red(tmp_path, monkeypatch):
    def boom(env):
        raise RuntimeError("401")
    monkeypatch.setattr(live_watch, "_client", boom)
    env = {"LIVE_WATCH_ENVIRONMENTS": "demo", "LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk",
           "LANGFUSE_BASE_URL": "https://example.test"}
    assert live_watch.main(["--out", str(tmp_path / "live.json")], env=env) == 1


# --- judging ------------------------------------------------------------------

def _payload(first=0, retry=0, p95=None, runs=1):
    return {"runs": runs, "window": {"hours": 26}, "environments": ["demo"],
            "latency_s": {"p95": p95}, "tavily": {"first_attempt_errors": first, "retry_errors": retry,
                                                 "runs_affected": 1, "by_kind": {"timeout": first}}}


def test_one_stray_first_attempt_error_is_not_a_finding():
    assert evaluate_live(_payload(first=1, p95=48.0)) == []


def test_the_november_pattern_fires():
    """11-16: one discovery and seven enrichment timeouts in one 117 s run."""
    findings = evaluate_live(_payload(first=8, p95=117.0))
    assert [(f.check, f.severity, f.case_id) for f in findings] == [("live_tavily_errors", "P2", LIVE_CASE_ID)]
    assert "8 first-attempt" in findings[0].observed


def test_any_error_after_retry_fires():
    """A retry that also failed lost the page outright."""
    assert [f.check for f in evaluate_live(_payload(retry=1))] == ["live_tavily_errors"]


def test_live_latency_over_the_ceiling_fires():
    findings = evaluate_live(_payload(p95=thresholds.LATENCY_CEILING_S + 1))
    assert [(f.check, f.severity) for f in findings] == [("live_latency_high", "P3")]


def test_a_quiet_window_vouches_for_nothing():
    """No visitor searched: silence is not a recovery, so no live issue closes."""
    assert live_scope(_payload(runs=0)) == frozenset()
    assert live_scope(_payload(runs=1)) == thresholds.LIVE_CHECKS


def test_watcher_judges_the_live_file(tmp_path, capsys):
    live = tmp_path / "live.json"
    live.write_text(json.dumps(_payload(first=5, p95=160.0)), encoding="utf-8")
    code = watch.main(["--tier", "A", "--case", "chandler-neurology", "--input", str(tmp_path / "none.json"),
                       "--no-issues", "--live", str(live)], env={})
    out = capsys.readouterr().out
    assert code == 1
    assert "Tavily errors in the live app" in out and "Live app latency high" in out
    assert "live app (demo, last 26 h): 1 run(s), p95 160 s" in out


def test_watcher_expects_the_live_file_when_asked():
    findings, scope, note = watch.evaluate_live_file(Path("/nonexistent/live.json"))
    assert [(f.check, f.case_id) for f in findings] == [("canary_did_not_run", LIVE_CASE_ID)]
    assert scope == {LIVE_CASE_ID: {"canary_did_not_run"}} and note is None
    assert watch.evaluate_live_file(None) == ([], {}, None)


def test_workflow_runs_the_live_watch_only_when_opted_in():
    import yaml

    text = (REPO / ".github" / "workflows" / "canary-tier-a.yml").read_text(encoding="utf-8")
    job = yaml.safe_load(text)["jobs"]["fetch-canary"]
    plan = next(s for s in job["steps"] if s.get("id") == "plan")
    live = next(s for s in job["steps"] if "evals.live_watch" in str(s.get("run", "")))
    assert plan["env"]["LIVE_WATCH_ENVIRONMENTS"] == "${{ vars.LIVE_WATCH_ENVIRONMENTS }}"
    assert live["env"]["LIVE_WATCH_ENVIRONMENTS"] == "${{ vars.LIVE_WATCH_ENVIRONMENTS }}"
    assert "secrets.LIVE_WATCH_ENVIRONMENTS" not in text
    assert live["if"] == "always() && steps.plan.outputs.live == 'true'"
    assert "--live evals/out/live_health.json" in plan["run"]
    steps = [s.get("name") for s in job["steps"]]
    assert steps.index(live["name"]) < steps.index("Evaluate, summarize, sync issues")
