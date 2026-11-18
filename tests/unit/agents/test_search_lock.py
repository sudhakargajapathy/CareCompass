"""One search at a time per process.

The app caches one orchestrator for the life of the process, so every
visitor shares it, and the cost tracker, the active trace and
`self.progress_callback` are process-wide. Two overlapping searches used to
cross: the second visitor's start reset the first one's cost card mid-run,
the first visitor's progress panel began showing the second's steps, and the
first run's `finally` then cleared the callback so the second's panel froze.
These drive two real overlapping runs through the real graph (agents mocked)
and pin that the second waits, says so, and that each run's numbers and
progress stay its own.
"""
import threading
import time
from unittest.mock import patch

import pytest

from agents import orchestrator as orch_module
from agents.orchestrator import ProviderMatchingOrchestrator
from tests.fixtures.mock_agent_responses import (
    MOCK_GATHER_PROVIDERS_RESULT,
    MOCK_SCORED_PROVIDERS_RESULT,
    MOCK_VALIDATION_RESULT,
)
from utils.cost_tracker import get_cost_tracker


@pytest.fixture
def orchestrator():
    with patch("agents.orchestrator.DataGathererAgent"), \
         patch("agents.orchestrator.PreferenceScorerAgent"), \
         patch("agents.orchestrator.CriticValidatorAgent"), \
         patch("agents.orchestrator.get_vector_store"):
        orch = ProviderMatchingOrchestrator()
        orch.preference_scorer.score_providers.return_value = MOCK_SCORED_PROVIDERS_RESULT
        orch.critic_validator.validate_rankings.return_value = MOCK_VALIDATION_RESULT
        yield orch
    assert not orch_module._SEARCH_LOCK.locked(), "a test leaked the search lock"


def _blocking_gatherer(orch, release_first: threading.Event, first_started: threading.Event):
    """Run 1's gather blocks until released; every gather records ONE Tavily
    credit, so a run's cost card must read exactly 1."""
    calls = {"n": 0}

    def gather(*args, **kwargs):
        calls["n"] += 1
        get_cost_tracker().record_tavily(depth="basic", agent="data_gatherer")
        if calls["n"] == 1:
            first_started.set()
            assert release_first.wait(10), "first search never released"
        return MOCK_GATHER_PROVIDERS_RESULT

    orch.data_gatherer.gather_providers.side_effect = gather
    return calls


def _run_streaming(orch, sink, out, key):
    out[key] = orch.execute_workflow_streaming(
        specialty="Neurology", location="Chandler, AZ", progress_callback=sink.append,
    )


def test_overlapping_searches_run_one_after_the_other(orchestrator):
    release, started = threading.Event(), threading.Event()
    calls = _blocking_gatherer(orchestrator, release, started)
    first_updates, second_updates, out = [], [], {}

    a = threading.Thread(target=_run_streaming, args=(orchestrator, first_updates, out, "a"))
    a.start()
    assert started.wait(5)
    b = threading.Thread(target=_run_streaming, args=(orchestrator, second_updates, out, "b"))
    b.start()

    # The second visitor is told why nothing is happening, and nothing runs.
    deadline = time.monotonic() + 5
    while not second_updates and time.monotonic() < deadline:
        time.sleep(0.01)
    assert [u.get("status") for u in second_updates] == ["waiting"]
    assert "Another search is finishing" in second_updates[0]["action"]
    assert calls["n"] == 1

    release.set()
    a.join(10)
    b.join(10)
    assert not a.is_alive() and not b.is_alive()
    assert calls["n"] == 2
    assert out["a"]["success"] and out["b"]["success"]

    # Each run's cost card holds its own spend only: one credit each. Without
    # the lock the second start reset the tracker under the first run.
    assert out["a"]["cost_summary"]["tavily"]["credits"] == 1
    assert out["b"]["cost_summary"]["tavily"]["credits"] == 1


def test_each_progress_panel_gets_only_its_own_steps(orchestrator):
    """Without the lock the second search replaced `self.progress_callback`
    while the first was running: the first panel lost the rest of its steps
    to the second visitor's panel."""
    release, started = threading.Event(), threading.Event()
    _blocking_gatherer(orchestrator, release, started)
    first_updates, second_updates, out = [], [], {}

    a = threading.Thread(target=_run_streaming, args=(orchestrator, first_updates, out, "a"))
    a.start()
    assert started.wait(5)
    seen_before_overlap = len(first_updates)
    b = threading.Thread(target=_run_streaming, args=(orchestrator, second_updates, out, "b"))
    b.start()
    time.sleep(0.1)
    release.set()
    a.join(10)
    b.join(10)

    steps = lambda updates: [u for u in updates if u.get("status") != "waiting"]
    # Both panels saw a whole run, and the same number of steps.
    assert len(first_updates) > seen_before_overlap
    assert len(steps(first_updates)) == len(steps(second_updates)) > 0
    assert first_updates[-1].get("progress_percentage") == second_updates[-1].get("progress_percentage")
    # The second panel's run steps all arrived after its waiting line.
    assert second_updates[0].get("status") == "waiting"
    assert orchestrator.progress_callback is None


def test_a_wait_past_the_limit_returns_busy_without_running(orchestrator, monkeypatch):
    monkeypatch.setattr(orch_module, "SEARCH_WAIT_S", 0.2)
    orchestrator.data_gatherer.gather_providers.return_value = MOCK_GATHER_PROVIDERS_RESULT
    updates = []
    assert orch_module._SEARCH_LOCK.acquire(timeout=1)  # another search holds the process
    try:
        busy = orchestrator.execute_workflow_streaming(
            specialty="Neurology", location="Chandler, AZ", progress_callback=updates.append,
        )
        plain = orchestrator.execute_workflow(specialty="Neurology", location="Chandler, AZ")
    finally:
        orch_module._SEARCH_LOCK.release()
    for result in (busy, plain):
        assert result["success"] is False
        assert result["workflow_summary"]["busy"] is True
        assert result["error_messages"] == [orch_module.BUSY_MESSAGE]
    orchestrator.data_gatherer.gather_providers.assert_not_called()
    assert [u.get("status") for u in updates] == ["waiting"]


def test_a_crashed_search_releases_the_lock(orchestrator):
    """The workflow's own handlers catch graph errors, so the escape that
    matters is one OUTSIDE them (here, opening the trace). Without the
    `finally`, every later search on the process would wait out the limit."""
    with patch.object(orch_module.tracing, "start_run", side_effect=RuntimeError("trace init died")):
        with pytest.raises(RuntimeError):
            orchestrator.execute_workflow_streaming(
                specialty="Neurology", location="Chandler, AZ", progress_callback=lambda u: None,
            )
        assert not orch_module._SEARCH_LOCK.locked()
        with pytest.raises(RuntimeError):
            orchestrator.execute_workflow(specialty="Neurology", location="Chandler, AZ")
        assert not orch_module._SEARCH_LOCK.locked()
    assert orchestrator.progress_callback is None


def test_a_single_search_never_waits(orchestrator):
    orchestrator.data_gatherer.gather_providers.return_value = MOCK_GATHER_PROVIDERS_RESULT
    updates = []
    result = orchestrator.execute_workflow_streaming(
        specialty="Neurology", location="Chandler, AZ", progress_callback=updates.append,
    )
    assert result["success"]
    assert all(u.get("status") != "waiting" for u in updates)


def test_the_app_shows_the_busy_message_not_a_failure():
    """'Check your input' would send the visitor hunting for a mistake they
    did not make: the busy result renders its own warning."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[3] / "app.py").read_text(encoding="utf-8")
    assert "from agents.orchestrator import BUSY_MESSAGE" in source
    busy_branch = source.index('.get("busy"):\n            # Not a failure of this search')
    generic = source.index('st.error("Search failed. Please check your input and try again.")')
    assert busy_branch < generic
    assert "st.warning(BUSY_MESSAGE)" in source[busy_branch:generic]
