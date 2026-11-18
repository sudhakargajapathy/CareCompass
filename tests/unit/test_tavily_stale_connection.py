"""Stale pooled Tavily connections: the 100+ s live searches.

The app keeps ONE orchestrator per process (`@st.cache_resource`), so one
`TavilyClient` and its `requests.Session` pool live across every search.
`requests` never expires an idle keep-alive socket. After the live Space had
sat idle for about a day, the far end had dropped them, and the next search
wrote into a dead socket: 30 of 34 first attempts hung to the 30 s timeout
while every retry on a new connection succeeded in 1-3 s (Langfuse, demo
environment, 2026-11-03..16). CI never sees it because each canary starts a
fresh process.

The fake server below reproduces that with a REAL `TavilyClient` over a REAL
socket. It answers the FIRST request on each connection, then keeps the
socket open and never answers again, which is what a silently dropped
keep-alive looks like from the client side. A new connection is answered
normally.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest
from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeout

from agents.data_gatherer import DataGathererAgent

_TIMEOUT_S = 1.5


class _OneAnswerPerConnection(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so the client pools the socket

    def handle(self):
        self.handle_one_request()  # serve the first request normally
        # Then go silent on this socket without closing it. The client
        # pooled it as healthy, so its next request on it can only time out.
        self.server.stop.wait()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.server.connections += 1
        body = json.dumps({"results": [{"url": "https://example.test/p",
                                        "raw_content": "ok"}],
                           "failed_results": []}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture
def stale_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OneAnswerPerConnection)
    server.daemon_threads = True
    server.stop = threading.Event()
    server.connections = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.stop.set()
    server.shutdown()
    server.server_close()


@pytest.fixture
def gatherer(stale_server):
    with patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
        agent = DataGathererAgent()
    host, port = stale_server.server_address
    agent.tavily_client = TavilyClient(api_key="tvly-test",
                                       api_base_url=f"http://{host}:{port}")
    return agent


def _extract(agent):
    started = time.monotonic()
    agent.tavily_client.extract(urls=["https://example.test/p"], timeout=_TIMEOUT_S)
    return time.monotonic() - started


def test_reproduces_the_hang_without_a_reset(gatherer):
    """The bug itself: a second search on the pooled socket times out."""
    _extract(gatherer)  # search 1 leaves its connection in the pool
    started = time.monotonic()
    with pytest.raises(TavilyTimeout):
        _extract(gatherer)  # search 2 reuses it and hangs to the timeout
    assert time.monotonic() - started >= _TIMEOUT_S * 0.9


def test_reset_makes_the_next_search_open_a_fresh_connection(gatherer, stale_server):
    _extract(gatherer)
    gatherer._refresh_tavily_connections()
    elapsed = _extract(gatherer)  # must not raise
    assert elapsed < _TIMEOUT_S / 2
    assert stale_server.connections == 2  # the second call really was a new socket


def test_gather_providers_resets_the_pool_at_the_start_of_every_search(gatherer, stale_server):
    """Wired into the search entry point, so back-to-back searches stay fast."""
    _extract(gatherer)
    # Input validation fails before any fetch, so this exercises only the
    # reset at the top of the method.
    result = gatherer.gather_providers(specialty="", location="")
    assert result["status"] != "success"
    assert _extract(gatherer) < _TIMEOUT_S / 2
    assert stale_server.connections == 2


def test_reset_never_fails_a_search():
    with patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
        agent = DataGathererAgent()

    agent.tavily_client = None  # clients failed to initialise
    agent._refresh_tavily_connections()

    agent.tavily_client = MagicMock()
    agent.tavily_client.session.close.side_effect = OSError("boom")
    agent._refresh_tavily_connections()  # swallowed, logged at debug
    agent.tavily_client.session.close.assert_called_once()

    agent.tavily_client = object()  # no session attribute at all
    agent._refresh_tavily_connections()


# --- Failing fast once a stall gets through ----------------------------------
#
# The pool reset above removes the cause. These guard the cost of any stall
# that still gets through (a socket that dies mid-run, a network blip): on the
# live Space each stalled first attempt ran the vendor's full 30 s timeout and
# then slept 2 s before a retry that succeeded in 1-3 s on a new socket.

import requests  # noqa: E402
from tavily.errors import UsageLimitExceededError  # noqa: E402

from agents import data_gatherer as dg  # noqa: E402


@pytest.mark.parametrize("exc", [
    TavilyTimeout(30),
    requests.exceptions.SSLError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred"),
    requests.exceptions.ConnectionError("Connection reset by peer"),
    requests.exceptions.ChunkedEncodingError("Connection broken"),
])
def test_connection_failures_retry_without_the_pause(exc):
    """Exactly the two signatures the live traces showed (30 s timeouts and
    fast SSL EOFs) plus their siblings: a new connection fixes them, so the
    2 s pause before the retry is pure latency."""
    assert dg._is_connection_failure(exc)
    assert dg._tavily_retry_pause(exc) == 0.0


@pytest.mark.parametrize("exc", [
    UsageLimitExceededError("rate limited"),
    requests.exceptions.HTTPError("502 Bad Gateway"),
    ValueError("unexpected payload"),
])
def test_vendor_side_failures_keep_the_pause(exc):
    """A 429 or 5xx is the vendor talking; retrying it instantly is the
    one case the pause exists for."""
    assert not dg._is_connection_failure(exc)
    assert dg._tavily_retry_pause(exc) == dg._TAVILY_RETRY_PAUSE_S > 0


def test_a_stalled_first_extract_costs_the_short_timeout_and_no_pause(
        gatherer, stale_server, monkeypatch):
    """End to end over a real socket: the second batch reuses the dead pooled
    connection, times out on the FIRST-ATTEMPT timeout (not the vendor's 30 s),
    and retries at once on a fresh connection, which succeeds. Before the fix
    this call ran 30 s and slept 2 s."""
    monkeypatch.setattr(dg.DataGathererAgent, "_EXTRACT_FIRST_ATTEMPT_TIMEOUT_S", 1.0)
    sleeps = []
    monkeypatch.setattr(dg.time, "sleep", sleeps.append)

    assert len(gatherer._extract_pages(["https://example.test/a"])) == 1
    started = time.monotonic()
    pages = gatherer._extract_pages(["https://example.test/b"])
    elapsed = time.monotonic() - started

    assert len(pages) == 1               # the retry recovered the page
    assert stale_server.connections == 2  # on a new socket
    assert sleeps == []                  # with no pause in between
    assert 0.9 <= elapsed < 2.5          # bounded by the short first timeout


def test_extract_attempts_use_the_short_then_the_full_timeout():
    """The first attempt carries the short timeout and the retry the vendor's
    full 30 s, so a slow-but-healthy batch still has the old budget."""
    with patch.object(dg.DataGathererAgent, "_initialize_clients", return_value=None):
        agent = dg.DataGathererAgent()
    agent.tavily_client = MagicMock()
    agent.tavily_client.extract.side_effect = [
        TavilyTimeout(15),
        {"results": [{"url": "https://example.test/p", "raw_content": "ok"}],
         "failed_results": []},
    ]
    with patch.object(dg.time, "sleep") as sleep:
        pages = agent._extract_pages(["https://example.test/p"])
    assert len(pages) == 1
    timeouts = [c.kwargs["timeout"] for c in agent.tavily_client.extract.call_args_list]
    assert timeouts == [dg.DataGathererAgent._EXTRACT_FIRST_ATTEMPT_TIMEOUT_S,
                        dg.DataGathererAgent._EXTRACT_RETRY_TIMEOUT_S]
    assert timeouts[0] < timeouts[1] == 30.0
    sleep.assert_not_called()


def _search_agent(side_effect):
    with patch.object(dg.DataGathererAgent, "_initialize_clients", return_value=None):
        agent = dg.DataGathererAgent()
    agent.tavily_client = MagicMock()
    agent.tavily_client.search.side_effect = side_effect
    return agent


def test_search_retries_a_dead_connection_at_once():
    agent = _search_agent([
        requests.exceptions.SSLError("UNEXPECTED_EOF_WHILE_READING"),
        {"results": [{"url": "https://example.test/p", "raw_content": "ok"}]},
    ])
    with patch.object(dg.time, "sleep") as sleep:
        results = agent._search_providers("neurology chandler az")
    assert len(results) == 1
    sleep.assert_not_called()


def test_search_still_pauses_before_retrying_a_vendor_error():
    agent = _search_agent([
        UsageLimitExceededError("rate limited"),
        {"results": [{"url": "https://example.test/p", "raw_content": "ok"}]},
    ])
    with patch.object(dg.time, "sleep") as sleep:
        results = agent._search_providers("neurology chandler az")
    assert len(results) == 1
    sleep.assert_called_once_with(dg._TAVILY_RETRY_PAUSE_S)
