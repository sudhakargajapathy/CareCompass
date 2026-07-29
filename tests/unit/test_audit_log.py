"""Unit tests for utils/audit_log.py.

Audit logging is observability. The one thing it must never do is fail the
action it is observing — and it did: `os.makedirs(os.path.dirname(path))` sat
OUTSIDE the OSError handler, so AUDIT_LOG_PATH="audit.log" (a bare filename,
whose dirname is "") raised FileNotFoundError into the login path
(utils/auth.py) and the search path (app.py), including app.py's own
`workflow_failed` handler.
"""

import json

import pytest

import utils.audit_log as audit_log


@pytest.fixture
def audit_path(tmp_path, monkeypatch):
    """Point the audit logger at a path under tmp_path and return a setter."""

    def _use(path_value, cwd=None):
        monkeypatch.chdir(cwd or tmp_path)
        monkeypatch.setenv("AUDIT_LOG_PATH", str(path_value))
        return path_value

    return _use


def _read_events(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TestPathShapes:
    def test_bare_filename_does_not_raise(self, audit_path, tmp_path):
        audit_path("audit.log")
        audit_log.log_audit_event("login_failed", user="alice", success=False)
        events = _read_events(tmp_path / "audit.log")
        assert events[0]["event_type"] == "login_failed"
        assert events[0]["success"] is False

    def test_nested_directory_is_created(self, audit_path, tmp_path):
        target = audit_path(tmp_path / "logs" / "deep" / "audit.log")
        audit_log.log_audit_event("workflow_started", user="bob")
        assert _read_events(target)[0]["user"] == "bob"

    def test_unwritable_path_degrades_silently(self, audit_path, caplog):
        audit_path("/proc/version/nope/audit.log")
        # Must not raise — the caller is mid-login or mid-search
        audit_log.log_audit_event("login_success", user="carol")
        assert "Failed to write audit log" in caplog.text


class TestRecordShape:
    def test_defaults(self, audit_path, tmp_path):
        target = audit_path(tmp_path / "audit.log")
        audit_log.log_audit_event("search")
        record = _read_events(target)[0]
        assert record["user"] == "anonymous"
        assert record["success"] is True
        assert record["details"] == {}
        assert record["timestamp"].endswith("Z")

    def test_details_are_preserved(self, audit_path, tmp_path):
        target = audit_path(tmp_path / "audit.log")
        audit_log.log_audit_event("rate_limited", details={"wait": 30})
        assert _read_events(target)[0]["details"] == {"wait": 30}

    def test_appends_rather_than_truncates(self, audit_path, tmp_path):
        target = audit_path(tmp_path / "audit.log")
        audit_log.log_audit_event("first")
        audit_log.log_audit_event("second")
        assert [e["event_type"] for e in _read_events(target)] == ["first", "second"]
