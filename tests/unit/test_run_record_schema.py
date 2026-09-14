"""The run-record contract and the account-verification script (observability P0).

The failure this scaffolding answers: a vendor-side search degradation cut the
discovery pool by two thirds and emptied enrichment, and it went unnoticed for
~3 weeks while every number that would have caught it was computed on every
run and thrown away. The run record keeps those numbers — one aggregates-only
row per run, carried as scores and metadata on the run's trace and exported
weekly into the public repository — and these tests pin the ways that record
could quietly stop being trustworthy:

  * a field that would carry provider identity into rows that ship publicly;
  * a field classified as both a descriptor and a structure, or as neither
    while not being a measurement either (P1 maps each class to a different
    carrier, so an unclassified field would silently vanish from the trace);
  * the verification script needing a network to say "keys are missing", or
    sending traces to the SDK's default (EU) host because the base URL was
    allowed to default.
"""

import re
import sys
import types
from pathlib import Path

import pytest

from evals import p0_verify
from utils.run_record import (
    DESCRIPTOR_FIELDS,
    RUN_RECORD_FIELDS,
    RUN_SOURCES,
    SCHEMA_VERSION,
    STRUCTURED_FIELDS,
    measurement_fields,
)

REPO = Path(__file__).resolve().parents[2]


class TestRunRecordContract:
    def test_fields_are_unique(self):
        assert len(set(RUN_RECORD_FIELDS)) == len(RUN_RECORD_FIELDS)

    def test_every_classified_field_exists_and_classes_do_not_overlap(self):
        fields = set(RUN_RECORD_FIELDS)
        assert DESCRIPTOR_FIELDS <= fields
        assert STRUCTURED_FIELDS <= fields
        assert not (DESCRIPTOR_FIELDS & STRUCTURED_FIELDS)

    def test_measurements_are_the_scalar_remainder(self):
        measured = measurement_fields()
        assert measured, "no measurements would mean no scores on the trace"
        assert not (set(measured) & DESCRIPTOR_FIELDS)
        assert not (set(measured) & STRUCTURED_FIELDS)
        assert set(measured) | DESCRIPTOR_FIELDS | STRUCTURED_FIELDS == set(RUN_RECORD_FIELDS)
        for name in ("pool_raw", "n_enriched", "cost_usd", "fallback_fired"):
            assert name in measured

    def test_schema_version_and_verify_source_exist(self):
        assert SCHEMA_VERSION >= 1
        assert "verify" in RUN_SOURCES and "user" in RUN_SOURCES

    def test_no_field_carries_provider_identity(self):
        # Aggregates only. A name, address, URL or review text field would
        # turn the health record into a second provider store — and the
        # exported rows are committed to the PUBLIC repository.
        forbidden = ("name", "address", "phone", "url", "text", "summary", "review_s")
        offenders = [f for f in RUN_RECORD_FIELDS if any(t in f for t in forbidden)]
        assert offenders == []


_ALL_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")


class TestMissingConfigurationReport:
    def test_names_every_absent_variable_in_report_order(self):
        assert p0_verify.missing_env({}) == list(_ALL_VARS)

    def test_blank_counts_as_missing(self):
        env = {name: "x" for name in _ALL_VARS}
        env["LANGFUSE_SECRET_KEY"] = "   "
        assert p0_verify.missing_env(env) == ["LANGFUSE_SECRET_KEY"]

    def test_every_required_variable_has_a_where_to_find_it(self):
        assert tuple(p0_verify.REQUIRED_ENV) == _ALL_VARS
        assert all(p0_verify.REQUIRED_ENV[v].strip() for v in _ALL_VARS)

    def test_main_reports_exit_2_and_contacts_nothing(self, monkeypatch, capsys):
        # The report about absent keys must never itself need a network: a
        # developer with no account runs this and gets the list, not a
        # traceback from a client constructor.
        for name in _ALL_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(p0_verify, "load_dotenv", lambda *a, **k: False)

        def must_not_run(*_a, **_k):
            raise AssertionError("a client was contacted with configuration missing")

        monkeypatch.setattr(p0_verify, "verify_langfuse", must_not_run)

        assert p0_verify.main([]) == 2
        out = capsys.readouterr().out
        for name in _ALL_VARS:
            assert name in out
        assert "nothing was contacted" in out

    def test_env_example_documents_the_variables_and_only_those(self):
        text = (REPO / ".env.example").read_text(encoding="utf-8")
        for name in _ALL_VARS:
            assert re.search(rf"^#?\s*{name}=", text, re.M), name
        # The retired database URL must not creep back: a secret with no
        # consumer is pure downside, and the store it fed no longer exists.
        assert "RUN_RECORD_DB_URL" not in text
        # The trace environment label is derived from ENV in the app, never
        # a second variable that could disagree with it.
        assert "LANGFUSE_TRACING_ENVIRONMENT=" not in text


class _FakeSpan:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def start_observation(self, **kw):
        self.log.append(("child", kw))
        return _FakeSpan(self.log, kw["name"])

    def update(self, **kw):
        self.log.append(("update", self.name, kw))
        return self

    def end(self, **kw):
        self.log.append(("end", self.name))
        return self


class _FakeLangfuse:
    instances = []

    def __init__(self, **kw):
        self.kw, self.log, self.auth = kw, [], True
        _FakeLangfuse.instances.append(self)

    @staticmethod
    def create_trace_id(*, seed=None):
        return f"trace-for-{seed}"

    def auth_check(self):
        self.log.append(("auth_check",))
        return self.auth

    def start_observation(self, **kw):
        self.log.append(("root", kw))
        return _FakeSpan(self.log, kw["name"])

    def create_score(self, **kw):
        self.log.append(("score", kw))

    def flush(self):
        self.log.append(("flush",))

    def get_trace_url(self, *, trace_id=None):
        return f"https://langfuse.example/trace/{trace_id}"

    def shutdown(self):
        self.log.append(("shutdown",))


@pytest.fixture
def fake_langfuse(monkeypatch):
    module = types.ModuleType("langfuse")
    module.Langfuse = _FakeLangfuse
    _FakeLangfuse.instances.clear()
    monkeypatch.setitem(sys.modules, "langfuse", module)
    return _FakeLangfuse


_LF_ENV = {
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
    "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com",
}


class TestLangfuseHelloWorld:
    def test_trace_is_seeded_by_run_id_scored_flushed_and_shut_down(self, fake_langfuse):
        result = p0_verify.verify_langfuse(_LF_ENV, "run-123", "abc123")
        client = fake_langfuse.instances[0]

        # The base URL is passed explicitly — the SDK's own default is the EU
        # host, and a US project's traces would vanish from the dashboard.
        assert client.kw["base_url"] == "https://us.cloud.langfuse.com"
        assert client.kw["public_key"] == "pk-lf-test"
        assert client.kw["secret_key"] == "sk-lf-test"
        assert client.kw["environment"] == "verify"

        root = next(e for e in client.log if e[0] == "root")[1]
        assert root["trace_context"] == {"trace_id": "trace-for-run-123"}
        assert root["metadata"]["source"] == "verify"
        assert root["metadata"]["schema_version"] == SCHEMA_VERSION
        score = next(e for e in client.log if e[0] == "score")[1]
        assert score["trace_id"] == "trace-for-run-123"
        assert result["trace_id"] == "trace-for-run-123"
        assert result["trace_url"].endswith("trace-for-run-123")

        kinds = [e[0] for e in client.log]
        assert kinds.index("flush") > kinds.index("score")
        assert kinds[-1] == "shutdown"
        assert ("end", "carecompass.p0_verify") in client.log
        assert ("end", "hello_world") in client.log

    def test_a_pasted_trailing_space_never_reaches_the_host(self, fake_langfuse):
        # The public repo's LANGFUSE_BASE_URL variable arrived with a trailing
        # space; a host ending in " " fails every call. Stripped, like tracing.
        p0_verify.verify_langfuse({**_LF_ENV, "LANGFUSE_BASE_URL": _LF_ENV["LANGFUSE_BASE_URL"] + " "}, "run-9", None)
        assert fake_langfuse.instances[0].kw["base_url"] == "https://us.cloud.langfuse.com"

    def test_rejected_keys_fail_before_any_span_is_opened(self, fake_langfuse):
        class Rejecting(_FakeLangfuse):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.auth = False

        sys.modules["langfuse"].Langfuse = Rejecting
        with pytest.raises(RuntimeError, match="rejected the keys"):
            p0_verify.verify_langfuse(_LF_ENV, "run-1", None)
        client = _FakeLangfuse.instances[0]
        assert not any(e[0] == "root" for e in client.log)
        assert client.log[-1] == ("shutdown",)

    def test_sdk_auth_exception_becomes_one_line_with_its_body(self, fake_langfuse):
        # The real SDK raises on a 401 with a header dump attached; the
        # owner should read the API's one-sentence body, not the headers.
        class Raising(_FakeLangfuse):
            def auth_check(self):
                err = RuntimeError("headers: {...}, status_code: 401")
                err.body = {"message": "Invalid credentials."}
                raise err

        sys.modules["langfuse"].Langfuse = Raising
        with pytest.raises(RuntimeError, match=r"Invalid credentials") as excinfo:
            p0_verify.verify_langfuse(_LF_ENV, "run-1", None)
        assert "headers" not in str(excinfo.value)

    def test_main_prints_one_ok_line_with_the_trace_url(self, fake_langfuse, monkeypatch, capsys):
        for name, value in _LF_ENV.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(p0_verify, "load_dotenv", lambda *a, **k: False)
        assert p0_verify.main([]) == 0
        out = capsys.readouterr().out
        assert "OK   Langfuse" in out and "https://langfuse.example/trace/" in out
        assert "sk-lf-test" not in out


class TestSecretsNeverEcho:
    def test_redact_scrubs_both_keys(self):
        cleaned = p0_verify._redact("auth failed for sk-lf-test (pk-lf-test)", _LF_ENV)
        assert "sk-lf-test" not in cleaned and "pk-lf-test" not in cleaned
        assert "***" in cleaned
