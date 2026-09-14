"""A fake Langfuse client that records the span tree, for tracing tests.

Every observation records its parent id, type, kwargs, updates, end state and
the thread that created it, so a test can assert the SHAPE the plan draws —
steps under the root, worker-thread spans under the open step, model calls
under their provider span — without a network.
"""

import sys
import threading
import types


class FakeSpan:
    def __init__(self, client, name, parent_id, as_type, **kw):
        self.client, self.name, self.parent_id, self.as_type = client, name, parent_id, as_type
        self.id = f"{name}#{len(client.spans)}"
        self.kw = dict(kw)
        self.updates = []
        self.ended = False
        self.thread = threading.current_thread().name
        self.trace_context = None
        client.spans.append(self)

    def start_observation(self, *, name, as_type="span", **kw):
        return FakeSpan(self.client, name, self.id, as_type, **kw)

    def update(self, **kw):
        self.updates.append(kw)
        return self

    def end(self, **kw):
        self.ended = True
        return self

    @property
    def merged(self):
        out = dict(self.kw)
        for update in self.updates:
            out.update({k: v for k, v in update.items() if v is not None})
        return out


class FakeLangfuse:
    instances = []

    def __init__(self, **kw):
        self.kw = kw
        self.spans = []
        self.scores = []
        self.flushes = 0
        FakeLangfuse.instances.append(self)

    @staticmethod
    def create_trace_id(*, seed=None):
        return f"trace-{seed}"

    def start_observation(self, *, name, as_type="span", trace_context=None, **kw):
        span = FakeSpan(self, name, None, as_type, **kw)
        span.trace_context = trace_context
        return span

    def create_score(self, **kw):
        self.scores.append(kw)

    def get_trace_url(self, *, trace_id=None):
        return f"https://lf.example/trace/{trace_id}"

    def flush(self):
        self.flushes += 1


class _NullContext:
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def install_fake_tracing(monkeypatch, env="ci"):
    """Tracing ON against the fake client; returns the propagated trace attrs."""
    from utils import tracing
    from utils.cost_tracker import get_cost_tracker

    module = types.ModuleType("langfuse")
    module.Langfuse = FakeLangfuse
    captured = {}

    def propagate_attributes(**kw):
        captured.update(kw)
        return _NullContext(**kw)

    module.propagate_attributes = propagate_attributes
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)
    monkeypatch.setenv("ENV", env)
    FakeLangfuse.instances.clear()
    tracing.reset_client()
    get_cost_tracker().reset()
    return captured


def by_name(client, name):
    return [s for s in client.spans if s.name == name]
