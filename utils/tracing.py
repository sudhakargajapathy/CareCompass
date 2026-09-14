"""Langfuse tracing: one run is one trace, one seam per call type, no-op without keys.

Why this exists. Tavily's August 2026 search overhaul degraded the pipeline
for about three weeks before anyone noticed, because nothing kept the
per-run numbers that would have shown it in a day. Every search now produces
one trace: a root observation seeded by the run id, one child per workflow
step, a generation per model call carrying usage AND cost, a tool span per
Tavily call carrying page counts and content hashes, and — at the end — the
run record (`utils.run_record`) attached as scores and metadata, so the
health of a run is a queryable row the day it happens.

The four rules this module enforces, each with the failure it prevents:

  * NO-OP WITHOUT KEYS. The unit suite is offline, local development needs
    no account, and the Space must run identically with tracing off. When
    the keys are absent no client is ever constructed — the SDK would
    otherwise log a warning per construction and try to export — and every
    helper degrades to a handle that records nothing. `LANGFUSE_BASE_URL`
    is REQUIRED alongside the keys: the SDK's default host is its EU cloud,
    and traces sent there would be invisible on the US dashboard the owner
    opens. Keys without a base URL disable tracing with a warning.

  * ONE SEAM, TWO SINKS. `generation()` is the only place a model call's
    usage is turned into cost, and it writes that cost to BOTH the in-app
    cost tracker and the span (`cost_details`). Our model ids are not in
    Langfuse's price table, so the trace would otherwise show $0 while the
    cost card showed $0.55 — two numbers for one call is how the two
    surfaces start disagreeing.

  * BODIES STAY OUT. Fetched pages are 30-60 KB each and 20+ per run; a
    trace carrying them would weigh ~1 MB and burn the free tier. Tool spans
    carry each page's character count and a sha256 (enough to prove two
    runs fetched the same bytes), never the text; prompts and outputs are
    recorded as a bounded preview (`PREVIEW_CHARS`) plus length and hash;
    and the SDK's mask hook caps every string the same way as a last line
    of defence, so nothing added later can leak a body by accident.

  * PARENTS ARE FOUND, NOT PASSED. Enrichment researches eight providers on
    worker threads, and the judge and critic shard across threads too; the
    OpenTelemetry context the SDK relies on does not follow a
    ThreadPoolExecutor, so naive child creation on a worker yields an
    orphan trace per call. The active workflow STEP is kept in a
    process-wide stack (one search runs at a time per process — the cost
    tracker already assumes exactly that), and a thread-local override lets
    a worker's own span (one per enriched provider) become the parent of
    everything that thread does inside it. Call sites never name a parent.

The trace ENVIRONMENT label comes from the app's own `ENV` knob
(`development` / `ci` / `production`), validated against the SDK's naming
rule and falling back to `development` — one variable names the tier for
the auth check, the traces and the run records, so the two can never drift.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from utils.config import get_config
from utils.cost_tracker import get_cost_tracker, llm_cost_usd, safe_usage

logger = logging.getLogger(__name__)

PREVIEW_CHARS = 4000
DEFAULT_ENVIRONMENT = "development"
ROOT_NAME = "carecompass.search"
# The SDK's own rule: lowercase alphanumerics, hyphens, underscores, and not
# starting with "langfuse". A label the SDK rejects would disable the
# environment dimension silently, so the fallback is explicit here.
_ENV_LABEL_RE = re.compile(r"^(?!langfuse)[a-z0-9_-]{1,40}$")


def environment_label(raw: Optional[str]) -> str:
    """The trace environment derived from ENV; `development` when unusable."""
    value = (raw or "").strip().lower()
    return value if _ENV_LABEL_RE.match(value) else DEFAULT_ENVIRONMENT


def git_sha() -> Optional[str]:
    """Short commit hash of the running checkout, or None when unknowable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    sha = os.environ.get("GITHUB_SHA") or os.environ.get("GIT_SHA") or ""
    return sha[:12] or None


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()


def preview(text: Optional[str], limit: int = PREVIEW_CHARS) -> Dict[str, Any]:
    """A bounded stand-in for a long string: length, hash, and a capped head."""
    text = text or ""
    head = text if len(text) <= limit else text[:limit] + " …[truncated]"
    return {"chars": len(text), "sha256": sha256_text(text), "preview": head}


def _cap(value: Any, limit: int = PREVIEW_CHARS) -> Any:
    """Recursively cap strings — the SDK mask hook and the last line of defence."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + " …[truncated]"
    if isinstance(value, dict):
        return {k: _cap(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cap(v, limit) for v in value]
    return value


def summarize(value: Any, depth: int = 0) -> Any:
    """A step's details, bounded: scalars kept, collections reduced to sizes.

    The finalize step's details are the whole workflow summary (provider
    rows, coverage tables, the cost card) — tens of KB that would be
    duplicated on every trace for nothing; the record carries the numbers.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _cap(value, 500)
    if isinstance(value, dict):
        if depth >= 2 or len(value) > 24:
            return {"keys": len(value)}
        return {str(k): summarize(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) <= 8 and all(isinstance(v, (str, int, float, bool)) or v is None for v in value):
            return [_cap(v, 200) for v in value]
        return {"items": len(value)}
    return str(type(value).__name__)


def mask(data: Any = None, **_: Any) -> Any:
    """SDK mask hook: called with the payload about to be exported."""
    return _cap(data)


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

_client: Any = None
_client_built = False
_client_lock = threading.Lock()


def _tracing_switched_off() -> bool:
    return os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().lower() in ("false", "0", "no", "off")


def _build_client() -> Any:
    config = get_config()
    if _tracing_switched_off():
        logger.info("Tracing off: LANGFUSE_TRACING_ENABLED is false")
        return None
    public_key = (config.LANGFUSE_PUBLIC_KEY or "").strip()
    secret_key = (config.LANGFUSE_SECRET_KEY or "").strip()
    base_url = (config.LANGFUSE_BASE_URL or "").strip()
    if not public_key or not secret_key:
        logger.info("Tracing off: no Langfuse keys configured")
        return None
    if not base_url:
        logger.warning(
            "Tracing off: LANGFUSE_PUBLIC_KEY/SECRET_KEY are set but LANGFUSE_BASE_URL "
            "is not — the SDK would default to its EU host and every trace would be "
            "invisible on the project's dashboard. Set the region host explicitly."
        )
        return None
    from langfuse import Langfuse  # lazy: never imported on the no-op path

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=environment_label(config.ENV),
        release=git_sha(),
        mask=mask,
        tracing_enabled=True,
    )
    atexit.register(flush)
    logger.info("Tracing on: Langfuse at %s, environment %s", base_url, environment_label(config.ENV))
    return client


def get_client() -> Any:
    """The process-wide Langfuse client, built once; None when tracing is off."""
    global _client, _client_built
    with _client_lock:
        if not _client_built:
            try:
                _client = _build_client()
            except Exception as exc:  # tracing must never fail the app
                logger.warning("Tracing off: Langfuse client could not be built (%s)", exc)
                _client = None
            _client_built = True
        return _client


def reset_client() -> None:
    """Forget the built client (tests, or after a key change)."""
    global _client, _client_built
    with _client_lock:
        _client = None
        _client_built = False


def flush() -> None:
    client = _client
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)


# --------------------------------------------------------------------------
# The active run, its step stack, and the thread-local parent override
# --------------------------------------------------------------------------

_active_run: Optional["RunTrace"] = None
_active_lock = threading.Lock()
_local = threading.local()


def active_run() -> Optional["RunTrace"]:
    return _active_run


def _thread_parent() -> Any:
    return getattr(_local, "parent", None)


class _NoopHandle:
    """What every helper yields when tracing is off: accepts everything, does nothing."""

    span = None
    trace_id = None

    def finish(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def update(self, **kwargs: Any) -> None:
        return None


class _SpanHandle:
    def __init__(self, span: Any):
        self.span = span

    def update(self, **kwargs: Any) -> None:
        try:
            self.span.update(**kwargs)
        except Exception as exc:
            logger.debug("span update failed: %s", exc)

    def finish(
        self,
        output: Any = None,
        metadata: Any = None,
        level: Optional[str] = None,
        cost_usd: Optional[float] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if metadata is not None:
            kwargs["metadata"] = metadata
        if level:
            kwargs["level"] = level
        if cost_usd is not None:
            # Tavily credits are money too. The first live trace summed model
            # spans only ($0.354) beside a cost card saying $0.442, because the
            # tool spans carried credits but no cost. Writing `cost_details`
            # here did NOT close that gap: a 2026-09-13 probe trace with one
            # observation of each type showed Langfuse keeps cost on
            # generation-like observations only (embedding + generation summed
            # to 0.027; the tool and span costs were dropped), so the trace
            # header's total is MODEL spend by construction. The dollar figure
            # therefore rides where the server keeps it — on this span's
            # metadata, readable per call — and the run record's `cost_usd`
            # score carries the card total for the charts. `cost_details`
            # stays: harmless, and self-correcting if the server ever honours
            # it on tool spans.
            cost = round(float(cost_usd), 6)
            kwargs["cost_details"] = {"total": cost}
            merged = dict(metadata) if isinstance(metadata, dict) else {}
            merged["cost_usd"] = cost
            kwargs["metadata"] = merged
        if kwargs:
            self.update(**kwargs)


class GenerationHandle(_SpanHandle):
    """A model call in flight: `finish(response)` is the one seam, two sinks."""

    def __init__(self, span: Any, model: str, agent: str):
        super().__init__(span)
        self.model, self.agent = model, agent
        self.started = time.perf_counter()
        self.input_tokens = self.output_tokens = 0
        self.cost_usd = 0.0

    def finish(  # type: ignore[override]
        self,
        response: Any = None,
        output_text: Optional[str] = None,
        stop_reason: Optional[str] = None,
        record_cost: bool = True,
    ) -> Tuple[int, int]:
        in_tokens, out_tokens = safe_usage(response) if response is not None else (0, 0)
        duration = time.perf_counter() - self.started
        self.input_tokens, self.output_tokens = in_tokens, out_tokens
        self.cost_usd = llm_cost_usd(self.model, in_tokens, out_tokens)
        if record_cost:
            get_cost_tracker().record_llm(
                self.model, in_tokens, out_tokens, agent=self.agent, duration_s=duration
            )
        if self.span is not None:
            kwargs: Dict[str, Any] = {
                "usage_details": {"input": in_tokens, "output": out_tokens, "total": in_tokens + out_tokens},
                "cost_details": {"total": round(self.cost_usd, 6)},
            }
            if output_text is not None:
                kwargs["output"] = preview(output_text)
            if stop_reason:
                kwargs["metadata"] = {"stop_reason": stop_reason}
            self.update(**kwargs)
        return in_tokens, out_tokens


class _NoopGeneration(GenerationHandle):
    def __init__(self, model: str, agent: str):
        super().__init__(None, model, agent)


class RunTrace:
    """One workflow run's trace: root span, step stack, and the record at the end."""

    def __init__(self, client: Any, run_id: str, root: Any, trace_id: str):
        self._client = client
        self.run_id = run_id
        self.root = root
        self.trace_id = trace_id
        self._steps: List[Tuple[str, Any]] = []
        self._lock = threading.Lock()
        self.finished = False

    # ---- parents -----------------------------------------------------------
    def current_parent(self) -> Any:
        parent = _thread_parent()
        if parent is not None:
            return parent
        with self._lock:
            return self._steps[-1][1] if self._steps else self.root

    # ---- steps (the orchestrator's _log_step seam) ------------------------
    def step_started(self, name: str, details: Optional[Dict[str, Any]] = None) -> None:
        try:
            parent = self.current_parent()
            span = parent.start_observation(name=f"step.{name}", as_type="span", input=summarize(details or {}))
            with self._lock:
                self._steps.append((name, span))
        except Exception as exc:
            logger.debug("step span open failed for %s: %s", name, exc)

    def step_finished(self, name: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """End the step's span, unwinding any nested step that never reported."""
        with self._lock:
            index = next((i for i in range(len(self._steps) - 1, -1, -1) if self._steps[i][0] == name), None)
            if index is None:
                return
            closing = self._steps[index:]
            del self._steps[index:]
        for _, span in reversed(closing):
            try:
                span.update(
                    output=summarize(details or {}),
                    level="ERROR" if status == "failed" else "DEFAULT",
                    status_message=(details or {}).get("error") if status == "failed" else None,
                )
                span.end()
            except Exception as exc:
                logger.debug("step span close failed for %s: %s", name, exc)

    # ---- the record --------------------------------------------------------
    def finish(self, record: Optional[Dict[str, Any]] = None, output: Any = None, error: Optional[str] = None) -> None:
        """Attach the run record (scores + metadata), end the root, flush."""
        from utils.run_record import DESCRIPTOR_FIELDS, STRUCTURED_FIELDS, measurement_fields

        if self.finished:
            return
        self.finished = True
        try:
            # Unwind any step still open (a node that raised past its own log).
            with self._lock:
                leftovers = list(self._steps)
                self._steps.clear()
            for _, span in reversed(leftovers):
                try:
                    span.end()
                except Exception:
                    pass
            record = record or {}
            for field in measurement_fields():
                value = record.get(field)
                if value is None:
                    continue
                if isinstance(value, bool):
                    self._client.create_score(
                        name=field, value=1 if value else 0, trace_id=self.trace_id, data_type="BOOLEAN"
                    )
                elif isinstance(value, (int, float)):
                    self._client.create_score(
                        name=field, value=float(value), trace_id=self.trace_id, data_type="NUMERIC"
                    )
            descriptors = {k: record.get(k) for k in DESCRIPTOR_FIELDS if record.get(k) is not None}
            structured = {k: record.get(k) for k in STRUCTURED_FIELDS if record.get(k) is not None}
            self.root.update(
                output=_cap(output) if output is not None else None,
                metadata={"run_record": {**descriptors, **structured}},
                level="ERROR" if error else "DEFAULT",
                status_message=error,
            )
        except Exception as exc:
            logger.warning("Attaching the run record to the trace failed: %s", exc)
        finally:
            try:
                self.root.end()
            except Exception:
                pass
            flush()


# --------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------

@contextmanager
def start_run(
    run_id: str,
    input: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Iterator[Optional[RunTrace]]:
    """Open the run's root trace (seeded by run_id); yields None when tracing is off.

    The caller finishes the run explicitly with `run.finish(record=...)`;
    this context manager only guarantees the root is ended and the active
    run cleared if the caller never gets that far.
    """
    global _active_run
    client = get_client()
    if client is None:
        yield None
        return
    run: Optional[RunTrace] = None
    try:
        from langfuse import Langfuse, propagate_attributes

        trace_id = Langfuse.create_trace_id(seed=run_id)
        trace_metadata = {k: str(v) for k, v in (metadata or {}).items() if v is not None}
        with propagate_attributes(trace_name=ROOT_NAME, tags=list(tags or []), metadata=trace_metadata):
            root = client.start_observation(
                trace_context={"trace_id": trace_id},
                name=ROOT_NAME,
                as_type="agent",
                input=_cap(input or {}),
                metadata=trace_metadata,
            )
        run = RunTrace(client, run_id, root, trace_id)
    except Exception as exc:
        logger.warning("Tracing: could not open the run trace (%s); continuing untraced", exc)
        yield None
        return
    with _active_lock:
        _active_run = run
    clean_exit = False
    try:
        yield run
        clean_exit = True
    finally:
        with _active_lock:
            if _active_run is run:
                _active_run = None
        if not run.finished:
            # The caller never reached finish(): an exception escaped, or it
            # simply forgot. Either way the root must not stay open — an
            # unended root is a trace that never exports.
            run.finished = True
            try:
                if not clean_exit:
                    run.root.update(level="ERROR", status_message="run aborted")
                run.root.end()
            except Exception:
                pass
            flush()


def trace_url(run: Optional[RunTrace]) -> Optional[str]:
    if run is None:
        return None
    try:
        return get_client().get_trace_url(trace_id=run.trace_id)
    except Exception:
        return None


@contextmanager
def generation(
    name: str,
    model: str,
    agent: str,
    prompt: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Iterator[GenerationHandle]:
    """A model call: yields a handle whose `finish(response, ...)` records usage and cost.

    Works with tracing off (the handle still records to the cost tracker), so
    a call site is written once and behaves the same in the suite, locally,
    and on the Space.
    """
    run = _active_run
    if run is None:
        yield _NoopGeneration(model, agent)
        return
    span = None
    try:
        parent = run.current_parent()
        span = parent.start_observation(
            name=name,
            as_type="generation",
            model=model,
            model_parameters={k: v for k, v in (params or {}).items() if isinstance(v, (str, int, float, bool))},
            input=preview(prompt) if prompt is not None else None,
            metadata={"agent": agent},
        )
    except Exception as exc:
        logger.debug("generation span open failed for %s: %s", name, exc)
    handle = GenerationHandle(span, model, agent)
    try:
        yield handle
    except Exception as exc:
        if span is not None:
            try:
                span.update(level="ERROR", status_message=str(exc)[:500])
            except Exception:
                pass
        raise
    finally:
        if span is not None:
            try:
                span.end()
            except Exception:
                pass


@contextmanager
def tool(name: str, input: Optional[Dict[str, Any]] = None) -> Iterator[_SpanHandle]:
    """A Tavily (or other external) call: page counts and hashes, never bodies."""
    run = _active_run
    if run is None:
        yield _NoopHandle()
        return
    span = None
    try:
        span = run.current_parent().start_observation(name=name, as_type="tool", input=_cap(input or {}))
    except Exception as exc:
        logger.debug("tool span open failed for %s: %s", name, exc)
    handle: _SpanHandle = _SpanHandle(span) if span is not None else _NoopHandle()
    try:
        yield handle
    except Exception as exc:
        if span is not None:
            try:
                span.update(level="ERROR", status_message=str(exc)[:500])
            except Exception:
                pass
        raise
    finally:
        if span is not None:
            try:
                span.end()
            except Exception:
                pass


@contextmanager
def span(name: str, input: Optional[Dict[str, Any]] = None) -> Iterator[_SpanHandle]:
    """A unit of work that OWNS its thread while open (one enriched provider).

    Everything the same thread traces inside this block nests under it — that
    is the thread-local override, and it is what keeps eight concurrent
    enrichment workers from all attaching to the step span as siblings.
    """
    run = _active_run
    if run is None:
        yield _NoopHandle()
        return
    created = None
    previous = _thread_parent()
    try:
        created = run.current_parent().start_observation(name=name, as_type="span", input=_cap(input or {}))
        _local.parent = created
    except Exception as exc:
        logger.debug("span open failed for %s: %s", name, exc)
    handle: _SpanHandle = _SpanHandle(created) if created is not None else _NoopHandle()
    try:
        yield handle
    except Exception as exc:
        if created is not None:
            try:
                created.update(level="ERROR", status_message=str(exc)[:500])
            except Exception:
                pass
        raise
    finally:
        _local.parent = previous
        if created is not None:
            try:
                created.end()
            except Exception:
                pass


def page_digest(pages: List[Dict[str, Any]], limit: int = 25) -> List[Dict[str, Any]]:
    """Per-page (url, chars, sha256) — what a tool span carries instead of bodies."""
    digest = []
    for page in pages[:limit]:
        raw = page.get("raw_content") or ""
        digest.append({"url": page.get("url"), "chars": len(raw), "sha256": sha256_text(raw)[:16]})
    return digest
