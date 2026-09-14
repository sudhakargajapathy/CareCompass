"""Vendor-error capture at the canary boundary.

The agents swallow vendor exceptions on purpose — a failed Tavily batch costs
that page, a failed extraction shard costs that shard, and the run goes on
(`gather_providers` itself returns `status: error` with a generic message).
That is right for a patient in front of the app and wrong for a monitor,
which needs to know WHICH vendor failed and whether the failure is the kind
that recovers on retry or the kind that means an unpaid bill. Log scraping
cannot tell: the gatherer logs `f"...: {e}"`, and a Tavily
`UsageLimitExceededError` stringifies without its class name.

So the canary wraps the vendor CLIENTS the agent holds, before the run, in
a recording proxy: every call goes through unchanged, every exception is
noted (vendor, class name, redacted message, fatal/transient) and re-raised
so the agent's own handling runs exactly as in production. Zero extra
credits, no paid probe, and the exception object itself is the evidence.

Fatal vs transient: auth, credit, quota and permission failures do not
recover on the gatherer's retry and are P1 on first sight. Anything else
(timeouts, connection resets, a 5xx) is transient and becomes a finding
only when the run then fetched nothing at all — the threshold module holds
that rule. The split is by exception class first and message markers
second, because the vendors disagree on classes: Anthropic reports credit
exhaustion as a 400 BadRequestError whose message says "credit balance is
too low", and OpenAI reports it as a 429 RateLimitError with
"insufficient_quota" — a RateLimitError alone is just a busy minute.
"""

import os
import re
from typing import Any, Dict, List, Optional

FATAL_CLASSES = frozenset({
    # tavily
    "InvalidAPIKeyError", "MissingAPIKeyError", "UsageLimitExceededError", "ForbiddenError",
    "TavilyKeylessLimitError", "KeylessUnsupportedEndpointError",
    # anthropic / openai
    "AuthenticationError", "PermissionDeniedError",
})
FATAL_MESSAGE_MARKERS = (
    "credit balance", "insufficient_quota", "quota", "billing", "invalid api key",
    "invalid x-api-key", "incorrect api key", "unauthorized", "authentication",
    "usage limit", "forbidden", "payment",
)
MESSAGE_CHARS = 300

_SECRET_ENV_NAMES = (
    "TAVILY_API_KEY", "APP_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "GITHUB_TOKEN",
)
# Token SHAPES, for the case where the value in the message is not the one
# in this process's environment (a key pasted into a config file, a second
# key in a vendor's own error text).
_TOKEN_SHAPES = re.compile(r"\b(?:sk-(?:ant-)?|tvly-|pk-lf-|sk-lf-|ghp_|github_pat_)[A-Za-z0-9_\-]{8,}")


def redact(text: Any, env: Optional[Dict[str, str]] = None, limit: int = MESSAGE_CHARS) -> str:
    """Scrub every configured secret and every token-shaped string; cap the length."""
    out = str(text or "")
    source = os.environ if env is None else env
    for name in _SECRET_ENV_NAMES:
        secret = (source.get(name) or "").strip()
        if len(secret) >= 8:
            out = out.replace(secret, "***")
    out = _TOKEN_SHAPES.sub("***", out)
    out = " ".join(out.split())
    return out[:limit] + ("…" if len(out) > limit else "")


def classify(exc: BaseException) -> str:
    """'fatal' for auth/credit/quota/permission failures, 'transient' otherwise."""
    if type(exc).__name__ in FATAL_CLASSES:
        return "fatal"
    message = str(exc).lower()
    return "fatal" if any(marker in message for marker in FATAL_MESSAGE_MARKERS) else "transient"


class VendorWatch:
    """Collects vendor exceptions seen through the proxies it hands out."""

    def __init__(self) -> None:
        self.errors: List[Dict[str, Any]] = []

    def note(self, vendor: str, exc: BaseException) -> None:
        self.errors.append({
            "vendor": vendor,
            "error": type(exc).__name__,
            "message": redact(str(exc)),
            "kind": classify(exc),
        })

    def wrap(self, target: Any, vendor: str) -> Any:
        return target if target is None else _Recording(target, self, vendor)

    def wrap_gatherer(self, agent: Any) -> None:
        """Wrap the clients a DataGathererAgent holds (missing ones are skipped)."""
        if agent is None:
            return
        for attr, vendor in (("tavily_client", "tavily"), ("anthropic_client", "anthropic")):
            client = getattr(agent, attr, None)
            if client is not None:
                setattr(agent, attr, self.wrap(client, vendor))

    def wrap_orchestrator(self, orchestrator: Any) -> None:
        """Wrap every vendor client the three agents hold (missing ones are skipped)."""
        self.wrap_gatherer(getattr(orchestrator, "data_gatherer", None))
        scorer = getattr(orchestrator, "preference_scorer", None)
        if scorer is not None and getattr(scorer, "openai_client", None) is not None:
            scorer.openai_client = self.wrap(scorer.openai_client, "openai")
        critic = getattr(orchestrator, "critic_validator", None)
        if critic is not None and getattr(critic, "anthropic_client", None) is not None:
            critic.anthropic_client = self.wrap(critic.anthropic_client, "anthropic")

    @property
    def fatal(self) -> List[Dict[str, Any]]:
        return [e for e in self.errors if e["kind"] == "fatal"]

    @property
    def transient(self) -> List[Dict[str, Any]]:
        return [e for e in self.errors if e["kind"] != "fatal"]


_PRIMITIVES = (str, bytes, int, float, bool, type(None), list, tuple, dict, set, frozenset)


class _Recording:
    """A transparent proxy: attributes resolve through it, calls are recorded.

    Nested resources (`client.messages.create`) work because every
    non-primitive attribute comes back wrapped, and a wrapped attribute is
    itself callable — so `client.messages` is a proxy, `.create` is a proxy,
    and the call on it is what gets recorded. Mock clients (every attribute
    callable) take the same path.
    """

    __slots__ = ("_target", "_watch", "_vendor")

    def __init__(self, target: Any, watch: VendorWatch, vendor: str) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_watch", watch)
        object.__setattr__(self, "_vendor", vendor)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if isinstance(attr, _PRIMITIVES) and not callable(attr):
            return attr
        return _Recording(attr, self._watch, self._vendor)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._target(*args, **kwargs)
        except Exception as exc:
            self._watch.note(self._vendor, exc)
            raise

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"<recording {self._vendor}: {self._target!r}>"
