"""Per-search cost and timing tracking for the multi-agent workflow.

Every LLM, embedding, and Tavily call records its usage here; the
orchestrator resets the tracker at workflow start and snapshots a summary
into the results, which the UI renders as a "search cost" card.

Prices are list prices per million tokens (checked 2026-07). They are
estimates for display, not billing records.
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

# USD per 1M tokens (input, output). Sonnet 5 has intro pricing of $2/$10
# through 2026-08-31; the list price is used here so estimates never
# understate.
PRICING_PER_MTOK: Dict[str, Dict[str, float]] = {
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "gpt-5.6-terra": {"input": 2.50, "output": 15.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}

# Tavily pay-as-you-go is roughly $0.008 per credit; a basic search costs
# 1 credit and an advanced search 2.
TAVILY_COST_PER_CREDIT = 0.008
TAVILY_CREDITS_PER_SEARCH = {"basic": 1, "advanced": 2}


def safe_usage(response: Any) -> Tuple[int, int]:
    """Extract (input_tokens, output_tokens) from any SDK response.

    Handles Anthropic messages (.usage.input_tokens/.output_tokens) and
    OpenAI chat/embeddings (.usage.prompt_tokens/.completion_tokens).
    Returns (0, 0) for anything else — including test Mocks — so cost
    capture can never break a workflow.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    for in_attr, out_attr in (("input_tokens", "output_tokens"), ("prompt_tokens", "completion_tokens")):
        raw_in = getattr(usage, in_attr, None)
        if raw_in is None:
            continue
        try:
            input_tokens = int(raw_in)
            output_tokens = int(getattr(usage, out_attr, 0) or 0)
            return max(input_tokens, 0), max(output_tokens, 0)
        except (TypeError, ValueError):
            continue

    return 0, 0


def _llm_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING_PER_MTOK.get(model)
    if pricing is None:
        return 0.0
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


class CostTracker:
    """Thread-safe accumulator for one workflow run's API usage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Clear all recorded usage for a new workflow run."""
        with self._lock:
            self._llm_calls: list = []
            self._tavily_searches: list = []
            self._embedding_tokens: int = 0
            self._embedding_model: str = "text-embedding-3-small"
            self._step_timings: Dict[str, float] = {}
            self._cache_hits: int = 0
            self._cache_misses: int = 0
            self._run_started: float = time.perf_counter()

    def record_llm(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent: str = "",
        duration_s: float = 0.0,
    ) -> None:
        """Record one LLM call's token usage."""
        entry = {
            "model": model,
            "agent": agent,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_s": round(duration_s, 2),
            "cost_usd": _llm_cost_usd(model, input_tokens, output_tokens),
            "priced": model in PRICING_PER_MTOK,
        }
        with self._lock:
            self._llm_calls.append(entry)
        if not entry["priced"]:
            logger.warning(f"No pricing entry for model {model}; cost recorded as $0")

    def record_tavily(self, depth: str = "basic", agent: str = "") -> None:
        """Record one Tavily search at the given depth."""
        credits = TAVILY_CREDITS_PER_SEARCH.get(depth, 1)
        with self._lock:
            self._tavily_searches.append({"depth": depth, "agent": agent, "credits": credits})

    def record_embeddings(self, tokens: int, model: str = "text-embedding-3-small") -> None:
        """Record embedding usage (token count is the input size)."""
        with self._lock:
            self._embedding_tokens += max(int(tokens or 0), 0)
            self._embedding_model = model

    def record_cache(self, hits: int = 0, misses: int = 0) -> None:
        """Record enrichment-cache outcomes for this search.

        Surfacing hit/miss on the cost card is how a warm run is verified
        without reading logs — the saving is otherwise invisible, since a
        cache hit shows up only as calls that did NOT happen.
        """
        with self._lock:
            self._cache_hits += max(int(hits or 0), 0)
            self._cache_misses += max(int(misses or 0), 0)

    @contextmanager
    def step(self, step_name: str) -> Iterator[None]:
        """Time a workflow step's wall clock."""
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self._step_timings[step_name] = round(elapsed, 2)

    def summary(self) -> Dict[str, Any]:
        """Snapshot totals for display: overall USD, per-model/agent breakdowns."""
        with self._lock:
            llm_calls = list(self._llm_calls)
            tavily = list(self._tavily_searches)
            embedding_tokens = self._embedding_tokens
            embedding_model = self._embedding_model
            step_timings = dict(self._step_timings)
            cache_hits = self._cache_hits
            cache_misses = self._cache_misses
            elapsed_s = round(time.perf_counter() - self._run_started, 2)

        by_model: Dict[str, Dict[str, Any]] = {}
        for call in llm_calls:
            slot = by_model.setdefault(
                call["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            )
            slot["calls"] += 1
            slot["input_tokens"] += call["input_tokens"]
            slot["output_tokens"] += call["output_tokens"]
            slot["cost_usd"] += call["cost_usd"]

        by_agent: Dict[str, float] = {}
        for call in llm_calls:
            agent = call["agent"] or "unknown"
            by_agent[agent] = by_agent.get(agent, 0.0) + call["cost_usd"]

        llm_cost = sum(call["cost_usd"] for call in llm_calls)
        tavily_credits = sum(s["credits"] for s in tavily)
        tavily_cost = tavily_credits * TAVILY_COST_PER_CREDIT
        embedding_cost = _llm_cost_usd(embedding_model, embedding_tokens, 0)

        return {
            "total_usd": round(llm_cost + tavily_cost + embedding_cost, 6),
            "cache": {
                "hits": cache_hits,
                "misses": cache_misses,
                "lookups": cache_hits + cache_misses,
            },
            "llm": {
                "cost_usd": round(llm_cost, 6),
                "calls": len(llm_calls),
                "input_tokens": sum(c["input_tokens"] for c in llm_calls),
                "output_tokens": sum(c["output_tokens"] for c in llm_calls),
                "by_model": {
                    model: {**slot, "cost_usd": round(slot["cost_usd"], 6)}
                    for model, slot in by_model.items()
                },
                "by_agent": {agent: round(cost, 6) for agent, cost in by_agent.items()},
            },
            "tavily": {
                "searches": len(tavily),
                "credits": tavily_credits,
                "cost_usd": round(tavily_cost, 6),
            },
            "embeddings": {
                "model": embedding_model,
                "tokens": embedding_tokens,
                "cost_usd": round(embedding_cost, 6),
            },
            "step_timings": step_timings,
            "elapsed_s": elapsed_s,
        }


_tracker: Optional[CostTracker] = None
_tracker_lock = threading.Lock()


def get_cost_tracker() -> CostTracker:
    """Get the process-wide cost tracker singleton."""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = CostTracker()
        return _tracker
