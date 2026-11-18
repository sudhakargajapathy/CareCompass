"""Live-app health: what the Space's own visitors hit, as counts.

Every other monitor here measures a FRESH process. The canaries start one
per run, and the weekly report exports only canary environments, so both
showed a healthy ~40 s while the live Space ran 100-120 s searches for two
weeks (2026-11-03..16). The cause lived only in a long-lived process: the
app keeps one orchestrator per process, its Tavily client pooled keep-alive
sockets that the far end dropped after about a day idle, and the next
search's first attempts hung to the 30 s timeout. Langfuse showed it the
whole time (34 first-attempt `tavily.extract` errors, every one in the
`demo` environment, none in `ci`), but nothing read that environment.

This reads it and writes COUNTS ONLY to `evals/out/live_health.json`:

  * runs in the window, and their latency p50 / p95 / max;
  * Tavily tool calls at level ERROR, split by attempt (a first-attempt
    error the retry rescued costs seconds; a retry error costs pages) and
    by kind (timeout / connection / other).

No query, location, provider, trace id or page reaches the file: visitor
searches are never published, and the artifact this lands in is readable
by anyone who can read the repository. The `attempt` and the status
message are read in memory and only counted.

Opt-in, like the schedules: the environments come from `--environments`
or the `LIVE_WATCH_ENVIRONMENTS` variable, and with neither set the script
measures nothing and says so (exit 0). The watcher judges the file
(`evals.watch --live`); this script only measures, and a failed pull is
exit 1 so the step goes red.
"""

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from evals.weekly_report import _call, _client, fetch_rows, parse_environments

DEFAULT_OUT = Path("evals/out/live_health.json")
# The Tier A workflow runs daily; 26 h covers one day plus slack for a late
# cron, so a run is counted at least once. Overlap only re-counts a run in
# the next window, which keeps an issue open one more day, never hides one.
WINDOW_HOURS = 26
TAVILY_TOOLS = ("tavily.extract", "tavily.search")
OBSERVATION_PAGE_SIZE = 100


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile; None for no values. With the live Space's
    one or two searches a day, p95 is effectively the slowest run, which is
    the honest reading of so few samples."""
    nums = sorted(float(v) for v in values)
    if not nums:
        return None
    rank = max(1, math.ceil(pct / 100.0 * len(nums)))
    return nums[rank - 1]


def _attempt(raw_input: Any) -> Optional[int]:
    """The `attempt` the gatherer records on every Tavily tool span's input.

    The v2 observations API returns input as a raw string; a span recorded
    by a different code version may carry a dict or nothing at all.
    """
    data = raw_input
    if isinstance(raw_input, str):
        try:
            data = json.loads(raw_input)
        except ValueError:
            return None
    if isinstance(data, dict):
        try:
            return int(data.get("attempt"))
        except (TypeError, ValueError):
            return None
    return None


def error_kind(message: Optional[str]) -> str:
    """timeout | connection | other, from a span's status message.

    The two live signatures were tavily-python's "Request timed out after
    30 seconds" and `SSLError ... UNEXPECTED_EOF_WHILE_READING` (a reused
    socket the far end had closed). Anything else is the vendor talking.
    """
    text = (message or "").lower()
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if any(marker in text for marker in ("ssl", "eof", "connection", "reset by peer", "broken pipe")):
        return "connection"
    return "other"


def _observations(client: Any, **kwargs) -> Iterable[Any]:
    """Every observation matching `kwargs`, following the v2 cursor."""
    cursor: Optional[str] = None
    while True:
        result = _call(client.api.observations.get_many, limit=OBSERVATION_PAGE_SIZE,
                       cursor=cursor, **kwargs)
        for item in result.data or []:
            yield item
        following = getattr(getattr(result, "meta", None), "cursor", None)
        # An empty page can still carry a cursor (filtered rows), so the
        # cursor alone decides; a cursor that does not move would loop.
        if not following or following == cursor:
            return
        cursor = following


def tavily_errors(client: Any, start: datetime, end: datetime,
                  environments: Sequence[str]) -> Dict[str, Any]:
    """Counts of Tavily tool calls at level ERROR in [start, end)."""
    wanted = {e.lower() for e in environments}
    counts: Dict[str, Any] = {
        "first_attempt_errors": 0,
        "retry_errors": 0,
        "unknown_attempt_errors": 0,
        "runs_affected": 0,
        "by_kind": {"timeout": 0, "connection": 0, "other": 0},
    }
    traces = set()
    for name in TAVILY_TOOLS:
        for obs in _observations(client, name=name, level="ERROR", environment=sorted(wanted),
                                 from_start_time=start, to_start_time=end, fields="core,basic,io"):
            # Re-checked here for the reason the weekly export re-checks: a
            # filter the server ignored would count the other Space's errors.
            if wanted and str(getattr(obs, "environment", "") or "").lower() not in wanted:
                continue
            attempt = _attempt(getattr(obs, "input", None))
            if attempt == 1:
                counts["first_attempt_errors"] += 1
            elif attempt is not None and attempt > 1:
                counts["retry_errors"] += 1
            else:
                counts["unknown_attempt_errors"] += 1
            counts["by_kind"][error_kind(getattr(obs, "status_message", None))] += 1
            trace_id = getattr(obs, "trace_id", None)
            if trace_id:
                traces.add(trace_id)
    counts["runs_affected"] = len(traces)
    return counts


def measure(client: Any, environments: Sequence[str], now: Optional[datetime] = None,
            window_hours: int = WINDOW_HOURS) -> Dict[str, Any]:
    """The live-health payload for the window ending `now`. Counts only."""
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(hours=window_hours)
    # Visitor searches only: a canary that happened to run under the live
    # label would otherwise vouch for the live process it did not run in.
    rows = [r for r in fetch_rows(client, start, end, environments) if (r.get("source") or "user") == "user"]
    latencies = [float(r["latency_s"]) for r in rows
                 if isinstance(r.get("latency_s"), (int, float)) and not isinstance(r.get("latency_s"), bool)]
    return {
        "generated_at": end.isoformat(timespec="seconds"),
        "window": {"start": start.isoformat(timespec="seconds"), "end": end.isoformat(timespec="seconds"),
                   "hours": window_hours},
        "environments": list(environments),
        "runs": len(rows),
        "latency_s": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": max(latencies) if latencies else None,
            "measured": len(latencies),
        },
        "tavily": tavily_errors(client, start, end, environments),
    }


def main(argv: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None) -> int:
    import os

    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description="Measure the live app's Tavily errors and latency (counts only)")
    parser.add_argument("--environments", default=None,
                        help="comma-separated trace environment labels of the live app "
                             "(default: the LIVE_WATCH_ENVIRONMENTS variable)")
    parser.add_argument("--hours", type=int, default=WINDOW_HOURS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    environments = parse_environments(args.environments if args.environments is not None
                                      else env.get("LIVE_WATCH_ENVIRONMENTS"))
    if not environments:
        # Unset must not mean "every environment": that would count the
        # canaries' own fresh processes as the live app and vouch for it.
        print("live watch: no environments configured (LIVE_WATCH_ENVIRONMENTS); nothing measured")
        return 0
    missing = [k for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL") if not env.get(k)]
    if missing:
        print(f"live watch: missing {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        payload = measure(_client(env), environments, window_hours=args.hours)
    except Exception as exc:  # the red step is the alert
        print(f"live watch: pull failed — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tavily = payload["tavily"]
    print(f"live watch [{', '.join(environments)}], last {args.hours} h: {payload['runs']} run(s), "
          f"p95 {payload['latency_s']['p95']} s, Tavily errors: {tavily['first_attempt_errors']} first-attempt, "
          f"{tavily['retry_errors']} after retry → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
