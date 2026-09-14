"""Tier A fetch canary: run discovery for each scheduled case, record the yield.

What it measures. The exact production discovery path — `gather_providers`
with enrichment off — so the numbers are the ones a live search would
produce: pages planned/fetched/empty per platform, listing rows per
platform, the deduped pool, whether the fetch-mode fallback or the ring
fired, credits and cost. In extract mode that is ~3 credits and a few cents
of model spend (the LLM runs only over pages the deterministic parsers
could not read; a healthy run has none). It is deliberately NOT a
parser-only harness: the incident this monitors was a vendor's index rot
that only the production fetch path could see.

What it does NOT do. It measures and writes; it never decides. The watcher
(`evals/watch.py`) is the single evaluator — it reads this script's JSON,
applies `evals/thresholds`, writes the run summary and syncs the issues.
The console preview of findings at the end of a local run calls the same
evaluator; nothing about a threshold lives here. The script always exits 0:
a crashed case is recorded as `status: crashed` with a redacted error and
the next case still runs, because a canary that dies on case one reports
nothing about cases two to four.

Tracing. Each case is one Langfuse trace (`source:canary`, `case:<id>`,
`tier:A`), the same root/step/tool shape a user search leaves, with the run
record attached as scores — so the daily canary draws the pool-size chart
the September incident needed, and a vendor error is on the failing tool span.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from evals.cases import CanaryCase, case_by_id, cases_for
from evals.thresholds import evaluate_fetch
from evals.vendor_watch import VendorWatch, redact
from utils import tracing
from utils.cost_tracker import get_cost_tracker
from utils.run_record import build_run_record

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("evals/out/canary_fetch.json")
TIER = "A"


def trace_args(case: CanaryCase, radius: float, run_id: str, config: Any) -> Dict[str, Any]:
    """Root-trace input/metadata/tags — the orchestrator's shape, canary-labelled."""
    from utils.geo import parse_location

    parts = parse_location(case.location)
    metadata = {
        "run_id": run_id,
        "source": "canary",
        "case_id": case.case_id,
        "tier": TIER,
        "git_sha": tracing.git_sha(),
        "tavily_mode": getattr(config, "TAVILY_MODE", None),
        "gatherer_model": getattr(config, "GATHERER_MODEL", None),
        "budget": getattr(config, "MAX_PROVIDERS_TO_ENRICH", None),
        "radius_miles": radius,
        "specialty": case.specialty,
        "city": parts.get("city"),
        "state": parts.get("state"),
    }
    return {
        "run_id": run_id,
        "input": {"specialty": case.specialty, "location": case.location, "radius_miles": radius},
        "metadata": metadata,
        "tags": ["source:canary", f"tier:{TIER}", f"case:{case.case_id}", f"mode:{metadata['tavily_mode']}"],
    }


def run_case(
    case: CanaryCase,
    radius: Optional[float] = None,
    agent_factory: Optional[Callable[[], Any]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Run discovery for one case and return its measurement (never raises)."""
    from utils.config import get_config

    config = get_config()
    radius = float(radius if radius is not None else config.DEFAULT_SEARCH_RADIUS)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"canary-{case.case_id}-{stamp}"
    tracker = get_cost_tracker()
    tracker.reset()
    watch = VendorWatch()
    started = time.perf_counter()
    agent: Any = None
    result: Dict[str, Any] = {}
    error: Optional[str] = None
    status = "crashed"

    with tracing.start_run(**trace_args(case, radius, run_id, config)) as run:
        if run is not None:
            run.step_started("gather_data", {
                "specialty": case.specialty, "location": case.location, "radius_miles": radius,
            })
        try:
            agent = (agent_factory or _default_factory)()
            watch.wrap_gatherer(agent)
            result = agent.gather_providers(
                case.specialty, case.location, enrich=False, radius_miles=radius
            )
            if not isinstance(result, dict):
                raise TypeError(f"gather_providers returned {type(result).__name__}")
            status = str(result.get("status") or "unknown")
        except Exception as exc:  # the next case must still run
            error = redact(f"{type(exc).__name__}: {exc}")
            logger.error("Canary %s crashed: %s", case.case_id, error)
            result = {"providers": [], "search_metadata": {}, "status": "crashed", "message": error}
        stats: Dict[str, Any] = {}
        if agent is not None and hasattr(agent, "fetch_stats"):
            try:
                stats = agent.fetch_stats() or {}
            except Exception as exc:
                logger.warning("fetch_stats unavailable: %s", exc)
        meta = result.get("search_metadata")
        if not isinstance(meta, dict):
            meta = {}
            result["search_metadata"] = meta
        meta["fetch_stats"] = stats
        elapsed = round(time.perf_counter() - started, 2)
        cost = tracker.summary()
        state = {
            "run_id": run_id,
            "run_source": "canary",
            "case_id": case.case_id,
            "specialty": case.specialty,
            "location": case.location,
            "insurance": None,
            "preferences": {"search_radius_miles": radius},
            "gathered_data": result,
            "scored_providers": {},
            "validation_results": {},
            "final_recommendations": [],
            "execution_log": [],
            "error_messages": [error] if error else [],
            "workflow_summary": {"cost_summary": cost},
        }
        record = build_run_record(state, config)
        record["extra"]["tier"] = TIER
        record["extra"]["vendor_errors"] = len(watch.errors)
        providers_found = len(result.get("providers") or []) if isinstance(result.get("providers"), list) else 0
        if run is not None:
            run.step_finished("gather_data", "failed" if error else "completed", {
                "status": status, "providers_found": providers_found, "elapsed_s": elapsed,
                **({"error": error} if error else {}),
            })
            run.finish(
                record=record,
                output={"status": status, "providers_found": providers_found, "cost_usd": cost.get("total_usd")},
                error=error,
            )
        trace_url = tracing.trace_url(run)

    return {
        "case_id": case.case_id,
        "specialty": case.specialty,
        "location": case.location,
        "tier": TIER,
        "status": status,
        "message": redact(result.get("message")) if result.get("message") else None,
        "error": error,
        "run_id": run_id,
        "trace_url": trace_url,
        "elapsed_s": elapsed,
        "record": record,
        "fetch_stats": stats,
        "vendor_errors": list(watch.errors),
    }


def _default_factory() -> Any:
    from agents.data_gatherer import create_data_gatherer

    return create_data_gatherer()


def select_cases(schedule: str, case_id: Optional[str] = None) -> Sequence[CanaryCase]:
    if case_id:
        case = case_by_id(case_id)
        if case is None:
            raise ValueError(f"unknown case {case_id!r}")
        return (case,)
    return cases_for(schedule)


def run(
    schedule: str = "daily",
    case_id: Optional[str] = None,
    radius: Optional[float] = None,
    out_path: Path = DEFAULT_OUT,
    agent_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Run every selected case, write the JSON payload, return it."""
    cases = select_cases(schedule, case_id)
    payload: Dict[str, Any] = {
        "tier": TIER,
        "schedule": schedule,
        "case_filter": case_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": tracing.git_sha(),
        "cases": [],
    }
    for case in cases:
        entry = run_case(case, radius=radius, agent_factory=agent_factory)
        payload["cases"].append(entry)
        _write(out_path, payload)  # after every case: a crash later still leaves what ran
    _write(out_path, payload)
    return payload


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def describe(entry: Dict[str, Any], case: CanaryCase) -> List[str]:
    """Console lines for one case: the numbers, then the findings preview."""
    r = entry.get("record") or {}
    rows = "/".join(
        str(r.get(f"rows_{s}")) if r.get(f"rows_{s}") is not None else "-" for s in ("hg", "wm", "vi")
    )
    cost = r.get("cost_usd")
    cost_text = f"${cost:.3f}" if isinstance(cost, (int, float)) else "$-"
    lines = [
        f"{case.case_id:26s} {str(entry.get('status')):12s} pool {r.get('pool_raw')}  rows {rows}  "
        f"pages {r.get('pages_fetched')}/{r.get('pages_planned')}  credits {r.get('tavily_credits')}  "
        f"{cost_text}  {entry.get('elapsed_s')} s"
    ]
    if entry.get("trace_url"):
        lines.append(f"    trace: {entry['trace_url']}")
    if entry.get("error"):
        lines.append(f"    error: {entry['error']}")
    for f in evaluate_fetch(r, case, entry.get("fetch_stats"), entry.get("vendor_errors"), entry.get("status")):
        lines.append(f"    {f.title}: {f.observed} (threshold {f.threshold})")
    return lines

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tier A fetch canary (measures; the watcher judges)")
    parser.add_argument("--schedule", choices=("daily", "weekly", "all"), default="daily")
    parser.add_argument("--case", default=None, help="run one case by id instead of the schedule")
    parser.add_argument("--radius", type=float, default=None, help="search radius in miles (default: config)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        cases = select_cases(args.schedule, args.case)
    except ValueError as exc:
        print(f"canary: {exc}", file=sys.stderr)
        return 2
    print(f"Tier A fetch canary — {args.schedule}: {', '.join(c.case_id for c in cases)}")
    payload = run(args.schedule, args.case, args.radius, args.out)
    for entry, case in zip(payload["cases"], cases):
        for line in describe(entry, case):
            print(line)
    print(f"wrote {args.out} — evaluate with: python -m evals.watch --tier A --schedule {args.schedule}"
          + (f" --case {args.case}" if args.case else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
