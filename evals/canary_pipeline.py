"""Tier B pipeline canary: the full workflow for each opted-in case, recorded.

Where Tier A (`evals/canary_fetch.py`) measures discovery alone at ~3
credits, Tier B runs `execute_workflow` end to end — discovery, enrichment,
the rubric judge, the critic — cold, exactly as a user search would, at
~$0.55 a case. It is the only monitor that can see the model-side
failures the run record names: an empty shortlist, a collapsed
validation, a truncated judge, `no_profile_found` climbing, a platform
missing from every card, cost or latency out of band. Weekly by design
(plan D6): the cost driver, and real user runs are free Tier B canaries in
between.

Same contract as Tier A: this script MEASURES and writes
`evals/out/canary_pipeline.json`; `evals/watch.py --tier B` judges. The
workflow opens its own trace (`execute_workflow(source=, case_id=)`), so
the trace URL and run id come back in the result. The run record is the
one the orchestrator attached (`workflow_summary["run_record"]`); when a
run fails before `_finalize_results` there is none, and the record is
built here from what the failure result carries — mostly None, which is
the honest shape of a run that did not complete.

`--source smoke` is the per-deploy check: the same run, tagged so the
charts can tell "we just deployed" from "it is Sunday". Vendor clients on
all three agents are wrapped for error capture (fatal vs transient, class
and redacted message) with production handling unchanged.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from evals import golden
from evals.cases import CanaryCase, case_by_id, tier_b_cases
from evals.thresholds import evaluate_pipeline
from evals.vendor_watch import VendorWatch, redact
from utils import tracing
from utils.run_record import build_run_record

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("evals/out/canary_pipeline.json")
TIER = "B"
SOURCES = ("canary", "smoke")


def _default_factory() -> Any:
    from agents.orchestrator import create_orchestrator

    return create_orchestrator()


def _state_from_result(result: Dict[str, Any], case: CanaryCase, source: str, radius: float) -> Dict[str, Any]:
    """Reassemble enough of the workflow state for `build_run_record`."""
    outputs = result.get("agent_outputs") or {}
    summary = result.get("workflow_summary") or {}
    return {
        "run_id": result.get("run_id"),
        "run_source": source,
        "case_id": case.case_id,
        "specialty": case.specialty,
        "location": case.location,
        "insurance": None,
        "preferences": {"search_radius_miles": radius},
        "gathered_data": outputs.get("data_gatherer") or {},
        "scored_providers": outputs.get("preference_scorer") or {},
        "validation_results": outputs.get("critic_validator") or {},
        "final_recommendations": result.get("final_recommendations") or [],
        "execution_log": result.get("execution_log") or [],
        "error_messages": result.get("error_messages") or [],
        "workflow_summary": summary if isinstance(summary, dict) else {},
    }


def run_case(
    case: CanaryCase,
    source: str = "canary",
    radius: Optional[float] = None,
    orchestrator_factory: Optional[Callable[[], Any]] = None,
    key_path: Path = golden.KEY_PATH,
) -> Dict[str, Any]:
    """Run the full workflow for one case and return its measurement (never raises)."""
    from utils.config import get_config

    config = get_config()
    radius = float(radius if radius is not None else config.DEFAULT_SEARCH_RADIUS)
    watch = VendorWatch()
    started = time.perf_counter()
    error: Optional[str] = None
    result: Dict[str, Any] = {}
    try:
        orchestrator = (orchestrator_factory or _default_factory)()
        watch.wrap_orchestrator(orchestrator)
        result = orchestrator.execute_workflow(
            case.specialty, case.location,
            preferences={"search_radius_miles": radius},
            use_cache=False,  # cold by design: a canary must not read or pollute a cache
            source=source, case_id=case.case_id,
        )
        if not isinstance(result, dict):
            raise TypeError(f"execute_workflow returned {type(result).__name__}")
    except Exception as exc:  # the next case must still run
        error = redact(f"{type(exc).__name__}: {exc}")
        logger.error("Pipeline canary %s crashed: %s", case.case_id, error)
        result = {"workflow_summary": {"workflow_failed": True, "error": error}, "error_messages": [error]}
    elapsed = round(time.perf_counter() - started, 2)
    summary = result.get("workflow_summary") if isinstance(result.get("workflow_summary"), dict) else {}
    failed = bool(summary.get("workflow_failed")) or error is not None
    status = "crashed" if error else ("error" if failed else "success")
    record = summary.get("run_record") if isinstance(summary.get("run_record"), dict) else None
    if record is None:
        record = build_run_record(_state_from_result(result, case, source, radius), config, pipeline=True)
    record["extra"]["tier"] = TIER
    record["extra"]["vendor_errors"] = len(watch.errors)
    gathered = (result.get("agent_outputs") or {}).get("data_gatherer") or {}
    meta = gathered.get("search_metadata") if isinstance(gathered, dict) else None
    fetch_stats = meta.get("fetch_stats") if isinstance(meta, dict) and isinstance(meta.get("fetch_stats"), dict) else {}
    failure_text = summary.get("error") if failed and not error else None
    # The golden cross-check: what THIS run read off the answer key's pages,
    # matched by URL, no extra fetch. Only when a key exists for the case.
    cross_check = None
    key = golden.load_key(key_path)
    if key and key.get("case_id") == case.case_id:
        scored = (result.get("agent_outputs") or {}).get("preference_scorer") or {}
        ranked = scored.get("ranked_providers") if isinstance(scored, dict) else None
        if not isinstance(ranked, list):
            ranked = gathered.get("providers") if isinstance(gathered, dict) else []
        try:
            cross_check = golden.cross_check_pipeline(key, ranked or [])
        except Exception as exc:  # a broken cross-check must not lose the run's measurement
            logger.warning("golden cross-check failed: %s", exc)
    return {
        "case_id": case.case_id,
        "specialty": case.specialty,
        "location": case.location,
        "tier": TIER,
        "source": source,
        "status": status,
        "message": redact(failure_text) if failure_text else None,
        "error": error or (redact(failure_text) if failure_text else None),
        "run_id": result.get("run_id") or record.get("run_id"),
        "trace_url": result.get("trace_url"),
        "elapsed_s": elapsed,
        "record": record,
        "fetch_stats": fetch_stats,
        "vendor_errors": list(watch.errors),
        "golden_cross_check": cross_check,
    }


def select_cases(case_id: Optional[str] = None) -> Sequence[CanaryCase]:
    if case_id:
        case = case_by_id(case_id)
        if case is None:
            raise ValueError(f"unknown case {case_id!r}")
        return (case,)
    return tier_b_cases()


def run(
    source: str = "canary",
    case_id: Optional[str] = None,
    radius: Optional[float] = None,
    out_path: Path = DEFAULT_OUT,
    orchestrator_factory: Optional[Callable[[], Any]] = None,
    key_path: Path = golden.KEY_PATH,
) -> Dict[str, Any]:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}: {' | '.join(SOURCES)}")
    cases = select_cases(case_id)
    payload: Dict[str, Any] = {
        "tier": TIER,
        "source": source,
        "case_filter": case_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": tracing.git_sha(),
        "cases": [],
    }
    for case in cases:
        payload["cases"].append(run_case(case, source=source, radius=radius, orchestrator_factory=orchestrator_factory, key_path=key_path))
        _write(out_path, payload)
    _write(out_path, payload)
    return payload


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def describe(entry: Dict[str, Any], case: CanaryCase) -> List[str]:
    r = entry.get("record") or {}
    cost = r.get("cost_usd")
    cost_text = f"${cost:.3f}" if isinstance(cost, (int, float)) else "$-"
    lines = [
        f"{case.case_id:26s} {str(entry.get('status')):12s} pool {r.get('pool_raw')}  enriched {r.get('n_enriched')}  "
        f"no_profile {r.get('n_no_profile_found')}  coverage {r.get('coverage_hg')}/{r.get('coverage_wm')}/{r.get('coverage_vi')}  "
        f"shortlist {r.get('shortlist_size')}  {cost_text}  {r.get('latency_s')} s"
    ]
    if entry.get("trace_url"):
        lines.append(f"    trace: {entry['trace_url']}")
    if entry.get("error"):
        lines.append(f"    error: {entry['error']}")
    cross = entry.get("golden_cross_check")
    if cross:
        lines.append(f"    golden cross-check: {cross['matched']} of {cross['key_pages']} key pages matched · "
                     f"agree {cross['agree']} · disagree {cross['disagree']} · identity {cross['identity']} · unread {cross['unread']}")
    for f in evaluate_pipeline(r, case, entry.get("vendor_errors"), entry.get("status"), entry.get("fetch_stats"), cross):
        lines.append(f"    {f.title}: {f.observed} (threshold {f.threshold})")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tier B pipeline canary (measures; the watcher judges)")
    parser.add_argument("--source", choices=SOURCES, default="canary", help="canary (scheduled) or smoke (per deploy)")
    parser.add_argument("--case", default=None, help="run one case by id instead of every Tier B case")
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        cases = select_cases(args.case)
    except ValueError as exc:
        print(f"canary: {exc}", file=sys.stderr)
        return 2
    print(f"Tier B pipeline canary — {args.source}: {', '.join(c.case_id for c in cases)}")
    payload = run(args.source, args.case, args.radius, args.out)
    for entry, case in zip(payload["cases"], cases):
        for line in describe(entry, case):
            print(line)
    print(f"wrote {args.out} — evaluate with: python -m evals.watch --tier B" + (f" --case {args.case}" if args.case else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
