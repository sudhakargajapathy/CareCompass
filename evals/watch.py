"""The watcher: read a canary's measurements, judge them, report, sync issues.

One evaluator for every tier. The canary scripts measure and write JSON;
this reads it, applies `evals/thresholds`, writes the run summary (the
step summary on Actions, stdout everywhere), mirrors the breaches into
GitHub issues, and FAILS THE JOB on any finding — the failure is what makes
the run page red and the email go out; the issue is what carries the
detail and the recovery.

A case the schedule expected and the JSON does not carry is itself a P1
(`canary_did_not_run`): a canary that silently stops is the failure mode
nobody planned for — GitHub disabling an idle cron — and the one no
threshold on a yield can see. A missing file is that finding for every
expected case.

Exit codes: 0 clean · 1 at least one finding · 2 usage error.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from evals.alerts import GitHubIssues, sync_issues
from evals.cases import CanaryCase, case_by_id, cases_for, tier_b_cases
from evals.summary import render_summary, write_step_summary
from evals.thresholds import (
    GOLDEN_CASE_ID,
    GOLDEN_CHECKS,
    REPORT_CASE_ID,
    Finding,
    checks_for_tier,
    evaluate_fetch,
    evaluate_golden,
    evaluate_pipeline,
    evaluate_report_freshness,
)

DEFAULT_INPUTS = {"A": Path("evals/out/canary_fetch.json"), "B": Path("evals/out/canary_pipeline.json")}
DEFAULT_REPORTS_DIR = Path("reports")


def load_results(path: Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def evaluate_results(
    tier: str, expected: Sequence[CanaryCase], payload: Optional[Mapping[str, Any]]
) -> Tuple[List[Finding], Dict[str, Set[str]]]:
    """Findings for every expected case, and the checks each case was evaluated on."""
    findings: List[Finding] = []
    evaluated: Dict[str, Set[str]] = {}
    entries = {
        str(c.get("case_id")): c
        for c in ((payload or {}).get("cases") or [])
        if isinstance(c, dict)
    }
    for case in expected:
        entry = entries.get(case.case_id)
        if entry is None:
            findings.append(Finding(
                "canary_did_not_run", "P1", case.case_id,
                "no result recorded" if payload else "no results file",
                "one result per scheduled case",
            ))
            evaluated[case.case_id] = {"canary_did_not_run"}
            continue
        record = entry.get("record") if isinstance(entry.get("record"), dict) else {}
        vendor_errors = entry.get("vendor_errors") if isinstance(entry.get("vendor_errors"), list) else []
        status = entry.get("status")
        if tier == "A":
            found = evaluate_fetch(record, case, entry.get("fetch_stats"), vendor_errors, status)
        else:
            found = evaluate_pipeline(record, case, vendor_errors, status, entry.get("fetch_stats"), entry.get("golden_cross_check"))
        findings.extend(found)
        evaluated[case.case_id] = set(checks_for_tier(tier))
    return findings, evaluated


def evaluate_report(reports_dir: Optional[Path]) -> Tuple[List[Finding], Dict[str, Set[str]]]:
    """The freshness watchdog: P3 when the weekly report's stamp is old.

    Evaluated only when a stamp exists — before the first report there is
    nothing to be stale — and always under its own pseudo-case, so a stale
    report has one issue with one lifetime, closed by the run that sees a
    fresh stamp.
    """
    if not reports_dir:
        return [], {}
    stamp = Path(reports_dir) / "latest.json"
    if not stamp.exists():
        return [], {}
    try:
        latest = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        latest = {"generated_at": "unreadable"}
    return evaluate_report_freshness(latest if isinstance(latest, dict) else {}), {REPORT_CASE_ID: {"report_stale"}}


def evaluate_golden_file(path: Optional[Path]) -> Tuple[List[Finding], Dict[str, Set[str]]]:
    """The golden set's grades, when this run was expected to produce them.

    Passing a path means the run SHOULD have graded the set (the weekly
    run does): a missing file is the golden-set case not running, the same
    P1 a missing canary case gets. Evaluated pages vouch for every golden
    check, so a page that recovered closes its issue.
    """
    if not path:
        return [], {}
    payload = load_results(path)
    if payload is None:
        return (
            [Finding("canary_did_not_run", "P1", GOLDEN_CASE_ID, "no golden results file", "one grade per key page")],
            {GOLDEN_CASE_ID: {"canary_did_not_run"}},
        )
    return evaluate_golden(payload), {GOLDEN_CASE_ID: set(GOLDEN_CHECKS) | {"canary_did_not_run"}}


def expected_cases(tier: str, schedule: str, case_id: Optional[str]) -> Optional[Sequence[CanaryCase]]:
    """Which cases this run should have measured; None for an unknown case id."""
    if case_id:
        case = case_by_id(case_id)
        return None if case is None else (case,)
    return tier_b_cases() if tier == "B" else cases_for(schedule)


def run_url_from_env(env: Mapping[str, str] = os.environ) -> Optional[str]:
    server, repo, run_id = env.get("GITHUB_SERVER_URL"), env.get("GITHUB_REPOSITORY"), env.get("GITHUB_RUN_ID")
    return f"{server}/{repo}/actions/runs/{run_id}" if server and repo and run_id else None


def main(argv: Optional[Sequence[str]] = None, env: Mapping[str, str] = os.environ) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a canary run, write the summary, sync issues")
    parser.add_argument("--tier", choices=("A", "B"), default="A")
    parser.add_argument("--schedule", choices=("daily", "weekly", "all"), default="daily")
    parser.add_argument("--case", default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--no-issues", action="store_true", help="skip the GitHub issue sync")
    parser.add_argument("--reports-dir", type=Path, default=None,
                        help="also check the weekly report's freshness stamp in this directory")
    parser.add_argument("--golden", type=Path, default=None,
                        help="the golden-set grades this run was expected to produce (evals/out/golden.json)")
    args = parser.parse_args(argv)

    expected = expected_cases(args.tier, args.schedule, args.case)
    if expected is None:
        print(f"watch: unknown case {args.case!r}", file=sys.stderr)
        return 2
    payload = load_results(args.input or DEFAULT_INPUTS[args.tier])
    findings, evaluated = evaluate_results(args.tier, expected, payload)
    report_findings, report_scope = evaluate_report(args.reports_dir)
    findings.extend(report_findings)
    evaluated.update(report_scope)
    golden_findings, golden_scope = evaluate_golden_file(args.golden)
    findings.extend(golden_findings)
    evaluated.update(golden_scope)
    golden_payload = load_results(args.golden) if args.golden else None
    run_url = run_url_from_env(env)
    notes: List[str] = []

    token, repo = env.get("GITHUB_TOKEN"), env.get("GITHUB_REPOSITORY")
    if args.no_issues:
        notes.append("issue sync skipped (--no-issues)")
    elif not (token and repo):
        notes.append("issue sync skipped (GITHUB_TOKEN / GITHUB_REPOSITORY not set)")
    else:
        trace_urls = {
            str(c.get("case_id")): c.get("trace_url")
            for c in ((payload or {}).get("cases") or []) if isinstance(c, dict)
        }
        try:
            result = sync_issues(GitHubIssues(token, repo), findings, evaluated, run_url, trace_urls)
            notes.append(result.describe())
            notes.extend(f"issue sync error: {e}" for e in result.errors)
        except Exception as exc:  # the summary and the exit code must still land
            notes.append(f"issue sync failed: {type(exc).__name__}: {exc}")

    text = render_summary(args.tier, args.schedule if not args.case else f"case {args.case}",
                          payload, findings, expected, run_url, notes, golden=golden_payload)
    print(text)
    write_step_summary(text, env.get("GITHUB_STEP_SUMMARY") or None)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
