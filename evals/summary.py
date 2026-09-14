"""The run summary: what the watcher writes to the run page and the console.

GitHub renders `$GITHUB_STEP_SUMMARY` at the top of the run page the failure
email links to, so this is the first thing a person sees after the email
subject. It leads with the worst severity, then the findings (observed vs
threshold), then the FIRST STEP for each — the runbook inline, because an
alert that makes the reader go find the runbook is an alert that gets
read tomorrow — then one row per case with the numbers the thresholds
compare against, and a trace link per case. Aggregates only: nothing here
names a provider.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from evals.cases import CanaryCase
from evals.thresholds import Finding, worst_severity

TIER_NAMES = {"A": "Tier A fetch canary", "B": "Tier B pipeline canary"}


def _cell(value: Any, fmt: Optional[str] = None) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if fmt and isinstance(value, (int, float)):
        return fmt.format(value)
    return str(value).replace("|", "\\|")


def _rows(record: Mapping[str, Any]) -> str:
    parts = [_cell(record.get(f"rows_{short}")) for short in ("hg", "wm", "vi")]
    return " / ".join(parts)


def _pages(record: Mapping[str, Any]) -> str:
    return f"{_cell(record.get('pages_fetched'))} / {_cell(record.get('pages_planned'))}"


def case_row(entry: Mapping[str, Any]) -> str:
    record = entry.get("record") or {}
    trace = entry.get("trace_url")
    cells = [
        _cell(entry.get("case_id")),
        _cell(entry.get("status")),
        _cell(record.get("pool_raw")),
        _rows(record),
        _pages(record),
        _cell(record.get("empty_bodies")),
        _cell(record.get("fallback_fired")),
        _cell(record.get("ring_fired")),
        _cell(record.get("tavily_credits")),
        _cell(record.get("cost_usd"), "${:.3f}"),
        _cell(record.get("latency_s"), "{:.0f} s"),
        f"[trace]({trace})" if trace else "—",
    ]
    return "| " + " | ".join(cells) + " |"


CASE_HEADER = (
    "| Case | Status | Pool | Rows hg / wm / vi | Pages fetched / planned | Empty | Fallback | Ring | Credits | Cost | Latency | Trace |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def findings_table(findings: Sequence[Finding]) -> str:
    lines = ["| Severity | Check | Case | Observed | Threshold |", "|---|---|---|---|---|"]
    for f in findings:
        lines.append(
            f"| {f.severity} | {_cell(f.title.split('] ', 1)[1].rsplit(' — ', 1)[0])} | {_cell(f.case_id)} "
            f"| {_cell(f.observed)} | {_cell(f.threshold)} |"
        )
    return "\n".join(lines)


def render_summary(
    tier: str,
    schedule: str,
    payload: Optional[Mapping[str, Any]],
    findings: Sequence[Finding],
    expected: Sequence[CanaryCase],
    run_url: Optional[str] = None,
    notes: Optional[Sequence[str]] = None,
    generated_at: Optional[datetime] = None,
    golden: Optional[Mapping[str, Any]] = None,
) -> str:
    when = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    title = TIER_NAMES.get(tier, f"Tier {tier}")
    worst = worst_severity(findings)
    entries = [c for c in (payload or {}).get("cases", []) if isinstance(c, dict)] if payload else []
    out: List[str] = [f"## {title} — {schedule} · {when}", ""]
    if worst:
        out.append(f"**Status: {worst} — {len(findings)} finding(s)** across {len(expected)} scheduled case(s).")
    else:
        out.append(f"**Status: clean** — {len(entries)} of {len(expected)} scheduled case(s) evaluated, nothing past a threshold.")
    if run_url:
        out.append(f"Run: {run_url}")
    out.append("")
    if findings:
        out += ["### Findings", "", findings_table(findings), "", "**First steps**", ""]
        seen = set()
        for f in findings:
            if f.identity in seen:
                continue
            seen.add(f.identity)
            out.append(f"- **{f.title}** — {f.first_step}")
        out.append("")
    if entries:
        out += ["### Cases", "", CASE_HEADER]
        out += [case_row(e) for e in entries]
        out.append("")
    if golden:
        out += golden_section(golden)
    vendor_notes = []
    for e in entries:
        transient = [v for v in (e.get("vendor_errors") or []) if v.get("kind") != "fatal"]
        if transient:
            kinds = sorted({f"{v.get('vendor')} {v.get('error')}" for v in transient})
            vendor_notes.append(f"{e.get('case_id')}: {len(transient)} transient vendor error(s) ({', '.join(kinds)})")
        if e.get("error"):
            vendor_notes.append(f"{e.get('case_id')}: {e['error']}")
        cross = e.get("golden_cross_check")
        if isinstance(cross, dict):
            vendor_notes.append(
                f"{e.get('case_id')}: golden cross-check — {cross.get('matched')} of {cross.get('key_pages')} key pages "
                f"matched, agree {cross.get('agree')}, disagree {cross.get('disagree')}, identity {cross.get('identity')}, "
                f"unread {cross.get('unread')}")
    all_notes = list(vendor_notes) + list(notes or [])
    if all_notes:
        out += ["### Notes", ""] + [f"- {n}" for n in all_notes] + [""]
    return "\n".join(out).rstrip() + "\n"


def golden_section(golden: Mapping[str, Any]) -> List[str]:
    """Counts by status, every page that is not ok, and the refresh proposals."""
    counts = golden.get("counts") or {}
    pages = [p for p in (golden.get("pages") or []) if isinstance(p, dict)]
    out = ["### Golden set", "",
           f"Key as of {_cell(golden.get('key_as_of'))} · " + ", ".join(f"{k} {v}" for k, v in counts.items() if v) + ".", ""]
    attention = [p for p in pages if p.get("status") not in ("ok", "world_drift")]
    if attention:
        out += ["| Status | Page | Notes |", "|---|---|---|"]
        out += [f"| {_cell(p.get('status'))} | {_cell(p.get('provider_id'))}/{_cell(p.get('platform'))} | {_cell('; '.join(str(n) for n in p.get('notes') or []))} |" for p in attention]
        out.append("")
    drift = [p for p in pages if p.get("status") == "world_drift"]
    if drift:
        out += ["**Refresh proposals** (the world moved within tolerance; no eyes needed):", ""]
        out += [f"- `python -m evals.refresh_fixtures --refresh {p.get('provider_id')}` — {p.get('platform')}: {'; '.join(str(n) for n in p.get('notes') or [])}" for p in drift]
        out.append("")
    return out


def write_step_summary(text: str, path: Optional[str] = None) -> bool:
    """Append to $GITHUB_STEP_SUMMARY when it is set; False outside Actions."""
    target = path if path is not None else os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return False
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
    return True
