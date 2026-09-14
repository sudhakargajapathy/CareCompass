"""The weekly report: the week's run records pulled from Langfuse, exported
to git as rows, rendered as one markdown page.

Two outputs, both public, both aggregates only:

  reports/run_records.jsonl   one line per run — the SYSTEM OF RECORD.
                              Append-only, deduped by run_id, every field
                              from utils.run_record.RUN_RECORD_FIELDS plus
                              the trace id and environment. Outlives the
                              trace store's retention window and any vendor
                              decision, needs no secret to read, and every
                              week's rows arrive as a reviewable diff.
  reports/YYYY-Www.md         the week's page: counts by source and
                              environment, the canary cases against their
                              floors, the pipeline runs' model-side health,
                              and per-source medians for cost and latency.
  reports/latest.json         the job's own stamp (when, which week, how
                              many rows) — the freshness watchdog the daily
                              watcher reads, because GitHub disables idle
                              crons silently and a report that stops being
                              written is invisible from inside the report.

Rows are built from exactly two places on the trace — the run record in
its metadata (descriptors + structured fields) and its scores (the
measurements) — never from the trace's input/output or any observation, so
nothing a page or a prompt contained can reach the public file. The
`verify` source (account-check traces) is dropped.

Reads only through the public API with the same three LANGFUSE_* keys
tracing uses. A failed pull is exit 1 — the job goes red, which is the P3
"export failed" alert by design; with ~90 days of retention a red week
loses nothing.
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from evals.cases import CASES, PLATFORM_LABELS
from evals.thresholds import COST_BAND_USD, LATENCY_CEILING_S, NO_PROFILE_FLOOR
from utils.run_record import DESCRIPTOR_FIELDS, RUN_RECORD_FIELDS, STRUCTURED_FIELDS, measurement_fields
from utils.tracing import ROOT_NAME

DEFAULT_DIR = Path("reports")
ROWS_FILE = "run_records.jsonl"
LATEST_FILE = "latest.json"
ROW_EXTRA_FIELDS = ("trace_id", "environment")
ROW_FIELDS: Tuple[str, ...] = RUN_RECORD_FIELDS + ROW_EXTRA_FIELDS
REPORTED_SOURCES = ("user", "canary", "smoke", "eval")
PAGE_SIZE = 100


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------

def week_bounds(now: Optional[datetime] = None, weeks_back: int = 1) -> Tuple[datetime, datetime, str]:
    """[Monday 00:00 UTC, next Monday 00:00 UTC) of the ISO week `weeks_back`
    weeks before the current one, and its label (YYYY-Www)."""
    now = now or datetime.now(timezone.utc)
    this_monday = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    start = this_monday - timedelta(weeks=weeks_back)
    end = start + timedelta(weeks=1)
    iso = start.isocalendar()
    return start, end, f"{iso[0]}-W{iso[1]:02d}"


# --------------------------------------------------------------------------
# Pulling rows
# --------------------------------------------------------------------------

def _tag_value(tags: Iterable[str], prefix: str) -> Optional[str]:
    for tag in tags or []:
        if isinstance(tag, str) and tag.startswith(prefix + ":"):
            return tag[len(prefix) + 1:]
    return None


def row_from_trace(trace: Any, scores: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """One exported row from a trace's run-record metadata and its scores.

    Fields come from the record's own classification: descriptors and
    structured fields from `metadata.run_record`, measurements from the
    scores (BOOLEAN scores back to bools). Anything else on the trace is
    ignored by construction. None when the trace carries no run record
    (a trace that is not a run).
    """
    metadata = getattr(trace, "metadata", None) or {}
    record = metadata.get("run_record") if isinstance(metadata, dict) else None
    if not isinstance(record, dict):
        return None
    tags = getattr(trace, "tags", None) or []
    source = record.get("source") or _tag_value(tags, "source") or "user"
    if source == "verify":
        return None
    values: Dict[str, Any] = {name: None for name in ROW_FIELDS}
    for name in DESCRIPTOR_FIELDS | STRUCTURED_FIELDS:
        if name in record:
            values[name] = record[name]
    by_name = {}
    for score in scores:
        name = getattr(score, "name", None)
        if name in measurement_fields():
            value = getattr(score, "value", None)
            if str(getattr(score, "data_type", "") or "").upper() == "BOOLEAN" and value is not None:
                value = bool(value)
            elif isinstance(value, float) and value.is_integer():
                value = int(value)  # scores come back as floats; a pool of 117.0 is a count
            by_name[name] = value
    values.update(by_name)
    values["source"] = source
    if not values.get("ts"):
        stamp = getattr(trace, "timestamp", None)
        values["ts"] = stamp.isoformat(timespec="seconds") if isinstance(stamp, datetime) else stamp
    values["trace_id"] = getattr(trace, "id", None)
    values["environment"] = getattr(trace, "environment", None)
    if not values.get("run_id"):
        values["run_id"] = f"trace:{values['trace_id']}"
    return values


def is_run(row: Mapping[str, Any]) -> bool:
    """A row is a run if it spent or measured something.

    The first dry run of this report (2026-09-13) listed 14 "user" runs of
    "Obscure Specialty in Remote Location" at $0.000 and 50 ms: the unit
    suite's mocked orchestrator, tracing for real because the development
    sandbox had gained the LANGFUSE_* keys. The suite no longer traces
    (conftest unsets the keys), but the system of record must not depend
    on every future environment getting that right — and a real run of the
    pipeline always fetches pages (credits, a planned page count, a pool)
    or spends on a model. A failure before any of that leaves a trace and
    a canary finding, not a row: half the mocked runs were mocked failures,
    and admitting "failed" here admitted them.
    """
    cost = row.get("cost_usd")
    credits = row.get("tavily_credits")
    if (isinstance(cost, (int, float)) and cost > 0) or (isinstance(credits, (int, float)) and credits > 0):
        return True
    return row.get("pool_raw") is not None or row.get("pages_planned") is not None


def is_pipeline(row: Mapping[str, Any]) -> bool:
    """Whether the orchestrator ran (vs a discovery-only fetch canary).

    The record states it (`extra.pipeline`); rows exported before that
    field existed fall back to the judge having scored anyone.
    """
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    if "pipeline" in extra:
        return bool(extra["pipeline"])
    return row.get("judge_applied") is not None or bool(row.get("shortlist_size"))


def _paged(fetch, **kwargs) -> Iterable[Any]:
    page = 1
    while True:
        result = fetch(page=page, limit=PAGE_SIZE, **kwargs)
        for item in result.data or []:
            yield item
        meta = getattr(result, "meta", None)
        total_pages = getattr(meta, "total_pages", None) or 1
        if page >= total_pages:
            return
        page += 1


def parse_environments(text: Optional[str]) -> List[str]:
    """A comma/space-separated list of trace environment labels → lowercase, deduped, in order; [] = all."""
    out: List[str] = []
    for part in (text or "").replace(",", " ").split():
        label = part.strip().lower()
        if label and label not in out:
            out.append(label)
    return out


def fetch_rows(client: Any, start: datetime, end: datetime,
               environments: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Every run in [start, end) as a row, via the public API.

    `environments` narrows the pull to traces whose environment label is in
    the list. Two Spaces (a private one and the public one) write to ONE
    Langfuse project with the same keys, and each repository's report must
    carry only its own Space's traffic — the private Space's searches must
    not be exported into the public repo's rows, nor the public Space's
    into the private one. The API filters server-side, and every row is
    checked AGAIN here: a filter the server ignored (an older deployment,
    a renamed parameter) would otherwise export the other Space's rows
    into a public file with nothing red to show for it. Empty = all, the
    pre-filter behaviour.
    """
    wanted = {label.lower() for label in (environments or []) if label}
    kwargs: Dict[str, Any] = dict(name=ROOT_NAME, from_timestamp=start, to_timestamp=end)
    if wanted:
        kwargs["environment"] = sorted(wanted)
    rows: List[Dict[str, Any]] = []
    for trace in _paged(client.api.trace.list, **kwargs):
        if wanted and str(getattr(trace, "environment", "") or "").lower() not in wanted:
            continue
        scores = list(_paged(client.api.scores.get_many, trace_id=trace.id))
        row = row_from_trace(trace, scores)
        if row is not None and is_run(row):
            rows.append(row)
    rows.sort(key=lambda r: str(r.get("ts") or ""))
    return rows


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def append_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Tuple[int, int]:
    """Append rows whose run_id is not already in the file. Returns (added, skipped)."""
    path = Path(path)
    existing = {r.get("run_id") for r in read_rows(path)}
    added = skipped = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row.get("run_id") in existing:
                skipped += 1
                continue
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            existing.add(row.get("run_id"))
            added += 1
    return added, skipped


def write_latest(directory: Path, label: str, added: int, total: int, now: Optional[datetime] = None,
                 environments: Optional[Sequence[str]] = None) -> Path:
    path = Path(directory) / LATEST_FILE
    path.write_text(json.dumps({
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "week": label,
        "rows_added": added,
        "rows_total": total,
        "environments": list(environments or []),  # [] = every environment was exported
    }, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _cell(value: Any, fmt: Optional[str] = None) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if fmt and isinstance(value, (int, float)):
        return fmt.format(value)
    return str(value).replace("|", "\\|")


def _median(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(statistics.median(nums), 3) if nums else None


def shortlist_stability(row: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """This run's shortlist against the previous pipeline run of the same case.

    Rolling, noisy by design: the judge and critic sample, so a week-over-
    week wobble is expected and a slide is the signal. Uses the ordered name
    hashes the record carries; the previous run is the latest EARLIER row
    for the case that carried hashes. None when there is no earlier run.
    """
    current = (row.get("extra") or {}).get("shortlist_hashes") if isinstance(row.get("extra"), dict) else None
    if not current or not row.get("case_id"):
        return None
    earlier = [
        r for r in history
        if r.get("case_id") == row.get("case_id") and r.get("run_id") != row.get("run_id")
        and str(r.get("ts") or "") < str(row.get("ts") or "")
        and isinstance(r.get("extra"), dict) and r["extra"].get("shortlist_hashes")
    ]
    if not earlier:
        return None
    previous = max(earlier, key=lambda r: str(r.get("ts") or ""))
    prior = previous["extra"]["shortlist_hashes"]
    shared = len(set(current) & set(prior))
    same_slot = sum(1 for a, b in zip(current, prior) if a == b)
    return {"previous_run_id": previous.get("run_id"), "previous_ts": previous.get("ts"),
            "size": len(current), "shared": shared, "same_slot": same_slot}


def _stability_cell(row: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> str:
    stability = shortlist_stability(row, history)
    if stability is None:
        return "first run" if (row.get("extra") or {}).get("shortlist_hashes") else "—"
    return f"{stability['shared']}/{stability['size']} shared · {stability['same_slot']} same slot"


def _pct(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{int(float(value) * 100 + 0.5)}%"  # half-up: 12.5% reads as 13%, not the format spec's 12%


def _latest_per_case(rows: Sequence[Mapping[str, Any]], source: str, pipeline: bool) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("source") != source or is_pipeline(row) != pipeline:
            continue
        case_id = row.get("case_id")
        if case_id and (case_id not in out or str(row.get("ts")) > str(out[case_id].get("ts"))):
            out[case_id] = row
    return out


def environments_scope(environments: Optional[Sequence[str]]) -> str:
    return (f"Traces filtered to environment(s) {', '.join(environments)}." if environments
            else "All environments.")


def render_report(label: str, start: datetime, end: datetime, rows: Sequence[Mapping[str, Any]],
                  generated_at: Optional[datetime] = None, history: Optional[Sequence[Mapping[str, Any]]] = None,
                  environments: Optional[Sequence[str]] = None) -> str:
    when = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    out: List[str] = [
        f"# CareCompass weekly report — {label}",
        "",
        f"Runs from {start:%Y-%m-%d} to {(end - timedelta(days=1)):%Y-%m-%d} (UTC), generated {when}. "
        f"Aggregates only; the rows are in `{ROWS_FILE}`. {environments_scope(environments)}",
        "",
        "## Runs",
        "",
        "| Source | Environment | Runs | Median cost | Median latency | Median pool | Median shortlist |",
        "|---|---|---|---|---|---|---|",
    ]
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("source")), str(row.get("environment"))), []).append(row)
    if not groups:
        out.append("| — | — | 0 | — | — | — | — |")
    for (source, env), group in sorted(groups.items()):
        out.append(
            f"| {source} | {env} | {len(group)} | {_cell(_median(r.get('cost_usd') for r in group), '${:.3f}')} "
            f"| {_cell(_median(r.get('latency_s') for r in group), '{:.0f} s')} "
            f"| {_cell(_median(r.get('pool_raw') for r in group), '{:.0f}')} "
            f"| {_cell(_median(r.get('shortlist_size') for r in group), '{:.0f}')} |"
        )
    out += ["", "## Live searches (source: user)", ""]
    live = [r for r in rows if r.get("source") == "user"]
    if live:
        out += [
            "| Time (UTC) | Specialty | City, ST | ZIP | Radius | Weights R / L / E | From | Pool | Shortlist | Cost | Latency |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in live:
            extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
            weights = extra.get("weights") if isinstance(extra.get("weights"), dict) else {}
            client = extra.get("client") if isinstance(extra.get("client"), dict) else {}
            where = " / ".join(x for x in (client.get("country"), client.get("region")) if x) or "—"
            w = " / ".join(_cell(weights.get(k), "{:.2f}") for k in ("rating", "location", "experience"))
            stamp = str(row.get("ts") or "")[:16].replace("T", " ")
            out.append(
                f"| {_cell(stamp)} | {_cell(row.get('specialty'))} | {_cell(row.get('city'))}, {_cell(row.get('state'))} "
                f"| {_cell(row.get('zip_present'))} | {_cell(row.get('radius_miles'))} mi | {w} | {where} "
                f"| {_cell(row.get('pool_raw'))} | {_cell(row.get('shortlist_size'))} | {_cell(row.get('cost_usd'), '${:.3f}')} "
                f"| {_cell(row.get('latency_s'), '{:.0f} s')} |"
            )
        out += ["", "\"From\" is the visitor's country / region as the hosting proxy reports it — never an address. "
                    "Weights are the search's own Nearby / Ratings / Experience settings, normalized."]
    else:
        out.append("No live (user) searches recorded this week.")
    out += ["", "## Tier A fetch canaries (latest run of the week per case)", ""]
    latest_a = _latest_per_case(rows, "canary", pipeline=False)
    if latest_a:
        out += [
            "| Case | Pool | Floor | Rows hg / wm / vi | Pages fetched / planned | Empty | Fallback | Ring | Credits |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for case in CASES:
            row = latest_a.get(case.case_id)
            if row is None:
                continue
            rows_cell = " / ".join(_cell(row.get(f"rows_{s}")) for s in ("hg", "wm", "vi"))
            out.append(
                f"| {case.case_id} | {_cell(row.get('pool_raw'))} | ≥ {case.pool_floor} | {rows_cell} "
                f"| {_cell(row.get('pages_fetched'))} / {_cell(row.get('pages_planned'))} | {_cell(row.get('empty_bodies'))} "
                f"| {_cell(row.get('fallback_fired'))} | {_cell(row.get('ring_fired'))} | {_cell(row.get('tavily_credits'))} |"
            )
    else:
        out.append("No Tier A canary runs this week.")
    out += ["", "## Pipeline runs (Tier B canaries and smoke checks)", ""]
    pipeline = [r for r in rows if r.get("source") in ("canary", "smoke") and is_pipeline(r)]
    all_history = list(history or []) + list(rows)
    if pipeline:
        out += [
            "| Case | Source | Pool | Enriched | no_profile_found | Coverage hg / wm / vi | Shortlist | Judge applied | Critic shards failed | Cost | Latency |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in pipeline:
            cov = " / ".join(_cell(row.get(f"coverage_{s}")) for s in ("hg", "wm", "vi"))
            out.append(
                f"| {_cell(row.get('case_id'))} | {_cell(row.get('source'))} | {_cell(row.get('pool_raw'))} | {_cell(row.get('n_enriched'))} "
                f"| {_cell(row.get('n_no_profile_found'))} | {cov} | {_cell(row.get('shortlist_size'))} | {_cell(row.get('judge_applied'))} "
                f"| {_cell(row.get('critic_shards_failed'))} | {_cell(row.get('cost_usd'), '${:.3f}')} | {_cell(row.get('latency_s'), '{:.0f} s')} |"
            )
        out += [
            "",
            f"Bands: cost ${COST_BAND_USD[0]:.2f}–${COST_BAND_USD[1]:.2f}, latency ≤ {LATENCY_CEILING_S:.0f} s, "
            f"no_profile_found < {NO_PROFILE_FLOOR} of the budget; platforms {', '.join(PLATFORM_LABELS.values())}.",
            "",
            "### Judge and critic compliance (from recorded outputs — no new model calls)",
            "",
            "| Case | Source | Evidence present | Neutral band | Conditional verdicts | Judge findings | Shortlist vs previous run |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in pipeline:
            out.append(
                f"| {_cell(row.get('case_id'))} | {_cell(row.get('source'))} | {_pct(row.get('evidence_present_rate'))} "
                f"| {_pct(row.get('neutral_band_rate'))} | {_pct(row.get('conditional_rate'))} | {_cell(row.get('judge_findings'))} "
                f"| {_stability_cell(row, all_history)} |"
            )
        out += [
            "",
            "Evidence present = judge rubric entries citing evidence; neutral band = entries reading \"no evidence\"; "
            "conditional = the critic's conditional verdicts over researched providers; judge findings = providers the critic "
            "flagged a citation on. Shortlist stability compares ordered name hashes with the previous pipeline run of the same "
            "case — the judge and critic sample, so a wobble is expected and a slide is the signal.",
        ]
    else:
        out.append("No pipeline canary or smoke runs this week.")
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _client(env: Mapping[str, str]) -> Any:
    from langfuse import Langfuse

    # Stripped, like utils.tracing does: the public repo's LANGFUSE_BASE_URL
    # variable arrived with a trailing space (visible in the first armed
    # run's log), which the canaries survived only because tracing strips —
    # passed raw, the host " " would have failed every API call here and
    # turned the first Monday report red.
    return Langfuse(
        public_key=env["LANGFUSE_PUBLIC_KEY"].strip(), secret_key=env["LANGFUSE_SECRET_KEY"].strip(),
        base_url=env["LANGFUSE_BASE_URL"].strip(), tracing_enabled=False,
    )


def generate(client: Any, out_dir: Path, weeks_back: int = 1, now: Optional[datetime] = None,
             dry_run: bool = False, environments: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    environments = list(environments or [])
    start, end, label = week_bounds(now, weeks_back)
    rows = fetch_rows(client, start, end, environments)
    history = read_rows(Path(out_dir) / ROWS_FILE)
    text = render_report(label, start, end, rows, now, history=history, environments=environments)
    result: Dict[str, Any] = {"week": label, "start": start, "end": end, "rows": len(rows), "report": text,
                              "environments": environments}
    if dry_run:
        return result
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    added, skipped = append_rows(out_dir / ROWS_FILE, rows)
    (out_dir / f"{label}.md").write_text(text, encoding="utf-8")
    total = len(read_rows(out_dir / ROWS_FILE))
    write_latest(out_dir, label, added, total, now, environments)
    result.update(added=added, skipped=skipped, total=total, report_path=str(out_dir / f"{label}.md"))
    return result


def main(argv: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None) -> int:
    import os

    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description="Export the week's run records and render the report")
    parser.add_argument("--weeks-back", type=int, default=1, help="1 = the last complete ISO week (default); 0 = this week so far")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="print the report; write nothing")
    parser.add_argument("--environments", default=None,
                        help="comma-separated trace environment labels to export (default: the REPORT_ENVIRONMENTS "
                             "variable; unset or empty = every environment)")
    args = parser.parse_args(argv)
    environments = parse_environments(args.environments if args.environments is not None
                                      else env.get("REPORT_ENVIRONMENTS"))
    missing = [k for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL") if not env.get(k)]
    if missing:
        print(f"weekly report: missing {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        result = generate(_client(env), args.out_dir, args.weeks_back, dry_run=args.dry_run, environments=environments)
    except Exception as exc:  # the red job IS the alert
        print(f"weekly report: export failed — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(result["report"])
    else:
        print(f"week {result['week']}: {result['rows']} run(s) pulled, {result['added']} row(s) added "
              f"({result['skipped']} already exported), {result['total']} total → {result['report_path']} "
              f"[{environments_scope(environments)}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
