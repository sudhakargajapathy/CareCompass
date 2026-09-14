# Run records and weekly reports

Written by the weekly-report job (`.github/workflows/weekly-report.yml`,
Mondays 06:00 UTC) from `evals/weekly_report.py`; public by design.

| File | What it is |
|---|---|
| `run_records.jsonl` | One line per run — the system of record. Every field of `utils.run_record.RUN_RECORD_FIELDS` (aggregates only: pool sizes, rows per platform, enrichment outcomes, judge/critic health, shortlist size, cost, credits, tokens, latency; the allowlisted inputs; the config snapshot) plus the trace id and environment. Append-only, deduped by `run_id`. |
| `YYYY-Www.md` | That ISO week's page: runs by source and environment, the week's live (user) searches — when, specialty, city, ZIP-present, radius, the three weights, and the visitor's country / region as the hosting proxy reports it — the fetch canaries against their floors, the pipeline runs' model-side health, per-source medians. |
| `latest.json` | The job's own stamp — when it last ran, which week, how many rows. The daily watcher raises a P3 "weekly report stale" issue when it is older than eight days. |

Nothing here names a provider, quotes a review, or carries a page: rows are
built from the run record on each trace and its scores, never from a
trace's input, output or observations. The `verify` source (account-check
traces) is excluded. A live-search row carries the search's own criteria
and, at most, the visitor's country / region, timezone and locale — never
an IP address: none is stored anywhere, and the salted pseudonym used to
count distinct visitors exists on the private trace only.
