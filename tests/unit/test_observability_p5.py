"""Observability P5: rolling shortlist stability and the compliance columns.

The record carries the shortlist as ORDERED HASHES of normalized names —
enough for the weekly report to say "4 of 5 shared with the previous run,
3 in the same slot" and nothing a reader could turn back into a doctor —
and the report's compliance table is computed from recorded outputs only
(plan D9: no model call is ever spent to measure a model).
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from evals import weekly_report
from utils.provider_key import normalized_name
from utils.run_record import RUN_RECORD_FIELDS, _shortlist_hashes, build_run_record

NOW = datetime(2026, 9, 13, 13, 0, tzinfo=timezone.utc)


def h(name):
    return hashlib.sha256(normalized_name(name).encode("utf-8")).hexdigest()[:12]


class TestShortlistHashes:
    def test_hashes_are_ordered_normalized_and_never_a_name(self):
        hashes = _shortlist_hashes([{"name": "Dr. Jane Kim, MD"}, {"name": "Kim, Jane"}, {"no": "name"}, "junk", {"name": ""}])
        assert hashes == [h("Dr. Jane Kim, MD"), h("Kim, Jane")]
        assert hashes[0] == hashes[1], "normalization: credentials and order do not change identity"
        assert all(len(x) == 12 and "kim" not in x for x in hashes)
        assert "shortlist_hashes" not in RUN_RECORD_FIELDS, "it rides in extra, not as a field of its own"

    def test_the_record_carries_them(self):
        state = {
            "run_id": "r", "run_source": "canary", "case_id": "c", "specialty": "Neurology", "location": "Chandler, AZ",
            "preferences": {}, "gathered_data": {}, "scored_providers": {}, "validation_results": {},
            "final_recommendations": [{"name": "Dr. A"}, {"name": "Dr. B"}], "execution_log": [], "error_messages": [],
            "workflow_summary": {},
        }
        config = SimpleNamespace(TAVILY_MODE="extract", GATHERER_MODEL="m", JUDGE_MODEL="j", CRITIC_MODEL="c",
                                 MAX_PROVIDERS_TO_ENRICH=8, DEFAULT_SEARCH_RADIUS=25)
        record = build_run_record(state, config, pipeline=True)
        assert record["extra"]["shortlist_hashes"] == [h("Dr. A"), h("Dr. B")] and record["shortlist_size"] == 2
        assert json.dumps(record["extra"])  # serializable for the trace and the rows


def row(run_id, ts, hashes, case_id="chandler-neurology", **over):
    r = {"run_id": run_id, "ts": ts.isoformat(), "case_id": case_id, "source": "canary", "environment": "ci",
         "extra": {"pipeline": True, "shortlist_hashes": hashes}, "pool_raw": 117, "n_enriched": 8, "n_no_profile_found": 0,
         "coverage_hg": 2, "coverage_wm": 8, "coverage_vi": 8, "shortlist_size": len(hashes), "judge_applied": 8,
         "critic_shards_failed": 0, "cost_usd": 0.44, "latency_s": 47.0,
         "evidence_present_rate": 0.9583, "neutral_band_rate": 0.0417, "conditional_rate": 0.125, "judge_findings": 2}
    r.update(over)
    return r


class TestStability:
    def test_compares_with_the_latest_earlier_run_of_the_same_case(self):
        week1 = row("w1", NOW - timedelta(days=14), ["a", "b", "c", "d", "e"])
        week2 = row("w2", NOW - timedelta(days=7), ["a", "b", "x", "d", "y"])
        other = row("o", NOW - timedelta(days=3), ["a", "b", "c", "d", "e"], case_id="phoenix-cardiology")
        week3 = row("w3", NOW, ["b", "a", "x", "z", "e"])
        history = [week1, week2, other]
        stability = weekly_report.shortlist_stability(week3, history + [week3])
        assert stability == {"previous_run_id": "w2", "previous_ts": week2["ts"], "size": 5, "shared": 3, "same_slot": 1}
        assert weekly_report.shortlist_stability(week1, history) is None, "no earlier run"
        assert weekly_report.shortlist_stability(row("n", NOW, []), history) is None, "no shortlist, nothing to compare"
        assert weekly_report._stability_cell(week3, history) == "3/5 shared · 1 same slot"
        assert weekly_report._stability_cell(week1, []) == "first run"
        assert weekly_report._stability_cell({"extra": {}}, []) == "—"

    def test_report_renders_compliance_and_stability_from_history(self, tmp_path):
        previous = row("prev", NOW - timedelta(days=7), ["a", "b", "c", "d", "e"])
        current = row("cur", NOW, ["a", "b", "c", "q", "e"])
        start, end, label = weekly_report.week_bounds(NOW, 0)
        text = weekly_report.render_report(label, start, end, [current], NOW, history=[previous])
        assert "### Judge and critic compliance" in text
        assert "| chandler-neurology | canary | 96% | 4% | 13% | 2 | 4/5 shared · 4 same slot |" in text
        alone = weekly_report.render_report(label, start, end, [current], NOW, history=[])
        assert "| 2 | first run |" in alone
        # generate() feeds the rows file back in as history, so week-over-week works across job runs.
        out = tmp_path / "reports"
        out.mkdir()
        (out / weekly_report.ROWS_FILE).write_text(json.dumps(previous) + "\n")
        trace = SimpleNamespace(id="t-cur", timestamp=NOW, tags=["source:canary"], environment="ci",
                                metadata={"run_record": {"run_id": "cur", "source": "canary", "case_id": "chandler-neurology",
                                                          "ts": NOW.isoformat(), "extra": current["extra"]}})
        scores = [SimpleNamespace(name=k, value=float(current[k]), data_type="NUMERIC", trace_id="t-cur")
                  for k in ("pool_raw", "cost_usd", "shortlist_size", "judge_applied", "evidence_present_rate")]
        api = SimpleNamespace(
            trace=SimpleNamespace(list=lambda **kw: SimpleNamespace(data=[trace] if kw["page"] == 1 else [], meta=SimpleNamespace(total_pages=1))),
            scores=SimpleNamespace(get_many=lambda **kw: SimpleNamespace(data=scores, meta=SimpleNamespace(total_pages=1))),
        )
        result = weekly_report.generate(SimpleNamespace(api=api), out, weeks_back=0, now=NOW)
        assert result["added"] == 1 and "4/5 shared · 4 same slot" in result["report"]
