"""Fixture freshness: fetch every answer-key page live and grade it.

Weekly (bundled into the Tier A weekly run). ~1 credit per 5 URLs — the
15-provider key is ~9 credits. Full bodies go to `evals/out/golden_bodies/`
(gitignored; the workflow uploads `evals/out/` as a 90-day private
artifact); the grades go to `evals/out/golden.json` for the watcher, which
turns identity failures and template drift into P2 issues, missing pages
and reviews-needed into P3, and lists world drift as refresh proposals.
Measures only; always exits 0 (a key that is missing is a P1 for the
watcher, `canary_did_not_run` on the golden-set case).
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from evals import golden
from utils import tracing

logger = logging.getLogger(__name__)


def _default_fetcher() -> Callable[[List[str]], Dict[str, str]]:
    from agents.data_gatherer import create_data_gatherer

    agent = create_data_gatherer()

    def fetch(urls: List[str]) -> Dict[str, str]:
        pages = agent._extract_pages(urls, purpose="golden set", stage="discovery")
        return {p["url"]: p.get("raw_content") or "" for p in pages if p.get("url")}

    return fetch


def run(
    key_path: Path = golden.KEY_PATH,
    fixture_dir: Path = golden.FIXTURE_DIR,
    bodies_dir: Path = golden.BODIES_DIR,
    out_path: Path = golden.OUT_PATH,
    fetcher: Optional[Callable[[List[str]], Dict[str, str]]] = None,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    key = golden.load_key(key_path)
    if key is None:
        logger.error("No answer key at %s — build one with evals.refresh_fixtures --build", key_path)
        return None
    targets = list(golden.iter_urls(key))
    urls = [url for _, url, _ in targets]
    bodies = (fetcher or _default_fetcher())(urls)
    grades: List[golden.PageGrade] = []
    for provider, url, entries in targets:
        live = bodies.get(url)
        platform = str(entries[0].get("platform"))
        fixtures = {}
        for entry in entries:
            path = golden.fixture_path(platform, provider["provider_id"], fixture_dir, entry.get("rating_pattern"))
            if path.exists():
                fixtures[entry.get("rating_pattern")] = path.read_text(encoding="utf-8")
        grades.append(golden.grade_url(provider, entries, live, fixtures))
        if live:
            target = golden.body_path(platform, provider["provider_id"], bodies_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(live, encoding="utf-8")
    payload = {
        "tier": "golden",
        "ts": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "git_sha": tracing.git_sha(),
        "key_as_of": key.get("as_of"),
        "case_id": key.get("case_id"),
        "counts": golden.summarize_grades(grades),
        "pages": [g.as_dict() for g in grades],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def describe(payload: Dict[str, Any]) -> List[str]:
    lines = [f"golden set ({payload.get('case_id')}, key as of {payload.get('key_as_of')}): "
             + ", ".join(f"{k} {v}" for k, v in payload["counts"].items() if v)]
    for page in payload["pages"]:
        if page["status"] != "ok":
            lines.append(f"  {page['status']:15s} {page['provider_id']}/{page['platform']}: {'; '.join(page['notes'])}")
    return lines


def bodies_fetcher(directory: Path, key: Mapping[str, Any]) -> Callable[[List[str]], Dict[str, str]]:
    """Re-grade from saved bodies (a downloaded artifact) — no credits, the
    runbook's "diff the private body artifact" step."""
    by_url: Dict[str, Path] = {}
    for provider, url, entries in golden.iter_urls(key):
        by_url[url] = golden.body_path(str(entries[0].get("platform")), provider["provider_id"], directory)

    def fetch(urls: List[str]) -> Dict[str, str]:
        return {u: by_url[u].read_text(encoding="utf-8") for u in urls if u in by_url and by_url[u].exists()}

    return fetch


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Grade the golden set against live pages (measures; the watcher judges)")
    parser.add_argument("--key", type=Path, default=golden.KEY_PATH)
    parser.add_argument("--out", type=Path, default=golden.OUT_PATH)
    parser.add_argument("--from-bodies", type=Path, default=None, help="grade saved bodies from this directory instead of fetching")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fetcher = None
    if args.from_bodies:
        key = golden.load_key(args.key)
        if key is None:
            return 0
        fetcher = bodies_fetcher(args.from_bodies, key)
    payload = run(key_path=args.key, out_path=args.out, fetcher=fetcher,
                  bodies_dir=args.from_bodies if args.from_bodies else golden.BODIES_DIR)
    if payload is None:
        return 0
    for line in describe(payload):
        print(line)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
