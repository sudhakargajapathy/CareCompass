"""Build, refresh and inspect the golden set — the runbook's instrument for
"a page reads wrong".

  --build            discover the case's providers on the production path,
                     fetch their profile pages, and write the answer key +
                     minimized fixtures (+ full bodies privately)
  --refresh PID      re-fetch one provider's pages and rewrite its key
                     entries and fixtures (the "world drift" follow-up)
  --inspect URL      fetch one page and print what the parser reads, the
                     minimized copy, its hash, and whether the copy agrees

The saved copy changes only through a commit a person reviews: this
script writes files; nothing merges them. ~3 credits for discovery plus
1 per 5 profile URLs.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from evals import golden
from evals.cases import case_by_id
from utils import tracing

logger = logging.getLogger(__name__)

DEFAULT_CASE = "chandler-neurology"
DEFAULT_N = 15
DEFAULT_MIN_PLATFORMS = 2


def _agent():
    from agents.data_gatherer import create_data_gatherer

    return create_data_gatherer()


def candidates_from_discovery(providers: Sequence[Dict[str, Any]], n: int, min_platforms: int) -> List[Dict[str, Any]]:
    """Providers with profile URLs on the most platforms first, name order second."""
    rows = []
    for p in providers:
        urls = {golden.platform_of(u): u for u in (p.get("platform_profile_urls") or {}).values() if golden.platform_of(u)}
        if len(urls) >= min_platforms and p.get("name"):
            rows.append((-len(urls), str(p["name"]), p, urls))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [{"name": name, "specialty": p.get("specialty"), "urls": urls} for _, name, p, urls in rows[:n]]


def write_provider(
    name: str, specialty: Optional[str], urls: Dict[str, str], bodies: Dict[str, str],
    fixture_dir: Path, bodies_dir: Path, now: Optional[datetime],
    existing_pages: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """One answer-key provider from fetched bodies; None when no page parsed a number.

    Identity is gated HERE too: a page naming someone else, or a URL whose
    slug contradicts the name, never enters the key — a stranger's stars as
    the answer would grade the parser right for reading the wrong person.
    Existing entries are MERGED: the fetched template replaces its own copy
    and other templates' copies stay (healthgrades serves A or B per fetch),
    and a URL whose fetch came back empty keeps whatever it had.
    """
    pid = golden.provider_id(name)
    kept: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {
        (str(e.get("url")), e.get("rating_pattern")): dict(e) for e in (existing_pages or [])
    }
    for platform in ("healthgrades", "webmd", "vitals"):
        url = urls.get(platform)
        text = bodies.get(url or "")
        if not url or not text or len(text.strip()) < golden.MIN_BODY_CHARS:
            continue
        entry = golden.page_entry(url, text, now)
        if entry.get("rating") is None and entry.get("review_count") is None:
            continue  # a page the parser reads nothing from is not an answer
        if not golden.identity_agrees(url, name, entry.get("page_provider_name")):
            logger.warning("golden: %s page for %s names %r — skipped", platform, name, entry.get("page_provider_name"))
            continue
        pattern = entry.get("rating_pattern")
        fixture = golden.fixture_path(platform, pid, fixture_dir, pattern)
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(golden.minimize(url, text), encoding="utf-8")
        body = golden.body_path(platform, pid, bodies_dir)
        body.parent.mkdir(parents=True, exist_ok=True)
        body.write_text(text, encoding="utf-8")
        entry["fixture"] = str(fixture)
        kept[(url, pattern)] = entry
    pages = sorted(kept.values(), key=lambda e: (("healthgrades", "webmd", "vitals").index(e["platform"]) if e.get("platform") in ("healthgrades", "webmd", "vitals") else 9, str(e.get("rating_pattern"))))
    if not pages:
        return None
    return {"provider_id": pid, "stated_name": name, "specialty_label": specialty, "pages": pages}


def build(
    case_id: str = DEFAULT_CASE, n: int = DEFAULT_N, min_platforms: int = DEFAULT_MIN_PLATFORMS,
    key_path: Path = golden.KEY_PATH, fixture_dir: Path = golden.FIXTURE_DIR, bodies_dir: Path = golden.BODIES_DIR,
    agent_factory: Callable[[], Any] = _agent, now: Optional[datetime] = None,
) -> Dict[str, Any]:
    case = case_by_id(case_id)
    if case is None:
        raise ValueError(f"unknown case {case_id!r}")
    agent = agent_factory()
    result = agent.gather_providers(case.specialty, case.location, enrich=False)
    chosen = candidates_from_discovery(result.get("providers") or [], n, min_platforms)
    urls = [u for c in chosen for u in c["urls"].values()]
    pages = agent._extract_pages(urls, purpose="golden set build", stage="discovery")
    bodies = {p["url"]: p.get("raw_content") or "" for p in pages if p.get("url")}
    providers = []
    for c in chosen:
        entry = write_provider(c["name"], c["specialty"], c["urls"], bodies, fixture_dir, bodies_dir, now)
        if entry:
            providers.append(entry)
    key = {
        "schema_version": golden.SCHEMA_VERSION,
        "case_id": case.case_id,
        "specialty": case.specialty,
        "location": case.location,
        "as_of": (now or datetime.now(timezone.utc)).date().isoformat(),
        "checked_by": "evals/refresh_fixtures.py --build (live fetch, production parsers)",
        "git_sha": tracing.git_sha(),
        "providers": providers,
    }
    golden.save_key(key, key_path)
    return key


def refresh(
    pid: str, key_path: Path = golden.KEY_PATH, fixture_dir: Path = golden.FIXTURE_DIR,
    bodies_dir: Path = golden.BODIES_DIR, agent_factory: Callable[[], Any] = _agent, now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    key = golden.load_key(key_path)
    if key is None:
        raise FileNotFoundError(f"no answer key at {key_path}")
    provider = next((p for p in key["providers"] if p["provider_id"] == pid), None)
    if provider is None:
        return None
    urls = {page["platform"]: page["url"] for page in provider["pages"]}
    agent = agent_factory()
    pages = agent._extract_pages(list(urls.values()), purpose="golden set refresh", stage="discovery")
    bodies = {p["url"]: p.get("raw_content") or "" for p in pages if p.get("url")}
    entry = write_provider(provider["stated_name"], provider.get("specialty_label"), urls, bodies, fixture_dir, bodies_dir, now,
                           existing_pages=provider["pages"])
    if entry is None:
        return None
    provider["pages"] = entry["pages"]
    key["as_of"] = (now or datetime.now(timezone.utc)).date().isoformat()
    golden.save_key(key, key_path)
    return provider


def inspect(url: str, agent_factory: Callable[[], Any] = _agent) -> List[str]:
    agent = agent_factory()
    pages = agent._extract_pages([url], purpose="golden set inspect", stage="discovery")
    text = next((p.get("raw_content") or "" for p in pages if p.get("url") == url), "")
    if not text:
        return [f"{url}: no body returned"]
    entry = golden.page_entry(url, text)
    mini = golden.minimize(url, text)
    lines = [f"{url}", f"  platform {entry['platform']}  chars {entry['fetched_chars']}  sha256 {entry['content_sha256'][:16]}…"]
    lines += [f"  {k}: {entry.get(k)}" for k in golden.FIELDS]
    lines.append(f"  minimized: {entry['minimized_chars']} chars, agrees with full parse: {entry['minimized_agrees']}")
    lines.append("  ---- minimized copy ----")
    lines += ["  | " + l for l in mini.splitlines()]
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build / refresh / inspect the golden set")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--refresh", metavar="PROVIDER_ID")
    group.add_argument("--inspect", metavar="URL")
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--min-platforms", type=int, default=DEFAULT_MIN_PLATFORMS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.inspect:
        for line in inspect(args.inspect):
            print(line)
        return 0
    if args.refresh:
        provider = refresh(args.refresh)
        if provider is None:
            print(f"refresh: no provider {args.refresh!r} in the key, or no page parsed", file=sys.stderr)
            return 2
        print(f"refreshed {provider['provider_id']}: {len(provider['pages'])} page(s)")
        return 0
    try:
        key = build(args.case, args.n, args.min_platforms)
    except ValueError as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 2
    for p in key["providers"]:
        pages = ", ".join(f"{pg['platform']}{'/' + pg['rating_pattern'] if pg.get('rating_pattern') else ''} {pg.get('rating')}/{pg.get('review_count')}{'' if pg['minimized_agrees'] else ' (fixture disagrees)'}" for pg in p["pages"])
        print(f"{p['provider_id']:32s} {pages}")
    print(f"answer key: {len(key['providers'])} providers, {sum(len(p['pages']) for p in key['providers'])} pages → {golden.KEY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
