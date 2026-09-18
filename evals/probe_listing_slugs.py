"""Probe every allowlisted specialty's city-listing URL on all three platforms.

A platform's specialty vocabulary is its own. "Cardiology" is
`cardiovascular-disease` on two of the three, primary care is
`family-practice` on one, and a slug the platform does not use returns a
~1.3 KB marketing stub rather than an error — so a wrong slug looks exactly
like a city with no providers. That is invisible in aggregate and total for
the affected specialty: discovery fetches one page fewer, every run, and no
surface says why.

Network tool, never part of the test suite. It costs Tavily credits (1 per 5
URLs), so it is run deliberately and its OUTPUT is what ships: the slug tables
in `utils/platform_urls.py`, pinned by `tests/unit/test_platform_urls.py`.

    python -m evals.probe_listing_slugs                    # every specialty
    python -m evals.probe_listing_slugs --specialty cardiology urology
    python -m evals.probe_listing_slugs --city "Chandler, AZ"
    python -m evals.probe_listing_slugs --url https://www.vitals.com/x/az/chandler

Reference cities default to two metros, because the question is whether the
SLUG works, not whether the city has doctors: a small town legitimately has no
page on some platforms, which would read as a broken slug.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.listing_parser import listing_result_count, parse_listing  # noqa: E402
from utils.platform_urls import (  # noqa: E402
    healthgrades_listing_url,
    vitals_listing_url,
    webmd_listing_url,
)
from utils.security import InputValidator  # noqa: E402

# Two metros on opposite coasts. A slug that works in one and not the other is
# a finding in itself; a slug that works in neither is the real target.
REFERENCE_CITIES = ("Chandler, AZ", "Chattanooga, TN")

# Below this, a body is the platform's generic marketing page rather than a
# directory. Measured: the stub served for an unknown vitals slug is 1,326
# chars in every city tried, while the thinnest REAL listing observed was
# ~4,000 with three entries.
_STUB_CHARS = 2000

BUILDERS = {
    "healthgrades": healthgrades_listing_url,
    "webmd": webmd_listing_url,
    "vitals": vitals_listing_url,
}


def _extract(urls: List[str], api_key: str) -> Dict[str, str]:
    """{url: body} for one batch, mirroring the agent's own fetch settings."""
    import requests

    bodies: Dict[str, str] = {}
    for start in range(0, len(urls), 20):
        batch = urls[start:start + 20]
        response = requests.post(
            "https://api.tavily.com/extract",
            json={"urls": batch, "extract_depth": "basic", "format": "markdown"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180,
        )
        response.raise_for_status()
        for result in response.json().get("results", []):
            bodies[result["url"]] = result.get("raw_content") or ""
    return bodies


def classify(url: str, body: str) -> Dict[str, Any]:
    """What this URL actually served: a directory, a stub, or nothing.

    `rows` is the deterministic parser's own read, so the verdict answers the
    question the pipeline asks rather than a proxy for it — a page that looks
    like a directory but parses to zero rows is just as useless.
    """
    rows = parse_listing(url, body) if body else []
    stated = listing_result_count(url, body) if body else None
    if not body:
        verdict = "no_body"
    elif len(body) < _STUB_CHARS:
        verdict = "stub"
    elif rows or stated:
        verdict = "ok"
    else:
        verdict = "unreadable"
    return {
        "url": url, "chars": len(body), "rows": len(rows),
        "stated_count": stated, "verdict": verdict,
    }


def probe(specialties: List[str], cities: List[str], api_key: str) -> List[Dict[str, Any]]:
    planned: List[Dict[str, Any]] = []
    for specialty in specialties:
        for city_state in cities:
            city, _, state = city_state.partition(",")
            for platform, builder in BUILDERS.items():
                url = builder(specialty, city.strip(), state.strip())
                if not url:
                    planned.append({
                        "specialty": specialty, "city": city_state, "platform": platform,
                        "url": None, "verdict": "no_url", "chars": 0, "rows": 0,
                        "stated_count": None,
                    })
                    continue
                planned.append({
                    "specialty": specialty, "city": city_state,
                    "platform": platform, "url": url,
                })
    urls = sorted({row["url"] for row in planned if row.get("url")})
    print(f"{len(urls)} distinct URL(s) ≈ {-(-len(urls) // 5)} Tavily credit(s)", file=sys.stderr)
    bodies = _extract(urls, api_key)
    for row in planned:
        if row.get("url"):
            row.update(classify(row["url"], bodies.get(row["url"], "")))
    return planned


def report(rows: List[Dict[str, Any]]) -> None:
    by_key: Dict[str, Dict[str, str]] = {}
    for row in rows:
        by_key.setdefault(row["specialty"], {})[
            f"{row['platform']}@{row['city']}"] = row["verdict"]
    platforms = list(BUILDERS)
    cities = sorted({row["city"] for row in rows})
    header = ["specialty"] + [f"{p[:2]}/{c.split(',')[0][:6]}" for c in cities for p in platforms]
    print(" | ".join(header))
    for specialty in sorted(by_key):
        line = [f"{specialty:<20}"]
        for city in cities:
            for platform in platforms:
                line.append(f"{by_key[specialty].get(f'{platform}@{city}', '-'):<10}")
        print(" | ".join(line))
    bad = [r for r in rows if r.get("verdict") not in ("ok", None)]
    print(f"\n{len(bad)} of {len(rows)} probes did not serve a readable directory", file=sys.stderr)
    for row in sorted(bad, key=lambda r: (r["specialty"], r["platform"])):
        print(f"  {row['verdict']:<11} {row['specialty']:<20} {row['platform']:<13} "
              f"{row.get('chars', 0):>6} chars  {row.get('url')}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specialty", nargs="*", default=None)
    parser.add_argument("--city", nargs="*", default=list(REFERENCE_CITIES))
    parser.add_argument("--url", nargs="*", default=None,
                        help="classify these URLs directly, bypassing the builders")
    parser.add_argument("--out", default="evals/out/listing_slugs.json")
    args = parser.parse_args(argv)

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        print("TAVILY_API_KEY is not set", file=sys.stderr)
        return 2

    if args.url:
        bodies = _extract(args.url, api_key)
        rows = [classify(url, bodies.get(url, "")) for url in args.url]
        for row in rows:
            print(json.dumps(row))
        return 0

    specialties = args.specialty or sorted(InputValidator.ALLOWED_SPECIALTIES)
    rows = probe(specialties, args.city, api_key)
    report(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
