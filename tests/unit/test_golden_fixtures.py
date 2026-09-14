"""The offline golden-set gate: the production parsers must still read every
saved fixture to the answer key's numbers.

This is the CI half of eval family 1. The key and the fixtures were built
from LIVE pages (evals/refresh_fixtures.py --build, 2026-09-13, Chandler
neurology, 15 providers × their platforms); each fixture is the subset of
a fetched page's lines that reproduces the parser's read. A parser change
that stops reading a real page fails here before it ships — the failure
the register recorded when the first profile parsers, inferred from
listing markup, returned tenure and nothing else on real profiles.

Also the privacy gate: a fixture is a claim about BYTES the public repo
carries, so every line is checked against the first-person rule the
minimizer applies, and no fixture may exceed a size that would mean a
page body got through.
"""

import json
import re
from pathlib import Path

import pytest

from evals import golden

REPO = Path(__file__).resolve().parents[2]
KEY = REPO / "evals" / "golden" / "answer_key.json"
FIRST_PERSON = re.compile(r"(?i)\b(i|i'm|i've|my|me|we|our|us)\b")
MAX_FIXTURE_BYTES = 12_000


def _pages():
    if not KEY.exists():
        return []
    key = json.loads(KEY.read_text(encoding="utf-8"))
    return [
        pytest.param(provider, page, id=f"{provider['provider_id']}/{page['platform']}")
        for provider, page in golden.iter_pages(key)
    ]


@pytest.mark.parametrize("provider,page", _pages())
def test_fixture_still_parses_to_the_answer_key(provider, page):
    fixture = REPO / page["fixture"]
    assert fixture.exists(), f"fixture missing: {fixture}"
    text = fixture.read_text(encoding="utf-8")
    fields = golden.parse_fields(page["url"], text)
    for name in golden.NUMERIC_FIELDS + ("page_provider_name", "rating_pattern"):
        assert fields.get(name) == page.get(name), f"{name}: parser reads {fields.get(name)!r}, key says {page.get(name)!r}"
    if page.get("minimized_agrees"):
        assert fields.get("location") == page.get("location")
    assert golden.name_overlap(fields.get("page_provider_name"), provider["stated_name"]) >= golden.NAME_OVERLAP_FLOOR


@pytest.mark.parametrize("provider,page", _pages())
def test_fixture_carries_no_review_prose(provider, page):
    text = (REPO / page["fixture"]).read_text(encoding="utf-8")
    assert len(text.encode("utf-8")) <= MAX_FIXTURE_BYTES
    for line in text.splitlines():
        structural = line.startswith("|") or "destination=" in line or line.startswith("#")
        if not structural:
            assert not FIRST_PERSON.search(line), f"first-person line in a public fixture: {line[:80]!r}"
            assert len(line) <= 200


def test_key_shape_and_coverage():
    if not KEY.exists():
        pytest.skip("no answer key built")
    key = json.loads(KEY.read_text(encoding="utf-8"))
    assert key["schema_version"] == golden.SCHEMA_VERSION and key["case_id"] == "chandler-neurology"
    providers = key["providers"]
    assert 10 <= len(providers) <= 20
    ids = [p["provider_id"] for p in providers]
    assert len(set(ids)) == len(ids)
    platforms = {pg["platform"] for _, pg in golden.iter_pages(key)}
    assert platforms == {"healthgrades", "webmd", "vitals"}
    assert all(pg["rating"] is not None or pg["review_count"] is not None for _, pg in golden.iter_pages(key))
    assert all(golden.platform_of(pg["url"]) == pg["platform"] for _, pg in golden.iter_pages(key))
