"""The golden set: an answer key of real profile pages, minimized fixtures
that reproduce the parsers' reads, and the grader that separates world
drift from template drift.

What it answers. The parsers in `utils/profile_parser.py` were written off
FETCHED pages, and the register's lesson is that a page's shape is a
property of the fetched BYTES, not of the page in a browser. So the
question "do the parsers still read what they read last week?" needs a
saved copy of last week's bytes and a fresh fetch — and the grade must
say WHICH moved: the world (a doctor gained reviews; the parser still
reads the page; refresh the key) or the template (the saved copy still
parses, today's copy parses to nothing; the platform changed its markup
and the parser is blind). Identity is graded FIRST: a URL that now shows
a different name is a stranger's stars, whatever the numbers say.

What is public. The answer key carries facts the app already displays —
the doctor's name, the platform URL, the stated rating/count/tenure, the
practice address — and the fixtures carry only the LINES the parsers
read: the H1, the rating and count lines, the tenure line, the address
line, the Locations links, the comparison-table rows, a few section
headings. Review paragraphs never pass the minimizer (first-person prose
is refused by rule, and every kept line is short). Full page bodies are
written to `evals/out/golden_bodies/` — gitignored, uploaded by the
workflow as a 90-day private artifact — and never to the repository. A
fixture proves itself at build time: the minimized copy must parse to the
same fields as the full page, or the entry records that it does not.

Tolerances (the answer key labels are a floor, not a target): a review
count may only GROW; a rating may move up to 0.3 before it needs eyes;
tenure changes need eyes; a different address is flagged, never failed
(doctors move; the app copes); the name must agree by token overlap and
by URL slug.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from utils.listing_parser import slug_contradicts_name
from utils.profile_parser import parse_profile, strip_data_uris
from utils.provenance import source_domain
from utils.provider_key import normalize_name_tokens

KEY_PATH = Path("evals/golden/answer_key.json")
FIXTURE_DIR = Path("evals/fixtures")
BODIES_DIR = Path("evals/out/golden_bodies")
OUT_PATH = Path("evals/out/golden.json")
SCHEMA_VERSION = 1

FIELDS: Tuple[str, ...] = ("rating", "review_count", "years_experience", "page_provider_name", "location", "rating_pattern")
NUMERIC_FIELDS = ("rating", "review_count", "years_experience")
RATING_TOLERANCE = 0.3
MIN_BODY_CHARS = 200
NAME_OVERLAP_FLOOR = 0.5

PLATFORMS = {"healthgrades.com": "healthgrades", "webmd.com": "webmd", "vitals.com": "vitals"}
STATUSES = ("ok", "world_drift", "needs_review", "template_drift", "identity", "missing")


def platform_of(url: str) -> Optional[str]:
    domain = source_domain(url) or ""
    for needle, name in PLATFORMS.items():
        if needle in domain:
            return name
    return None


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def provider_id(name: str) -> str:
    """A stable slug from the name's identity tokens, in name order."""
    keep = normalize_name_tokens(name)
    tokens = [t for t in re.split(r"[^a-z0-9]+", str(name or "").lower()) if t and t in keep]
    return "-".join(tokens) or "unknown"


def parse_fields(url: str, text: str) -> Dict[str, Any]:
    """The comparable fields the production parser reads from a page."""
    found = parse_profile(url, text or "")
    return {k: found[k] for k in FIELDS if found.get(k) is not None}


def name_overlap(a: Optional[str], b: Optional[str]) -> float:
    ta, tb = normalize_name_tokens(a), normalize_name_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# --------------------------------------------------------------------------
# Minimizer
# --------------------------------------------------------------------------

_HEADING = re.compile(r"^#{1,4}\s")
_HEADING_KEEP = re.compile(
    r"(?i)(rating|review|experience|location|compare|overview|about|office|patient|faq|insurance|practice)")
_KEEP = re.compile(
    r"(?i)(\bratings?\b|\breviews?\b|star rating|likelihood of recommending|out of\s*5|"
    r"years?\s+(?:of\s+)?exp|get directions|destination=|tel:\s*\[|current profile|view profile|"
    r"#rating-overview|\b[A-Z]{2},?\s*\d{5}\b)")
_BARE_NUMBER = re.compile(r"^\d(?:\.\d+)?$")
_RATINGS_PAREN = re.compile(r"^\(\s*\d+\s+Ratings?\s*\)$", re.I)
_TABLE_ROW = re.compile(r"^\s*\|")
_FIRST_PERSON = re.compile(r"(?i)\b(i|i'm|i've|my|me|we|our|us)\b")
_LINE_CAP = 200
_STRUCTURAL_CAP = 800


def _structural(line: str) -> bool:
    return bool(_TABLE_ROW.match(line) or "destination=" in line or _HEADING.match(line))


def minimize(url: str, text: str) -> str:
    """The lines the parsers read, and nothing a reviewer wrote.

    Kept: the H1 (identity), section headings the parsers anchor on, any
    line naming a rating/count/tenure/address/directions/comparison cell,
    a bare number immediately above a "(N Ratings)" line (webmd's header
    pair spans two lines), and comparison-table rows. Refused: anything
    over 200 chars that is not structural, and any line in the first
    person — review prose is first-person, the platforms' own lines are
    not.
    """
    lines = [l.rstrip() for l in strip_data_uris(text or "").splitlines()]
    kept: List[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        keep = False
        if s.startswith("# "):
            keep = True
        elif _HEADING.match(s):
            keep = bool(_HEADING_KEEP.search(s))
        elif _TABLE_ROW.match(s):
            keep = True
        elif _BARE_NUMBER.match(s):
            nxt = next((lines[j].strip() for j in range(i + 1, min(i + 3, len(lines))) if lines[j].strip()), "")
            keep = bool(_RATINGS_PAREN.match(nxt))
        elif _KEEP.search(s):
            keep = True
        if not keep:
            continue
        if not _structural(s) and (len(s) > _LINE_CAP or _FIRST_PERSON.search(s)):
            continue
        if len(s) > _STRUCTURAL_CAP:
            continue
        kept.append(s)
    return "\n".join(kept) + ("\n" if kept else "")


def minimized_agrees(url: str, full_text: str, minimized_text: str) -> bool:
    """Does the minimized copy parse to the same fields as the full page?"""
    return parse_fields(url, full_text) == parse_fields(url, minimized_text)


# --------------------------------------------------------------------------
# Answer key
# --------------------------------------------------------------------------

def load_key(path: Path = KEY_PATH) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_key(key: Mapping[str, Any], path: Path = KEY_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(key, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def fixture_path(platform: str, pid: str, root: Path = FIXTURE_DIR, pattern: Optional[str] = None) -> Path:
    """One fixture per (page, template). healthgrades serves the SAME URL as
    template A or B on different fetches — the first freshness run saw 7 of
    15 flip within twenty minutes — so a URL can hold two saved copies, and
    each must keep parsing."""
    name = f"{pid}.{pattern}.md" if pattern else f"{pid}.md"
    return Path(root) / platform / name


def body_path(platform: str, pid: str, root: Path = BODIES_DIR) -> Path:
    return Path(root) / platform / f"{pid}.md"


def page_entry(url: str, text: str, now: Optional[datetime] = None) -> Dict[str, Any]:
    """One answer-key page from a fetched body: the parse, the hash, the
    minimized copy's agreement."""
    fields = parse_fields(url, text)
    mini = minimize(url, text)
    return {
        "platform": platform_of(url),
        "url": url,
        **{k: fields.get(k) for k in FIELDS},
        "content_sha256": content_hash(text),
        "fetched_chars": len(text or ""),
        "minimized_chars": len(mini),
        "minimized_agrees": minimized_agrees(url, text, mini),
        "as_of": (now or datetime.now(timezone.utc)).date().isoformat(),
    }


def iter_pages(key: Mapping[str, Any]):
    for provider in key.get("providers") or []:
        for page in provider.get("pages") or []:
            yield provider, page


def iter_urls(key: Mapping[str, Any]):
    """(provider, url, entries) — every saved template copy of one URL together."""
    for provider in key.get("providers") or []:
        by_url: Dict[str, List[Dict[str, Any]]] = {}
        for page in provider.get("pages") or []:
            by_url.setdefault(str(page.get("url")), []).append(page)
        for url, entries in by_url.items():
            yield provider, url, entries


def identity_agrees(url: str, stated_name: str, page_name: Optional[str]) -> bool:
    """The page is about the stated person: name tokens overlap and the URL slug does not contradict."""
    if page_name and stated_name and name_overlap(page_name, stated_name) < NAME_OVERLAP_FLOOR:
        return False
    return not (stated_name and slug_contradicts_name(url, stated_name))


# --------------------------------------------------------------------------
# Grader
# --------------------------------------------------------------------------

@dataclass
class PageGrade:
    provider_id: str
    platform: str
    url: str
    status: str
    expected: Dict[str, Any] = field(default_factory=dict)
    observed: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    content_sha256: Optional[str] = None
    fetched_chars: int = 0
    changed_bytes: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def grade_page(
    provider: Mapping[str, Any], page: Mapping[str, Any], live_text: Optional[str],
    fixture_text: Optional[str] = None,
) -> PageGrade:
    """Identity first, then the numbers against their tolerances.

      missing         the page came back empty or absent
      identity        the page names someone else (token overlap < 0.5 with
                      the stated name, or the URL slug contradicts it)
      template_drift  the parser reads nothing from today's copy while it
                      still reads the saved copy — the markup moved
      needs_review    a count went DOWN, a rating moved ≥ 0.3, or tenure
                      changed: not impossible, but a person decides
      world_drift     numbers moved within tolerance (count up, rating
                      < 0.3): refresh the key, no eyes needed
      ok              nothing moved
    """
    pid = str(provider.get("provider_id"))
    platform = str(page.get("platform") or platform_of(str(page.get("url"))) or "unknown")
    url = str(page.get("url"))
    expected = {k: page.get(k) for k in FIELDS if page.get(k) is not None}
    grade = PageGrade(provider_id=pid, platform=platform, url=url, status="ok", expected=expected)
    if not live_text or len(live_text.strip()) < MIN_BODY_CHARS:
        grade.status = "missing"
        grade.fetched_chars = len(live_text or "")
        grade.notes.append("empty or absent body")
        return grade
    grade.content_sha256 = content_hash(live_text)
    grade.fetched_chars = len(live_text)
    grade.changed_bytes = grade.content_sha256 != page.get("content_sha256")
    observed = parse_fields(url, live_text)
    grade.observed = observed

    stated = str(provider.get("stated_name") or "")
    page_name = observed.get("page_provider_name")
    if page_name and stated and name_overlap(page_name, stated) < NAME_OVERLAP_FLOOR:
        grade.status = "identity"
        grade.notes.append(f"page names {page_name!r}, key says {stated!r}")
        return grade
    if stated and slug_contradicts_name(url, stated):
        grade.status = "identity"
        grade.notes.append("URL slug contradicts the stated name")
        return grade

    had_pair = expected.get("rating") is not None or expected.get("review_count") is not None
    has_pair = observed.get("rating") is not None or observed.get("review_count") is not None
    if had_pair and not has_pair:
        stored_reads = bool(fixture_text) and any(
            parse_fields(url, fixture_text).get(k) is not None for k in ("rating", "review_count"))
        grade.status = "template_drift"
        grade.notes.append(
            "parser reads no rating or count from today's copy"
            + (" while the saved copy still parses" if stored_reads else " — and none from the saved copy either (parser regression?)"))
        return grade

    review: List[str] = []
    drift: List[str] = []
    exp_c, obs_c = expected.get("review_count"), observed.get("review_count")
    if exp_c is not None and obs_c is not None:
        if obs_c < exp_c:
            review.append(f"review_count went DOWN {exp_c} → {obs_c}")
        elif obs_c > exp_c:
            drift.append(f"review_count {exp_c} → {obs_c}")
    elif exp_c is not None and obs_c is None:
        drift.append(f"review_count {exp_c} → none")
    elif exp_c is None and obs_c is not None:
        drift.append(f"review_count none → {obs_c}")
    exp_r, obs_r = expected.get("rating"), observed.get("rating")
    if exp_r is not None and obs_r is not None:
        if abs(float(obs_r) - float(exp_r)) >= RATING_TOLERANCE:
            review.append(f"rating moved {exp_r} → {obs_r}")
        elif float(obs_r) != float(exp_r):
            drift.append(f"rating {exp_r} → {obs_r}")
    exp_y, obs_y = expected.get("years_experience"), observed.get("years_experience")
    if exp_y is not None and obs_y is not None and exp_y != obs_y:
        review.append(f"years_experience {exp_y} → {obs_y}")
    exp_a, obs_a = expected.get("location"), observed.get("location")
    if exp_a and obs_a and exp_a != obs_a:
        drift.append(f"address {exp_a!r} → {obs_a!r} (flagged, never failed)")
    if expected.get("rating_pattern") and observed.get("rating_pattern") and expected["rating_pattern"] != observed["rating_pattern"]:
        drift.append(f"template {expected['rating_pattern']} → {observed['rating_pattern']}")

    if review:
        grade.status = "needs_review"
        grade.notes.extend(review + drift)
    elif drift:
        grade.status = "world_drift"
        grade.notes.extend(drift)
    return grade


def grade_url(
    provider: Mapping[str, Any], entries: Sequence[Mapping[str, Any]], live_text: Optional[str],
    fixtures: Optional[Mapping[Optional[str], str]] = None,
) -> PageGrade:
    """Grade one URL against the saved copy of the template it served today.

    healthgrades serves template A or B per fetch; comparing a star-template
    fetch against a likelihood-template entry would report "count none →
    38" every other week. So the live parse picks the entry with the same
    `rating_pattern` (or the only entry, for platforms without templates);
    a template the key has never seen is a refresh proposal ("add a copy"),
    not drift. Identity and emptiness are graded before any of that.
    """
    fixtures = fixtures or {}
    first = entries[0]
    if not live_text or len(live_text.strip()) < MIN_BODY_CHARS:
        return grade_page(provider, first, live_text, None)
    observed = parse_fields(str(first.get("url")), live_text)
    pattern = observed.get("rating_pattern")
    match = next((e for e in entries if e.get("rating_pattern") == pattern), None)
    if match is None and pattern is None and len(entries) == 1:
        match = first
    if match is not None:
        return grade_page(provider, match, live_text, fixtures.get(match.get("rating_pattern")))
    grade = grade_page(provider, first, live_text, fixtures.get(first.get("rating_pattern")))
    if grade.status in ("identity", "missing"):
        return grade
    known = ", ".join(str(e.get("rating_pattern")) for e in entries)
    grade.status = "world_drift"
    grade.notes = [f"new template {pattern!r} (key holds {known}): add a copy"]
    return grade


# --------------------------------------------------------------------------
# Cross-check against a pipeline run (Tier B)
# --------------------------------------------------------------------------

CROSS_STATUSES = ("agree", "disagree", "identity", "unread")


def _num(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        return None


def cross_check_pipeline(key: Mapping[str, Any], providers: Sequence[Mapping[str, Any]],
                         tolerance: float = RATING_TOLERANCE) -> Dict[str, Any]:
    """Grade what a LIVE pipeline run read off golden-set pages, against the key.

    The golden freshness check fetches the pages on its own and asks "do the
    parsers still read them?". This asks the other half: when the real
    pipeline researched a golden doctor, did ITS read of that page agree
    with the key? A page is matched by URL — the pipeline's observations
    carry the profile URL they were read from — so nothing is fetched.

      agree      the pipeline's rating is within tolerance of a saved copy
      disagree   it is not — a refresh or a parser drift the freshness run
                 will name; either way a person looks
      identity   the pipeline attached this URL to a provider whose name
                 does not match the key — the dedupe/URL-union failure
                 class, a stranger's page on someone's card
      unread     the pipeline knew the URL but read no rating from it this
                 run (coverage, not correctness)

    Counts are reported but never graded: a listing row says "88 ratings"
    where the profile's star template says "38 reviews" and template B
    says nothing, so a count comparison across sources would flag every
    healthy run.
    """
    from utils.provenance import canonical_profile_url

    index: Dict[str, Tuple[Mapping[str, Any], str, List[Mapping[str, Any]]]] = {}
    for provider, url, entries in iter_urls(key):
        index[canonical_profile_url(url)] = (provider, url, list(entries))
    pages: List[Dict[str, Any]] = []
    seen: set = set()
    for p in providers or []:
        if not isinstance(p, dict) or p.get("enrichment_outcome") not in ("enriched", "cached"):
            continue
        name = str(p.get("name") or "")
        observations: Dict[str, Mapping[str, Any]] = {}
        for obs in p.get("review_observations") or []:
            if isinstance(obs, dict) and obs.get("source_url"):
                observations.setdefault(canonical_profile_url(obs["source_url"]), obs)
        known = {canonical_profile_url(u) for u in (p.get("platform_profile_urls") or {}).values() if u}
        for curl in sorted(set(observations) | known):
            hit = index.get(curl)
            if hit is None or curl in seen:
                continue
            seen.add(curl)
            gp, url, entries = hit
            page: Dict[str, Any] = {
                "provider_id": gp.get("provider_id"), "platform": entries[0].get("platform"), "url": url,
                "key": [{"rating_pattern": e.get("rating_pattern"), "rating": e.get("rating"),
                         "review_count": e.get("review_count")} for e in entries],
                "observed": None, "status": "unread", "notes": [],
            }
            if name and name_overlap(name, str(gp.get("stated_name") or "")) < NAME_OVERLAP_FLOOR:
                page["status"] = "identity"
                page["notes"].append("the pipeline attached this page to a provider whose name does not match the key")
                pages.append(page)
                continue
            obs = observations.get(curl)
            if obs is None:
                page["notes"].append("URL known to the pipeline; no observation read from it this run")
                pages.append(page)
                continue
            rating, count = _num(obs.get("rating")), _num(obs.get("review_count"))
            page["observed"] = {"rating": rating, "review_count": int(count) if count is not None else None,
                                "via": obs.get("extraction_source")}
            if rating is None:
                page["notes"].append("observation carries no rating")
            else:
                key_ratings = [_num(e.get("rating")) for e in entries if _num(e.get("rating")) is not None]
                if any(abs(rating - k) < tolerance for k in key_ratings):
                    page["status"] = "agree"
                else:
                    page["status"] = "disagree"
                    page["notes"].append(f"pipeline rating {rating} vs key {key_ratings}")
            pages.append(page)
    counts = {status: sum(1 for pg in pages if pg["status"] == status) for status in CROSS_STATUSES}
    return {"matched": len(pages), "key_pages": sum(1 for _ in iter_urls(key)), **counts, "pages": pages}


def summarize_grades(grades: Sequence[PageGrade]) -> Dict[str, int]:
    counts = {s: 0 for s in STATUSES}
    for g in grades:
        counts[g.status] = counts.get(g.status, 0) + 1
    return counts
