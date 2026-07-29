"""Content-aware excerpting: hand the extractor signal, not site chrome.

Scraped pages open with nav bars, cookie banners, and link lists; a blind
head-truncation (raw_content[:2000]) often feeds the LLM exactly that while
the provider list sits below the fold. This module strips boilerplate-looking
lines, then spends the same fixed character budget on a few disjoint windows
centered where the anchor keywords actually hit.

Design notes (from the gap #7 review):
- Multiple windows, not one — directory pages spread providers across
  10-20K chars, so a single "densest cluster" window still misses most.
- Callers pick the anchors: the candidate pass anchors on the specialty and
  review vocabulary; the enrichment pass anchors on the provider's NAME
  (which doubles as an identity guard).
- Head-truncation of the cleaned text is the explicit fallback whenever
  anchors are absent or never hit, so behavior can only improve.
"""

import re
import textwrap
from typing import List

# Upper bound on a provider's review_summary when it is handed to a reasoning
# model. The gatherer asks for 3-4 sentences, which run ~700-800 chars, so this
# is headroom against a pathological response — NOT a budget to spend. Anything
# that routinely trims real summaries is a bug: see clip_words below.
SUMMARY_MAX_CHARS = 2000

# Chrome phrases that mark a short line as navigation/boilerplate. Matched
# only on lines under _NAV_LINE_MAX_LEN so prose mentioning "search" or
# "menu" is never dropped.
_NAV_MARKERS = (
    "cookie", "sign in", "log in", "login", "skip to", "privacy policy",
    "terms of", "subscribe", "advertisement", "javascript", "menu",
    "back to top", "all rights reserved", "©",
)
_NAV_LINE_MAX_LEN = 80
_MIN_LINE_LEN = 4

# Lines carrying the data the extraction passes exist to find. Exempt from the
# chrome filters, which were deleting them: a profile header states its numbers
# compactly, and compact numeric text is exactly what "symbol-heavy" means.
_SIGNAL_RE = re.compile(
    r"\d[\d.,]*\s*(?:/\s*5|out\s+of\s+5|stars?\b)"        # 4.9/5, 4.9 out of 5
    r"|\(\s*\d[\d,]*\s*(?:reviews?|ratings?)?\s*\)"        # (127), (31 reviews)
    r"|\b\d[\d,]*\s+(?:reviews?|ratings?)\b"               # 127 reviews
    r"|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"                    # 480-555-1234
    r"|\b\d+\+?\s*(?:years?|yrs?)\b"                       # 26 years, 30+ yrs
    r"|\brating\s*:",                                      # Rating: 4.8
    re.IGNORECASE,
)


def clip_words(text: str, max_chars: int) -> str:
    """Bound text without cutting mid-word, and make a cut that fires VISIBLE.

    A silent truncation is worse than a short one. A model handed a fragment
    that stops mid-clause has no way to know it is a fragment, so it reasons
    about the fragment as if it were the whole document — the judge, fed a
    summary severed at "However, practice-level compla", scored the provider's
    access as "no evidence" and told the patient the summary "begins to mention
    practice-level complaints without providing their details". The trailing
    "…" is the signal that lets a reader distinguish "nothing was said" from
    "the rest was cut".

    Returns the input byte-identical when it fits — a bound that reshapes text
    it did not need to touch is its own source of drift.
    """
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return textwrap.shorten(cleaned, width=max_chars, placeholder=" …")


def strip_boilerplate(text: str) -> str:
    """Drop lines that look like site chrome; keep prose-like lines."""
    kept: List[str] = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if len(stripped) < _MIN_LINE_LEN:
            continue
        # A line stating a rating, a review count, a tenure or a phone number
        # is the highest-value datum on the page and is exempt from the
        # chrome filters below. Both of them were deleting exactly these:
        # profile headers state the numbers compactly ("4.9 (127)",
        # "Rating: 4.8/5"), which reads as symbol-heavy, and a directory
        # entry is a single markdown link, which read as a nav row. The
        # enrichment pass exists to harvest these lines, and it was being
        # handed pages with them already removed.
        if _SIGNAL_RE.search(stripped):
            kept.append(stripped)
            continue
        lowered = stripped.lower()
        if len(stripped) < _NAV_LINE_MAX_LEN and any(m in lowered for m in _NAV_MARKERS):
            continue
        # Link rows ("[Find a Doctor](https://…) | [Locations](…)") carry no
        # extractable content. Counting `](` and `http` SEPARATELY rather than
        # summing them: a single markdown link contributes one of each, so the
        # sum hit the threshold of 2 on its own and every one-link directory
        # entry — the line naming the provider — was dropped.
        link_count = max(stripped.count("]("), stripped.count("http"))
        if link_count >= 2 and len(stripped) < 160:
            continue
        # Symbol-heavy short lines (separators, breadcrumb rows)
        letters = sum(c.isalpha() for c in stripped)
        if letters / len(stripped) < 0.5 and len(stripped) < 60:
            continue
        kept.append(stripped)
    return "\n".join(kept)


# Shortest anchor we will look for. Two, not three, because the three-char
# floor was a proxy for "don't match inside other words" — a job word
# boundaries do properly (see `_anchor_pattern`). At three, every provider
# whose SURNAME is two letters lost their name anchor entirely: An, Ho, Li,
# Ng, Wu, Yu, Oh. The enrichment pass anchors on the surname as its decisive
# priority, so for those providers it anchored on nothing but the page's
# generic review vocabulary — measured at 1 priority hit on a profile page
# where a normal surname scores 101.
#
# That is not hypothetical: Dr. Andrea An is the provider whose card exposed
# the header-extraction failure across two live runs on 2026-07-28.
_MIN_ANCHOR_LEN = 2


def _anchor_pattern(anchor: str) -> str:
    """Anchor as a regex, word-bounded only where a boundary is meaningful.

    `\\b` asserts a word/non-word transition, so appending it to an anchor that
    ends in punctuation ("Dr.", "rating:") asserts a WORD character follows the
    punctuation and the anchor stops matching at all. Apply each boundary only
    when the adjacent character is one a boundary can be defined against.

    A trailing PLURAL is admitted before that boundary. Bounding both ends
    fixed "an" matching inside "management" and simultaneously broke the thing
    the vocabulary anchors exist for: `\\breview\\b` scores ZERO on "Read all
    reviews", and `\\brating\\b` misses "70 patient ratings" — which is how a
    review count is written on every one of the five platforms. The 2026-07-29
    field run showed the consequence, a healthgrades profile fetched at 44,138
    chars that yielded no rating+count pair while the city directory page did.

    Only "s", and only at the end: "reviewer" and "rated" stay out, because an
    anchor that matches a different part of speech drifts away from the number
    it is aimed at. On a two-letter surname the extra branch is harmless
    ("wus", "ans" are not words).
    """
    escaped = re.escape(anchor)
    prefix = r"\b" if anchor[:1].isalnum() or anchor[:1] == "_" else ""
    if anchor[-1:].isalpha():
        # Alphabetic tail: allow the plural, then require the boundary.
        return f"{prefix}{escaped}s?\\b"
    suffix = r"\b" if anchor[-1:].isalnum() or anchor[-1:] == "_" else ""
    return f"{prefix}{escaped}{suffix}"


def _hit_positions(lowered: str, anchors: List[str]) -> List[int]:
    """Positions where any anchor occurs as a WHOLE WORD.

    Substring matching made short anchors unusable — "an" hits inside "and",
    "many", "manner", "answering" on every page — which is why the length floor
    existed. Bounding the match removes the reason for the floor rather than
    living with its cost.
    """
    return sorted({
        match.start()
        for anchor in anchors or []
        if anchor and len(str(anchor).strip()) >= _MIN_ANCHOR_LEN
        for match in re.finditer(_anchor_pattern(str(anchor).strip().lower()), lowered)
    })


def _pick_clusters(hits: List[int], window_size: int, limit: int) -> List[List[int]]:
    """The `limit` densest clusters of hit positions.

    A gap wider than one window starts a new cluster; densest clusters win.
    """
    if not hits or limit <= 0:
        return []
    clusters: List[List[int]] = [[hits[0]]]
    for position in hits[1:]:
        if position - clusters[-1][-1] > window_size:
            clusters.append([position])
        else:
            clusters[-1].append(position)
    clusters.sort(key=len, reverse=True)
    return clusters[:limit]


def build_excerpt(
    text: str,
    anchors: List[str],
    budget: int = 2000,
    max_windows: int = 3,
    priority_anchors: List[str] = None,
    include_head: bool = False,
    head_chars: int = None,
) -> str:
    """Excerpt of ≤ budget chars: boilerplate-stripped, anchor-centered.

    Finds case-insensitive anchor hits in the cleaned text, groups them into
    clusters, keeps the max_windows densest clusters (re-ordered by document
    position), and cuts one window around each — all sharing the single
    budget.

    priority_anchors (e.g. the provider's name in the enrichment pass) are
    decisive but NOT exclusive: they claim windows first, so dense generic
    vocabulary ("review" forty times around the wrong doctor) can't drown the
    one mention that identifies the right one — but one window is reserved
    for the regular anchors. Priority-exclusive selection had a real cost: on
    a doctor's own profile the surname hits dozens of times in the review
    comments, so every window landed there and the per-domain hints aimed at
    the page's rating/experience header were never consulted. Regular anchors
    take every window when priorities never hit; a plain head-truncation of
    the cleaned text is the final fallback.

    include_head reserves one window at the START of the cleaned text,
    regardless of anchors. Selection is otherwise by DENSITY, and a fact stated
    ONCE can never win a density contest against a table that repeats its
    vocabulary — which is exactly how a review platform lays out a profile:
    "3.4 out of 5 (23 ratings)" on one header line, then a percentage
    distribution and hundreds of comments below. On the 2026-07-25 run the
    extractor received Kuniyoshi's Healthgrades percentage table and correctly
    declined to derive a rating from it, having never been shown the header
    twenty lines above that stated the average outright. No amount of added
    vocabulary fixes that; position does.

    head_chars SIZES that reservation independently. Without it the head takes
    `budget // len(chosen)` like any other window — equal sizing that was never
    argued for, it just fell out of the arithmetic. The two window kinds are not
    doing the same job: the head holds ONE stated fact at a known position,
    while the density windows hold prose that is worth whatever room is left.
    At the enrichment pass's 2000/3 that made the head 666 chars, and a review
    platform's rating line sits ~900-1100 chars into the cleaned text — so the
    reservation landed just short of the fact it exists to capture, and did so
    NON-DETERMINISTICALLY across runs of the same search (see the sweep beside
    `_ENRICHMENT_EXCERPT_BUDGET`). Omitted, behaviour is exactly as before.
    """
    cleaned = strip_boilerplate(text)
    if len(cleaned) <= budget:
        return cleaned

    window_size = budget // max_windows
    lowered = cleaned.lower()
    priority_hits = _hit_positions(lowered, priority_anchors)
    regular_hits = _hit_positions(lowered, anchors)

    # The head claims its window first; anchors compete for what's left.
    anchor_windows = max(1, max_windows - 1) if include_head else max_windows

    if priority_hits and regular_hits:
        clusters = _pick_clusters(priority_hits, window_size, max(1, anchor_windows - 1))
        clusters += _pick_clusters(regular_hits, window_size, 1)
    else:
        clusters = _pick_clusters(priority_hits or regular_hits, window_size, anchor_windows)

    if include_head and head_chars is None:
        # Position 0 is a cluster of one; the sort below restores document
        # order, so it lands first. Appended unconditionally — a cluster that
        # merely BEGINS near the head is not evidence the head is covered,
        # because each window is centered on its cluster's midpoint, and an
        # anchor that runs from the header through hundreds of review comments
        # centers far down the page. When the spans genuinely do overlap, the
        # merge below joins them and no text repeats.
        #
        # Only on the legacy path. With head_chars set the head is carved out
        # below at its own size instead of competing as a cluster.
        clusters.append([0])

    if not clusters:
        # Head-only, or nothing anchored: a prefix of the cleaned text IS the
        # head, and a wider one than head_chars would have given.
        return cleaned[:budget]

    # Restore document order; overlapping spans merge below, so a regular
    # cluster landing inside a priority one costs no duplicated text.
    chosen = sorted(clusters, key=lambda c: c[0])

    spans = []
    anchor_budget = budget
    if include_head and head_chars is not None:
        head_end = min(len(cleaned), head_chars)
        spans.append((0, head_end))
        # What the head takes, the density windows do not get. The caller sizes
        # the reservation; the budget still bounds the whole excerpt.
        anchor_budget = max(0, budget - head_end)

    per_window = anchor_budget // len(chosen)
    for cluster in chosen:
        center = (cluster[0] + cluster[-1]) // 2
        start = max(0, center - per_window // 2)
        spans.append((start, min(len(cleaned), start + per_window)))

    # Sort by SPAN start, not by cluster start. Each window is centered on its
    # cluster's midpoint, so a cluster that begins early can produce a span that
    # begins late — a name anchor hitting the header and then every review
    # comment below spans the page and centers halfway down it. The merge below
    # assumes ascending order; fed an out-of-order span it treats the later
    # window as overlapping and silently absorbs the earlier one, which is
    # exactly how the reserved head window disappeared.
    spans.sort()

    # Merge overlapping/adjacent spans so no text repeats
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    excerpt = " … ".join(cleaned[start:end].strip() for start, end in merged)
    return excerpt[:budget]
