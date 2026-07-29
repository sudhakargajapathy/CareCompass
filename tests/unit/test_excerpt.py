"""Unit tests for utils/excerpt.py — content-aware page excerpting."""

from utils.excerpt import (
    SUMMARY_MAX_CHARS,
    _hit_positions,
    build_excerpt,
    clip_words,
    strip_boilerplate,
)

NAV_CHROME = "\n".join([
    "Skip to main content",
    "Accept all cookies",
    "Sign in | Register",
    "[Find a Doctor](https://x.com/find) | [Locations](https://x.com/loc)",
    "Menu",
    "© 2026 HealthSite. All rights reserved",
])

PROSE = (
    "Dr. Andrea An is a neurology specialist in Chandler with over 25 years of experience. "
    "Patients praise her thorough approach to migraine treatment and her clear communication. "
    "One review notes that appointments run on time and questions are always answered fully. "
)


class TestStripBoilerplate:
    def test_drops_chrome_keeps_prose(self):
        cleaned = strip_boilerplate(NAV_CHROME + "\n" + PROSE)
        assert "cookies" not in cleaned.lower()
        assert "sign in" not in cleaned.lower()
        assert "find a doctor" not in cleaned.lower()
        assert "Dr. Andrea An" in cleaned

    def test_prose_mentioning_nav_words_survives(self):
        # Marker words only strike SHORT lines — long prose is kept
        line = ("The clinic asks patients to sign in at the front desk and review "
                "their insurance details before every neurology appointment there.")
        assert line in strip_boilerplate(line)


class TestBuildExcerpt:
    def test_short_text_returned_whole(self):
        assert build_excerpt(PROSE, ["neurology"], budget=2000) == PROSE.strip()

    def test_head_fallback_when_no_anchor_hits(self):
        text = ("filler words here " * 300)  # ~5400 chars, no anchors
        excerpt = build_excerpt(text, ["neurology"], budget=500)
        assert len(excerpt) <= 500
        assert excerpt == strip_boilerplate(text)[:500]

    def test_window_centers_on_deep_content_not_head(self):
        # The ASCII-diagram scenario: 3K of filler, then the provider list
        filler = ("generic wellness marketing text without key terms " * 70)  # ~3.4K
        page = filler + PROSE + ("closing filler text " * 50)
        excerpt = build_excerpt(page, ["neurology", "review", "dr."], budget=800)
        assert "Dr. Andrea An" in excerpt
        assert len(excerpt) <= 800

    def test_multiple_far_apart_clusters_both_covered(self):
        doctor_a = "Dr. Alpha Neuro is a neurology physician praised in patient reviews. "
        doctor_b = "Dr. Beta Brain, another neurology specialist, also has strong reviews. "
        page = doctor_a + ("padding text " * 300) + doctor_b  # ~4K apart
        excerpt = build_excerpt(page, ["neurology", "review"], budget=1200, max_windows=3)
        assert "Dr. Alpha Neuro" in excerpt
        assert "Dr. Beta Brain" in excerpt
        assert len(excerpt) <= 1200

    def test_budget_is_a_hard_cap(self):
        page = (PROSE * 40)
        assert len(build_excerpt(page, ["neurology"], budget=1000)) <= 1000

    def test_anchor_matching_is_case_insensitive(self):
        page = ("padding " * 400) + "NEUROLOGY CARE from Dr. Loud Voice here."
        excerpt = build_excerpt(page, ["neurology"], budget=400)
        assert "Dr. Loud Voice" in excerpt

    def test_short_anchors_ignored(self):
        # 1-2 char anchors ("An") would hit inside random words — skipped
        page = ("random padding words " * 200) + "Dr. Target Person, neurology."
        excerpt = build_excerpt(page, ["An", "neurology"], budget=400)
        assert "Dr. Target Person" in excerpt

    def test_priority_anchor_beats_dense_generic_vocab(self):
        # 40 "review" mentions around the wrong doctor must not out-vote the
        # single mention of the right one
        wrong = "Dr. Wrong Person, a cardiology physician, has many reviews here. " * 40
        right = "Dr. Ortega earns consistent praise in neurology patient reviews. "
        page = wrong + right
        excerpt = build_excerpt(
            page, anchors=["review", "rating"], budget=1000,
            priority_anchors=["Dr. Maria Ortega", "Ortega"],
        )
        assert "Dr. Ortega earns consistent praise" in excerpt

    def test_vocab_fallback_when_priority_never_hits(self):
        page = ("padding " * 400) + "Neurology reviews praise Dr. Someone Else here."
        excerpt = build_excerpt(
            page, anchors=["review", "neurology"], budget=400,
            priority_anchors=["Dr. Absent Person"],
        )
        assert "Dr. Someone Else" in excerpt

    def test_regular_anchors_keep_a_window_when_priority_also_hits(self):
        # The real-world shape of a healthgrades profile: the rating and
        # tenure the page LEADS with, then a long tail of review comments
        # that repeat the surname. Priority-exclusive selection spent every
        # window on the comments and never read the header — which cost a
        # real provider his cross-platform rating pair and left a stale
        # years-of-experience value on the card.
        header = "4 Star Rating Based on 31 reviews. 30+ years of experience. "
        comments = "Dr. Khan listened carefully and explained everything clearly. " * 60
        page = header + comments
        excerpt = build_excerpt(
            page, anchors=["years of experience", "rating"], budget=900,
            priority_anchors=["Dr. Mohammad B. Khan", "Khan"],
        )
        assert "30+ years of experience" in excerpt   # regular anchors got a window
        assert "Dr. Khan listened carefully" in excerpt  # priority still decisive


class TestClipWords:
    """The bound both reasoning-model payloads share (judge + critic)."""

    def test_text_that_fits_is_returned_byte_identical(self):
        # A bound that reshapes text it did not need to touch is its own
        # source of drift — the judge and critic must be able to compare
        # strings, not "roughly the same paragraph".
        summary = (
            "Patients consistently praise Dr. Rabin as exceptionally caring. "
            "However, practice-level complaints emerge regarding scheduling "
            "difficulties, long wait times, and administrative issues."
        )
        assert clip_words(summary, SUMMARY_MAX_CHARS) == summary

    def test_a_real_summary_is_nowhere_near_the_bound(self):
        # Live summaries measured 725-732 chars. The bound exists for a
        # pathological response, not to trim ordinary output — if this ever
        # fails, the gatherer prompt changed, not the clip.
        assert SUMMARY_MAX_CHARS >= 2000

    def test_over_budget_breaks_on_a_word_and_says_so(self):
        # The defect this whole helper exists for: `[:400]` severed a summary
        # at "However, practice-level compla" with no marker, and the judge
        # then reported that the summary "begins to mention practice-level
        # complaints without providing their details" — reasoning about a
        # fragment as though it were the whole document.
        long_text = "Patients report long wait times at this office. " * 80
        clipped = clip_words(long_text, 300)

        assert len(clipped) <= 300
        assert clipped.endswith(" …")                      # the cut announces itself
        assert clipped[:-2].rstrip().split()[-1] in long_text.split()
        assert "compla" not in clipped                     # never a word fragment

    def test_none_and_blank_are_empty_not_crashes(self):
        assert clip_words(None, 100) == ""
        assert clip_words("   ", 100) == ""


class TestHeadWindow:
    """A fact stated ONCE cannot win a density contest against a table."""

    # The shape of a review-platform profile: the overall rating and the
    # provider's tenure appear on one header line and nowhere else; below it
    # sit hundreds of surname-heavy comments, then a percentage distribution
    # that repeats the review vocabulary dozens of times.
    HEADER = ("Dr. Ellen Kuniyoshi, MD. Neurology. 3.4 out of 5 (23 ratings). "
              "28 years of experience.\n")
    COMMENTS = " ".join(
        "Review: Dr. Kuniyoshi was thorough. Kuniyoshi listened. "
        "I would see Dr. Kuniyoshi again. rating review patient"
        for _ in range(40)
    )
    TABLE = " ".join(
        "5 star 48% 4 star 9% 3 star 4% 2 star 0% 1 star 39% ratings review reviews"
        for _ in range(20)
    )
    PAGE = HEADER + COMMENTS + TABLE
    ANCHORS = ["rating", "review", "years of experience"]

    def _excerpt(self, include_head):
        return build_excerpt(self.PAGE, anchors=self.ANCHORS,
                             priority_anchors=["Kuniyoshi"], budget=2000,
                             include_head=include_head)

    def test_density_alone_misses_the_header(self):
        """The 2026-07-25 failure, reproduced. The extractor was handed the
        percentage table, correctly refused to derive a rating from it, and was
        never shown the line stating the average outright — so the provider
        scored as though Healthgrades had no rating at all."""
        without = self._excerpt(include_head=False)
        assert "3.4 out of 5" not in without
        assert "28 years" not in without

    def test_reserving_the_head_recovers_the_stated_pair(self):
        with_head = self._excerpt(include_head=True)
        assert "3.4 out of 5" in with_head
        assert "(23 ratings)" in with_head
        # The per-domain hint aimed at Healthgrades' tenure line never reached
        # it either — same window, same cause.
        assert "28 years of experience" in with_head

    def test_the_head_does_not_cost_the_anchors_their_windows(self):
        """One window of three, not the whole budget: the review body must
        still be represented or the summary narrows to a header."""
        with_head = self._excerpt(include_head=True)
        assert "Kuniyoshi was thorough" in with_head
        assert len(with_head) <= 2000

    def test_head_window_is_off_by_default(self):
        """Opt-in: the candidate pass reads listing pages, where the head is a
        page title rather than a rating."""
        assert self._excerpt(include_head=False) == build_excerpt(
            self.PAGE, anchors=self.ANCHORS, priority_anchors=["Kuniyoshi"], budget=2000
        )

    def test_short_pages_are_unaffected(self):
        short = "Dr. Ellen Kuniyoshi. 3.4 out of 5 (23 ratings)."
        assert build_excerpt(short, anchors=self.ANCHORS, include_head=True) == short

    def test_an_early_span_is_never_absorbed_by_a_later_one(self):
        """Regression: spans were merged in CLUSTER order, not span order.

        A name anchor that hits the header and then every review comment forms
        one cluster spanning the page, whose window centers halfway down it. The
        reserved head span then arrived "after" that window in the merge list,
        looked like an overlap, and was absorbed — so the head window was built,
        counted against the budget, and silently thrown away. Found only because
        the wiring test failed while the helper's own test passed."""
        page = self.HEADER + self.COMMENTS + self.TABLE
        out = build_excerpt(page, anchors=self.ANCHORS,
                            # Anchors on the header AND throughout the body —
                            # the shape that produced the swallowed span.
                            priority_anchors=["Dr. Ellen Kuniyoshi", "Kuniyoshi"],
                            budget=2000, include_head=True)
        assert out.startswith("Dr. Ellen Kuniyoshi")
        assert "3.4 out of 5" in out


# ---- the enrichment head reservation, and short surnames ----
#
# Both come from the 2026-07-28 live runs, where the SAME search ranked
# Dr. Andrea An #1 and then #5 — because run 1 read webmd as "4.5/5" with no
# count and run 2 read "4.5/5 (61 reviews)".

_HEADER_FACT = "4.1 out of 5 (70 ratings)"
_PROSE_FACT = "never heard back"


def _profile_page(surname: str = "An", first: str = "Andrea") -> str:
    """A review-platform PROFILE, shaped like the real thing.

    Load-bearing properties, each one a reason a fact goes missing:
      * ~18K chars — a real profile runs 10-40K, and a short fixture passes
        `len(cleaned) <= budget` and returns whole, testing nothing.
      * the rating is stated ONCE, ~1000 chars into the CLEANED text, which is
        past a 666-char head reservation and nowhere near it.
      * the doctor is named uniformly every ~130 chars through the reviews, so
        `_pick_clusters` merges the whole body into ONE cluster.
      * one complaint sits at ~76% depth, where that merged cluster's
        midpoint-centred window does not reach.
    """
    full = f"Dr. {first} {surname}, MD"
    return (
        "Home\nFind a Doctor\nSign in\nMenu\nPrivacy Policy\nSkip to main content\n"
        + f"{full}\nNeurology\nChandler, AZ 85224\nAccepting new patients\n"
        + "Overview  Locations  Insurance  Reviews  About  Compare\n"
        + (f"Dr. {surname} is a neurologist in Chandler, Arizona affiliated with multiple "
           "hospitals in the area, including Chandler Regional Medical Center. Providers "
           "are practicing physicians who see patients in an office setting. ") * 4
        + f"\n{_HEADER_FACT}\n21 years of experience\n"
        + (f"About Dr. {surname}. Board certified. Education and training details follow. "
           "Hospital affiliations and office locations are listed below. ") * 25
        + ("5 star 62%\n4 star 14%\n3 star 6%\n2 star 4%\n1 star 14%\n"
           "Likelihood of recommending 4.1\nExplains conditions well 4.2\n"
           "Trust in provider's decisions 4.0\nScheduling appointments 3.6\n") * 8
        + (f"Dr. {surname} was thorough and took time to explain everything to me. Excellent "
           "bedside manner and an accurate diagnosis. I felt heard during the visit "
           "and would recommend her to anyone needing a neurologist. ") * 40
        + f"\nI was told I would be called about my MRI approval but {_PROSE_FACT} "
          "from the office. Reaching anyone through the answering service is very "
          "difficult and follow-up on test results did not happen.\n"
        + (f"Great doctor overall, the wait was about 10-13 minutes. Staff were pleasant "
           f"and the office was easy to find. Dr. {surname} answered all my questions. ") * 30
    )


def _enrich(page: str, surname: str = "An", **overrides) -> str:
    """The enrichment pass's own excerpt call, as `_extract_review_data_only` makes it."""
    kwargs = dict(
        anchors=["review", "rating", "Neurology"],
        budget=2000, max_windows=3,
        priority_anchors=[f"Dr. Andrea {surname}, MD", surname],
        include_head=True,
    )
    kwargs.update(overrides)
    return build_excerpt(page, **kwargs)


def test_a_sized_head_reservation_reaches_a_profile_rating_header():
    """The rating is stated once, ~1000 chars in. `budget // max_windows` gave
    the head 666 and it landed ~400 short — capturing the header on some runs
    and not others, which is what moved a provider four ranks between two runs
    of the same search."""
    page = _profile_page()

    assert _HEADER_FACT not in _enrich(page), "the shipped 2000/3 sizing should miss it"
    assert _HEADER_FACT in _enrich(page, head_chars=1200)


def test_the_reservation_is_the_cheap_route_to_the_header():
    """The budget alone DOES eventually reach the header — at 4000, twice
    today's tokens for one fact, and still no prose. The reservation gets it at
    2000, where the budget alone does not.

    Stated as a cost argument rather than "the budget never works", because the
    sweep says it does work, expensively. A test claiming otherwise would be
    false and would block a future budget raise for the wrong reason.

    RE-MEASURED 2026-07-29. This test previously asserted the same thing at
    3000, and that stopped being true when `_anchor_pattern` learned to match
    the plural: the header reads "4.1 out of 5 (70 ratings)", so the ordinary
    `rating` anchor now lands ON it and gives a SECOND, independent route in.
    The old number was not wrong when measured — it was measured against a
    matcher that scored zero on "reviews" and "ratings".

        budget  windows  head          old matcher   with plurals
          2000        3   666(default)    miss           miss
          3000        3  1000(default)    miss           HIT     <- moved
          3000        4  1200(reserved)   hit            hit
          4000        3   default         hit            hit

    This is the second time an excerpt measurement went stale on an anchor
    change. Re-sweep after touching `_anchor_pattern`; never
    patch the assertion."""
    page = _profile_page()

    assert _HEADER_FACT not in _enrich(page, budget=2000)
    assert _HEADER_FACT in _enrich(page, budget=2000, head_chars=1200)

    # ...and the expensive route, recorded so the trade stays visible
    assert _HEADER_FACT in _enrich(page, budget=4000)


def test_the_reservation_still_earns_its_place_at_the_shipped_sizing():
    """The plural fix did NOT make the head reservation redundant.

    At the sizing actually shipped — `_ENRICHMENT_EXCERPT_BUDGET` 3000 over
    `_ENRICHMENT_EXCERPT_WINDOWS` 4 — the anchor route still does not reach the
    header on its own, because four windows over 3000 chars are 750 wide and
    the clustering puts them in the review body:

        head=0    miss      head=900   miss
        head=666  miss      head=1100  HIT
                            head=1200  HIT   <- shipped, with margin

    The two routes fail independently, which is the argument for keeping both:
    the anchor route needs the page to WORD its count ("70 ratings"), and the
    reservation needs the header to sit inside 1200 chars. A page that renders
    "4.1 ★ · 70" defeats the first; a 44,138-char profile defeats the second."""
    page = _profile_page()
    shipped = dict(budget=3000, max_windows=4)

    assert _HEADER_FACT not in _enrich(page, include_head=False, **shipped)
    assert _HEADER_FACT not in _enrich(page, head_chars=900, **shipped)
    assert _HEADER_FACT in _enrich(page, head_chars=1200, **shipped)


def test_enrichment_window_count_is_a_staircase_not_a_flat_line():
    """Round 12 measured "more windows buy nothing" for DISCOVERY. Enrichment
    does NOT behave that way, and assuming it did would have left the window
    count at 3: `window_size = budget // max_windows` feeds back into
    clustering, so narrower windows split the review body into more clusters
    and the output steps up — 3 -> 2103 chars, 4/5/6 -> 2406, 8/10 -> 2557.

    Asserts the step and the plateau we actually sit on. The next step (8) is
    rejected on span WIDTH, not char count — see the constant's comment."""
    page = _profile_page()
    kw = dict(budget=3000, head_chars=1200)

    at_three = _enrich(page, max_windows=3, **kw)
    at_four = _enrich(page, max_windows=4, **kw)

    assert len(at_four) > len(at_three), "3 -> 4 is a real gain, not a wash"
    for windows in (5, 6):
        assert _enrich(page, max_windows=windows, **kw) == at_four


def test_a_two_letter_surname_still_anchors():
    """`_hit_positions` required anchors of 3+ chars as a proxy for "don't match
    inside other words". Every provider with a two-letter surname — An, Ho, Li,
    Ng, Wu, Yu, Oh — therefore lost their name anchor entirely, and the
    enrichment pass anchored on nothing but generic review vocabulary.

    Dr. Andrea An is the provider whose card exposed this across two live runs.
    """
    page = _profile_page(surname="An")
    hits = _hit_positions(strip_boilerplate(page).lower(), ["Dr. Andrea An, MD", "An"])

    assert len(hits) > 50, f"a named doctor should anchor throughout their own profile, got {len(hits)}"


def test_a_short_anchor_does_not_match_inside_other_words():
    """The reason the length floor existed, now handled properly: "an" must not
    hit inside "and", "many", "manner", "answering" — which is every page."""
    text = "and many a manner of answering came, and android bananas abound"

    assert _hit_positions(text, ["an"]) == []
    # `lowered` is the function's contract — callers lowercase before calling.
    assert _hit_positions("dr. an saw me; an was thorough", ["an"]) == [4, 15]


def test_anchors_ending_in_punctuation_still_match():
    """A word boundary asserts a word/non-word transition, so appending one to
    an anchor ending in punctuation asserts a WORD character follows the
    punctuation — and the anchor stops matching at all. `rating:` and `Dr.` are
    both live anchor shapes."""
    assert _hit_positions("rating: 4.8 stars", ["rating:"]) == [0]
    assert _hit_positions("dr. an is a neurologist", ["dr."]) == [0]
