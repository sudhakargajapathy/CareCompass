"""Unit tests for pure helpers in app.py (superlatives, rubric markup,
judge-consistency findings)."""

import pytest

from app import (
    _empty_shortlist_notice,
    _judge_findings,
    refinement_note_markup,
    _pool_highlights,
    _rubric_markup,
)


def test_pool_highlights_awards_each_superlative():
    providers = [
        {"computed_distance_miles": 8.0, "blended_review_count": 80, "years_experience": 10},
        {"computed_distance_miles": 2.0, "review_count": 12, "years_experience": 25},
        {"computed_distance_miles": 5.0, "blended_review_count": 40, "years_experience": 5},
    ]
    highlights = _pool_highlights(providers)
    assert "Closest" in highlights[1]           # 2.0 mi is nearest
    assert "Most reviewed" in highlights[0]      # 80 reviews (blend preferred)
    assert "Most experienced" in highlights[1]   # 25 years


def test_pool_highlights_needs_two_to_compare():
    # Only one provider has a computed distance -> "Closest" is meaningless
    providers = [
        {"computed_distance_miles": 3.0, "review_count": 10, "years_experience": 8},
        {"review_count": 20, "years_experience": 12},
    ]
    highlights = _pool_highlights(providers)
    assert all("Closest" not in labels for labels in highlights.values())


def test_pool_highlights_ties_break_to_higher_rank():
    providers = [
        {"review_count": 50, "years_experience": 10},
        {"review_count": 50, "years_experience": 10},
    ]
    highlights = _pool_highlights(providers)
    assert "Most reviewed" in highlights.get(0, [])
    assert "Most experienced" in highlights.get(0, [])
    assert 1 not in highlights                   # the tie went to the earlier index


def test_pool_highlights_empty_pool_is_safe():
    assert _pool_highlights([]) == {}


def test_closest_chip_respects_city_centroid_uncertainty():
    """The chip must rank on the same effective distance the SCORER uses.

    The scorer adds CITY_CENTROID_MARGIN_MILES to a city-precision figure, so
    a ZIP-measured 8.5 mi beats a city-estimated 8.0 mi. Reading the raw
    number here put "Closest" on the provider the algorithm had ranked
    farther — the badge contradicting the ordering printed beside it.
    """
    providers = [
        {"computed_distance_miles": 8.0, "distance_precision": "city"},
        {"computed_distance_miles": 8.5, "distance_precision": "zip"},
    ]
    highlights = _pool_highlights(providers)
    assert "Closest" in highlights.get(1, [])
    assert "Closest" not in highlights.get(0, [])


def test_closest_chip_is_withheld_when_a_centroid_is_shared():
    """One city centroid is ONE coordinate shared by every provider in it —
    the 2026-07-25 pool showed an identical 8.0 mi ten times. Awarding
    "Closest" to whichever sorted first would overstate what was measured."""
    pool = [
        {"computed_distance_miles": 8.0, "distance_precision": "city"}
        for _ in range(10)
    ]
    assert _pool_highlights(pool) == {}


def test_closest_chip_still_awarded_on_a_real_difference():
    providers = [
        {"computed_distance_miles": 2.0, "distance_precision": "zip"},
        {"computed_distance_miles": 9.0, "distance_precision": "zip"},
    ]
    assert "Closest" in _pool_highlights(providers).get(0, [])


# ---- _rubric_markup: every band explains itself, scored or not ----

_RUBRIC = {"review_substance": 48.0, "red_flags": 30.0, "practical_access": 10.0}


def test_rubric_renders_a_real_citation():
    markup = _rubric_markup({
        "ai_rubric": _RUBRIC,
        "ai_evidence": {"review_substance": 'patients call him "very thorough"'},
    })
    assert "cc-bar-quote" in markup
    assert "very thorough" in markup
    assert "&quot;" in markup                    # quotes escaped, not raw
    assert "cc-bar-nodata" not in markup.split("Red flags")[0]


@pytest.mark.parametrize("sentinel", ["no evidence", "No evidence.", "none", "N/A", ""])
def test_uncited_criterion_gets_a_note_not_a_blank(sentinel):
    # An uncited criterion used to render as an empty gap under the bar.
    markup = _rubric_markup({
        "ai_rubric": _RUBRIC,
        "ai_evidence": {"practical_access": sentinel},
    })
    assert "cc-bar-nodata" in markup
    assert "No scheduling or wait-time evidence was cited" in markup
    assert "&ldquo;no evidence" not in markup.lower()   # never posed as a quote


def test_absent_notes_never_claim_anything_about_the_sources():
    """The note may describe the JUDGE'S CITATION only.

    Live failure: practical_access rendered "No details on scheduling, wait
    times, or office responsiveness IN THE SOURCES" directly beneath a review
    summary describing long wait times — which the judge had itself quoted under
    red_flags. This branch observes one thing, that no snippet arrived for the
    key; it cannot see what the sources contain."""
    markup = _rubric_markup({"ai_rubric": _RUBRIC, "ai_evidence": {}})

    for overclaim in ("in the sources", "across the sources", "found in", "was found"):
        assert overclaim not in markup.lower(), f"note claims {overclaim!r} about the corpus"
    assert markup.lower().count("was cited") + markup.lower().count("were cited") == 3


def test_absent_note_stays_true_beside_a_high_score():
    """The branch reads only the evidence string, never the score, so a nearly
    full bar can reach this text. "Nothing was cited" survives that; "nothing
    was found" would have rendered under a 48/50 bar."""
    markup = _rubric_markup({
        "ai_rubric": {"review_substance": 48.0, "red_flags": 30.0, "practical_access": 20.0},
        "ai_evidence": {"review_substance": "none"},
    })
    assert "48/50" in markup
    assert "No review text was cited for this criterion." in markup
    assert "found" not in markup.lower()


def test_missing_evidence_key_gets_the_same_note():
    markup = _rubric_markup({"ai_rubric": _RUBRIC, "ai_evidence": {}})
    assert markup.count("cc-bar-nodata") == 3          # all three bands explained


def test_rubric_absent_renders_nothing():
    assert _rubric_markup({"ai_evidence": {"review_substance": "x"}}) == ""
    assert _rubric_markup({"ai_rubric": {}}) == ""


# ---- judge-consistency findings: the critic auditing OUR judge ----
#
# Round 5 gave the critic the judge's rubric and asked it to flag criteria
# parked in a neutral band while the summary held evidence for them. The
# finding was then logged to stderr and nowhere else — app.py configures
# logging with no FileHandler, so it died with the container and never
# appeared in the app. These helpers give it three destinations.

_FINDING = (
    'practical_access scored 10/20 "no evidence" though the summary '
    "describes long wait times."
)


def _validation(*entries) -> dict:
    return {"top_provider_validation": {"top_provider_validations": list(entries)}}


def test_judge_findings_extracts_name_and_text():
    findings = _judge_findings(_validation(
        {"provider_name": "Dr. Brian Rabin, MD", "rank": 1,
         "recommendation_adjustments": _FINDING},
    ))
    assert findings == [("Dr. Brian Rabin, MD", _FINDING)]


def test_judge_findings_covers_providers_below_the_shortlist():
    """The critic validates EVERY ranked provider while the cards render only
    the top 5. A signal about our own scoring must not stop at rank 5 — which
    is why this reads the raw critic entries, not each card's critic_review."""
    findings = _judge_findings(_validation(
        {"provider_name": "Dr. Top", "rank": 1, "recommendation_adjustments": ""},
        {"provider_name": "Dr. Ninth", "rank": 9, "recommendation_adjustments": _FINDING},
    ))
    assert findings == [("Dr. Ninth", _FINDING)]


@pytest.mark.parametrize("entry", [
    {"provider_name": "Dr. Clean", "rank": 1},                              # absent
    {"provider_name": "Dr. Clean", "rank": 1, "recommendation_adjustments": ""},
    {"provider_name": "Dr. Clean", "rank": 1, "recommendation_adjustments": "   "},
    {"provider_name": "Dr. Clean", "rank": 1, "recommendation_adjustments": None},
])
def test_judge_findings_skips_empty_adjustments(entry):
    """Clean judge scoring is the expected case and must produce nothing —
    otherwise the panel shows a concern on every single search."""
    assert _judge_findings(_validation(entry)) == []


@pytest.mark.parametrize("payload", [{}, None, {"top_provider_validation": {}}])
def test_judge_findings_tolerates_missing_structure(payload):
    assert _judge_findings(payload) == []


# ---- the WIRING, not just the helpers ----
#
# Round 29 removed the judge-consistency note from the Responsible-AI panel
# (owner call: the surviving findings were cosmetic, and a patient-facing
# count with no readable detail raised questions the panel could not answer).
# `_judge_findings` itself STAYS — the Detailed Agent Analysis critic tab and
# the audit-log event still consume it — so these tests now guard the split
# in the new direction: the finding must reach the developer surfaces and
# must NOT reach the panel.

def _panel_markup(monkeypatch, *entries) -> str:
    """The Responsible-AI panel's composed HTML, captured before Streamlit."""
    import app as app_module

    captured = []
    monkeypatch.setattr(app_module, "render_html", captured.append)
    app_module.render_validation_insights({
        "agent_outputs": {"critic_validator": {"validation_results": {
            "bias_analysis": {"bias_assessment": {"severity": "low", "detected_biases": []}},
            "top_provider_validation": {"top_provider_validations": list(entries)},
            "final_recommendations": {"recommendation_confidence": "high"},
        }}},
        "workflow_summary": {},
    })
    return "".join(captured)


def test_panel_omits_the_judge_note_even_when_a_finding_exists(monkeypatch):
    """Round 29: the judge-consistency count LEFT the panel (owner call).

    Before this, a real finding rendered "Judge review: 1 inconsistency was
    found" in the patient-facing panel. The finding itself was never
    readable there (its text names internal rubric criteria), so the row
    raised a question the panel could not answer. It now renders only on
    the developer surfaces — asserted by the companion test below.
    """
    markup = _panel_markup(
        monkeypatch,
        {"provider_name": "Dr. Rabin", "rank": 1, "recommendation_adjustments": _FINDING},
    )
    assert "Judge review:" not in markup
    assert "inconsistenc" not in markup
    assert "practical_access" not in markup
    assert "Responsible AI review" in markup          # the panel itself still rendered


def test_judge_findings_still_reach_the_developer_surfaces():
    """The panel removal must not take the developer destinations with it.

    Wiring assert on source: the Detailed Agent Analysis critic tab still
    renders the findings under its heading, and the audit-log event is
    still emitted from the harvest path. A helper-only check would let
    either consumer be deleted with the suite green.
    """
    import inspect
    import app as app_module

    src = inspect.getsource(app_module)
    assert "Judge consistency findings" in src
    assert "judge_evidence_inconsistency" in src
    # And the helper still extracts the finding those surfaces render.
    findings = _judge_findings(_validation(
        {"provider_name": "Dr. Rabin", "rank": 1, "recommendation_adjustments": _FINDING},
    ))
    assert findings == [("Dr. Rabin", _FINDING)]


def test_panel_stays_silent_on_a_clean_run(monkeypatch):
    markup = _panel_markup(
        monkeypatch,
        {"provider_name": "Dr. Clean", "rank": 1, "recommendation_adjustments": ""},
    )
    assert "Judge review:" not in markup
    assert "Responsible AI review" in markup          # the panel itself still rendered


# ---- Round-6 panel fixes: what a PATIENT actually sees ----

def _panel(monkeypatch, *, bias=None, validations=(), considerations=()):
    """The Responsible-AI panel's composed HTML."""
    import app as app_module

    captured = []
    monkeypatch.setattr(app_module, "render_html", captured.append)
    app_module.render_validation_insights({
        "agent_outputs": {"critic_validator": {"validation_results": {
            "bias_analysis": {"bias_assessment": bias or {
                "severity": "low", "detected_biases": [], "explanation": ""}},
            "top_provider_validation": {"top_provider_validations": list(validations)},
            "final_recommendations": {
                "recommendation_confidence": "high",
                "key_findings": ["Detected potential biases in ranking methodology"],
                "important_considerations": list(considerations),
            },
        }}},
        "workflow_summary": {},
    })
    return "".join(captured)


# Change 0 — pass verdicts must not be counted as inconsistencies

_LIVE_PASS = ("Judge correctly scored practical_access low (5) reflecting "
              "wait-time complaints; scoring matches evidence.")
_REAL_FINDING = ('practical_access scored 10/20 "no evidence" though the '
                 'summary describes long wait times.')


def test_pass_verdicts_never_reach_the_patient_count(monkeypatch):
    """The live run rendered "10 inconsistencies were found" when the true
    count of judge errors was zero. (Round 29 removed the count from the
    panel entirely, which subsumes this — the assertions stay because they
    would catch the note quietly returning.)"""
    markup = _panel(monkeypatch, validations=[
        {"provider_name": f"Dr. {i}", "rank": i, "recommendation_adjustments": _LIVE_PASS}
        for i in range(1, 11)
    ])
    assert "Judge review:" not in markup
    assert "inconsistenc" not in markup


def test_a_real_finding_feeds_the_dev_surfaces_but_not_the_panel(monkeypatch):
    """The pass-verdict filter still matters after the panel removal: the
    dev tab and audit event must carry the ONE real finding, not the pass
    verdict beside it — while the panel carries neither."""
    validations = [
        {"provider_name": "Dr. A", "rank": 1, "recommendation_adjustments": _REAL_FINDING},
        {"provider_name": "Dr. B", "rank": 2, "recommendation_adjustments": _LIVE_PASS},
    ]
    findings = _judge_findings(_validation(*validations))
    assert findings == [("Dr. A", _REAL_FINDING)]

    markup = _panel(monkeypatch, validations=validations)
    assert "inconsistenc" not in markup


# Change 2 — the panel gets the plain register, never the technical one

def test_panel_shows_plain_explanation_and_never_the_technical_one(monkeypatch):
    markup = _panel(monkeypatch, bias={
        "severity": "medium",
        "detected_biases": ["Top pick is not the highest-rated provider"],
        "explanation": "The top result wins on experience and distance, not on review score.",
        "technical_explanation": "adjusted_rating 4.16 vs 4.72; weighted_contribution -4.07.",
    })
    assert "wins on experience and distance" in markup
    for jargon in ("adjusted_rating", "weighted_contribution", "4.16"):
        assert jargon not in markup


def test_technical_explanation_never_substitutes_for_a_missing_plain_one(monkeypatch):
    """The fallback runs only toward the developer surface. An unconstrained
    string must not reach the panel because the plain field was empty."""
    markup = _panel(monkeypatch, bias={
        "severity": "high", "detected_biases": ["something"],
        "explanation": "",
        "technical_explanation": "adjusted_rating 4.16 drove the ordering.",
    })
    assert "adjusted_rating" not in markup


# Change 3 — the hardcoded section is gone

def test_key_findings_section_is_not_rendered(monkeypatch):
    """Even when the key is populated, the panel must not show it — it restated
    the tile directly above it in worse language."""
    markup = _panel(monkeypatch)
    assert "Key findings" not in markup
    assert "Detected potential biases in ranking methodology" not in markup


# Change 4 — blind spots are OUR gaps, named honestly

def test_blind_spots_render_under_an_honest_heading_without_the_prefix(monkeypatch):
    markup = _panel(monkeypatch, considerations=[
        "Consider Recency and trend of reviews: no weighting for whether feedback is current",
    ])
    assert "What this ranking doesn't capture" in markup
    assert "Before you book" not in markup
    assert "Consider Recency" not in markup
    assert "Recency and trend of reviews" in markup


@pytest.mark.parametrize("raw,expected", [
    ("Consider Recency and trend of reviews", "Recency and trend of reviews"),
    ("consider specialty/scope match", "Specialty/scope match"),
    ("Recency of reviews", "Recency of reviews"),      # no prefix to strip
    ("Consider ", ""),                                  # prefix only -> dropped
])
def test_strip_consider_prefix(raw, expected):
    from app import _strip_consider_prefix
    assert _strip_consider_prefix(raw) == expected


# Change 5 — plural agreement and the footer

@pytest.mark.parametrize("count,expected", [(1, "1 potential bias flagged"),
                                            (3, "3 potential biases flagged")])
def test_bias_tile_pluralizes(monkeypatch, count, expected):
    markup = _panel(monkeypatch, bias={
        "severity": "medium",
        "detected_biases": [f"bias {i}" for i in range(count)],
        "explanation": "Something worth knowing.",
    })
    assert expected in markup


def test_footer_claims_only_what_is_true_and_drops_security_trivia(monkeypatch):
    """The allowlist claim is real (security.py ALLOWED_SPECIALTIES) but applied
    to specialty alone; location is regex-validated. XSS escaping is
    implementation trivia, and unevidenced self-praise undercuts a panel whose
    subject is independent critique."""
    markup = _panel(monkeypatch)
    assert "whitelist" not in markup
    assert "allowlist" in markup
    assert "escaped before rendering" not in markup

    # The critic claim must not outrun what the critic now does. Round 10
    # bounded it to the research budget, so "every ranking" became false for
    # ranks past the cut — the same class of overclaim as the XSS boast above,
    # and this assertion is the reason it was caught.
    assert "every ranking" not in markup

    # Round 13 broke the NEXT version of it. "Every provider we RESEARCHED is
    # reviewed independently" became false on this very page: `not_critiqued`
    # counts a researched provider the critic never returned an entry for, and
    # that count renders in the withheld note a few lines above this footer.
    # The claim is now scoped to what the shortlist gate enforces.
    assert "every provider we researched is reviewed" not in markup
    assert "every provider we recommend has been reviewed independently" in markup


# ---- Round 7: user_guidance stops being computed-and-dropped ----

def _panel_with_guidance(monkeypatch, considerations=(), guidance=()):
    import app as app_module
    captured = []
    monkeypatch.setattr(app_module, "render_html", captured.append)
    app_module.render_validation_insights({
        "agent_outputs": {"critic_validator": {"validation_results": {
            "bias_analysis": {"bias_assessment": {"severity": "low",
                                                  "detected_biases": [], "explanation": ""}},
            "top_provider_validation": {"top_provider_validations": []},
            "final_recommendations": {
                "recommendation_confidence": "high",
                "important_considerations": list(considerations),
                "user_guidance": list(guidance),
            },
        }}},
        "workflow_summary": {},
    })
    return "".join(captured)


def test_user_guidance_now_reaches_the_panel(monkeypatch):
    """It was populated by the critic and rendered nowhere — real patient
    guidance discarded while blind spots were dressed up as guidance."""
    # A realistic EARNED entry. The fixture used to be "Review detailed
    # provider information beyond just rankings" — the unconditional filler
    # string round 10 deleted from the critic — which read as though the panel
    # were being tested against real output when nothing produced it.
    markup = _panel_with_guidance(
        monkeypatch, guidance=["Exercise additional caution in provider selection"])
    assert "Exercise additional caution in provider selection" in markup
    assert "What this ranking doesn't capture" in markup


def test_guidance_and_blind_spots_share_the_section(monkeypatch):
    markup = _panel_with_guidance(
        monkeypatch,
        considerations=["Consider Recency and trend of reviews"],
        guidance=["Exercise additional caution in provider selection"])
    assert "Recency and trend of reviews" in markup
    assert "Exercise additional caution in provider selection" in markup


def test_guidance_duplicating_a_blind_spot_is_not_shown_twice(monkeypatch):
    markup = _panel_with_guidance(
        monkeypatch,
        considerations=["Consider Recency of reviews"],
        guidance=["Recency of reviews", "recency of REVIEWS"])
    assert markup.count("ecency of reviews") == 1


def test_the_section_stays_absent_when_there_is_nothing_to_say(monkeypatch):
    markup = _panel_with_guidance(monkeypatch)
    assert "What this ranking doesn't capture" not in markup


# ---- Round 7: the card says when a distance is city-level ----

def test_card_marks_a_city_level_distance(monkeypatch):
    """A whole pool showing an identical "8.0 mi" must not read as ten
    measurements that happen to agree."""
    from app import render_provider_card
    import app as app_module
    captured = []
    monkeypatch.setattr(app_module, "render_html", captured.append)
    monkeypatch.setattr(app_module, "st", app_module.st)
    try:
        render_provider_card(
            {"name": "Dr. X", "specialty": "Neurology", "location": "Gilbert, AZ",
             "computed_distance_miles": 8.0, "distance_precision": "city",
             "final_score": 80}, 0, {})
    except Exception:
        pass  # Streamlit containers aren't available; the markup is what matters
    assert any("city-level" in c for c in captured)


def test_cost_card_names_the_ring_expansion(monkeypatch):
    """Ring expansion is the one part of a search whose cost varies on a
    decision the code makes silently. Its log line goes to stdout rather than
    `logs/` (only audit.log is written there), and the debug tab read the
    SINGULAR `search_metadata["query"]`, never `query_count` — so a run whose
    Tavily bill doubled looked identical to one that didn't."""
    import app as app_module
    captured = []
    monkeypatch.setattr(app_module, "render_html", captured.append)

    cost = {"total_usd": 0.5, "elapsed_s": 100.0,
            "tavily": {"searches": 15, "credits": 25, "cost_usd": 0.1}}

    app_module.render_cost_card(cost, {"query_count": 5, "ring_expanded": True})
    assert "5 discovery queries" in captured[-1]
    assert "expanded to nearby cities" in captured[-1]

    captured.clear()
    app_module.render_cost_card(cost, {"query_count": 3, "ring_expanded": False})
    assert "3 discovery queries" in captured[-1]
    assert "expanded to nearby cities" not in captured[-1]


def test_cost_card_without_metadata_still_renders():
    """The second argument is optional — the no_results path passes what it has."""
    import app as app_module
    rendered = []
    app_module.render_html = rendered.append
    app_module.render_cost_card({"total_usd": 0.1, "elapsed_s": 5.0,
                                 "tavily": {"searches": 3, "credits": 3, "cost_usd": 0.01}})
    assert "discovery queries" not in rendered[-1]


# ---- the refinement note lives INSIDE the Responsible-AI panel ----


def test_refinement_note_returns_markup_not_render():
    """Markup-returning, so the panel can interpolate it beside the bias and
    judge notes. It used to render as a sibling block immediately after the
    panel's own render_html — visually adjacent, structurally unrelated."""
    import app as app_module
    markup = app_module.refinement_note_markup(
        {"applied": True, "moves": [{"name": "Dr. Alpha", "from": 6, "to": 5,
                                     "reasons": ["stronger platform agreement"]}]}
    )
    assert "Refined by independent critic review" in markup
    assert "Dr. Alpha" in markup
    # Destination plus magnitude, NOT "#from &rarr; #to". This assertion used to
    # require the arrow; both of its numbers are ranks in the full refined pool,
    # which includes withheld providers, so "#6 &rarr; #5" invited the reader to
    # compare against card #5 — a different doctor on the 2026-07-29 run.
    assert "now #5" in markup
    assert "up 1 place" in markup
    assert "&rarr;" not in markup


def test_refinement_note_prefers_the_rank_the_reader_can_see():
    """`display_rank` is the provider's actual position on the page (card
    ordinal, or row in "Other providers considered"); `to` is its rank in the
    full refined pool, which names a position nobody renders. When the
    orchestrator supplies the former, it must win. (The original example used
    display_rank 11 — a position now filtered out of the panel entirely, so
    the principle is pinned with an in-budget move.)"""
    import app as app_module
    markup = app_module.refinement_note_markup(
        {"applied": True, "moves": [{"name": "Dr. Alpha", "from": 7, "to": 2,
                                     "display_rank": 5, "reasons": ["docked"]}]}
    )
    assert "now #5" in markup
    assert "now #2" not in markup


def test_reconciliation_points_at_the_refinement_note_without_duplicating_it():
    """Round 31. On the first always-on-read run (2026-08-11) the caveat's
    "After it: X is now #7, Y is now #8" list repeated, name for name, the
    refinement bullets rendered three lines below — the panel saying one
    thing twice. The caveat now names nobody and points at the refinement
    note, the single source for destinations. It fires only when the prose
    actually contains a numbered position (here "ranked 3rd")."""
    import app as app_module
    line = app_module._reorder_reconciliation({
        "workflow_summary": {"refinement": {"moves": [
            {"name": "Dr. De Lima", "from": 7, "to": 4, "display_rank": 5}
        ]}}
    }, "Dr. De Lima (ranked 3rd) has the strongest reviews.")
    assert "from before the independent critic review" in line
    assert "Refined by independent critic review" in line
    assert "De Lima" not in line
    assert "is now #" not in line


def test_no_caveat_when_the_prose_names_no_positions():
    """THE round-31 bug: the 2026-08-11 clean read said "the top three
    doctors" and "a couple of highly-rated doctors dropped down" — no
    numbered position anywhere — yet the line said "Positions mentioned
    above are from before..." — a caveat about nothing. Group phrasings
    survive the reorder (the researched top of the list is stable), so
    only numbered positions trigger it."""
    import app as app_module
    moves = {"workflow_summary": {"refinement": {"moves": [
        {"name": "Dr. Kan Yu", "from": 9, "to": 7, "display_rank": 7}
    ]}}}
    clean_read = ("The top three doctors are separated by very small margins "
                  "and are all strong choices; a couple of highly-rated "
                  "doctors dropped down the list.")
    assert app_module._reorder_reconciliation(moves, clean_read) == ""
    # ...and each numbered-position shape does trigger it.
    for prose in ("now sits at #3", "ranked 3rd overall", "Rank 2 edges ahead"):
        assert app_module._reorder_reconciliation(moves, prose) != ""


def test_moves_landing_below_the_recommendations_stay_out_of_the_panel():
    """The panel is a patient surface: a provider whose new position is a row
    in "Other providers considered" is, to that reader, an other provider —
    naming them sends the reader hunting below the shortlist for context the
    panel should carry itself. Both note builders read the SAME filter, and
    the headline counts the filtered moves, or the panel contradicts itself
    on one screen ("changed 2" above one bullet)."""
    import app as app_module
    refinement = {"applied": True, "adjusted_count": 2, "moves": [
        {"name": "Dr. Near", "from": 6, "to": 5, "display_rank": 5,
         "reasons": ["docked"]},
        {"name": "Dr. Far", "from": 9, "to": 15, "display_rank": 15,
         "reasons": ["rejected"]},
    ]}

    markup = app_module.refinement_note_markup(refinement)
    assert "Dr. Near" in markup
    assert "Dr. Far" not in markup
    assert "changed 1 recommendation(s)" in markup

    # The caveat reads the same filter: with only below-the-fold moves the
    # visible top positions never moved, so ordinal-bearing prose gets no
    # caveat either.
    only_far = {"workflow_summary": {"refinement": {
        "applied": True, "moves": [refinement["moves"][1]]}}}
    assert app_module._reorder_reconciliation(only_far, "ranked 3rd") == ""


def test_all_moves_below_the_fold_say_so_instead_of_no_changes():
    """Filtering every move must not read as "confirmed the original order" —
    the validator DID act, just on no position this panel shows."""
    import app as app_module
    markup = app_module.refinement_note_markup(
        {"applied": True, "moves": [
            {"name": "Dr. Far", "from": 9, "to": 15, "display_rank": 15,
             "reasons": ["rejected"]},
        ]}
    )
    assert "below the recommendations" in markup
    assert "Dr. Far" not in markup
    assert "confirmed the original order" not in markup


def test_refinement_note_is_silent_when_there_is_nothing_to_report():
    """Same conditional discipline as the other two notes."""
    import app as app_module
    assert app_module.refinement_note_markup({}) == ""
    assert app_module.refinement_note_markup(None) == ""
    assert app_module.refinement_note_markup({"applied": False, "moves": []}) == ""
    # The loop ran and agreed — that IS worth saying.
    assert "confirmed the original order" in app_module.refinement_note_markup(
        {"applied": True, "moves": []}
    )


def test_panel_wiring_carries_the_refinement_note(monkeypatch):
    """WIRING guard. A helper test alone would let `{refinement_note}` be
    deleted from the panel's f-string with the suite still green — the same
    reason the judge note has one."""
    import app as app_module
    captured = []
    monkeypatch.setattr(app_module, "render_html", captured.append)

    app_module.render_validation_insights({
        "agent_outputs": {"critic_validator": {"validation_results": {
            "bias_analysis": {"bias_assessment": {"severity": "low", "detected_biases": []}},
            "top_provider_validation": {"top_provider_validations": []},
            "final_recommendations": {"recommendation_confidence": "high"},
        }}},
        "workflow_summary": {"refinement": {
            "applied": True,
            "moves": [{"name": "Dr. Kuniyoshi", "from": 6, "to": 5,
                       "reasons": ["critic confirmed platform agreement"]}],
        }},
    })

    panel = "".join(captured)
    assert "Refined by independent critic review" in panel
    assert "Dr. Kuniyoshi" in panel
    # Inside the panel card, not a sibling block after it.
    assert panel.index("Responsible AI review") < panel.index("Refined by independent critic review")
    assert "Refined by independent critic review" in panel.split("cc-cost-note")[0]


# ---- Round 13: withheld providers ----

from app import _withheld_note   # noqa: E402


def test_no_withheld_note_when_nothing_was_withheld():
    """A permanent "0 withheld" row trains the eye to skip the row that matters
    — the same reason the hardcoded "0 inconsistencies" line was deleted."""
    assert _withheld_note({}) == ""
    assert _withheld_note({"total": 0, "no_data": 0, "pipeline_failures": 0}) == ""
    # Never-researched providers alone are already explained by the expander's
    # own group caption; this callout is for providers we tried to assess.
    assert _withheld_note({"total": 4, "no_data": 0, "pipeline_failures": 0,
                           "not_researched": 4}) == ""


def test_the_note_separates_our_failures_from_coverage_gaps():
    """Two different claims. "We couldn't find enough about them" is a gap in
    what the web holds; "our scoring didn't finish" is our defect. Collapsing
    them would tell a patient a provider was unverifiable when in fact we simply
    failed to score them."""
    markup = _withheld_note({"total": 3, "no_data": 2, "pipeline_failures": 1})

    assert "couldn't find enough verified information" in markup
    assert "our own scoring didn't finish" in markup
    assert "2 providers were" in markup and "1 provider was" in markup


def test_the_note_carries_counts_but_never_names_or_jargon():
    """Same rule as the judge-consistency note: no provider is named beside an
    admission that our pipeline slipped, and internal stage vocabulary stays on
    the developer surface."""
    markup = _withheld_note({"total": 2, "no_data": 1, "pipeline_failures": 1})

    for jargon in ("not_judged", "not_critiqued", "no_profile_found",
                   "identity_rejected", "over_budget", "enrichment_outcome",
                   "ai_rubric", "critic_review"):
        assert jargon not in markup
    assert "Other providers considered" in markup   # says where to find them


def test_the_note_points_at_where_the_providers_are_listed():
    """Withheld does not mean hidden. If the callout didn't say where they went,
    it would read as a deletion."""
    markup = _withheld_note({"total": 1, "no_data": 1, "pipeline_failures": 0})
    assert "listed under" in markup


# ---- the execution timeline: which agent gets blamed for the wall clock ----

def _timeline_rows(monkeypatch, execution_log) -> list:
    """The step labels the timeline renders, in display order."""
    import app as app_module

    rows = []
    monkeypatch.setattr(app_module.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(app_module.st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(app_module.st, "json", lambda *a, **k: None)
    monkeypatch.setattr(app_module.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(app_module.st, "info", lambda *a, **k: None)

    class _Expander:
        def __init__(self, label, **kwargs):
            rows.append(label)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(app_module.st, "expander", _Expander)
    app_module.render_execution_timeline(execution_log)
    return rows


def _step(step, status, elapsed=None):
    details = {} if elapsed is None else {"elapsed_s": elapsed}
    return {"step": step, "status": status, "timestamp": "t", "details": details}


def test_timeline_names_enrichment_as_its_own_step(monkeypatch):
    """"Preference Scoring — 70.1s" was the longest row on the 2026-07-28
    timeline, and ~54s of it was review enrichment: a Tavily search plus a Haiku
    extraction per provider, i.e. DataGathererAgent work running inside the
    scoring node. The panel named the wrong agent as the bottleneck.

    Asserted on the composed rows, not the step_info dict: a table test would
    stay green if the row were never rendered."""
    rows = _timeline_rows(monkeypatch, [
        _step("gather_data", "started"), _step("gather_data", "completed", 38.9),
        _step("score_providers", "started"),
        _step("enrich_reviews", "started"), _step("enrich_reviews", "completed", 54.0),
        _step("score_providers", "completed", 15.1),
    ])

    assert any("Review Enrichment" in row for row in rows)
    enrichment = next(row for row in rows if "Review Enrichment" in row)
    assert "DataGathererAgent" in enrichment


def test_timeline_orders_enrichment_before_the_scorer(monkeypatch):
    """Enrichment runs INSIDE the scoring node, so its "started" entry lands
    after the scorer's in the log. Grouping by first appearance printed it after
    the step it happens before — a timeline that misstates the order is worse
    than one that omits the row."""
    rows = _timeline_rows(monkeypatch, [
        _step("gather_data", "started"), _step("gather_data", "completed", 38.9),
        _step("score_providers", "started"),
        _step("enrich_reviews", "started"), _step("enrich_reviews", "completed", 54.0),
        _step("score_providers", "completed", 15.1),
        _step("validate_rankings", "started"), _step("validate_rankings", "completed", 36.2),
    ])

    labels = [row for row in rows]
    enrich_at = next(i for i, r in enumerate(labels) if "Review Enrichment" in r)
    score_at = next(i for i, r in enumerate(labels) if "Preference Scoring" in r)
    gather_at = next(i for i, r in enumerate(labels) if "Data Gathering" in r)

    assert gather_at < enrich_at < score_at


# ---- ring expansion: what it bought, not just that it fired ----

def _agent_workflow_captions(monkeypatch, workflow_results) -> list:
    """Every st.caption emitted by the Detailed Agent Analysis surface."""
    import app as app_module

    captions = []
    for name in ("subheader", "markdown", "json", "write", "info", "divider"):
        monkeypatch.setattr(app_module.st, name, lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app_module.st, "caption", lambda text, *a, **k: captions.append(str(text)))
    monkeypatch.setattr(app_module, "render_html", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_render_withheld_detail", lambda *a, **k: None)

    class _Ctx:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(app_module.st, "expander", _Ctx)
    monkeypatch.setattr(app_module.st, "columns", lambda n, *a, **k: [_Ctx() for _ in range(n if isinstance(n, int) else len(n))])
    monkeypatch.setattr(app_module.st, "tabs", lambda labels, *a, **k: [_Ctx() for _ in labels])

    app_module.render_agent_workflow(workflow_results)
    return captions


def test_ring_expansion_reports_what_it_bought(monkeypatch):
    """`ring_expanded` told a developer the ring FIRED. It never said whether
    the two extra searches contributed anything a patient saw — so the cost/
    breadth decision on MIN_CANDIDATE_POOL had no evidence behind it.

    Asserted on the composed captions, not the summary dict: a data-only test
    would stay green if the row were never rendered."""
    captions = _agent_workflow_captions(monkeypatch, {
        "agent_outputs": {"data_gatherer": {"search_metadata": {
            "queries": ["q1", "q2", "q3", "ring1", "ring2"],
            "query_count": 5, "ring_expanded": True,
        }}},
        "workflow_summary": {
            "ring_contribution": {"added": 3, "researched": 2, "shortlisted": 0},
        },
    })

    ring_line = next((c for c in captions if "nearby cities" in c), None)
    assert ring_line is not None, captions
    assert "3 candidate(s) added" in ring_line
    assert "2 researched" in ring_line
    assert "0 reached the recommendations" in ring_line


def test_no_ring_row_when_the_ring_did_not_fire(monkeypatch):
    """A permanent row reading "0 added" would train the eye to skip the row
    that matters — the same doctrine that deleted the hardcoded
    "0 inconsistencies" line from the Responsible-AI panel."""
    captions = _agent_workflow_captions(monkeypatch, {
        "agent_outputs": {"data_gatherer": {"search_metadata": {
            "queries": ["q1", "q2", "q3"], "query_count": 3, "ring_expanded": False,
        }}},
        "workflow_summary": {
            "ring_contribution": {"added": 0, "researched": 0, "shortlisted": 0},
        },
    })

    assert not any("nearby cities" in c for c in captions), captions


# ---- the panel must not contradict the cards it sits above ----

_MOVES = {
    "applied": True,
    "adjusted_count": 1,
    "moves": [
        {"name": "Dr. Mohammad B. Khan, MD", "from": 3, "to": 6,
         "reasons": ["critic marked it 'conditional' (-8)", "2 red flag(s) (-8)"]},
        {"name": "Dr. Julie Lockwood, MD", "from": 7, "to": 4, "reasons": []},
    ],
}


def test_a_displaced_provider_is_not_credited_with_critic_feedback():
    """Dr. Lockwood moved #7 -> #4 with no adjustment of her own — she rose
    because Dr. Khan fell. The fallback string said "critic feedback", which
    asserts the validator said something about her. Same invented causality as
    round 6, on the same panel."""
    markup = refinement_note_markup(_MOVES)

    assert "moved as others were re-scored" in markup
    khan, lockwood = markup.index("Khan"), markup.index("Lockwood")
    assert "critic feedback" not in markup[lockwood:]
    # ...and the provider who WAS adjusted still shows why
    assert "conditional" in markup[khan:lockwood]


def test_the_headline_counts_adjustments_not_rows_that_moved():
    """"re-ordered 4 recommendation(s)" for a run where the critic docked
    exactly one; the other three were that one's wake. `adjusted_count` was
    computed three lines from `moves` in refine_rankings and never read."""
    markup = refinement_note_markup(_MOVES)

    assert "changed 1 recommendation(s)" in markup
    assert "moved 1 more" in markup


def test_the_headline_falls_back_when_adjusted_count_is_absent():
    """Older summaries — and any caller that builds this dict by hand — have no
    `adjusted_count`. Deriving it from the reasons is exact, not an estimate."""
    without = {k: v for k, v in _MOVES.items() if k != "adjusted_count"}

    assert "changed 1 recommendation(s)" in refinement_note_markup(without)


def test_the_bias_note_says_its_positions_predate_the_reordering(monkeypatch):
    """The bias analysis runs BEFORE refine_rankings, so its ordinals describe
    the pre-refinement order — and the panel renders them above cards numbered
    by the final one.

    On 2026-07-28: "Dr. Khan (ranked 3rd) has a higher review score" three
    lines above "Dr. Mohammad B. Khan, MD #3 -> #6", with his card numbered 6.

    Round 29 moved the caveat from the header parenthetical to the closing
    line; round 31 made it POINT at the refinement note instead of
    duplicating its destinations, and gated it on the prose actually
    naming a position — here "(ranked 3rd)" does, so it renders.
    """
    import app as app_module

    captured = []
    monkeypatch.setattr(app_module, "render_html", captured.append)
    app_module.render_validation_insights({
        "agent_outputs": {"critic_validator": {"validation_results": {
            "bias_analysis": {"bias_assessment": {
                "severity": "medium",
                "detected_biases": ["Dr. Khan (ranked 3rd) has a higher review score."],
                "explanation": "This list leans on experience.",
            }},
            "top_provider_validation": {"top_provider_validations": []},
            "final_recommendations": {"recommendation_confidence": "high"},
        }}},
        "workflow_summary": {"refinement": _MOVES},
    })
    markup = "".join(captured)

    assert "from before the independent critic review" in markup
    # The destination lives ONCE, in the refinement note the caveat points
    # at — the old "is now #N" duplication is gone.
    assert "now #6" in markup, "the reader must be able to reconcile the ordinal"
    assert "is now #" not in markup


def test_no_reconciliation_line_when_nothing_moved(monkeypatch):
    """A permanent "nothing changed" line trains the eye to skip the row that
    matters — the doctrine that deleted the hardcoded "0 inconsistencies"."""
    import app as app_module

    captured = []
    monkeypatch.setattr(app_module, "render_html", captured.append)
    app_module.render_validation_insights({
        "agent_outputs": {"critic_validator": {"validation_results": {
            "bias_analysis": {"bias_assessment": {
                "severity": "medium", "detected_biases": ["Something."],
                "explanation": "Explanation.",
            }},
            "top_provider_validation": {"top_provider_validations": []},
            "final_recommendations": {"recommendation_confidence": "high"},
        }}},
        "workflow_summary": {"refinement": {"applied": True, "moves": []}},
    })
    markup = "".join(captured)

    assert "Positions above are from before" not in markup


class TestRatingWithoutCountCount:
    """The Data Gatherer panel's count of pages that gave a rating and no
    count — the shape that puts "— listing page" on a card while the doctor's
    own profile sat in `enrichment_sources` all along."""

    @staticmethod
    def _row(*yields):
        return {"name": "Dr. X", "sources": [
            {"url": f"https://p{i}.com", "kind": "profile", "yielded": y}
            for i, y in enumerate(yields)
        ]}

    def test_counts_a_rating_with_no_count(self):
        from app import _rating_without_count_pages
        coverage = [self._row({"rating": 4.1, "review_count": None})]
        assert _rating_without_count_pages(coverage) == 1

    def test_a_full_pair_is_not_counted(self):
        from app import _rating_without_count_pages
        coverage = [self._row({"rating": 4.5, "review_count": 61})]
        assert _rating_without_count_pages(coverage) == 0

    def test_a_page_that_yielded_nothing_is_not_counted(self):
        """"We fetched it and got nothing" is a different failure with a
        different fix — conflating them is what the field made us untangle."""
        from app import _rating_without_count_pages
        assert _rating_without_count_pages([self._row(None)]) == 0

    def test_a_zero_count_counts_as_missing(self):
        from app import _rating_without_count_pages
        assert _rating_without_count_pages([self._row({"rating": 4.1, "review_count": 0})]) == 1

    def test_counts_pages_not_providers(self):
        from app import _rating_without_count_pages
        coverage = [
            self._row({"rating": 4.1, "review_count": None},
                      {"rating": 3.8, "review_count": None}),
            self._row({"rating": 4.5, "review_count": 61}),
        ]
        assert _rating_without_count_pages(coverage) == 2

    def test_a_cache_hit_with_no_sources_is_survivable(self):
        """`sources` is absent on a cache hit — see `_enrich_one`."""
        from app import _rating_without_count_pages
        assert _rating_without_count_pages([{"name": "Dr. Cached"}]) == 0
        assert _rating_without_count_pages([{"name": "Dr. C", "sources": None}]) == 0
        assert _rating_without_count_pages(None) == 0

    def test_malformed_rows_do_not_raise(self):
        from app import _rating_without_count_pages
        assert _rating_without_count_pages([None, {"sources": ["junk", None]}]) == 0


class TestExtractionPathSplit:
    """The panel's parser-vs-model split. Both paths read the SAME fetched
    pages, so one coverage number cannot say which one came back empty — and
    they need different fixes: markup re-read off a fetched page, versus an
    excerpt window that missed the header. The parser's failures are the silent
    half, because it degrades to the model rather than to an error."""

    @staticmethod
    def _row(*yields):
        return {"name": "Dr. X", "sources": [
            {"url": f"https://p{i}.com", "kind": "profile", "yielded": y}
            for i, y in enumerate(yields)
        ]}

    def test_splits_by_via(self):
        from app import _pages_by_extraction_path
        coverage = [self._row(
            {"rating": 5.0, "review_count": 9, "via": "profile_parser"},
            {"rating": 4.2, "review_count": 30, "via": "llm"},
            {"rating": 3.8, "review_count": None, "via": "profile_parser"},
        )]
        assert _pages_by_extraction_path(coverage) == (2, 1)

    def test_a_row_predating_the_key_counts_as_the_model(self):
        """A cached or older row has no `via`. The model is what produced every
        observation before the parser existed, so that is the honest default —
        the alternative would credit the parser with pages it never read."""
        from app import _pages_by_extraction_path
        assert _pages_by_extraction_path([self._row({"rating": 4.1})]) == (0, 1)

    def test_pages_that_yielded_nothing_are_in_neither_column(self):
        from app import _pages_by_extraction_path
        assert _pages_by_extraction_path([self._row(None, None)]) == (0, 0)

    def test_cache_hits_and_malformed_rows_are_survivable(self):
        from app import _pages_by_extraction_path
        assert _pages_by_extraction_path([{"name": "Dr. Cached"}]) == (0, 0)
        assert _pages_by_extraction_path([None, {"sources": ["junk", None]}]) == (0, 0)
        assert _pages_by_extraction_path(None) == (0, 0)

    def test_the_panel_RENDERS_the_split(self, monkeypatch):
        """Driven through `render_agent_workflow`, not by rebuilding the
        caption here: a helper-only test lets the f-string be deleted with the
        suite still green, and a test that assembles the string it then asserts
        on proves only that Python concatenates."""
        from unittest.mock import MagicMock
        import app as app_module

        captured = []

        class Box(MagicMock):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class Recorder(MagicMock):
            def caption(self, text="", **kwargs):
                captured.append(str(text))

            def markdown(self, text="", **kwargs):
                captured.append(str(text))

            def columns(self, spec, **kwargs):
                width = spec if isinstance(spec, int) else len(spec)
                return [Box() for _ in range(width)]

            def tabs(self, labels, **kwargs):
                return [Box() for _ in labels]

            def expander(self, *args, **kwargs):
                return Box()

        monkeypatch.setattr(app_module, "st", Recorder())
        app_module.render_agent_workflow({
            "agent_outputs": {"data_gatherer": {"status": "success", "search_metadata": {}}},
            "workflow_summary": {"review_coverage": [
                self._row({"rating": 5.0, "review_count": 9, "via": "profile_parser"},
                          {"rating": 4.2, "review_count": 30, "via": "llm"})]},
        })

        assert "1 page(s) read by parser, 1 by model" in "".join(captured)

    def test_the_panel_actually_calls_it(self):
        """The WIRING. Every assertion above passes with the call site deleted
        from the coverage panel — the known failure mode for the
        judge note, one panel over. The coverage block lives inside a Streamlit
        tab, so the composed-markup route used for the Responsible-AI panel is
        not available; this reads the source, which is the same guard
        `test_round_nine` uses for the discovery prompt."""
        import inspect
        import app as app_module

        source = inspect.getsource(app_module.render_agent_workflow)
        assert "_rating_without_count_pages(coverage)" in source, (
            "the coverage caption no longer counts rating-only pages"
        )
        assert "gave a rating with no count" in source


class TestSharedAddressesAreFlagged:
    """Two providers came back on one street address at one distance.

    `location_source` has recorded where an address was read since the
    model-address guard went in, but nothing rendered it — so the one question
    the field exists to answer ("where did this come from?") still could not be
    asked from the UI. The duplicate itself is the visible symptom, so the panel
    names it and points at the provenance beside it.
    """

    ADDRESS = "1450 S Dobson Rd Ste B122, Mesa, AZ 85202"

    def _coverage(self, *locations):
        return [{"name": f"Dr. {i}", "location": loc} for i, loc in enumerate(locations)]

    def test_a_repeated_address_is_returned(self):
        from app import _shared_addresses
        assert _shared_addresses(
            self._coverage(self.ADDRESS, self.ADDRESS, "9 Elm St, Chandler, AZ 85224")
        ) == [self.ADDRESS.lower()]

    def test_distinct_addresses_say_nothing(self):
        """Silence on a clean run — a permanent '0 shared' row trains the eye
        to skip the row that matters."""
        from app import _shared_addresses
        assert _shared_addresses(self._coverage("9 Elm St, Chandler AZ", "4 Oak Rd, Mesa AZ")) == []

    def test_whitespace_and_case_do_not_hide_a_duplicate(self):
        from app import _shared_addresses
        assert _shared_addresses(
            self._coverage("1450 S Dobson Rd,  Mesa, AZ", "1450 s dobson rd, Mesa, AZ")
        ) == ["1450 s dobson rd, mesa, az"]

    def test_missing_and_malformed_rows_are_ignored(self):
        from app import _shared_addresses
        assert _shared_addresses([None, {}, {"location": ""}, {"location": None}]) == []
        assert _shared_addresses(None) == []

    def test_the_panel_RENDERS_the_warning(self, monkeypatch):
        """Driven through `render_agent_workflow` — a helper-only test lets the
        caption be deleted with the suite still green."""
        from unittest.mock import MagicMock
        import app as app_module

        captured = []

        class Box(MagicMock):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class Recorder(MagicMock):
            def caption(self, text="", **kwargs):
                captured.append(str(text))

            def markdown(self, text="", **kwargs):
                captured.append(str(text))

            def columns(self, spec, **kwargs):
                width = spec if isinstance(spec, int) else len(spec)
                return [Box() for _ in range(width)]

            def tabs(self, labels, **kwargs):
                return [Box() for _ in labels]

            def expander(self, *args, **kwargs):
                return Box()

        monkeypatch.setattr(app_module, "st", Recorder())
        app_module.render_agent_workflow({
            "agent_outputs": {"data_gatherer": {"status": "success", "search_metadata": {}}},
            "workflow_summary": {"review_coverage": self._coverage(
                self.ADDRESS, self.ADDRESS, "9 Elm St, Chandler, AZ 85224")},
        })

        rendered = "".join(captured)
        assert "claimed by more than one provider" in rendered
        assert "location_source" in rendered, "the reader is not pointed at the provenance"

    def test_the_panel_actually_calls_it(self):
        """The WIRING, read off the source — the coverage block lives inside a
        Streamlit tab, so the composed-markup route is not available."""
        import inspect
        import app as app_module

        source = inspect.getsource(app_module.render_agent_workflow)
        assert "_shared_addresses(coverage)" in source


def test_review_coverage_carries_the_address_and_its_source():
    """The panel can only show provenance the orchestrator puts in the row.

    Both halves fail silently on their own: drop the key here and the caption
    above has nothing to count, with no error anywhere.
    """
    import inspect
    from agents.orchestrator import ProviderMatchingOrchestrator

    source = inspect.getsource(ProviderMatchingOrchestrator)
    assert '"location": provider.get("location")' in source
    assert '"location_source": provider.get("location_source")' in source


class TestAddressConflictsReachThePanel:
    """The gatherer's location record is worthless if no surface names it."""

    def _coverage(self):
        return [
            {"name": "Dr. Andre Hagevik, MD",
             "address_conflict": {"distinct_count": 2, "addresses": []}},
            {"name": "Dr. Clean", "location": "9 Elm St, Chandler, AZ"},
        ]

    def test_the_helper_names_the_conflicted(self):
        from app import _address_conflicts
        assert _address_conflicts(self._coverage()) == ["Dr. Andre Hagevik, MD"]

    def test_a_clean_run_says_nothing(self):
        from app import _address_conflicts
        assert _address_conflicts([{"name": "Dr. Clean"}]) == []
        assert _address_conflicts(None) == []

    def test_the_panel_RENDERS_the_warning(self, monkeypatch):
        from unittest.mock import MagicMock
        import app as app_module

        captured = []

        class Box(MagicMock):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class Recorder(MagicMock):
            def caption(self, text="", **kwargs):
                captured.append(str(text))

            def markdown(self, text="", **kwargs):
                captured.append(str(text))

            def columns(self, spec, **kwargs):
                width = spec if isinstance(spec, int) else len(spec)
                return [Box() for _ in range(width)]

            def tabs(self, labels, **kwargs):
                return [Box() for _ in labels]

            def expander(self, *args, **kwargs):
                return Box()

        monkeypatch.setattr(app_module, "st", Recorder())
        app_module.render_agent_workflow({
            "agent_outputs": {"data_gatherer": {"status": "success", "search_metadata": {}}},
            "workflow_summary": {"review_coverage": self._coverage()},
        })

        rendered = "".join(captured)
        assert "multiple practice locations" in rendered
        assert "Dr. Andre Hagevik, MD" in rendered
        assert "address_conflict" in rendered, \
            "the reader is not pointed at the JSON field with the detail"

    def test_the_panel_actually_calls_it(self):
        import inspect
        import app as app_module

        source = inspect.getsource(app_module.render_agent_workflow)
        assert "_address_conflicts(coverage)" in source


class TestAddressConflictCardNote:
    """The location set must reach the one reader who can act on it.

    Dr. Kumar's multi-office record rendered only on the developer surface
    while his card asserted one address with no hint the others existed.
    The note names CITIES, not street addresses, and the wording follows
    what actually happened: a completed nearest-trusted selection tells the
    member the CLOSEST office is shown; anything else claims only "more
    than one location". Selection never suppresses the note — choosing the
    nearest office is not a certification the doctor sits there today.
    """

    def _provider(self, *addresses, selection=None, legacy_resolution=None):
        conflict = {
            "addresses": [{"address": a, "source": f"profile_parser:https://x/{i}"}
                          for i, a in enumerate(addresses)],
            "distinct_count": len(addresses),
        }
        if selection is not None:
            conflict["selection"] = selection
        if legacy_resolution is not None:
            # A cached row written by the retired cluster-vote code: it
            # carries "resolution" and no "selection"/"distinct_count".
            conflict.pop("distinct_count")
            conflict["resolution"] = legacy_resolution
        return {"name": "Dr. Harvinder Kumar", "address_conflict": conflict}

    def test_the_note_names_the_cities(self):
        from app import _address_conflict_note
        note = _address_conflict_note(self._provider(
            "1450 S Dobson Rd Ste B122, Mesa, AZ 85202",
            "9321 W Thomas Rd Ste 205, Phoenix, AZ 85037",
        ))

        assert "more than one practice location" in note
        assert "Mesa" in note and "Phoenix" in note
        assert "confirm" in note.lower()
        assert "Dobson" not in note, "street addresses stay on the developer surface"

    def test_a_completed_selection_says_the_closest_is_shown(self):
        """The member deserves to know the address was chosen FOR them and
        that the others exist — "closest to you is shown" is the wording
        that makes the choice legible instead of silently member-relative."""
        from app import _address_conflict_note
        note = _address_conflict_note(self._provider(
            "1450 S Dobson Rd Ste B122, Mesa, AZ 85202",
            "9321 W Thomas Rd Ste 205, Phoenix, AZ 85037",
            selection={"resolved": True, "method": "nearest_trusted",
                       "chosen": "1450 S Dobson Rd Ste B122, Mesa, AZ 85202"},
        ))

        assert "practices at 2 locations" in note
        assert "closest to you is shown" in note
        assert "confirm" in note.lower()

    def test_an_unresolved_selection_claims_only_what_is_true(self):
        """Nothing trusted, or nothing geocodable: the shown address was NOT
        chosen for the member, so the note must not say "closest"."""
        from app import _address_conflict_note
        note = _address_conflict_note(self._provider(
            "1450 S Dobson Rd Ste B122, Mesa, AZ 85202",
            "14520 W Granite Valley Dr, Sun City West, AZ 85375",
            selection={"resolved": False, "reason": "no_trusted_address"},
        ))

        assert "more than one practice location" in note
        assert "closest" not in note

    def test_a_legacy_cached_row_still_notes(self):
        """Rows written by the retired cluster-vote code carry "resolution"
        and no "selection" — for the whole TTL. They fall to the generic
        wording rather than to silence or a false "closest" claim."""
        from app import _address_conflict_note
        note = _address_conflict_note(self._provider(
            "1450 S Dobson Rd Ste B122, Mesa, AZ 85202",
            "9321 W Thomas Rd Ste 205, Phoenix, AZ 85037",
            legacy_resolution={"resolved": True,
                               "chosen": "9321 W Thomas Rd Ste 205, Phoenix, AZ 85037"},
        ))

        assert "more than one practice location" in note
        assert "closest" not in note

    def test_a_same_city_set_keeps_the_note_without_a_parenthetical(self):
        """Several offices in one big city must not lose the note just
        because the city names collapse to one — Phoenix alone spans more
        than many metros."""
        from app import _address_conflict_note
        note = _address_conflict_note(self._provider(
            "100 W Thomas Rd, Phoenix, AZ 85013",
            "9321 W Thomas Rd Ste 205, Phoenix, AZ 85037",
        ))

        assert "more than one practice location" in note
        assert "(" not in note, "one distinct city name earns no parenthetical"

    def test_no_conflict_no_note(self):
        from app import _address_conflict_note
        assert _address_conflict_note({"name": "Dr. Clean"}) == ""
        assert _address_conflict_note({"address_conflict": {"addresses": []}}) == ""

    def test_the_card_actually_renders_it(self):
        """A helper nobody calls is a note nobody reads — same wiring
        discipline as the panel's judge note."""
        import inspect
        import app as app_module

        source = inspect.getsource(app_module.render_provider_card)
        assert "_address_conflict_note(provider)" in source
        assert "note_bits.append(address_note)" in source


def test_review_coverage_carries_tenure_provenance_and_the_conflict():
    """The triage row must answer "where did each ranked number come from" —
    tenure alongside the address, plus the platforms-disagree flag. Dropping
    any key here fails silently: the caption above counts nothing and the JSON
    simply lacks the field."""
    import inspect
    from agents.orchestrator import ProviderMatchingOrchestrator

    source = inspect.getsource(ProviderMatchingOrchestrator)
    assert '"years_experience": provider.get("years_experience")' in source
    assert '"experience_source": provider.get("experience_source")' in source
    assert '"address_conflict": provider.get("address_conflict")' in source


class TestTheUIWritesNoSearchConfig:
    """`get_orchestrator` must not write depth or budget env vars at all.

    Successor to the fast-mode guard (TestFastModeTogglesTheBudgetWithout-
    OwningIt), whose principle — the UI may depart from config but must
    restore it exactly — died with the toggle in 2026-08-07's removal. The
    stronger form survives it: the os.environ channel between the UI and
    Config was the app's proven bug source twice (a hardcoded restore of
    "10" out-voting every deployment's budget; a depth guard that nearly
    cost basic-depth runs an 8x richer extraction payload), so with fast
    mode gone the UI writes NOTHING about search depth or the research
    budget. Reintroducing either write — a new toggle, a helpful override —
    goes red here, not into a deployment hunt.
    """

    def test_get_orchestrator_takes_only_the_fhir_flag(self):
        import inspect
        import app as app_module

        fn = getattr(app_module.get_orchestrator, "__wrapped__", app_module.get_orchestrator)
        assert list(inspect.signature(fn).parameters) == ["fhir_enabled"]

    def test_depth_and_budget_env_are_untouched(self, monkeypatch):
        import os
        from unittest.mock import MagicMock
        import app as app_module

        monkeypatch.setattr(app_module, "create_orchestrator", MagicMock())
        # Sentinels a UI write would clobber; snapshot/restore explicitly so
        # this test cannot leak its sentinels into later Config() reads (the
        # exact hygiene bug the retired fast-mode test documented).
        originals = {
            key: os.environ.pop(key, None)
            for key in ("TAVILY_SEARCH_DEPTH", "MAX_PROVIDERS_TO_ENRICH")
        }
        os.environ["TAVILY_SEARCH_DEPTH"] = "sentinel-depth"
        os.environ["MAX_PROVIDERS_TO_ENRICH"] = "99"
        try:
            fn = getattr(app_module.get_orchestrator, "__wrapped__", app_module.get_orchestrator)
            fn(False)
            assert os.environ["TAVILY_SEARCH_DEPTH"] == "sentinel-depth"
            assert os.environ["MAX_PROVIDERS_TO_ENRICH"] == "99"
        finally:
            for key, value in originals.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class TestJoinCacheInventory:
    """`_join_cache_inventory` — stored rows annotated with THIS run's view.

    The open cache defect: a repeat search reused 1 of an expected ~5, and
    naming the drifted component (name variant vs discovery city) took a
    manual diff of two runs' coverage panels. The join puts the stored basis
    and this run's basis on one row, so a miss carries its own diagnosis.
    """

    @staticmethod
    def _stored(name, basis, key=None):
        from utils.provider_key import basis_cache_key

        return {
            "provider_key": key or basis_cache_key(basis),
            "name": name,
            "stored_basis": basis,
            "expired": False,
        }

    def test_key_match_names_the_provider_and_its_outcome(self):
        import app as app_module

        rows, counts = app_module._join_cache_inventory(
            [self._stored("Dr. Andrea An, MD", "an andrea|chandler az")],
            [{"name": "Dr. Andrea An, MD", "outcome": "cached",
              "cache_basis": "an andrea|chandler az"}],
        )

        assert counts == {"key_matches": 1, "basis_drift": 0}
        assert "Dr. Andrea An, MD" in rows[0]["this_run"]
        assert "cached" in rows[0]["this_run"]

    def test_basis_drift_shows_both_bases_side_by_side(self):
        """Same physician, different key — the exact shape the 7-of-8-miss
        investigation is hunting. The annotation must carry BOTH basis
        strings so the drifted component (here: a middle initial the name
        variant kept) is readable without a second run."""
        import app as app_module

        rows, counts = app_module._join_cache_inventory(
            [self._stored("Andrea M An, MD", "an andrea m|chandler az")],
            [{"name": "Dr. Andrea An", "outcome": "enriched",
              "cache_basis": "an andrea|chandler az"}],
        )

        assert counts == {"key_matches": 0, "basis_drift": 1}
        note = rows[0]["this_run"]
        assert "DIFFERENT key" in note
        assert "an andrea m|chandler az" in note
        assert "an andrea|chandler az" in note

    def test_unrelated_rows_carry_no_annotation(self):
        """A row from some other city's search is not evidence about this
        run — annotating it would bury the two rows that are."""
        import app as app_module

        rows, counts = app_module._join_cache_inventory(
            [self._stored("Dr. John Smith, MD", "john smith|phoenix az")],
            [{"name": "Dr. Andrea An", "outcome": "enriched",
              "cache_basis": "an andrea|chandler az"}],
        )

        assert counts == {"key_matches": 0, "basis_drift": 0}
        assert "this_run" not in rows[0]

    def test_single_token_names_only_drift_against_single_token_names(self):
        """Same rule as provider dedupe: a bare surname overlaps 1.0 with any
        full name sharing it, so mixed-cardinality pairs never merge — a
        stored "An" must not claim this run's "Dr. Andrea An"."""
        import app as app_module

        rows, counts = app_module._join_cache_inventory(
            [self._stored("An", "an|chandler az")],
            [{"name": "Dr. Andrea An", "outcome": "enriched",
              "cache_basis": "an andrea|chandler az"}],
        )

        assert counts == {"key_matches": 0, "basis_drift": 0}
        assert "this_run" not in rows[0]

    def test_the_panel_actually_renders_the_inventory(self):
        """Wiring guard, both halves: a helper-only test would let the render
        block be deleted with the suite still green. The Data Gatherer tab
        must call the store's inventory() and join it via
        _join_cache_inventory."""
        import inspect
        import app as app_module

        source = inspect.getsource(app_module.render_agent_workflow)
        assert "get_vector_store().inventory()" in source
        assert "_join_cache_inventory(" in source
        assert "Provider cache inventory" in source


class TestPositiveOnlyMoverWording:
    """A mover whose only entry is the near-uniform +2 gets displacement
    phrasing, per the fallback recorded at _CONFIDENCE_ADJUSTMENT.

    First sighting 2026-08-07 (the first Opus 5 run): Kumar's bullet read
    "now #4 (up 2 places) — high critic confidence (+2)", crediting the +2
    for a climb caused by two -8s above him — Vandian climbed the SAME two
    places with a 0 adjustment. The +2 stays visible (the user's R3 call:
    it is earned signal and it moves scores) but as an endorsement beside
    the move, never as its cause; and the headline stops counting such
    movers as "changed" — the same no-variance-is-not-a-finding rule that
    keeps the +2 out of refinement_findings and adjusted_count.
    """

    # The exact four-move shape of the 2026-08-07 run.
    _RUN = {"applied": True, "moves": [
        {"name": "Dr. Harvinder Kumar", "from": 6, "to": 4, "display_rank": 4,
         "reasons": ["high critic confidence (+2)"]},
        {"name": "Dr. Vardges Vandian, DO", "from": 7, "to": 5, "display_rank": 5,
         "reasons": []},
        {"name": "Dr. Yeeshu Arora", "from": 5, "to": 6, "display_rank": 6,
         "reasons": ["critic marked it 'conditional' (-8)",
                     "high critic confidence (+2)"]},
        {"name": "Dr. Brian Lep Rabin, MD", "from": 4, "to": 7, "display_rank": 7,
         "reasons": ["critic marked it 'conditional' (-8)",
                     "1 red flag(s) from critic review (-4)",
                     "high critic confidence (+2)"]},
    ]}

    def test_the_plus_two_is_an_endorsement_not_a_cause(self):
        import app as app_module

        markup = app_module.refinement_note_markup(self._RUN)
        kumar = markup[markup.index("Kumar"):markup.index("Vandian")]

        assert "moved as others were re-scored" in kumar
        assert "critic endorsement" in kumar
        assert "high critic confidence (+2)" in kumar, "the +2 stays visible"
        # The old shape — the reasons list as the bullet's sole detail,
        # em-dash asserting causation — must be gone for this mover.
        assert "(up 2 places) — high critic confidence" not in markup

    def test_docked_movers_keep_causal_wording(self):
        import app as app_module

        markup = app_module.refinement_note_markup(self._RUN)
        arora = markup[markup.index("Arora"):markup.index("Rabin")]

        assert "conditional" in arora
        assert "moved as others were re-scored" not in arora

    def test_the_headline_counts_docked_movers_only(self):
        """"changed 3" counted Kumar; the docked movers number 2."""
        import app as app_module

        markup = app_module.refinement_note_markup(self._RUN)

        assert "changed 2 recommendation(s)" in markup
        assert "moved 2 more" in markup

    def test_an_unclassifiable_reason_keeps_the_old_causal_branch(self):
        """The predicate is "every entry provably positive", not "any entry
        negative": a reason with no sign marker stays in the causal branch
        and still counts as a change — the conservative default for a shape
        production has never emitted."""
        import app as app_module

        markup = app_module.refinement_note_markup({"applied": True, "moves": [
            {"name": "Dr. Odd", "from": 6, "to": 5, "display_rank": 5,
             "reasons": ["docked"]},
        ]})

        assert "changed 1 recommendation(s)" in markup
        assert "critic endorsement" not in markup


class TestInFlightSearchGuard:
    """One search at a time, and reruns reattach instead of orphaning it.

    2026-08-08: the owner toggled a sidebar switch mid-search. The rerun
    killed the page script; the workflow thread kept running headless
    (invisible, unstoppable, still billing); the redrawn page looked idle;
    the natural second click launched a second complete workflow — ~$0.50
    duplicated and a cost card merging two runs. The job (future + queue +
    pool) now lives in session state so a rerun reattaches to it, and a
    click while one is running starts nothing.
    """

    @staticmethod
    def _fake_orchestrator(result):
        from unittest.mock import MagicMock

        orchestrator = MagicMock()

        def run(specialty, location, preferences, progress_callback, use_cache):
            progress_callback({"agent_name": "DataGathererAgent",
                               "action": "searching", "progress_percentage": 30})
            progress_callback({"agent_name": "CriticValidatorAgent",
                               "action": "validating", "progress_percentage": 90})
            return result

        orchestrator.execute_workflow_streaming.side_effect = run
        return orchestrator

    def test_job_round_trip_renders_progress_and_returns_results(self):
        from unittest.mock import MagicMock
        import app as app_module

        result = {"success": True, "final_recommendations": []}
        job = app_module._start_search_job(
            self._fake_orchestrator(result),
            {"specialty": "Neurology", "location": "Chandler, AZ",
             "preferences": {}, "use_cache": True},
        )
        status, progress = MagicMock(), MagicMock()

        assert app_module._drain_search_job(job, status, progress) is result
        assert status.write.call_count >= 2, "progress must reach the widgets"
        assert job["pool"]._shutdown, "the worker pool is released on harvest"

    def test_a_second_drain_of_the_same_job_reattaches(self):
        """The rerun path: the widgets die with the old script run, the job
        does not — a fresh drain against fresh widgets must still hand back
        the same result rather than raising or hanging."""
        from unittest.mock import MagicMock
        import app as app_module

        result = {"success": True}
        job = app_module._start_search_job(
            self._fake_orchestrator(result),
            {"specialty": "Neurology", "location": "Chandler, AZ",
             "preferences": {}},
        )
        app_module._drain_search_job(job, MagicMock(), MagicMock())

        assert app_module._drain_search_job(job, MagicMock(), MagicMock()) is result

    def test_main_prefers_the_running_job_over_starting_another(self):
        """Wiring guard: the button path must consult the stash BEFORE
        creating a job, say out loud that no second search started, and the
        drain must key on the stashed job — a helper-only test would let the
        guard be unwired with the suite green."""
        import inspect
        import app as app_module

        source = inspect.getsource(app_module.main)
        assert 'st.session_state.get("search_job")' in source
        assert "A search is already running" in source
        assert "_start_search_job(" in source
        assert "_drain_search_job(" in source
        assert "execute_with_live_progress" not in source


# ---------------------------------------------------------------------------
# Round 26: the UI-truth set + the simulated network check.

import inspect
from types import SimpleNamespace
from urllib.parse import urlparse

import app


class TestAgentModelLabelsDeriveFromConfig:
    """The How-it-works strip said "Claude Sonnet 5" for the critic while the
    cost card on the same page printed claude-opus-4-8 — the strip predated
    the label dict and was never rewired — and the dict itself was
    hand-maintained: the 2026-08-09 model revert had to hand-edit it, and an
    env-var flip of CRITIC_MODEL (the designed rollback lever) would have
    left the UI naming a model no call ever used. Labels now derive from the
    same config the agents read, so the strip cannot disagree with the cost
    card again."""

    @staticmethod
    def _stub(critic="claude-opus-5"):
        return SimpleNamespace(
            GATHERER_MODEL="claude-haiku-4-5",
            JUDGE_MODEL="gpt-5.6-terra",
            CRITIC_MODEL=critic,
        )

    def test_labels_follow_the_config(self, monkeypatch):
        monkeypatch.setattr(app, "get_config", lambda: self._stub("claude-opus-5"))
        assert app.agent_models()["CriticValidatorAgent"] == "Claude Opus 5"
        monkeypatch.setattr(app, "get_config", lambda: self._stub("claude-opus-4-8"))
        assert app.agent_models()["CriticValidatorAgent"] == "Claude Opus 4.8"

    def test_an_unknown_model_id_is_shown_raw_not_guessed(self, monkeypatch):
        """The raw ID is the honest fallback: it can never contradict the
        cost card, which prints raw IDs."""
        monkeypatch.setattr(app, "get_config", lambda: self._stub("claude-nova-9"))
        assert app.agent_models()["CriticValidatorAgent"] == "claude-nova-9"

    def test_the_pipeline_strip_renders_the_configured_critic(self, monkeypatch):
        monkeypatch.setattr(app, "get_config", lambda: self._stub("claude-opus-5"))
        strip = app._pipeline_strip_html()
        assert "Claude Opus 5" in strip
        assert "Sonnet" not in strip
        monkeypatch.setattr(app, "get_config", lambda: self._stub("claude-opus-4-8"))
        assert "Claude Opus 4.8" in app._pipeline_strip_html()

    def test_the_header_and_progress_lines_are_wired_to_the_helpers(self):
        """Wiring guard both ways: pasting model literals back into the strip
        (or the progress line) would leave every helper test green."""
        header_src = inspect.getsource(app.render_header)
        assert "_pipeline_strip_html()" in header_src
        for model_token in ("Sonnet", "Opus", "Haiku", "Terra", "4.5", "4.8"):
            assert model_token not in header_src, model_token
        assert "agent_models()" in inspect.getsource(app._drain_search_job)


class TestSearchFormCaptionTellsTheMeasuredTruth:
    """"about 30–60 seconds and costs a few cents" rendered directly above a
    cost card reading "$0.6671 / 144.1s" on the 2026-07-28 review run —
    photographed in one frame, the first promise the demo broke. The caption
    states the measured band; round 29 tightened it to the owner's "50–60
    cents" wording after the 2026-08-09/10 run pair measured $0.5592 cold /
    $0.3497 warm — the earlier "$0.40–0.70" span was wide enough to read
    as a hedge. The pocket-change lie stays banned either way."""

    def test_the_old_lies_are_gone_and_the_measured_band_is_stated(self):
        src = inspect.getsource(app.render_search_form)
        assert "a few cents" not in src
        assert "30–60 seconds" not in src
        assert "about a minute" in src
        assert "50–60 cents" in src
        assert "$0.40" not in src      # the superseded band must not linger


class TestUnderTheHoodNamesTheEngineering:
    """ChromaDB, the code-computed geo pipeline and the test suite were
    invisible on screen — a technical reviewer saw agents and scores while
    the engineering lived only in the repo. The guardrail clause is scoped
    to exactly what is built (input sanitization + the specialty allowlist):
    scraped-page injection defense is an OPEN hole and must not be claimed."""

    def test_the_expander_names_the_stack(self):
        src = inspect.getsource(app.render_header)
        assert "ChromaDB" in src
        assert "never estimated by a model" in src
        assert "allowlist" in src
        # Floors with headroom, owner-bumped at round 33 (suite was 1,237,
        # rounds 32): the stale floor must not linger anywhere in the
        # function, comments included.
        assert "1,000+" in src
        assert "950+" not in src

    def test_no_injection_defense_is_claimed(self):
        assert "injection" not in inspect.getsource(app.render_header).lower()


class TestTheGithubLinkLandsOnARepo:
    """"View the project on GitHub" is the single most likely click a
    technical reviewer makes, and it dead-ended on the profile page for
    eleven days after the curated public repo existed. The default must be
    repo-shaped: owner plus repository, never a bare profile."""

    def test_the_default_is_a_repo_not_a_profile(self):
        path = urlparse(app.PORTFOLIO_GITHUB_URL).path.strip("/")
        assert path.count("/") >= 1, app.PORTFOLIO_GITHUB_URL


class TestNetworkCheckHonestyRails:
    """Simulated coverage must say so wherever it appears: a bare
    "in-network" against a real physician's name from invented data is
    fabricated coverage. Both chip surfaces and the caption carry the label;
    the card chip renders BOTH states because a mixed pool is the demo's
    point and a card with no chip reads as "never checked"; the patient
    caption never names an env var."""

    def test_panel_chips_carry_the_simulated_label(self):
        verified = {"status": "verified", "source": "simulated", "payer": "Aetna"}
        no_record = {"status": "no_record", "source": "simulated", "payer": "Aetna"}
        verified_chip = app._network_panel_chip("Dr. A", verified)
        no_record_chip = app._network_panel_chip("Dr. A", no_record)
        assert "simulated" in verified_chip and "in network" in verified_chip
        assert "simulated" in no_record_chip and "not in this plan" in no_record_chip

    def test_real_mode_chips_are_unchanged(self):
        chip = app._network_panel_chip("Dr. A", {"status": "verified", "source": "live"})
        assert "simulated" not in chip and "in-network" in chip

    def test_the_simulated_caption_is_patient_copy(self):
        note = app._network_check_note("simulated")
        assert "simulated for this demo" in note
        assert "Plan-Net" in note
        assert "FHIR_USE_MOCK" not in note  # env vars are developer copy
        assert "confirm" not in note.lower()  # the card already says it once

    def test_sandbox_and_live_captions_survive(self):
        assert "FHIR_USE_MOCK" in app._network_check_note("sandbox")
        assert "live FHIR directory" in app._network_check_note(None)

    def test_card_chip_renders_both_simulated_states_labeled(self):
        verified = {"status": "verified", "source": "simulated", "payer": "Aetna"}
        no_record = {"status": "no_record", "source": "simulated", "payer": "Aetna"}
        verified_chip = app._network_card_chip(verified)
        no_record_chip = app._network_card_chip(no_record)
        assert "simulated" in verified_chip and "Aetna" in verified_chip
        assert no_record_chip and "simulated" in no_record_chip
        # Real mode unchanged: a non-sandbox no_record still renders nothing.
        assert app._network_card_chip({"status": "no_record", "source": "live"}) == ""

    def test_the_panel_and_card_wiring_use_the_helpers(self):
        panel_src = inspect.getsource(app.render_network_check)
        # The exact branch expression, not the bare token: the docstring also
        # says FHIR_USE_MOCK, so a token search stayed green with the branch
        # replaced by `if False:` (caught by the revert-in-isolation pass).
        assert "if get_config().FHIR_USE_MOCK:" in panel_src
        assert "simulate_network_batch(" in panel_src
        assert "_network_panel_chip" in panel_src
        assert "_network_check_note" in panel_src
        assert "_network_card_chip" in inspect.getsource(app.render_provider_card)


class TestCosmeticTruthPass:
    """Round 26b, from the owner's deployed-run review: the finished bar, the
    tab icon, tooltip accuracy, the retired cache toggle, and How-it-works
    carrying the resume's claims."""

    def test_main_backstops_the_finished_bar(self):
        """The orchestrator emits the final 100; this is the belt-and-braces
        so a dropped event can never again strand a 'complete' header above
        an unfinished bar."""
        assert "progress_bar.progress(100)" in inspect.getsource(app.main)

    def test_the_page_icon_is_the_compass_mark(self):
        icon = app._page_icon()
        assert icon.endswith("logo.svg"), icon
        with open(icon, encoding="utf-8") as f:
            content = f.read()
        assert content.lstrip().startswith("<svg")
        # Wiring: the header actually uses the helper.
        assert "page_icon=_page_icon()" in inspect.getsource(app.render_header)

    def test_the_cache_toggle_is_retired(self):
        """Always-on made the toggle surface noise: its default was the only
        state anyone ever saw. The controls that remain are the clear-cache
        button (user) and the TTL env knob (dev). The old label is banned
        from the whole module so the toggle cannot quietly return."""
        full_source = inspect.getsource(app)
        assert "Use cached provider data" not in full_source
        assert '"use_cache"' not in inspect.getsource(app.render_search_form)

    def test_stale_tooltip_claims_are_gone(self):
        """The radius help said "not searched or ranked" — wrong on both
        verbs (discovery always runs the same three searches; unmeasurable
        distances are never dropped) — and the clear-cache help said "a few
        cents" for a measured ~20-cent cold delta ($0.5699 cold vs $0.3845
        warm), the same understatement the run caption was called out for."""
        form_src = inspect.getsource(app.render_search_form)
        assert "not searched or ranked" not in form_src
        assert "measurably farther" in form_src
        main_src = inspect.getsource(app.main)
        assert "a few cents" not in main_src
        assert "20 cents" in main_src

    def test_how_it_works_carries_the_resume_claims(self):
        """Every load-bearing resume claim needs an on-screen counterpart —
        the paragraph previously omitted the cross-family critic, the
        Responsible-AI panel, the cost/latency card, and the hardening
        rounds, which lived only in the repo."""
        src = inspect.getsource(app.render_header)
        assert "different model family" in src
        assert "Responsible-AI panel" in src
        assert "per-agent traces" in src
        assert "cites review evidence for every criterion" in src
        # Floor bumped at round 33 (32 rounds shipped); the stale "20+"
        # must not linger in the function, comments included.
        assert "30+" in src and "field-test-and-fix rounds" in src
        assert "20+" not in src


class TestLocationPickerWiring:
    """Rounds 28–29: the WHERE row is selection-only, inside ONE card.

    Round 28 made every location field dataset-backed but had to leave the
    pickers OUTSIDE st.form (a form batches widget state until submit, so
    the city list could never react to a state change) — and the owner
    read the result as two disconnected boxes. Round 29 retired the form
    instead of the pickers: one st.container(border=True) card, a plain
    st.button, and the ZIP promoted from text_input to a selectbox fed by
    zips_for_city — so a mismatched ZIP is unreachable from the UI, not
    merely caught. sanitize_location stays as the server-side enforcement
    for every non-UI path, exactly the specialty pattern."""

    def test_the_form_is_retired_for_one_bordered_card(self):
        """st.form and dependent selectboxes are mutually exclusive, so the
        form went, not the pickers. The submit control must be st.button —
        the form-scoped submit widget raises outside a form context. The
        container assert carries the key: the docstring also says
        "st.container(border=True)", so a looser assert would stay green
        with the call itself deleted."""
        src = inspect.getsource(app.render_search_form)
        assert "st.form(" not in src
        assert "form_submit_button" not in src
        assert 'st.container(border=True, key="search_card")' in src
        assert "_location_picker()" in src
        assert "st.button(" in src

    def test_the_card_key_matches_the_theme_hook(self):
        """The Hearth card styling moved from [data-testid="stForm"] to the
        st-key-search_card class Streamlit derives from the container's
        key. The two live in different files, so renaming either side
        silently un-styles the card — this is the only thing binding
        them."""
        from utils.theme import HEARTH_CSS
        assert ".st-key-search_card" in HEARTH_CSS
        assert "stForm" not in HEARTH_CSS

    def test_the_picker_is_selection_only_and_dataset_backed(self):
        """No free-text field survives in the picker: state, city AND zip
        are all selectboxes drawing from the vendored GeoNames dataset."""
        src = inspect.getsource(app._location_picker)
        assert "known_states()" in src
        assert "cities_for_state(" in src
        assert "zips_for_city(" in src
        assert "text_input" not in src
        assert "Location (City, State ZIP)" not in inspect.getsource(app.render_search_form)

    def test_the_submit_path_keeps_the_server_side_allowlist(self):
        """Round 28's UI-side ZIP↔city precheck was DELETED in round 29 —
        the ZIP dropdown offers only in-city ZIPs, making the mismatch it
        caught unreachable from the UI, and a check for an unreachable
        state is banned by house rule. sanitize_location remains the
        enforcement for every other path and must stay on the submit
        path."""
        src = inspect.getsource(app.render_search_form)
        assert "city_state_for_zip(" not in src
        assert "sanitize_location(" in src

    def test_under_the_hood_claims_every_field_now(self):
        """The copy upgrade the allowlist earns: from "specialties checked
        against an allowlist" to every search field."""
        src = inspect.getsource(app.render_header)
        assert "every search field checked against an allowlist" in src
        assert "ZIP verified against the chosen city" in src


# ---------------------------------------------------------------------------
# Round 29: the owner's deployed-run pass (docked risers, tile scope, panel
# copy, status persistence, motion smoothing) plus the run-pair gaps.


class TestDockedRiserWording:
    """Both 2026-08-09 live runs rendered "now #4 (up 27 places) — critic
    marked it 'conditional' (-8)...": penalties cannot explain an upward
    move, but the em-dash asserted they did. The rise is the partition
    (researched providers sort above never-researched ones) plus others'
    drops; the findings must stay visible — they DID lower the provider's
    own score — but as findings, never as the cause of the rise."""

    _DOCKED_RISER = {"applied": True, "moves": [
        {"name": "Dr. Riser", "from": 31, "to": 4, "display_rank": 4,
         "reasons": ["critic marked it 'conditional' (-8)"]},
    ]}

    def test_a_docked_riser_gets_displacement_framing_with_findings(self):
        markup = refinement_note_markup(self._DOCKED_RISER)
        assert "up 27 places" in markup
        assert "moved up as unresearched providers were set aside" in markup
        assert "still lowered this provider&#x27;s own score" in markup \
            or "still lowered this provider's own score" in markup
        # The finding itself stays visible — hiding it would discard the
        # critic's work to fix a grammar problem.
        assert "conditional" in markup

    def test_a_docked_faller_keeps_plain_causal_framing(self):
        """Downward moves are the case the em-dash grammar was written for:
        their own penalty moved them, and naming it as the cause is fair."""
        markup = refinement_note_markup({"applied": True, "moves": [
            {"name": "Dr. Faller", "from": 2, "to": 5, "display_rank": 5,
             "reasons": ["critic marked it 'conditional' (-8)"]},
        ]})
        assert "down 3 places" in markup
        assert "moved up as unresearched providers were set aside" not in markup
        assert "conditional" in markup

    def test_docked_risers_still_count_as_changed(self):
        """The headline counts DOCKED providers; the displacement framing of
        the movement must not reclassify a docked riser as a mere
        displacement."""
        markup = refinement_note_markup(self._DOCKED_RISER)
        assert "changed 1 recommendation(s)" in markup

    def test_the_no_extra_cost_claim_is_gone(self):
        """Owner call (round 29): the closing clause described
        refine_rankings being free post-processing but read as a claim
        about the critic review itself — the run's single most expensive
        call. Banned from the note and from the function source, comments
        included."""
        markup = refinement_note_markup(self._DOCKED_RISER)
        assert "no extra API cost" not in markup
        assert "no extra API cost" not in inspect.getsource(app.refinement_note_markup)


class TestRedFlagsTileScope:
    """The 2026-08-09 run pair: the tile read "5 raised on top picks" while
    all five flags sat on providers ranked #6-#8 and every actual card said
    "Critic approved". The tile now counts flags from the providers ON the
    cards (workflow_results["final_recommendations"]), not from every
    validation the critic returned."""

    @staticmethod
    def _panel_with_flags(monkeypatch, recommendations):
        import app as app_module

        captured = []
        monkeypatch.setattr(app_module, "render_html", captured.append)
        app_module.render_validation_insights({
            "agent_outputs": {"critic_validator": {"validation_results": {
                "bias_analysis": {"bias_assessment": {"severity": "low", "detected_biases": []}},
                "top_provider_validation": {"top_provider_validations": [
                    # The critic's own validation list still carries flags for
                    # a provider BELOW the shortlist — the exact shape that
                    # produced the wrong count.
                    {"provider_name": "Dr. Below", "rank": 7,
                     "red_flags": ["long waits", "billing disputes"]},
                ]},
                "final_recommendations": {"recommendation_confidence": "high"},
            }}},
            "workflow_summary": {},
            "final_recommendations": recommendations,
        })
        return "".join(captured)

    def test_flags_below_the_shortlist_do_not_count(self, monkeypatch):
        markup = self._panel_with_flags(monkeypatch, recommendations=[
            {"provider": {"name": "Dr. Card", "critic_review": {"red_flags": []}}},
        ])
        assert "None raised" in markup
        assert "raised on top picks" not in markup

    def test_flags_on_a_card_still_count(self, monkeypatch):
        markup = self._panel_with_flags(monkeypatch, recommendations=[
            {"provider": {"name": "Dr. Card",
                          "critic_review": {"red_flags": ["billing disputes"]}}},
        ])
        assert "1 raised on top picks" in markup


class TestPanelCopyRound29:
    """Owner's #5: the header parenthetical goes, the caveat moves to the
    bottom line, and the note names the critic review as independent."""

    def test_the_bias_header_parenthetical_is_gone(self):
        src = inspect.getsource(app.render_validation_insights)
        assert "read before the independent" not in src
        assert "re-ordered the list" not in src

    def test_the_reconciliation_line_carries_the_caveat_at_the_bottom(self):
        """Round 31 rewording: the caveat fires only for prose that names a
        numbered position, and points at the refinement note instead of
        duplicating its "After it" destinations."""
        line = app._reorder_reconciliation({
            "workflow_summary": {"refinement": {"moves": [
                {"name": "Dr. Moved", "from": 6, "to": 5, "display_rank": 5}
            ]}}
        }, "Dr. Moved (ranked 3rd) leads on reviews.")
        assert "from before the independent critic review" in line
        assert "Refined by independent critic review" in line
        assert "After it:" not in line


class TestStatusPersistence:
    """Owner's #3: enabling the network-check toggle reran the script and the
    "agents are working" expander — the run's step-by-step record —
    vanished, because the status widget only existed while a job was being
    drained. The drain now keeps every rendered line on the job, every
    harvest outcome snapshots (label, state, bar, lines) into
    session_state, and a job-less rerun replays the snapshot as a
    collapsed completed st.status."""

    def test_the_drain_accumulates_lines_on_the_job(self):
        src = inspect.getsource(app._drain_search_job)
        assert 'job.setdefault("lines", [])' in src
        assert 'job["progress"]' in src

    def test_the_snapshot_captures_label_state_and_lines(self):
        import streamlit as st

        job = {"lines": ["**DataGathererAgent** — searching"], "progress": 40}
        app._snapshot_run_progress(job, "Search complete in 52.5s", "complete",
                                   progress=100)
        snap = st.session_state.last_run_progress
        assert snap["label"] == "Search complete in 52.5s"
        assert snap["state"] == "complete"
        assert snap["progress"] == 100
        assert snap["lines"] == ["**DataGathererAgent** — searching"]
        # The snapshot must COPY the lines: the job dict dies with the pool.
        assert snap["lines"] is not job["lines"]

    def test_the_snapshot_defaults_to_the_jobs_last_progress(self):
        import streamlit as st

        app._snapshot_run_progress({"lines": [], "progress": 55},
                                   "Search failed", "error")
        assert st.session_state.last_run_progress["progress"] == 55

    def test_every_harvest_outcome_snapshots_and_the_rerun_replays(self):
        """Wiring: all four status.update sites snapshot (success,
        no-results, generic failure, exception), and main's job-less branch
        replays. A helper-only test would let any of the five call sites
        be deleted with the suite green."""
        src = inspect.getsource(app.main)
        assert src.count("_snapshot_run_progress(") == 4
        assert "_render_last_run_status()" in src

    def test_the_replay_renders_the_snapshot(self, monkeypatch):
        import streamlit as st

        writes, statuses = [], []

        class _Status:
            def __init__(self, label, state=None, expanded=None):
                statuses.append((label, state, expanded))

            def progress(self, value):
                writes.append(("progress", value))

            def write(self, line):
                writes.append(("write", line))

        st.session_state.last_run_progress = {
            "label": "Search complete in 52.5s", "state": "complete",
            "progress": 100, "lines": ["**Agent** — step one"],
        }
        monkeypatch.setattr(app.st, "status", _Status)
        app._render_last_run_status()

        assert statuses == [("Search complete in 52.5s", "complete", False)]
        assert ("progress", 100) in writes
        assert ("write", "**Agent** — step one") in writes

    def test_the_replay_is_silent_without_a_snapshot(self, monkeypatch):
        import streamlit as st

        st.session_state.last_run_progress = None
        called = []
        monkeypatch.setattr(app.st, "status",
                            lambda *a, **k: called.append(a))
        app._render_last_run_status()
        assert called == []


class TestExpanderAnimationRetired:
    """Round 33: the expander open animation is GONE, banned by token.

    Three rounds tried to tame it — round 29 shipped the height
    transition, round 30 cut it to height-only after Chrome repainted
    opening text garbled, round 31 disabled scroll anchoring on the
    measured scroll container to stop the dip-and-spring — and on
    2026-08-11 the owner reported both the text flicker and the bounce
    STILL alive on the deployed build. Animating layout height
    re-rasterizes the text under it every frame and makes the browser
    re-derive scroll position mid-flight; each patch moved that fight
    instead of ending it. Instant open has neither failure mode. The
    bans cover the whole sheet, comments included (the round-31 trap:
    a comment quoting a banned spec keeps it grep-alive)."""

    def test_the_animation_tokens_are_banned_from_the_sheet(self):
        from utils.theme import HEARTH_CSS
        assert "::details-content" not in HEARTH_CSS
        assert "interpolate-size" not in HEARTH_CSS
        assert "transition: height" not in HEARTH_CSS
        # The anchoring opt-out existed only to protect the animation;
        # with nothing animating layout, browser-default anchoring is
        # useful again and the opt-out must not linger.
        assert "overflow-anchor" not in HEARTH_CSS
        # Round 30's bans hold: the paint-state animation that scrambled
        # opening text must not return in any form.
        assert "allow-discrete" not in HEARTH_CSS
        assert "content-visibility" not in HEARTH_CSS

    def test_smooth_programmatic_scrolling_survives_the_retirement(self):
        """Round 29's stepped-scroll fix is NOT part of the retirement: it
        eases the programmatic/anchor scrolls Streamlit issues on reruns,
        moves only the viewport (never layout), and predates the expander
        defects. It keeps its reduced-motion gate."""
        from utils.theme import HEARTH_CSS
        assert "scroll-behavior: smooth" in HEARTH_CSS
        assert "prefers-reduced-motion: no-preference" in HEARTH_CSS

    def test_cards_and_expanders_stay_containment_scoped(self):
        """Round 30's containment is independent of the animation and
        stays: without it, one card's hover lift invalidates layout and
        paint across the whole heavy results tree — the owner's scroll
        jitter. layout+paint only, never size: the boxes grow with
        content."""
        from utils.theme import HEARTH_CSS
        assert HEARTH_CSS.count("contain: layout paint;") >= 2
        assert "contain: strict" not in HEARTH_CSS
        assert "contain: size" not in HEARTH_CSS


class TestScrollbarGutterStability:
    """Round 34: the jitter that SURVIVED the animation retirement.

    Owner, after the round-33 deploy: the dip-and-spring is resolved,
    but opening an expander still "causes some words to move to the
    next line". The mechanism has nothing to do with animation: the
    open changes page height; when that crosses the viewport threshold
    on a classic-scrollbar platform, the browser inserts the vertical
    scrollbar into the scroll container, the content narrows ~15px,
    and every line of text on the page rewraps. Reserving the gutter
    permanently makes scrollbar arrival width-neutral."""

    def test_the_gutter_is_reserved_on_the_measured_scroll_container(self):
        from utils.theme import HEARTH_CSS
        assert "scrollbar-gutter: stable" in HEARTH_CSS
        rule_at = HEARTH_CSS.index("scrollbar-gutter: stable")
        selector = 'html, [data-testid="stMain"]'
        selector_at = HEARTH_CSS.rindex(selector, 0, rule_at)
        # The measured scroll container's selector immediately owns the
        # declaration — nothing between them but the opening brace.
        between = HEARTH_CSS[selector_at + len(selector):rule_at]
        assert between.strip() == "{"

    def test_stability_is_not_gated_behind_reduced_motion(self):
        """Layout stability is not motion: a reduced-motion user's text
        rewraps on scrollbar arrival exactly like everyone else's, so
        the gutter must sit OUTSIDE the motion gate."""
        from utils.theme import HEARTH_CSS
        media_start = HEARTH_CSS.index("@media (prefers-reduced-motion: no-preference)")
        depth, media_end = 0, None
        for i in range(media_start, len(HEARTH_CSS)):
            if HEARTH_CSS[i] == "{":
                depth += 1
            elif HEARTH_CSS[i] == "}":
                depth -= 1
                if depth == 0:
                    media_end = i
                    break
        assert media_end is not None
        assert "scrollbar-gutter" not in HEARTH_CSS[media_start:media_end]


# ---------------------------------------------------------------------------
# Round 30: the critic's read on clean runs, smoothing v2, radius alignment.


class TestCriticsReadOnCleanRuns:
    """Round 30 (owner call): the bias explanation renders on EVERY run.

    It used to be discarded unless a bias was flagged, which hid the
    cross-family validator's best prose — "the top two are within a small
    margin, both strong choices; the lower ranks fell on review evidence"
    is the critic earning its keep, and on a portfolio surface its absence
    read as the critic having nothing to say. The HEADER is the honesty
    rail: "Bias check" names a finding, so a clean run renders under
    "Independent critic's read" instead — same slot, no manufactured
    alarm."""

    @staticmethod
    def _markup(monkeypatch, bias):
        import app as app_module

        captured = []
        monkeypatch.setattr(app_module, "render_html", captured.append)
        app_module.render_validation_insights({
            "agent_outputs": {"critic_validator": {"validation_results": {
                "bias_analysis": {"bias_assessment": bias},
                "top_provider_validation": {"top_provider_validations": []},
                "final_recommendations": {"recommendation_confidence": "high"},
            }}},
            "workflow_summary": {},
        })
        return "".join(captured)

    _CLEAN_READ = ("The top two providers are separated by a very small "
                   "margin - both are strong choices.")

    def test_a_clean_run_renders_the_read_under_the_honest_header(self, monkeypatch):
        markup = self._markup(monkeypatch, {
            "severity": "low", "detected_biases": [],
            "explanation": self._CLEAN_READ,
        })
        assert "Independent critic&#x27;s read" in markup
        assert "separated by a very small" in markup
        assert "Bias check:" not in markup

    def test_a_flagged_run_keeps_the_bias_check_header(self, monkeypatch):
        markup = self._markup(monkeypatch, {
            "severity": "medium",
            "detected_biases": ["Distance is deciding the top spot."],
            "explanation": "Location outweighs the rating evidence here.",
        })
        assert "Bias check:" in markup
        assert "Independent critic&#x27;s read" not in markup
        assert "Distance is deciding the top spot." in markup

    def test_an_empty_explanation_still_renders_nothing(self, monkeypatch):
        """The conditional-note discipline survives: always-on means "on
        every run the critic wrote something", never a permanent empty
        row."""
        markup = self._markup(monkeypatch, {
            "severity": "low", "detected_biases": [], "explanation": "",
        })
        assert "Independent critic" not in markup
        assert "Bias check:" not in markup


class TestRadiusChipsCarryTheUnit:
    """Round 30 (owner): "10 mi | 25 mi | 50 mi" chips, sized and aligned
    to the weight controls below. The bare numbers were a wrap fix for the
    OLD narrow form column; the radius now sits in the [2, 1] row's last
    third — directly above the Experience control — where the suffixed
    chips fit."""

    def test_every_chip_states_the_unit(self):
        from app import SEARCH_RADIUS_OPTIONS, SEARCH_RADIUS_DEFAULT
        assert all(label.endswith(" mi") for label in SEARCH_RADIUS_OPTIONS)
        assert SEARCH_RADIUS_DEFAULT.endswith(" mi")

    def test_the_label_no_longer_repeats_the_unit(self):
        src = inspect.getsource(app.render_search_form)
        assert '"Search within"' in src
        assert "Search within (miles)" not in src


# ---------------------------------------------------------------------------
# Round 31: the caveat earns its render, the card is one grid, the page
# stops bouncing.


class TestRound31Alignment:
    def test_the_where_row_is_equal_thirds(self):
        """One 3-column grid down the card (owner): State above Nearby,
        City above Ratings, ZIP above Search-within above Experience. The
        column-context recorder in test_platform_parsers asserts the spec;
        this pins the source so the two cannot drift apart silently."""
        src = inspect.getsource(app._location_picker)
        assert 'st.columns(3, vertical_alignment="bottom")' in src
        assert "[1, 2, 1]" not in src


class TestPatientRegisterAvoidsNumberedPositions:
    def test_the_prompt_prefers_groups_over_numbered_positions(self):
        """The companion to the caveat gate: fewer stale ordinals at the
        source. The reorder runs after the analysis, so every numbered
        position the model writes into the patient register forces a
        timing caveat onto the panel; names and groups do not."""
        import json
        from unittest.mock import MagicMock, patch

        from agents.critic_validator import CriticValidatorAgent

        with patch("agents.critic_validator.Anthropic"):
            critic = CriticValidatorAgent()
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content[0].text = json.dumps({
            "bias_assessment": {"detected_biases": [], "severity": "low",
                                "explanation": "x", "technical_explanation": "y"}
        })
        critic.anthropic_client.messages.create.return_value = response
        critic._analyze_ranking_bias([{"name": "Dr. A"}], {})
        prompt = (critic.anthropic_client.messages.create
                  .call_args.kwargs["messages"][0]["content"])

        assert "not by numbered position" in prompt
        assert "forces a timing caveat onto the panel" in prompt
        assert "stay fine in technical_explanation" in prompt


def test_panel_passes_the_rendered_prose_to_the_caveat(monkeypatch):
    """WIRING: the caveat must be gated on the prose the reader SEES. A
    helper-only test would let the call site pass a constant and the
    2026-08-11 caveat-about-nothing would return with the suite green."""
    import app as app_module

    captured = []
    monkeypatch.setattr(app_module, "render_html", captured.append)
    workflow = {
        "agent_outputs": {"critic_validator": {"validation_results": {
            "bias_analysis": {"bias_assessment": {
                "severity": "low", "detected_biases": [],
                "explanation": ("The top three doctors are separated by "
                                "very small margins."),
            }},
            "top_provider_validation": {"top_provider_validations": []},
            "final_recommendations": {"recommendation_confidence": "high"},
        }}},
        "workflow_summary": {"refinement": {"applied": True, "moves": [
            {"name": "Dr. Kan Yu", "from": 9, "to": 7, "display_rank": 7,
             "reasons": ["docked"]},
        ]}},
    }
    markup = "".join(captured) if captured else ""
    app_module.render_validation_insights(workflow)
    markup = "".join(captured)
    assert "Independent critic&#x27;s read" in markup
    assert "from before the independent critic review" not in markup

    # The same run with a numbered position in the read DOES get the caveat.
    captured.clear()
    workflow["agent_outputs"]["critic_validator"]["validation_results"][
        "bias_analysis"]["bias_assessment"]["explanation"] = (
        "The doctor ranked 3rd has the strongest reviews.")
    app_module.render_validation_insights(workflow)
    markup = "".join(captured)
    assert "from before the independent critic review" in markup


class TestEmptyShortlistNotice:
    """Round 32: the zero-card page must say WHY (2026-08-11 incident).

    An exhausted Anthropic credit balance failed all four critic calls, every
    researched provider was withheld `not_critiqued`, and the page's only
    output was "No provider recommendations found. Try adjusting your search
    criteria." — advice written for a coverage gap, rendered for a pipeline
    failure, above a search that had just FOUND 22 providers. Every surface
    that named the actual cause (withheld reasons, cost card, agent
    internals) was gated behind a non-empty shortlist.
    """

    def _results(self, by_reason, found=64, critic_error=""):
        withheld = {
            "total": sum(by_reason.values()),
            "by_reason": dict(by_reason),
            "pipeline_failures": by_reason.get("not_judged", 0)
            + by_reason.get("not_critiqued", 0),
            "pipeline_failure_names": [],
            "no_data": sum(
                by_reason.get(r, 0)
                for r in ("no_profile_found", "identity_rejected", "failed")
            ),
            "not_researched": by_reason.get("over_budget", 0),
        }
        results = {
            "workflow_summary": {
                "withheld": withheld,
                "total_providers_found": found,
            },
            "agent_outputs": {},
        }
        if critic_error:
            results["agent_outputs"] = {
                "critic_validator": {
                    "validation_results": {
                        "top_provider_validation": {
                            "top_provider_validations": [],
                            "overall_ranking_validity": {
                                "status": "error",
                                "confidence": "low",
                                "summary": critic_error,
                            },
                        }
                    }
                }
            }
        return results

    def test_pipeline_failure_gets_retry_framing_not_criteria_advice(self):
        """The observed incident shape: not_critiqued=7, no_profile_found=1,
        over_budget=56. The message must name the found count, own the
        failure, advise a retry — and must not be the old criteria advice."""
        message, detail = _empty_shortlist_notice(
            self._results(
                {"not_critiqued": 7, "no_profile_found": 1, "over_budget": 56},
                critic_error=(
                    "Validation could not be completed — Validation error: "
                    "Error code: 400 - credit balance is too low"
                ),
            )
        )
        assert "64" in message
        assert "on our side" in message
        assert "again" in message.lower()
        assert "adjusting your search criteria" not in message
        assert "credit balance is too low" in detail

    def test_failed_lookups_count_as_our_side(self):
        """`failed` is an errored lookup — infrastructure, not coverage. A
        Tavily outage must not read as "these providers have no reviews"."""
        message, _ = _empty_shortlist_notice(self._results({"failed": 8}))
        assert "on our side" in message

    def test_coverage_gap_keeps_the_widening_advice(self):
        """When every researched provider was a genuine coverage gap, the
        old advice was honest — widening is the right move, and the message
        must not claim a service failure that didn't happen."""
        message, detail = _empty_shortlist_notice(
            self._results({"no_profile_found": 6, "identity_rejected": 2})
        )
        assert "radius" in message or "nearby city" in message
        assert "on our side" not in message
        assert detail == ""

    def test_our_failures_outrank_coverage_in_the_framing(self):
        """Mixed reasons take the retry framing: telling the user to widen a
        search our own stage broke buries the lede."""
        message, _ = _empty_shortlist_notice(
            self._results({"not_judged": 1, "no_profile_found": 7})
        )
        assert "on our side" in message

    def test_missing_summary_falls_back_to_an_honest_generic(self):
        message, detail = _empty_shortlist_notice({"workflow_summary": {}})
        assert message
        assert detail == ""

    def test_a_healthy_critic_yields_no_detail(self):
        """The developer detail is the critic's collapse reason — a critic
        that validated fine contributes nothing, whatever its confidence."""
        results = self._results({"no_profile_found": 8})
        results["agent_outputs"] = {
            "critic_validator": {
                "validation_results": {
                    "top_provider_validation": {
                        "top_provider_validations": [{"rank": 1}],
                        "overall_ranking_validity": {
                            "status": "validated",
                            "confidence": "high",
                            "summary": "Fine.",
                        },
                    }
                }
            }
        }
        _, detail = _empty_shortlist_notice(results)
        assert detail == ""

    def test_the_old_blanket_advice_is_gone_from_the_source(self):
        """The one-size sentence must not quietly return beside the notice."""
        import inspect

        import app as app_module

        assert (
            "No provider recommendations found. Try adjusting your search criteria."
            not in inspect.getsource(app_module)
        )

    def test_main_renders_diagnostics_on_the_empty_shortlist_path(self):
        """The wiring IS the feature: the notice, the per-row withheld
        reasons and the cost card must all be reachable with zero cards, or
        the run that most needs explaining renders one sentence."""
        import inspect

        import app as app_module

        source = inspect.getsource(app_module.main)
        assert "_empty_shortlist_notice(" in source
        # success path + empty-shortlist path
        assert source.count("render_other_providers(") >= 2
        # success + empty-shortlist + found-nobody paths
        assert source.count("render_cost_card(") >= 3
