"""Unit tests for pure helpers in app.py (superlatives, rubric markup,
judge-consistency findings)."""

import pytest

from app import (
    _judge_findings,
    refinement_note_markup,
    _judge_findings_note,
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


def test_judge_findings_note_is_empty_when_there_is_no_concern():
    """"Show only when a concern is present": a permanent "0 inconsistencies"
    row would train the eye to skip the row that matters."""
    assert _judge_findings_note([]) == ""


def test_judge_findings_note_reports_the_count_and_the_no_score_effect():
    note = _judge_findings_note([("Dr. Rabin", _FINDING)])
    assert "1 inconsistency was found" in note
    assert "no provider's ranking was changed" in note.lower()
    assert 'class="cc-why"' in note


def test_judge_findings_note_pluralizes():
    note = _judge_findings_note([("Dr. A", _FINDING), ("Dr. B", _FINDING)])
    assert "2 inconsistencies were found" in note


def test_judge_findings_note_leaks_no_jargon_and_names_nobody():
    """The guard on the patient/developer split. The finding text names
    internal rubric criteria, and the judge is explicitly forbidden from
    putting that vocabulary in front of a patient (preference_scorer's
    scoring_rules). The raw text belongs on the developer surfaces only — and
    no provider gets named beside an admission that our judge slipped.
    """
    note = _judge_findings_note([("Dr. Brian Rabin, MD", _FINDING)])

    for jargon in ("practical_access", "review_substance", "red_flags", "10/20"):
        assert jargon not in note
    assert "Rabin" not in note
    assert _FINDING not in note


# ---- the WIRING, not just the helpers ----
#
# Guarding the helpers alone would let someone delete `{judge_note}` from the
# panel's f-string with every test still green — which is exactly how round 5's
# `[:400]` survived four field-test rounds unasserted.

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


def test_panel_renders_the_judge_note_when_a_finding_exists(monkeypatch):
    markup = _panel_markup(
        monkeypatch,
        {"provider_name": "Dr. Rabin", "rank": 1, "recommendation_adjustments": _FINDING},
    )
    assert "Judge review:" in markup
    assert "1 inconsistency was found" in markup
    # ...and the raw finding still does not reach the patient-facing panel
    assert "practical_access" not in markup


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
    count of judge errors was zero."""
    markup = _panel(monkeypatch, validations=[
        {"provider_name": f"Dr. {i}", "rank": i, "recommendation_adjustments": _LIVE_PASS}
        for i in range(1, 11)
    ])
    assert "Judge review:" not in markup
    assert "inconsistenc" not in markup


def test_a_real_finding_still_reaches_the_patient_count(monkeypatch):
    markup = _panel(monkeypatch, validations=[
        {"provider_name": "Dr. A", "rank": 1, "recommendation_adjustments": _REAL_FINDING},
        {"provider_name": "Dr. B", "rank": 2, "recommendation_adjustments": _LIVE_PASS},
    ])
    assert "1 inconsistency was found" in markup


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
    assert "Refined by critic review" in markup
    assert "Dr. Alpha" in markup
    assert "#6 &rarr; #5" in markup


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
    assert "Refined by critic review" in panel
    assert "Dr. Kuniyoshi" in panel
    # Inside the panel card, not a sibling block after it.
    assert panel.index("Responsible AI review") < panel.index("Refined by critic review")
    assert "Refined by critic review" in panel.split("cc-cost-note")[0]


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

    assert "before the independent review re-ordered" in markup
    assert "Khan, MD is now #6" in markup, "the reader must be able to reconcile the ordinal"


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
