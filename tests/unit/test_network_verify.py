"""Tests for the FHIR network-verification prototype (fhir/verify.py)."""

from unittest.mock import patch

from fhir.verify import simulate_network_batch, verify_network


def _with_mock_fhir(monkeypatch):
    monkeypatch.setenv("FHIR_ENABLED", "true")
    monkeypatch.setenv("FHIR_USE_MOCK", "true")


def test_sandbox_verifies_a_directory_provider(monkeypatch):
    """A provider that exists in the sandbox directory verifies."""
    _with_mock_fhir(monkeypatch)

    # Dr. Sarah Chen is in the mock directory's Neurology data
    result = verify_network("Dr. Sarah Chen", "Aetna", specialty="Neurology", location="Phoenix, AZ")

    assert result["status"] == "verified"
    assert result["source"] == "sandbox"
    assert "Chen" in result["matched_name"]


def test_sandbox_no_record_is_not_a_verdict(monkeypatch):
    """A real-world provider absent from the sandbox returns no_record."""
    _with_mock_fhir(monkeypatch)

    result = verify_network(
        "Dr. Leslie Zuniga", "Aetna", specialty="Neurology", location="Phoenix, AZ"
    )

    assert result["status"] == "no_record"
    assert result["source"] == "sandbox"


def test_unavailable_on_missing_inputs_or_errors(monkeypatch):
    _with_mock_fhir(monkeypatch)

    assert verify_network("", "Aetna")["status"] == "unavailable"
    assert verify_network("Dr. X", "")["status"] == "unavailable"

    with patch("fhir.client.create_fhir_client", side_effect=RuntimeError("boom")):
        result = verify_network("Dr. X", "Aetna", specialty="Neurology")
    assert result["status"] == "unavailable"


# ---------------------------------------------------------------------------
# The SIMULATED directory (demo mode). The sandbox ships static demo doctors
# while the live search finds real ones, so every check answered "no record"
# and the FHIR feature demoed as a dead end (2026-07-31 screenshot: 5/5 grey
# against every payer). Membership is a pure function of (pool, payer) —
# stored nowhere, deterministic, payer-sensitive, labeled simulated on every
# answer.

_DEMO_POOL = [
    "Dr. Charles Vandian, DO",
    "Dr. Kishore Kumar, MD",
    "Dr. Yeeshu Arora, MD",
    "Dr. Andrea An, MD",
    "Dr. Todd Levine, MD",
]


def test_simulated_membership_is_deterministic():
    """Same pool + payer -> the same answer every run, test and demo. Stored
    nowhere on purpose: a static seed file would rot as the live pool changes
    between runs, and the repo rule forbids committing provider data."""
    assert simulate_network_batch(_DEMO_POOL, "Cigna") == simulate_network_batch(
        _DEMO_POOL, "Cigna"
    )


def test_a_payer_switch_redeals_coverage():
    """The demo moment: switching payers re-checks the SAME pool and the
    verdicts visibly move. Fixtures verified against the shipped hash — an
    earlier candidate pairing hit the ~1% case where four payers dealt this
    exact pool identically, which is why these two payers are pinned rather
    than chosen for realism."""
    aetna = {
        n
        for n, v in simulate_network_batch(_DEMO_POOL, "Aetna").items()
        if v["status"] == "verified"
    }
    united = {
        n
        for n, v in simulate_network_batch(_DEMO_POOL, "UnitedHealth").items()
        if v["status"] == "verified"
    }
    assert aetna != united


def test_every_pool_of_two_or_more_is_mixed():
    """Never all-green and never all-grey: the quota is ranked over the pool
    and clamped to [1, n-1], deliberately not an independent 70% coin flip
    per provider — at 70% per head a 5-provider pool comes up all-green in
    ~17% of demos, and an all-anything network check reads as a feature that
    does nothing."""
    for n in range(2, len(_DEMO_POOL) + 1):
        statuses = {
            v["status"]
            for v in simulate_network_batch(_DEMO_POOL[:n], "Aetna").values()
        }
        assert statuses == {"verified", "no_record"}, n


def test_results_are_labeled_simulated_and_keyed_by_the_name_asked():
    """The honesty rail plus the matcher bypass: every answer says simulated
    (a bare "in-network" against a real physician's name from invented data
    would be fabricated coverage), and results are keyed by the exact name
    asked about, so the name matcher's known shared-surname false positive
    cannot occur in demo mode."""
    results = simulate_network_batch(_DEMO_POOL, "Medicare")
    assert set(results) == set(_DEMO_POOL)
    for name, result in results.items():
        assert result["source"] == "simulated"
        assert result["payer"] == "Medicare"
        expected_match = name if result["status"] == "verified" else None
        assert result["matched_name"] == expected_match


def test_single_provider_and_blank_names():
    """A pool of one cannot be mixed — it falls to a deterministic per-name
    draw; blank names are dropped rather than dealt coverage."""
    lone = simulate_network_batch(["Dr. Andrea An, MD"], "Aetna")
    assert lone == simulate_network_batch(["Dr. Andrea An, MD"], "Aetna")
    assert list(lone.values())[0]["status"] in {"verified", "no_record"}
    assert simulate_network_batch(["", "   "], "Aetna") == {}
