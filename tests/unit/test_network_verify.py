"""Tests for the FHIR network-verification prototype (fhir/verify.py)."""

from unittest.mock import patch

from fhir.verify import verify_network


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
