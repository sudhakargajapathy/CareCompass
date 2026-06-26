"""Unit tests for FHIR R4 resource conversion utilities."""

import pytest

from utils.fhir import (
    provider_to_fhir_practitioner,
    provider_to_fhir_practitioner_role,
    provider_to_fhir_bundle,
    providers_to_fhir_bundle,
    fhir_practitioner_to_provider,
    fhir_bundle_to_providers,
)
from tests.fixtures.mock_providers import SAMPLE_PROVIDERS, PROVIDER_WITH_MISSING_DATA
from tests.fixtures.mock_fhir_data import (
    MOCK_FHIR_PRACTITIONER,
    MOCK_FHIR_PRACTITIONER_ROLE,
    MOCK_FHIR_BUNDLE,
    MOCK_FHIR_PRACTITIONER_MINIMAL,
    MOCK_FHIR_BUNDLE_EMPTY,
)


# ---------------------------------------------------------------------------
# Provider → FHIR Practitioner
# ---------------------------------------------------------------------------

class TestProviderToFHIRPractitioner:
    """Tests for converting internal provider dicts to FHIR Practitioner."""

    def test_basic_conversion(self):
        provider = SAMPLE_PROVIDERS[0]  # Dr. Sarah Johnson
        result = provider_to_fhir_practitioner(provider)

        assert result["resourceType"] == "Practitioner"
        assert result["active"] is True
        assert result["id"]  # non-empty UUID
        assert result["name"][0]["family"] == "Johnson"
        assert "Sarah" in result["name"][0]["given"]

    def test_phone_included(self):
        provider = SAMPLE_PROVIDERS[0]
        result = provider_to_fhir_practitioner(provider)

        assert "telecom" in result
        assert result["telecom"][0]["system"] == "phone"
        assert result["telecom"][0]["value"] == provider["phone"]

    def test_address_parsed(self):
        provider = SAMPLE_PROVIDERS[0]
        result = provider_to_fhir_practitioner(provider)

        assert "address" in result
        assert result["address"][0]["text"] == "Phoenix, AZ"

    def test_specialty_in_qualification(self):
        provider = SAMPLE_PROVIDERS[0]
        result = provider_to_fhir_practitioner(provider)

        assert "qualification" in result
        specialties = [
            q["code"]["coding"][0]["display"]
            for q in result["qualification"]
            if "coding" in q.get("code", {})
        ]
        assert "Neurology" in specialties

    def test_years_experience_in_qualification(self):
        provider = SAMPLE_PROVIDERS[0]  # 15 years
        result = provider_to_fhir_practitioner(provider)

        experience_quals = [
            q for q in result["qualification"]
            if "experience" in q.get("code", {}).get("text", "")
        ]
        assert len(experience_quals) == 1
        assert "15" in experience_quals[0]["code"]["text"]

    def test_missing_phone_excluded(self):
        result = provider_to_fhir_practitioner(PROVIDER_WITH_MISSING_DATA)
        assert "telecom" not in result

    def test_missing_data_still_valid(self):
        result = provider_to_fhir_practitioner(PROVIDER_WITH_MISSING_DATA)
        assert result["resourceType"] == "Practitioner"
        assert result["name"][0]["text"] == "Dr. John Doe"

    def test_deterministic_id(self):
        """Same provider name should produce same id."""
        provider = SAMPLE_PROVIDERS[0]
        id1 = provider_to_fhir_practitioner(provider)["id"]
        id2 = provider_to_fhir_practitioner(provider)["id"]
        assert id1 == id2

    def test_different_providers_different_ids(self):
        id1 = provider_to_fhir_practitioner(SAMPLE_PROVIDERS[0])["id"]
        id2 = provider_to_fhir_practitioner(SAMPLE_PROVIDERS[1])["id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# Provider → FHIR PractitionerRole
# ---------------------------------------------------------------------------

class TestProviderToFHIRPractitionerRole:
    """Tests for converting internal provider dicts to FHIR PractitionerRole."""

    def test_basic_role(self):
        provider = SAMPLE_PROVIDERS[0]
        role = provider_to_fhir_practitioner_role(provider, "pract-001")

        assert role["resourceType"] == "PractitionerRole"
        assert role["practitioner"]["reference"] == "Practitioner/pract-001"
        assert role["active"] is True

    def test_specialty_present(self):
        provider = SAMPLE_PROVIDERS[0]
        role = provider_to_fhir_practitioner_role(provider, "pract-001")

        assert role["specialty"][0]["text"] == "Neurology"

    def test_rating_extension(self):
        provider = SAMPLE_PROVIDERS[0]  # rating 4.8
        role = provider_to_fhir_practitioner_role(provider, "pract-001")

        rating_exts = [
            e for e in role["extension"] if e["url"].endswith("provider-rating")
        ]
        assert len(rating_exts) == 1
        assert rating_exts[0]["valueDecimal"] == 4.8

    def test_review_count_extension(self):
        provider = SAMPLE_PROVIDERS[0]  # 127 reviews
        role = provider_to_fhir_practitioner_role(provider, "pract-001")

        count_exts = [
            e for e in role["extension"] if e["url"].endswith("review-count")
        ]
        assert len(count_exts) == 1
        assert count_exts[0]["valueInteger"] == 127

    def test_insurance_extensions(self):
        provider = SAMPLE_PROVIDERS[0]  # string insurance
        role = provider_to_fhir_practitioner_role(provider, "pract-001")

        ins_exts = [
            e for e in role["extension"] if e["url"].endswith("insurance-accepted")
        ]
        assert len(ins_exts) >= 1

    def test_no_rating_no_extension(self):
        role = provider_to_fhir_practitioner_role(PROVIDER_WITH_MISSING_DATA, "pract-x")
        # No rating (None), no review_count (0 stored as None-ish), no insurance
        assert "extension" not in role or len(role.get("extension", [])) == 0


# ---------------------------------------------------------------------------
# Provider → FHIR Bundle (single)
# ---------------------------------------------------------------------------

class TestProviderToFHIRBundle:
    """Tests for converting a single provider to a FHIR Bundle."""

    def test_bundle_structure(self):
        provider = SAMPLE_PROVIDERS[0]
        bundle = provider_to_fhir_bundle(provider)

        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert len(bundle["entry"]) == 2

    def test_bundle_contains_practitioner_and_role(self):
        provider = SAMPLE_PROVIDERS[0]
        bundle = provider_to_fhir_bundle(provider)

        types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        assert "Practitioner" in types
        assert "PractitionerRole" in types

    def test_role_references_practitioner(self):
        provider = SAMPLE_PROVIDERS[0]
        bundle = provider_to_fhir_bundle(provider)

        practitioner = bundle["entry"][0]["resource"]
        role = bundle["entry"][1]["resource"]

        expected_ref = f"Practitioner/{practitioner['id']}"
        assert role["practitioner"]["reference"] == expected_ref


# ---------------------------------------------------------------------------
# Providers list → FHIR searchset Bundle
# ---------------------------------------------------------------------------

class TestProvidersToFHIRBundle:
    """Tests for converting multiple providers to a FHIR searchset Bundle."""

    def test_multi_provider_bundle(self):
        bundle = providers_to_fhir_bundle(SAMPLE_PROVIDERS)

        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "searchset"
        assert bundle["total"] == len(SAMPLE_PROVIDERS)
        # 2 entries per provider (Practitioner + PractitionerRole)
        assert len(bundle["entry"]) == len(SAMPLE_PROVIDERS) * 2

    def test_empty_list_produces_empty_bundle(self):
        bundle = providers_to_fhir_bundle([])

        assert bundle["total"] == 0
        assert bundle["entry"] == []


# ---------------------------------------------------------------------------
# FHIR → Provider (reverse conversion)
# ---------------------------------------------------------------------------

class TestFHIRToProvider:
    """Tests for converting FHIR resources back to internal provider dicts."""

    def test_practitioner_only(self):
        provider = fhir_practitioner_to_provider(MOCK_FHIR_PRACTITIONER)

        assert provider["name"] == "Dr. Sarah Johnson"
        assert provider["phone"] == "(602) 555-1234"
        assert provider["location"] == "Phoenix, AZ"
        assert provider["specialty"] == "Neurology"
        assert provider["years_experience"] == 15

    def test_practitioner_with_role(self):
        provider = fhir_practitioner_to_provider(
            MOCK_FHIR_PRACTITIONER, MOCK_FHIR_PRACTITIONER_ROLE
        )

        assert provider["rating"] == 4.8
        assert provider["review_count"] == 127
        assert "Aetna" in provider["insurance_accepted"]
        assert "Blue Cross Blue Shield" in provider["insurance_accepted"]

    def test_minimal_practitioner(self):
        provider = fhir_practitioner_to_provider(MOCK_FHIR_PRACTITIONER_MINIMAL)

        assert provider["name"] == "John Doe"
        assert "phone" not in provider
        assert "location" not in provider

    def test_empty_bundle(self):
        providers = fhir_bundle_to_providers(MOCK_FHIR_BUNDLE_EMPTY)
        assert providers == []

    def test_bundle_roundtrip_count(self):
        providers = fhir_bundle_to_providers(MOCK_FHIR_BUNDLE)
        assert len(providers) == 2

    def test_bundle_roundtrip_names(self):
        providers = fhir_bundle_to_providers(MOCK_FHIR_BUNDLE)
        names = {p["name"] for p in providers}
        assert "Dr. Sarah Johnson" in names
        assert "Dr. Michael Chen" in names

    def test_bundle_roundtrip_ratings(self):
        providers = fhir_bundle_to_providers(MOCK_FHIR_BUNDLE)
        ratings = {p["name"]: p.get("rating") for p in providers}
        assert ratings["Dr. Sarah Johnson"] == 4.8
        assert ratings["Dr. Michael Chen"] == 4.5


# ---------------------------------------------------------------------------
# Round-trip: Provider → FHIR → Provider
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Tests that provider → FHIR → provider preserves key fields."""

    def test_single_provider_roundtrip(self):
        original = SAMPLE_PROVIDERS[0]
        bundle = provider_to_fhir_bundle(original)
        restored_list = fhir_bundle_to_providers(bundle)

        assert len(restored_list) == 1
        restored = restored_list[0]

        assert restored["name"] == original["name"]
        assert restored["specialty"] == original["specialty"]
        assert restored["rating"] == original["rating"]
        assert restored["review_count"] == original["review_count"]
        assert restored["years_experience"] == original["years_experience"]

    def test_multi_provider_roundtrip(self):
        originals = SAMPLE_PROVIDERS[:3]
        bundle = providers_to_fhir_bundle(originals)
        restored = fhir_bundle_to_providers(bundle)

        assert len(restored) == 3
        original_names = {p["name"] for p in originals}
        restored_names = {p["name"] for p in restored}
        assert original_names == restored_names

    def test_roundtrip_preserves_insurance_list(self):
        """Provider with list-type insurance should roundtrip correctly."""
        provider = {
            "name": "Dr. Test Provider",
            "specialty": "Cardiology",
            "location": "Boston, MA",
            "phone": "(617) 555-0000",
            "rating": 4.6,
            "review_count": 50,
            "insurance_accepted": ["Aetna", "Cigna", "Medicare"],
        }
        bundle = provider_to_fhir_bundle(provider)
        restored = fhir_bundle_to_providers(bundle)[0]

        assert set(restored["insurance_accepted"]) == {"Aetna", "Cigna", "Medicare"}
