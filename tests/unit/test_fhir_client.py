"""Unit tests for FHIR client, mock client, transformer, and specialty codes."""

import pytest
from unittest.mock import patch

from fhir.mock_client import MockFHIRClient
from fhir.transformer import FHIRToProviderTransformer
from fhir.specialty_codes import get_snomed_code, get_specialty_name, SPECIALTY_TO_SNOMED
from fhir.client import FHIRClientProtocol, create_fhir_client


# ---------------------------------------------------------------------------
# Specialty codes
# ---------------------------------------------------------------------------


class TestSpecialtyCodes:
    def test_known_specialty_returns_code(self):
        assert get_snomed_code("Neurology") == "394591006"
        assert get_snomed_code("Cardiology") == "394579002"

    def test_unknown_specialty_returns_none(self):
        assert get_snomed_code("Nonexistent Specialty") is None

    def test_reverse_lookup(self):
        assert get_specialty_name("394591006") == "Neurology"
        assert get_specialty_name("000000000") is None

    def test_all_specialties_have_unique_codes(self):
        codes = list(SPECIALTY_TO_SNOMED.values())
        assert len(codes) == len(set(codes)), "Duplicate SNOMED codes found"


# ---------------------------------------------------------------------------
# Mock FHIR Client
# ---------------------------------------------------------------------------


class TestMockFHIRClient:
    @pytest.fixture
    def client(self):
        return MockFHIRClient()

    def test_is_available(self, client):
        assert client.is_available() is True

    def test_search_neurology_phoenix(self, client):
        bundle = client.search_practitioners("Neurology", "Phoenix, AZ")
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "searchset"
        assert bundle["total"] > 0
        assert len(bundle["entry"]) > 0

    def test_search_returns_proper_fhir_structure(self, client):
        bundle = client.search_practitioners("Cardiology", "Phoenix, AZ")
        for entry in bundle["entry"]:
            resource = entry["resource"]
            assert resource["resourceType"] == "Practitioner"
            assert "name" in resource
            assert "id" in resource

    def test_search_filters_by_specialty(self, client):
        neuro_bundle = client.search_practitioners("Neurology", "Phoenix, AZ")
        cardio_bundle = client.search_practitioners("Cardiology", "Phoenix, AZ")

        neuro_ids = {e["resource"]["id"] for e in neuro_bundle["entry"]}
        cardio_ids = {e["resource"]["id"] for e in cardio_bundle["entry"]}

        # Neurology and cardiology should have no overlap
        assert neuro_ids.isdisjoint(cardio_ids)

    def test_search_filters_by_insurance_network(self, client):
        aetna_bundle = client.search_practitioners("Neurology", "Phoenix, AZ", insurance_network="Aetna")
        assert aetna_bundle["total"] > 0

        # All returned practitioners should be in Aetna network
        for entry in aetna_bundle["entry"]:
            org_names = [org.get("name", "") for org in entry.get("_organizations", [])]
            assert any("Aetna" in name for name in org_names), \
                f"Expected Aetna in organizations, got {org_names}"

    def test_search_with_no_match_returns_empty(self, client):
        bundle = client.search_practitioners("Neurology", "Anchorage, AK")
        assert bundle["total"] == 0
        assert len(bundle["entry"]) == 0

    def test_search_respects_count_limit(self, client):
        bundle = client.search_practitioners("Neurology", "Phoenix, AZ", count=1)
        assert len(bundle["entry"]) <= 1

    def test_get_practitioner_by_id(self, client):
        practitioner = client.get_practitioner("prac-1001")
        assert practitioner["id"] == "prac-1001"
        assert practitioner["resourceType"] == "Practitioner"

    def test_get_practitioner_unknown_id(self, client):
        practitioner = client.get_practitioner("nonexistent")
        assert practitioner == {}

    def test_get_practitioner_roles(self, client):
        roles = client.get_practitioner_roles("prac-1001")
        assert len(roles) > 0
        assert all(r["resourceType"] == "PractitionerRole" for r in roles)

    def test_get_location(self, client):
        loc = client.get_location("loc-phoenix-1")
        assert loc["resourceType"] == "Location"
        assert loc["address"]["city"] == "Phoenix"

    def test_entries_have_attached_resources(self, client):
        bundle = client.search_practitioners("Neurology", "Phoenix, AZ")
        for entry in bundle["entry"]:
            assert "_roles" in entry
            assert "_locations" in entry
            assert "_organizations" in entry
            assert len(entry["_roles"]) > 0
            assert len(entry["_locations"]) > 0


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class TestFHIRToProviderTransformer:
    @pytest.fixture
    def transformer(self):
        return FHIRToProviderTransformer()

    @pytest.fixture
    def mock_bundle(self):
        client = MockFHIRClient()
        return client.search_practitioners("Neurology", "Phoenix, AZ")

    def test_transform_bundle_returns_list(self, transformer, mock_bundle):
        providers = transformer.transform_bundle(mock_bundle)
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_transformed_provider_has_required_fields(self, transformer, mock_bundle):
        providers = transformer.transform_bundle(mock_bundle)
        required_fields = [
            "name", "specialty", "location", "phone", "rating",
            "review_count", "review_summary", "review_sentiment",
            "insurance_accepted", "data_source", "fhir_metadata",
        ]
        for provider in providers:
            for field in required_fields:
                assert field in provider, f"Missing field: {field}"

    def test_provider_name_format(self, transformer, mock_bundle):
        providers = transformer.transform_bundle(mock_bundle)
        for provider in providers:
            assert provider["name"].startswith("Dr."), f"Expected Dr. prefix, got: {provider['name']}"

    def test_fhir_metadata_present(self, transformer, mock_bundle):
        providers = transformer.transform_bundle(mock_bundle)
        for provider in providers:
            meta = provider["fhir_metadata"]
            assert "practitioner_id" in meta
            assert "npi" in meta
            assert meta["network_verified"] is True
            assert "networks" in meta

    def test_data_source_is_fhir(self, transformer, mock_bundle):
        providers = transformer.transform_bundle(mock_bundle)
        for provider in providers:
            assert provider["data_source"] == "fhir"

    def test_rating_defaults_to_zero(self, transformer, mock_bundle):
        providers = transformer.transform_bundle(mock_bundle)
        for provider in providers:
            assert provider["rating"] == 0.0

    def test_review_fields_default(self, transformer, mock_bundle):
        providers = transformer.transform_bundle(mock_bundle)
        for provider in providers:
            assert provider["review_summary"] == "No reviews available"
            assert provider["review_sentiment"] == "unknown"

    def test_insurance_extracted(self, transformer, mock_bundle):
        providers = transformer.transform_bundle(mock_bundle)
        # At least one provider should have insurance info
        has_insurance = any(len(p["insurance_accepted"]) > 0 for p in providers)
        assert has_insurance, "Expected at least one provider with insurance data"

    def test_npi_extracted(self, transformer, mock_bundle):
        providers = transformer.transform_bundle(mock_bundle)
        for provider in providers:
            npi = provider["fhir_metadata"]["npi"]
            assert npi is not None
            assert len(npi) == 10  # US NPIs are 10 digits

    def test_education_extracted(self, transformer, mock_bundle):
        providers = transformer.transform_bundle(mock_bundle)
        has_education = any(p.get("education") is not None for p in providers)
        assert has_education, "Expected at least one provider with education data"

    def test_years_experience_estimated(self, transformer, mock_bundle):
        providers = transformer.transform_bundle(mock_bundle)
        has_experience = any(p.get("years_experience") is not None for p in providers)
        assert has_experience

    def test_transform_empty_bundle(self, transformer):
        providers = transformer.transform_bundle({})
        assert providers == []

    def test_transform_invalid_bundle(self, transformer):
        providers = transformer.transform_bundle({"resourceType": "Patient"})
        assert providers == []

    def test_transform_bundle_with_no_entries(self, transformer):
        bundle = {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []}
        providers = transformer.transform_bundle(bundle)
        assert providers == []


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateFHIRClient:
    @patch("fhir.client.get_config")
    def test_creates_mock_when_configured(self, mock_config):
        mock_config.return_value.FHIR_USE_MOCK = True
        client = create_fhir_client()
        assert isinstance(client, MockFHIRClient)

    @patch("fhir.client.get_config")
    def test_mock_client_satisfies_protocol(self, mock_config):
        mock_config.return_value.FHIR_USE_MOCK = True
        client = create_fhir_client()
        assert isinstance(client, FHIRClientProtocol)
