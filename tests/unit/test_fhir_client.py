"""Unit tests for the FHIR client package: specialty codes, mock client, transformer."""

import pytest
from unittest.mock import patch

from fhir.mock_client import MockFHIRClient
from fhir.transformer import FHIRToProviderTransformer
from fhir.specialty_codes import get_snomed_code, get_specialty_name, SPECIALTY_TO_SNOMED
from fhir.client import FHIRClientProtocol, create_fhir_client


class TestSpecialtyCodes:
    def test_lookup_and_reverse_lookup(self):
        assert get_snomed_code("Neurology") == "394591006"
        assert get_snomed_code("Cardiology") == "394579002"
        assert get_snomed_code("Nonexistent Specialty") is None
        assert get_specialty_name("394591006") == "Neurology"
        assert get_specialty_name("000000000") is None

    def test_all_specialties_have_unique_codes(self):
        codes = list(SPECIALTY_TO_SNOMED.values())
        assert len(codes) == len(set(codes)), "Duplicate SNOMED codes found"


class TestMockFHIRClient:
    @pytest.fixture
    def client(self):
        return MockFHIRClient()

    def test_search_returns_well_formed_bundle(self, client):
        assert client.is_available() is True

        bundle = client.search_practitioners("Neurology", "Phoenix, AZ")
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "searchset"
        assert bundle["total"] > 0

        for entry in bundle["entry"]:
            resource = entry["resource"]
            assert resource["resourceType"] == "Practitioner"
            assert "name" in resource and "id" in resource
            # Roles, locations, and organizations ride along for the transformer
            assert entry["_roles"] and entry["_locations"]
            assert "_organizations" in entry

    def test_search_filters_by_specialty(self, client):
        neuro = client.search_practitioners("Neurology", "Phoenix, AZ")
        cardio = client.search_practitioners("Cardiology", "Phoenix, AZ")

        neuro_ids = {e["resource"]["id"] for e in neuro["entry"]}
        cardio_ids = {e["resource"]["id"] for e in cardio["entry"]}
        assert neuro_ids.isdisjoint(cardio_ids)

    def test_search_filters_by_insurance_network(self, client):
        bundle = client.search_practitioners("Neurology", "Phoenix, AZ", insurance_network="Aetna")
        assert bundle["total"] > 0
        for entry in bundle["entry"]:
            org_names = [org.get("name", "") for org in entry.get("_organizations", [])]
            assert any("Aetna" in name for name in org_names)

    def test_search_edge_cases(self, client):
        no_match = client.search_practitioners("Neurology", "Anchorage, AK")
        assert no_match["total"] == 0 and no_match["entry"] == []

        limited = client.search_practitioners("Neurology", "Phoenix, AZ", count=1)
        assert len(limited["entry"]) <= 1

    def test_resource_lookups(self, client):
        practitioner = client.get_practitioner("prac-1001")
        assert practitioner["id"] == "prac-1001"
        assert practitioner["resourceType"] == "Practitioner"
        assert client.get_practitioner("nonexistent") == {}

        roles = client.get_practitioner_roles("prac-1001")
        assert roles and all(r["resourceType"] == "PractitionerRole" for r in roles)

        location = client.get_location("loc-phoenix-1")
        assert location["resourceType"] == "Location"
        assert location["address"]["city"] == "Phoenix"


class TestFHIRToProviderTransformer:
    @pytest.fixture
    def transformer(self):
        return FHIRToProviderTransformer()

    @pytest.fixture
    def providers(self, transformer):
        bundle = MockFHIRClient().search_practitioners("Neurology", "Phoenix, AZ")
        return transformer.transform_bundle(bundle)

    def test_provider_shape_and_defaults(self, providers):
        assert providers, "Expected transformed providers from the mock bundle"

        required_fields = [
            "name", "specialty", "location", "phone", "rating",
            "review_count", "review_summary", "review_sentiment",
            "insurance_accepted", "data_source", "fhir_metadata",
        ]
        for provider in providers:
            for field in required_fields:
                assert field in provider, f"Missing field: {field}"
            assert provider["name"].startswith("Dr.")
            assert provider["data_source"] == "fhir"
            # Directory data has no review signal; scorer defaults must hold
            assert provider["rating"] == 0.0
            assert provider["review_summary"] == "No reviews available"
            assert provider["review_sentiment"] == "unknown"

    def test_fhir_metadata_and_extracted_fields(self, providers):
        for provider in providers:
            meta = provider["fhir_metadata"]
            assert meta["network_verified"] is True
            assert "practitioner_id" in meta and "networks" in meta
            assert meta["npi"] is not None and len(meta["npi"]) == 10

        assert any(p["insurance_accepted"] for p in providers)
        assert any(p.get("education") is not None for p in providers)
        assert any(p.get("years_experience") is not None for p in providers)

    @pytest.mark.parametrize(
        "bundle",
        [
            {},
            {"resourceType": "Patient"},
            {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []},
        ],
    )
    def test_transform_handles_empty_and_invalid_bundles(self, transformer, bundle):
        assert transformer.transform_bundle(bundle) == []


class TestCreateFHIRClient:
    @patch("fhir.client.get_config")
    def test_factory_returns_mock_client_satisfying_protocol(self, mock_config):
        mock_config.return_value.FHIR_USE_MOCK = True
        client = create_fhir_client()
        assert isinstance(client, MockFHIRClient)
        assert isinstance(client, FHIRClientProtocol)
