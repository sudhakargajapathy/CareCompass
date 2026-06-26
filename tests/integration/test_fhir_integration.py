"""Integration tests for FHIR within the DataGathererAgent pipeline.

These tests use FHIR_ENABLED=true and FHIR_USE_MOCK=true to exercise
the full FHIR → transformer → merge pipeline without live API calls.
Tavily and Anthropic calls are mocked to isolate the FHIR integration logic.
"""

import pytest
from unittest.mock import patch, MagicMock

from agents.data_gatherer import DataGathererAgent, _NAME_MATCH_THRESHOLD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Config with FHIR enabled, mock mode, and fake API keys."""
    config = MagicMock()
    config.TAVILY_API_KEY = "fake-tavily-key"
    config.ANTHROPIC_API_KEY = "fake-anthropic-key"
    config.FHIR_ENABLED = True
    config.FHIR_USE_MOCK = True
    config.MAX_PROVIDERS_PER_SEARCH = 20
    config.MAX_PROVIDERS_TO_ENRICH = 3
    return config


@pytest.fixture
def tavily_providers():
    """Simulated Tavily-extracted providers overlapping with mock FHIR data."""
    return [
        {
            "name": "Dr. Sarah Chen",
            "specialty": "Neurology",
            "location": "Phoenix, AZ",
            "phone": "602-555-0101",
            "rating": 4.8,
            "review_count": 127,
            "review_summary": "Patients praise Dr. Chen for her thorough explanations and compassionate care.",
            "review_sentiment": "positive",
            "insurance_accepted": ["Aetna", "Blue Cross"],
            "distance": 3.2,
        },
        {
            "name": "Dr. John Smith",  # No FHIR match
            "specialty": "Neurology",
            "location": "Phoenix, AZ",
            "phone": "602-555-9999",
            "rating": 4.2,
            "review_count": 45,
            "review_summary": "Good neurologist with reasonable wait times.",
            "review_sentiment": "positive",
            "insurance_accepted": ["Cigna"],
            "distance": 5.0,
        },
    ]


# ---------------------------------------------------------------------------
# Name matching tests
# ---------------------------------------------------------------------------


class TestNameTokenOverlap:
    def test_exact_match(self):
        score = DataGathererAgent._name_token_overlap("Dr. Sarah Chen", "Dr. Sarah Chen")
        assert score == 1.0

    def test_partial_match(self):
        score = DataGathererAgent._name_token_overlap("Dr. Sarah Chen", "Sarah Chen MD")
        assert score >= _NAME_MATCH_THRESHOLD

    def test_no_match(self):
        score = DataGathererAgent._name_token_overlap("Dr. Sarah Chen", "Dr. John Smith")
        assert score < _NAME_MATCH_THRESHOLD

    def test_prefix_stripped(self):
        score = DataGathererAgent._name_token_overlap("Dr. Martinez", "Martinez MD")
        assert score == 1.0

    def test_empty_strings(self):
        assert DataGathererAgent._name_token_overlap("", "") == 0.0
        assert DataGathererAgent._name_token_overlap("Dr. Chen", "") == 0.0


# ---------------------------------------------------------------------------
# FHIR gathering tests
# ---------------------------------------------------------------------------


class TestFHIRGatherProviders:
    @patch("agents.data_gatherer.get_config")
    @patch("agents.data_gatherer.TavilyClient")
    @patch("agents.data_gatherer.Anthropic")
    def test_fhir_providers_returned(self, mock_anthropic_cls, mock_tavily_cls, mock_get_config, mock_config):
        """FHIR mock client returns providers when enabled."""
        mock_get_config.return_value = mock_config
        agent = DataGathererAgent()

        assert agent.fhir_client is not None
        providers = agent._gather_fhir_providers("Neurology", "Phoenix, AZ")
        assert len(providers) > 0
        assert all(p.get("data_source") == "fhir" for p in providers)

    @patch("agents.data_gatherer.get_config")
    @patch("agents.data_gatherer.TavilyClient")
    @patch("agents.data_gatherer.Anthropic")
    def test_fhir_providers_have_npi(self, mock_anthropic_cls, mock_tavily_cls, mock_get_config, mock_config):
        """FHIR providers include NPI in metadata."""
        mock_get_config.return_value = mock_config
        agent = DataGathererAgent()

        providers = agent._gather_fhir_providers("Neurology", "Phoenix, AZ")
        for provider in providers:
            assert provider["fhir_metadata"]["npi"] is not None

    @patch("agents.data_gatherer.get_config")
    @patch("agents.data_gatherer.TavilyClient")
    @patch("agents.data_gatherer.Anthropic")
    def test_fhir_network_filter(self, mock_anthropic_cls, mock_tavily_cls, mock_get_config, mock_config):
        """FHIR results filtered by insurance network."""
        mock_get_config.return_value = mock_config
        agent = DataGathererAgent()

        all_providers = agent._gather_fhir_providers("Neurology", "Phoenix, AZ")
        aetna_providers = agent._gather_fhir_providers("Neurology", "Phoenix, AZ", "Aetna")

        assert len(aetna_providers) <= len(all_providers)
        for p in aetna_providers:
            networks = [n.lower() for n in p["fhir_metadata"].get("networks", [])]
            assert any("aetna" in n for n in networks)


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------


class TestMergeFHIRAndTavily:
    @patch("agents.data_gatherer.get_config")
    @patch("agents.data_gatherer.TavilyClient")
    @patch("agents.data_gatherer.Anthropic")
    def test_merge_combines_sources(self, mock_anthropic_cls, mock_tavily_cls, mock_get_config, mock_config, tavily_providers):
        """Merged list contains both matched and unmatched providers."""
        mock_get_config.return_value = mock_config
        agent = DataGathererAgent()

        fhir_providers = agent._gather_fhir_providers("Neurology", "Phoenix, AZ")
        merged = agent._merge_fhir_and_tavily_providers(fhir_providers, tavily_providers)

        # Should have more providers than either source alone
        assert len(merged) >= max(len(fhir_providers), len(tavily_providers))

    @patch("agents.data_gatherer.get_config")
    @patch("agents.data_gatherer.TavilyClient")
    @patch("agents.data_gatherer.Anthropic")
    def test_matched_provider_has_both_sources(self, mock_anthropic_cls, mock_tavily_cls, mock_get_config, mock_config, tavily_providers):
        """Matched provider gets data_source='fhir+tavily'."""
        mock_get_config.return_value = mock_config
        agent = DataGathererAgent()

        fhir_providers = agent._gather_fhir_providers("Neurology", "Phoenix, AZ")
        merged = agent._merge_fhir_and_tavily_providers(fhir_providers, tavily_providers)

        combined = [p for p in merged if p.get("data_source") == "fhir+tavily"]
        assert len(combined) > 0, "Expected at least one FHIR+Tavily merged provider"

        for p in combined:
            # Should have FHIR metadata
            assert p["fhir_metadata"]["network_verified"] is True
            assert p["fhir_metadata"]["npi"] is not None
            # Should have Tavily rating/reviews
            assert p["rating"] > 0
            assert p["review_count"] is not None

    @patch("agents.data_gatherer.get_config")
    @patch("agents.data_gatherer.TavilyClient")
    @patch("agents.data_gatherer.Anthropic")
    def test_unmatched_tavily_kept(self, mock_anthropic_cls, mock_tavily_cls, mock_get_config, mock_config, tavily_providers):
        """Tavily providers without FHIR match are kept with data_source='tavily'."""
        mock_get_config.return_value = mock_config
        agent = DataGathererAgent()

        fhir_providers = agent._gather_fhir_providers("Neurology", "Phoenix, AZ")
        merged = agent._merge_fhir_and_tavily_providers(fhir_providers, tavily_providers)

        tavily_only = [p for p in merged if p.get("data_source") == "tavily"]
        # Dr. John Smith has no FHIR match
        assert any("Smith" in p["name"] for p in tavily_only)

    @patch("agents.data_gatherer.get_config")
    @patch("agents.data_gatherer.TavilyClient")
    @patch("agents.data_gatherer.Anthropic")
    def test_unmatched_fhir_kept(self, mock_anthropic_cls, mock_tavily_cls, mock_get_config, mock_config, tavily_providers):
        """FHIR providers without Tavily match are kept with data_source='fhir'."""
        mock_get_config.return_value = mock_config
        agent = DataGathererAgent()

        fhir_providers = agent._gather_fhir_providers("Neurology", "Phoenix, AZ")
        merged = agent._merge_fhir_and_tavily_providers(fhir_providers, tavily_providers)

        fhir_only = [p for p in merged if p.get("data_source") == "fhir"]
        # FHIR has 4 neurology practitioners, only 1 matches Tavily (Dr. Chen)
        assert len(fhir_only) > 0

    @patch("agents.data_gatherer.get_config")
    @patch("agents.data_gatherer.TavilyClient")
    @patch("agents.data_gatherer.Anthropic")
    def test_no_duplicates(self, mock_anthropic_cls, mock_tavily_cls, mock_get_config, mock_config, tavily_providers):
        """No provider name should appear more than once."""
        mock_get_config.return_value = mock_config
        agent = DataGathererAgent()

        fhir_providers = agent._gather_fhir_providers("Neurology", "Phoenix, AZ")
        merged = agent._merge_fhir_and_tavily_providers(fhir_providers, tavily_providers)

        names = [p["name"] for p in merged]
        # Allow for close but not identical names (FHIR format may differ)
        # Check no exact duplicates
        assert len(names) == len(set(names)), f"Duplicate names found: {names}"


# ---------------------------------------------------------------------------
# FHIR metadata in search_metadata
# ---------------------------------------------------------------------------


class TestGatherProvidersFHIRMetadata:
    @patch("agents.data_gatherer.get_config")
    @patch("agents.data_gatherer.TavilyClient")
    @patch("agents.data_gatherer.Anthropic")
    def test_search_metadata_includes_fhir_count(self, mock_anthropic_cls, mock_tavily_cls, mock_get_config, mock_config):
        """gather_providers() includes fhir_count in search_metadata."""
        mock_get_config.return_value = mock_config

        # Mock Tavily to return no results (isolate FHIR)
        mock_tavily_instance = MagicMock()
        mock_tavily_instance.search.return_value = {"results": []}
        mock_tavily_cls.return_value = mock_tavily_instance

        # Need to mock validate_search_params to allow test inputs
        with patch("agents.data_gatherer.validate_search_params") as mock_validate:
            mock_validate.return_value = {
                "is_valid": True,
                "specialty": "Neurology",
                "location": "Phoenix, AZ",
                "insurance": None,
                "errors": [],
            }

            agent = DataGathererAgent()
            result = agent.gather_providers("Neurology", "Phoenix, AZ")

            assert "fhir_count" in result["search_metadata"]
            assert result["search_metadata"]["fhir_count"] > 0
            assert result["search_metadata"]["fhir_enabled"] is True

    @patch("agents.data_gatherer.get_config")
    @patch("agents.data_gatherer.TavilyClient")
    @patch("agents.data_gatherer.Anthropic")
    def test_fhir_disabled_returns_zero_fhir(self, mock_anthropic_cls, mock_tavily_cls, mock_get_config, mock_config):
        """When FHIR is disabled, fhir_count is 0."""
        mock_config.FHIR_ENABLED = False
        mock_get_config.return_value = mock_config

        mock_tavily_instance = MagicMock()
        mock_tavily_instance.search.return_value = {"results": []}
        mock_tavily_cls.return_value = mock_tavily_instance

        with patch("agents.data_gatherer.validate_search_params") as mock_validate:
            mock_validate.return_value = {
                "is_valid": True,
                "specialty": "Neurology",
                "location": "Phoenix, AZ",
                "insurance": None,
                "errors": [],
            }

            agent = DataGathererAgent()
            result = agent.gather_providers("Neurology", "Phoenix, AZ")

            assert result["search_metadata"].get("fhir_count", 0) == 0
