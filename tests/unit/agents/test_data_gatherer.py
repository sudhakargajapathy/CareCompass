"""Unit tests for the DataGathererAgent."""

import pytest
from unittest.mock import MagicMock, patch
from agents.data_gatherer import DataGathererAgent
from tests.fixtures.mock_agent_responses import (
    MOCK_TAVILY_SEARCH_RESPONSE,
    MOCK_CLAUDE_EXTRACTION_RESPONSE,
    MOCK_GATHER_PROVIDERS_RESULT
)

@pytest.fixture
def data_gatherer():
    """Fixture to create a DataGathererAgent with mocked clients."""
    with patch.object(DataGathererAgent, '_initialize_clients', return_value=None):
        agent = DataGathererAgent()
        agent.tavily_client = MagicMock()
        agent.anthropic_client = MagicMock()
        return agent

def test_build_search_query(data_gatherer: DataGathererAgent):
    """Test the _build_search_query method."""
    query = data_gatherer._build_search_query(
        specialty="Cardiology",
        location="New York, NY",
        insurance="Aetna"
    )
    assert "Cardiology" in query
    assert "New York, NY" in query
    assert "Aetna" in query
    assert "reviews" in query

@patch('agents.data_gatherer.DataGathererAgent._search_providers')
def test_gather_providers_success(mock_search, data_gatherer: DataGathererAgent):
    """Test the main gather_providers method for a successful run."""
    
    mock_search.return_value = MOCK_TAVILY_SEARCH_RESPONSE['results']
    
    # Mock the extraction method
    with patch.object(data_gatherer, '_extract_provider_data', return_value=MOCK_GATHER_PROVIDERS_RESULT['providers']) as mock_extract:
        
        result = data_gatherer.gather_providers(
            specialty="Neurology",
            location="Phoenix, AZ"
        )
        
        assert result['status'] == 'success'
        assert len(result['providers']) == 2
        assert result['providers'][0]['name'] == "Dr. Emily Carter"
        mock_search.assert_called_once()
        mock_extract.assert_called_once()

def test_extract_provider_data_success(data_gatherer: DataGathererAgent):
    """Test the _extract_provider_data method for successful extraction."""
    
    # Mock the Anthropic client's response
    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_CLAUDE_EXTRACTION_RESPONSE
    data_gatherer.anthropic_client.messages.create.return_value = mock_response
    
    providers = data_gatherer._extract_provider_data(
        search_results=MOCK_TAVILY_SEARCH_RESPONSE['results'],
        specialty="Neurology",
        location="Phoenix, AZ"
    )
    
    assert len(providers) == 2
    assert providers[0]['name'] == "Dr. Emily Carter"
    assert providers[1]['name'] == "Dr. Ben Adams"
    assert providers[0]['rating'] == 4.8
    assert "BCBS" in providers[0]['insurance_accepted']

def test_search_providers_api_failure(data_gatherer: DataGathererAgent):
    """Test how _search_providers handles a Tavily API failure."""
    
    data_gatherer.tavily_client.search.side_effect = Exception("API Error")
    
    results = data_gatherer._search_providers(query="test query")
    
    assert results == []

def test_extract_provider_data_json_error(data_gatherer: DataGathererAgent):
    """Test _extract_provider_data with a malformed JSON response from Claude."""
    
    mock_response = MagicMock()
    mock_response.content[0].text = "This is not valid JSON"
    data_gatherer.anthropic_client.messages.create.return_value = mock_response
    
    providers = data_gatherer._extract_provider_data(
        search_results=MOCK_TAVILY_SEARCH_RESPONSE['results'],
        specialty="Neurology",
        location="Phoenix, AZ"
    )
    
    assert providers == []
