"""Unit tests for the CriticValidatorAgent."""

import pytest
from unittest.mock import MagicMock, patch
import json
from agents.critic_validator import CriticValidatorAgent
from tests.fixtures.mock_agent_responses import (
    MOCK_RANKED_PROVIDERS,
    MOCK_BIAS_ANALYSIS_RESPONSE,
    MOCK_ALTERNATIVE_RANKINGS_RESPONSE,
    MOCK_VALIDATION_RESPONSE
)

@pytest.fixture
def critic_validator():
    """Fixture to create a CriticValidatorAgent with a mocked Anthropic client."""
    with patch.object(CriticValidatorAgent, '_initialize_client', return_value=None):
        agent = CriticValidatorAgent()
        agent.anthropic_client = MagicMock()
        return agent

def test_analyze_ranking_bias_success(critic_validator: CriticValidatorAgent):
    """Test the _analyze_ranking_bias method for successful analysis."""
    
    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_BIAS_ANALYSIS_RESPONSE
    critic_validator.anthropic_client.messages.create.return_value = mock_response

    analysis = critic_validator._analyze_ranking_bias(MOCK_RANKED_PROVIDERS, {})
    
    assert "bias_assessment" in analysis
    assert analysis["bias_assessment"]["severity"] == "low"

def test_generate_alternative_rankings_success(critic_validator: CriticValidatorAgent):
    """Test the _generate_alternative_rankings method."""

    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_ALTERNATIVE_RANKINGS_RESPONSE
    critic_validator.anthropic_client.messages.create.return_value = mock_response
    
    alternatives = critic_validator._generate_alternative_rankings(MOCK_RANKED_PROVIDERS, {})
    
    assert len(alternatives) == 1
    assert alternatives[0]["scenario_name"] == "Quality-First Perspective"

def test_validate_top_recommendations_success(critic_validator: CriticValidatorAgent):
    """Test the _validate_top_recommendations method."""

    mock_response = MagicMock()
    mock_response.content[0].text = MOCK_VALIDATION_RESPONSE
    critic_validator.anthropic_client.messages.create.return_value = mock_response

    validation = critic_validator._validate_top_recommendations(MOCK_RANKED_PROVIDERS)
    
    assert "top_provider_validations" in validation
    assert len(validation["top_provider_validations"]) == 1
    assert validation["top_provider_validations"][0]["validation_status"] == "approved"

@patch('agents.critic_validator.CriticValidatorAgent._analyze_ranking_bias')
@patch('agents.critic_validator.CriticValidatorAgent._generate_alternative_rankings')
@patch('agents.critic_validator.CriticValidatorAgent._validate_top_recommendations')
def test_validate_rankings_main_method(mock_validate, mock_alternatives, mock_bias, critic_validator: CriticValidatorAgent):
    """Test the main validate_rankings method."""
    
    mock_bias.return_value = json.loads(MOCK_BIAS_ANALYSIS_RESPONSE)
    mock_alternatives.return_value = json.loads(MOCK_ALTERNATIVE_RANKINGS_RESPONSE)
    mock_validate.return_value = json.loads(MOCK_VALIDATION_RESPONSE)
    
    result = critic_validator.validate_rankings(MOCK_RANKED_PROVIDERS, {})
    
    assert result['status'] == 'success'
    assert 'bias_analysis' in result['validation_results']
    assert 'alternative_rankings' in result['validation_results']
    assert 'top_provider_validation' in result['validation_results']
    mock_bias.assert_called_once()
    mock_alternatives.assert_called_once()
    mock_validate.assert_called_once()

def test_validate_rankings_no_providers(critic_validator: CriticValidatorAgent):
    """Test that validate_rankings handles an empty list of providers."""
    
    result = critic_validator.validate_rankings([], {})
    
    assert result['status'] == 'no_providers'
