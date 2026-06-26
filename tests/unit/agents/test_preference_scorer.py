"""Unit tests for the PreferenceScorerAgent."""

import pytest
from unittest.mock import MagicMock, patch
from agents.preference_scorer import PreferenceScorerAgent, calculate_rating_score_with_confidence
from tests.fixtures.mock_agent_responses import MOCK_GATHER_PROVIDERS_RESULT, MOCK_OPENAI_RESPONSE

@pytest.fixture
def preference_scorer():
    """Fixture to create a PreferenceScorerAgent with a mocked OpenAI client."""
    with patch.object(PreferenceScorerAgent, '_initialize_client', return_value=None):
        agent = PreferenceScorerAgent()
        agent.openai_client = MagicMock()
        return agent

def test_calculate_rating_score_with_confidence():
    """Test the rating score calculation logic."""
    # Test high confidence
    result = calculate_rating_score_with_confidence(rating=4.5, review_count=100)
    assert result['score'] > 80
    assert result['confidence'] == 'high'

    # Test low confidence
    result = calculate_rating_score_with_confidence(rating=4.5, review_count=3)
    assert result['confidence'] == 'low'
    
    # Test no rating
    result = calculate_rating_score_with_confidence(rating=0, review_count=0)
    assert result['score'] == 0
    assert result['confidence'] == 'no_rating'

def test_calculate_base_scores(preference_scorer: PreferenceScorerAgent):
    """Test the _calculate_base_scores method."""
    providers = MOCK_GATHER_PROVIDERS_RESULT['providers']
    preferences = {"rating_weight": 0.5, "location_weight": 0.3, "insurance_priority": 0.2}
    
    scored_providers = preference_scorer._calculate_base_scores(providers, preferences)
    
    assert len(scored_providers) == 2
    assert "base_score" in scored_providers[0]
    assert scored_providers[0]['base_score'] > 0
    assert "score_breakdown" in scored_providers[0]

@patch('agents.preference_scorer.PreferenceScorerAgent._generate_ai_rankings')
def test_score_providers_success(mock_ai_rankings, preference_scorer: PreferenceScorerAgent):
    """Test the main score_providers method for a successful run."""
    
    # Let AI ranking return providers in the same order
    mock_ai_rankings.side_effect = lambda providers, prefs: providers

    providers = MOCK_GATHER_PROVIDERS_RESULT['providers']
    preferences = {"rating_weight": 0.5, "location_weight": 0.3, "insurance_priority": 0.2}

    result = preference_scorer.score_providers(providers, preferences)
    
    assert result['status'] == 'success'
    assert len(result['ranked_providers']) == 2
    assert "final_score" in result['ranked_providers'][0]
    assert result['ranked_providers'][0]['final_rank'] == 1

def test_generate_ai_rankings_success(preference_scorer: PreferenceScorerAgent):
    """Test the _generate_ai_rankings method for successful AI ranking."""
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = MOCK_OPENAI_RESPONSE
    preference_scorer.openai_client.chat.completions.create.return_value = mock_response

    providers = preference_scorer._calculate_base_scores(MOCK_GATHER_PROVIDERS_RESULT['providers'], {})
    
    ranked_providers = preference_scorer._generate_ai_rankings(providers, {})
    
    assert len(ranked_providers) == 2
    assert "ai_rank" in ranked_providers[0]
    assert ranked_providers[0]['ai_rank'] == 1
    assert "Excellent ratings" in ranked_providers[0]['ai_strengths']

def test_score_providers_no_providers(preference_scorer: PreferenceScorerAgent):
    """Test that score_providers handles an empty list of providers."""
    
    result = preference_scorer.score_providers([], {})
    
    assert result['status'] == 'no_providers'
    assert len(result['ranked_providers']) == 0
