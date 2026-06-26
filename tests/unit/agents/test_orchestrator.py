"""Unit tests for the ProviderMatchingOrchestrator."""

import pytest
from unittest.mock import MagicMock, patch
from agents.orchestrator import ProviderMatchingOrchestrator, WorkflowState
from tests.fixtures.mock_agent_responses import (
    MOCK_GATHER_PROVIDERS_RESULT,
    MOCK_SCORED_PROVIDERS_RESULT,
    MOCK_VALIDATION_RESULT,
)

@pytest.fixture
def orchestrator():
    """Fixture to create a ProviderMatchingOrchestrator with mocked agents."""
    with patch('agents.orchestrator.DataGathererAgent') as mock_data_gatherer, \
         patch('agents.orchestrator.PreferenceScorerAgent') as mock_scorer, \
         patch('agents.orchestrator.CriticValidatorAgent') as mock_validator, \
         patch('agents.orchestrator.get_vector_store') as mock_vector_store:
        
        orchestrator = ProviderMatchingOrchestrator()
        
        # Configure mock agents
        orchestrator.data_gatherer.gather_providers.return_value = MOCK_GATHER_PROVIDERS_RESULT
        orchestrator.preference_scorer.score_providers.return_value = MOCK_SCORED_PROVIDERS_RESULT
        orchestrator.critic_validator.validate_rankings.return_value = MOCK_VALIDATION_RESULT
        
        yield orchestrator

def test_orchestrator_initialization(orchestrator: ProviderMatchingOrchestrator):
    """Test that the orchestrator and its workflow are initialized correctly."""
    assert orchestrator.workflow is not None
    assert orchestrator.data_gatherer is not None
    assert orchestrator.preference_scorer is not None
    assert orchestrator.critic_validator is not None

def test_execute_workflow_success(orchestrator: ProviderMatchingOrchestrator):
    """Test a full, successful execution of the workflow."""
    
    result = orchestrator.execute_workflow(
        specialty="Neurology",
        location="Phoenix, AZ"
    )
    
    assert result['success']
    assert len(result['final_recommendations']) > 0
    assert result['workflow_summary']['total_providers_found'] == 2
    
    # Check that agents were called
    orchestrator.data_gatherer.gather_providers.assert_called_once()
    orchestrator.preference_scorer.score_providers.assert_called_once()
    orchestrator.critic_validator.validate_rankings.assert_called_once()

def test_execute_workflow_data_gathering_fails(orchestrator: ProviderMatchingOrchestrator):
    """Test workflow failure when the data gathering step fails."""
    
    # Simulate a failure in the data gatherer
    orchestrator.data_gatherer.gather_providers.return_value = {
        "status": "no_results",
        "providers": []
    }
    
    result = orchestrator.execute_workflow(
        specialty="Obscure Specialty",
        location="Remote Location"
    )
    
    assert not result['success']
    assert "data gathering" in result['error_messages'][0].lower()
    
    # Check that subsequent agents were not called
    orchestrator.preference_scorer.score_providers.assert_not_called()
    orchestrator.critic_validator.validate_rankings.assert_not_called()

def test_check_data_gathering_success(orchestrator: ProviderMatchingOrchestrator):
    """Test the conditional edge logic for data gathering."""
    
    # Success case
    state = {"gathered_data": {"status": "success"}}
    assert orchestrator._check_data_gathering_success(state) == "success"
    
    # Error case
    state = {"gathered_data": {"status": "error"}}
    assert orchestrator._check_data_gathering_success(state) == "error"
