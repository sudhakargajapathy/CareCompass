"""Shared pytest fixtures for all tests."""
import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from tests.fixtures.mock_providers import SAMPLE_PROVIDERS, PROVIDER_WITH_MISSING_DATA, PROVIDER_LOW_REVIEWS
from tests.fixtures.mock_api_responses import (
    MOCK_TAVILY_SEARCH_RESULTS,
    MOCK_CLAUDE_EXTRACTION_RESPONSE,
    MOCK_GPT_RANKING_RESPONSE,
    MOCK_CLAUDE_VALIDATION_RESPONSE,
    MOCK_CLAUDE_VALIDATION_MARKDOWN_WRAPPED
)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-12345")
    monkeypatch.setenv("APP_ANTHROPIC_API_KEY", "test-anthropic-key-67890")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key-abcde")
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "./test_chroma_db")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_providers")


@pytest.fixture
def mock_env_missing_openai(monkeypatch):
    """Environment with missing OpenAI key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("APP_ANTHROPIC_API_KEY", "test-anthropic-key-67890")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key-abcde")


@pytest.fixture
def mock_env_missing_all(monkeypatch):
    """Environment with all API keys missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("APP_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


@pytest.fixture
def sample_providers():
    """Sample provider data for testing."""
    return SAMPLE_PROVIDERS.copy()


@pytest.fixture
def provider_with_missing_data():
    """Provider with missing/incomplete data."""
    return PROVIDER_WITH_MISSING_DATA.copy()


@pytest.fixture
def provider_low_reviews():
    """Provider with low review count for Bayesian testing."""
    return PROVIDER_LOW_REVIEWS.copy()


@pytest.fixture
def mock_tavily_response():
    """Mock Tavily search API response."""
    return MOCK_TAVILY_SEARCH_RESULTS.copy()


@pytest.fixture
def mock_claude_response():
    """Mock Anthropic Claude response."""
    return MOCK_CLAUDE_EXTRACTION_RESPONSE.copy()


@pytest.fixture
def mock_claude_invalid_json():
    """Mock Claude response with invalid JSON."""
    from tests.fixtures.mock_api_responses import MOCK_CLAUDE_INVALID_JSON_RESPONSE
    return MOCK_CLAUDE_INVALID_JSON_RESPONSE.copy()


@pytest.fixture
def mock_gpt_response():
    """Mock OpenAI GPT response."""
    return MOCK_GPT_RANKING_RESPONSE.copy()


@pytest.fixture
def mock_claude_validation_response():
    """Mock Claude Sonnet validation response."""
    return MOCK_CLAUDE_VALIDATION_RESPONSE.copy()


@pytest.fixture
def mock_claude_markdown_wrapped():
    """Mock Claude response with markdown-wrapped JSON."""
    return MOCK_CLAUDE_VALIDATION_MARKDOWN_WRAPPED.copy()


@pytest.fixture
def temp_chroma_db(tmp_path):
    """Create a temporary ChromaDB directory for testing."""
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    yield str(chroma_dir)
    # Cleanup handled by tmp_path fixture


@pytest.fixture
def mock_tavily_client():
    """Mock TavilyClient for testing."""
    mock_client = Mock()
    mock_client.search = Mock(return_value={"results": MOCK_TAVILY_SEARCH_RESULTS})
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    mock_client = Mock()
    mock_message = Mock()
    mock_message.content = MOCK_CLAUDE_EXTRACTION_RESPONSE["content"]
    mock_message.model = "claude-3-5-haiku-20241022"
    mock_client.messages.create = Mock(return_value=mock_message)
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()

    # Mock embeddings
    mock_embedding = Mock()
    mock_embedding.embedding = [0.1] * 1536  # 1536-dimensional vector
    mock_embeddings_response = Mock()
    mock_embeddings_response.data = [mock_embedding]
    mock_client.embeddings.create = Mock(return_value=mock_embeddings_response)

    # Mock chat completions
    mock_choice = Mock()
    mock_choice.message.content = MOCK_GPT_RANKING_RESPONSE["choices"][0]["message"]["content"]
    mock_completion = Mock()
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create = Mock(return_value=mock_completion)

    return mock_client


@pytest.fixture
def mock_vector_store():
    """Mock ProviderVectorStore for unit tests."""
    mock_store = Mock()
    mock_store.add_providers = Mock(return_value=True)
    mock_store.search_providers = Mock(return_value=[])
    mock_store.get_collection_stats = Mock(return_value={"provider_count": 0})
    mock_store.clear_collection = Mock()
    return mock_store


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.OPENAI_API_KEY = "test-openai-key"
    config.ANTHROPIC_API_KEY = "test-anthropic-key"
    config.TAVILY_API_KEY = "test-tavily-key"
    config.CHROMA_PERSIST_DIRECTORY = "./test_chroma_db"
    config.CHROMA_COLLECTION_NAME = "test_providers"
    config.EMBEDDING_MODEL = "text-embedding-3-small"
    config.DEFAULT_SEARCH_RADIUS = 25
    config.MAX_PROVIDERS_PER_SEARCH = 20
    config.validate_api_keys = Mock(return_value=True)
    config.get_missing_keys = Mock(return_value=[])
    config.is_production = Mock(return_value=False)
    return config


@pytest.fixture
def mock_workflow_state():
    """Mock WorkflowState for orchestrator testing."""
    return {
        "specialty": "Neurology",
        "location": "Phoenix, AZ",
        "insurance": "Aetna",
        "preferences": {
            "location_weight": 0.4,
            "rating_weight": 0.3,
            "insurance_weight": 0.3
        },
        "gathered_data": {},
        "scored_providers": {},
        "validation_results": {},
        "current_step": "initialize",
        "workflow_id": "test-workflow-123",
        "error_messages": [],
        "execution_log": [],
        "final_recommendations": [],
        "workflow_summary": {}
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # This ensures clean state for each test
    yield
    # Cleanup code would go here if needed


@pytest.fixture
def mock_progress_callback():
    """Mock progress callback for streaming tests."""
    return Mock()
