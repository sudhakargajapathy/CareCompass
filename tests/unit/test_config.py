"""Unit tests for utils/config.py configuration management."""
import pytest
import os
from utils.config import Config, get_config, check_environment


class TestConfigValidation:
    """Tests for API key validation."""

    def test_validate_api_keys_all_present(self, mock_env_vars):
        """Test validation when all API keys are present."""
        config = Config()
        validation = config.validate_api_keys()

        assert validation["openai"] is True
        assert validation["anthropic"] is True
        assert validation["tavily"] is True

    def test_validate_api_keys_missing_openai(self, mock_env_missing_openai):
        """Test validation when OpenAI key is missing."""
        config = Config()
        validation = config.validate_api_keys()

        assert validation["openai"] is False
        assert validation["anthropic"] is True
        assert validation["tavily"] is True

    def test_validate_api_keys_all_missing(self, mock_env_missing_all):
        """Test validation when all keys are missing."""
        config = Config()
        validation = config.validate_api_keys()

        assert validation["openai"] is False
        assert validation["anthropic"] is False
        assert validation["tavily"] is False

    def test_validate_api_keys_empty_string(self, monkeypatch):
        """Test validation treats empty strings as invalid."""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("APP_ANTHROPIC_API_KEY", "valid-key")
        monkeypatch.setenv("TAVILY_API_KEY", "   ")  # Whitespace only

        config = Config()
        validation = config.validate_api_keys()

        assert validation["openai"] is False
        assert validation["anthropic"] is True
        assert validation["tavily"] is False


class TestGetMissingKeys:
    """Tests for get_missing_keys method."""

    def test_get_missing_keys_none_missing(self, mock_env_vars):
        """Test when no keys are missing."""
        config = Config()
        missing = config.get_missing_keys()

        assert missing == []
        assert len(missing) == 0

    def test_get_missing_keys_one_missing(self, mock_env_missing_openai):
        """Test when one key is missing."""
        config = Config()
        missing = config.get_missing_keys()

        assert "openai" in missing
        assert len(missing) == 1

    def test_get_missing_keys_all_missing(self, mock_env_missing_all):
        """Test when all keys are missing."""
        config = Config()
        missing = config.get_missing_keys()

        assert set(missing) == {"openai", "anthropic", "tavily"}
        assert len(missing) == 3

    def test_get_missing_keys_multiple_missing(self, monkeypatch):
        """Test when multiple keys are missing."""
        monkeypatch.setenv("OPENAI_API_KEY", "valid-key")
        monkeypatch.delenv("APP_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        config = Config()
        missing = config.get_missing_keys()

        assert "anthropic" in missing
        assert "tavily" in missing
        assert "openai" not in missing
        assert len(missing) == 2


class TestEnvironmentSettings:
    """Tests for environment-related settings."""

    def test_is_production_true(self, monkeypatch):
        """Test is_production returns True for production env."""
        monkeypatch.setenv("ENV", "production")
        config = Config()

        assert config.is_production() is True

    def test_is_production_true_case_insensitive(self, monkeypatch):
        """Test is_production is case insensitive."""
        monkeypatch.setenv("ENV", "PRODUCTION")
        config = Config()

        assert config.is_production() is True

    def test_is_production_false_development(self, monkeypatch):
        """Test is_production returns False for development."""
        monkeypatch.setenv("ENV", "development")
        config = Config()

        assert config.is_production() is False

    def test_is_production_false_default(self, monkeypatch):
        """Test is_production returns False by default."""
        monkeypatch.delenv("ENV", raising=False)
        config = Config()

        assert config.is_production() is False
        assert config.ENV == "development"


class TestDefaultValues:
    """Tests for default configuration values."""

    def test_default_search_radius(self, monkeypatch):
        """Test DEFAULT_SEARCH_RADIUS has correct default."""
        monkeypatch.delenv("DEFAULT_SEARCH_RADIUS", raising=False)
        config = Config()

        assert config.DEFAULT_SEARCH_RADIUS == 25

    def test_default_search_radius_custom(self, monkeypatch):
        """Test DEFAULT_SEARCH_RADIUS can be customized."""
        monkeypatch.setenv("DEFAULT_SEARCH_RADIUS", "50")
        config = Config()

        assert config.DEFAULT_SEARCH_RADIUS == 50

    def test_max_providers_per_search_default(self, monkeypatch):
        """Test MAX_PROVIDERS_PER_SEARCH has correct default."""
        monkeypatch.delenv("MAX_PROVIDERS_PER_SEARCH", raising=False)
        config = Config()

        assert config.MAX_PROVIDERS_PER_SEARCH == 20

    def test_max_providers_per_search_custom(self, monkeypatch):
        """Test MAX_PROVIDERS_PER_SEARCH can be customized."""
        monkeypatch.setenv("MAX_PROVIDERS_PER_SEARCH", "30")
        config = Config()

        assert config.MAX_PROVIDERS_PER_SEARCH == 30

    def test_embedding_model_default(self, monkeypatch):
        """Test EMBEDDING_MODEL has correct default."""
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        config = Config()

        assert config.EMBEDDING_MODEL == "text-embedding-3-small"

    def test_app_name_default(self, monkeypatch):
        """Test APP_NAME has correct default."""
        monkeypatch.delenv("APP_NAME", raising=False)
        config = Config()

        assert config.APP_NAME == "CareCompass"

    def test_debug_default_false(self, monkeypatch):
        """Test DEBUG defaults to False."""
        monkeypatch.delenv("DEBUG", raising=False)
        config = Config()

        assert config.DEBUG is False

    def test_debug_true(self, monkeypatch):
        """Test DEBUG can be set to True."""
        monkeypatch.setenv("DEBUG", "true")
        config = Config()

        assert config.DEBUG is True

    def test_debug_case_insensitive(self, monkeypatch):
        """Test DEBUG is case insensitive."""
        monkeypatch.setenv("DEBUG", "TRUE")
        config = Config()

        assert config.DEBUG is True


class TestChromaSettings:
    """Tests for ChromaDB configuration."""

    def test_chroma_persist_directory_default(self, monkeypatch):
        """Test CHROMA_PERSIST_DIRECTORY has correct default."""
        monkeypatch.delenv("CHROMA_PERSIST_DIRECTORY", raising=False)
        config = Config()

        assert config.CHROMA_PERSIST_DIRECTORY == "./chroma_db"

    def test_chroma_persist_directory_custom(self, monkeypatch):
        """Test CHROMA_PERSIST_DIRECTORY can be customized."""
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "/tmp/test_chroma")
        config = Config()

        assert config.CHROMA_PERSIST_DIRECTORY == "/tmp/test_chroma"

    def test_chroma_collection_name_default(self, monkeypatch):
        """Test CHROMA_COLLECTION_NAME has correct default."""
        monkeypatch.delenv("CHROMA_COLLECTION_NAME", raising=False)
        config = Config()

        assert config.CHROMA_COLLECTION_NAME == "healthcare_providers"

    def test_chroma_collection_name_custom(self, monkeypatch):
        """Test CHROMA_COLLECTION_NAME can be customized."""
        monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_providers")
        config = Config()

        assert config.CHROMA_COLLECTION_NAME == "test_providers"


class TestFactoryFunctions:
    """Tests for factory and helper functions."""

    def test_get_config_returns_config_instance(self, mock_env_vars):
        """Test get_config returns a Config instance."""
        config = get_config()

        assert isinstance(config, Config)
        assert hasattr(config, "OPENAI_API_KEY")
        assert hasattr(config, "validate_api_keys")

    def test_check_environment_all_keys_present(self, mock_env_vars):
        """Test check_environment when all keys are valid."""
        is_valid, missing_keys = check_environment()

        assert is_valid is True
        assert missing_keys == []

    def test_check_environment_missing_keys(self, mock_env_missing_openai):
        """Test check_environment when keys are missing."""
        is_valid, missing_keys = check_environment()

        assert is_valid is False
        assert "openai" in missing_keys
        assert len(missing_keys) == 1

    def test_check_environment_all_missing(self, mock_env_missing_all):
        """Test check_environment when all keys are missing."""
        is_valid, missing_keys = check_environment()

        assert is_valid is False
        assert len(missing_keys) == 3
        assert set(missing_keys) == {"openai", "anthropic", "tavily"}


class TestConfigIntegrity:
    """Tests for configuration integrity and edge cases."""

    def test_api_keys_loaded_from_env(self, mock_env_vars):
        """Test API keys are properly loaded from environment."""
        config = Config()

        assert config.OPENAI_API_KEY == "test-openai-key-12345"
        assert config.ANTHROPIC_API_KEY == "test-anthropic-key-67890"
        assert config.TAVILY_API_KEY == "test-tavily-key-abcde"

    def test_config_attributes_exist(self, mock_env_vars):
        """Test all expected config attributes exist."""
        config = Config()

        # API Keys
        assert hasattr(config, "OPENAI_API_KEY")
        assert hasattr(config, "ANTHROPIC_API_KEY")
        assert hasattr(config, "TAVILY_API_KEY")

        # App Settings
        assert hasattr(config, "APP_NAME")
        assert hasattr(config, "ENV")
        assert hasattr(config, "DEBUG")

        # ChromaDB
        assert hasattr(config, "CHROMA_PERSIST_DIRECTORY")
        assert hasattr(config, "CHROMA_COLLECTION_NAME")

        # Search Parameters
        assert hasattr(config, "DEFAULT_SEARCH_RADIUS")
        assert hasattr(config, "MAX_PROVIDERS_PER_SEARCH")
        assert hasattr(config, "EMBEDDING_MODEL")

    def test_integer_parsing_handles_invalid_input(self, monkeypatch):
        """Test integer config values handle invalid input gracefully."""
        # Note: This will raise ValueError in current implementation
        # This test documents the behavior - consider adding error handling
        monkeypatch.setenv("DEFAULT_SEARCH_RADIUS", "invalid")

        with pytest.raises(ValueError):
            config = Config()

    def test_multiple_config_instances_share_values(self, mock_env_vars):
        """Test multiple Config instances see same environment values."""
        config1 = Config()
        config2 = Config()

        assert config1.OPENAI_API_KEY == config2.OPENAI_API_KEY
        assert config1.DEFAULT_SEARCH_RADIUS == config2.DEFAULT_SEARCH_RADIUS
