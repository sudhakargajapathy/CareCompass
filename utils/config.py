"""Configuration management for CareCompass healthcare provider matching system."""

import os
from typing import Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for managing API keys and application settings."""

    def __init__(self):
        """Initialize configuration by reading environment variables."""
        # API Keys
        self.OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.ANTHROPIC_API_KEY: Optional[str] = os.getenv("APP_ANTHROPIC_API_KEY")
        self.TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")

        # Application Settings
        self.APP_NAME: str = os.getenv("APP_NAME", "CareCompass")
        self.ENV: str = os.getenv("ENV", "development")
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

        # Authentication Settings
        self.AUTH_DATABASE_URL: Optional[str] = os.getenv("AUTH_DATABASE_URL") or os.getenv("DATABASE_URL")
        self.AUTH_BOOTSTRAP_ADMIN: bool = os.getenv("AUTH_BOOTSTRAP_ADMIN", "true").lower() == "true"
        self.APP_ADMIN_USERNAME: str = os.getenv("APP_ADMIN_USERNAME", "admin")
        self.APP_ADMIN_PASSWORD: Optional[str] = os.getenv("APP_ADMIN_PASSWORD")
        self.AUTH_MAX_FAILED_ATTEMPTS: int = int(os.getenv("AUTH_MAX_FAILED_ATTEMPTS", "5"))
        self.AUTH_LOCKOUT_SECONDS: int = int(os.getenv("AUTH_LOCKOUT_SECONDS", "300"))

        # Audit Logging
        self.AUDIT_LOG_PATH: str = os.getenv("AUDIT_LOG_PATH", "./logs/audit.log")

        # Rate Limiting
        self.RATE_LIMIT_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
        self.RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

        # ChromaDB Settings
        self.CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        self.CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "healthcare_providers")

        # Search Parameters
        try:
            self.DEFAULT_SEARCH_RADIUS: int = int(os.getenv("DEFAULT_SEARCH_RADIUS", "25"))
        except ValueError:
            raise ValueError("DEFAULT_SEARCH_RADIUS must be a valid integer")

        try:
            self.MAX_PROVIDERS_PER_SEARCH: int = int(os.getenv("MAX_PROVIDERS_PER_SEARCH", "20"))
        except ValueError:
            raise ValueError("MAX_PROVIDERS_PER_SEARCH must be a valid integer")

        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.MAX_PROVIDERS_TO_ENRICH: int = int(os.getenv("MAX_PROVIDERS_TO_ENRICH", "5"))

        # FHIR Integration Settings
        self.FHIR_ENABLED: bool = os.getenv("FHIR_ENABLED", "false").lower() == "true"
        self.FHIR_USE_MOCK: bool = os.getenv("FHIR_USE_MOCK", "true").lower() == "true"
        self.FHIR_BASE_URL: Optional[str] = os.getenv("FHIR_BASE_URL")
        self.FHIR_CLIENT_ID: Optional[str] = os.getenv("FHIR_CLIENT_ID")
        self.FHIR_CLIENT_SECRET: Optional[str] = os.getenv("FHIR_CLIENT_SECRET")
        self.FHIR_TOKEN_URL: Optional[str] = os.getenv("FHIR_TOKEN_URL")
        self.FHIR_REQUEST_TIMEOUT: int = int(os.getenv("FHIR_REQUEST_TIMEOUT", "30"))
        self.FHIR_MAX_PAGES: int = int(os.getenv("FHIR_MAX_PAGES", "5"))

    def validate_api_keys(self) -> dict[str, bool]:
        """Validate that all required API keys are present.

        Returns:
            dict: Dictionary with API key names and their validation status
        """
        return {
            "openai": self.OPENAI_API_KEY is not None and self.OPENAI_API_KEY.strip() != "",
            "anthropic": self.ANTHROPIC_API_KEY is not None and self.ANTHROPIC_API_KEY.strip() != "",
            "tavily": self.TAVILY_API_KEY is not None and self.TAVILY_API_KEY.strip() != "",
        }

    def get_missing_keys(self) -> list[str]:
        """Get list of missing API keys.

        Returns:
            list: List of missing API key names
        """
        validation = self.validate_api_keys()
        return [key for key, is_valid in validation.items() if not is_valid]

    def validate_fhir_config(self) -> dict[str, Any]:
        """Validate FHIR configuration settings.

        Returns:
            dict with 'is_valid' bool and 'errors' list
        """
        errors: list[str] = []

        if not self.FHIR_ENABLED:
            return {"is_valid": True, "errors": [], "note": "FHIR is disabled"}

        if self.FHIR_USE_MOCK:
            return {"is_valid": True, "errors": [], "note": "Using mock FHIR client"}

        # Real client requires endpoint configuration
        if not self.FHIR_BASE_URL:
            errors.append("FHIR_BASE_URL is required for real FHIR client")
        if not self.FHIR_TOKEN_URL:
            errors.append("FHIR_TOKEN_URL is required for OAuth2 authentication")
        if not self.FHIR_CLIENT_ID:
            errors.append("FHIR_CLIENT_ID is required for OAuth2 authentication")
        if not self.FHIR_CLIENT_SECRET:
            errors.append("FHIR_CLIENT_SECRET is required for OAuth2 authentication")

        return {"is_valid": len(errors) == 0, "errors": errors}

    def is_production(self) -> bool:
        """Check if running in production environment.

        Returns:
            bool: True if environment is production
        """
        return self.ENV.lower() == "production"


def get_config() -> Config:
    """Get configuration instance.

    Returns:
        Config: Configuration instance
    """
    return Config()


def check_environment() -> tuple[bool, list[str]]:
    """Check if environment is properly configured.

    Returns:
        tuple: (is_valid, missing_keys)
    """
    config = get_config()
    missing_keys = config.get_missing_keys()
    return len(missing_keys) == 0, missing_keys
