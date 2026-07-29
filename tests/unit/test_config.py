"""Unit tests for utils/config.py configuration management."""
import pytest
from utils.config import Config, get_config, check_environment


class TestApiKeyValidation:
    def test_all_keys_present(self, mock_env_vars):
        config = Config()
        assert config.validate_api_keys() == {"openai": True, "anthropic": True, "tavily": True}
        assert config.get_missing_keys() == []

    @pytest.mark.parametrize(
        "unset, expected_missing",
        [
            (["OPENAI_API_KEY"], {"openai"}),
            (["APP_ANTHROPIC_API_KEY", "TAVILY_API_KEY"], {"anthropic", "tavily"}),
            (["OPENAI_API_KEY", "APP_ANTHROPIC_API_KEY", "TAVILY_API_KEY"], {"openai", "anthropic", "tavily"}),
        ],
    )
    def test_missing_keys_reported(self, mock_env_vars, monkeypatch, unset, expected_missing):
        for var in unset:
            monkeypatch.delenv(var, raising=False)
        assert set(Config().get_missing_keys()) == expected_missing

    def test_blank_keys_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("APP_ANTHROPIC_API_KEY", "valid-key")
        monkeypatch.setenv("TAVILY_API_KEY", "   ")  # whitespace only

        validation = Config().validate_api_keys()
        assert validation == {"openai": False, "anthropic": True, "tavily": False}


class TestEnvironmentSettings:
    @pytest.mark.parametrize(
        "env_value, expected",
        [("production", True), ("PRODUCTION", True), ("development", False), (None, False)],
    )
    def test_is_production(self, monkeypatch, env_value, expected):
        if env_value is None:
            monkeypatch.delenv("ENV", raising=False)
        else:
            monkeypatch.setenv("ENV", env_value)
        assert Config().is_production() is expected


class TestDefaultsAndOverrides:
    def test_defaults(self, monkeypatch):
        for var in (
            "DEFAULT_SEARCH_RADIUS", "MAX_PROVIDERS_PER_SEARCH", "EMBEDDING_MODEL",
            "APP_NAME", "DEBUG", "CHROMA_PERSIST_DIRECTORY", "CHROMA_COLLECTION_NAME",
            "TAVILY_SEARCH_DEPTH", "MAX_PROVIDERS_TO_ENRICH",
            "PROVIDER_CACHE_TTL_DAYS",
            "GATHERER_MODEL", "JUDGE_MODEL", "CRITIC_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)

        config = Config()
        assert config.DEFAULT_SEARCH_RADIUS == 25
        assert config.MAX_PROVIDERS_PER_SEARCH == 20
        assert config.EMBEDDING_MODEL == "text-embedding-3-small"
        assert config.APP_NAME == "CareCompass"
        assert config.DEBUG is False
        assert config.CHROMA_PERSIST_DIRECTORY == "./chroma_db"
        assert config.CHROMA_COLLECTION_NAME == "healthcare_providers"
        assert config.TAVILY_SEARCH_DEPTH == "basic"
        # THE RESEARCH BUDGET — it gates enrichment, the judge and the critic,
        # so this number scales the bill close to linearly. Equal to
        # ENRICHMENT_MAX_WORKERS by intent, which makes enrichment one wave.
        assert config.MAX_PROVIDERS_TO_ENRICH == 8
        assert config.ENRICHMENT_MAX_WORKERS == 8
        assert config.PROVIDER_CACHE_TTL_DAYS == 7
        # Per-role model knobs: critic defaults to the deepest-reasoning model;
        # judge must be the FULL terra id (bare "gpt-5.6" routes to Sol at 2x)
        assert config.GATHERER_MODEL == "claude-haiku-4-5"
        assert config.JUDGE_MODEL == "gpt-5.6-terra"
        assert config.CRITIC_MODEL == "claude-opus-4-8"

    def test_cache_ttl_is_overridable(self, monkeypatch):
        monkeypatch.setenv("PROVIDER_CACHE_TTL_DAYS", "3.5")
        assert Config().PROVIDER_CACHE_TTL_DAYS == 3.5

    def test_cache_ttl_zero_disables_reuse(self, monkeypatch):
        """0 forces every search cold without discarding stored data — the
        env-side equivalent of the sidebar's cold-run toggle."""
        monkeypatch.setenv("PROVIDER_CACHE_TTL_DAYS", "0")
        assert Config().PROVIDER_CACHE_TTL_DAYS == 0

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_SEARCH_RADIUS", "50")
        monkeypatch.setenv("MAX_PROVIDERS_PER_SEARCH", "10")
        monkeypatch.setenv("DEBUG", "TRUE")
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "/custom/path")
        monkeypatch.setenv("TAVILY_SEARCH_DEPTH", "advanced")
        monkeypatch.setenv("CRITIC_MODEL", "claude-sonnet-5")

        config = Config()
        assert config.DEFAULT_SEARCH_RADIUS == 50
        assert config.MAX_PROVIDERS_PER_SEARCH == 10
        assert config.DEBUG is True
        assert config.CHROMA_PERSIST_DIRECTORY == "/custom/path"
        assert config.TAVILY_SEARCH_DEPTH == "advanced"
        assert config.CRITIC_MODEL == "claude-sonnet-5"

    def test_invalid_integer_raises(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_SEARCH_RADIUS", "not-a-number")
        with pytest.raises(ValueError, match="DEFAULT_SEARCH_RADIUS"):
            Config()


class TestFactoryFunctions:
    def test_get_config_returns_fresh_instance(self, mock_env_vars):
        config = get_config()
        assert isinstance(config, Config)
        assert config.OPENAI_API_KEY is not None

    def test_check_environment(self, mock_env_vars, monkeypatch):
        is_valid, missing = check_environment()
        assert is_valid is True and missing == []

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        is_valid, missing = check_environment()
        assert is_valid is False and missing == ["openai"]


class TestResearchBudgetDefaults:
    """The three knobs that decide how much of a search costs, and the two
    relationships between their DEFAULTS that comments claim but nothing
    checked."""

    def test_ring_trigger_defaults_to_the_research_budget(self):
        """`MIN_CANDIDATE_POOL == MAX_PROVIDERS_TO_ENRICH` is what gives the
        ring trigger a derivation rather than a preference: ring out exactly
        when the home city cannot fill the research budget.

        Both moved 10 -> 8 together on 2026-07-28. Moving one alone would leave
        the other's comment describing a rule the code no longer follows —
        which is how `MIN_CANDIDATE_POOL`'s justification would quietly become
        fiction.

        This pins the shipped DEFAULTS and deliberately does NOT constrain the
        env vars. It is not round 4's clamp, which forced the two into step at
        runtime and was removed on purpose: `MIN_CANDIDATE_POOL=5` ("only
        rescue genuinely sparse towns") stays a legitimate setting.
        """
        config = Config()
        assert config.MIN_CANDIDATE_POOL == config.MAX_PROVIDERS_TO_ENRICH

    def test_the_budget_defaults_to_one_enrichment_wave(self):
        """Enrichment costs ceil(budget / workers) waves. At a budget of 10
        against 8 workers the second wave carried two providers and still cost
        a full wave — ~18s of latency for 20% of the work.

        Also a defaults-only property. Raising the budget alone brings waves
        back, which is correct behaviour and not something to forbid.
        """
        config = Config()
        waves = -(-config.MAX_PROVIDERS_TO_ENRICH // config.ENRICHMENT_MAX_WORKERS)
        assert waves == 1

    def test_the_knobs_are_still_independently_overridable(self, monkeypatch):
        """The defaults agree; the env is free. Guards against someone reading
        the two tests above as an invariant and clamping them in code."""
        monkeypatch.setenv("MIN_CANDIDATE_POOL", "5")
        monkeypatch.setenv("MAX_PROVIDERS_TO_ENRICH", "12")
        monkeypatch.setenv("ENRICHMENT_MAX_WORKERS", "4")

        config = Config()
        assert config.MIN_CANDIDATE_POOL == 5
        assert config.MAX_PROVIDERS_TO_ENRICH == 12
        assert config.ENRICHMENT_MAX_WORKERS == 4
