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
        # THE RESEARCH BUDGET. One cut, pinned before enrichment and then
        # honoured by the judge and the critic too — so this gates all three
        # stages, and providers past it reach no model at all (marked
        # `over_budget`, still ranked and still listed, never carded).
        #
        # It scales the bill close to linearly, because the two stages it now
        # gates are where the money is:
        #
        #     enrichment   ~9% of a run
        #     judge        ~20%
        #     critic       ~54%    <- Opus 4.8, two parallel calls
        #
        # Until round 10 this capped enrichment ALONE — rationing the 9% stage
        # while the 74% pair ran over the whole pool, and scoring providers
        # nobody had researched (a rubric on an empty record grades OUR
        # coverage and reports it as their quality). The percentages are from
        # a measured live run and shift with pool size.
        #
        # The comment here previously called this "a RUNAWAY GUARD, not a
        # rationing device", which was true only while it capped one cheap
        # stage. It rations.
        #
        # ENRICHMENT_POOL_SIZE was retired here: it existed solely as the
        # tier-1 boundary of a tiered budget that, at the observed pool size of
        # 10 against a cap of 10, never rationed anything. Its clamp invariant
        # (round 4) went with it — there is no second knob left to fall out of
        # step with this one.
        #
        # 10 -> 8 (2026-07-28). Two reasons, one measured and one structural:
        #
        #   * It equals ENRICHMENT_MAX_WORKERS, so enrichment is exactly ONE
        #     wave. At 10 against 8 workers the second wave carried two
        #     providers and still cost a full wave — ~18s of pure latency for
        #     20% of the work. The alignment is a PROPERTY of these two
        #     defaults, not an invariant: raise this knob alone and waves come
        #     back, which is correct behaviour, just slower.
        #   * It gates the two most expensive stages (judge ~20%, critic ~54%),
        #     so the cut is close to linear in both.
        #
        # What it COSTS is shortlist headroom. The shortlist is 5, drawn only
        # from providers that completed all three stages, so a budget of 10 left
        # room for 5 to come back unrecommendable and a budget of 8 leaves room
        # for 3. The 2026-07-28 run had at least 6 recommendable out of 10, so
        # 8 should still fill a page — but a short shortlist is the correct
        # failure here (see the register), not a bug to pad around, and this is
        # the knob to raise if pages start arriving with four cards.
        self.MAX_PROVIDERS_TO_ENRICH: int = int(os.getenv("MAX_PROVIDERS_TO_ENRICH", "8"))

        # How many providers enrichment researches CONCURRENTLY. Each one is an
        # independent Tavily search plus a Haiku extraction (~18s serially), so
        # this sets the number of waves the stage runs in, not the amount of
        # work: ceil(MAX_PROVIDERS_TO_ENRICH / this).
        #
        # It was a hardcoded 4, which at the standard budget of 10 meant THREE
        # sequential waves — the last of them half empty and still costing a
        # full wave. The 2026-07-28 field run measured the enrichment stage at
        # ~54s of a 145s search, and the timeline blamed the preference scorer
        # for all of it (see the orchestrator's enrich_reviews step).
        #
        # 8 rather than 10: it takes 10 providers from three waves to two, and
        # keeps a bound if MAX_PROVIDERS_TO_ENRICH is ever raised — an unbounded
        # pool would fan out as wide as the budget and invite provider-side
        # throttling. `_search_providers` already retries once on a transient
        # failure, which is the backstop if a burst is throttled anyway.
        self.ENRICHMENT_MAX_WORKERS: int = int(os.getenv("ENRICHMENT_MAX_WORKERS", "8"))

        # Whether the rubric judge scores the pool in TWO concurrent calls
        # (~24.4s -> ~14s) instead of one. Off restores the single call and
        # changes nothing else — see `_should_split_judge`.
        #
        # This is the only one of Phase 2's three splits with a knob, and the
        # asymmetry is the point. Discovery's pages and the critic's verdicts
        # are per-item and independent: nothing in either prompt compares one
        # item to another, so a split cannot change an answer. The judge is
        # different in kind — its bands are anchored PROSE, and a model
        # calibrates prose against the examples in the call with it. Splitting
        # it is a bet that the anchors are absolute enough, and the only way to
        # settle that bet is a live run comparing the halves. Until one has
        # been done, the bet needs an off switch that is not a code change.
        self.JUDGE_PARALLEL_ENABLED: bool = os.getenv(
            "JUDGE_PARALLEL_ENABLED", "true"
        ).lower() == "true"

        # Enrichment cache: how long stored review/tenure/insurance evidence
        # stays usable. Seven rather than five because ratings and review
        # counts drift slowly, while a 7-day window nearly doubles the hit rate
        # over a 5-day one. Set to 0 to disable reuse entirely (every search
        # runs cold) without touching the stored data.
        self.PROVIDER_CACHE_TTL_DAYS: float = float(
            os.getenv("PROVIDER_CACHE_TTL_DAYS", "7")
        )

        # Per-role model knobs — env flips instead of code changes for model
        # experiments. GATHERER (extraction, highest call volume) and CRITIC
        # are Anthropic model IDs; JUDGE is an OpenAI model ID (the scorer
        # uses the OpenAI client — cross-family judge/critic independence is
        # deliberate: the validator shouldn't share the scorer's blind
        # spots). Critic defaults to Opus 4.8: bias detection and red-flag
        # analysis are the deepest-reasoning role, and its output reorders
        # the final list.
        self.GATHERER_MODEL: str = os.getenv("GATHERER_MODEL", "claude-haiku-4-5")
        # Judge default is the FULL id "gpt-5.6-terra" — the bare "gpt-5.6"
        # alias routes to Sol, the $5/$30 frontier tier (2x Terra's price).
        self.JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gpt-5.6-terra")
        self.CRITIC_MODEL: str = os.getenv("CRITIC_MODEL", "claude-opus-4-8")

        # Multi-query recall (discovery). A single web query returns a pool
        # pre-clustered by whatever "best-of" pages exist for one city; several
        # phrasings surface more distinct providers, and — only when the home
        # pool comes back THIN — the search rings out to nearby cities inside
        # DEFAULT_SEARCH_RADIUS.
        self.MULTI_QUERY_ENABLED: bool = os.getenv("MULTI_QUERY_ENABLED", "true").lower() == "true"
        # Ring-expansion trigger — the tunable knob governing cost vs breadth.
        #
        # Set equal to MAX_PROVIDERS_TO_ENRICH so the rule has a derivation:
        # ring out exactly when the home city cannot fill the research budget.
        # The equality is INTENT, not an enforced invariant (round 4's clamp
        # between two knobs was deliberately removed) — lowering this alone is
        # a valid "only rescue genuinely sparse towns" setting.
        #
        # MIN_DISTINCT_LOCATIONS was deleted in round 10 along with the
        # clustering trigger it fed; see the note at the trigger in
        # agents/data_gatherer.py for why no unit repairs that metric.
        #
        # 10 -> 8 (2026-07-28), moved WITH MAX_PROVIDERS_TO_ENRICH so the
        # derivation above survives rather than becoming a stale comment beside
        # a changed number.
        #
        # This does NOT promise the ring stops firing. Round 12 measured
        # discovery recall at ~7 names from a 15-entry listicle, so a home pool
        # of 7 still trips a threshold of 8 — which is the ring behaving as the
        # rescue it was built to be. What changes is its BLAST RADIUS: with the
        # budget also at 8, a home pool of 7 leaves the ring at most ONE
        # researched slot to fill, where at a budget of 10 it could fill three,
        # each costing an enrichment search, a judge slot and an Opus verdict.
        # `ring_contribution` on the next run says whether even that one earns
        # its place; going to 6 or 7 is the setting that would stop it outright.
        self.MIN_CANDIDATE_POOL: int = int(os.getenv("MIN_CANDIDATE_POOL", "8"))
        self.MAX_RING_CITIES: int = int(os.getenv("MAX_RING_CITIES", "2"))
        # Tavily depth: "basic" is 1 credit and fast, "advanced" is 2 credits
        # and slower but digs deeper. The UI's fast-demo toggle sets this.
        self.TAVILY_SEARCH_DEPTH: str = os.getenv("TAVILY_SEARCH_DEPTH", "basic")

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
