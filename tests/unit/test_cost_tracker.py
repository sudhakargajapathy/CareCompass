"""Unit tests for the per-search cost tracker."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from utils.cost_tracker import (
    CostTracker,
    PRICING_PER_MTOK,
    TAVILY_COST_PER_CREDIT,
    get_cost_tracker,
    safe_usage,
)


@pytest.mark.unit
class TestSafeUsage:
    def test_anthropic_shape(self):
        response = Mock()
        response.usage = Mock(spec=["input_tokens", "output_tokens"])
        response.usage.input_tokens = 1200
        response.usage.output_tokens = 340
        assert safe_usage(response) == (1200, 340)

    def test_openai_shape(self):
        response = Mock()
        response.usage = Mock(spec=["prompt_tokens", "completion_tokens"])
        response.usage.prompt_tokens = 800
        response.usage.completion_tokens = 200
        assert safe_usage(response) == (800, 200)

    def test_bare_mock_and_none_are_harmless(self):
        # conftest fixtures use bare Mock() clients: usage attrs are Mocks,
        # which must degrade to zero instead of raising
        assert safe_usage(Mock()) == (0, 0)
        assert safe_usage(None) == (0, 0)
        no_usage = Mock(spec=[])
        assert safe_usage(no_usage) == (0, 0)


@pytest.mark.unit
class TestCostTracker:
    def test_llm_cost_math(self):
        tracker = CostTracker()
        tracker.record_llm("claude-sonnet-5", 1_000_000, 1_000_000, agent="critic_validator")

        summary = tracker.summary()
        pricing = PRICING_PER_MTOK["claude-sonnet-5"]
        assert summary["llm"]["cost_usd"] == pytest.approx(pricing["input"] + pricing["output"])
        assert summary["llm"]["by_agent"]["critic_validator"] > 0
        assert summary["total_usd"] == pytest.approx(summary["llm"]["cost_usd"])

    def test_opus_critic_pricing_entry(self):
        """The critic's default model must be priced — an unpriced model
        silently reports $0 and the cost card would understate."""
        tracker = CostTracker()
        tracker.record_llm("claude-opus-4-8", 1_000_000, 1_000_000, agent="critic_validator")

        summary = tracker.summary()
        pricing = PRICING_PER_MTOK["claude-opus-4-8"]
        assert pricing == {"input": 5.00, "output": 25.00}
        assert summary["llm"]["cost_usd"] == pytest.approx(30.00)

    def test_default_judge_model_is_priced(self):
        """Whatever JUDGE_MODEL defaults to must have a pricing row — the
        judge's silent-failure mode means an unpriced model would degrade
        rankings AND report $0 for it (terra: $2.50/$15)."""
        from utils.config import Config

        judge_default = Config().JUDGE_MODEL
        assert judge_default in PRICING_PER_MTOK
        assert PRICING_PER_MTOK["gpt-5.6-terra"] == {"input": 2.50, "output": 15.00}

        tracker = CostTracker()
        tracker.record_llm("gpt-5.6-terra", 1_000_000, 1_000_000, agent="preference_scorer")
        assert tracker.summary()["llm"]["cost_usd"] == pytest.approx(17.50)

    def test_unknown_model_records_zero_cost(self):
        tracker = CostTracker()
        tracker.record_llm("mystery-model", 5000, 5000)

        summary = tracker.summary()
        assert summary["llm"]["calls"] == 1
        assert summary["llm"]["cost_usd"] == 0.0

    def test_tavily_credit_math(self):
        tracker = CostTracker()
        tracker.record_tavily(depth="basic")
        tracker.record_tavily(depth="advanced")

        tavily = tracker.summary()["tavily"]
        assert tavily["searches"] == 2
        assert tavily["credits"] == 3  # basic=1 + advanced=2
        assert tavily["cost_usd"] == pytest.approx(3 * TAVILY_COST_PER_CREDIT)

    def test_embeddings_and_step_timings(self):
        tracker = CostTracker()
        tracker.record_embeddings(10_000)
        with tracker.step("gather_data"):
            pass

        summary = tracker.summary()
        assert summary["embeddings"]["tokens"] == 10_000
        assert summary["embeddings"]["cost_usd"] > 0
        assert "gather_data" in summary["step_timings"]

    def test_reset_clears_run(self):
        tracker = CostTracker()
        tracker.record_llm("claude-haiku-4-5", 100, 100)
        tracker.record_tavily()
        tracker.reset()

        summary = tracker.summary()
        assert summary["total_usd"] == 0.0
        assert summary["llm"]["calls"] == 0
        assert summary["tavily"]["searches"] == 0

    def test_concurrent_recording_is_lossless(self):
        # The critic validator and enrichment loop record from worker threads
        tracker = CostTracker()
        with ThreadPoolExecutor(max_workers=8) as executor:
            for _ in range(100):
                executor.submit(tracker.record_llm, "claude-haiku-4-5", 10, 10, "data_gatherer")

        assert tracker.summary()["llm"]["calls"] == 100

    def test_singleton(self):
        assert get_cost_tracker() is get_cost_tracker()
