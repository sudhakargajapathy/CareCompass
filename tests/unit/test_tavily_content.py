"""Tavily `content`: the chunks knob that steers it, and the bound on it.

Two changes are guarded here, and they are related by one fact that was
misread for a long time: BOTH extraction prompts have always pasted Tavily's
`content` field into the page block, alongside the anchored excerpt of
`raw_content`. It was never an unused field. That means

  * `chunks_per_source` was already steering extraction input, unset, at the
    vendor's default of 3 — so raising it is a live change, not a no-op; and
  * `content` was the one field in that block with no bound, while its
    neighbour `raw_content` had been budgeted since round 12.

Field measurements behind the numbers live beside the constants
(`TAVILY_CHUNKS_PER_SOURCE` in utils/config.py, `_CONTENT_MAX_CHARS` in
agents/data_gatherer.py).
"""
from unittest.mock import MagicMock, patch

import pytest

from agents.data_gatherer import (
    DataGathererAgent,
    _CONTENT_MAX_CHARS,
    _DISCOVERY_EXCERPT_BUDGET,
)


@pytest.fixture
def gatherer():
    with patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
        agent = DataGathererAgent()
        agent.tavily_client = MagicMock()
        agent.anthropic_client = MagicMock()
        return agent


class TestChunksPerSourceReachesTavily:
    """The knob has to arrive in the REQUEST, at either depth."""

    @pytest.mark.parametrize("depth", ["basic", "advanced"])
    def test_knob_is_sent_at_both_depths(self, gatherer, depth):
        """Measured: basic honours chunks_per_source exactly as advanced does
        (cps 1 -> 5 took content from 4,724 to 37,298 chars at BOTH depths).

        The first version of this call site withheld the parameter on basic,
        on the widely-documented belief that it is advanced-only and that the
        (since-removed) fast-demo toggle would otherwise trip an API
        rejection. Basic accepts it and honours it, so that guard would have
        cost any basic-depth deployment an 8x richer extraction input to
        prevent a rejection that never happens. Depth stays an env knob
        (TAVILY_SEARCH_DEPTH), so both parametrizations remain reachable.
        """
        gatherer.config.TAVILY_CHUNKS_PER_SOURCE = 5
        gatherer.tavily_client.search.return_value = {"results": []}

        gatherer._search_providers("q", search_depth=depth)

        kwargs = gatherer.tavily_client.search.call_args.kwargs
        assert kwargs["search_depth"] == depth
        assert kwargs["chunks_per_source"] == 5

    def test_knob_follows_config_not_a_literal(self, gatherer):
        """A hardcoded 5 would pass the test above while ignoring the knob."""
        gatherer.config.TAVILY_CHUNKS_PER_SOURCE = 2
        gatherer.tavily_client.search.return_value = {"results": []}

        gatherer._search_providers("q", search_depth="advanced")

        assert gatherer.tavily_client.search.call_args.kwargs["chunks_per_source"] == 2


def _captured_prompt(gatherer, call):
    """Run `call` and return the prompt text handed to Claude."""
    response = MagicMock()
    response.content = [MagicMock(text="[]")]
    gatherer.anthropic_client.messages.create.return_value = response
    with patch("agents.data_gatherer.get_cost_tracker", return_value=MagicMock()):
        call()
    kwargs = gatherer.anthropic_client.messages.create.call_args.kwargs
    return kwargs["messages"][0]["content"]


class TestContentIsBounded:
    """`content` is clipped like its neighbour, on BOTH extraction paths.

    Asserted on the composed prompt, not on `clip_words` — a helper-only test
    would stay green if the call site went back to pasting the field whole,
    which is the exact regression this guards.
    """

    def test_discovery_block_clips_content(self, gatherer):
        """A vendor field with no bound scales with a knob nobody re-reads."""
        huge = "reviewed " * 4000                       # ~36,000 chars
        pages = [{
            "title": "Top Neurologists", "url": "https://www.healthgrades.com/x",
            "content": huge, "raw_content": "",
        }]

        prompt = _captured_prompt(
            gatherer, lambda: gatherer._extract_page_shard(pages, "Neurology", "Chandler, AZ")
        )

        assert len(huge) > _CONTENT_MAX_CHARS * 5, "probe must exceed the bound"
        assert huge not in prompt
        content_line = next(
            line for line in prompt.splitlines() if line.startswith("Content: ")
        )
        assert len(content_line) <= _CONTENT_MAX_CHARS + len("Content: ") + 8
        assert content_line.rstrip().endswith("…"), "a silent cut reads as the whole page"

    def test_enrichment_block_clips_content(self, gatherer):
        """The enrichment pass builds its own block and had the same defect."""
        huge = "rated " * 4000
        results = [{
            "title": "Dr. Hemant Pandey", "url": "https://www.healthgrades.com/physician/x",
            "content": huge, "raw_content": "", "score": 0.9,
        }]

        prompt = _captured_prompt(
            gatherer,
            lambda: gatherer._extract_review_data_only(
                results, "Dr. Hemant Pandey, MD", specialty="Neurology",
                provider_location="Chandler, AZ",
            ),
        )

        assert huge not in prompt
        content_line = next(
            line for line in prompt.splitlines() if line.startswith("Content: ")
        )
        assert len(content_line) <= _CONTENT_MAX_CHARS + len("Content: ") + 8

    def test_short_content_is_passed_through_untouched(self, gatherer):
        """Clipping must not reshape text it did not need to touch — a bound
        that rewrites short input is its own source of drift."""
        short = "Rated 3.8 out of 5 3.8 from 88 ratings"
        pages = [{
            "title": "t", "url": "https://www.healthgrades.com/x",
            "content": short, "raw_content": "",
        }]

        prompt = _captured_prompt(
            gatherer, lambda: gatherer._extract_page_shard(pages, "Neurology", "Chandler, AZ")
        )

        assert f"Content: {short}" in prompt

    def test_content_bound_leaves_room_for_the_excerpt(self, gatherer):
        """`content` and the excerpt are complementary, so the bound on one
        must not be able to crowd out the other.

        Measured across a discovery pass: `content` carries the roster
        (65 provider entries vs the excerpt's 36, 55 rating+count vs 20) while
        the excerpt carries what the per-domain anchors target (11 mentions of
        years-of-experience vs 5, 47 of insurance vs 26). Setting the content
        bound at or above the excerpt budget would let one page's chunks eat
        the share the anchors depend on.
        """
        assert _CONTENT_MAX_CHARS < _DISCOVERY_EXCERPT_BUDGET
