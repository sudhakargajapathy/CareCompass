"""Phase 2: the three one-call-over-N-items stages, split across concurrent calls.

Discovery extraction (25.3s), the rubric judge (24.4s) and the critic's deep
validation (36.9s) were 87s of a 97.8s measured run, each spending it inside a
single model call over independent items. Enrichment got this treatment in
round 14; these three never did.

What these tests guard is NOT the latency — a unit suite cannot see it. It is
the four ways a split silently corrupts a result that a single call could not:

  * a page or a provider lands in no shard, or in two
  * a shard's answer binds to a provider that shard never read
  * one failed shard costs the whole pool instead of its own items
  * a merged summary field claims more than any shard could see
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.critic_validator import (
    CriticValidatorAgent,
    _MIN_PROVIDERS_TO_SPLIT_VALIDATION,
    _merge_validation_shards,
)
from agents.data_gatherer import DataGathererAgent, _MIN_PAGES_TO_SHARD
from agents.preference_scorer import (
    PreferenceScorerAgent,
    _MIN_PROVIDERS_TO_SPLIT_JUDGE,
    _judge_token_budget,
)


@pytest.fixture
def data_gatherer():
    with patch.object(DataGathererAgent, "_initialize_clients", return_value=None):
        agent = DataGathererAgent()
        agent.tavily_client = MagicMock()
        agent.anthropic_client = MagicMock()
        return agent


@pytest.fixture
def preference_scorer():
    with patch.object(PreferenceScorerAgent, "_initialize_client", return_value=None):
        agent = PreferenceScorerAgent()
        agent.openai_client = MagicMock()
        return agent


@pytest.fixture
def critic_validator():
    with patch.object(CriticValidatorAgent, "_initialize_client", return_value=None):
        agent = CriticValidatorAgent()
        agent.anthropic_client = MagicMock()
        return agent


def _pages(count):
    return [
        {"title": f"Page {i}", "url": f"https://example{i}.com/p",
         "content": f"page {i} content", "raw_content": f"body {i}"}
        for i in range(count)
    ]


def _extraction_response(names):
    payload = [
        {"name": name, "specialty": "Neurology", "location": "Chandler, AZ",
         "rating": 4.5, "review_count": 20}
        for name in names
    ]
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content[0].text = json.dumps(payload)
    return response


def _judge_response(entries):
    response = MagicMock()
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.content = json.dumps(entries)
    return response


def _judge_entry(index, name, score=40):
    return {
        "provider_index": index,
        "provider_name": name,
        "scores": {"review_substance": score, "red_flags": 20, "practical_access": 10},
        "evidence": {"review_substance": f"cited for {name}"},
        "reasoning": f"reasoning for {name}",
        "strengths": [], "concerns": [],
    }


# ---------------------------------------------------------------------------
# Discovery extraction
# ---------------------------------------------------------------------------

class TestDiscoveryExtractionSplit:
    def test_a_full_page_list_is_read_by_two_concurrent_calls(self, data_gatherer):
        data_gatherer.anthropic_client.messages.create.return_value = _extraction_response(["Dr. A"])
        data_gatherer._extract_provider_data(_pages(12), "Neurology", "Chandler, AZ")
        assert data_gatherer.anthropic_client.messages.create.call_count == 2

    def test_the_page_floor_is_observed_on_both_sides(self, data_gatherer):
        """Below the floor an extra preamble buys about two pages of
        parallelism, so the split is refused rather than tuned.

        BOTH sides, because a one-sided assertion driven by the constant moves
        with it: lowering the floor to 1 makes `_MIN_PAGES_TO_SHARD - 1` zero
        pages, which trivially stays one call, and the test reported the
        deleted floor as guarded."""
        data_gatherer.anthropic_client.messages.create.return_value = _extraction_response(["Dr. A"])

        data_gatherer._extract_provider_data(
            _pages(_MIN_PAGES_TO_SHARD - 1), "Neurology", "Chandler, AZ"
        )
        assert data_gatherer.anthropic_client.messages.create.call_count == 1

        data_gatherer.anthropic_client.messages.create.reset_mock()
        data_gatherer._extract_provider_data(
            _pages(_MIN_PAGES_TO_SHARD), "Neurology", "Chandler, AZ"
        )
        assert data_gatherer.anthropic_client.messages.create.call_count == 2

    def test_every_page_reaches_exactly_one_prompt(self, data_gatherer):
        """A page in no shard loses the providers named on it — and the pool
        then reads as thin, which is the condition that fires the ring and
        imports cities nobody searched for. A page in TWO shards is paid for
        twice."""
        data_gatherer.anthropic_client.messages.create.return_value = _extraction_response(["Dr. A"])
        data_gatherer._extract_provider_data(_pages(12), "Neurology", "Chandler, AZ")

        prompts = [
            call.kwargs["messages"][0]["content"]
            for call in data_gatherer.anthropic_client.messages.create.call_args_list
        ]
        for i in range(12):
            hits = [p for p in prompts if f"https://example{i}.com/p" in p]
            assert len(hits) == 1, f"page {i} reached {len(hits)} prompts, expected 1"

    def test_providers_from_every_shard_are_merged(self, data_gatherer):
        """The merge is the whole point: half the providers arriving would look
        exactly like a thin web."""
        responses = [
            _extraction_response(["Dr. First", "Dr. Second"]),
            _extraction_response(["Dr. Third"]),
        ]
        data_gatherer.anthropic_client.messages.create.side_effect = responses

        providers = data_gatherer._extract_provider_data(_pages(12), "Neurology", "Chandler, AZ")
        assert sorted(p["name"] for p in providers) == ["Dr. First", "Dr. Second", "Dr. Third"]

    def test_one_failed_shard_does_not_cost_the_other(self, data_gatherer):
        """`future.result()` re-raises, so an exception escaping a shard would
        return [] for every page rather than for that shard's pages."""
        data_gatherer.anthropic_client.messages.create.side_effect = [
            RuntimeError("api exploded"),
            _extraction_response(["Dr. Survivor"]),
        ]
        providers = data_gatherer._extract_provider_data(_pages(12), "Neurology", "Chandler, AZ")
        assert [p["name"] for p in providers] == ["Dr. Survivor"]


# ---------------------------------------------------------------------------
# Rubric judge
# ---------------------------------------------------------------------------

class TestJudgeSplit:
    @staticmethod
    def _pool(count):
        return [
            {"name": f"Dr. Number {i}", "base_score": 90 - i, "review_summary": "Fine.",
             "rating": 4.0, "review_count": 10}
            for i in range(count)
        ]

    def test_a_full_pool_is_scored_by_two_concurrent_calls(self, preference_scorer):
        preference_scorer.openai_client.chat.completions.create.return_value = _judge_response([])
        preference_scorer._generate_ai_rankings(self._pool(8), {})
        assert preference_scorer.openai_client.chat.completions.create.call_count == 2

    def test_a_small_pool_stays_one_call(self, preference_scorer):
        """At 3 providers a shard holds one, and a rubric applied to a single
        provider has nothing in its own call to be consistent with."""
        preference_scorer.openai_client.chat.completions.create.return_value = _judge_response([])
        preference_scorer._generate_ai_rankings(
            self._pool(_MIN_PROVIDERS_TO_SPLIT_JUDGE - 1), {}
        )
        assert preference_scorer.openai_client.chat.completions.create.call_count == 1

    def test_the_floor_never_produces_a_single_provider_shard(self, preference_scorer):
        """The property the floor was chosen FOR, not the number itself.

        Asserting "below the floor, one call" using the constant is
        self-consistent at ANY value — lowering it to 2 keeps that assertion
        green while producing exactly the shard the floor exists to prevent: a
        provider alone in a call, with nothing in it to be consistent with. So
        assert the consequence. 4 is the smallest pool whose halves both hold
        at least two."""
        preference_scorer.openai_client.chat.completions.create.return_value = _judge_response([])
        preference_scorer._generate_ai_rankings(
            self._pool(_MIN_PROVIDERS_TO_SPLIT_JUDGE), {}
        )

        calls = preference_scorer.openai_client.chat.completions.create.call_args_list
        assert len(calls) == 2, "at the floor the pool must actually split"
        for call in calls:
            prompt = call.kwargs["messages"][1]["content"]
            block = prompt.split("<providers_data>\n")[1].split("\n</providers_data>")[0]
            assert len(json.loads(block)) >= 2, (
                "a shard holding one provider is what the floor exists to prevent"
            )

    def test_the_config_knob_reverts_to_one_call(self, preference_scorer):
        """D1's escape hatch: two calls CAN calibrate differently, and turning
        that off must not require a code change."""
        preference_scorer.config.JUDGE_PARALLEL_ENABLED = False
        preference_scorer.openai_client.chat.completions.create.return_value = _judge_response([])
        preference_scorer._generate_ai_rankings(self._pool(8), {})
        assert preference_scorer.openai_client.chat.completions.create.call_count == 1

    def test_every_provider_is_scored_exactly_once(self, preference_scorer):
        """The indices in each shard are GLOBAL, so the union must be the pool
        and no provider may appear twice."""
        preference_scorer.openai_client.chat.completions.create.return_value = _judge_response([])
        preference_scorer._generate_ai_rankings(self._pool(8), {})

        sent = []
        for call in preference_scorer.openai_client.chat.completions.create.call_args_list:
            prompt = call.kwargs["messages"][1]["content"]
            block = prompt.split("<providers_data>\n")[1].split("\n</providers_data>")[0]
            sent.extend(json.loads(block))

        assert sorted(e["provider_index"] for e in sent) == list(range(8))
        assert sorted(e["name"] for e in sent) == sorted(f"Dr. Number {i}" for i in range(8))

    def test_each_shard_scores_its_own_providers(self, preference_scorer):
        """The binding test. Shard indices are global, so shard B's entry for
        index 5 must land on provider 5 — not on B's own second element."""
        pool = self._pool(8)

        def respond(**kwargs):
            prompt = kwargs["messages"][1]["content"]
            block = prompt.split("<providers_data>\n")[1].split("\n</providers_data>")[0]
            shown = json.loads(block)
            return _judge_response([
                _judge_entry(rec["provider_index"], rec["name"], score=30 + rec["provider_index"])
                for rec in shown
            ])

        preference_scorer.openai_client.chat.completions.create.side_effect = respond
        ranked = preference_scorer._generate_ai_rankings(pool, {})

        for i, provider in enumerate(ranked):
            assert provider["ai_rubric"]["review_substance"] == 30 + i, (
                f"provider {i} carries another provider's rubric"
            )
            assert f"Dr. Number {i}" in provider["ai_evidence"]["review_substance"]

    def test_an_entry_for_another_shards_provider_is_dropped(self, preference_scorer, caplog):
        """In range for the pool, but never shown to this call — so the model
        invented it. Without the guard, shard B echoing index 0 overwrites the
        rubric shard A wrote for a provider B never read."""
        pool = self._pool(8)
        shard_indices = []

        def respond(**kwargs):
            prompt = kwargs["messages"][1]["content"]
            block = prompt.split("<providers_data>\n")[1].split("\n</providers_data>")[0]
            shown = json.loads(block)
            mine = sorted(rec["provider_index"] for rec in shown)
            shard_indices.append(mine)
            # Every call claims index 0, whether or not it was shown it.
            return _judge_response([_judge_entry(0, "Dr. Number 0", score=10 * len(shard_indices))])

        preference_scorer.openai_client.chat.completions.create.side_effect = respond
        with caplog.at_level("WARNING"):
            ranked = preference_scorer._generate_ai_rankings(pool, {})

        owner = 1 if 0 in shard_indices[0] else 2
        assert ranked[0]["ai_rubric"]["review_substance"] == 10 * owner, (
            "the shard that was never shown provider 0 overwrote its rubric"
        )
        assert "never shown" in caplog.text

    def test_no_shard_is_all_strong_or_all_weak(self, preference_scorer):
        """`providers` arrives in core-rank order, so CONTIGUOUS halves would
        give one call only strong providers and the other only weak ones. The
        rubric's bands are anchored prose, and prose calibrates against the
        examples in the call with it — a shard with no strong provider has
        nothing to calibrate the top of the scale against. That is the whole
        risk D1 accepted; dealing the pool is what bounds it."""
        preference_scorer.openai_client.chat.completions.create.return_value = _judge_response([])
        preference_scorer._generate_ai_rankings(self._pool(8), {})

        for call in preference_scorer.openai_client.chat.completions.create.call_args_list:
            prompt = call.kwargs["messages"][1]["content"]
            block = prompt.split("<providers_data>\n")[1].split("\n</providers_data>")[0]
            indices = [rec["provider_index"] for rec in json.loads(block)]
            # Pool index IS rank here (base_score descends), so a shard must
            # reach into both the top and the bottom half of the pool.
            assert min(indices) < 4 <= max(indices), (
                f"shard {sorted(indices)} covers only one end of the ranking"
            )

    def test_the_token_budget_is_scaled_to_the_shard(self, preference_scorer):
        """The ceiling bounds what ONE call returns. Sending the pool's budget
        to each of two calls asks for twice the headroom the response needs —
        and would pass just as well if the split stopped happening."""
        preference_scorer.openai_client.chat.completions.create.return_value = _judge_response([])
        preference_scorer._generate_ai_rankings(self._pool(8), {})

        budgets = [
            call.kwargs["max_completion_tokens"]
            for call in preference_scorer.openai_client.chat.completions.create.call_args_list
        ]
        assert budgets == [_judge_token_budget(4), _judge_token_budget(4)]

    def test_one_failed_shard_leaves_the_others_rubrics_intact(self, preference_scorer):
        """A failed judge call costs its own providers a rubric — they fall to
        the neutral 50 and are withheld from the shortlist for lack of one —
        and must not cost the pool."""
        pool = self._pool(8)
        calls = {"n": 0}

        def respond(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("judge exploded")
            prompt = kwargs["messages"][1]["content"]
            block = prompt.split("<providers_data>\n")[1].split("\n</providers_data>")[0]
            shown = json.loads(block)
            return _judge_response([
                _judge_entry(rec["provider_index"], rec["name"]) for rec in shown
            ])

        preference_scorer.openai_client.chat.completions.create.side_effect = respond
        ranked = preference_scorer._generate_ai_rankings(pool, {})

        scored = [p for p in ranked if p.get("ai_rubric")]
        unscored = [p for p in ranked if not p.get("ai_rubric")]
        assert len(scored) == 4 and len(unscored) == 4
        assert all(p["ai_score"] == 50.0 for p in unscored), (
            "providers of the failed shard must fall back to neutral, not punitive"
        )


# ---------------------------------------------------------------------------
# Critic deep validation
# ---------------------------------------------------------------------------

class TestValidationSplit:
    @staticmethod
    def _pool(count):
        return [
            {"name": f"Dr. Number {i}", "final_score": 90 - i, "rating": 4.0,
             "review_count": 10, "review_summary": "Fine.", "review_sentiment": "positive"}
            for i in range(count)
        ]

    @staticmethod
    def _response(entries, confidence="high"):
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content[0].text = json.dumps({
            "top_provider_validations": entries,
            "overall_ranking_validity": {
                "status": "validated", "confidence": confidence,
                "summary": "ok", "improvement_suggestions": [],
            },
        })
        return response

    def test_a_full_pool_is_validated_by_two_concurrent_calls(self, critic_validator):
        critic_validator.anthropic_client.messages.create.return_value = self._response([])
        critic_validator._validate_top_recommendations(self._pool(8))
        assert critic_validator.anthropic_client.messages.create.call_count == 2

    def test_a_small_pool_stays_one_call(self, critic_validator):
        critic_validator.anthropic_client.messages.create.return_value = self._response([])
        critic_validator._validate_top_recommendations(
            self._pool(_MIN_PROVIDERS_TO_SPLIT_VALIDATION - 1)
        )
        assert critic_validator.anthropic_client.messages.create.call_count == 1

    def test_the_floor_never_produces_a_single_provider_shard(self, critic_validator):
        """Same shape as the judge's floor, same reason it is asserted as a
        consequence rather than as a number: "find the real differences
        between these providers" has no meaning for a call holding one."""
        critic_validator.anthropic_client.messages.create.return_value = self._response([])
        critic_validator._validate_top_recommendations(
            self._pool(_MIN_PROVIDERS_TO_SPLIT_VALIDATION)
        )

        calls = critic_validator.anthropic_client.messages.create.call_args_list
        assert len(calls) == 2, "at the floor the pool must actually split"
        for call in calls:
            prompt = call.kwargs["messages"][0]["content"]
            sent = json.loads(
                prompt.split("TOP PROVIDERS TO VALIDATE:\n")[1].split("\n\nEach provider")[0]
            )
            assert len(sent) >= 2, (
                "a shard holding one provider is what the floor exists to prevent"
            )

    def test_verdicts_from_every_shard_survive_the_merge(self, critic_validator):
        critic_validator.anthropic_client.messages.create.side_effect = [
            self._response([{"provider_name": "Dr. Number 0", "rank": 1,
                             "validation_status": "approved"}]),
            self._response([{"provider_name": "Dr. Number 1", "rank": 2,
                             "validation_status": "conditional"}]),
        ]
        merged = critic_validator._validate_top_recommendations(self._pool(8))
        names = [e["provider_name"] for e in merged["top_provider_validations"]]
        assert sorted(names) == ["Dr. Number 0", "Dr. Number 1"]

    def test_no_shard_is_all_strong_or_all_weak(self, critic_validator):
        """The DIFFERENTIATION CHECK asks the critic to find the real
        differences among the providers in front of it. Contiguous halves give
        one call the top four and the other the bottom four — each internally
        similar, which is precisely the uniform-verdict failure that check
        exists to prevent."""
        critic_validator.anthropic_client.messages.create.return_value = self._response([])
        critic_validator._validate_top_recommendations(self._pool(8))

        for call in critic_validator.anthropic_client.messages.create.call_args_list:
            prompt = call.kwargs["messages"][0]["content"]
            sent = json.loads(
                prompt.split("TOP PROVIDERS TO VALIDATE:\n")[1].split("\n\nEach provider")[0]
            )
            ranks = [e["rank"] for e in sent]
            assert min(ranks) <= 4 < max(ranks), (
                f"shard with ranks {sorted(ranks)} covers only one end of the ranking"
            )

    def test_the_prompt_warns_that_ranks_are_not_consecutive(self, critic_validator):
        """Dealt shards carry ranks 1,3,5,7 — a model told to list providers
        "in ranking order" and handed gaps may helpfully renumber them, and
        `_generate_final_recommendations` looks the top three up BY RANK."""
        critic_validator.anthropic_client.messages.create.return_value = self._response([])
        critic_validator._validate_top_recommendations(self._pool(8))
        for call in critic_validator.anthropic_client.messages.create.call_args_list:
            prompt = call.kwargs["messages"][0]["content"]
            assert "may not be consecutive" in prompt
            assert "Never renumber them." in prompt

    def test_notes_are_capped_at_two_sentences(self, critic_validator):
        """Phase 2 trimmed 3 -> 2. The notes render as one card line, where the
        third sentence was routinely a restatement of the first."""
        critic_validator.anthropic_client.messages.create.return_value = self._response([])
        critic_validator._validate_top_recommendations(self._pool(8))
        prompt = critic_validator.anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "at most 2 sentences" in prompt
        assert "at most 3 sentences" not in prompt


class TestMergingValidationShards:
    """`overall_ranking_validity` describes a ranking NO shard saw whole."""

    @staticmethod
    def _shard(entries, confidence, status="validated", suggestions=None):
        return {
            "top_provider_validations": entries,
            "overall_ranking_validity": {
                "status": status, "confidence": confidence,
                "summary": f"{confidence} summary",
                "improvement_suggestions": suggestions or [],
            },
        }

    def test_confidence_falls_to_the_least_confident_shard(self):
        """It reaches the patient — a "low" adds a caution line to
        user_guidance. A ranking is only as validated as its weakest half, and
        averaging would let a confident half mask one the critic could not
        vouch for."""
        merged = _merge_validation_shards([
            self._shard([{"rank": 1}], "high"),
            self._shard([{"rank": 2}], "low"),
        ])
        assert merged["overall_ranking_validity"]["confidence"] == "low"

    def test_a_shard_that_returned_nothing_does_not_drag_confidence_down(self):
        """Its fallback is a hardcoded "low" that represents no judgment about
        the ranking — its providers are separately marked not_critiqued. Letting
        it vote would turn one failed call into a caution line about providers
        the critic actually approved."""
        empty = self._shard([], "low", status="error")
        merged = _merge_validation_shards([self._shard([{"rank": 1}], "high"), empty])
        assert merged["overall_ranking_validity"]["confidence"] == "high"
        assert merged["overall_ranking_validity"]["status"] == "validated"

    def test_total_failure_reports_the_error_shape(self):
        merged = _merge_validation_shards([
            self._shard([], "low", status="error"),
            self._shard([], "low", status="error"),
        ])
        assert merged["top_provider_validations"] == []
        assert merged["overall_ranking_validity"]["status"] == "error"

    def test_suggestions_union_without_duplicates(self):
        merged = _merge_validation_shards([
            self._shard([{"rank": 1}], "high", suggestions=["widen the search", "verify hours"]),
            self._shard([{"rank": 2}], "high", suggestions=["verify hours", "check parking"]),
        ])
        assert merged["overall_ranking_validity"]["improvement_suggestions"] == [
            "widen the search", "verify hours", "check parking",
        ]

    def test_malformed_shards_are_skipped_not_fatal(self):
        merged = _merge_validation_shards([None, self._shard([{"rank": 1}], "medium"), {}])
        assert len(merged["top_provider_validations"]) == 1
        assert merged["overall_ranking_validity"]["confidence"] == "medium"
