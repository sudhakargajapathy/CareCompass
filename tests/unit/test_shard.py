"""The Phase 2 partition helper.

The load-bearing test here is `TestPickingTheWrongMode` — the wrap-up plan
specified round-robin for discovery extraction on the reasoning that it
"preserves round 12's interleave doctrine", and it does the exact opposite.
These tests exist so the next person to reach for the "obvious" mode sees
what it costs before they ship it.
"""

from itertools import zip_longest

from utils.shard import contiguous_shards, round_robin_shards


class TestRoundRobin:
    def test_deals_like_cards(self):
        assert round_robin_shards([1, 2, 3, 4, 5, 6], 2) == [[1, 3, 5], [2, 4, 6]]

    def test_relative_order_survives_within_a_shard(self):
        """Each shard must stay best-first: the judge and critic both receive
        core-rank order, and a consumer reading a shard as a ranking would be
        reading a shuffled one otherwise."""
        shards = round_robin_shards(list(range(10)), 3)
        for shard in shards:
            assert shard == sorted(shard)

    def test_uneven_pools_differ_by_at_most_one(self):
        shards = round_robin_shards(list(range(7)), 2)
        assert [len(s) for s in shards] == [4, 3]

    def test_empty_shards_are_dropped(self):
        """Three shards over two items must not produce an empty third — an
        empty shard downstream is a model call with no work in it."""
        assert round_robin_shards([1, 2], 3) == [[1], [2]]

    def test_degenerate_inputs(self):
        assert round_robin_shards([], 2) == []
        assert round_robin_shards([1], 2) == [[1]]
        assert round_robin_shards([1, 2, 3], 1) == [[1, 2, 3]]


class TestContiguous:
    def test_cuts_consecutive_blocks(self):
        assert contiguous_shards([1, 2, 3, 4, 5, 6], 2) == [[1, 2, 3], [4, 5, 6]]

    def test_remainder_goes_to_the_earliest_shards(self):
        assert contiguous_shards(list(range(7)), 2) == [[0, 1, 2, 3], [4, 5, 6]]

    def test_empty_shards_are_dropped(self):
        assert contiguous_shards([1, 2], 3) == [[1], [2]]

    def test_degenerate_inputs(self):
        assert contiguous_shards([], 2) == []
        assert contiguous_shards([1], 2) == [[1]]
        assert contiguous_shards([1, 2, 3], 1) == [[1, 2, 3]]


class TestNothingIsLost:
    """Whatever the mode, every item must land in exactly one shard. A split
    that drops a page drops the providers named on it, and the candidate pool
    reads as thin for a reason no log would explain."""

    def test_round_robin_is_a_partition(self):
        items = list(range(18))
        flat = [item for shard in round_robin_shards(items, 2) for item in shard]
        assert sorted(flat) == items

    def test_contiguous_is_a_partition(self):
        items = list(range(18))
        flat = [item for shard in contiguous_shards(items, 2) for item in shard]
        assert sorted(flat) == items


class TestPickingTheWrongMode:
    """Discovery's page list is built by zip_longest over
    (review-platform, everything-else), so it ALREADY alternates with period
    2. That is the shape on which the two modes behave oppositely."""

    @staticmethod
    def _interleaved_pages():
        profiles = [f"profile-{i}" for i in range(6)]
        directories = [f"directory-{i}" for i in range(6)]
        pages = []
        for pair in zip_longest(profiles, directories):
            pages.extend(p for p in pair if p is not None)
        return pages

    def test_round_robin_segregates_the_interleave(self):
        """The failure the plan would have shipped: one call gets every
        profile page (one physician each) and the other every directory page.
        Round 12's whole finding was that profile pages must not evict the
        many-name pages that fill the candidate pool — this hands one call a
        list made entirely of them."""
        first, second = round_robin_shards(self._interleaved_pages(), 2)
        assert all(page.startswith("profile") for page in first)
        assert all(page.startswith("directory") for page in second)

    def test_contiguous_preserves_the_interleave(self):
        """Each half is itself alternating, so both calls read a mix of
        one-name profile pages and many-name directory pages."""
        for shard in contiguous_shards(self._interleaved_pages(), 2):
            kinds = {page.split("-")[0] for page in shard}
            assert kinds == {"profile", "directory"}, (
                f"shard lost a page kind entirely: {shard}"
            )

    def test_contiguous_would_segregate_a_ranking(self):
        """The mirror image, and why the judge and critic use the other mode:
        on a SORTED list contiguous halves split strong from weak, leaving one
        call with no strong provider to calibrate the top of the rubric
        against."""
        ranking = list(range(100, 0, -10))  # best-first scores
        strong, weak = contiguous_shards(ranking, 2)
        assert min(strong) > max(weak)

        # Dealing the same list gives both halves the full range.
        dealt = round_robin_shards(ranking, 2)
        for shard in dealt:
            assert max(shard) >= 90 and min(shard) <= 20
