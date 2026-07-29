"""Partition a work list into shards for parallel LLM calls.

Phase 2 splits three stages that were each ONE model call over N independent
items (discovery extraction, the rubric judge, the critic's deep validation).
Enrichment already had this treatment in round 14; these three never did, and
together they were ~87s of a 97.8s run.

TWO MODES, AND THEY ARE NOT INTERCHANGEABLE. Which one is correct depends on
what the list's ORDER already means, and picking by habit gets it backwards —
the wrap-up plan specified round-robin for discovery on the reasoning that it
"preserves round 12's interleave doctrine", which is the exact opposite of
what it does there:

  round_robin  when the order carries a GRADIENT each shard must span.
               The judge and the critic receive providers in core-rank order,
               so contiguous halves hand one call the strong providers and the
               other the weak ones. The rubric is anchored, so scores are meant
               to be absolute — but a shard with no strong example in it has
               nothing to calibrate the top of the scale against, which is
               exactly the divergence splitting the judge risks.

  contiguous   when the order is ALREADY an interleave that alternation
               destroys. Discovery's page list is built by zip_longest over
               (review-platform, everything-else), so it alternates with
               period 2 — a round-robin split on that list separates the two
               kinds PERFECTLY, handing one call every profile page and the
               other every directory page. Round 12's whole finding was that
               profile pages naming one physician must not evict the
               many-name directory pages that fill the candidate pool.
               Contiguous halves are each themselves interleaved.

Both drop empty shards, so a short list simply yields fewer shards rather
than empty model calls.
"""

from typing import Any, List, Sequence


def round_robin_shards(items: Sequence[Any], shards: int) -> List[List[Any]]:
    """Deal `items` into `shards` lists like cards, preserving relative order.

    Use when the input order is a ranking and each shard must span its range.
    """
    if shards <= 1 or len(items) <= 1:
        return [list(items)] if items else []

    dealt: List[List[Any]] = [[] for _ in range(shards)]
    for position, item in enumerate(items):
        dealt[position % shards].append(item)
    return [shard for shard in dealt if shard]


def contiguous_shards(items: Sequence[Any], shards: int) -> List[List[Any]]:
    """Cut `items` into `shards` consecutive blocks, largest first.

    Use when the input order is already an interleave that dealing would
    segregate. Remainder items go to the earliest shards, so sizes differ by
    at most one.
    """
    if shards <= 1 or len(items) <= 1:
        return [list(items)] if items else []

    size, remainder = divmod(len(items), shards)
    cut: List[List[Any]] = []
    start = 0
    for position in range(shards):
        stop = start + size + (1 if position < remainder else 0)
        if start < stop:
            cut.append(list(items[start:stop]))
        start = stop
    return cut
