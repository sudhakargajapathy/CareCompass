"""Standalone network-verification prototype against a payer FHIR directory.

This module is deliberately decoupled from candidate gathering and from
scoring: it answers one question — "does the payer's directory have a record
of this provider in-network?" — and the UI presents the answer as labeled
evidence. It never moves rankings.

Against the sandbox directory (FHIR_USE_MOCK=true, the default) a "no_record"
answer only means the provider isn't in the demo data, so callers must treat
it as *unverified*, never as a penalty. Point FHIR_USE_MOCK=false plus the
FHIR_* credentials at a real Plan-Net endpoint and the same call verifies for
real — that seam is the point of the prototype.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Share of a pool marked in-network by the simulated directory. A quota over
# the RANKED pool, deliberately not an independent per-provider coin flip: at
# 70% per head, a 5-provider pool comes up all-green ~17% of demos — and an
# all-green (or all-grey) network check reads as a feature that does nothing,
# which is exactly how the sandbox demoed before this existed (2026-07-31
# screenshot: 5/5 "no record" against every payer, a dead end).
_SIMULATED_IN_NETWORK_SHARE = 0.7


def _simulation_rank(provider_name: str, payer: str) -> str:
    """Deterministic per-(provider, payer) ordering key for the quota.

    Keyed on the NORMALIZED NAME rather than the full provider cache key
    (name + city): enrichment sharpens a provider's address between runs, and
    a location-bearing key would re-deal that provider's coverage whenever it
    did — a doctor flipping in and out of network across two identical demos
    reads as flakiness, not simulation. Names are stable across runs; the
    payer in the hash is what re-deals coverage on a payer switch.
    """
    normalized = " ".join(sorted(_name_tokens(provider_name))) or str(provider_name).lower()
    return hashlib.sha256(f"{normalized}|{payer}".encode("utf-8")).hexdigest()


def simulate_network_batch(provider_names: List[str], payer: str) -> Dict[str, Dict[str, Any]]:
    """Deterministic SIMULATED coverage for a pool of providers under one payer.

    The demo-mode replacement for querying the sandbox directory: the sandbox
    ships static demo doctors while the live search finds real ones, so every
    check answered "no record" and the FHIR feature demoed as a dead end.
    Membership here is a pure function of (names, payer) — stored nowhere,
    same answer every run and every test, nothing to go stale, no provider
    data committed — and it sidesteps the name matcher entirely: results are
    keyed by the exact name asked about, so the matcher's known
    shared-surname false positive cannot occur in demo mode. Real mode
    (FHIR_USE_MOCK=false) never reaches this function.

    Every result is labeled ``"source": "simulated"`` and the UI must render
    that label on every chip and in the panel caption — a bare "in-network"
    against a real physician's name from invented data would be fabricated
    coverage.

    The quota guarantees a MIXED result: rank the pool by
    ``sha256(name|payer)`` and mark the top ~70% in-network, clamped so a
    pool of two or more always shows at least one of each state. A pool of
    one cannot be mixed; it falls to a deterministic per-name 70% draw.
    """
    names = [str(n) for n in provider_names if str(n or "").strip()]
    results: Dict[str, Dict[str, Any]] = {}
    if not names:
        return results

    def _result(name: str, in_network: bool) -> Dict[str, Any]:
        return {
            "status": "verified" if in_network else "no_record",
            "source": "simulated",
            "payer": payer,
            "matched_name": name if in_network else None,
        }

    if len(names) == 1:
        name = names[0]
        draw = int(_simulation_rank(name, payer), 16) % 10 < 7
        results[name] = _result(name, draw)
        return results

    ranked = sorted(names, key=lambda name: _simulation_rank(name, payer))
    # ~70% in-network, with the payer also choosing floor vs ceil — the count
    # moving (3 vs 4 of 5) is a far more visible re-deal on a payer switch
    # than one name swapping, and with only C(5,1) possible single-out deals
    # a 7-payer list provably repeats them (pigeonhole; measured: the demo
    # pool dealt An-out under four of the seven payers). Clamped to [1, n-1]
    # so a pool of two or more is always mixed.
    share = _SIMULATED_IN_NETWORK_SHARE * len(ranked)
    ceil_bit = int(hashlib.sha256(f"quota|{payer}".encode("utf-8")).hexdigest(), 16) % 2
    quota = int(share) + (1 if ceil_bit and share != int(share) else 0)
    quota = max(1, min(len(ranked) - 1, quota))
    in_network = set(ranked[:quota])
    for name in names:
        results[name] = _result(name, name in in_network)
    return results

# Credentials/titles that carry no identity signal for name matching
_NAME_NOISE = {"dr", "md", "do", "phd", "np", "pa", "dds", "mbbs", "jr", "sr"}


def _name_tokens(name: Any) -> set:
    return {
        token
        for token in re.findall(r"[a-z]+", str(name or "").lower())
        if len(token) > 1 and token not in _NAME_NOISE
    }


def verify_network(
    provider_name: str,
    payer: str,
    specialty: Optional[str] = None,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    """Look a provider up in the payer's FHIR directory.

    Returns:
        {"status": "verified" | "no_record" | "unavailable",
         "source": "sandbox" | "live" | None,
         "matched_name": str | None}

        verified    — a directory entry for this payer matches the name
        no_record   — directory reachable, but no matching entry (from the
                      sandbox this means "not in demo data": treat as
                      unverified, never as evidence of non-acceptance)
        unavailable — directory not configured or errored
    """
    if not provider_name or not payer:
        return {"status": "unavailable", "source": None, "matched_name": None}

    try:
        from fhir.client import create_fhir_client
        from fhir.transformer import FHIRToProviderTransformer
        from utils.config import get_config

        config = get_config()
        client = create_fhir_client()
        if not client.is_available():
            return {"status": "unavailable", "source": None, "matched_name": None}
        source = "sandbox" if config.FHIR_USE_MOCK else "live"

        bundle = client.search_practitioners(
            specialty=specialty or "",
            location=location or "",
            insurance_network=payer,
            count=50,
        )
        directory_entries = FHIRToProviderTransformer().transform_bundle(bundle)

        target = _name_tokens(provider_name)
        if not target:
            return {"status": "unavailable", "source": source, "matched_name": None}

        for entry in directory_entries:
            candidate = _name_tokens(entry.get("name"))
            if not candidate:
                continue
            overlap = len(target & candidate) / max(min(len(target), len(candidate)), 1)
            if overlap >= 0.5:
                return {
                    "status": "verified",
                    "source": source,
                    "matched_name": entry.get("name"),
                }

        return {"status": "no_record", "source": source, "matched_name": None}

    except Exception as exc:
        logger.warning("Network verification unavailable: %s", exc)
        return {"status": "unavailable", "source": None, "matched_name": None}
