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

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

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
