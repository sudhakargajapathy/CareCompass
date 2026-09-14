"""Coarse, privacy-safe facts about the visitor who started a search.

What is captured, and why only this:

  country / region   from the headers the hosting proxy adds to the request
                     (Cloudflare's `cf-ipcountry`, Vercel's, CloudFront's);
                     absent when the host adds none. Coarse enough to
                     publish: the weekly report's rows are public.
  timezone / locale  from the browser, via Streamlit's request context.
  visitor            a salted SHA-256 of the client IP, first 16 hex — a
                     stable pseudonym so "distinct visitors" can be counted
                     on the trace store. Computed ONLY when VISITOR_HASH_SALT
                     is set (no salt, no id), and it rides on the trace's
                     metadata alone: the run record, and therefore the
                     public rows, never carry it.

The raw IP address is never stored anywhere — not on the trace, not in
the record, not in a log. This is a healthcare-adjacent app whose run
records ship to a public repository; an address that finds a person has
no place in it, and nothing the monitoring needs requires one.

Everything is wrapped so that a missing context (a worker thread, a unit
test, a future Streamlit) yields an empty dict rather than an exception:
capturing a fact about the visitor must never cost them the search.
"""

import hashlib
import os
from typing import Any, Dict, Mapping, Optional

COUNTRY_HEADERS = ("cf-ipcountry", "x-vercel-ip-country", "cloudfront-viewer-country", "x-country-code")
REGION_HEADERS = ("cf-region-code", "x-vercel-ip-country-region", "cloudfront-viewer-country-region", "cf-region")
SALT_ENV = "VISITOR_HASH_SALT"
_MAX_CHARS = 40


def _clean(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text[:_MAX_CHARS] if text else None


def _header(headers: Any, names: tuple) -> Optional[str]:
    if headers is None:
        return None
    lowered: Dict[str, Any] = {}
    try:
        lowered = {str(k).lower(): v for k, v in dict(headers).items()}
    except Exception:
        for name in names:
            try:
                value = headers.get(name)
            except Exception:
                value = None
            if value:
                return _clean(value)
        return None
    for name in names:
        if lowered.get(name):
            return _clean(lowered[name])
    return None


def visitor_id(ip: Optional[str], salt: Optional[str]) -> Optional[str]:
    """A salted pseudonym for an IP, or None without a salt or an address."""
    if not ip or not salt:
        return None
    return hashlib.sha256(f"{salt}:{ip}".encode("utf-8")).hexdigest()[:16]


def capture_client_context(context: Any = None, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The coarse visitor facts for this request; {} when nothing is known."""
    env = os.environ if env is None else env
    try:
        if context is None:
            import streamlit as st

            context = st.context
        out: Dict[str, Any] = {}
        headers = getattr(context, "headers", None)
        country = _header(headers, COUNTRY_HEADERS)
        region = _header(headers, REGION_HEADERS)
        if country:
            out["country"] = country.upper()
        if region:
            out["region"] = region
        timezone = _clean(getattr(context, "timezone", None))
        locale = _clean(getattr(context, "locale", None))
        if timezone:
            out["timezone"] = timezone
        if locale:
            out["locale"] = locale
        visitor = visitor_id(_clean(getattr(context, "ip_address", None)), env.get(SALT_ENV))
        if visitor:
            out["visitor"] = visitor
        return out
    except Exception:
        return {}


# The keys that may leave the trace: coarse facts only. `visitor` is
# deliberately not here — the record and the public rows never see it.
PUBLIC_KEYS = ("country", "region", "timezone", "locale")


def public_subset(client: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(client, Mapping):
        return None
    out = {k: client[k] for k in PUBLIC_KEYS if client.get(k)}
    return out or None
