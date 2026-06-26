"""Inject basic security headers via meta tags for Streamlit."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def inject_security_headers() -> None:
    """Inject CSP and referrer policy meta tags.

    Note: For production, enforce headers at the reverse proxy/load balancer.
    """
    csp = (
        "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; "
        "img-src 'self' data: blob: https:; "
        "style-src 'self' 'unsafe-inline' https:; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; "
        "connect-src 'self' https: wss:; "
        "frame-ancestors 'none';"
    )

    components.html(
        f"""
        <meta http-equiv="Content-Security-Policy" content="{csp}">
        <meta http-equiv="Referrer-Policy" content="strict-origin-when-cross-origin">
        """,
        height=0,
        width=0,
    )

    # Streamlit requires a layout call after injection for consistent rendering
    st.markdown("", unsafe_allow_html=True)
