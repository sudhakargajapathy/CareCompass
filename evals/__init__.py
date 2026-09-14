"""Evaluation and monitoring harness.

Everything under `evals/` is PUBLIC and carries facts, not data: schemas,
verification and probe scripts, minimized structural fixtures, hashes. Full
fetched page bodies and raw API responses never live here — the house rule
that predates this directory ("never commit provider data or raw API
responses") applies to it in full.
"""
