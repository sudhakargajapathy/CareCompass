#!/usr/bin/env python
"""Account verification: prove the Langfuse project works before wiring it.

Run from the repository root once the keys are in `.env` (or exported):

    venv/bin/python evals/p0_verify.py

What it proves: the key pair authenticates against the configured host, a
hello-world trace (root span, child span, one score) is flushed, and the
trace's URL is printed for the owner to open.

Exit codes: 0 verified · 2 configuration missing (the report names every
absent variable and where it comes from; nothing is contacted) · 1 the check
failed (the error text is printed; no secret is ever echoed).

Decisions carried inline so they are not re-derived:

  * `LANGFUSE_BASE_URL` is REQUIRED, not defaulted. The SDK sends an unset
    host to its EU cloud; the project lives on the US region, and a verify
    that "passed" against the wrong region would leave every later trace
    invisible on the dashboard the owner actually opens.
  * The run_id SEEDS the trace id (`Langfuse.create_trace_id(seed=run_id)`),
    the joining rule the instrumented pipeline will use: one id across the
    trace, the exported run record and the audit log.
  * The client import is LAZY, so the missing-configuration path is provably
    offline (pinned by test): a report about absent keys must not itself
    need a network.
  * Langfuse is the ONLY store. A second half of this script once verified a
    Supabase `run_metrics` table; it was retired the same day it was
    written (2026-09-12) — `utils/run_record` carries the reasoning.
"""

import argparse
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402  (already a project dependency)

from utils.run_record import SCHEMA_VERSION  # noqa: E402

US_LANGFUSE_HOST = "https://us.cloud.langfuse.com"

# Variable → where the owner finds its value. Printed verbatim by the
# missing-configuration report, so it must stay a complete instruction.
REQUIRED_ENV: Dict[str, str] = {
    "LANGFUSE_PUBLIC_KEY": "Langfuse → project → Settings → API Keys (pk-lf-…)",
    "LANGFUSE_SECRET_KEY": "same screen, shown ONCE when the key pair is created (sk-lf-…)",
    "LANGFUSE_BASE_URL": f"the project's region host — {US_LANGFUSE_HOST} for a US project",
}


def missing_env(env: Mapping[str, str]) -> List[str]:
    """Names of the required variables that are unset or blank, in report order."""
    return [name for name in REQUIRED_ENV if not (env.get(name) or "").strip()]


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(REPO_ROOT),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    sha = os.environ.get("GITHUB_SHA", "")
    return sha[:12] or None


def _redact(text: str, env: Mapping[str, str]) -> str:
    """Scrub every configured key from an error line."""
    for name in ("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"):
        secret = env.get(name) or ""
        if len(secret) >= 4:
            text = text.replace(secret, "***")
    return text


def verify_langfuse(env: Mapping[str, str], run_id: str, git_sha: Optional[str]) -> Dict[str, Optional[str]]:
    """Authenticate, emit a hello-world trace seeded by run_id, flush, return its URL."""
    from langfuse import Langfuse  # lazy: the missing-config path stays offline

    client = Langfuse(
        public_key=env["LANGFUSE_PUBLIC_KEY"],
        secret_key=env["LANGFUSE_SECRET_KEY"],
        base_url=env["LANGFUSE_BASE_URL"].strip(),  # a pasted trailing space must not become part of the host
        environment="verify",
        release=git_sha or None,
    )
    try:
        # The SDK RAISES on a 401 (its UnauthorizedError carries a header dump)
        # and returns False on other failures; both become one short line.
        try:
            authenticated = client.auth_check()
        except Exception as exc:
            detail = getattr(exc, "body", None) or getattr(exc, "status_code", None) or exc
            raise RuntimeError(
                f"Langfuse rejected the keys ({detail}) — check the pk/sk pair belongs "
                "to one project and LANGFUSE_BASE_URL is that project's region host"
            ) from exc
        if not authenticated:
            raise RuntimeError(
                "Langfuse rejected the keys (auth_check returned False) — check the "
                "pk/sk pair belongs to one project and LANGFUSE_BASE_URL is that "
                "project's region host"
            )
        trace_id = Langfuse.create_trace_id(seed=run_id)
        root = client.start_observation(
            trace_context={"trace_id": trace_id},
            name="carecompass.p0_verify",
            as_type="span",
            input={"run_id": run_id, "phase": "P0"},
            metadata={"source": "verify", "git_sha": git_sha, "schema_version": SCHEMA_VERSION},
        )
        child = root.start_observation(
            name="hello_world", as_type="span", output={"ok": True}
        )
        child.end()
        root.update(output={"ok": True})
        root.end()
        client.create_score(
            name="p0_verify_ok",
            value=1,
            trace_id=trace_id,
            data_type="BOOLEAN",
            comment="account verification: hello-world trace",
        )
        client.flush()
        return {"trace_id": trace_id, "trace_url": client.get_trace_url(trace_id=trace_id)}
    finally:
        client.shutdown()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the Langfuse project keys with a hello-world trace.")
    parser.parse_args(argv)

    load_dotenv()  # .env from the cwd upward; exported variables always win
    env = os.environ
    missing = missing_env(env)
    if missing:
        print("P0 verify: configuration missing — nothing was contacted.")
        for name in missing:
            print(f"  {name:<20} <- {REQUIRED_ENV[name]}")
        print("Set them in .env (see .env.example) or export them, then re-run.")
        return 2

    run_id = uuid.uuid4().hex
    git_sha = _git_sha()
    print(f"P0 verify: run_id {run_id} · git {git_sha or 'unknown'}")
    try:
        result = verify_langfuse(env, run_id, git_sha)
    except Exception as exc:  # one line, never a traceback, never a secret
        print(f"  FAIL Langfuse  {type(exc).__name__}: {_redact(str(exc), env)}")
        print("P0 verify: the check failed.")
        return 1
    print(f"  OK   Langfuse  host {env['LANGFUSE_BASE_URL']} · trace {result['trace_url'] or result['trace_id']}")
    print("P0 verify: all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
