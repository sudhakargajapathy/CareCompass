"""Authentication module for CareCompass application."""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import bcrypt
import streamlit as st

from .audit_log import log_audit_event
from .config import get_config

logger = logging.getLogger(__name__)


class DatabaseAuthenticator:
    """Database-backed authentication with roles and rate limits."""

    def __init__(self) -> None:
        config = get_config()
        self.db_url = config.AUTH_DATABASE_URL
        self.max_failed_attempts = config.AUTH_MAX_FAILED_ATTEMPTS
        self.lockout_seconds = config.AUTH_LOCKOUT_SECONDS
        self.is_configured = bool(self.db_url)

        if "login_attempts" not in st.session_state:
            st.session_state.login_attempts = {}

        if self.is_configured:
            try:
                self._ensure_schema()
                if config.AUTH_BOOTSTRAP_ADMIN:
                    self._ensure_admin_user()
            except Exception as exc:
                logger.error("Auth database initialization failed: %s", exc)
                self.is_configured = False

    def _get_connection(self):
        import psycopg2
        return psycopg2.connect(self.db_url)

    def _ensure_schema(self) -> None:
        query = """
        CREATE TABLE IF NOT EXISTS app_users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_login TIMESTAMPTZ
        );
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
            conn.commit()

    def _ensure_admin_user(self) -> None:
        config = get_config()
        admin_username = config.APP_ADMIN_USERNAME
        admin_password = config.APP_ADMIN_PASSWORD

        if not admin_password:
            logger.warning("APP_ADMIN_PASSWORD not set; admin bootstrap skipped")
            return

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM app_users WHERE username = %s", (admin_username,))
                exists = cur.fetchone()
                if not exists:
                    password_hash = self._hash_password(admin_password)
                    cur.execute(
                        """
                        INSERT INTO app_users (username, password_hash, role)
                        VALUES (%s, %s, %s)
                        """,
                        (admin_username, password_hash, "admin"),
                    )
            conn.commit()

    @staticmethod
    def _hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def _verify_password(password: str, password_hash: str) -> bool:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

    def _check_rate_limit(self, username: str) -> Tuple[bool, int]:
        if username not in st.session_state.login_attempts:
            return True, 0

        attempts_data = st.session_state.login_attempts[username]
        failed_attempts = attempts_data.get("count", 0)
        last_attempt_time = attempts_data.get("last_attempt", 0)

        if failed_attempts >= self.max_failed_attempts:
            wait_time = self.lockout_seconds - (time.time() - last_attempt_time)
            if wait_time > 0:
                return False, int(wait_time)
            st.session_state.login_attempts[username] = {"count": 0, "last_attempt": 0}

        return True, 0

    def _record_failed_attempt(self, username: str) -> None:
        if username not in st.session_state.login_attempts:
            st.session_state.login_attempts[username] = {"count": 0, "last_attempt": 0}

        st.session_state.login_attempts[username]["count"] += 1
        st.session_state.login_attempts[username]["last_attempt"] = time.time()

    def _reset_failed_attempts(self, username: str) -> None:
        if username in st.session_state.login_attempts:
            st.session_state.login_attempts[username] = {"count": 0, "last_attempt": 0}

    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        is_allowed, wait_time = self._check_rate_limit(username)
        if not is_allowed:
            log_audit_event("login_rate_limited", user=username, success=False, details={"wait": wait_time})
            return False, None

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT password_hash, role, is_active
                        FROM app_users
                        WHERE username = %s
                        """,
                        (username,),
                    )
                    row = cur.fetchone()
        except Exception as exc:
            logger.error("Authentication query failed: %s", exc)
            log_audit_event("login_error", user=username, success=False, details={"error": str(exc)})
            return False, None

        if not row:
            self._record_failed_attempt(username)
            log_audit_event("login_failed", user=username, success=False)
            return False, None

        password_hash, role, is_active = row
        if not is_active:
            log_audit_event("login_blocked_inactive", user=username, success=False)
            return False, None

        if self._verify_password(password, password_hash):
            self._reset_failed_attempts(username)
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE app_users SET last_login = NOW() WHERE username = %s",
                        (username,),
                    )
                conn.commit()
            log_audit_event("login_success", user=username)
            return True, role

        self._record_failed_attempt(username)
        log_audit_event("login_failed", user=username, success=False)
        return False, None

    def login_form(self) -> Optional[str]:
        st.markdown("## 🔐 Login to CareCompass")
        st.markdown("Please login to access the healthcare provider matching system.")

        if st.session_state.get("authenticated", False):
            return st.session_state.get("username")

        if not self.is_configured:
            if self._is_dev_mode():
                st.info("Dev mode: authentication bypassed. Auto-logged in as dev_admin.")
                return "dev_admin"
            st.error("Authentication database is not configured.")
            st.code("Set AUTH_DATABASE_URL or DATABASE_URL to a Postgres connection string.")
            return None

        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password")
                    return None

                is_allowed, wait_time = self._check_rate_limit(username)
                if not is_allowed:
                    st.error(f"⚠️ Too many failed login attempts. Please try again in {wait_time} seconds.")
                    return None

                authenticated, role = self.authenticate(username, password)
                if authenticated:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_role = role
                    st.session_state.login_time = time.time()
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    attempts_left = self.max_failed_attempts - st.session_state.login_attempts.get(username, {}).get(
                        "count", 0
                    )
                    if attempts_left > 0:
                        st.error(f"❌ Invalid credentials. {attempts_left} attempts remaining.")
                    else:
                        st.error("⚠️ Account locked. Too many failed attempts.")

        return None

    def logout(self) -> None:
        if st.session_state.get("authenticated", False):
            user = st.session_state.get("username")
            log_audit_event("logout", user=user)
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_role = None
            st.session_state.login_time = None
            st.rerun()

    def _is_dev_mode(self) -> bool:
        """Check if running in development mode without a database."""
        config = get_config()
        return not self.is_configured and not config.is_production()

    def require_authentication(self) -> bool:
        if st.session_state.get("authenticated", False):
            login_time = st.session_state.get("login_time", 0)
            if time.time() - login_time > 1800:
                user = st.session_state.get("username")
                log_audit_event("session_timeout", user=user, success=False)
                self.logout()
                return False
            return True

        # Dev mode: auto-authenticate when no database is configured
        if self._is_dev_mode():
            st.session_state.authenticated = True
            st.session_state.username = "dev_admin"
            st.session_state.user_role = "admin"
            st.session_state.login_time = time.time()
            logger.info("Dev mode: auto-authenticated as dev_admin")
            return True

        return False

    @staticmethod
    def is_admin() -> bool:
        return st.session_state.get("user_role") == "admin"


def get_authenticator() -> DatabaseAuthenticator:
    if "authenticator" not in st.session_state:
        st.session_state.authenticator = DatabaseAuthenticator()

    return st.session_state.authenticator
