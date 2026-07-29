"""Unit tests for single-key Fernet encryption."""

import pytest
from cryptography.fernet import Fernet

from utils.encryption import DataEncryption, generate_encryption_key


@pytest.mark.unit
class TestDataEncryption:
    def test_round_trip_with_env_key(self, monkeypatch):
        monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
        encryptor = DataEncryption()

        payload = {"name": "Dr. Sarah Johnson", "rating": 4.8, "insurance": ["Aetna"]}
        token = encryptor.encrypt_data(payload)

        assert isinstance(token, str)
        assert encryptor.decrypt_data(token) == payload

    def test_missing_key_falls_back_to_ephemeral(self, monkeypatch, caplog):
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        encryptor = DataEncryption()

        assert "ephemeral" in caplog.text.lower()
        assert encryptor.decrypt_data(encryptor.encrypt_data("demo")) == "demo"

    def test_invalid_key_falls_back_to_ephemeral(self, monkeypatch, caplog):
        monkeypatch.setenv("ENCRYPTION_KEY", "not-a-fernet-key")
        encryptor = DataEncryption()

        assert "not a valid fernet key" in caplog.text.lower()
        assert encryptor.decrypt_data(encryptor.encrypt_data([1, 2, 3])) == [1, 2, 3]

    def test_garbage_token_returns_none(self, monkeypatch):
        monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
        encryptor = DataEncryption()

        assert encryptor.decrypt_data("definitely-not-a-token") is None

    def test_key_id_argument_is_accepted_and_ignored(self, monkeypatch):
        # Older call sites passed a key id from the retired key-ring scheme
        monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
        encryptor = DataEncryption()

        token = encryptor.encrypt_data({"a": 1})
        assert encryptor.decrypt_data(token, key_id="legacy") == {"a": 1}

    def test_generate_encryption_key_is_valid(self):
        key = generate_encryption_key()
        Fernet(key.encode())  # raises if invalid
