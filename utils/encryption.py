"""Encryption utilities for secure data storage."""

import os
import base64
import logging
import json
from typing import Any, Dict, Optional, Tuple
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class DataEncryption:
    """Handles encryption and decryption of sensitive data."""

    def __init__(self):
        """Initialize encryption with key derivation."""
        self.ciphers: Dict[str, Fernet] = {}
        self.current_key_id: Optional[str] = None
        self._initialize_ciphers()

    # Salt must be deterministic so the same passphrase yields the same Fernet
    # key across restarts — ciphertext would otherwise become undecryptable.
    _SALT_PREFIX = b"carecompass_salt_2024"
    _PBKDF2_ITERATIONS = 600_000  # OWASP 2023 guidance for PBKDF2-HMAC-SHA256

    def _normalize_key(self, raw_key: str, key_id: Optional[str] = None) -> bytes:
        key_bytes = raw_key.encode() if isinstance(raw_key, str) else raw_key

        if len(key_bytes) != 44:
            salt = self._SALT_PREFIX + b":" + (key_id or "default").encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self._PBKDF2_ITERATIONS,
                backend=default_backend()
            )
            key_bytes = base64.urlsafe_b64encode(kdf.derive(key_bytes))

        return key_bytes

    def _initialize_ciphers(self) -> None:
        keyring = os.getenv("ENCRYPTION_KEYRING")
        encryption_key = os.getenv("ENCRYPTION_KEY")

        if keyring:
            entries = [item.strip() for item in keyring.split(",") if item.strip()]
            for entry in entries:
                if ":" in entry:
                    key_id, key_value = entry.split(":", 1)
                else:
                    key_id = f"key_{len(self.ciphers) + 1}"
                    key_value = entry

                try:
                    normalized_key = self._normalize_key(key_value, key_id)
                    self.ciphers[key_id] = Fernet(normalized_key)
                except Exception as exc:
                    logger.error("Failed to load encryption key %s: %s", key_id, exc)

            if self.ciphers:
                self.current_key_id = next(iter(self.ciphers.keys()))
        else:
            if not encryption_key:
                logger.warning("No ENCRYPTION_KEY found in environment. Generating temporary key.")
                logger.warning("⚠️  IMPORTANT: Set ENCRYPTION_KEY in .env for persistent encryption!")
                encryption_key = Fernet.generate_key().decode()
                os.environ["ENCRYPTION_KEY"] = encryption_key

            normalized_key = self._normalize_key(encryption_key, "default")
            self.ciphers["default"] = Fernet(normalized_key)
            self.current_key_id = "default"

        if not self.ciphers:
            raise RuntimeError("No valid encryption keys configured")

        logger.info("Data encryption initialized with %s key(s)", len(self.ciphers))

    def encrypt_data(self, data: Any) -> str:
        """Encrypt data for storage.

        Args:
            data: Data to encrypt (will be JSON serialized)

        Returns:
            Encrypted data as base64 string
        """
        encrypted, _ = self.encrypt_data_with_key_id(data)
        return encrypted

    def encrypt_data_with_key_id(self, data: Any) -> Tuple[str, str]:
        """Encrypt data and return the key id used.

        Args:
            data: Data to encrypt

        Returns:
            Tuple of encrypted data and key id
        """
        if not self.current_key_id:
            raise RuntimeError("No active encryption key configured")

        json_data = json.dumps(data)
        cipher = self.ciphers[self.current_key_id]
        encrypted = cipher.encrypt(json_data.encode())
        return base64.b64encode(encrypted).decode(), self.current_key_id

    def decrypt_data(self, encrypted_data: str, key_id: Optional[str] = None) -> Any:
        """Decrypt data from storage.

        Args:
            encrypted_data: Encrypted data as base64 string
            key_id: Optional key id to use for decryption

        Returns:
            Decrypted and deserialized data
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())

            if key_id and key_id in self.ciphers:
                return self._decrypt_with_cipher(self.ciphers[key_id], encrypted_bytes)

            for cipher in self.ciphers.values():
                try:
                    return self._decrypt_with_cipher(cipher, encrypted_bytes)
                except InvalidToken:
                    continue

        except Exception as e:
            logger.error("Decryption failed: %s", e)

        return None

    @staticmethod
    def _decrypt_with_cipher(cipher: Fernet, encrypted_bytes: bytes) -> Any:
        decrypted = cipher.decrypt(encrypted_bytes)
        return json.loads(decrypted.decode())

    def encrypt_dict_values(self, data_dict: Dict[str, Any], fields_to_encrypt: list) -> Dict[str, Any]:
        """Encrypt specific fields in a dictionary."""
        result = data_dict.copy()

        for field in fields_to_encrypt:
            if field in result and result[field]:
                try:
                    result[field] = self.encrypt_data(result[field])
                except Exception as e:
                    logger.error("Failed to encrypt field %s: %s", field, e)

        return result

    def decrypt_dict_values(self, data_dict: Dict[str, Any], fields_to_decrypt: list) -> Dict[str, Any]:
        """Decrypt specific fields in a dictionary."""
        result = data_dict.copy()

        for field in fields_to_decrypt:
            if field in result and result[field]:
                try:
                    decrypted = self.decrypt_data(result[field])
                    if decrypted is not None:
                        result[field] = decrypted
                except Exception as e:
                    logger.error("Failed to decrypt field %s: %s", field, e)
                    result[field] = None

        return result


def generate_encryption_key() -> str:
    """Generate a new Fernet encryption key."""
    return Fernet.generate_key().decode()


def get_encryptor() -> DataEncryption:
    """Get singleton instance of DataEncryption."""
    if not hasattr(get_encryptor, '_instance'):
        get_encryptor._instance = DataEncryption()

    return get_encryptor._instance
