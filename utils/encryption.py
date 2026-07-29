"""Single-key Fernet encryption for data at rest.

ENCRYPTION_KEY must be a Fernet key (generate one with
generate_encryption_key() or `python -c "from cryptography.fernet import
Fernet; print(Fernet.generate_key().decode())"`). Without one, an
ephemeral key is generated so local demos still work — anything encrypted
with it is unreadable after the process exits.
"""

import json
import logging
import os
from typing import Any, Optional

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class DataEncryption:
    """Encrypts/decrypts JSON-serializable data with a single Fernet key."""

    def __init__(self):
        key = os.getenv("ENCRYPTION_KEY")
        if key:
            try:
                self.cipher = Fernet(key.encode())
            except ValueError:
                logger.warning(
                    "ENCRYPTION_KEY is not a valid Fernet key; using an ephemeral key. "
                    "Generate one with utils.encryption.generate_encryption_key()."
                )
                self.cipher = Fernet(Fernet.generate_key())
        else:
            logger.warning(
                "No ENCRYPTION_KEY set; using an ephemeral key. "
                "Set ENCRYPTION_KEY in .env for persistent encryption."
            )
            self.cipher = Fernet(Fernet.generate_key())

    def encrypt_data(self, data: Any) -> str:
        """Encrypt JSON-serializable data, returning a Fernet token string."""
        return self.cipher.encrypt(json.dumps(data).encode()).decode()

    def decrypt_data(self, encrypted_data: str, key_id: Optional[str] = None) -> Any:
        """Decrypt a token produced by encrypt_data.

        Args:
            encrypted_data: Fernet token string
            key_id: Ignored; accepted for backwards compatibility with the
                retired multi-key call sites

        Returns:
            The decrypted data, or None if the token can't be decrypted
        """
        try:
            return json.loads(self.cipher.decrypt(encrypted_data.encode()).decode())
        except Exception as e:
            logger.error("Decryption failed: %s", e)
            return None


def generate_encryption_key() -> str:
    """Generate a new Fernet encryption key."""
    return Fernet.generate_key().decode()


def get_encryptor() -> DataEncryption:
    """Get singleton instance of DataEncryption."""
    if not hasattr(get_encryptor, "_instance"):
        get_encryptor._instance = DataEncryption()

    return get_encryptor._instance
