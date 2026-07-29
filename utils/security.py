"""Security utilities for input validation and sanitization."""

import re
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates and sanitizes user inputs to prevent injection attacks."""

    # Allowed medical specialties (whitelist approach)
    ALLOWED_SPECIALTIES = {
        "neurology", "cardiology", "dermatology", "orthopedics",
        "gastroenterology", "endocrinology", "psychiatry", "oncology",
        "rheumatology", "pulmonology", "family medicine", "internal medicine",
        "pediatrics", "obstetrics", "gynecology", "urology", "ophthalmology",
        "otolaryngology", "anesthesiology", "radiology", "pathology",
        "emergency medicine", "general surgery", "plastic surgery"
    }

    @staticmethod
    def sanitize_specialty(specialty: str) -> Optional[str]:
        """Validate and sanitize medical specialty input.

        Args:
            specialty: User-provided specialty string

        Returns:
            Sanitized specialty or None if invalid
        """
        if not specialty:
            return None

        # Normalize to lowercase and remove extra whitespace
        normalized = " ".join(specialty.lower().strip().split())

        # Check against whitelist
        if normalized not in InputValidator.ALLOWED_SPECIALTIES:
            logger.warning(f"Invalid specialty attempted: {specialty}")
            return None

        return normalized.title()

    @staticmethod
    def sanitize_location(location: str, max_length: int = 200) -> Optional[str]:
        """Validate and sanitize location input.

        Args:
            location: User-provided location string
            max_length: Maximum allowed length

        Returns:
            Sanitized location or None if invalid
        """
        if not location or not location.strip():
            return None

        # Remove extra whitespace
        location = location.strip()

        # Check length
        if len(location) > max_length:
            logger.warning(f"Location exceeds max length: {len(location)} chars")
            return None

        # Only allow alphanumeric, spaces, commas, periods, hyphens
        # This prevents prompt injection attempts
        if not re.match(r'^[a-zA-Z0-9\s,.\-]+$', location):
            logger.warning(f"Location contains invalid characters: {location}")
            return None

        return location

    @staticmethod
    def sanitize_insurance(insurance: str) -> Optional[str]:
        """Validate and sanitize insurance input.

        Args:
            insurance: User-provided insurance string

        Returns:
            Sanitized insurance or None if invalid
        """
        if not insurance or not insurance.strip():
            return None

        # Whitelist of common insurance providers
        allowed_insurance = {
            "aetna", "blue cross blue shield", "bcbs", "cigna",
            "unitedhealth", "united healthcare", "medicare", "medicaid",
            "humana", "kaiser", "anthem", "other"
        }

        normalized = insurance.lower().strip()

        # Check against whitelist (partial match for variations)
        if not any(allowed in normalized for allowed in allowed_insurance):
            logger.warning(f"Invalid insurance provider: {insurance}")
            return None

        return insurance.strip()

    @staticmethod
    def sanitize_notes(notes: str, max_length: int = 500) -> str:
        """Validate and sanitize user notes/preferences.

        Args:
            notes: User-provided notes
            max_length: Maximum allowed length

        Returns:
            Sanitized notes (empty string if invalid)
        """
        if not notes or not notes.strip():
            return ""

        # Remove extra whitespace
        notes = " ".join(notes.strip().split())

        # Enforce length limit
        if len(notes) > max_length:
            logger.warning(f"Notes exceed max length: {len(notes)} chars")
            notes = notes[:max_length]

        # Remove potentially dangerous patterns
        # Prevent prompt injection keywords
        dangerous_patterns = [
            r'ignore\s+(all\s+)?previous\s+instructions?',
            r'ignore\s+(all\s+)?above',
            r'disregard\s+(all\s+)?previous',
            r'system\s+prompt',
            r'<\|.*?\|>',  # Special tokens
            r'\[INST\]',   # Instruction markers
            r'###\s*system',
            r'###\s*assistant',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, notes, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected in notes: {pattern}")
                notes = re.sub(pattern, '[REDACTED]', notes, flags=re.IGNORECASE)

        return notes

    @staticmethod
    def validate_preferences(preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize user preferences.

        Args:
            preferences: User preference dictionary

        Returns:
            Sanitized preferences dictionary
        """
        sanitized = {}

        # Validate numeric weights (must be 0-1). ONLY the three real
        # ranking weights survive: emitting legacy keys with defaults
        # (insurance_priority: 0.0, notes: "") advertised phantom ranking
        # factors to the critic — live, the bias analysis flagged
        # "insurance excluded by weight 0.0" as a methodology bias when
        # insurance was never a ranking factor at all.
        for key in ['location_weight', 'rating_weight', 'experience_weight']:
            value = preferences.get(key, 0.0)
            try:
                value = float(value)
                # Clamp to valid range
                value = max(0.0, min(1.0, value))
                sanitized[key] = value
            except (ValueError, TypeError):
                logger.warning(f"Invalid preference value for {key}: {value}")
                sanitized[key] = 0.0

        return sanitized


class PromptSanitizer:
    """Sanitizes inputs before inserting into LLM prompts."""

    @staticmethod
    def escape_for_prompt(text: str) -> str:
        """Escape text for safe insertion into prompts.

        Args:
            text: Text to escape

        Returns:
            Escaped text safe for prompt insertion
        """
        if not text:
            return ""

        # Remove or escape special characters that could break prompt structure
        # Replace newlines with spaces
        text = text.replace('\n', ' ').replace('\r', ' ')

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove quote marks that could break JSON or prompt structure
        text = text.replace('"', "'")

        return text

    @staticmethod
    def create_delimited_section(label: str, content: str, delimiter: str = "```") -> str:
        """Create a clearly delimited section in a prompt.

        Args:
            label: Section label
            content: Content to include
            delimiter: Delimiter to use

        Returns:
            Delimited section string
        """
        return f"{delimiter} {label}\n{content}\n{delimiter}"


def validate_search_params(specialty: str, location: str, insurance: Optional[str] = None) -> Dict[str, Any]:
    """Validate all search parameters at once.

    Args:
        specialty: Medical specialty
        location: Search location
        insurance: Insurance provider (optional)

    Returns:
        Dictionary with validation results and sanitized values
    """
    validator = InputValidator()

    sanitized_specialty = validator.sanitize_specialty(specialty)
    sanitized_location = validator.sanitize_location(location)
    sanitized_insurance = validator.sanitize_insurance(insurance) if insurance else None

    is_valid = sanitized_specialty is not None and sanitized_location is not None

    return {
        "is_valid": is_valid,
        "specialty": sanitized_specialty,
        "location": sanitized_location,
        "insurance": sanitized_insurance,
        "errors": [
            "Invalid specialty" if sanitized_specialty is None else None,
            "Invalid location" if sanitized_location is None else None,
        ]
    }
