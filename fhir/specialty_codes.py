"""SNOMED CT specialty code mapping for FHIR Provider Directory queries.

Maps UI-facing specialty names to SNOMED CT codes used in FHIR
PractitionerRole.specialty searches.
"""

from typing import Optional

# SNOMED CT codes for medical specialties
# Source: SNOMED CT International Edition, mapped to US Core specialty value set
SPECIALTY_TO_SNOMED: dict[str, str] = {
    "Neurology": "394591006",
    "Cardiology": "394579002",
    "Dermatology": "394582007",
    "Orthopedics": "394801008",
    "Gastroenterology": "394584008",
    "Endocrinology": "394583002",
    "Psychiatry": "394587001",
    "Oncology": "394593009",
    "Rheumatology": "394810000",
    "Pulmonology": "418112009",
    "Family Medicine": "419772000",
    "Internal Medicine": "419192003",
    "Ophthalmology": "394594003",
    "Urology": "394612005",
    "Nephrology": "394589003",
    "Allergy and Immunology": "408439002",
    "Otolaryngology": "418960008",
    "Pediatrics": "394537008",
    "Obstetrics and Gynecology": "394585009",
    "Anesthesiology": "394577000",
    "Emergency Medicine": "773568002",
}

# Reverse lookup: SNOMED code -> specialty name
_SNOMED_TO_SPECIALTY: dict[str, str] = {v: k for k, v in SPECIALTY_TO_SNOMED.items()}


def get_snomed_code(specialty_name: str) -> Optional[str]:
    """Look up the SNOMED CT code for a specialty name.

    Args:
        specialty_name: UI-facing specialty name (e.g., "Neurology")

    Returns:
        SNOMED CT code string, or None if not found
    """
    return SPECIALTY_TO_SNOMED.get(specialty_name)


def get_specialty_name(snomed_code: str) -> Optional[str]:
    """Reverse-look up the specialty name from a SNOMED CT code.

    Args:
        snomed_code: SNOMED CT code (e.g., "394591006")

    Returns:
        Specialty name string, or None if not found
    """
    return _SNOMED_TO_SPECIALTY.get(snomed_code)
