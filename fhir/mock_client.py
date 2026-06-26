"""Mock FHIR client with realistic in-memory provider data for development and testing.

Provides ~15 practitioners across multiple specialties in the Phoenix/Scottsdale AZ
metro area, distributed across major insurance networks.
"""

import logging
from typing import Dict, List, Optional

from .specialty_codes import get_snomed_code

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock data: practitioners, roles, locations, organizations
# ---------------------------------------------------------------------------

_ORGANIZATIONS = {
    "org-aetna": {
        "resourceType": "Organization",
        "id": "org-aetna",
        "name": "Aetna",
        "type": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/organization-type", "code": "ins", "display": "Insurance Company"}]}],
    },
    "org-bcbs": {
        "resourceType": "Organization",
        "id": "org-bcbs",
        "name": "Blue Cross Blue Shield",
        "type": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/organization-type", "code": "ins", "display": "Insurance Company"}]}],
    },
    "org-cigna": {
        "resourceType": "Organization",
        "id": "org-cigna",
        "name": "Cigna",
        "type": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/organization-type", "code": "ins", "display": "Insurance Company"}]}],
    },
    "org-united": {
        "resourceType": "Organization",
        "id": "org-united",
        "name": "UnitedHealth",
        "type": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/organization-type", "code": "ins", "display": "Insurance Company"}]}],
    },
    "org-medicare": {
        "resourceType": "Organization",
        "id": "org-medicare",
        "name": "Medicare",
        "type": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/organization-type", "code": "ins", "display": "Insurance Company"}]}],
    },
}

_LOCATIONS = {
    "loc-phoenix-1": {
        "resourceType": "Location",
        "id": "loc-phoenix-1",
        "name": "Phoenix Medical Plaza",
        "address": {
            "line": ["1234 N Central Ave"],
            "city": "Phoenix",
            "state": "AZ",
            "postalCode": "85004",
        },
        "telecom": [{"system": "phone", "value": "602-555-0101"}],
    },
    "loc-phoenix-2": {
        "resourceType": "Location",
        "id": "loc-phoenix-2",
        "name": "Banner Health Clinic",
        "address": {
            "line": ["5678 E Camelback Rd"],
            "city": "Phoenix",
            "state": "AZ",
            "postalCode": "85018",
        },
        "telecom": [{"system": "phone", "value": "602-555-0202"}],
    },
    "loc-phoenix-3": {
        "resourceType": "Location",
        "id": "loc-phoenix-3",
        "name": "Phoenix Neuro Center",
        "address": {
            "line": ["910 W McDowell Rd"],
            "city": "Phoenix",
            "state": "AZ",
            "postalCode": "85007",
        },
        "telecom": [{"system": "phone", "value": "602-555-0303"}],
    },
    "loc-scottsdale-1": {
        "resourceType": "Location",
        "id": "loc-scottsdale-1",
        "name": "Scottsdale Specialty Clinic",
        "address": {
            "line": ["2200 N Scottsdale Rd"],
            "city": "Scottsdale",
            "state": "AZ",
            "postalCode": "85257",
        },
        "telecom": [{"system": "phone", "value": "480-555-0401"}],
    },
    "loc-scottsdale-2": {
        "resourceType": "Location",
        "id": "loc-scottsdale-2",
        "name": "Mayo Clinic Scottsdale",
        "address": {
            "line": ["13400 E Shea Blvd"],
            "city": "Scottsdale",
            "state": "AZ",
            "postalCode": "85259",
        },
        "telecom": [{"system": "phone", "value": "480-555-0502"}],
    },
    "loc-tempe-1": {
        "resourceType": "Location",
        "id": "loc-tempe-1",
        "name": "Tempe Medical Group",
        "address": {
            "line": ["1100 S Rural Rd"],
            "city": "Tempe",
            "state": "AZ",
            "postalCode": "85281",
        },
        "telecom": [{"system": "phone", "value": "480-555-0601"}],
    },
    "loc-mesa-1": {
        "resourceType": "Location",
        "id": "loc-mesa-1",
        "name": "Mesa Healthcare Center",
        "address": {
            "line": ["330 E Main St"],
            "city": "Mesa",
            "state": "AZ",
            "postalCode": "85201",
        },
        "telecom": [{"system": "phone", "value": "480-555-0701"}],
    },
}

# Each tuple: (practitioner, list_of_roles)
# Roles link a practitioner to a specialty, location, and insurance network.
_MOCK_PRACTITIONERS: List[Dict] = [
    # --- Neurology (4 practitioners) ---
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-1001",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "1234567890"}],
            "name": [{"family": "Chen", "given": ["Sarah"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "602-555-0101"}],
            "qualification": [
                {"code": {"text": "MD - Johns Hopkins University School of Medicine"}, "period": {"start": "2008"}},
                {"code": {"text": "Board Certified - American Board of Psychiatry and Neurology"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-1001",
                "practitioner": {"reference": "Practitioner/prac-1001"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394591006", "display": "Neurology"}]}],
                "location": [{"reference": "Location/loc-phoenix-1"}],
                "organization": {"reference": "Organization/org-aetna"},
            },
            {
                "resourceType": "PractitionerRole",
                "id": "role-1001b",
                "practitioner": {"reference": "Practitioner/prac-1001"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394591006", "display": "Neurology"}]}],
                "location": [{"reference": "Location/loc-phoenix-1"}],
                "organization": {"reference": "Organization/org-bcbs"},
            },
        ],
    },
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-1002",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "2345678901"}],
            "name": [{"family": "Martinez", "given": ["Roberto"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "602-555-0303"}],
            "qualification": [
                {"code": {"text": "MD - University of Arizona College of Medicine"}, "period": {"start": "2012"}},
                {"code": {"text": "Board Certified - Neurology"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-1002",
                "practitioner": {"reference": "Practitioner/prac-1002"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394591006", "display": "Neurology"}]}],
                "location": [{"reference": "Location/loc-phoenix-3"}],
                "organization": {"reference": "Organization/org-cigna"},
            },
        ],
    },
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-1003",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "3456789012"}],
            "name": [{"family": "Patel", "given": ["Anita"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "480-555-0502"}],
            "qualification": [
                {"code": {"text": "MD - Mayo Clinic Alix School of Medicine"}, "period": {"start": "2005"}},
                {"code": {"text": "Board Certified - Neurology, Epilepsy"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-1003",
                "practitioner": {"reference": "Practitioner/prac-1003"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394591006", "display": "Neurology"}]}],
                "location": [{"reference": "Location/loc-scottsdale-2"}],
                "organization": {"reference": "Organization/org-united"},
            },
            {
                "resourceType": "PractitionerRole",
                "id": "role-1003b",
                "practitioner": {"reference": "Practitioner/prac-1003"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394591006", "display": "Neurology"}]}],
                "location": [{"reference": "Location/loc-scottsdale-2"}],
                "organization": {"reference": "Organization/org-medicare"},
            },
        ],
    },
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-1004",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "4567890123"}],
            "name": [{"family": "Thompson", "given": ["James"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "480-555-0601"}],
            "qualification": [
                {"code": {"text": "DO - Midwestern University"}, "period": {"start": "2015"}},
                {"code": {"text": "Board Certified - Neurology"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-1004",
                "practitioner": {"reference": "Practitioner/prac-1004"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394591006", "display": "Neurology"}]}],
                "location": [{"reference": "Location/loc-tempe-1"}],
                "organization": {"reference": "Organization/org-aetna"},
            },
        ],
    },
    # --- Cardiology (4 practitioners) ---
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-2001",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "5678901234"}],
            "name": [{"family": "Williams", "given": ["Emily"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "602-555-0202"}],
            "qualification": [
                {"code": {"text": "MD - Stanford University School of Medicine"}, "period": {"start": "2007"}},
                {"code": {"text": "Board Certified - Cardiovascular Disease"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-2001",
                "practitioner": {"reference": "Practitioner/prac-2001"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394579002", "display": "Cardiology"}]}],
                "location": [{"reference": "Location/loc-phoenix-2"}],
                "organization": {"reference": "Organization/org-bcbs"},
            },
            {
                "resourceType": "PractitionerRole",
                "id": "role-2001b",
                "practitioner": {"reference": "Practitioner/prac-2001"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394579002", "display": "Cardiology"}]}],
                "location": [{"reference": "Location/loc-phoenix-2"}],
                "organization": {"reference": "Organization/org-medicare"},
            },
        ],
    },
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-2002",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "6789012345"}],
            "name": [{"family": "Kumar", "given": ["Rajesh"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "480-555-0401"}],
            "qualification": [
                {"code": {"text": "MD - Harvard Medical School"}, "period": {"start": "2003"}},
                {"code": {"text": "Board Certified - Interventional Cardiology"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-2002",
                "practitioner": {"reference": "Practitioner/prac-2002"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394579002", "display": "Cardiology"}]}],
                "location": [{"reference": "Location/loc-scottsdale-1"}],
                "organization": {"reference": "Organization/org-aetna"},
            },
            {
                "resourceType": "PractitionerRole",
                "id": "role-2002b",
                "practitioner": {"reference": "Practitioner/prac-2002"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394579002", "display": "Cardiology"}]}],
                "location": [{"reference": "Location/loc-scottsdale-1"}],
                "organization": {"reference": "Organization/org-cigna"},
            },
        ],
    },
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-2003",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "7890123456"}],
            "name": [{"family": "Johnson", "given": ["Michael"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "480-555-0701"}],
            "qualification": [
                {"code": {"text": "MD - Creighton University School of Medicine"}, "period": {"start": "2010"}},
                {"code": {"text": "Board Certified - Cardiology"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-2003",
                "practitioner": {"reference": "Practitioner/prac-2003"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394579002", "display": "Cardiology"}]}],
                "location": [{"reference": "Location/loc-mesa-1"}],
                "organization": {"reference": "Organization/org-united"},
            },
        ],
    },
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-2004",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "8901234567"}],
            "name": [{"family": "Garcia", "given": ["Maria"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "602-555-0101"}],
            "qualification": [
                {"code": {"text": "MD - University of Arizona College of Medicine"}, "period": {"start": "2014"}},
                {"code": {"text": "Board Certified - Cardiology"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-2004",
                "practitioner": {"reference": "Practitioner/prac-2004"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394579002", "display": "Cardiology"}]}],
                "location": [{"reference": "Location/loc-phoenix-1"}],
                "organization": {"reference": "Organization/org-cigna"},
            },
        ],
    },
    # --- Dermatology (3 practitioners) ---
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-3001",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "9012345678"}],
            "name": [{"family": "Lee", "given": ["Jessica"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "480-555-0401"}],
            "qualification": [
                {"code": {"text": "MD - UCLA David Geffen School of Medicine"}, "period": {"start": "2011"}},
                {"code": {"text": "Board Certified - Dermatology"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-3001",
                "practitioner": {"reference": "Practitioner/prac-3001"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394582007", "display": "Dermatology"}]}],
                "location": [{"reference": "Location/loc-scottsdale-1"}],
                "organization": {"reference": "Organization/org-bcbs"},
            },
            {
                "resourceType": "PractitionerRole",
                "id": "role-3001b",
                "practitioner": {"reference": "Practitioner/prac-3001"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394582007", "display": "Dermatology"}]}],
                "location": [{"reference": "Location/loc-scottsdale-1"}],
                "organization": {"reference": "Organization/org-aetna"},
            },
        ],
    },
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-3002",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "0123456789"}],
            "name": [{"family": "Brown", "given": ["David"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "602-555-0202"}],
            "qualification": [
                {"code": {"text": "MD - Columbia University Vagelos College of Physicians and Surgeons"}, "period": {"start": "2009"}},
                {"code": {"text": "Board Certified - Dermatology, Dermatopathology"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-3002",
                "practitioner": {"reference": "Practitioner/prac-3002"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394582007", "display": "Dermatology"}]}],
                "location": [{"reference": "Location/loc-phoenix-2"}],
                "organization": {"reference": "Organization/org-united"},
            },
        ],
    },
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-3003",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "1122334455"}],
            "name": [{"family": "Anderson", "given": ["Karen"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "480-555-0601"}],
            "qualification": [
                {"code": {"text": "DO - A.T. Still University"}, "period": {"start": "2016"}},
                {"code": {"text": "Board Certified - Dermatology"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-3003",
                "practitioner": {"reference": "Practitioner/prac-3003"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394582007", "display": "Dermatology"}]}],
                "location": [{"reference": "Location/loc-tempe-1"}],
                "organization": {"reference": "Organization/org-medicare"},
            },
            {
                "resourceType": "PractitionerRole",
                "id": "role-3003b",
                "practitioner": {"reference": "Practitioner/prac-3003"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394582007", "display": "Dermatology"}]}],
                "location": [{"reference": "Location/loc-tempe-1"}],
                "organization": {"reference": "Organization/org-cigna"},
            },
        ],
    },
    # --- Orthopedics (2 practitioners) ---
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-4001",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "2233445566"}],
            "name": [{"family": "Wilson", "given": ["Thomas"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "602-555-0303"}],
            "qualification": [
                {"code": {"text": "MD - University of Pennsylvania Perelman School of Medicine"}, "period": {"start": "2006"}},
                {"code": {"text": "Board Certified - Orthopedic Surgery"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-4001",
                "practitioner": {"reference": "Practitioner/prac-4001"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394801008", "display": "Orthopedics"}]}],
                "location": [{"reference": "Location/loc-phoenix-3"}],
                "organization": {"reference": "Organization/org-aetna"},
            },
            {
                "resourceType": "PractitionerRole",
                "id": "role-4001b",
                "practitioner": {"reference": "Practitioner/prac-4001"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394801008", "display": "Orthopedics"}]}],
                "location": [{"reference": "Location/loc-phoenix-3"}],
                "organization": {"reference": "Organization/org-bcbs"},
            },
        ],
    },
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-4002",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "3344556677"}],
            "name": [{"family": "Nguyen", "given": ["Lisa"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "480-555-0502"}],
            "qualification": [
                {"code": {"text": "MD - Mayo Clinic Alix School of Medicine"}, "period": {"start": "2013"}},
                {"code": {"text": "Board Certified - Orthopedic Surgery, Sports Medicine"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-4002",
                "practitioner": {"reference": "Practitioner/prac-4002"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394801008", "display": "Orthopedics"}]}],
                "location": [{"reference": "Location/loc-scottsdale-2"}],
                "organization": {"reference": "Organization/org-united"},
            },
            {
                "resourceType": "PractitionerRole",
                "id": "role-4002b",
                "practitioner": {"reference": "Practitioner/prac-4002"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394801008", "display": "Orthopedics"}]}],
                "location": [{"reference": "Location/loc-scottsdale-2"}],
                "organization": {"reference": "Organization/org-medicare"},
            },
        ],
    },
    # --- Gastroenterology (2 practitioners) ---
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-5001",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "4455667788"}],
            "name": [{"family": "Davis", "given": ["Robert"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "602-555-0101"}],
            "qualification": [
                {"code": {"text": "MD - Baylor College of Medicine"}, "period": {"start": "2004"}},
                {"code": {"text": "Board Certified - Gastroenterology"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-5001",
                "practitioner": {"reference": "Practitioner/prac-5001"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394584008", "display": "Gastroenterology"}]}],
                "location": [{"reference": "Location/loc-phoenix-1"}],
                "organization": {"reference": "Organization/org-bcbs"},
            },
            {
                "resourceType": "PractitionerRole",
                "id": "role-5001b",
                "practitioner": {"reference": "Practitioner/prac-5001"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394584008", "display": "Gastroenterology"}]}],
                "location": [{"reference": "Location/loc-phoenix-1"}],
                "organization": {"reference": "Organization/org-cigna"},
            },
        ],
    },
    {
        "practitioner": {
            "resourceType": "Practitioner",
            "id": "prac-5002",
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": "5566778899"}],
            "name": [{"family": "Taylor", "given": ["Michelle"], "prefix": ["Dr."]}],
            "telecom": [{"system": "phone", "value": "480-555-0701"}],
            "qualification": [
                {"code": {"text": "MD - University of Michigan Medical School"}, "period": {"start": "2011"}},
                {"code": {"text": "Board Certified - Gastroenterology, Hepatology"}},
            ],
        },
        "roles": [
            {
                "resourceType": "PractitionerRole",
                "id": "role-5002",
                "practitioner": {"reference": "Practitioner/prac-5002"},
                "specialty": [{"coding": [{"system": "http://snomed.info/sct", "code": "394584008", "display": "Gastroenterology"}]}],
                "location": [{"reference": "Location/loc-mesa-1"}],
                "organization": {"reference": "Organization/org-aetna"},
            },
        ],
    },
]


def _parse_location(location_str: str) -> tuple[Optional[str], Optional[str]]:
    """Parse 'City, State' into (city, state)."""
    parts = [p.strip() for p in location_str.split(",")]
    city = parts[0] if parts else None
    state = parts[1] if len(parts) > 1 else None
    return city, state


def _location_matches(loc_resource: Dict, city: Optional[str], state: Optional[str]) -> bool:
    """Check if a Location resource matches city/state."""
    addr = loc_resource.get("address", {})
    loc_city = addr.get("city", "").lower()
    loc_state = addr.get("state", "").lower()

    if city and state:
        return loc_city == city.lower() or loc_state == state.lower()
    if city:
        return loc_city == city.lower()
    if state:
        return loc_state == state.lower()
    return True


def _specialty_matches(role: Dict, specialty: str) -> bool:
    """Check if a PractitionerRole specialty matches by SNOMED code or display text."""
    snomed_code = get_snomed_code(specialty)
    for spec in role.get("specialty", []):
        for coding in spec.get("coding", []):
            if snomed_code and coding.get("code") == snomed_code:
                return True
            if coding.get("display", "").lower() == specialty.lower():
                return True
    return False


def _network_matches(role: Dict, network: str) -> bool:
    """Check if a PractitionerRole belongs to the specified insurance network."""
    org_ref = role.get("organization", {}).get("reference", "")
    org_id = org_ref.split("/")[-1] if "/" in org_ref else org_ref
    org = _ORGANIZATIONS.get(org_id, {})
    return network.lower() in org.get("name", "").lower()


class MockFHIRClient:
    """Mock FHIR client with in-memory data for development and testing."""

    def __init__(self):
        logger.info("MockFHIRClient initialized with %d practitioners", len(_MOCK_PRACTITIONERS))

    def search_practitioners(
        self,
        specialty: str,
        location: str,
        insurance_network: str | None = None,
        count: int = 20,
    ) -> Dict:
        """Search mock practitioners with filtering."""
        city, state = _parse_location(location)

        matched_entries = []
        seen_practitioner_ids: set[str] = set()

        for record in _MOCK_PRACTITIONERS:
            practitioner = record["practitioner"]
            prac_id = practitioner["id"]

            # Skip if already matched this practitioner
            if prac_id in seen_practitioner_ids:
                continue

            # Check if any role matches specialty + location + network
            matching_roles = []
            for role in record["roles"]:
                if not _specialty_matches(role, specialty):
                    continue

                # Check location
                for loc_ref in role.get("location", []):
                    loc_id = loc_ref.get("reference", "").split("/")[-1]
                    loc_resource = _LOCATIONS.get(loc_id)
                    if loc_resource and _location_matches(loc_resource, city, state):
                        if insurance_network is None or _network_matches(role, insurance_network):
                            matching_roles.append(role)
                            break

            if matching_roles:
                seen_practitioner_ids.add(prac_id)
                # Build entry with practitioner + all matching roles + locations
                entry = {
                    "fullUrl": f"urn:uuid:{prac_id}",
                    "resource": practitioner,
                    "search": {"mode": "match"},
                }

                # Attach roles and resolved locations/orgs as contained references
                entry["_roles"] = matching_roles
                entry["_locations"] = []
                entry["_organizations"] = []

                for r in matching_roles:
                    for loc_ref in r.get("location", []):
                        loc_id = loc_ref.get("reference", "").split("/")[-1]
                        if loc_id in _LOCATIONS:
                            entry["_locations"].append(_LOCATIONS[loc_id])

                    org_ref = r.get("organization", {}).get("reference", "")
                    org_id = org_ref.split("/")[-1] if "/" in org_ref else ""
                    if org_id in _ORGANIZATIONS:
                        entry["_organizations"].append(_ORGANIZATIONS[org_id])

                matched_entries.append(entry)

            if len(matched_entries) >= count:
                break

        bundle = {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": len(matched_entries),
            "entry": matched_entries,
        }

        logger.info(
            "Mock FHIR search: specialty=%s, location=%s, network=%s -> %d results",
            specialty, location, insurance_network, len(matched_entries),
        )
        return bundle

    def get_practitioner(self, practitioner_id: str) -> Dict:
        """Retrieve a single practitioner by ID."""
        for record in _MOCK_PRACTITIONERS:
            if record["practitioner"]["id"] == practitioner_id:
                return record["practitioner"]
        return {}

    def get_practitioner_roles(self, practitioner_id: str) -> List[Dict]:
        """Retrieve roles for a practitioner."""
        for record in _MOCK_PRACTITIONERS:
            if record["practitioner"]["id"] == practitioner_id:
                return record["roles"]
        return []

    def get_location(self, location_id: str) -> Dict:
        """Retrieve a location by ID."""
        return _LOCATIONS.get(location_id, {})

    def is_available(self) -> bool:
        """Mock client is always available."""
        return True
