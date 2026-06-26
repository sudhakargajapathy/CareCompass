"""Mock FHIR R4 resources for testing provider ↔ FHIR conversion."""

MOCK_FHIR_PRACTITIONER = {
    "resourceType": "Practitioner",
    "id": "pract-001",
    "active": True,
    "name": [
        {
            "use": "official",
            "text": "Dr. Sarah Johnson",
            "family": "Johnson",
            "given": ["Sarah"],
            "prefix": ["Dr."],
        }
    ],
    "telecom": [
        {"system": "phone", "value": "(602) 555-1234", "use": "work"}
    ],
    "address": [
        {
            "use": "work",
            "text": "Phoenix, AZ",
            "city": "Phoenix",
            "state": "AZ",
        }
    ],
    "qualification": [
        {
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "display": "Neurology",
                    }
                ],
                "text": "Neurology",
            }
        },
        {
            "code": {"text": "15 years experience"}
        },
    ],
}

MOCK_FHIR_PRACTITIONER_ROLE = {
    "resourceType": "PractitionerRole",
    "id": "role-001",
    "active": True,
    "practitioner": {"reference": "Practitioner/pract-001"},
    "specialty": [
        {
            "coding": [
                {"system": "http://snomed.info/sct", "display": "Neurology"}
            ],
            "text": "Neurology",
        }
    ],
    "extension": [
        {
            "url": "http://carecompass.example.org/fhir/StructureDefinition/provider-rating",
            "valueDecimal": 4.8,
        },
        {
            "url": "http://carecompass.example.org/fhir/StructureDefinition/review-count",
            "valueInteger": 127,
        },
        {
            "url": "http://carecompass.example.org/fhir/StructureDefinition/insurance-accepted",
            "valueString": "Aetna",
        },
        {
            "url": "http://carecompass.example.org/fhir/StructureDefinition/insurance-accepted",
            "valueString": "Blue Cross Blue Shield",
        },
    ],
}

MOCK_FHIR_BUNDLE = {
    "resourceType": "Bundle",
    "id": "bundle-001",
    "type": "searchset",
    "total": 2,
    "entry": [
        {
            "fullUrl": "urn:uuid:pract-001",
            "resource": MOCK_FHIR_PRACTITIONER,
        },
        {
            "fullUrl": "urn:uuid:role-001",
            "resource": MOCK_FHIR_PRACTITIONER_ROLE,
        },
        {
            "fullUrl": "urn:uuid:pract-002",
            "resource": {
                "resourceType": "Practitioner",
                "id": "pract-002",
                "active": True,
                "name": [
                    {
                        "use": "official",
                        "text": "Dr. Michael Chen",
                        "family": "Chen",
                        "given": ["Michael"],
                        "prefix": ["Dr."],
                    }
                ],
                "telecom": [
                    {"system": "phone", "value": "(480) 555-5678", "use": "work"}
                ],
                "address": [
                    {
                        "use": "work",
                        "text": "Scottsdale, AZ",
                        "city": "Scottsdale",
                        "state": "AZ",
                    }
                ],
                "qualification": [
                    {
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "display": "Neurology",
                                }
                            ],
                            "text": "Neurology",
                        }
                    }
                ],
            },
        },
        {
            "fullUrl": "urn:uuid:role-002",
            "resource": {
                "resourceType": "PractitionerRole",
                "id": "role-002",
                "active": True,
                "practitioner": {"reference": "Practitioner/pract-002"},
                "specialty": [
                    {
                        "coding": [
                            {"system": "http://snomed.info/sct", "display": "Neurology"}
                        ],
                        "text": "Neurology",
                    }
                ],
                "extension": [
                    {
                        "url": "http://carecompass.example.org/fhir/StructureDefinition/provider-rating",
                        "valueDecimal": 4.5,
                    },
                    {
                        "url": "http://carecompass.example.org/fhir/StructureDefinition/review-count",
                        "valueInteger": 89,
                    },
                    {
                        "url": "http://carecompass.example.org/fhir/StructureDefinition/insurance-accepted",
                        "valueString": "Most major insurance plans",
                    },
                ],
            },
        },
    ],
}

# Minimal practitioner with no optional fields
MOCK_FHIR_PRACTITIONER_MINIMAL = {
    "resourceType": "Practitioner",
    "id": "pract-minimal",
    "active": True,
    "name": [
        {
            "use": "official",
            "text": "John Doe",
            "family": "Doe",
            "given": ["John"],
            "prefix": [],
        }
    ],
}

# Empty bundle
MOCK_FHIR_BUNDLE_EMPTY = {
    "resourceType": "Bundle",
    "id": "bundle-empty",
    "type": "searchset",
    "total": 0,
    "entry": [],
}
