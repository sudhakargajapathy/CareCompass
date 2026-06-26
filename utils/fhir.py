"""FHIR R4 resource conversion utilities for CareCompass provider data.

Converts between internal provider dictionaries and FHIR R4 resources
(Practitioner, PractitionerRole, Location, Bundle).
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FHIR_RESOURCE_TYPE_PRACTITIONER = "Practitioner"
FHIR_RESOURCE_TYPE_PRACTITIONER_ROLE = "PractitionerRole"
FHIR_RESOURCE_TYPE_LOCATION = "Location"
FHIR_RESOURCE_TYPE_BUNDLE = "Bundle"


def _parse_name(full_name: str) -> Dict[str, Any]:
    """Parse a full name string into FHIR HumanName components.

    Args:
        full_name: e.g. "Dr. Sarah Johnson"

    Returns:
        FHIR HumanName dict with family, given, prefix.
    """
    parts = full_name.strip().split()
    prefix = []
    given = []
    family = ""

    for i, part in enumerate(parts):
        cleaned = part.rstrip(".,")
        if cleaned.lower() in ("dr", "md", "do", "phd", "np", "pa"):
            prefix.append(cleaned + "." if not cleaned.endswith(".") else cleaned)
        elif i == len(parts) - 1:
            family = part
        else:
            given.append(part)

    return {
        "use": "official",
        "text": full_name,
        "family": family,
        "given": given if given else [full_name],
        "prefix": prefix if prefix else [],
    }


def _parse_phone(phone: str) -> Optional[Dict[str, str]]:
    """Convert a phone string to a FHIR ContactPoint.

    Args:
        phone: e.g. "(602) 555-1234" or "N/A"

    Returns:
        FHIR ContactPoint dict, or None if not available.
    """
    if not phone or phone.upper() == "N/A":
        return None
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 7:
        return None
    return {
        "system": "phone",
        "value": phone,
        "use": "work",
    }


def _parse_location(location_str: str) -> Dict[str, Any]:
    """Convert a location string to a FHIR Address.

    Args:
        location_str: e.g. "Phoenix, AZ" or "123 Health St, Phoenix, AZ"

    Returns:
        FHIR Address dict.
    """
    parts = [p.strip() for p in location_str.split(",")]
    address: Dict[str, Any] = {"use": "work", "text": location_str}

    if len(parts) >= 2:
        address["city"] = parts[-2].strip()
        address["state"] = parts[-1].strip()
    if len(parts) >= 3:
        address["line"] = [", ".join(parts[:-2])]

    return address


def provider_to_fhir_practitioner(provider: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an internal provider dict to a FHIR R4 Practitioner resource.

    Args:
        provider: Internal provider dictionary with name, specialty, location, etc.

    Returns:
        FHIR R4 Practitioner resource dict.
    """
    name = provider.get("name", "")
    practitioner: Dict[str, Any] = {
        "resourceType": FHIR_RESOURCE_TYPE_PRACTITIONER,
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, name)),
        "active": True,
        "name": [_parse_name(name)],
    }

    phone_contact = _parse_phone(provider.get("phone", ""))
    if phone_contact:
        practitioner["telecom"] = [phone_contact]

    if provider.get("location"):
        practitioner["address"] = [_parse_location(provider["location"])]

    qualifications = []
    if provider.get("specialty"):
        qualifications.append({
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "display": provider["specialty"],
                }],
                "text": provider["specialty"],
            }
        })
    if provider.get("years_experience"):
        qualifications.append({
            "code": {
                "text": f"{provider['years_experience']} years experience",
            }
        })
    if qualifications:
        practitioner["qualification"] = qualifications

    return practitioner


def provider_to_fhir_practitioner_role(
    provider: Dict[str, Any], practitioner_id: str
) -> Dict[str, Any]:
    """Convert provider data to a FHIR PractitionerRole resource.

    Captures specialty, services, insurance, and rating as extensions.

    Args:
        provider: Internal provider dictionary.
        practitioner_id: The FHIR Practitioner resource id to reference.

    Returns:
        FHIR R4 PractitionerRole resource dict.
    """
    role: Dict[str, Any] = {
        "resourceType": FHIR_RESOURCE_TYPE_PRACTITIONER_ROLE,
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"role-{provider.get('name', '')}")),
        "active": True,
        "practitioner": {"reference": f"Practitioner/{practitioner_id}"},
    }

    if provider.get("specialty"):
        role["specialty"] = [{
            "coding": [{
                "system": "http://snomed.info/sct",
                "display": provider["specialty"],
            }],
            "text": provider["specialty"],
        }]

    extensions = []

    rating = provider.get("rating")
    if rating is not None and rating > 0:
        extensions.append({
            "url": "http://carecompass.example.org/fhir/StructureDefinition/provider-rating",
            "valueDecimal": rating,
        })

    review_count = provider.get("review_count")
    if review_count:
        extensions.append({
            "url": "http://carecompass.example.org/fhir/StructureDefinition/review-count",
            "valueInteger": review_count,
        })

    insurance = provider.get("insurance_accepted")
    if insurance:
        if isinstance(insurance, list):
            for plan in insurance:
                extensions.append({
                    "url": "http://carecompass.example.org/fhir/StructureDefinition/insurance-accepted",
                    "valueString": plan,
                })
        elif isinstance(insurance, str):
            extensions.append({
                "url": "http://carecompass.example.org/fhir/StructureDefinition/insurance-accepted",
                "valueString": insurance,
            })

    if extensions:
        role["extension"] = extensions

    return role


def provider_to_fhir_bundle(provider: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single provider to a FHIR Bundle containing Practitioner + PractitionerRole.

    Args:
        provider: Internal provider dictionary.

    Returns:
        FHIR R4 Bundle resource dict.
    """
    practitioner = provider_to_fhir_practitioner(provider)
    role = provider_to_fhir_practitioner_role(provider, practitioner["id"])

    return {
        "resourceType": FHIR_RESOURCE_TYPE_BUNDLE,
        "id": str(uuid.uuid4()),
        "type": "collection",
        "entry": [
            {"resource": practitioner, "fullUrl": f"urn:uuid:{practitioner['id']}"},
            {"resource": role, "fullUrl": f"urn:uuid:{role['id']}"},
        ],
    }


def providers_to_fhir_bundle(providers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a list of providers to a single FHIR searchset Bundle.

    Args:
        providers: List of internal provider dictionaries.

    Returns:
        FHIR R4 Bundle with all Practitioner and PractitionerRole resources.
    """
    entries = []
    for provider in providers:
        practitioner = provider_to_fhir_practitioner(provider)
        role = provider_to_fhir_practitioner_role(provider, practitioner["id"])
        entries.append({"resource": practitioner, "fullUrl": f"urn:uuid:{practitioner['id']}"})
        entries.append({"resource": role, "fullUrl": f"urn:uuid:{role['id']}"})

    return {
        "resourceType": FHIR_RESOURCE_TYPE_BUNDLE,
        "id": str(uuid.uuid4()),
        "type": "searchset",
        "total": len(providers),
        "entry": entries,
    }


def fhir_practitioner_to_provider(
    practitioner: Dict[str, Any],
    role: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert FHIR Practitioner (+ optional PractitionerRole) back to internal provider dict.

    Args:
        practitioner: FHIR R4 Practitioner resource.
        role: Optional FHIR R4 PractitionerRole resource with extensions.

    Returns:
        Internal provider dictionary.
    """
    provider: Dict[str, Any] = {}

    # Extract name
    names = practitioner.get("name", [])
    if names:
        provider["name"] = names[0].get("text", "")
    else:
        provider["name"] = ""

    # Extract phone
    telecoms = practitioner.get("telecom", [])
    for t in telecoms:
        if t.get("system") == "phone":
            provider["phone"] = t.get("value", "")
            break

    # Extract address / location
    addresses = practitioner.get("address", [])
    if addresses:
        provider["location"] = addresses[0].get("text", "")

    # Extract specialty from qualifications
    for qual in practitioner.get("qualification", []):
        code = qual.get("code", {})
        codings = code.get("coding", [])
        if codings and codings[0].get("system") == "http://snomed.info/sct":
            provider["specialty"] = codings[0].get("display", "")
            break
        elif code.get("text") and "experience" not in code["text"]:
            provider["specialty"] = code["text"]
            break

    # Extract years_experience
    for qual in practitioner.get("qualification", []):
        text = qual.get("code", {}).get("text", "")
        match = re.search(r"(\d+)\s+years?\s+experience", text)
        if match:
            provider["years_experience"] = int(match.group(1))
            break

    # Extract data from PractitionerRole extensions
    if role:
        for ext in role.get("extension", []):
            url = ext.get("url", "")
            if url.endswith("provider-rating"):
                provider["rating"] = ext.get("valueDecimal", 0.0)
            elif url.endswith("review-count"):
                provider["review_count"] = ext.get("valueInteger")
            elif url.endswith("insurance-accepted"):
                provider.setdefault("insurance_accepted", [])
                provider["insurance_accepted"].append(ext.get("valueString", ""))

        # Extract specialty from role if not already set
        if not provider.get("specialty"):
            specialties = role.get("specialty", [])
            if specialties:
                provider["specialty"] = specialties[0].get("text", "")

    return provider


def fhir_bundle_to_providers(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a FHIR Bundle back to a list of internal provider dicts.

    Pairs Practitioner resources with their PractitionerRole by reference.

    Args:
        bundle: FHIR R4 Bundle resource.

    Returns:
        List of internal provider dictionaries.
    """
    practitioners = {}
    roles = {}

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType")
        rid = resource.get("id", "")

        if rtype == FHIR_RESOURCE_TYPE_PRACTITIONER:
            practitioners[rid] = resource
        elif rtype == FHIR_RESOURCE_TYPE_PRACTITIONER_ROLE:
            ref = resource.get("practitioner", {}).get("reference", "")
            pract_id = ref.replace("Practitioner/", "")
            roles[pract_id] = resource

    providers = []
    for pid, practitioner in practitioners.items():
        role = roles.get(pid)
        providers.append(fhir_practitioner_to_provider(practitioner, role))

    return providers
