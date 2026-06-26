"""Transform FHIR R4 resources into CareCompass provider dictionaries.

Maps Practitioner, PractitionerRole, Location, and Organization resources
into the flat provider dict schema expected by the scoring and UI layers.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FHIRToProviderTransformer:
    """Transforms FHIR Bundle search results into CareCompass provider dicts."""

    def transform_bundle(self, bundle: Dict) -> List[Dict[str, Any]]:
        """Transform a FHIR searchset Bundle into a list of provider dicts.

        Each bundle entry is expected to have the Practitioner resource plus
        helper keys ``_roles``, ``_locations``, and ``_organizations`` attached
        by the client (mock or real).

        Args:
            bundle: FHIR R4 Bundle (searchset) dict

        Returns:
            List of provider dicts matching the CareCompass schema
        """
        if not bundle or bundle.get("resourceType") != "Bundle":
            logger.warning("Invalid or empty FHIR bundle")
            return []

        entries = bundle.get("entry", [])
        providers: List[Dict[str, Any]] = []

        for entry in entries:
            practitioner = entry.get("resource", {})
            roles = entry.get("_roles", [])
            locations = entry.get("_locations", [])
            organizations = entry.get("_organizations", [])

            if practitioner.get("resourceType") != "Practitioner":
                continue

            try:
                provider = self.transform_practitioner(
                    practitioner, roles, locations, organizations
                )
                if provider:
                    providers.append(provider)
            except Exception as e:
                prac_id = practitioner.get("id", "unknown")
                logger.warning("Failed to transform practitioner %s: %s", prac_id, e)

        logger.info("Transformed %d FHIR entries into provider dicts", len(providers))
        return providers

    def transform_practitioner(
        self,
        practitioner: Dict,
        roles: List[Dict],
        locations: List[Dict],
        organizations: Optional[List[Dict]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Transform a single practitioner and related resources.

        Args:
            practitioner: FHIR Practitioner resource
            roles: Associated PractitionerRole resources
            locations: Associated Location resources
            organizations: Associated Organization resources

        Returns:
            Provider dict matching CareCompass schema, or None on failure
        """
        organizations = organizations or []

        # --- Name ---
        name = self._extract_name(practitioner)
        if not name:
            return None

        # --- NPI ---
        npi = self._extract_npi(practitioner)

        # --- Phone ---
        phone = self._extract_phone(practitioner, locations)

        # --- Specialty (from first matching role) ---
        specialty = self._extract_specialty(roles)

        # --- Location address ---
        location_str = self._extract_location(locations)

        # --- Insurance networks ---
        insurance_accepted = self._extract_insurance(organizations)

        # --- Education & experience ---
        education, years_experience = self._extract_qualifications(practitioner)

        provider: Dict[str, Any] = {
            "name": name,
            "specialty": specialty,
            "location": location_str,
            "phone": phone,
            "rating": 0.0,
            "review_count": None,
            "review_summary": "No reviews available",
            "review_sentiment": "unknown",
            "insurance_accepted": insurance_accepted,
            "distance": None,
            "education": education,
            "years_experience": years_experience,
            "data_source": "fhir",
            "fhir_metadata": {
                "practitioner_id": practitioner.get("id", ""),
                "npi": npi,
                "network_verified": True,
                "fhir_source": "mock" if not practitioner.get("meta") else "live",
                "networks": insurance_accepted,
            },
        }

        return provider

    # ------------------------------------------------------------------
    # Private extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_name(practitioner: Dict) -> str:
        """Format practitioner name as 'Dr. First Last'."""
        names = practitioner.get("name", [])
        if not names:
            return ""

        name_obj = names[0]
        prefix = name_obj.get("prefix", [""])[0] if name_obj.get("prefix") else "Dr."
        given = " ".join(name_obj.get("given", []))
        family = name_obj.get("family", "")

        if given and family:
            return f"{prefix} {given} {family}".strip()
        return family or given

    @staticmethod
    def _extract_npi(practitioner: Dict) -> Optional[str]:
        """Extract NPI from practitioner identifiers."""
        for identifier in practitioner.get("identifier", []):
            if "us-npi" in identifier.get("system", ""):
                return identifier.get("value")
        return None

    @staticmethod
    def _extract_phone(practitioner: Dict, locations: List[Dict]) -> str:
        """Extract phone number from practitioner or location telecom."""
        # Try practitioner telecom first
        for telecom in practitioner.get("telecom", []):
            if telecom.get("system") == "phone":
                return telecom.get("value", "N/A")

        # Fall back to first location telecom
        for loc in locations:
            for telecom in loc.get("telecom", []):
                if telecom.get("system") == "phone":
                    return telecom.get("value", "N/A")

        return "N/A"

    @staticmethod
    def _extract_specialty(roles: List[Dict]) -> str:
        """Extract specialty display text from practitioner roles."""
        for role in roles:
            for spec in role.get("specialty", []):
                for coding in spec.get("coding", []):
                    display = coding.get("display")
                    if display:
                        return display
        return "General Practice"

    @staticmethod
    def _extract_location(locations: List[Dict]) -> str:
        """Format the first location's address."""
        if not locations:
            return "N/A"

        loc = locations[0]
        addr = loc.get("address", {})
        parts = []

        lines = addr.get("line", [])
        if lines:
            parts.append(", ".join(lines))

        city = addr.get("city", "")
        state = addr.get("state", "")
        zip_code = addr.get("postalCode", "")

        city_state = f"{city}, {state}" if city and state else city or state
        if city_state:
            parts.append(city_state)
        if zip_code:
            parts.append(zip_code)

        return ", ".join(parts) if parts else "N/A"

    @staticmethod
    def _extract_insurance(organizations: List[Dict]) -> List[str]:
        """Extract unique insurance network names from organizations."""
        networks: list[str] = []
        seen: set[str] = set()

        for org in organizations:
            name = org.get("name", "")
            if name and name not in seen:
                networks.append(name)
                seen.add(name)

        return networks

    @staticmethod
    def _extract_qualifications(practitioner: Dict) -> tuple[Optional[str], Optional[int]]:
        """Extract education and estimated years of experience.

        Returns:
            (education_string, years_experience) tuple
        """
        education = None
        years_experience = None
        current_year = datetime.now().year

        for qual in practitioner.get("qualification", []):
            text = qual.get("code", {}).get("text", "")

            # Education: look for MD/DO school mentions
            if any(keyword in text for keyword in ["MD", "DO", "University", "School", "College"]):
                education = text

                # Estimate experience from graduation period
                period = qual.get("period", {})
                start_year_str = period.get("start", "")
                if start_year_str:
                    try:
                        start_year = int(start_year_str[:4])
                        years_experience = current_year - start_year
                    except (ValueError, IndexError):
                        pass

        return education, years_experience
