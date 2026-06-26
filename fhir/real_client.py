"""Real FHIR R4 client for querying live payer Provider Directory endpoints.

Handles OAuth2 client credentials authentication, FHIR search parameter
construction, bundle pagination, and error recovery (429 retry, 401 refresh).
"""

import logging
import time
from typing import Dict, List, Optional

import requests

from utils.config import get_config
from .specialty_codes import get_snomed_code

logger = logging.getLogger(__name__)


class RealFHIRClient:
    """FHIR client for live payer Provider Directory APIs."""

    def __init__(self):
        self.config = get_config()
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0

        # Validate required config
        fhir_validation = self.config.validate_fhir_config()
        if not fhir_validation["is_valid"]:
            raise ValueError(
                "FHIR client configuration incomplete: "
                + "; ".join(fhir_validation["errors"])
            )

        self._base_url = self.config.FHIR_BASE_URL.rstrip("/")
        self._token_url = self.config.FHIR_TOKEN_URL
        self._client_id = self.config.FHIR_CLIENT_ID
        self._client_secret = self.config.FHIR_CLIENT_SECRET
        self._timeout = self.config.FHIR_REQUEST_TIMEOUT
        self._max_pages = self.config.FHIR_MAX_PAGES

        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        })

        logger.info("RealFHIRClient initialized for %s", self._base_url)

    # ------------------------------------------------------------------
    # OAuth2 token management
    # ------------------------------------------------------------------

    def _ensure_token(self) -> None:
        """Obtain or refresh the OAuth2 access token."""
        if self._access_token and time.time() < self._token_expires_at - 30:
            return  # Token still valid (with 30s buffer)

        logger.info("Requesting new OAuth2 token from %s", self._token_url)

        try:
            resp = requests.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": "system/*.read",
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            token_data = resp.json()

            self._access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires_at = time.time() + expires_in

            self._session.headers["Authorization"] = f"Bearer {self._access_token}"
            logger.info("OAuth2 token obtained, expires in %ds", expires_in)

        except requests.RequestException as e:
            logger.error("OAuth2 token request failed: %s", e)
            raise RuntimeError(f"Failed to obtain FHIR access token: {e}") from e

    # ------------------------------------------------------------------
    # HTTP request with retry logic
    # ------------------------------------------------------------------

    def _request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Make an authenticated GET request with retry logic.

        Handles:
        - 401: Token refresh and retry
        - 429: Rate-limit backoff and retry
        - Timeouts and general HTTP errors
        """
        self._ensure_token()

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)

                if resp.status_code == 401 and attempt < max_retries:
                    logger.warning("401 Unauthorized — refreshing token (attempt %d)", attempt + 1)
                    self._access_token = None  # Force refresh
                    self._ensure_token()
                    continue

                if resp.status_code == 429 and attempt < max_retries:
                    retry_after = int(resp.headers.get("Retry-After", "5"))
                    logger.warning("429 Rate limited — waiting %ds (attempt %d)", retry_after, attempt + 1)
                    time.sleep(min(retry_after, 30))  # Cap wait at 30s
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.Timeout:
                logger.warning("Request timed out for %s (attempt %d)", url, attempt + 1)
                if attempt >= max_retries:
                    raise
            except requests.RequestException as e:
                logger.error("FHIR request failed: %s", e)
                if attempt >= max_retries:
                    raise

        return {}

    # ------------------------------------------------------------------
    # FHIR search operations
    # ------------------------------------------------------------------

    def search_practitioners(
        self,
        specialty: str,
        location: str,
        insurance_network: str | None = None,
        count: int = 20,
    ) -> Dict:
        """Search for practitioners via the FHIR Provider Directory.

        Constructs FHIR search parameters and follows pagination links.
        """
        params: Dict[str, str] = {
            "_count": str(count),
            "_include": "PractitionerRole:practitioner,PractitionerRole:location,PractitionerRole:organization",
        }

        # Map specialty to SNOMED code
        snomed_code = get_snomed_code(specialty)
        if snomed_code:
            params["specialty"] = f"http://snomed.info/sct|{snomed_code}"
        else:
            params["specialty:text"] = specialty

        # Parse location into city/state
        parts = [p.strip() for p in location.split(",")]
        if parts:
            params["location.address-city"] = parts[0]
        if len(parts) > 1:
            params["location.address-state"] = parts[1]

        # Network filter
        if insurance_network:
            params["organization.name"] = insurance_network

        url = f"{self._base_url}/PractitionerRole"

        # Fetch first page
        bundle = self._request(url, params=params)
        if not bundle or bundle.get("resourceType") != "Bundle":
            return {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []}

        # Follow pagination
        all_entries = list(bundle.get("entry", []))
        pages_fetched = 1

        while pages_fetched < self._max_pages:
            next_url = self._get_next_link(bundle)
            if not next_url:
                break

            bundle = self._request(next_url)
            if not bundle or not bundle.get("entry"):
                break

            all_entries.extend(bundle["entry"])
            pages_fetched += 1

        # Reorganize entries: group by Practitioner
        result_bundle = self._reorganize_bundle(all_entries)

        logger.info(
            "FHIR search complete: %d entries across %d pages -> %d practitioners",
            len(all_entries), pages_fetched, result_bundle["total"],
        )
        return result_bundle

    def get_practitioner(self, practitioner_id: str) -> Dict:
        """Retrieve a single Practitioner resource."""
        url = f"{self._base_url}/Practitioner/{practitioner_id}"
        return self._request(url)

    def get_practitioner_roles(self, practitioner_id: str) -> List[Dict]:
        """Retrieve PractitionerRole resources for a practitioner."""
        url = f"{self._base_url}/PractitionerRole"
        params = {"practitioner": f"Practitioner/{practitioner_id}"}
        bundle = self._request(url, params=params)
        return [e.get("resource", {}) for e in bundle.get("entry", [])]

    def get_location(self, location_id: str) -> Dict:
        """Retrieve a single Location resource."""
        url = f"{self._base_url}/Location/{location_id}"
        return self._request(url)

    def is_available(self) -> bool:
        """Check if the FHIR endpoint is reachable."""
        try:
            self._ensure_token()
            resp = self._session.get(
                f"{self._base_url}/metadata",
                timeout=self._timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_next_link(bundle: Dict) -> Optional[str]:
        """Extract the 'next' pagination URL from a Bundle."""
        for link in bundle.get("link", []):
            if link.get("relation") == "next":
                return link.get("url")
        return None

    def _reorganize_bundle(self, entries: List[Dict]) -> Dict:
        """Reorganize _include'd entries into per-practitioner groups.

        The FHIR server returns Practitioner, PractitionerRole, Location,
        and Organization resources as flat entries. This method groups them
        into the structure the transformer expects.
        """
        practitioners: Dict[str, Dict] = {}
        roles_by_prac: Dict[str, List[Dict]] = {}
        locations: Dict[str, Dict] = {}
        organizations: Dict[str, Dict] = {}

        for entry in entries:
            resource = entry.get("resource", {})
            res_type = resource.get("resourceType", "")
            res_id = resource.get("id", "")

            if res_type == "Practitioner":
                practitioners[res_id] = resource
            elif res_type == "PractitionerRole":
                prac_ref = resource.get("practitioner", {}).get("reference", "")
                prac_id = prac_ref.split("/")[-1] if "/" in prac_ref else prac_ref
                roles_by_prac.setdefault(prac_id, []).append(resource)
            elif res_type == "Location":
                locations[res_id] = resource
            elif res_type == "Organization":
                organizations[res_id] = resource

        # Build grouped entries
        grouped_entries = []
        for prac_id, practitioner in practitioners.items():
            roles = roles_by_prac.get(prac_id, [])

            # Resolve location and organization references
            entry_locations = []
            entry_organizations = []

            for role in roles:
                for loc_ref in role.get("location", []):
                    loc_id = loc_ref.get("reference", "").split("/")[-1]
                    if loc_id in locations:
                        entry_locations.append(locations[loc_id])

                org_ref = role.get("organization", {}).get("reference", "")
                org_id = org_ref.split("/")[-1] if "/" in org_ref else ""
                if org_id in organizations:
                    entry_organizations.append(organizations[org_id])

            grouped_entries.append({
                "fullUrl": f"{self._base_url}/Practitioner/{prac_id}",
                "resource": practitioner,
                "search": {"mode": "match"},
                "_roles": roles,
                "_locations": entry_locations,
                "_organizations": entry_organizations,
            })

        return {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": len(grouped_entries),
            "entry": grouped_entries,
        }
