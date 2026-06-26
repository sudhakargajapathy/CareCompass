"""FHIR client protocol and factory for CareCompass.

Defines the interface all FHIR clients must implement and provides
a factory that selects mock vs real based on configuration.
"""

import logging
from typing import Dict, List, Protocol, runtime_checkable

from utils.config import get_config

logger = logging.getLogger(__name__)


@runtime_checkable
class FHIRClientProtocol(Protocol):
    """Protocol defining the interface for FHIR Provider Directory clients."""

    def search_practitioners(
        self,
        specialty: str,
        location: str,
        insurance_network: str | None = None,
        count: int = 20,
    ) -> Dict:
        """Search for practitioners matching criteria.

        Args:
            specialty: Medical specialty name (e.g., "Neurology")
            location: City, State string (e.g., "Phoenix, AZ")
            insurance_network: Payer network name (optional)
            count: Maximum results to return

        Returns:
            FHIR R4 Bundle (searchset) as a dict
        """
        ...

    def get_practitioner(self, practitioner_id: str) -> Dict:
        """Retrieve a single Practitioner resource.

        Args:
            practitioner_id: FHIR resource ID

        Returns:
            FHIR Practitioner resource dict
        """
        ...

    def get_practitioner_roles(self, practitioner_id: str) -> List[Dict]:
        """Retrieve PractitionerRole resources linked to a practitioner.

        Args:
            practitioner_id: FHIR Practitioner resource ID

        Returns:
            List of FHIR PractitionerRole resource dicts
        """
        ...

    def get_location(self, location_id: str) -> Dict:
        """Retrieve a single Location resource.

        Args:
            location_id: FHIR resource ID

        Returns:
            FHIR Location resource dict
        """
        ...

    def is_available(self) -> bool:
        """Check if the FHIR client is available and configured.

        Returns:
            True if client is ready for use
        """
        ...


def create_fhir_client() -> FHIRClientProtocol:
    """Factory function to create the appropriate FHIR client.

    Returns MockFHIRClient when FHIR_USE_MOCK is true, otherwise
    returns RealFHIRClient.

    Returns:
        A FHIR client implementing FHIRClientProtocol
    """
    config = get_config()

    if config.FHIR_USE_MOCK:
        from .mock_client import MockFHIRClient
        logger.info("Creating mock FHIR client")
        return MockFHIRClient()
    else:
        from .real_client import RealFHIRClient
        logger.info("Creating real FHIR client")
        return RealFHIRClient()
