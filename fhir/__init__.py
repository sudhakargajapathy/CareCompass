"""FHIR R4 Provider Directory integration for CareCompass.

This package provides FHIR-compliant healthcare provider data retrieval,
including a mock client for development/testing and a real client for
live payer endpoints.
"""

from .client import FHIRClientProtocol, create_fhir_client
from .transformer import FHIRToProviderTransformer

__all__ = [
    "FHIRClientProtocol",
    "create_fhir_client",
    "FHIRToProviderTransformer",
]
