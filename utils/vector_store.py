"""ChromaDB vector store operations for healthcare provider data."""

import logging
from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import json

from .config import get_config
from .encryption import get_encryptor

logger = logging.getLogger(__name__)


class ProviderVectorStore:
    """ChromaDB vector store for healthcare provider data with semantic search capabilities."""

    def __init__(self):
        """Initialize the vector store with persistent ChromaDB client and OpenAI embeddings."""
        self.config = get_config()
        self.client = None
        self.collection = None
        self.openai_client = None
        self.encryptor = get_encryptor()
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize ChromaDB and OpenAI clients."""
        try:
            # Initialize ChromaDB with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.config.CHROMA_PERSIST_DIRECTORY,
                settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.CHROMA_COLLECTION_NAME,
                metadata={"description": "Healthcare provider profiles for semantic matching"}
            )

            # Initialize OpenAI client for embeddings
            if self.config.OPENAI_API_KEY:
                self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            else:
                raise ValueError("OpenAI API key not found in configuration")

            logger.info(f"Vector store initialized with collection: {self.config.CHROMA_COLLECTION_NAME}")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    def _create_provider_text(self, provider: Dict[str, Any]) -> str:
        """Create searchable text representation of provider.

        Args:
            provider: Provider data dictionary

        Returns:
            Combined text for embedding
        """
        text_parts = []

        # Basic info
        if provider.get("name"):
            text_parts.append(f"Provider: {provider['name']}")

        if provider.get("specialty"):
            text_parts.append(f"Specialty: {provider['specialty']}")

        if provider.get("location"):
            text_parts.append(f"Location: {provider['location']}")

        # Additional details
        if provider.get("phone"):
            text_parts.append(f"Phone: {provider['phone']}")

        if provider.get("insurance_accepted"):
            insurances = ", ".join(provider["insurance_accepted"])
            text_parts.append(f"Insurance: {insurances}")

        if provider.get("services"):
            services = ", ".join(provider["services"])
            text_parts.append(f"Services: {services}")

        return " | ".join(text_parts)

    def add_providers(self, providers: List[Dict[str, Any]]) -> bool:
        """Add provider data to the vector store.

        Args:
            providers: List of provider dictionaries

        Returns:
            bool: Success status
        """
        try:
            if not providers:
                logger.warning("No providers to add")
                return True

            documents = []
            metadatas = []
            ids = []
            embeddings = []

            for i, provider in enumerate(providers):
                # Create searchable text
                doc_text = self._create_provider_text(provider)
                documents.append(doc_text)

                # Encrypt sensitive provider data before storing
                encrypted_provider, key_id = self.encryptor.encrypt_data_with_key_id(provider)

                # Create metadata (ChromaDB requires string values)
                # Store non-sensitive fields in plain text for searching
                # Store full provider data encrypted
                metadata = {
                    "name": str(provider.get("name", "")),
                    "specialty": str(provider.get("specialty", "")),
                    "location": str(provider.get("location", "")),
                    "phone_encrypted": "yes",  # Indicator that phone is encrypted
                    "rating": str(provider.get("rating", "0")),
                    "distance": str(provider.get("distance", "0")),
                    "raw_data_encrypted": encrypted_provider,  # Store encrypted full provider data
                    "raw_data_key_id": key_id
                }
                metadatas.append(metadata)

                # Create unique ID
                provider_id = f"provider_{i}_{hash(provider.get('name', ''))}"
                ids.append(provider_id)

                # Get embedding
                embedding = self._get_embedding(doc_text)
                embeddings.append(embedding)

            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )

            logger.info(f"Added {len(providers)} providers to vector store")
            return True

        except Exception as e:
            logger.error(f"Failed to add providers: {e}")
            return False

    def search_providers(
        self,
        query: str,
        specialty: Optional[str] = None,
        location: Optional[str] = None,
        insurance: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for providers using semantic similarity.

        Args:
            query: Search query text
            specialty: Filter by specialty
            location: Filter by location
            insurance: Filter by insurance accepted
            max_results: Maximum number of results

        Returns:
            List of matching provider dictionaries with similarity scores
        """
        try:
            # Create search query
            search_text = query
            if specialty:
                search_text += f" {specialty}"
            if location:
                search_text += f" {location}"
            if insurance:
                search_text += f" {insurance}"

            # Get query embedding
            query_embedding = self._get_embedding(search_text)

            # Build where clause for filtering
            where_clause = {}
            if specialty:
                where_clause["specialty"] = {"$contains": specialty}

            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )

            # Process results
            providers = []
            if results["ids"] and results["ids"][0]:
                for i, provider_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]

                    # Decrypt provider data
                    if "raw_data_encrypted" in metadata:
                        provider_data = self.encryptor.decrypt_data(metadata["raw_data_encrypted"], metadata.get("raw_data_key_id"))
                    else:
                        # Fallback for unencrypted data (backwards compatibility)
                        provider_data = json.loads(metadata.get("raw_data", "{}"))

                    if provider_data:
                        provider_data["similarity_score"] = 1 - distance  # Convert distance to similarity
                        provider_data["search_rank"] = i + 1
                        providers.append(provider_data)

            logger.info(f"Found {len(providers)} providers for query: {query}")
            return providers

        except Exception as e:
            logger.error(f"Failed to search providers: {e}")
            return []

    def filter_by_metadata(
        self,
        filters: Dict[str, str],
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Filter providers by metadata criteria.

        Args:
            filters: Dictionary of field-value pairs to filter by
            max_results: Maximum number of results

        Returns:
            List of matching provider dictionaries
        """
        try:
            # Build where clause
            where_clause = {}
            for field, value in filters.items():
                if value and value.strip():
                    where_clause[field] = {"$contains": value.strip()}

            # Query collection
            results = self.collection.query(
                query_embeddings=[self._get_embedding("healthcare provider")],  # Dummy embedding
                n_results=max_results,
                where=where_clause if where_clause else None,
                include=["metadatas"]
            )

            # Process results
            providers = []
            if results["ids"] and results["ids"][0]:
                for metadata in results["metadatas"][0]:
                    # Decrypt provider data
                    if "raw_data_encrypted" in metadata:
                        provider_data = self.encryptor.decrypt_data(metadata["raw_data_encrypted"], metadata.get("raw_data_key_id"))
                    else:
                        # Fallback for unencrypted data (backwards compatibility)
                        provider_data = json.loads(metadata.get("raw_data", "{}"))

                    if provider_data:
                        providers.append(provider_data)

            logger.info(f"Filtered {len(providers)} providers with criteria: {filters}")
            return providers

        except Exception as e:
            logger.error(f"Failed to filter providers: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about the provider collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "total_providers": count,
                "collection_name": self.config.CHROMA_COLLECTION_NAME
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"total_providers": 0, "collection_name": "unknown"}

    def clear_collection(self) -> bool:
        """Clear all data from the collection.

        Returns:
            bool: Success status
        """
        try:
            # Delete and recreate collection
            self.client.delete_collection(name=self.config.CHROMA_COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=self.config.CHROMA_COLLECTION_NAME,
                metadata={"description": "Healthcare provider profiles for semantic matching"}
            )
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False


# Singleton instance
_vector_store_instance: Optional[ProviderVectorStore] = None


def get_vector_store() -> ProviderVectorStore:
    """Get singleton vector store instance.

    Returns:
        ProviderVectorStore instance
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = ProviderVectorStore()
    return _vector_store_instance