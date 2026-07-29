"""ChromaDB vector store operations for healthcare provider data."""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import json

from .config import get_config
from .cost_tracker import get_cost_tracker, safe_usage
from .encryption import get_encryptor
from .provider_key import provider_cache_key, resolve_cache_key

logger = logging.getLogger(__name__)

# Bumped whenever the stored record shape changes incompatibly. Records written
# under an older version are ignored on read (and cleared on first use), because
# a cache that silently serves a stale SHAPE is worse than a cold start.
CACHE_SCHEMA_VERSION = "2"

# Enrichment-derived fields — the expensive half of a search, and the only
# thing worth storing. Everything else is either cheap to recompute or depends
# on the CURRENT search (see _CACHE_EXCLUDED below).
CACHEABLE_FIELDS = (
    "review_observations",
    "review_summary",
    "review_sentiment",
    "review_source_url",
    "insurance_accepted",
    "insurance_source_url",
    "years_experience",
    "location",
)

# Evidence that only enrichment can produce. `location` is excluded on purpose:
# discovery already supplies it, so it cannot serve as proof that a search
# learned anything. See `cacheable_payload`.
SUBSTANTIVE_CACHE_FIELDS = tuple(f for f in CACHEABLE_FIELDS if f != "location")

# Values the extractor writes when it found NOTHING. They are indistinguishable
# from evidence by an emptiness test, which is how they defeated the guard.
_PLACEHOLDER_VALUES = frozenset(
    {"no reviews available", "unknown", "n/a", "na", "none", "not available"}
)


def _is_placeholder(value: Any) -> bool:
    """True for a stand-in the extractor emits in place of missing data."""
    return (
        isinstance(value, str)
        and value.strip().lower() in _PLACEHOLDER_VALUES
    )

# Never cached, and asserted in tests. Distance depends on the USER's location:
# a Chandler distance restored into a Phoenix search would be wrong AND would
# read as measured rather than imputed. Scores depend on per-search weights.
_CACHE_EXCLUDED = (
    "computed_distance_miles",
    "location_match",
    "location_evidence",
    "distance",
    "base_score",
    "final_score",
    "refined_score",
    "ai_score",
    "rank",
)


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

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get OpenAI embeddings for many texts in a single API call.

        Args:
            texts: Texts to embed

        Returns:
            One embedding per input text, in input order
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.config.EMBEDDING_MODEL,
                input=texts
            )
            tokens, _ = safe_usage(response)
            get_cost_tracker().record_embeddings(tokens, model=self.config.EMBEDDING_MODEL)
            # response.data is index-ordered to match the input list
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Failed to get batch embeddings: {e}")
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

            for i, provider in enumerate(providers):
                # Create searchable text
                doc_text = self._create_provider_text(provider)
                documents.append(doc_text)

                # Encrypt sensitive provider data before storing
                encrypted_provider = self.encryptor.encrypt_data(provider)

                # Create metadata (ChromaDB requires string values)
                # Store non-sensitive fields in plain text for searching
                # Store full provider data encrypted
                metadata = {
                    "provider_key": provider_cache_key(
                        provider.get("name"), provider.get("location")
                    ),
                    "name": str(provider.get("name", "")),
                    "specialty": str(provider.get("specialty", "")),
                    "location": str(provider.get("location", "")),
                    "phone_encrypted": "yes",  # Indicator that phone is encrypted
                    "rating": str(provider.get("rating", "0")),
                    "distance": str(provider.get("distance", "0")),
                    "raw_data_encrypted": encrypted_provider  # Full provider data, encrypted
                }
                metadatas.append(metadata)

                # Deterministic ID. The previous scheme was
                # f"provider_{i}_{hash(name)}" — Python salts str hashes per
                # process, and `i` is the list index, so the same provider got
                # a fresh ID on every restart AND at every rank. Nothing could
                # ever be looked up, and the collection grew duplicates without
                # bound.
                ids.append(metadata["provider_key"])

            # One embeddings request for all providers instead of one per provider
            embeddings = self._get_embeddings_batch(documents)

            # upsert, not add: re-encountering a provider must REPLACE its row.
            self.collection.upsert(
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
                        provider_data = self.encryptor.decrypt_data(metadata["raw_data_encrypted"])
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
                        provider_data = self.encryptor.decrypt_data(metadata["raw_data_encrypted"])
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

    # ---- enrichment cache -------------------------------------------------
    #
    # A KEYED lookup, deliberately not a similarity search. Two neurologists in
    # one city produce near-identical embeddings — same specialty, same town,
    # similar profile text — so cosine nearest-neighbour would happily return
    # one physician's enriched reviews for another, and the resulting card would
    # look complete and well-evidenced while being wrong. That is the exact
    # failure `_observation_is_same_person` and `_name_token_overlap` exist to
    # prevent; re-introducing it here would undo them one layer down, where
    # nothing is watching. The embedding stays on the record for a possible
    # "similar providers" feature — it just never decides identity.

    @staticmethod
    def cacheable_payload(provider: Dict[str, Any]) -> Dict[str, Any]:
        """The enrichment-derived subset worth storing, and nothing else.

        Returns {} when nothing SUBSTANTIVE was learned. `location` alone does
        not count: discovery already supplies it, so a provider whose
        enrichment found nothing would otherwise produce a truthy payload, get
        stored, and be served as a fresh hit for the whole TTL — the failed
        lookup masquerading as cached evidence.

        Neither do the extractor's PLACEHOLDERS. `_extract_provider_data` sets
        `review_summary = "No reviews available"` and `review_sentiment =
        "unknown"` on every provider whether or not anything was found, and
        both fields are substantive — so the emptiness test alone admitted
        them and the guard never fired for the case it exists to catch. A
        provider whose enrichment found nothing was cached, then served for
        the full TTL and excluded from every live retry, and the placeholder
        could overwrite a real summary a later candidate pass had found.
        """
        payload = {
            field: provider[field]
            for field in CACHEABLE_FIELDS
            if provider.get(field) not in (None, "", [], {})
            and not _is_placeholder(provider.get(field))
        }
        if not any(field in payload for field in SUBSTANTIVE_CACHE_FIELDS):
            return {}
        return payload

    def get_cached_providers(
        self,
        providers: List[Dict[str, Any]],
        ttl_days: Optional[float] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """Fetch fresh cached enrichment for the given providers, by key.

        One `.get` for the whole pool: no embedding call, no network request.

        Returns (payload_by_key, stale_keys) — stale keys are reported so the
        caller can log them; they are treated exactly like misses.
        """
        if not providers:
            return {}, []

        ttl = self.config.PROVIDER_CACHE_TTL_DAYS if ttl_days is None else ttl_days
        keys = [
            resolve_cache_key(p) for p in providers
        ]

        try:
            result = self.collection.get(
                where={"provider_key": {"$in": keys}},
                include=["metadatas"],
            )
        except Exception as e:
            # A cache that cannot be read is a cold start, never an error.
            logger.warning(f"Cache lookup failed, treating as all-miss: {e}")
            return {}, []

        now = time.time()
        cutoff = ttl * 86400.0
        fresh: Dict[str, Dict[str, Any]] = {}
        stale: List[str] = []

        for metadata in (result.get("metadatas") or []):
            if not metadata:
                continue
            key = metadata.get("provider_key")
            if not key:
                continue

            if str(metadata.get("schema_version", "")) != CACHE_SCHEMA_VERSION:
                stale.append(key)
                continue

            try:
                age = now - float(metadata.get("enriched_at_epoch", 0) or 0)
            except (TypeError, ValueError):
                stale.append(key)
                continue

            # Freshness is compared HERE rather than as a Chroma numeric filter
            # so the rule is unit-testable without a database.
            if age >= cutoff:
                stale.append(key)
                continue

            payload = self.encryptor.decrypt_data(metadata.get("raw_data_encrypted"))
            if isinstance(payload, dict) and payload:
                payload["cached_enriched_at"] = metadata.get("enriched_at_iso", "")
                fresh[key] = payload

        return fresh, stale

    def upsert_enriched_providers(self, providers: List[Dict[str, Any]]) -> int:
        """Store enrichment results under deterministic keys. Returns count written.

        Only providers carrying real enrichment are written — storing an empty
        payload would let a failed lookup masquerade as a fresh cache hit for
        the next `PROVIDER_CACHE_TTL_DAYS`, which is strictly worse than a miss.
        """
        writable = [
            (p, self.cacheable_payload(p))
            for p in (providers or [])
            if p.get("name")
        ]
        writable = [(p, payload) for p, payload in writable if payload]
        if not writable:
            return 0

        try:
            now = time.time()
            stamped_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            ids, documents, metadatas = [], [], []
            for provider, payload in writable:
                key = resolve_cache_key(provider)
                ids.append(key)
                documents.append(self._create_provider_text(provider))
                metadatas.append({
                    "provider_key": key,
                    "name": str(provider.get("name", "")),
                    "specialty": str(provider.get("specialty", "")),
                    "location": str(provider.get("location", "")),
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "enriched_at_epoch": now,
                    "enriched_at_iso": stamped_iso,
                    "raw_data_encrypted": self.encryptor.encrypt_data(payload),
                })

            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=self._get_embeddings_batch(documents),
            )
            logger.info(f"Cached enrichment for {len(ids)} providers")
            return len(ids)

        except Exception as e:
            # Failing to WRITE the cache must never fail the search.
            logger.warning(f"Failed to cache enriched providers: {e}")
            return 0

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