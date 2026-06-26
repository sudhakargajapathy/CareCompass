"""Unit tests for utils/vector_store.py ChromaDB vector store operations."""
import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from utils.vector_store import ProviderVectorStore, get_vector_store


class TestVectorStoreInitialization:
    """Tests for ProviderVectorStore initialization."""

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_initialize_clients_success(self, mock_openai, mock_chroma, mock_env_vars):
        """Test successful client initialization."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        store = ProviderVectorStore()

        assert store.client is not None
        assert store.collection is not None
        assert store.openai_client is not None
        mock_chroma.assert_called_once()

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_initialize_missing_api_key(self, mock_openai, mock_chroma, mock_env_missing_openai):
        """Test initialization fails gracefully without OpenAI key."""
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            store = ProviderVectorStore()

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_initialize_chroma_failure(self, mock_openai, mock_chroma, mock_env_vars):
        """Test initialization handles ChromaDB failures."""
        mock_chroma.side_effect = Exception("ChromaDB connection failed")

        with pytest.raises(Exception, match="ChromaDB connection failed"):
            store = ProviderVectorStore()


class TestCreateProviderText:
    """Tests for _create_provider_text method."""

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_create_provider_text_complete_data(self, mock_openai, mock_chroma, mock_env_vars, sample_providers):
        """Test provider text creation with complete data."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        store = ProviderVectorStore()
        provider = sample_providers[0]
        text = store._create_provider_text(provider)

        assert "Provider: Dr. Sarah Johnson" in text
        assert "Specialty: Neurology" in text
        assert "Location: Phoenix, AZ" in text
        assert "Phone: (602) 555-1234" in text
        assert "Insurance:" in text
        assert "Services:" in text
        assert "|" in text  # Check separator is used

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_create_provider_text_minimal_data(self, mock_openai, mock_chroma, mock_env_vars):
        """Test provider text creation with minimal data."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        store = ProviderVectorStore()
        provider = {"name": "Dr. Test", "specialty": "Test Specialty"}
        text = store._create_provider_text(provider)

        assert "Provider: Dr. Test" in text
        assert "Specialty: Test Specialty" in text
        assert "Phone:" not in text
        assert "Insurance:" not in text

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_create_provider_text_empty_provider(self, mock_openai, mock_chroma, mock_env_vars):
        """Test provider text creation with empty provider data."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        store = ProviderVectorStore()
        text = store._create_provider_text({})

        assert text == ""


class TestGetEmbedding:
    """Tests for _get_embedding method."""

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_get_embedding_success(self, mock_openai, mock_chroma, mock_env_vars):
        """Test successful embedding generation."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock OpenAI response
        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding_obj]
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = mock_response

        store = ProviderVectorStore()
        embedding = store._get_embedding("test text")

        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
        mock_openai_instance.embeddings.create.assert_called_once()

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_get_embedding_api_failure(self, mock_openai, mock_chroma, mock_env_vars):
        """Test embedding generation handles API failures."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.side_effect = Exception("API rate limit exceeded")

        store = ProviderVectorStore()

        with pytest.raises(Exception, match="API rate limit exceeded"):
            store._get_embedding("test text")


class TestAddProviders:
    """Tests for add_providers method."""

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_add_providers_single(self, mock_openai, mock_chroma, mock_env_vars, sample_providers):
        """Test adding a single provider."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock OpenAI embedding
        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding_obj]
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = mock_response

        store = ProviderVectorStore()
        result = store.add_providers([sample_providers[0]])

        assert result is True
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert len(call_args["documents"]) == 1
        assert len(call_args["metadatas"]) == 1
        assert len(call_args["ids"]) == 1
        assert len(call_args["embeddings"]) == 1

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_add_providers_batch(self, mock_openai, mock_chroma, mock_env_vars, sample_providers):
        """Test adding multiple providers in batch."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock OpenAI embedding
        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding_obj]
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = mock_response

        store = ProviderVectorStore()
        result = store.add_providers(sample_providers)

        assert result is True
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert len(call_args["documents"]) == len(sample_providers)
        assert len(call_args["metadatas"]) == len(sample_providers)

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_add_providers_empty_list(self, mock_openai, mock_chroma, mock_env_vars):
        """Test adding empty provider list."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        store = ProviderVectorStore()
        result = store.add_providers([])

        assert result is True
        mock_collection.add.assert_not_called()

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_add_providers_metadata_format(self, mock_openai, mock_chroma, mock_env_vars, sample_providers):
        """Test metadata is properly formatted as strings."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding_obj]
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = mock_response

        store = ProviderVectorStore()
        store.add_providers([sample_providers[0]])

        call_args = mock_collection.add.call_args[1]
        metadata = call_args["metadatas"][0]

        # Check all metadata values are strings
        assert isinstance(metadata["name"], str)
        assert isinstance(metadata["specialty"], str)
        assert isinstance(metadata["rating"], str)
        assert isinstance(metadata["raw_data"], str)

        # Check raw_data is valid JSON
        provider_data = json.loads(metadata["raw_data"])
        assert provider_data["name"] == sample_providers[0]["name"]

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_add_providers_failure_handling(self, mock_openai, mock_chroma, mock_env_vars, sample_providers):
        """Test add_providers handles failures gracefully."""
        mock_collection = Mock()
        mock_collection.add.side_effect = Exception("Database write error")
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding_obj]
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = mock_response

        store = ProviderVectorStore()
        result = store.add_providers([sample_providers[0]])

        assert result is False


class TestSearchProviders:
    """Tests for search_providers method."""

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_search_providers_basic(self, mock_openai, mock_chroma, mock_env_vars, sample_providers):
        """Test basic semantic search."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock search results
        mock_collection.query.return_value = {
            "ids": [[f"provider_0_{hash(sample_providers[0]['name'])}"]],
            "metadatas": [[{"raw_data": json.dumps(sample_providers[0])}]],
            "distances": [[0.2]]
        }

        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding_obj]
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = mock_response

        store = ProviderVectorStore()
        results = store.search_providers("neurologist in Phoenix")

        assert len(results) == 1
        assert results[0]["name"] == sample_providers[0]["name"]
        assert "similarity_score" in results[0]
        assert "search_rank" in results[0]
        assert results[0]["search_rank"] == 1

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_search_providers_with_filters(self, mock_openai, mock_chroma, mock_env_vars, sample_providers):
        """Test search with specialty filter."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_collection.query.return_value = {
            "ids": [[f"provider_0_{hash(sample_providers[0]['name'])}"]],
            "metadatas": [[{"raw_data": json.dumps(sample_providers[0])}]],
            "distances": [[0.15]]
        }

        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding_obj]
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = mock_response

        store = ProviderVectorStore()
        results = store.search_providers(
            query="doctor",
            specialty="Neurology",
            location="Phoenix, AZ"
        )

        # Verify where clause was used
        call_args = mock_collection.query.call_args[1]
        assert "where" in call_args
        assert call_args["where"] is not None

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_search_providers_no_results(self, mock_openai, mock_chroma, mock_env_vars):
        """Test search returns empty list when no matches."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_collection.query.return_value = {
            "ids": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding_obj]
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = mock_response

        store = ProviderVectorStore()
        results = store.search_providers("unknown specialty")

        assert results == []

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_search_providers_similarity_calculation(self, mock_openai, mock_chroma, mock_env_vars, sample_providers):
        """Test similarity score is correctly calculated from distance."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_collection.query.return_value = {
            "ids": [[f"provider_0_{hash(sample_providers[0]['name'])}"]],
            "metadatas": [[{"raw_data": json.dumps(sample_providers[0])}]],
            "distances": [[0.3]]  # Distance 0.3 should give similarity 0.7
        }

        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding_obj]
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = mock_response

        store = ProviderVectorStore()
        results = store.search_providers("test")

        assert results[0]["similarity_score"] == pytest.approx(0.7)


class TestFilterByMetadata:
    """Tests for filter_by_metadata method."""

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_filter_by_metadata_single_field(self, mock_openai, mock_chroma, mock_env_vars, sample_providers):
        """Test filtering by single metadata field."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_collection.query.return_value = {
            "ids": [[f"provider_0_{hash(sample_providers[0]['name'])}"]],
            "metadatas": [[{"raw_data": json.dumps(sample_providers[0])}]]
        }

        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding_obj]
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = mock_response

        store = ProviderVectorStore()
        results = store.filter_by_metadata({"specialty": "Neurology"})

        assert len(results) == 1
        assert results[0]["name"] == sample_providers[0]["name"]

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_filter_by_metadata_empty_filters(self, mock_openai, mock_chroma, mock_env_vars):
        """Test filtering with empty filter dictionary."""
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_collection.query.return_value = {
            "ids": [[]],
            "metadatas": [[]]
        }

        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding_obj]
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = mock_response

        store = ProviderVectorStore()
        results = store.filter_by_metadata({})

        assert results == []


class TestCollectionManagement:
    """Tests for collection management methods."""

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_get_collection_stats(self, mock_openai, mock_chroma, mock_env_vars):
        """Test getting collection statistics."""
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        store = ProviderVectorStore()
        stats = store.get_collection_stats()

        assert stats["total_providers"] == 42
        assert "collection_name" in stats

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_clear_collection(self, mock_openai, mock_chroma, mock_env_vars):
        """Test clearing collection."""
        mock_collection = Mock()
        mock_client = mock_chroma.return_value
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.create_collection.return_value = mock_collection

        store = ProviderVectorStore()
        result = store.clear_collection()

        assert result is True
        mock_client.delete_collection.assert_called_once()
        mock_client.create_collection.assert_called_once()

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_clear_collection_failure(self, mock_openai, mock_chroma, mock_env_vars):
        """Test clear_collection handles failures."""
        mock_collection = Mock()
        mock_client = mock_chroma.return_value
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.delete_collection.side_effect = Exception("Delete failed")

        store = ProviderVectorStore()
        result = store.clear_collection()

        assert result is False


class TestSingletonPattern:
    """Tests for singleton pattern implementation."""

    @patch('utils.vector_store.chromadb.PersistentClient')
    @patch('utils.vector_store.OpenAI')
    def test_get_vector_store_singleton(self, mock_openai, mock_chroma, mock_env_vars):
        """Test get_vector_store returns singleton instance."""
        # Reset singleton
        import utils.vector_store
        utils.vector_store._vector_store_instance = None

        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        store1 = get_vector_store()
        store2 = get_vector_store()

        assert store1 is store2
        mock_chroma.assert_called_once()  # Should only initialize once
