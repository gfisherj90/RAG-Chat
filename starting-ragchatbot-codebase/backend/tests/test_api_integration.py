"""
Integration tests for the API endpoints.
Tests the actual FastAPI app to diagnose "query failed" errors.
"""
import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient


class TestAPIQueryEndpoint:
    """Tests for the /api/query endpoint."""

    @pytest.fixture
    def mock_rag_system(self):
        """Create mock RAG system for API testing."""
        rag = Mock()
        rag.session_manager = Mock()
        rag.session_manager.create_session = Mock(return_value="test-session-123")
        rag.query = Mock(return_value=(
            "This is a test response",
            [{"text": "Source 1", "link": "https://example.com/1"}]
        ))
        return rag

    @pytest.fixture
    def client(self, mock_rag_system):
        """Create test client with mocked RAG system."""
        with patch('app.RAGSystem') as mock_rag_class, \
             patch('app.config'):
            mock_rag_class.return_value = mock_rag_system

            # Import app after patching
            from app import app
            return TestClient(app)

    def test_query_endpoint_returns_200_on_success(self, client, mock_rag_system):
        """Query endpoint should return 200 with valid response."""
        response = client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        print(f"\n=== API Response ===")
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_query_endpoint_handles_rag_exception(self, client, mock_rag_system):
        """Query endpoint should return 500 when RAG system throws."""
        mock_rag_system.query.side_effect = Exception("Database error")

        response = client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        print(f"\n=== API Error Response ===")
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")

        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]

    def test_query_endpoint_with_session_id(self, client, mock_rag_system):
        """Query endpoint should use provided session_id."""
        response = client.post(
            "/api/query",
            json={"query": "Follow up question", "session_id": "existing-session"}
        )

        assert response.status_code == 200
        mock_rag_system.query.assert_called_with("Follow up question", "existing-session")


class TestAPIWithRealRAGSystem:
    """Tests using the real RAG system (no mocking)."""

    @pytest.fixture
    def real_client(self):
        """Create test client with real RAG system."""
        from app import app
        return TestClient(app)

    def test_real_query_endpoint(self, real_client):
        """Test actual query endpoint with real system."""
        response = real_client.post(
            "/api/query",
            json={"query": "What courses are available?"}
        )

        print(f"\n=== Real API Response ===")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Answer preview: {data.get('answer', '')[:200]}...")
            print(f"Sources count: {len(data.get('sources', []))}")
            print(f"Session ID: {data.get('session_id')}")
        else:
            print(f"Error: {response.text}")

        # This test is diagnostic - captures actual behavior
        if response.status_code != 200:
            print(f"\n!!! FOUND THE BUG - Status {response.status_code} !!!")
            print(f"Error details: {response.text}")

    def test_courses_endpoint(self, real_client):
        """Test the courses endpoint."""
        response = real_client.get("/api/courses")

        print(f"\n=== Courses Endpoint ===")
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")

        assert response.status_code == 200


class TestSourceItemSerialization:
    """Tests for source item serialization in API responses."""

    def test_source_item_model_accepts_dict_with_text_and_link(self):
        """Verify SourceItem accepts dict format from RAG system."""
        from app import SourceItem

        # This is what RAG system returns
        source_dict = {"text": "Course A - Lesson 1", "link": "https://example.com"}

        # Should be able to create SourceItem from dict
        source = SourceItem(**source_dict)
        assert source.text == "Course A - Lesson 1"
        assert source.link == "https://example.com"

    def test_source_item_model_accepts_dict_without_link(self):
        """Verify SourceItem accepts dict without link."""
        from app import SourceItem

        source_dict = {"text": "Course A - Lesson 1"}

        source = SourceItem(**source_dict)
        assert source.text == "Course A - Lesson 1"
        assert source.link is None

    def test_query_response_model_serializes_sources(self):
        """Verify QueryResponse can serialize source dicts."""
        from app import QueryResponse, SourceItem

        # Create response with source dicts (as RAG system returns them)
        sources = [
            {"text": "Source 1", "link": "https://example.com/1"},
            {"text": "Source 2", "link": None}
        ]

        response = QueryResponse(
            answer="Test answer",
            sources=[SourceItem(**s) for s in sources],
            session_id="test-session"
        )

        # Verify serialization works
        response_dict = response.model_dump()
        assert len(response_dict["sources"]) == 2
        assert response_dict["sources"][0]["text"] == "Source 1"
