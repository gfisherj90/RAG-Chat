"""Shared test fixtures for RAG system tests."""

import pytest
from unittest.mock import Mock, MagicMock, patch


@pytest.fixture
def mock_rag_system():
    """Create a mock RAGSystem for API testing."""
    mock_system = Mock()
    mock_system.session_manager = Mock()
    mock_system.session_manager.create_session = Mock(return_value="test-session-123")
    mock_system.query = Mock(return_value=("Test answer", ["source1.pdf", "source2.pdf"]))
    mock_system.get_course_analytics = Mock(return_value={
        "total_courses": 3,
        "course_titles": ["Python Basics", "Data Science 101", "Machine Learning"]
    })
    return mock_system


@pytest.fixture
def sample_query_request():
    """Sample query request data."""
    return {
        "query": "What is machine learning?",
        "session_id": None
    }


@pytest.fixture
def sample_query_request_with_session():
    """Sample query request data with existing session."""
    return {
        "query": "Tell me more about neural networks",
        "session_id": "existing-session-456"
    }


@pytest.fixture
def mock_vector_store():
    """Create mock vector store for component tests."""
    mock_store = Mock()
    mock_store.get_course_count = Mock(return_value=5)
    mock_store.get_existing_course_titles = Mock(return_value=["Course A", "Course B"])
    mock_store.search = Mock(return_value=[])
    return mock_store


@pytest.fixture
def mock_ai_generator():
    """Create mock AI generator for component tests."""
    mock_generator = Mock()
    mock_generator.generate_response = Mock(return_value="Generated response")
    return mock_generator


@pytest.fixture
def mock_session_manager():
    """Create mock session manager for component tests."""
    mock_manager = Mock()
    mock_manager.create_session = Mock(return_value="new-session-id")
    mock_manager.get_conversation_history = Mock(return_value=[])
    mock_manager.add_exchange = Mock()
    return mock_manager
