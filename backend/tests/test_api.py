"""API endpoint tests for the RAG system FastAPI application.

This module defines a test-specific FastAPI app to avoid import issues
with static file mounting in the main app.py.
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional
from unittest.mock import Mock, patch


class QueryRequest(BaseModel):
    """Request model for course queries."""
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries."""
    answer: str
    sources: List[str]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics."""
    total_courses: int
    course_titles: List[str]


def create_test_app(mock_rag_system):
    """Create a test FastAPI app with mocked RAG system."""
    app = FastAPI(title="Course Materials RAG System - Test")

    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"status": "ok", "message": "RAG System API"}

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources."""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics."""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


class TestRootEndpoint:
    """Tests for the root health check endpoint."""

    def test_root_returns_ok_status(self, mock_rag_system):
        """GET / should return health check response."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data


class TestQueryEndpoint:
    """Tests for the /api/query endpoint."""

    def test_query_without_session_creates_new_session(self, mock_rag_system, sample_query_request):
        """POST /api/query without session_id should create new session."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert data["answer"] == "Test answer"
        assert data["sources"] == ["source1.pdf", "source2.pdf"]
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_existing_session_uses_provided_session(
        self, mock_rag_system, sample_query_request_with_session
    ):
        """POST /api/query with session_id should use existing session."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post("/api/query", json=sample_query_request_with_session)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-456"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with(
            "Tell me more about neural networks",
            "existing-session-456"
        )

    def test_query_calls_rag_system_query(self, mock_rag_system, sample_query_request):
        """POST /api/query should call RAG system query method."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        client.post("/api/query", json=sample_query_request)

        mock_rag_system.query.assert_called_once()
        call_args = mock_rag_system.query.call_args
        assert call_args[0][0] == "What is machine learning?"

    def test_query_returns_422_for_missing_query_field(self, mock_rag_system):
        """POST /api/query without query field should return 422."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post("/api/query", json={"session_id": "test"})

        assert response.status_code == 422

    def test_query_returns_422_for_empty_body(self, mock_rag_system):
        """POST /api/query with empty body should return 422."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post("/api/query", json={})

        assert response.status_code == 422

    def test_query_returns_500_on_rag_system_error(self, mock_rag_system, sample_query_request):
        """POST /api/query should return 500 when RAG system raises exception."""
        mock_rag_system.query.side_effect = Exception("Database connection failed")
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


class TestCoursesEndpoint:
    """Tests for the /api/courses endpoint."""

    def test_courses_returns_course_statistics(self, mock_rag_system):
        """GET /api/courses should return course count and titles."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Python Basics" in data["course_titles"]

    def test_courses_calls_get_course_analytics(self, mock_rag_system):
        """GET /api/courses should call RAG system get_course_analytics."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        client.get("/api/courses")

        mock_rag_system.get_course_analytics.assert_called_once()

    def test_courses_returns_empty_list_when_no_courses(self, mock_rag_system):
        """GET /api/courses should handle empty course list."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_returns_500_on_analytics_error(self, mock_rag_system):
        """GET /api/courses should return 500 when analytics fails."""
        mock_rag_system.get_course_analytics.side_effect = Exception("Vector store unavailable")
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Vector store unavailable" in response.json()["detail"]


class TestRequestValidation:
    """Tests for request validation behavior."""

    def test_query_accepts_string_query(self, mock_rag_system):
        """Query endpoint should accept valid string query."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post("/api/query", json={"query": "test query"})

        assert response.status_code == 200

    def test_query_rejects_non_string_query(self, mock_rag_system):
        """Query endpoint should reject non-string query values."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post("/api/query", json={"query": 123})

        assert response.status_code == 422

    def test_query_accepts_null_session_id(self, mock_rag_system):
        """Query endpoint should accept null session_id."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post("/api/query", json={"query": "test", "session_id": None})

        assert response.status_code == 200


class TestResponseFormat:
    """Tests for response format compliance."""

    def test_query_response_contains_required_fields(self, mock_rag_system, sample_query_request):
        """Query response should contain answer, sources, and session_id."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.post("/api/query", json=sample_query_request)

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert isinstance(data["sources"], list)

    def test_courses_response_contains_required_fields(self, mock_rag_system):
        """Courses response should contain total_courses and course_titles."""
        app = create_test_app(mock_rag_system)
        client = TestClient(app)

        response = client.get("/api/courses")

        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
