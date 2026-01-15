"""
Integration tests for VectorStore - tests the real ChromaDB interaction.
These tests help diagnose the "query failed" issue by testing actual search behavior.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStoreSearch:
    """Integration tests for VectorStore.search() with real ChromaDB."""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create a temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def vector_store(self, temp_chroma_path):
        """Create a VectorStore instance with temp storage."""
        return VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

    @pytest.fixture
    def sample_course(self):
        """Create a sample course."""
        return Course(
            title="Python Programming 101",
            course_link="https://example.com/python101",
            instructor="John Doe",
            lessons=[
                Lesson(lesson_number=1, title="Introduction to Python", lesson_link="https://example.com/py/1"),
                Lesson(lesson_number=2, title="Variables and Types", lesson_link="https://example.com/py/2"),
                Lesson(lesson_number=3, title="Control Flow", lesson_link="https://example.com/py/3"),
            ]
        )

    @pytest.fixture
    def sample_chunks(self):
        """Create sample course chunks."""
        return [
            CourseChunk(
                content="Python is a high-level programming language known for its simplicity.",
                course_title="Python Programming 101",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Variables in Python are dynamically typed. You can store integers, strings, and floats.",
                course_title="Python Programming 101",
                lesson_number=2,
                chunk_index=1
            ),
            CourseChunk(
                content="Control flow in Python includes if statements, for loops, and while loops.",
                course_title="Python Programming 101",
                lesson_number=3,
                chunk_index=2
            ),
        ]

    def test_search_returns_results_when_content_exists(self, vector_store, sample_course, sample_chunks):
        """Search should return results when matching content exists."""
        # Add course metadata and content
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search for content
        results = vector_store.search(query="Python programming language")

        assert not results.is_empty(), "Search should return results"
        assert len(results.documents) > 0
        assert "Python" in results.documents[0]

    def test_search_returns_empty_when_no_content(self, vector_store):
        """Search should return empty results when no content exists."""
        results = vector_store.search(query="Python basics")

        # Should return empty, not error
        assert results.is_empty()
        assert results.error is None  # No error, just empty

    def test_search_with_course_filter(self, vector_store, sample_course, sample_chunks):
        """Search with course_name filter should only search within that course."""
        # Add course metadata and content
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search with course filter
        results = vector_store.search(
            query="programming",
            course_name="Python"  # Partial match
        )

        assert not results.is_empty()
        # All results should be from the Python course
        for meta in results.metadata:
            assert "Python" in meta.get("course_title", "")

    def test_search_with_nonexistent_course_returns_error(self, vector_store, sample_course, sample_chunks):
        """Search with nonexistent course_name should return error."""
        # Add course metadata and content
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search with invalid course filter
        results = vector_store.search(
            query="programming",
            course_name="NonexistentCourse123"
        )

        # Should return error about course not found
        assert results.error is not None
        assert "No course found" in results.error

    def test_search_with_lesson_filter(self, vector_store, sample_course, sample_chunks):
        """Search with lesson_number filter should only search within that lesson."""
        # Add course metadata and content
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search for lesson 2 content
        results = vector_store.search(
            query="variables",
            lesson_number=2
        )

        # Should find content from lesson 2
        assert not results.is_empty()
        for meta in results.metadata:
            assert meta.get("lesson_number") == 2

    def test_course_resolution_works(self, vector_store, sample_course):
        """Test that course name resolution via semantic search works."""
        vector_store.add_course_metadata(sample_course)

        # Try to resolve partial course name
        resolved = vector_store._resolve_course_name("Python")

        assert resolved == "Python Programming 101"

    def test_course_resolution_returns_none_for_unknown(self, vector_store):
        """Course resolution should return None for unknown courses."""
        resolved = vector_store._resolve_course_name("Completely Unknown Course")

        assert resolved is None

    def test_get_course_outline(self, vector_store, sample_course):
        """Test getting course outline with lessons."""
        vector_store.add_course_metadata(sample_course)

        outline = vector_store.get_course_outline("Python")

        assert outline is not None
        assert outline["course_title"] == "Python Programming 101"
        assert outline["course_link"] == "https://example.com/python101"
        assert len(outline["lessons"]) == 3
        assert outline["lessons"][0]["lesson_number"] == 1


class TestVectorStoreWithExistingDB:
    """Tests that check the actual production database state."""

    def test_production_db_has_courses(self):
        """Check if the production database has any courses loaded."""
        import os

        # Path to the actual production chroma_db
        prod_chroma_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "chroma_db"
        )

        if not os.path.exists(prod_chroma_path):
            pytest.skip("Production chroma_db not found")

        vector_store = VectorStore(
            chroma_path=prod_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        # Check if courses exist
        course_count = vector_store.get_course_count()
        course_titles = vector_store.get_existing_course_titles()

        print(f"\n=== Production Database State ===")
        print(f"Course count: {course_count}")
        print(f"Course titles: {course_titles}")

        # This test is informational - it helps diagnose if DB is empty
        assert course_count >= 0, "Should be able to count courses"

    def test_production_search_for_any_content(self):
        """Attempt a search in the production database."""
        import os

        prod_chroma_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "chroma_db"
        )

        if not os.path.exists(prod_chroma_path):
            pytest.skip("Production chroma_db not found")

        vector_store = VectorStore(
            chroma_path=prod_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        # Try a generic search
        results = vector_store.search(query="course content lesson")

        print(f"\n=== Production Search Results ===")
        print(f"Documents found: {len(results.documents)}")
        print(f"Error: {results.error}")
        if results.documents:
            print(f"First result preview: {results.documents[0][:200]}...")

        # If results are empty and no error, DB might be empty
        if results.is_empty() and not results.error:
            print("WARNING: Search returned no results - database may be empty!")


class TestSearchResultsDataclass:
    """Tests for SearchResults dataclass behavior."""

    def test_from_chroma_handles_empty_results(self):
        """SearchResults.from_chroma should handle empty ChromaDB results."""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.is_empty()
        assert results.error is None

    def test_from_chroma_handles_normal_results(self):
        """SearchResults.from_chroma should correctly parse ChromaDB results."""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course_title': 'A'}, {'course_title': 'B'}]],
            'distances': [[0.1, 0.2]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert not results.is_empty()
        assert len(results.documents) == 2
        assert results.documents[0] == 'doc1'

    def test_empty_creates_error_results(self):
        """SearchResults.empty should create results with error message."""
        results = SearchResults.empty("Test error message")

        assert results.is_empty()
        assert results.error == "Test error message"
