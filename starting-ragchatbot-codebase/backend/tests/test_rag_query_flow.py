import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass


@dataclass
class MockConfig:
    """Mock configuration for RAGSystem."""
    ANTHROPIC_API_KEY: str = "test-key"
    ANTHROPIC_MODEL: str = "test-model"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"


class TestRAGSystemQueryFlow:
    """Tests for RAGSystem.query() method flow."""

    @pytest.fixture
    def mock_ai_generator(self):
        """Create mock AIGenerator."""
        generator = Mock()
        generator.generate_response = Mock(return_value="AI generated response")
        return generator

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock VectorStore."""
        store = Mock()
        store.search = Mock()
        store.get_lesson_link = Mock(return_value=None)
        store.get_course_outline = Mock()
        return store

    @pytest.fixture
    def mock_session_manager(self):
        """Create mock SessionManager."""
        manager = Mock()
        manager.get_conversation_history = Mock(return_value=None)
        manager.add_exchange = Mock()
        return manager

    @pytest.fixture
    def mock_document_processor(self):
        """Create mock DocumentProcessor."""
        return Mock()

    @pytest.fixture
    def rag_system(self, mock_ai_generator, mock_vector_store, mock_session_manager, mock_document_processor):
        """Create RAGSystem with mocked components."""
        with patch('rag_system.DocumentProcessor', return_value=mock_document_processor), \
             patch('rag_system.VectorStore', return_value=mock_vector_store), \
             patch('rag_system.AIGenerator', return_value=mock_ai_generator), \
             patch('rag_system.SessionManager', return_value=mock_session_manager):
            from rag_system import RAGSystem
            system = RAGSystem(MockConfig())
            return system

    def test_query_calls_ai_generator(self, rag_system, mock_ai_generator):
        """Query should call AIGenerator.generate_response."""
        rag_system.query("What is Python?")

        mock_ai_generator.generate_response.assert_called_once()

    def test_query_passes_tools_to_ai_generator(self, rag_system, mock_ai_generator):
        """Query should pass tool definitions to AIGenerator."""
        rag_system.query("Search for Python courses")

        call_kwargs = mock_ai_generator.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) == 2  # search + outline tools

    def test_query_passes_tool_manager_to_ai_generator(self, rag_system, mock_ai_generator):
        """Query should pass tool_manager to AIGenerator."""
        rag_system.query("Find course content")

        call_kwargs = mock_ai_generator.generate_response.call_args[1]
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is not None

    def test_query_returns_ai_response(self, rag_system, mock_ai_generator):
        """Query should return the AI-generated response."""
        mock_ai_generator.generate_response.return_value = "Test response"

        response, sources = rag_system.query("Test question")

        assert response == "Test response"

    def test_query_includes_session_history_when_provided(self, rag_system, mock_ai_generator, mock_session_manager):
        """When session_id provided, should include conversation history."""
        mock_session_manager.get_conversation_history.return_value = "Previous Q: What is X?\nPrevious A: X is..."

        rag_system.query("Follow up question", session_id="session123")

        call_kwargs = mock_ai_generator.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "Previous Q: What is X?\nPrevious A: X is..."

    def test_query_updates_session_history(self, rag_system, mock_ai_generator, mock_session_manager):
        """After query, should update session history."""
        mock_ai_generator.generate_response.return_value = "The answer"

        rag_system.query("My question", session_id="session456")

        mock_session_manager.add_exchange.assert_called_once_with(
            "session456",
            "My question",
            "The answer"
        )

    def test_query_returns_sources_from_tool_manager(self, rag_system):
        """Query should return sources captured from tool execution."""
        # Set up sources in the search tool
        rag_system.search_tool.last_sources = [
            {"text": "Course A - Lesson 1", "link": "https://example.com/1"},
            {"text": "Course B - Lesson 2", "link": "https://example.com/2"}
        ]

        response, sources = rag_system.query("Find content")

        assert len(sources) == 2
        assert sources[0]["text"] == "Course A - Lesson 1"

    def test_query_resets_sources_after_retrieval(self, rag_system):
        """After query, tool sources should be reset."""
        rag_system.search_tool.last_sources = [{"text": "Source"}]

        rag_system.query("Question")

        assert rag_system.search_tool.last_sources == []


class TestRAGSystemToolWiring:
    """Tests verifying tools are properly wired in RAGSystem."""

    @pytest.fixture
    def rag_system(self):
        """Create RAGSystem with mocked dependencies."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs_class, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            # Configure mock VectorStore instance
            mock_vs = Mock()
            mock_vs_class.return_value = mock_vs

            from rag_system import RAGSystem
            system = RAGSystem(MockConfig())
            return system

    def test_search_tool_registered(self, rag_system):
        """CourseSearchTool should be registered in tool_manager."""
        assert "search_course_content" in rag_system.tool_manager.tools

    def test_outline_tool_registered(self, rag_system):
        """CourseOutlineTool should be registered in tool_manager."""
        assert "get_course_outline" in rag_system.tool_manager.tools

    def test_tool_definitions_have_correct_structure(self, rag_system):
        """Tool definitions should have required fields."""
        definitions = rag_system.tool_manager.get_tool_definitions()

        for definition in definitions:
            assert "name" in definition
            assert "description" in definition
            assert "input_schema" in definition


class TestRAGSystemErrorHandling:
    """Tests for error handling in RAGSystem query flow."""

    @pytest.fixture
    def mock_ai_generator(self):
        """Create mock AIGenerator."""
        generator = Mock()
        return generator

    @pytest.fixture
    def rag_system(self, mock_ai_generator):
        """Create RAGSystem with mocked dependencies."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator', return_value=mock_ai_generator), \
             patch('rag_system.SessionManager'):
            from rag_system import RAGSystem
            system = RAGSystem(MockConfig())
            return system

    def test_query_handles_ai_generator_exception(self, rag_system, mock_ai_generator):
        """When AIGenerator raises exception, query should propagate it."""
        mock_ai_generator.generate_response.side_effect = Exception("API error")

        with pytest.raises(Exception) as exc_info:
            rag_system.query("Test query")

        assert "API error" in str(exc_info.value)

    def test_query_handles_empty_response(self, rag_system, mock_ai_generator):
        """When AIGenerator returns empty string, should handle gracefully."""
        mock_ai_generator.generate_response.return_value = ""

        response, sources = rag_system.query("Test")

        assert response == ""


class TestQueryPromptConstruction:
    """Tests for how query prompts are constructed."""

    @pytest.fixture
    def mock_ai_generator(self):
        """Create mock AIGenerator that captures calls."""
        generator = Mock()
        generator.generate_response = Mock(return_value="Response")
        return generator

    @pytest.fixture
    def rag_system(self, mock_ai_generator):
        """Create RAGSystem with mocked dependencies."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator', return_value=mock_ai_generator), \
             patch('rag_system.SessionManager'):
            from rag_system import RAGSystem
            system = RAGSystem(MockConfig())
            return system

    def test_query_includes_user_question_in_prompt(self, rag_system, mock_ai_generator):
        """User's question should be included in the prompt."""
        rag_system.query("What courses cover machine learning?")

        call_kwargs = mock_ai_generator.generate_response.call_args[1]
        prompt = call_kwargs["query"]
        assert "machine learning" in prompt

    def test_query_wraps_question_appropriately(self, rag_system, mock_ai_generator):
        """Query should wrap user question in appropriate prompt format."""
        rag_system.query("Test question here")

        call_kwargs = mock_ai_generator.generate_response.call_args[1]
        prompt = call_kwargs["query"]
        # The prompt should include context about course materials
        assert "course materials" in prompt.lower() or "Test question here" in prompt


class TestSourceRetrieval:
    """Tests for source retrieval after queries."""

    @pytest.fixture
    def rag_system(self):
        """Create RAGSystem with mocked dependencies."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs_class, \
             patch('rag_system.AIGenerator') as mock_ai_class, \
             patch('rag_system.SessionManager'):
            mock_vs = Mock()
            mock_vs.get_lesson_link = Mock(return_value="https://example.com/lesson")
            mock_vs_class.return_value = mock_vs

            mock_ai = Mock()
            mock_ai.generate_response = Mock(return_value="Answer")
            mock_ai_class.return_value = mock_ai

            from rag_system import RAGSystem
            system = RAGSystem(MockConfig())
            return system

    def test_sources_from_search_tool_returned(self, rag_system):
        """Sources from CourseSearchTool should be returned in query response."""
        rag_system.search_tool.last_sources = [
            {"text": "Python Course - Lesson 3", "link": "https://example.com/py3"}
        ]

        response, sources = rag_system.query("Python question")

        assert len(sources) == 1
        assert sources[0]["text"] == "Python Course - Lesson 3"
        assert sources[0]["link"] == "https://example.com/py3"

    def test_sources_from_outline_tool_returned(self, rag_system):
        """Sources from CourseOutlineTool should be returned in query response."""
        # Clear search tool sources, set outline tool sources
        rag_system.search_tool.last_sources = []
        rag_system.outline_tool.last_sources = [
            {"text": "MCP Course", "link": "https://example.com/mcp"}
        ]

        response, sources = rag_system.query("What's in MCP course?")

        assert len(sources) == 1
        assert sources[0]["text"] == "MCP Course"

    def test_empty_sources_when_no_tool_used(self, rag_system):
        """When no tool is used, sources should be empty."""
        rag_system.search_tool.last_sources = []
        rag_system.outline_tool.last_sources = []

        response, sources = rag_system.query("General question")

        assert sources == []
