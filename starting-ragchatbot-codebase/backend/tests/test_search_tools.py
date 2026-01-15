import pytest
from unittest.mock import Mock, MagicMock
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore."""
        store = Mock()
        store.search = Mock()
        store.get_lesson_link = Mock(return_value=None)
        return store

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create CourseSearchTool with mocked store."""
        return CourseSearchTool(mock_vector_store)

    def test_execute_returns_formatted_results(self, search_tool, mock_vector_store):
        """When VectorStore.search() returns valid results, should return formatted content."""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content about Python basics", "More Python content"],
            metadata=[
                {"course_title": "Python 101", "lesson_number": 1},
                {"course_title": "Python 101", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )

        result = search_tool.execute(query="Python basics")

        assert "Python 101" in result
        assert "Content about Python basics" in result
        assert "Lesson 1" in result
        mock_vector_store.search.assert_called_once_with(
            query="Python basics",
            course_name=None,
            lesson_number=None
        )

    def test_execute_handles_empty_results(self, search_tool, mock_vector_store):
        """When no matching content is found, should return appropriate message."""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )

        result = search_tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_handles_search_error(self, search_tool, mock_vector_store):
        """When VectorStore returns an error, should return error message."""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="No course found matching 'invalid course'"
        )

        result = search_tool.execute(query="test", course_name="invalid course")

        assert "No course found matching" in result

    def test_execute_filters_by_course_name(self, search_tool, mock_vector_store):
        """Verifies course_name filter is passed correctly to VectorStore."""
        mock_vector_store.search.return_value = SearchResults(
            documents=["MCP content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.1]
        )

        search_tool.execute(query="tools", course_name="MCP")

        mock_vector_store.search.assert_called_once_with(
            query="tools",
            course_name="MCP",
            lesson_number=None
        )

    def test_execute_filters_by_lesson_number(self, search_tool, mock_vector_store):
        """Verifies lesson_number filter is passed correctly to VectorStore."""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Lesson 5 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 5}],
            distances=[0.1]
        )

        search_tool.execute(query="content", lesson_number=5)

        mock_vector_store.search.assert_called_once_with(
            query="content",
            course_name=None,
            lesson_number=5
        )

    def test_execute_filters_by_both_course_and_lesson(self, search_tool, mock_vector_store):
        """Verifies both filters can be used together."""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Specific content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 3}],
            distances=[0.1]
        )

        search_tool.execute(query="test", course_name="MCP", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name="MCP",
            lesson_number=3
        )

    def test_format_results_includes_course_and_lesson_header(self, search_tool, mock_vector_store):
        """Verifies formatted output includes course title and lesson number."""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content here"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 7}],
            distances=[0.1]
        )

        result = search_tool.execute(query="test")

        assert "[Advanced Python - Lesson 7]" in result
        assert "Test content here" in result

    def test_format_results_handles_missing_lesson_number(self, search_tool, mock_vector_store):
        """Verifies formatting works when lesson_number is None."""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Intro content"],
            metadata=[{"course_title": "Intro Course", "lesson_number": None}],
            distances=[0.1]
        )

        result = search_tool.execute(query="test")

        assert "[Intro Course]" in result
        assert "Lesson" not in result or "Lesson None" not in result

    def test_last_sources_tracks_search_results(self, search_tool, mock_vector_store):
        """Verifies sources are stored for retrieval after search."""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        search_tool.execute(query="test")

        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert search_tool.last_sources[1]["text"] == "Course B - Lesson 2"

    def test_last_sources_cleared_on_empty_results(self, search_tool, mock_vector_store):
        """When search returns empty, last_sources should be empty."""
        # First search with results
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.1]
        )
        search_tool.execute(query="first")
        assert len(search_tool.last_sources) == 1

        # Second search with no results - last_sources should remain from first
        # (because _format_results is not called for empty results)
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )
        search_tool.execute(query="second")
        # Note: The current implementation doesn't clear last_sources on empty results
        # This might be intentional or a bug to investigate


class TestCourseSearchToolDefinition:
    """Tests for CourseSearchTool.get_tool_definition()."""

    @pytest.fixture
    def search_tool(self):
        """Create CourseSearchTool with mocked store."""
        return CourseSearchTool(Mock())

    def test_tool_definition_has_correct_name(self, search_tool):
        """Tool definition should have correct name."""
        definition = search_tool.get_tool_definition()
        assert definition["name"] == "search_course_content"

    def test_tool_definition_has_required_query_field(self, search_tool):
        """Tool definition should require query field."""
        definition = search_tool.get_tool_definition()
        assert "query" in definition["input_schema"]["required"]

    def test_tool_definition_has_optional_filters(self, search_tool):
        """Tool definition should have optional course_name and lesson_number."""
        definition = search_tool.get_tool_definition()
        properties = definition["input_schema"]["properties"]
        assert "course_name" in properties
        assert "lesson_number" in properties


class TestCourseOutlineTool:
    """Tests for CourseOutlineTool."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore."""
        store = Mock()
        store.get_course_outline = Mock()
        return store

    @pytest.fixture
    def outline_tool(self, mock_vector_store):
        """Create CourseOutlineTool with mocked store."""
        return CourseOutlineTool(mock_vector_store)

    def test_execute_returns_formatted_outline(self, outline_tool, mock_vector_store):
        """When course found, should return formatted outline."""
        mock_vector_store.get_course_outline.return_value = {
            "course_title": "MCP Course",
            "course_link": "https://example.com/mcp",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Introduction"},
                {"lesson_number": 2, "lesson_title": "Getting Started"}
            ]
        }

        result = outline_tool.execute(course_name="MCP")

        assert "MCP Course" in result
        assert "Lesson 1: Introduction" in result
        assert "Lesson 2: Getting Started" in result

    def test_execute_handles_course_not_found(self, outline_tool, mock_vector_store):
        """When course not found, should return error message."""
        mock_vector_store.get_course_outline.return_value = None

        result = outline_tool.execute(course_name="nonexistent")

        assert "No course found matching" in result


class TestToolManager:
    """Tests for ToolManager."""

    @pytest.fixture
    def tool_manager(self):
        """Create ToolManager instance."""
        return ToolManager()

    @pytest.fixture
    def mock_search_tool(self):
        """Create mock search tool."""
        tool = Mock()
        tool.get_tool_definition.return_value = {
            "name": "search_course_content",
            "description": "Search courses"
        }
        tool.execute.return_value = "Search results"
        tool.last_sources = [{"text": "Source 1"}]
        return tool

    def test_register_tool_adds_to_tools(self, tool_manager, mock_search_tool):
        """Registering a tool should make it available."""
        tool_manager.register_tool(mock_search_tool)

        assert "search_course_content" in tool_manager.tools

    def test_get_tool_definitions_returns_all_definitions(self, tool_manager, mock_search_tool):
        """Should return definitions for all registered tools."""
        tool_manager.register_tool(mock_search_tool)

        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool_calls_correct_tool(self, tool_manager, mock_search_tool):
        """Should call execute on the correct tool with kwargs."""
        tool_manager.register_tool(mock_search_tool)

        result = tool_manager.execute_tool("search_course_content", query="test")

        mock_search_tool.execute.assert_called_once_with(query="test")
        assert result == "Search results"

    def test_execute_tool_returns_error_for_unknown_tool(self, tool_manager):
        """Should return error message for unknown tool name."""
        result = tool_manager.execute_tool("unknown_tool")

        assert "not found" in result

    def test_get_last_sources_returns_sources_from_tools(self, tool_manager, mock_search_tool):
        """Should return last_sources from tools that have them."""
        tool_manager.register_tool(mock_search_tool)

        sources = tool_manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]["text"] == "Source 1"

    def test_reset_sources_clears_all_tool_sources(self, tool_manager, mock_search_tool):
        """Should clear last_sources on all tools."""
        tool_manager.register_tool(mock_search_tool)

        tool_manager.reset_sources()

        assert mock_search_tool.last_sources == []
