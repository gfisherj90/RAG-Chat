import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class MockContentBlock:
    """Mock for Anthropic API content blocks."""
    def __init__(self, block_type, text=None, tool_id=None, name=None, input_data=None):
        self.type = block_type
        if text is not None:
            self.text = text
        if tool_id is not None:
            self.id = tool_id
        if name is not None:
            self.name = name
        if input_data is not None:
            self.input = input_data


class MockResponse:
    """Mock for Anthropic API response."""
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class TestToolManagerReceivesCorrectToolName:
    """Tests verifying tool_manager receives correct tool names from AI."""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with mocked Anthropic client."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(api_key="test-key", model="test-model")
            return generator

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock VectorStore."""
        store = Mock()
        store.search = Mock(return_value=SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1]
        ))
        store.get_lesson_link = Mock(return_value=None)
        return store

    @pytest.fixture
    def real_tool_manager(self, mock_vector_store):
        """Create real ToolManager with mocked VectorStore."""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        return manager

    def test_tool_manager_receives_search_course_content_call(self, ai_generator, real_tool_manager, mock_vector_store):
        """When Claude requests search_course_content, ToolManager should execute it."""
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_123",
                name="search_course_content",
                input_data={"query": "Python basics"}
            )]
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Here is the answer")]
        )
        ai_generator.client.messages.create = Mock(side_effect=[tool_use_response, final_response])

        result = ai_generator.generate_response(
            query="Tell me about Python",
            tools=real_tool_manager.get_tool_definitions(),
            tool_manager=real_tool_manager
        )

        # Verify the vector store search was called (indicating tool executed)
        mock_vector_store.search.assert_called_once()

    def test_unknown_tool_name_handled_gracefully(self, ai_generator, real_tool_manager):
        """When Claude requests unknown tool, should handle gracefully."""
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_unknown",
                name="nonexistent_tool",
                input_data={"query": "test"}
            )]
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="I couldn't find that")]
        )
        ai_generator.client.messages.create = Mock(side_effect=[tool_use_response, final_response])

        # Should not raise exception
        result = ai_generator.generate_response(
            query="Test query",
            tools=real_tool_manager.get_tool_definitions(),
            tool_manager=real_tool_manager
        )

        assert result == "I couldn't find that"


class TestToolParametersPassedCorrectly:
    """Tests verifying tool parameters are passed correctly."""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with mocked Anthropic client."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(api_key="test-key", model="test-model")
            return generator

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock VectorStore."""
        store = Mock()
        store.search = Mock(return_value=SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1]
        ))
        store.get_lesson_link = Mock(return_value=None)
        return store

    @pytest.fixture
    def real_tool_manager(self, mock_vector_store):
        """Create real ToolManager with mocked VectorStore."""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        return manager

    def test_query_parameter_passed_to_tool(self, ai_generator, real_tool_manager, mock_vector_store):
        """Verify query parameter reaches the tool."""
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_1",
                name="search_course_content",
                input_data={"query": "machine learning algorithms"}
            )]
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Answer")]
        )
        ai_generator.client.messages.create = Mock(side_effect=[tool_use_response, final_response])

        ai_generator.generate_response(
            query="Tell me about ML",
            tools=real_tool_manager.get_tool_definitions(),
            tool_manager=real_tool_manager
        )

        mock_vector_store.search.assert_called_once_with(
            query="machine learning algorithms",
            course_name=None,
            lesson_number=None
        )

    def test_course_name_parameter_passed_to_tool(self, ai_generator, real_tool_manager, mock_vector_store):
        """Verify course_name parameter reaches the tool."""
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_1",
                name="search_course_content",
                input_data={"query": "introduction", "course_name": "MCP Course"}
            )]
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Answer")]
        )
        ai_generator.client.messages.create = Mock(side_effect=[tool_use_response, final_response])

        ai_generator.generate_response(
            query="Tell me about MCP intro",
            tools=real_tool_manager.get_tool_definitions(),
            tool_manager=real_tool_manager
        )

        mock_vector_store.search.assert_called_once_with(
            query="introduction",
            course_name="MCP Course",
            lesson_number=None
        )

    def test_lesson_number_parameter_passed_to_tool(self, ai_generator, real_tool_manager, mock_vector_store):
        """Verify lesson_number parameter reaches the tool."""
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_1",
                name="search_course_content",
                input_data={"query": "content", "lesson_number": 5}
            )]
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Answer")]
        )
        ai_generator.client.messages.create = Mock(side_effect=[tool_use_response, final_response])

        ai_generator.generate_response(
            query="What's in lesson 5",
            tools=real_tool_manager.get_tool_definitions(),
            tool_manager=real_tool_manager
        )

        mock_vector_store.search.assert_called_once_with(
            query="content",
            course_name=None,
            lesson_number=5
        )

    def test_all_parameters_passed_together(self, ai_generator, real_tool_manager, mock_vector_store):
        """Verify all parameters are passed when provided together."""
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_1",
                name="search_course_content",
                input_data={
                    "query": "advanced topics",
                    "course_name": "Python 101",
                    "lesson_number": 10
                }
            )]
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Answer")]
        )
        ai_generator.client.messages.create = Mock(side_effect=[tool_use_response, final_response])

        ai_generator.generate_response(
            query="What advanced topics in Python lesson 10",
            tools=real_tool_manager.get_tool_definitions(),
            tool_manager=real_tool_manager
        )

        mock_vector_store.search.assert_called_once_with(
            query="advanced topics",
            course_name="Python 101",
            lesson_number=10
        )


class TestToolResultIncludedInNextMessage:
    """Tests verifying tool results are included in subsequent API calls."""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with mocked Anthropic client."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(api_key="test-key", model="test-model")
            return generator

    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager."""
        manager = Mock()
        manager.execute_tool = Mock(return_value="[Python 101 - Lesson 1]\nPython is a programming language.")
        return manager

    def test_tool_result_sent_back_to_claude(self, ai_generator, mock_tool_manager):
        """Tool execution result should be sent in the next message to Claude."""
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_abc",
                name="search_course_content",
                input_data={"query": "Python"}
            )]
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Final answer")]
        )
        ai_generator.client.messages.create = Mock(side_effect=[tool_use_response, final_response])

        ai_generator.generate_response(
            query="Tell me about Python",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Check the second call included tool results
        second_call_kwargs = ai_generator.client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Last message should contain tool result
        last_message = messages[-1]
        assert last_message["role"] == "user"
        assert last_message["content"][0]["type"] == "tool_result"
        assert last_message["content"][0]["tool_use_id"] == "tool_abc"
        assert "Python is a programming language" in last_message["content"][0]["content"]

    def test_tool_error_sent_back_to_claude(self, ai_generator, mock_tool_manager):
        """Tool execution errors should be sent in the next message."""
        mock_tool_manager.execute_tool = Mock(side_effect=Exception("Database error"))

        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_err",
                name="search_course_content",
                input_data={"query": "test"}
            )]
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Error response")]
        )
        ai_generator.client.messages.create = Mock(side_effect=[tool_use_response, final_response])

        ai_generator.generate_response(
            query="Search query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        second_call_kwargs = ai_generator.client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        last_message = messages[-1]

        assert "Tool execution failed" in last_message["content"][0]["content"]
        assert "Database error" in last_message["content"][0]["content"]


class TestMultipleToolCallsHandled:
    """Tests for handling multiple sequential tool calls."""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with mocked Anthropic client."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(api_key="test-key", model="test-model")
            return generator

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock VectorStore."""
        store = Mock()
        store.search = Mock(return_value=SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1]
        ))
        store.get_lesson_link = Mock(return_value=None)
        store.get_course_outline = Mock(return_value={
            "course_title": "Test Course",
            "course_link": "https://example.com",
            "lessons": [{"lesson_number": 1, "lesson_title": "Intro"}]
        })
        return store

    @pytest.fixture
    def real_tool_manager(self, mock_vector_store):
        """Create real ToolManager with both tools."""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        return manager

    def test_two_different_tools_in_sequence(self, ai_generator, real_tool_manager, mock_vector_store):
        """Should handle two different tools called sequentially."""
        # First round: outline tool
        round1_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_1",
                name="get_course_outline",
                input_data={"course_name": "Test Course"}
            )]
        )
        # Second round: search tool
        round2_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_2",
                name="search_course_content",
                input_data={"query": "specific topic"}
            )]
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Combined answer")]
        )
        ai_generator.client.messages.create = Mock(
            side_effect=[round1_response, round2_response, final_response]
        )

        result = ai_generator.generate_response(
            query="Get outline and then search for topic",
            tools=real_tool_manager.get_tool_definitions(),
            tool_manager=real_tool_manager
        )

        assert result == "Combined answer"
        mock_vector_store.get_course_outline.assert_called_once()
        mock_vector_store.search.assert_called_once()

    def test_same_tool_called_twice(self, ai_generator, real_tool_manager, mock_vector_store):
        """Should handle same tool called twice in sequence."""
        tool_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_x",
                name="search_course_content",
                input_data={"query": "topic"}
            )]
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Final")]
        )
        ai_generator.client.messages.create = Mock(
            side_effect=[tool_response, tool_response, final_response]
        )

        ai_generator.generate_response(
            query="Search twice",
            tools=real_tool_manager.get_tool_definitions(),
            tool_manager=real_tool_manager
        )

        # Should have been called twice (once per round)
        assert mock_vector_store.search.call_count == 2
