import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.ai_generator import AIGenerator


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


@pytest.fixture
def ai_generator():
    """Create AIGenerator with mocked Anthropic client."""
    with patch('backend.ai_generator.anthropic.Anthropic'):
        generator = AIGenerator(api_key="test-key", model="test-model")
        return generator


@pytest.fixture
def mock_tool_manager():
    """Create mock tool manager."""
    manager = Mock()
    manager.execute_tool = Mock(return_value="Tool result content")
    return manager


@pytest.fixture
def sample_tools():
    """Sample tool definition."""
    return [{
        "name": "search_course_content",
        "description": "Search course content",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
    }]


class TestNoToolsProvided:
    """Tests when no tools are provided to generate_response."""

    def test_returns_text_response_directly(self, ai_generator):
        """When no tools provided, should make single API call and return text."""
        mock_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Direct answer")]
        )
        ai_generator.client.messages.create = Mock(return_value=mock_response)

        result = ai_generator.generate_response(query="What is Python?")

        assert result == "Direct answer"
        assert ai_generator.client.messages.create.call_count == 1


class TestToolProvidedButNotUsed:
    """Tests when tools are provided but Claude doesn't use them."""

    def test_returns_text_when_claude_skips_tool(self, ai_generator, sample_tools, mock_tool_manager):
        """When Claude responds with text only, should return without tool execution."""
        mock_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="I can answer without searching")]
        )
        ai_generator.client.messages.create = Mock(return_value=mock_response)

        result = ai_generator.generate_response(
            query="What is 2+2?",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        assert result == "I can answer without searching"
        assert ai_generator.client.messages.create.call_count == 1
        mock_tool_manager.execute_tool.assert_not_called()


class TestSingleToolCallRound:
    """Tests for queries requiring exactly one tool call."""

    def test_executes_tool_and_returns_final_response(self, ai_generator, sample_tools, mock_tool_manager):
        """Should execute tool once and return final text response."""
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
            content=[MockContentBlock("text", text="Based on the search, here is the answer")]
        )
        ai_generator.client.messages.create = Mock(side_effect=[tool_use_response, final_response])

        result = ai_generator.generate_response(
            query="Tell me about Python basics in the course",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        assert result == "Based on the search, here is the answer"
        assert ai_generator.client.messages.create.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )


class TestTwoSequentialToolCallRounds:
    """Tests for complex queries requiring two sequential tool calls."""

    def test_executes_two_tools_sequentially(self, ai_generator, sample_tools, mock_tool_manager):
        """Should support two rounds of tool calls before final response."""
        # First round: Claude requests first search
        round1_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_1",
                name="search_course_content",
                input_data={"query": "course X lesson 4"}
            )]
        )
        # Second round: Claude requests follow-up search
        round2_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_2",
                name="search_course_content",
                input_data={"query": "similar topic course"}
            )]
        )
        # Final response after both searches
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Complete answer from both searches")]
        )
        ai_generator.client.messages.create = Mock(
            side_effect=[round1_response, round2_response, final_response]
        )

        result = ai_generator.generate_response(
            query="Find a course with similar topic to lesson 4 of course X",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        assert result == "Complete answer from both searches"
        assert ai_generator.client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2


class TestMaxRoundsReached:
    """Tests for when maximum tool rounds are exhausted."""

    def test_forces_final_response_after_max_rounds(self, ai_generator, sample_tools, mock_tool_manager):
        """When max rounds reached, should make final call without tools."""
        # Both rounds request tools
        tool_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_n",
                name="search_course_content",
                input_data={"query": "search"}
            )]
        )
        forced_final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Forced final answer")]
        )
        ai_generator.client.messages.create = Mock(
            side_effect=[tool_response, tool_response, forced_final_response]
        )

        result = ai_generator.generate_response(
            query="Complex multi-step query",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        assert result == "Forced final answer"
        # 2 rounds with tools + 1 final without tools = 3 calls
        assert ai_generator.client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify final call was made without tools
        final_call_kwargs = ai_generator.client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call_kwargs


class TestToolExecutionFailure:
    """Tests for graceful handling of tool execution errors."""

    def test_continues_with_error_message_in_results(self, ai_generator, sample_tools, mock_tool_manager):
        """When tool execution raises exception, should include error and continue."""
        mock_tool_manager.execute_tool = Mock(side_effect=Exception("Database connection failed"))

        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_fail",
                name="search_course_content",
                input_data={"query": "test"}
            )]
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="I encountered an error but will respond")]
        )
        ai_generator.client.messages.create = Mock(side_effect=[tool_use_response, final_response])

        result = ai_generator.generate_response(
            query="Search for something",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        assert result == "I encountered an error but will respond"
        assert ai_generator.client.messages.create.call_count == 2

        # Verify error message was passed in tool results
        second_call_kwargs = ai_generator.client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        assert "Tool execution failed" in tool_result_msg["content"][0]["content"]


class TestNoToolManagerProvided:
    """Tests when tool_manager is None but tools are available."""

    def test_returns_text_when_no_tool_manager(self, ai_generator, sample_tools):
        """When tool_manager is None, should return any available text."""
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock("text", text="I would like to search"),
                MockContentBlock(
                    "tool_use",
                    tool_id="tool_x",
                    name="search_course_content",
                    input_data={"query": "test"}
                )
            ]
        )
        ai_generator.client.messages.create = Mock(return_value=tool_use_response)

        result = ai_generator.generate_response(
            query="Search for something",
            tools=sample_tools,
            tool_manager=None
        )

        assert result == "I would like to search"
        assert ai_generator.client.messages.create.call_count == 1


class TestMessageAccumulation:
    """Tests verifying correct message history across rounds."""

    def test_messages_accumulate_correctly(self, ai_generator, sample_tools, mock_tool_manager):
        """Verify message history is built correctly across rounds."""
        round1_response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="tool_1",
                name="search_course_content",
                input_data={"query": "first search"}
            )]
        )
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Final")]
        )
        ai_generator.client.messages.create = Mock(side_effect=[round1_response, final_response])

        ai_generator.generate_response(
            query="Test query",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Check second API call has accumulated messages
        second_call_kwargs = ai_generator.client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test query"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_1"


class TestExtractText:
    """Tests for the _extract_text helper method."""

    def test_extracts_text_from_response(self, ai_generator):
        """Should extract text from content blocks."""
        response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock("text", text="Extracted text")]
        )

        result = ai_generator._extract_text(response)

        assert result == "Extracted text"

    def test_returns_empty_string_when_no_text(self, ai_generator):
        """Should return empty string when no text block exists."""
        response = MockResponse(
            stop_reason="tool_use",
            content=[MockContentBlock(
                "tool_use",
                tool_id="t1",
                name="tool",
                input_data={}
            )]
        )

        result = ai_generator._extract_text(response)

        assert result == ""

    def test_extracts_first_text_from_mixed_content(self, ai_generator):
        """Should extract first text block from mixed content."""
        response = MockResponse(
            stop_reason="end_turn",
            content=[
                MockContentBlock("tool_use", tool_id="t1", name="tool", input_data={}),
                MockContentBlock("text", text="First text"),
                MockContentBlock("text", text="Second text")
            ]
        )

        result = ai_generator._extract_text(response)

        assert result == "First text"
