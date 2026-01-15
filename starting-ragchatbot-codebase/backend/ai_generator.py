import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Maximum number of sequential tool call rounds per query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search within course lesson content for specific topics or details
2. **get_course_outline**: Get the complete structure of a course including course title, course link, and all lessons with their numbers and titles

Tool Selection:
- Use **get_course_outline** for:
  - Questions about course structure, syllabus, or outline
  - "What lessons are in [course]?"
  - "What is the outline of [course]?"
  - "What topics does [course] cover?"
  - When you need the course link or lesson list
  - Always include the course title, course link, and full lesson list (number and title) in your response

- Use **search_course_content** for:
  - Questions about specific content within lessons
  - "What does [course] say about [topic]?"
  - Detailed information from course materials

Sequential Tool Usage:
- You may use up to 2 tool calls per query when needed
- For multi-step questions, use tools sequentially:
  1. First call: gather initial information (e.g., get course outline to find lesson titles)
  2. Second call: use that information for a refined search (e.g., search for content related to a specific lesson topic)
- After receiving tool results, evaluate if additional information is needed before providing your final answer
- Example: "Find courses similar to lesson 4 of MCP" → first get MCP outline → then search for that topic

Tool Usage Guidelines:
- Synthesize all tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**: Provide direct answers only — no reasoning process or tool explanations
- **After tool results**: Always provide a complete response based on the information gathered

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        # Initialize context
        system_content = self._build_system_content(conversation_history)
        messages = [{"role": "user", "content": query}]
        api_params = self._build_api_params(messages, system_content, tools)

        # Tool calling loop - up to MAX_TOOL_ROUNDS iterations
        for round_num in range(self.MAX_TOOL_ROUNDS):
            response = self.client.messages.create(**api_params)

            # Termination: No tool use requested - return text response
            if response.stop_reason != "tool_use":
                return self._extract_text(response)

            # Termination: Tool use requested but no tool_manager
            if not tool_manager:
                return self._extract_text(response)

            # Execute tools for this round
            tool_results = self._execute_round_tools(response, tool_manager)

            # Accumulate messages for next round
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            api_params["messages"] = messages

        # Termination: Max rounds reached - force final response without tools
        return self._force_final_response(messages, system_content)

    def _build_system_content(self, conversation_history: Optional[str]) -> str:
        """Build system content with optional conversation history."""
        if conversation_history:
            return f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
        return self.SYSTEM_PROMPT

    def _build_api_params(self, messages: List[Dict[str, Any]],
                          system_content: str,
                          tools: Optional[List]) -> Dict[str, Any]:
        """Build API parameters with optional tools."""
        params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = {"type": "auto"}
        return params

    def _execute_round_tools(self, response, tool_manager) -> List[Dict[str, Any]]:
        """Execute all tool_use blocks in a response and return results."""
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                except Exception as e:
                    tool_result = f"Tool execution failed: {str(e)}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        return tool_results

    def _force_final_response(self, messages: List[Dict[str, Any]],
                               system_content: str) -> str:
        """Make final API call without tools to force a text response."""
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        final_response = self.client.messages.create(**final_params)
        return self._extract_text(final_response)

    def _extract_text(self, response) -> str:
        """Extract text content from API response."""
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text
        return ""