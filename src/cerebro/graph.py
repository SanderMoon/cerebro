"""LangGraph ReAct agent for Cerebro."""

from datetime import UTC, datetime
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from cerebro.context import Context
from cerebro.state import InputState, State
from cerebro.tools import BASE_TOOLS
from cerebro.utils import load_chat_model


def create_graph(
    tools: Optional[List[Any]] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> Any:
    """Create and compile the agent graph.

    Args:
        tools: Tools the agent can use. Defaults to BASE_TOOLS.
            When brain-mcp is available, the TUI prepends its tools here.
        checkpointer: Optional checkpointer for persisting conversation state
            across turns. Pass AsyncSqliteSaver for durable chat history.

    Returns:
        A compiled LangGraph graph.
    """
    if tools is None:
        tools = BASE_TOOLS

    async def call_model(
        state: State, runtime: Runtime[Context]
    ) -> Dict[str, List[AIMessage]]:
        model = load_chat_model(runtime.context.model).bind_tools(tools)

        system_message = runtime.context.system_prompt.format(
            system_time=datetime.now(tz=UTC).isoformat()
        )

        response = cast(
            AIMessage,
            await model.ainvoke(
                [{"role": "system", "content": system_message}, *state.messages]
            ),
        )

        if state.is_last_step and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, I could not find an answer to your question in the specified number of steps.",
                    )
                ]
            }

        return {"messages": [response]}

    def route_model_output(state: State) -> Literal["__end__", "tools"]:
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage):
            raise ValueError(
                f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
            )
        if not last_message.tool_calls:
            return "__end__"
        return "tools"

    builder = StateGraph(State, input_schema=InputState, context_schema=Context)
    builder.add_node(call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge("__start__", "call_model")
    builder.add_conditional_edges("call_model", route_model_output)
    builder.add_edge("tools", "call_model")

    return builder.compile(checkpointer=checkpointer, name="Cerebro")


# Default graph for LangGraph Studio (Studio manages its own checkpointing)
graph = create_graph()
