"""Built-in tools available to the agent.

Brain tools (todos, notes, projects, etc.) are loaded separately at runtime
from the brain-mcp MCP server. This module only defines tools that are always
available regardless of whether brain-mcp is running.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from cerebro.context import Context


async def web_search(query: str) -> Optional[dict[str, Any]]:
    """Search the web for current information using Tavily.

    Use this for questions about current events or topics not covered by the
    brain tools.
    """
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


BASE_TOOLS: List[Callable[..., Any]] = [web_search]
