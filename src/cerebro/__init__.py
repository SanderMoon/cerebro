"""Cerebro — personal AI assistant built on LangGraph.

Connects to a local-brain MCP server to give the agent access to your
todos, notes, and projects. Exposes a compiled LangGraph graph for use
with LangGraph Studio or the built-in TUI.
"""

from cerebro.graph import graph

__all__ = ["graph"]
