"""Runtime configuration for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from . import prompts


@dataclass(kw_only=True)
class Context:
    """Runtime-configurable parameters passed to every graph run."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={"description": "System prompt sent to the model on every turn."},
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-haiku-4-5-20251001",
        metadata={"description": "LLM in 'provider/model-name' format."},
    )

    max_search_results: int = field(
        default=10,
        metadata={"description": "Maximum Tavily results returned per web search."},
    )

    def __post_init__(self) -> None:
        """Override defaults with matching uppercase environment variables."""
        for f in fields(self):
            if not f.init:
                continue
            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))
