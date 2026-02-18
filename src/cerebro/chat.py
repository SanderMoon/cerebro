"""Textual TUI chat interface for Cerebro.

Launches an interactive chat session backed by the LangGraph agent.
MCP tools (brain-mcp) are loaded at startup and conversation history is
persisted to SQLite so threads survive across sessions.

Usage:
    cerebro          # after `pip install -e .`
    python -m react_agent.chat
"""

from __future__ import annotations

import shutil
import uuid
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.widgets import Footer, Header, Input, Static

from cerebro.context import Context as AgentContext  # passed to astream_events as context=
from cerebro.graph import create_graph
from cerebro.tools import BASE_TOOLS

# Where chat history is stored
_DATA_DIR = Path.home() / ".local" / "share" / "cerebro"
_DB_PATH = _DATA_DIR / "chat.db"
_THREAD_FILE = _DATA_DIR / "last_thread.txt"


# ---------------------------------------------------------------------------
# Widget: a single chat message
# ---------------------------------------------------------------------------


class MessageWidget(Static):
    """Renders one chat turn (user or assistant) with optional tool-call hints."""

    DEFAULT_CSS = """
    MessageWidget {
        padding: 1 2;
        margin: 0 0 1 0;
    }
    MessageWidget.user {
        border-left: thick #5bc0f8;
        background: $surface;
    }
    MessageWidget.assistant {
        border-left: thick #7bc67e;
        background: $panel;
    }
    """

    def __init__(self, role: str, content: str = "", **kwargs: Any) -> None:
        self._role = role
        self._content = content
        self._tool_calls: List[str] = []
        super().__init__(self._build_markup(), **kwargs)
        self.add_class(role)

    def _build_markup(self, streaming: bool = False) -> str:
        if self._role == "user":
            header = "[bold #5bc0f8]You[/bold #5bc0f8]"
        else:
            header = "[bold #7bc67e]Cerebro[/bold #7bc67e]"

        parts = [header, ""]

        for tool in self._tool_calls:
            parts.append(f"[dim]  ↳ {tool}[/dim]")

        if self._tool_calls:
            parts.append("")

        if self._content:
            parts.append(self._content)
        elif streaming:
            parts.append("[dim]thinking…[/dim]")

        return "\n".join(parts)

    def set_content(
        self,
        content: str,
        tool_calls: List[str] | None = None,
        streaming: bool = False,
    ) -> None:
        """Update displayed content in-place (used during streaming)."""
        self._content = content
        if tool_calls is not None:
            self._tool_calls = tool_calls
        self.update(self._build_markup(streaming=streaming))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


class ChatApp(App[None]):
    """Cerebro — terminal chat powered by LangGraph + brain-mcp."""

    TITLE = "Cerebro"
    CSS = """
    Screen {
        background: $background;
    }
    #messages {
        height: 1fr;
        overflow-y: auto;
        padding: 1 2;
    }
    #input {
        dock: bottom;
        height: 3;
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+n", "new_thread", "New thread"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._exit_stack = AsyncExitStack()
        self._mcp_client: Any = None  # kept alive so connections aren't GC'd
        self.graph: Any = None
        self.thread_id: str = _daily_thread_id()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ScrollableContainer(id="messages")
        yield Input(placeholder="Message Cerebro… (Enter to send)", id="input")
        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_mount(self) -> None:
        self.sub_title = f"thread: {self.thread_id}"
        self._initialize()  # fire-and-forget worker

    async def on_unmount(self) -> None:
        await self._exit_stack.aclose()

    @work(exclusive=True)
    async def _initialize(self) -> None:
        """Start MCP client and SQLite checkpointer, then compile the graph."""
        _DATA_DIR.mkdir(parents=True, exist_ok=True)

        tools = list(BASE_TOOLS)

        # --- MCP tools (optional — graceful fallback if brain-mcp not found) ---
        if shutil.which("brain-mcp"):
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient

                # In langchain-mcp-adapters >= 0.1.0 the client is not a context
                # manager; connections are established lazily on get_tools().
                self._mcp_client = MultiServerMCPClient(
                    {
                        "local-brain": {
                            "transport": "stdio",
                            "command": "brain-mcp",
                            "args": [],
                        }
                    }
                )
                mcp_tools = await self._mcp_client.get_tools()
                tools = mcp_tools + tools
                self.notify(
                    f"brain-mcp connected ({len(mcp_tools)} tools)",
                    timeout=3,
                )
            except Exception as exc:
                self.notify(
                    f"brain-mcp failed to connect: {exc}",
                    severity="warning",
                    timeout=6,
                )
        else:
            self.notify(
                "brain-mcp not found in PATH — running without brain tools",
                severity="warning",
                timeout=6,
            )

        # --- SQLite checkpointer for persistent history ---
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        checkpointer = await self._exit_stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(_DB_PATH))
        )

        self.graph = create_graph(tools, checkpointer=checkpointer)

        # Focus the input now that we're ready
        self.query_one("#input", Input).focus()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        content = event.value.strip()
        if not content or self.graph is None:
            return
        event.input.clear()
        self._send_message(content)

    # ------------------------------------------------------------------
    # Message streaming
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def _send_message(self, content: str) -> None:
        container = self.query_one("#messages", ScrollableContainer)

        # Show user message
        user_widget = MessageWidget("user", content)
        await container.mount(user_widget)
        container.scroll_end(animate=False)

        # Placeholder for assistant reply
        ai_widget = MessageWidget("assistant")
        await container.mount(ai_widget)
        ai_widget.set_content("", streaming=True)  # show "thinking…"
        container.scroll_end(animate=False)

        full_text = ""
        tool_calls: List[str] = []

        try:
            config: dict = {"configurable": {"thread_id": self.thread_id}}
            # LangGraph 1.x passes the context_schema instance via context=, not configurable.
            ctx = AgentContext()
            async for event in self.graph.astream_events(
                {"messages": [("human", content)]},
                config=config,
                context=ctx,
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_tool_start":
                    tool_name = event.get("name", "tool")
                    if tool_name not in tool_calls:
                        tool_calls.append(tool_name)
                    ai_widget.set_content(full_text, tool_calls=tool_calls, streaming=True)
                    container.scroll_end(animate=False)

                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        chunk_content = getattr(chunk, "content", None)
                        added = ""
                        if isinstance(chunk_content, str):
                            added = chunk_content
                        elif isinstance(chunk_content, list):
                            # Anthropic streams content as a list of typed blocks
                            for block in chunk_content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    added += block.get("text", "")
                        if added:
                            full_text += added
                            ai_widget.set_content(
                                full_text, tool_calls=tool_calls, streaming=True
                            )
                            container.scroll_end(animate=False)

        except Exception as exc:
            import traceback
            ai_widget.set_content(
                f"[red]Error:[/red] {exc}\n\n[dim]{traceback.format_exc()}[/dim]"
            )
        else:
            ai_widget.set_content(full_text or "(no response)", tool_calls=tool_calls)
        finally:
            container.scroll_end(animate=False)
            _save_thread_id(self.thread_id)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_new_thread(self) -> None:
        """Start a fresh conversation thread."""
        self.thread_id = f"chat-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.sub_title = f"thread: {self.thread_id}"
        self.notify("New thread started", timeout=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _daily_thread_id() -> str:
    """Return a thread ID scoped to today's date, restored from last session if same day."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_id = f"chat-{today}"

    if _THREAD_FILE.exists():
        saved = _THREAD_FILE.read_text().strip()
        # If saved thread is from today, resume it
        if saved.startswith(f"chat-{today}"):
            return saved

    return daily_id


def _save_thread_id(thread_id: str) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _THREAD_FILE.write_text(thread_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the Cerebro TUI."""
    from dotenv import load_dotenv
    load_dotenv()
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()
