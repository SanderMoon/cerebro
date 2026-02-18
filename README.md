# Cerebro

A personal AI assistant built on [LangGraph](https://github.com/langchain-ai/langgraph). Cerebro connects to your [local-brain](https://github.com/sandermoonemans/local-brain) knowledge base via MCP, giving the agent direct access to your todos, notes, projects, and daily logs through a terminal chat interface.

## Features

- **Terminal chat (TUI)** — clean chat interface built with [Textual](https://textual.textualize.io/), streaming responses token by token
- **Brain tools** — reads and writes todos, notes, and projects via the `brain-mcp` MCP server
- **Web search** — falls back to Tavily for current events and general knowledge
- **Persistent history** — conversations are saved to SQLite; each day resumes the same thread automatically
- **LangGraph Studio** — open the graph visually with `langgraph dev` for debugging

## Prerequisites

1. **[local-brain](https://github.com/sandermoonemans/local-brain-init)** — install and configure your brain, then make sure `brain-mcp` is available in your `$PATH`
2. **Anthropic API key** — Cerebro defaults to Claude Haiku; any `anthropic/` model works
3. **Python ≥ 3.11** and [uv](https://github.com/astral-sh/uv)

## Installation

```bash
git clone https://github.com/sandermoonemans/cerebro
cd cerebro
cp .env.example .env
# Add your API key(s) to .env
uv sync
```

## Usage

```bash
uv run cerebro
```

### Keybindings

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Ctrl+N` | Start a new thread |
| `Ctrl+C` | Quit |

Chat history is stored in `~/.local/share/cerebro/chat.db`. Each calendar day gets its own thread by default; `Ctrl+N` starts a fresh one at any time.

## Configuration

`.env` file:

```bash
ANTHROPIC_API_KEY=your-key-here   # required
TAVILY_API_KEY=your-key-here      # optional, enables web search
```

The model and system prompt can also be overridden via environment variables (`MODEL`, `SYSTEM_PROMPT`) or by editing `src/cerebro/context.py`.

## Development

```bash
# Unit tests
uv run pytest tests/unit_tests/

# LangGraph Studio (visual graph debugger)
uv run langgraph dev
```

## Architecture

```
src/cerebro/
├── graph.py      # create_graph() factory + default compiled graph
├── chat.py       # Textual TUI entry point
├── context.py    # Runtime config (model, system prompt, search results)
├── prompts.py    # Default system prompt
├── state.py      # Agent state schema
├── tools.py      # web_search tool (Tavily)
└── utils.py      # load_chat_model helper
```

The agent loop: `call_model → [tools →] call_model → response`

Brain tools (todos, notes, projects, search within brain, etc.) are loaded at TUI startup from the `brain-mcp` MCP server and injected into the graph. If `brain-mcp` is not found in `$PATH`, the agent falls back to web search only.
