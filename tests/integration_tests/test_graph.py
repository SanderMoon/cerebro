import pytest

from cerebro import graph
from cerebro.context import Context

pytestmark = pytest.mark.anyio


async def test_cerebro_simple_response() -> None:
    """Smoke test: agent returns a response without tool calls for a simple query."""
    res = await graph.ainvoke(
        {"messages": [("user", "Reply with exactly the word PONG and nothing else.")]},
        context=Context(),
    )

    assert "pong" in str(res["messages"][-1].content).lower()
