import os

from cerebro.context import Context


def test_context_init() -> None:
    context = Context(model="anthropic/claude-haiku-4-5-20251001")
    assert context.model == "anthropic/claude-haiku-4-5-20251001"


def test_context_init_with_env_vars() -> None:
    os.environ["MODEL"] = "anthropic/claude-sonnet-4-5"
    context = Context()
    assert context.model == "anthropic/claude-sonnet-4-5"


def test_context_init_with_env_vars_and_passed_values() -> None:
    os.environ["MODEL"] = "anthropic/claude-sonnet-4-5"
    # Passing a value different from the default means the explicit arg wins
    context = Context(model="anthropic/claude-opus-4-5")
    assert context.model == "anthropic/claude-opus-4-5"
