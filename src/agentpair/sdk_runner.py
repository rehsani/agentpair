"""The Runner that talks to Claude.

Every other module works against the Runner protocol, so this is the only file that
imports the SDK and the only one that costs money to execute.

Each turn opens its own client and closes it, so one role's context never reaches
another's. That is necessary between the implementer and the reviewer, whose independence
is the thing being tested. It is not necessary between the implementer's own two turns,
and applying it there is expensive: the first measured run showed the implementer moving
380k input tokens across two cold sessions against 139k for the single-session baseline,
because the second session rediscovers the files it just wrote.
"""

from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from agentpair.metrics import usage_from_result
from agentpair.models import AgentTurn

DEFAULT_MAX_TURNS = 30
DEFAULT_MAX_BUDGET_USD = 2.0

# The API's own stop reasons for a message that ended of its own accord. "tool_use" and
# "pause_turn" end a message mid-loop and the loop continues, so neither is an early end.
# Anything else ("max_tokens", "refusal", "model_context_window_exceeded") is.
NORMAL_STOP_REASONS = frozenset({"end_turn", "stop", "stop_sequence", "tool_use", "pause_turn", "success"})

# The CLI's reasons for the query loop ending. "completed" is the normal one; "max_turns",
# "aborted_streaming" and "aborted_tools" are not. The two vocabularies are disjoint and
# must be tested separately: "completed" is a normal terminal_reason and not a stop_reason
# at all, and testing one set against both fields reports every successful turn as cut off.
NORMAL_TERMINAL_REASONS = frozenset({"completed", "success", "end_turn"})


async def sdk_runner(
    role: str,
    system: str,
    prompt: str,
    model: str,
    workspace: Path,
    tools: list[str],
    max_turns: int = DEFAULT_MAX_TURNS,
    max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
) -> AgentTurn:
    """Run one turn against Claude and return what it said and what it cost.

    Args:
        role: Label for the role, used in the accounting entry.
        system: The role brief, appended to the Claude Code preset.
        prompt: The message for this turn.
        model: Model identifier to run as.
        workspace: Directory the agent is scoped to.
        tools: The tools this role may reach. Passed as `tools`, which controls what is
            available, not only as `allowed_tools`, which merely skips permission prompts
            that bypassPermissions already skips. An empty list is a reasoning-only turn.
        max_turns: Ceiling on the agent's internal tool loop. Each round trip resends the
            whole context, so an uncapped loop is the main way a single turn runs away.
        max_budget_usd: Hard spend ceiling for this one turn.

    Returns:
        The assistant text and the turn's accounting.

    Raises:
        RuntimeError: If the session ended without a result message, which means no
            usage was reported and the turn cannot be accounted for.
    """
    options = ClaudeAgentOptions(
        system_prompt={"type": "preset", "preset": "claude_code", "append": system},
        tools=tools,
        allowed_tools=tools,
        permission_mode="bypassPermissions",
        model=model,
        cwd=str(workspace),
        setting_sources=[],
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
    )

    parts: list[str] = []
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                parts.extend(text_blocks(message))
            elif isinstance(message, ResultMessage):
                return AgentTurn(
                    text=assemble_text(parts),
                    usage=usage_from_result(role, message),
                    cut_short=_cut_short_reason(message),
                )
    raise RuntimeError(f"{role} turn ended without a result message")


def text_blocks(message: AssistantMessage) -> list[str]:
    """Return the text of each text block in one assistant message, in order.

    Args:
        message: One assistant message of the turn.

    Returns:
        One entry per text block, with tool calls and their results left out.
    """
    return [block.text for block in message.content if isinstance(block, TextBlock)]


def assemble_text(parts: list[str]) -> str:
    """Join a turn's text blocks into the response the parsers read.

    A tool-using turn narrates in one message and answers in another, so its text arrives
    as separate blocks with the break between them dropped. Joining those with "" glues
    the answer onto the end of the narration and puts it mid-line, and the fence opener
    every role's answer begins with has to start a line to be found. So the blocks are
    rejoined with the break that separated them. A block ends where a message or a tool
    call ends, never mid-sentence, so the break cannot land inside a line the model wrote
    as one line.

    Args:
        parts: The turn's text blocks, in the order they arrived.

    Returns:
        The turn's full response text.
    """
    return "\n".join(parts)


def _cut_short_reason(message: ResultMessage) -> str | None:
    """Return why a turn ended early, if it did.

    A turn stopped by the turn or budget cap returns whatever text it had reached, which
    for a role that owes JSON is an unparseable answer. Without this the cap firing is
    indistinguishable from the model ignoring the requested format, which is exactly the
    confusion the failure reasons exist to remove. The inverse matters just as much: a
    normal turn reported as cut short makes the loop discard a genuine parse error and
    blame a cap that never fired, so each field is tested against its own vocabulary.

    Args:
        message: The SDK result message ending the turn.

    Returns:
        A short reason, or None when the turn ran to completion normally.
    """
    if getattr(message, "is_error", False):
        return f"the turn errored: {_error_detail(message)}"
    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason and str(stop_reason).lower() not in NORMAL_STOP_REASONS:
        return f"the turn stopped early: {stop_reason}"
    terminal_reason = getattr(message, "terminal_reason", None)
    if terminal_reason and str(terminal_reason).lower() not in NORMAL_TERMINAL_REASONS:
        return f"the turn stopped early: {terminal_reason}"
    subtype = getattr(message, "subtype", None)
    if subtype and subtype != "success":
        return f"the turn stopped early: {subtype}"
    return None


def _error_detail(message: ResultMessage) -> str:
    """Name what went wrong on a result flagged as an error.

    An API failure on the last turn arrives as is_error with subtype "success", so the
    subtype alone would report "the turn errored: success". The HTTP status and the error
    list are where the CLI puts the actual cause in that case.

    Args:
        message: The SDK result message ending the turn, with is_error set.

    Returns:
        The most specific cause available, or a stated absence of one.
    """
    subtype = getattr(message, "subtype", None)
    if subtype and subtype != "success":
        return str(subtype)
    api_error_status = getattr(message, "api_error_status", None)
    if api_error_status:
        return f"API status {api_error_status}"
    errors = getattr(message, "errors", None)
    if errors:
        return "; ".join(str(error) for error in errors)
    return "no reason reported"
