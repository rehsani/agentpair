"""Tests for reading what the SDK returns: the turn's text, and its result message.

`_cut_short_reason` is a pure function over a plain dataclass, so it is exercised against
real `ResultMessage` objects rather than a stand-in. Both directions matter: a cap read as
a format failure loses the reason, and a normal turn read as a cap makes the loop discard
the real parse error and blame a cap that never fired.

Text assembly is exercised against real `AssistantMessage` objects for the same reason, and
against the matchers that read the assembled text, because the cost of getting it wrong is
not a bad string, it is a matcher silently failing on a response that satisfies it.
"""

from __future__ import annotations

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

from agentpair.parsing import extract_block, parse_probe, parse_verdict
from agentpair.sdk_runner import _cut_short_reason, assemble_text, text_blocks


def make_result(**overrides) -> ResultMessage:
    """Build a result message for a normal, successful turn, with fields overridden.

    Args:
        **overrides: Fields to replace on the default successful result.

    Returns:
        The result message.
    """
    fields = {
        "subtype": "success",
        "duration_ms": 1000,
        "duration_api_ms": 900,
        "is_error": False,
        "num_turns": 3,
        "session_id": "session-1",
        "stop_reason": None,
        "terminal_reason": "completed",
        "total_cost_usd": 0.01,
        "usage": {},
    }
    return ResultMessage(**(fields | overrides))


def test_a_completed_turn_is_not_cut_short() -> None:
    assert _cut_short_reason(make_result()) is None


def test_an_end_turn_stop_reason_is_not_cut_short() -> None:
    assert _cut_short_reason(make_result(stop_reason="end_turn")) is None


def test_a_tool_use_stop_reason_is_not_cut_short() -> None:
    assert _cut_short_reason(make_result(stop_reason="tool_use")) is None


def test_the_turn_cap_is_named() -> None:
    result = make_result(subtype="error_max_turns", terminal_reason="max_turns")
    assert _cut_short_reason(result) == "the turn stopped early: max_turns"


def test_the_output_cap_is_named() -> None:
    assert _cut_short_reason(make_result(stop_reason="max_tokens")) == "the turn stopped early: max_tokens"


def test_an_interrupted_turn_is_named() -> None:
    result = make_result(terminal_reason="aborted_streaming")
    assert _cut_short_reason(result) == "the turn stopped early: aborted_streaming"


def test_an_api_error_reports_its_status_not_its_subtype() -> None:
    result = make_result(is_error=True, subtype="success", api_error_status=529)
    assert _cut_short_reason(result) == "the turn errored: API status 529"


def test_an_api_error_with_no_status_reports_the_error_list() -> None:
    result = make_result(is_error=True, subtype="success", errors=["overloaded_error"])
    assert _cut_short_reason(result) == "the turn errored: overloaded_error"


def test_an_error_subtype_is_reported_as_the_cause() -> None:
    result = make_result(is_error=True, subtype="error_during_execution")
    assert _cut_short_reason(result) == "the turn errored: error_during_execution"


def make_messages(*blocks: list[str]) -> list[AssistantMessage]:
    """Build one assistant message per group of text blocks, as a tool-using turn emits.

    Args:
        *blocks: One list of block texts per assistant message of the turn.

    Returns:
        The assistant messages, each carrying its text blocks in order.
    """
    return [AssistantMessage(content=[TextBlock(text=text) for text in group], model="claude") for group in blocks]


def assemble(*blocks: list[str]) -> str:
    """Assemble a turn's text the way the runner does.

    Args:
        *blocks: One list of block texts per assistant message of the turn.

    Returns:
        The turn's full response text.
    """
    parts: list[str] = []
    for message in make_messages(*blocks):
        parts.extend(text_blocks(message))
    return assemble_text(parts)


def test_a_turn_of_one_block_is_that_block() -> None:
    assert assemble(['```json\n{"probe": null}\n```']) == '```json\n{"probe": null}\n```'


def test_tool_calls_and_their_ids_are_not_part_of_the_text() -> None:
    message = AssistantMessage(
        content=[
            TextBlock(text="I will read duration.py first."),
            ToolUseBlock(id="t1", name="Read", input={"file_path": "duration.py"}),
        ],
        model="claude",
    )
    assert text_blocks(message) == ["I will read duration.py first."]


def test_narration_and_answer_are_separated_by_the_break_that_was_between_them() -> None:
    # The shape of every tool-using turn: it says what it is about to do, reads a file,
    # then answers. Joined with "" the fence opener lands mid-line, no fence is found at
    # all, and the orchestrator pays for three retries before recording a probe failure.
    text = assemble(["I will read duration.py and test_duration.py first."], ['```json\n{"probe": null}\n```'])
    assert text == 'I will read duration.py and test_duration.py first.\n```json\n{"probe": null}\n```'
    assert parse_probe(text) is None


def test_a_verdict_after_narration_is_still_read() -> None:
    text = assemble(["Reading the test file.", '```json\n{"favours": "implementer", "reason": "5415"}\n```'])
    assert parse_verdict(text).favours == "implementer"


def test_a_fence_after_narration_still_opens_a_line() -> None:
    text = assemble(["Here is the probe."], ['```json\n{"probe": "print(5415)"}\n```'])
    assert extract_block(text, "json") == '{"probe": "print(5415)"}'
    assert parse_probe(text) == "print(5415)"


def test_two_blocks_in_one_message_are_separated_too() -> None:
    blocks = ["thinking out loud", '```json\n{"favours": "reviewer"}\n```']
    assert assemble(blocks) == 'thinking out loud\n```json\n{"favours": "reviewer"}\n```'
    assert parse_verdict(assemble(blocks)).favours == "reviewer"


def test_a_turn_that_said_nothing_is_empty() -> None:
    assert assemble() == ""
