"""Tests that the prompt templates carry the placeholders the loop will fill.

A typo in a placeholder name surfaces at run time, after a model call has already been
paid for, so it is pinned here instead.
"""

from __future__ import annotations

from agentpair import roles
from agentpair.parsing import parse_probe, parse_verdict


def test_implementer_system_substitutes_module_and_test() -> None:
    filled = roles.IMPLEMENTER_SYSTEM.substitute(
        module="duration.py",
        test="test_duration.py",
        style=roles.DEFAULT_STYLE,
    )
    assert "Write duration.py and test_duration.py" in filled
    assert "$" not in filled


def test_baseline_system_substitutes_module_and_test() -> None:
    filled = roles.BASELINE_SYSTEM.substitute(
        module="duration.py",
        test="test_duration.py",
        style=roles.DEFAULT_STYLE,
    )
    assert "Write duration.py and test_duration.py" in filled
    assert "$" not in filled


def test_style_can_be_overridden() -> None:
    filled = roles.IMPLEMENTER_SYSTEM.substitute(
        module="duration.py",
        test="test_duration.py",
        style="Use tabs and no docstrings.",
    )
    assert "Use tabs and no docstrings." in filled
    assert "PEP 8" not in filled


def test_implementer_response_substitutes_findings_and_keeps_the_json_shape() -> None:
    filled = roles.IMPLEMENTER_RESPONSE.substitute(findings="f1: the parser drops seconds")
    assert "f1: the parser drops seconds" in filled
    assert '"accept"' in filled


def test_reviewer_system_asks_for_evidence() -> None:
    assert '"evidence"' in roles.REVIEWER_SYSTEM
    assert "```json" in roles.REVIEWER_SYSTEM


def test_orchestrator_probe_asks_for_the_json_the_loop_parses() -> None:
    assert "```json" in roles.ORCHESTRATOR_PROBE_SYSTEM
    assert '"probe"' in roles.ORCHESTRATOR_PROBE_SYSTEM
    assert '{"probe": null}' in roles.ORCHESTRATOR_PROBE_SYSTEM


def test_orchestrator_rule_asks_for_the_json_the_loop_parses() -> None:
    assert "```json" in roles.ORCHESTRATOR_RULE_SYSTEM
    assert '"favours"' in roles.ORCHESTRATOR_RULE_SYSTEM
    assert '"implementer" or "reviewer"' in roles.ORCHESTRATOR_RULE_SYSTEM


def test_the_example_answers_parse_with_the_parsers_that_read_them() -> None:
    # The prompt and the parser are two statements of one contract, and a prompt that
    # teaches an unparseable shape is the failure four rounds of this kept finding.
    probe = roles.PROBE_FORMAT[roles.PROBE_FORMAT.index('{"probe"') :]
    assert parse_probe(f"```json\n{probe.splitlines()[0]}\n```")
    assert parse_probe('```json\n{"probe": null}\n```') is None
    verdict = roles.VERDICT_FORMAT[roles.VERDICT_FORMAT.index('{"favours"') :]
    assert parse_verdict(f"```json\n{verdict.splitlines()[0]}\n```").favours == "reviewer"


def test_the_answer_brief_asks_for_the_decision_list_not_a_build() -> None:
    filled = roles.IMPLEMENTER_ANSWER_SYSTEM.substitute(
        module="duration.py",
        test="test_duration.py",
        style=roles.DEFAULT_STYLE,
    )
    assert "$" not in filled
    assert "duration.py" in filled and "test_duration.py" in filled
    assert "decision list" in filled
    # The build brief's two instructions are what pulled against the decisions array.
    assert "Write duration.py" not in filled
    assert "Report briefly what you built" not in filled


def test_the_answer_brief_carries_the_style_override() -> None:
    filled = roles.IMPLEMENTER_ANSWER_SYSTEM.substitute(
        module="duration.py",
        test="test_duration.py",
        style="Use tabs and no docstrings.",
    )
    assert "Use tabs and no docstrings." in filled
    assert "PEP 8" not in filled
