"""Tests for pulling findings, decisions and verdicts out of model prose."""

from __future__ import annotations

import pytest

from agentpair.parsing import ParseError, extract_block, parse_decisions, parse_findings, parse_probe, parse_verdict


def test_extract_takes_the_last_matching_fence() -> None:
    text = 'Format is:\n```json\n[{"id": "example"}]\n```\nMy answer:\n```json\n[{"id": "f1"}]\n```'
    assert extract_block(text, "json") == '[{"id": "f1"}]'


def test_extract_ignores_other_languages() -> None:
    with pytest.raises(ParseError, match="no ```json block"):
        extract_block("```python\nprint(1)\n```", "json")


def test_parse_findings_reads_claim_and_evidence() -> None:
    text = '```json\n[{"id": "f1", "claim": "drops seconds", "evidence": "duration.py:12"}]\n```'
    findings = parse_findings(text)
    assert findings[0].id == "f1"
    assert findings[0].evidence == "duration.py:12"


def test_parse_findings_defaults_missing_evidence_to_empty() -> None:
    findings = parse_findings('```json\n[{"id": "f1", "claim": "drops seconds"}]\n```')
    assert findings[0].evidence == ""


def test_parse_findings_accepts_an_explicit_empty_review() -> None:
    assert parse_findings("Nothing to raise.\n```json\n[]\n```") == []


def test_parse_findings_refuses_a_missing_block() -> None:
    with pytest.raises(ParseError, match="no ```json block"):
        parse_findings("The code looks fine to me.")


def test_parse_findings_refuses_malformed_json() -> None:
    with pytest.raises(ParseError, match="not valid JSON"):
        parse_findings('```json\n[{"id": "f1",}]\n```')


def test_parse_findings_refuses_a_non_array() -> None:
    with pytest.raises(ParseError, match="must be a JSON array"):
        parse_findings('```json\n{"id": "f1"}\n```')


def test_parse_findings_refuses_a_missing_claim() -> None:
    with pytest.raises(ParseError, match=r"missing \['claim'\]"):
        parse_findings('```json\n[{"id": "f1", "evidence": "duration.py:12"}]\n```')


def test_parse_findings_refuses_duplicate_ids() -> None:
    text = '```json\n[{"id": "f1", "claim": "a"}, {"id": "f1", "claim": "b"}]\n```'
    with pytest.raises(ParseError, match="duplicate finding id"):
        parse_findings(text)


def test_parse_decisions_reads_accept_and_reason() -> None:
    text = '```json\n[{"id": "f1", "accept": false, "reason": "the spec says 5415"}]\n```'
    decisions = parse_decisions(text)
    assert decisions[0].finding_id == "f1"
    assert decisions[0].accept is False
    assert decisions[0].reason == "the spec says 5415"


def test_parse_decisions_defaults_a_missing_reason_to_empty() -> None:
    assert parse_decisions('```json\n[{"id": "f1", "accept": true}]\n```')[0].reason == ""


def test_parse_decisions_refuses_a_missing_accept() -> None:
    with pytest.raises(ParseError, match=r"missing \['accept'\]"):
        parse_decisions('```json\n[{"id": "f1"}]\n```')


def test_parse_decisions_refuses_two_answers_to_one_finding() -> None:
    text = '```json\n[{"id": "f1", "accept": true}, {"id": "f1", "accept": false}]\n```'
    with pytest.raises(ParseError, match="duplicate decision"):
        parse_decisions(text)


def test_parse_findings_refuses_a_blank_claim() -> None:
    with pytest.raises(ParseError, match="claim must not be empty"):
        parse_findings('```json\n[{"id": "f1", "claim": "   "}]\n```')


def test_parse_findings_refuses_a_blank_id() -> None:
    with pytest.raises(ParseError, match="id must not be empty"):
        parse_findings('```json\n[{"id": "", "claim": "drops seconds"}]\n```')


def test_parse_findings_refuses_a_null_claim_rather_than_reading_it_as_none() -> None:
    with pytest.raises(ParseError, match="claim must be a string, got NoneType"):
        parse_findings('```json\n[{"id": "f1", "claim": null}]\n```')


@pytest.mark.parametrize(
    ("evidence", "rendered"),
    [
        ('["mod.py:2", "mod.py:5"]', "mod.py:2; mod.py:5"),
        ("12", "12"),
        ("false", "false"),
        ('{"file": "mod.py", "line": 2}', '{"file": "mod.py", "line": 2}'),
        ("[]", ""),
    ],
)
def test_non_string_evidence_is_rendered_rather_than_costing_a_retry(evidence: str, rendered: str) -> None:
    # The format asks for a string, but nothing branches on evidence: it is only pasted
    # into a later prompt, so refusing it burned a paid turn over a container choice.
    findings = parse_findings(f'```json\n[{{"id": "f1", "claim": "a", "evidence": {evidence}}}]\n```')
    assert findings[0].evidence == rendered


def test_one_oddly_shaped_evidence_no_longer_discards_the_findings_beside_it() -> None:
    findings = parse_findings(
        '```json\n[{"id": "f1", "claim": "a", "evidence": ["mod.py:2", "mod.py:5"]},'
        ' {"id": "f2", "claim": "b", "evidence": "mod.py:9"}]\n```'
    )
    assert [finding.id for finding in findings] == ["f1", "f2"]
    assert findings[0].evidence == "mod.py:2; mod.py:5"


def test_parse_decisions_refuses_a_blank_id() -> None:
    with pytest.raises(ParseError, match="id must not be empty"):
        parse_decisions('```json\n[{"id": " ", "accept": true}]\n```')


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('[{"id": "f1", "claim": "a", "evidence": null}]', ""),
        ('[{"id": "f1", "claim": "a", "evidence": ""}]', ""),
        ('[{"id": "f1", "claim": "a", "evidence": "   "}]', ""),
        ('[{"id": "f1", "claim": "a"}]', ""),
        ('[{"id": "f1", "claim": "a", "evidence": " duration.py:12 "}]', "duration.py:12"),
    ],
)
def test_a_null_or_absent_evidence_is_read_as_no_evidence(body: str, expected: str) -> None:
    findings = parse_findings(f"```json\n{body}\n```")
    assert findings[0].evidence == expected


def test_one_null_evidence_does_not_discard_the_other_findings() -> None:
    text = (
        '```json\n[{"id": "f1", "claim": "a", "evidence": "duration.py:1"},'
        ' {"id": "f2", "claim": "b", "evidence": null},'
        ' {"id": "f3", "claim": "c", "evidence": "duration.py:3"}]\n```'
    )
    findings = parse_findings(text)
    assert [finding.id for finding in findings] == ["f1", "f2", "f3"]
    assert findings[1].evidence == ""


def test_a_null_reason_is_read_as_no_reason() -> None:
    assert parse_decisions('```json\n[{"id": "f1", "accept": false, "reason": null}]\n```')[0].reason == ""


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('[{"id": null, "claim": "a"}]', "id must be a string, got NoneType"),
        ('[{"id": "f1", "claim": null}]', "claim must be a string, got NoneType"),
    ],
)
def test_a_required_field_that_is_null_is_still_refused(body: str, message: str) -> None:
    with pytest.raises(ParseError, match=message):
        parse_findings(f"```json\n{body}\n```")


def fence(body: str) -> str:
    """Wrap a JSON body in the fence every role answers in.

    Args:
        body: The JSON text the model wrote.

    Returns:
        The body inside a ```json fence.
    """
    return f"```json\n{body}\n```"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('{"favours": "implementer"}', "implementer"),
        ('{"favours": "reviewer"}', "reviewer"),
        ('{"favours": "reviewer", "reason": "duration.py:2 returns 5400"}', "reviewer"),
        ('{"favours": "IMPLEMENTER"}', "implementer"),
        ('{"favours": " reviewer "}', "reviewer"),
        ('{"reason": "the probe printed 5415", "favours": "implementer"}', "implementer"),
    ],
)
def test_parse_verdict_reads_the_side_the_orchestrator_wrote(body: str, expected: str) -> None:
    assert parse_verdict(fence(body)).favours == expected


def test_the_recorded_reason_is_the_one_the_object_carried() -> None:
    verdict = parse_verdict(fence('{"favours": "reviewer", "reason": " duration.py:2 returns 5400 "}'))
    assert verdict.reason == "duration.py:2 returns 5400"


@pytest.mark.parametrize("body", ['{"favours": "reviewer"}', '{"favours": "reviewer", "reason": null}'])
def test_a_ruling_with_no_reason_is_still_a_ruling(body: str) -> None:
    # The side is the answer, and refusing a ruling over a missing reason costs a paid turn
    # and leaves the dispute unresolved, which is worse than recording an empty reason.
    verdict = parse_verdict(fence(body))
    assert verdict.favours == "reviewer"
    assert verdict.reason == ""


def test_a_non_string_reason_does_not_throw_away_the_side_that_was_named() -> None:
    # favours is the answer and stays strict; reason is prose and is rendered, so a badly
    # shaped reason no longer costs three retries and an unresolved dispute.
    verdict = parse_verdict(fence('{"favours": "reviewer", "reason": 12}'))
    assert verdict.favours == "reviewer"
    assert verdict.reason == "12"


def test_a_non_string_decision_reason_is_rendered() -> None:
    decisions = parse_decisions(fence('[{"id": "f1", "accept": false, "reason": ["too slow", "and wrong"]}]'))
    assert decisions[0].reason == "too slow; and wrong"


def test_accept_written_as_a_string_is_still_refused() -> None:
    # accept decides whether a finding becomes an Acceptance or a Dispute, so coercing it
    # would invent a decision the implementer did not make.
    with pytest.raises(ParseError, match="accept must be true or false"):
        parse_decisions(fence('[{"id": "f1", "accept": "true"}]'))


def test_a_non_string_favours_is_still_refused() -> None:
    with pytest.raises(ParseError, match="favours must be a string"):
        parse_verdict(fence('{"favours": ["reviewer"]}'))


def test_a_verdict_is_read_after_the_reasoning_that_names_the_other_side() -> None:
    # The common shape: prose that argues about both sides, then the answer. Under the old
    # prose contract the side was taken from wherever a keyword appeared, and every round
    # of that contract had a shape where the prose won.
    text = (
        "The implementer says this favours the implementer, and FAVOURS: IMPLEMENTER is the\n"
        "answer it wants. But duration.py:2 returns 5400.\n\n" + fence('{"favours": "reviewer", "reason": "5400"}')
    )
    assert parse_verdict(text).favours == "reviewer"


def test_the_last_verdict_fence_is_the_answer() -> None:
    text = (
        "The format is:\n" + fence('{"favours": "implementer"}') + "\nMy ruling:\n" + fence('{"favours": "reviewer"}')
    )
    assert parse_verdict(text).favours == "reviewer"


# The five shapes that round six measured recording the loser as the winner, and the sixth
# that the ruling prompt literally asked for and that the prose parser refused. None of
# them carries a verdict object, so each is now a clean refusal, which is recorded as an
# unresolved dispute and is visible. The point is that none can be read as a side.
PROSE_RULINGS = [
    "The options are FAVOURS: IMPLEMENTER or FAVOURS: REVIEWER.\n\nFAVOURS: REVIEWER",
    "I must answer either\n\nFAVOURS: IMPLEMENTER\n\nor\n\nFAVOURS: REVIEWER\n\n...I rule FAVOURS: REVIEWER.",
    "```\nFAVOURS: IMPLEMENTER\n```\n\nHaving read m.py:2 I rule.\n\nFAVOURS: REVIEWER",
    "Nothing in the spec favours: the implementer here.\n\nFAVOURS: REVIEWER",
    "The evidence disfavours: implementer on this point.\n\nFAVOURS: REVIEWER",
    "FAVOURS: REVIEWER\nThe implementer said this FAVOURS: IMPLEMENTER, but m.py:2 shows otherwise.",
]


@pytest.mark.parametrize("text", PROSE_RULINGS)
def test_a_ruling_written_as_prose_is_refused_and_never_inverted(text: str) -> None:
    with pytest.raises(ParseError, match="no ```json block"):
        parse_verdict(text)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('{"reason": "the probe printed 5415"}', 'missing "favours"'),
        ('{"favours": "neither"}', 'favours must be "implementer" or "reviewer"'),
        ('{"favours": "the reviewer"}', 'favours must be "implementer" or "reviewer"'),
        ('{"favours": ""}', 'favours must be "implementer" or "reviewer"'),
        ('{"favours": null}', "favours must be a string, got NoneType"),
        ('{"favours": ["reviewer"]}', "favours must be a string, got list"),
        ('["reviewer"]', "verdict block must be a JSON object, got list"),
        ('{"favours": "reviewer",}', "not valid JSON"),
    ],
)
def test_parse_verdict_refuses_anything_that_is_not_one_of_the_two_sides(body: str, message: str) -> None:
    with pytest.raises(ParseError, match=message):
        parse_verdict(fence(body))


def test_parse_verdict_refuses_a_response_with_no_fence() -> None:
    with pytest.raises(ParseError, match="no ```json block"):
        parse_verdict("I cannot tell from what I was given.")


def test_parse_probe_reads_a_one_line_script() -> None:
    assert parse_probe(fence('{"probe": "print(5415)"}')) == "print(5415)"


def test_a_multi_line_script_arrives_as_one_json_string() -> None:
    # The break between the two statements is a \n inside the string, so no line of the
    # response is part of the script and nothing about the response's layout can change it.
    body = '{"probe": "import duration\\nprint(duration.parse_duration(\'1h30m\'))"}'
    assert parse_probe(fence(body)) == "import duration\nprint(duration.parse_duration('1h30m'))"


def test_a_script_that_prints_three_backticks_still_truncates_the_fence() -> None:
    # Unchanged from the previous contract and deliberately not fixed: the fence regex ends
    # at the next ``` wherever it is. Pinned so the limit is stated rather than assumed
    # away, and it is a parse failure, which is retried and then recorded, not a wrong probe.
    with pytest.raises(ParseError, match="not valid JSON"):
        parse_probe(fence('{"probe": "print(\'```\')"}'))


def test_a_null_probe_is_the_orchestrator_declining_to_run_anything() -> None:
    assert parse_probe(fence('{"probe": null}')) is None


def test_a_probe_quoting_the_disputed_code_is_still_the_probe() -> None:
    # Under the old contract this response carried two fences and the outcome turned on
    # which was tagged and on whether the refusal phrase appeared above them.
    text = (
        "The disputed function is:\n\n```python\ndef parse_duration(text):\n    return 1\n```\n\n"
        "NO PROBE is not what I mean, here is what to run:\n\n" + fence('{"probe": "print(5415)"}')
    )
    assert parse_probe(text) == "print(5415)"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('{"reason": "nothing to run"}', 'missing "probe"'),
        ('{"probe": ""}', "probe must not be empty"),
        ('{"probe": "   "}', "probe must not be empty"),
        ('{"probe": 5415}', "probe must be a string or null, got int"),
        ('{"probe": ["print(1)"]}', "probe must be a string or null, got list"),
        ("null", "probe block must be a JSON object, got NoneType"),
    ],
)
def test_parse_probe_refuses_an_answer_that_is_neither_a_script_nor_a_refusal(body: str, message: str) -> None:
    with pytest.raises(ParseError, match=message):
        parse_probe(fence(body))


@pytest.mark.parametrize(
    "text",
    [
        "NO PROBE",
        "Nothing I could run would settle a naming disagreement.",
        "```python\nprint(5415)\n```",
        "```\nprint(5415)\n```",
    ],
)
def test_a_probe_turn_that_ignored_the_format_is_refused_rather_than_guessed_at(text: str) -> None:
    with pytest.raises(ParseError, match="no ```json block"):
        parse_probe(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('```json\n{"probe": null}\n```', '{"probe": null}'),
        ('```JSON\n{"probe": null}\n```', '{"probe": null}'),
        ('```json \n{"probe": null}\n```', '{"probe": null}'),
        ('```json\r\n{"probe": null}\n```', '{"probe": null}'),
        ('```json\n[]\n```\nbetter:\n```json\n{"probe": null}\n```', '{"probe": null}'),
    ],
)
def test_the_json_fence_is_found_however_the_tag_is_cased(text: str, expected: str) -> None:
    assert extract_block(text, "json") == expected


@pytest.mark.parametrize(
    "text",
    [
        "```\n[]\n```",
        "```python\nprint(1)\n```",
        "```bash\necho 1\n```",
        "no fence at all",
        "    []\n",
    ],
)
def test_a_fence_that_is_not_tagged_json_is_refused(text: str) -> None:
    # The lone-untagged-fence fallback is gone with the python probe it existed for. It
    # guessed that an untagged block was the answer, and every guess about which block was
    # the answer has cost a round.
    with pytest.raises(ParseError, match="no ```json block"):
        extract_block(text, "json")


def test_a_decision_matches_its_finding_whatever_case_each_side_used() -> None:
    findings = parse_findings(fence('[{"id": "F1", "claim": "a"}, {"id": "f2", "claim": "b"}]'))
    decisions = parse_decisions(fence('[{"id": "f1", "accept": true}, {"id": "F2", "accept": false, "reason": "no"}]'))
    by_id = {finding.id: finding for finding in findings}
    assert [decision.finding_id in by_id for decision in decisions] == [True, True]


def test_ids_that_differ_only_by_case_are_the_duplicate_they_are() -> None:
    with pytest.raises(ParseError, match="duplicate finding id"):
        parse_findings(fence('[{"id": "f1", "claim": "a"}, {"id": "F1", "claim": "b"}]'))


def test_two_decisions_answering_one_finding_in_different_cases_are_refused() -> None:
    with pytest.raises(ParseError, match="duplicate decision"):
        parse_decisions(fence('[{"id": "f1", "accept": true}, {"id": "F1", "accept": false, "reason": "no"}]'))
