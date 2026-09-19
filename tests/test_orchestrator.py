"""End to end tests of the pair loop with scripted model responses.

The runner is injected, so these exercise the whole protocol, including a real probe
subprocess, without an SDK session or a paid call.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from agentpair.arbitration import SelfArbitration
from agentpair.models import AgentTurn, RoleUsage, RunFailed
from agentpair.orchestrator import run_pair


class ScriptedRunner:
    """Returns a queued response per role, and records the order roles were called in."""

    def __init__(self, script: dict[str, list[str]]) -> None:
        for role, texts in script.items():
            if isinstance(texts, str):
                raise TypeError(f"script for {role!r} must be a list of responses, not a string")
        self.script = {role: list(texts) for role, texts in script.items()}
        self.calls: list[str] = []

    async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
        self.calls.append(role)
        queued = self.script.get(role) or ["(nothing scripted)"]
        text = queued.pop(0) if queued else "(exhausted)"
        usage = RoleUsage(role=role, input_tokens=10, output_tokens=5, cost_usd=0.001, seconds=0.1)
        return AgentTurn(text=text, usage=usage)


async def run(script: dict[str, list[str]], workspace: Path):
    runner = ScriptedRunner(script)
    record = await run_pair(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=workspace,
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="haiku",
        runner=runner,
    )
    return record, runner


REVIEW_ONE = '```json\n[{"id": "f1", "claim": "returns 5400", "evidence": "duration.py:12"}]\n```'
REJECT_ONE = '```json\n[{"id": "f1", "accept": false, "reason": "it returns 5415"}]\n```'
ACCEPT_ONE = '```json\n[{"id": "f1", "accept": true, "reason": "fixed"}]\n```'
ACCEPT_AND_REJECT = (
    '```json\n[{"id": "f1", "accept": true, "reason": "applied"},'
    ' {"id": "f2", "accept": false, "reason": "the spec says 5415"}]\n```'
)

# The orchestrator answers in the same fence the other two roles do.
PROBE = '```json\n{"probe": "print(5415)"}\n```'
NO_PROBE = '```json\n{"probe": null}\n```'
RULE_FOR_IMPLEMENTER = '```json\n{"favours": "implementer", "reason": "the probe printed 5415 not 5400"}\n```'
RULE_FOR_REVIEWER = '```json\n{"favours": "reviewer", "reason": "duration.py:12 returns 5400"}\n```'


async def test_clean_review_stops_after_two_turns(tmp_path: Path) -> None:
    record, runner = await run(
        {"implementer": ["built it"], "reviewer": ["nothing wrong\n```json\n[]\n```"]},
        tmp_path,
    )
    assert record.findings == []
    assert record.resolutions == []
    assert runner.calls == ["implementer", "reviewer"]
    assert record.total_tokens == 30


async def test_accepted_finding_produces_no_dispute(tmp_path: Path) -> None:
    record, runner = await run(
        {"implementer": ["built it", ACCEPT_ONE], "reviewer": [REVIEW_ONE]},
        tmp_path,
    )
    assert len(record.findings) == 1
    assert record.disputes == []
    assert "orchestrator_probe" not in runner.calls


async def test_rejection_is_settled_with_a_probe(tmp_path: Path) -> None:
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [PROBE],
            "orchestrator_rule": [RULE_FOR_IMPLEMENTER],
        },
        tmp_path,
    )
    resolution = record.resolutions[0]
    assert resolution.method == "judged_with_probe"
    assert resolution.favours == "implementer"
    assert "observation: 5415" in resolution.rationale
    assert Path(resolution.probe_path).exists()
    assert runner.calls.count("orchestrator_probe") == 1


async def test_rejection_the_orchestrator_cannot_probe_is_judged(tmp_path: Path) -> None:
    record, runner = await run(
        {
            "implementer": ["built it", '```json\n[{"id": "f1", "accept": false, "reason": "the name is clear"}]\n```'],
            "reviewer": ['```json\n[{"id": "f1", "claim": "bad name", "evidence": "duration.py:4"}]\n```'],
            "orchestrator_probe": [NO_PROBE],
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        },
        tmp_path,
    )
    resolution = record.resolutions[0]
    assert resolution.method == "judged"
    assert resolution.probe_path is None
    assert "orchestrator_rule" in runner.calls


async def test_a_broken_probe_is_retried_then_succeeds(tmp_path: Path) -> None:
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": ["no fence here", '```json\n{"probe": "def broken("}\n```', PROBE],
            "orchestrator_rule": [RULE_FOR_IMPLEMENTER],
        },
        tmp_path,
    )
    assert runner.calls.count("orchestrator_probe") == 3
    assert record.resolutions[0].method == "judged_with_probe"


async def test_a_probe_that_never_runs_still_gets_a_ruling(tmp_path: Path) -> None:
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": ["no fence here"] * 3,
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        },
        tmp_path,
    )
    assert runner.calls.count("orchestrator_probe") == 3
    assert record.unresolved == []
    assert record.resolutions[0].method == "judged"
    assert record.resolutions[0].favours == "reviewer"
    assert len(record.probe_failures) == 1
    assert "after 3 attempts" in record.probe_failures[0]


async def test_prose_around_the_probe_object_does_not_change_it(tmp_path: Path) -> None:
    # The response says NO PROBE and carries a probe, which under the prose contract was a
    # contradiction the loop had to arbitrate and got wrong twice. The object is the answer.
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [f"This is not a NO PROBE case.\n{PROBE}"],
            "orchestrator_rule": [RULE_FOR_IMPLEMENTER],
        },
        tmp_path,
    )
    assert runner.calls.count("orchestrator_probe") == 1
    assert record.resolutions[0].method == "judged_with_probe"
    assert "observation: 5415" in record.resolutions[0].rationale
    assert record.probe_failures == []


async def test_an_unusable_verdict_leaves_a_reason(tmp_path: Path) -> None:
    record, _ = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": ["NO PROBE"],
            "orchestrator_rule": ["I cannot tell from what I was given."],
        },
        tmp_path,
    )
    assert record.resolutions == []
    assert "no usable verdict" in record.unresolved[0]


async def test_unanswered_finding_is_recorded_not_treated_as_accepted(tmp_path: Path) -> None:
    record, _ = await run(
        {
            "implementer": ["built it", '```json\n[{"id": "f1", "accept": true}]\n```'],
            "reviewer": ['```json\n[{"id": "f1", "claim": "a"}, {"id": "f2", "claim": "b"}]\n```'],
        },
        tmp_path,
    )
    assert record.unanswered == ["f2"]
    assert record.disputes == []


async def test_a_decision_naming_no_finding_is_a_protocol_error(tmp_path: Path) -> None:
    record, _ = await run(
        {
            "implementer": ["built it", '```json\n[{"id": "f9", "accept": false, "reason": "no"}]\n```'],
            "reviewer": [REVIEW_ONE],
        },
        tmp_path,
    )
    assert record.protocol_errors == ["f9: decision names no known finding"]
    assert record.unresolved == []
    assert record.unanswered == ["f1"]


async def test_an_unparseable_review_keeps_the_usage_already_paid_for(tmp_path: Path) -> None:
    record, _ = await run(
        {"implementer": ["built it"], "reviewer": ["The code looks fine to me."]},
        tmp_path,
    )
    assert record.failure is not None
    assert "reviewer answer did not parse" in record.failure
    assert "after 3 attempts" in record.failure
    # One implementer turn plus three reviewer attempts, all of them paid for.
    assert record.total_tokens == 60
    assert record.total_cost_usd > 0


async def test_a_cut_short_turn_is_named_as_the_cause(tmp_path: Path) -> None:
    class CutShortRunner(ScriptedRunner):
        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            turn = await super().__call__(role, system, prompt, model, workspace, tools)
            if role == "reviewer":
                return AgentTurn(text=turn.text, usage=turn.usage, cut_short="the turn stopped early: max_turns")
            return turn

    runner = CutShortRunner({"implementer": ["built it"], "reviewer": ["partial answer with no fence"]})
    record = await run_pair(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="haiku",
        runner=runner,
    )
    assert "max_turns" in record.failure
    assert "did not parse" not in record.failure


async def test_an_unparseable_decision_does_not_mark_every_finding_unanswered(tmp_path: Path) -> None:
    record, _ = await run(
        {"implementer": ["built it", "I disagree with all of it."], "reviewer": [REVIEW_ONE]},
        tmp_path,
    )
    assert record.failure is not None
    assert record.unanswered == []
    assert record.disputes == []


async def test_the_orchestrator_model_is_recorded(tmp_path: Path) -> None:
    runner = ScriptedRunner({"implementer": ["built it"], "reviewer": ["```json\n[]\n```"]})
    record = await run_pair(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="haiku",
        runner=runner,
    )
    assert record.orchestrator_model == "haiku"


@pytest.mark.parametrize(
    ("orchestrator_model", "named"),
    [
        ("opus", "reviewer"),
        ("OPUS", "reviewer"),
        ("Opus", "reviewer"),
        (" opus ", "reviewer"),
        ("\topus", "reviewer"),
        ("sonnet", "implementer"),
        ("Sonnet ", "implementer"),
    ],
)
async def test_a_disputants_model_as_orchestrator_is_refused_by_the_library(
    tmp_path: Path,
    orchestrator_model: str,
    named: str,
) -> None:
    runner = ScriptedRunner({"implementer": ["built it"], "reviewer": ["```json\n[]\n```"]})
    with pytest.raises(SelfArbitration, match=named):
        await run_pair(
            task_id="duration",
            spec="write parse_duration",
            module="duration.py",
            test="test_duration.py",
            workspace=tmp_path,
            implementer_model="sonnet",
            reviewer_model="opus",
            orchestrator_model=orchestrator_model,
            runner=runner,
        )
    assert runner.calls == []


async def test_self_arbitration_is_allowed_when_the_caller_asks_for_it(tmp_path: Path) -> None:
    runner = ScriptedRunner({"implementer": ["built it"], "reviewer": ["```json\n[]\n```"]})
    record = await run_pair(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="Opus",
        runner=runner,
        allow_self_arbitration=True,
    )
    assert record.orchestrator_model == "Opus"


async def test_a_cut_short_turn_whose_text_parses_is_still_recorded(tmp_path: Path) -> None:
    runner = CutShortOn({"implementer": ["built it"], "reviewer": ["```json\n[]\n```"]}, "reviewer")
    record = await pair_with(runner, tmp_path)
    assert record.failure is None
    assert record.findings == []
    assert record.cut_short == ["reviewer: the turn stopped early: max_turns"]


async def test_a_run_with_no_cap_firing_records_no_cut_short(tmp_path: Path) -> None:
    record, _ = await run({"implementer": ["built it"], "reviewer": ["```json\n[]\n```"]}, tmp_path)
    assert record.cut_short == []


async def test_every_cut_short_turn_is_recorded_with_its_role(tmp_path: Path) -> None:
    class CutShortAll(ScriptedRunner):
        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            turn = await super().__call__(role, system, prompt, model, workspace, tools)
            return AgentTurn(text=turn.text, usage=turn.usage, cut_short="the turn stopped early: max_turns")

    runner = CutShortAll(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [NO_PROBE],
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        }
    )
    record = await pair_with(runner, tmp_path)
    assert [line.split(":")[0] for line in record.cut_short] == [
        "implementer",
        "reviewer",
        "implementer",
        "orchestrator_probe",
        "orchestrator_rule",
        "implementer",
    ]


async def test_a_raising_turn_carries_the_paid_usage_out_with_the_failure(tmp_path: Path) -> None:
    class RaisingRunner(ScriptedRunner):
        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            turn = await super().__call__(role, system, prompt, model, workspace, tools)
            if role == "reviewer":
                raise RuntimeError("transport died")
            return turn

    runner = RaisingRunner({"implementer": ["built it"], "reviewer": ["never seen"]})
    with pytest.raises(RunFailed) as raised:
        await run_pair(
            task_id="duration",
            spec="write parse_duration",
            module="duration.py",
            test="test_duration.py",
            workspace=tmp_path,
            implementer_model="sonnet",
            reviewer_model="opus",
            orchestrator_model="haiku",
            runner=runner,
        )
    record = raised.value.record
    assert record.failure == "RuntimeError: transport died"
    assert record.total_cost_usd == 0.001
    assert record.total_tokens == 15


async def test_a_declined_probe_with_trailing_prose_is_read_as_declined(tmp_path: Path) -> None:
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [f"{NO_PROBE}\n\nThis is a naming question, nothing to run."],
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        },
        tmp_path,
    )
    assert runner.calls.count("orchestrator_probe") == 1
    assert record.resolutions[0].method == "judged"
    assert record.probe_failures == []


class CutShortOn(ScriptedRunner):
    """Marks one named role's turns as cut short, as a turn or budget cap would."""

    def __init__(self, script, role: str) -> None:
        super().__init__(script)
        self.cut_role = role

    async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
        turn = await super().__call__(role, system, prompt, model, workspace, tools)
        if role == self.cut_role:
            return AgentTurn(text=turn.text, usage=turn.usage, cut_short="the turn stopped early: max_turns")
        return turn


async def pair_with(runner, workspace: Path):
    """Run the pair loop with a prepared runner.

    Args:
        runner: The runner to use.
        workspace: Directory the run works in.

    Returns:
        The run record.
    """
    return await run_pair(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=workspace,
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="haiku",
        runner=runner,
    )


async def test_a_cut_short_probe_turn_is_named_not_blamed_on_the_format(tmp_path: Path) -> None:
    runner = CutShortOn(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": ["I will write a script that"] * 3,
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        },
        "orchestrator_probe",
    )
    record = await pair_with(runner, tmp_path)
    assert record.probe_failures == [
        "f1: the probe turn was unusable because the turn stopped early: max_turns (after 3 attempts)"
    ]
    assert record.resolutions[0].method == "judged"


async def test_a_cut_short_ruling_is_named_not_blamed_on_the_format(tmp_path: Path) -> None:
    runner = CutShortOn(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [NO_PROBE],
            "orchestrator_rule": ["Weighing the two positions,"] * 3,
        },
        "orchestrator_rule",
    )
    record = await pair_with(runner, tmp_path)
    assert record.resolutions == []
    assert record.unresolved == [
        "f1: the orchestrator's ruling was unusable because the turn stopped early: max_turns (after 3 attempts)"
    ]


DECLINED_QUOTING_PYTHON = (
    f"For reference, the disputed function is:\n\n```\ndef parse_duration(text):\n"
    f"    return 1\n```\n\nThis is a naming disagreement, not a behavioural one.\n\n{NO_PROBE}"
)
DECLINED_QUOTING_PROSE = f"```\nrename f -> duration_seconds\n```\n\nNothing to run here.\n\n{NO_PROBE}"


async def test_a_fence_quoted_beside_a_declined_probe_is_not_run_as_the_probe(tmp_path: Path) -> None:
    # The code under discussion, quoted in a fence, beside an answer that declines to run
    # anything. Running the quotation records the dispute as settled with a probe, on an
    # observation nobody asked for, and tells the ruling turn that code was run.
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [DECLINED_QUOTING_PYTHON],
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        },
        tmp_path,
    )
    resolution = record.resolutions[0]
    assert runner.calls.count("orchestrator_probe") == 1
    assert resolution.method == "judged"
    assert resolution.probe_path is None
    assert "observation" not in resolution.rationale
    assert record.probe_failures == []
    assert not list(tmp_path.glob("probe_*.py"))


async def test_the_ruling_turn_is_not_told_a_probe_ran_when_none_did(tmp_path: Path) -> None:
    class PromptCapturingRunner(ScriptedRunner):
        """Keeps the prompt each role was given, so the ruling turn's premise can be read."""

        def __init__(self, script) -> None:
            super().__init__(script)
            self.prompts: dict[str, str] = {}

        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            self.prompts[role] = prompt
            return await super().__call__(role, system, prompt, model, workspace, tools)

    runner = PromptCapturingRunner(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [DECLINED_QUOTING_PYTHON],
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        }
    )
    record = await pair_with(runner, tmp_path)
    assert "You ran a probe" not in runner.prompts["orchestrator_rule"]
    assert "Nothing you could run would settle this" in runner.prompts["orchestrator_rule"]
    assert record.resolutions[0].method == "judged"


async def test_a_non_python_quote_beside_a_declined_probe_costs_no_retries(tmp_path: Path) -> None:
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [DECLINED_QUOTING_PROSE],
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        },
        tmp_path,
    )
    assert runner.calls.count("orchestrator_probe") == 1
    assert record.probe_failures == []
    assert record.resolutions[0].method == "judged"


@pytest.mark.parametrize(
    "answer",
    [
        "```python\nprint(5415)\n```",
        "```\nprint(5415)\n```",
        "NO PROBE",
    ],
)
async def test_a_probe_in_the_old_prose_format_is_retried_and_never_run(tmp_path: Path, answer: str) -> None:
    # The python fence, the untagged fence that stood in for it, and the NO PROBE line are
    # all gone. Each is now an answer that does not carry the object, which is retried and
    # then recorded as a probe failure, and the dispute is still ruled on.
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [answer] * 3,
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        },
        tmp_path,
    )
    assert runner.calls.count("orchestrator_probe") == 3
    assert record.probe_failures == ["f1: no ```json block found in the response (after 3 attempts)"]
    assert record.resolutions[0].method == "judged"
    assert not list(tmp_path.glob("probe_*.py"))


async def test_the_recorded_side_is_the_one_the_ruling_object_names(tmp_path: Path) -> None:
    # Prose that names the losing side, in the form the old contract matched on, above an
    # object that rules the other way. Every inversion the reviews found was this shape.
    ruling = (
        "The options are FAVOURS: IMPLEMENTER or FAVOURS: REVIEWER.\n\n"
        "The implementer argues this FAVOURS: IMPLEMENTER, but duration.py:12 returns 5400.\n\n"
        f"{RULE_FOR_REVIEWER}"
    )
    record, _ = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [NO_PROBE],
            "orchestrator_rule": [ruling],
        },
        tmp_path,
    )
    resolution = record.resolutions[0]
    assert resolution.favours == "reviewer"
    # The reasoning recorded is the one the object carried, not the prose above it.
    assert resolution.rationale == "duration.py:12 returns 5400"


async def test_a_ruling_that_is_only_the_format_restated_is_unresolved(tmp_path: Path) -> None:
    restated = "Answer in one of two ways:\nFAVOURS: IMPLEMENTER\nor\nFAVOURS: REVIEWER"
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [NO_PROBE],
            "orchestrator_rule": [restated] * 3,
        },
        tmp_path,
    )
    assert runner.calls.count("orchestrator_rule") == 3
    assert record.resolutions == []
    assert record.unresolved == [
        "f1: the orchestrator gave no usable verdict: no ```json block found in the response (after 3 attempts)"
    ]


async def test_a_blank_orchestrator_model_is_refused_before_any_turn(tmp_path: Path) -> None:
    runner = ScriptedRunner({"implementer": ["built it"]})
    for blank in ("", "   ", "\t\n"):
        with pytest.raises(ValueError, match="orchestrator_model must name the model"):
            await run_pair(
                task_id="duration",
                spec="write parse_duration",
                module="duration.py",
                test="test_duration.py",
                workspace=tmp_path,
                implementer_model="sonnet",
                reviewer_model="opus",
                orchestrator_model=blank,
                runner=runner,
            )
    assert runner.calls == []


async def test_an_accepted_finding_is_recorded_with_the_reason_given(tmp_path: Path) -> None:
    # Acceptance is the outcome of almost every finding and was the one with no record.
    record, _ = await run(
        {"implementer": ["built it", ACCEPT_ONE], "reviewer": [REVIEW_ONE]},
        tmp_path,
    )
    assert [(entry.finding.id, entry.reason) for entry in record.accepted] == [("f1", "fixed")]
    assert record.accepted[0].finding.claim == "returns 5400"
    assert record.disputes == []
    assert record.unanswered == []


async def test_acceptance_is_recorded_beside_a_rejection_of_another_finding(tmp_path: Path) -> None:
    record, _ = await run(
        {
            "implementer": ["built it", ACCEPT_AND_REJECT],
            "reviewer": ['```json\n[{"id": "f1", "claim": "a"}, {"id": "f2", "claim": "b"}]\n```'],
            "orchestrator_probe": [NO_PROBE],
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        },
        tmp_path,
    )
    assert [entry.finding.id for entry in record.accepted] == ["f1"]
    assert [dispute.finding.id for dispute in record.disputes] == ["f2"]
    assert record.accepted[0].reason == "applied"


async def test_an_accepted_finding_with_no_reason_is_still_recorded(tmp_path: Path) -> None:
    record, _ = await run(
        {
            "implementer": ["built it", '```json\n[{"id": "f1", "accept": true}]\n```'],
            "reviewer": [REVIEW_ONE],
        },
        tmp_path,
    )
    assert [(entry.finding.id, entry.reason) for entry in record.accepted] == [("f1", "")]


async def test_a_clean_review_accepts_nothing(tmp_path: Path) -> None:
    record, _ = await run({"implementer": ["built it"], "reviewer": ["```json\n[]\n```"]}, tmp_path)
    assert record.accepted == []


async def test_a_malformed_ruling_is_retried_and_the_second_answer_is_used(tmp_path: Path) -> None:
    # A ruling that does not parse is the orchestrator's instrument failing, on the same
    # terms as a probe that will not compile, so it is retried rather than abandoning a
    # dispute that has already cost four paid turns.
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [NO_PROBE],
            "orchestrator_rule": ['```json\n{"favours": "neither"}\n```', RULE_FOR_REVIEWER],
        },
        tmp_path,
    )
    assert runner.calls.count("orchestrator_rule") == 2
    assert record.resolutions[0].favours == "reviewer"
    assert record.unresolved == []


async def test_the_retry_tells_the_ruling_turn_what_was_wrong_with_its_last_answer(tmp_path: Path) -> None:
    class PromptLog(ScriptedRunner):
        """Keeps every prompt each role was given, in order."""

        def __init__(self, script) -> None:
            super().__init__(script)
            self.prompts: dict[str, list[str]] = {}

        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            self.prompts.setdefault(role, []).append(prompt)
            return await super().__call__(role, system, prompt, model, workspace, tools)

    runner = PromptLog(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [NO_PROBE],
            "orchestrator_rule": ["I cannot tell.", RULE_FOR_REVIEWER],
        }
    )
    await pair_with(runner, tmp_path)
    retry = runner.prompts["orchestrator_rule"][1]
    assert "no ```json block found in the response" in retry
    assert "Answer in the format you were given" in retry


async def test_every_attempt_at_a_ruling_is_recorded_when_the_cap_keeps_firing(tmp_path: Path) -> None:
    # The turns that retry are turns, and each one's cap firing reaches the record whether
    # or not the text it returned then parsed.
    runner = CutShortOn(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [NO_PROBE],
            "orchestrator_rule": ["Weighing the two positions,", "Weighing them again,", RULE_FOR_REVIEWER],
        },
        "orchestrator_rule",
    )
    record = await pair_with(runner, tmp_path)
    assert record.cut_short == ["orchestrator_rule: the turn stopped early: max_turns"] * 3
    assert record.resolutions[0].favours == "reviewer"
    assert record.unresolved == []


async def test_every_probe_attempt_is_recorded_when_the_cap_keeps_firing(tmp_path: Path) -> None:
    runner = CutShortOn(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": ["I will write a script that", "still writing", PROBE],
            "orchestrator_rule": [RULE_FOR_IMPLEMENTER],
        },
        "orchestrator_probe",
    )
    record = await pair_with(runner, tmp_path)
    assert record.cut_short == ["orchestrator_probe: the turn stopped early: max_turns"] * 3
    assert record.resolutions[0].method == "judged_with_probe"
    assert record.probe_failures == []


async def test_a_run_records_when_it_started(tmp_path: Path) -> None:
    record, _ = await run({"implementer": ["built it"], "reviewer": ["```json\n[]\n```"]}, tmp_path)
    started = datetime.fromisoformat(record.started_at)
    assert started.tzinfo is not None
    assert started.utcoffset().total_seconds() == 0


async def test_a_ruling_for_the_reviewer_sends_the_implementer_back_to_apply_it(tmp_path: Path) -> None:
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE, "applied the ruling"],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [PROBE],
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        },
        tmp_path,
    )
    assert [resolution.favours for resolution in record.resolutions] == ["reviewer"]
    assert record.sent_back_after_ruling == ["f1"]
    # The remediation turn is the implementer's third, and it comes after the ruling.
    assert runner.calls == [
        "implementer",
        "reviewer",
        "implementer",
        "orchestrator_probe",
        "orchestrator_rule",
        "implementer",
    ]


async def test_a_ruling_for_the_implementer_changes_nothing_and_costs_no_extra_turn(tmp_path: Path) -> None:
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [PROBE],
            "orchestrator_rule": [RULE_FOR_IMPLEMENTER],
        },
        tmp_path,
    )
    assert [resolution.favours for resolution in record.resolutions] == ["implementer"]
    assert record.sent_back_after_ruling == []
    assert runner.calls.count("implementer") == 2


async def test_an_unresolved_dispute_is_never_sent_back_to_be_applied(tmp_path: Path) -> None:
    record, runner = await run(
        {
            "implementer": ["built it", REJECT_ONE],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [NO_PROBE],
            "orchestrator_rule": ["I cannot decide this."] * 3,
        },
        tmp_path,
    )
    assert record.resolutions == []
    assert len(record.unresolved) == 1
    assert record.sent_back_after_ruling == []
    assert runner.calls.count("implementer") == 2


async def test_the_remediation_turn_is_shown_the_ruling_that_overrode_it(tmp_path: Path) -> None:
    prompts: list[str] = []

    class Recording(ScriptedRunner):
        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            prompts.append(prompt)
            return await super().__call__(role, system, prompt, model, workspace, tools)

    runner = Recording(
        {
            "implementer": ["built it", REJECT_ONE, "applied it"],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [PROBE],
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        }
    )
    await run_pair(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="haiku",
        runner=runner,
    )
    remediation = prompts[-1]
    assert "f1: returns 5400" in remediation
    assert "you rejected it, saying: it returns 5415" in remediation
    assert "duration.py:12 returns 5400" in remediation
    assert "duration.py" in remediation and "test_duration.py" in remediation


async def test_two_upheld_rulings_are_applied_in_one_turn(tmp_path: Path) -> None:
    review_two = (
        '```json\n[{"id": "f1", "claim": "returns 5400", "evidence": "duration.py:12"},'
        ' {"id": "f2", "claim": "no test for days", "evidence": "test_duration.py has none"}]\n```'
    )
    reject_two = (
        '```json\n[{"id": "f1", "accept": false, "reason": "it returns 5415"},'
        ' {"id": "f2", "accept": false, "reason": "days are out of scope"}]\n```'
    )
    record, runner = await run(
        {
            "implementer": ["built it", reject_two, "applied both"],
            "reviewer": [review_two],
            "orchestrator_probe": [NO_PROBE, NO_PROBE],
            "orchestrator_rule": [RULE_FOR_REVIEWER, RULE_FOR_REVIEWER],
        },
        tmp_path,
    )
    assert record.sent_back_after_ruling == ["f1", "f2"]
    assert runner.calls.count("implementer") == 3


async def test_a_remediation_turn_stopped_by_a_cap_is_recorded(tmp_path: Path) -> None:
    class CutShort(ScriptedRunner):
        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            turn = await super().__call__(role, system, prompt, model, workspace, tools)
            if turn.text == "partial":
                return AgentTurn(text=turn.text, usage=turn.usage, cut_short="the turn hit its budget")
            return turn

    runner = CutShort(
        {
            "implementer": ["built it", REJECT_ONE, "partial"],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [NO_PROBE],
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        }
    )
    record = await run_pair(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="haiku",
        runner=runner,
    )
    assert record.sent_back_after_ruling == ["f1"]
    assert record.cut_short == ["implementer: the turn hit its budget"]


async def test_the_orchestrator_is_shown_the_spec_and_the_filenames(tmp_path: Path) -> None:
    prompts: dict[str, str] = {}

    class Recording(ScriptedRunner):
        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            prompts.setdefault(role, prompt)
            return await super().__call__(role, system, prompt, model, workspace, tools)

    runner = Recording(
        {
            "implementer": ["built it", REJECT_ONE, "applied it"],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [NO_PROBE],
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        }
    )
    await run_pair(
        task_id="duration",
        spec="parse durations like 1h30m into seconds",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="haiku",
        runner=runner,
    )
    # Both turns rule on the dispute, so both need the spec and the names of the files.
    for role in ("orchestrator_probe", "orchestrator_rule"):
        assert "parse durations like 1h30m into seconds" in prompts[role]
        assert "duration.py" in prompts[role]
        assert "test_duration.py" in prompts[role]
        assert "Reviewer claims: returns 5400" in prompts[role]


async def test_the_probe_turn_is_told_which_module_to_import(tmp_path: Path) -> None:
    prompts: list[str] = []

    class Recording(ScriptedRunner):
        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            if role == "orchestrator_probe":
                prompts.append(prompt)
            return await super().__call__(role, system, prompt, model, workspace, tools)

    runner = Recording(
        {
            "implementer": ["built it", REJECT_ONE, "applied it"],
            "reviewer": [REVIEW_ONE],
            "orchestrator_probe": [PROBE],
            "orchestrator_rule": [RULE_FOR_REVIEWER],
        }
    )
    await run_pair(
        task_id="duration",
        spec="parse durations",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="haiku",
        runner=runner,
    )
    assert "It was implemented in duration.py, with tests in test_duration.py." in prompts[0]


async def test_a_reviewer_that_misformats_once_is_re_asked_and_its_findings_kept(tmp_path: Path) -> None:
    record, runner = await run(
        {
            "implementer": ["built it", ACCEPT_ONE],
            # A review wrapped in an object used to end the run outright.
            "reviewer": ['```json\n{"findings": [{"id": "f1", "claim": "returns 5400"}]}\n```', REVIEW_ONE],
        },
        tmp_path,
    )
    assert record.failure is None
    assert [finding.id for finding in record.findings] == ["f1"]
    assert runner.calls.count("reviewer") == 2


async def test_the_reviewer_retry_is_told_what_was_wrong_with_its_last_answer(tmp_path: Path) -> None:
    prompts: list[str] = []

    class Recording(ScriptedRunner):
        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            if role == "reviewer":
                prompts.append(prompt)
            return await super().__call__(role, system, prompt, model, workspace, tools)

    runner = Recording({"implementer": ["built it", ACCEPT_ONE], "reviewer": ["no fence at all", REVIEW_ONE]})
    await run_pair(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="haiku",
        runner=runner,
    )
    assert "Your last answer could not be used" in prompts[1]
    assert "no ```json block found" in prompts[1]
    # The task and the files are still in the re-ask, not only the complaint.
    assert "Review duration.py and test_duration.py" in prompts[1]


async def test_a_reviewer_cut_short_every_time_names_the_cap_not_the_format(tmp_path: Path) -> None:
    class CutShortReviewer(ScriptedRunner):
        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            turn = await super().__call__(role, system, prompt, model, workspace, tools)
            if role == "reviewer":
                return AgentTurn(text=turn.text, usage=turn.usage, cut_short="the turn stopped early: max_turns")
            return turn

    runner = CutShortReviewer({"implementer": ["built it"], "reviewer": ["partial"] * 3})
    record = await run_pair(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="haiku",
        runner=runner,
    )
    assert record.failure == "reviewer answer unusable because the turn stopped early: max_turns (after 3 attempts)"
    assert "did not parse" not in record.failure
    assert len(record.cut_short) == 3


async def test_a_clean_review_costs_exactly_one_reviewer_turn(tmp_path: Path) -> None:
    record, runner = await run(
        {"implementer": ["built it"], "reviewer": ["```json\n[]\n```"]},
        tmp_path,
    )
    assert record.findings == []
    assert record.failure is None
    assert runner.calls.count("reviewer") == 1


async def test_an_implementer_that_misformats_once_is_re_asked_and_its_decisions_kept(tmp_path: Path) -> None:
    record, runner = await run(
        {
            # A decision list wrapped in an object used to end the run outright.
            "implementer": ["built it", '```json\n{"decisions": [{"id": "f1", "accept": true}]}\n```', ACCEPT_ONE],
            "reviewer": [REVIEW_ONE],
        },
        tmp_path,
    )
    assert record.failure is None
    assert [acceptance.finding.id for acceptance in record.accepted] == ["f1"]
    assert runner.calls.count("implementer") == 3


async def test_the_decision_retry_tells_the_implementer_its_edits_already_stand(tmp_path: Path) -> None:
    prompts: list[str] = []

    class Recording(ScriptedRunner):
        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            if role == "implementer":
                prompts.append(prompt)
            return await super().__call__(role, system, prompt, model, workspace, tools)

    runner = Recording(
        {"implementer": ["built it", "I disagree with all of it.", ACCEPT_ONE], "reviewer": [REVIEW_ONE]}
    )
    await run_pair(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="haiku",
        runner=runner,
    )
    retry = prompts[2]
    assert "Your last answer could not be used" in retry
    # The turn edits files, so the re-ask must not read as an order to apply them again.
    assert "do not apply them a second time" in retry
    assert "f1: returns 5400" in retry


async def test_an_implementer_unreadable_every_time_fails_the_run_after_three_attempts(tmp_path: Path) -> None:
    record, runner = await run(
        {"implementer": ["built it", "no", "still no", "nope"], "reviewer": [REVIEW_ONE]},
        tmp_path,
    )
    assert "implementer answer did not parse" in record.failure
    assert "after 3 attempts" in record.failure
    assert runner.calls.count("implementer") == 4
    assert record.unanswered == []
    assert record.disputes == []


async def test_a_readable_decision_costs_exactly_one_implementer_answer_turn(tmp_path: Path) -> None:
    record, runner = await run(
        {"implementer": ["built it", ACCEPT_ONE], "reviewer": [REVIEW_ONE]},
        tmp_path,
    )
    assert record.failure is None
    assert runner.calls.count("implementer") == 2
