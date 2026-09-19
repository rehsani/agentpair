"""Tests for the command line: the arbitration guard, the console line, and the arms loop.

The model runner is the only thing replaced. Everything else here is the real command
line, parsed from a real argv, writing a real results file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentpair import cli
from agentpair.models import Dispute, Finding, Resolution, RoleUsage, RunRecord


def make_args(argv: list[str], monkeypatch: pytest.MonkeyPatch):
    """Parse a real command line.

    Args:
        argv: Arguments after the program name.
        monkeypatch: Fixture used to set sys.argv.

    Returns:
        The parsed namespace.
    """
    monkeypatch.setattr("sys.argv", ["agentpair", *argv])
    return cli.parse_args()


def test_the_judge_flag_is_required_for_the_pair_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    args = make_args(["--task", "t.md", "--module", "m.py"], monkeypatch)
    with pytest.raises(SystemExit, match="--judge is required"):
        cli.resolve_judge(args)


def test_the_refusal_names_both_ways_out(monkeypatch: pytest.MonkeyPatch) -> None:
    args = make_args(["--task", "t.md", "--module", "m.py", "--reviewer-model", "opus"], monkeypatch)
    with pytest.raises(SystemExit, match="(?s)--judge reviewer.*--judge haiku"):
        cli.resolve_judge(args)


def test_judge_reviewer_reuses_the_reviewer_model_and_accepts_that_it_is_a_disputant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = make_args(
        ["--task", "t.md", "--module", "m.py", "--reviewer-model", "opus", "--judge", "reviewer"], monkeypatch
    )
    assert cli.resolve_judge(args) == ("opus", True)


@pytest.mark.parametrize("spelling", ["reviewer", "Reviewer", "REVIEWER", " reviewer "])
def test_the_reviewer_choice_is_read_however_it_is_cased(spelling: str, monkeypatch: pytest.MonkeyPatch) -> None:
    args = make_args(["--task", "t.md", "--module", "m.py", "--judge", spelling], monkeypatch)
    assert cli.resolve_judge(args) == ("opus", True)


def test_a_third_model_judges_as_a_third_party(monkeypatch: pytest.MonkeyPatch) -> None:
    args = make_args(["--task", "t.md", "--module", "m.py", "--judge", "haiku"], monkeypatch)
    assert cli.resolve_judge(args) == ("haiku", False)


def test_the_implementer_model_is_never_accepted_as_the_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    # The implementer has already written down its position, so there is no opt-in for it.
    args = make_args(
        ["--task", "t.md", "--module", "m.py", "--implementer-model", "sonnet", "--judge", "sonnet"],
        monkeypatch,
    )
    with pytest.raises(SystemExit, match="the implementer's"):
        cli.resolve_judge(args)


def test_naming_the_reviewers_model_directly_points_at_the_reviewer_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    args = make_args(
        ["--task", "t.md", "--module", "m.py", "--implementer-model", "sonnet", "--judge", "opus"],
        monkeypatch,
    )
    with pytest.raises(SystemExit, match="(?s)reviewer's.*--judge reviewer to choose that deliberately"):
        cli.resolve_judge(args)


@pytest.mark.parametrize(
    ("argv", "named"),
    [
        (["--judge", "Opus"], "reviewer"),
        (["--judge", " opus"], "reviewer"),
        (["--judge", "\topus\n"], "reviewer"),
        (["--implementer-model", "sonnet", "--judge", "Sonnet"], "implementer"),
        (["--reviewer-model", "Haiku", "--judge", "haiku"], "reviewer"),
        (["--implementer-model", "GPT-5", "--judge", "gpt-5"], "implementer"),
    ],
)
def test_the_guard_is_not_defeated_by_case_or_whitespace(
    argv: list[str],
    named: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = make_args(["--task", "t.md", "--module", "m.py", *argv], monkeypatch)
    with pytest.raises(SystemExit, match=f"(?s)refusing to run.*{named}"):
        cli.resolve_judge(args)


@pytest.mark.parametrize(
    "argv",
    [
        ["--judge", "haiku"],
        ["--judge", "opusx"],
        ["--judge", "claude-opus-4-5-20251101"],
        ["--reviewer-model", "opus", "--judge", "sonnet-4-6"],
    ],
)
def test_a_name_that_is_merely_similar_is_still_allowed(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    # The guard is nominal: "claude-opus-4-5-20251101" is the reviewer's model under
    # another name and nothing here can tell. The docstring says so.
    args = make_args(["--task", "t.md", "--module", "m.py", *argv], monkeypatch)
    assert cli.resolve_judge(args) == (argv[-1], False)


def make_record() -> RunRecord:
    """Return a finished pair record with one resolved dispute.

    Returns:
        The record.
    """
    finding = Finding(id="f1", claim="drops seconds", evidence="duration.py:12")
    dispute = Dispute(finding=finding, implementer_position="it returns 5415")
    return RunRecord(
        task_id="duration",
        arm="pair",
        implementer_model="sonnet",
        reviewer_model="opus",
        orchestrator_model="haiku",
        findings=[finding],
        disputes=[dispute],
        resolutions=[Resolution(dispute=dispute, favours="implementer", method="judged", rationale="clear")],
        usage=[RoleUsage(role="reviewer", input_tokens=10, output_tokens=5, cost_usd=0.02, seconds=1.0)],
        suite_status="passed",
        tests_passed=3,
    )


def test_summarise_stays_quiet_when_nothing_degraded() -> None:
    line = cli.summarise(make_record())
    assert "protocol_errors" not in line
    assert "probe_failures" not in line
    assert "suite=passed" in line


def test_summarise_reports_protocol_errors_and_probe_failures() -> None:
    record = make_record()
    record.protocol_errors = ["f9: decision names no known finding"]
    record.probe_failures = ["f1: the script did not compile (after 3 attempts)"]
    line = cli.summarise(record)
    assert "protocol_errors=1" in line
    assert "probe_failures=1" in line


def test_summarise_reports_turns_that_ended_early() -> None:
    record = make_record()
    record.cut_short = ["reviewer: the turn stopped early: max_turns"]
    line = cli.summarise(record)
    assert "cut_short=1" in line
    assert "FAILED" not in line


class ScriptedRunner:
    """Returns a queued response per role, so the arms loop runs without a paid call."""

    def __init__(self, script: dict[str, list[str]]) -> None:
        self.script = {role: list(texts) for role, texts in script.items()}

    async def __call__(self, role, system, prompt, model, workspace, tools):
        from agentpair.models import AgentTurn

        queued = self.script.get(role) or ["(nothing scripted)"]
        text = queued.pop(0) if queued else "(exhausted)"
        usage = RoleUsage(role=role, input_tokens=10, output_tokens=5, cost_usd=0.25, seconds=0.1)
        return AgentTurn(text=text, usage=usage)


class RaisingRunner(ScriptedRunner):
    """Answers the implementer, then dies on the review, as a transport error would."""

    async def __call__(self, role, system, prompt, model, workspace, tools):
        turn = await super().__call__(role, system, prompt, model, workspace, tools)
        if role == "reviewer":
            raise RuntimeError("transport died")
        return turn


async def test_a_crashed_pair_run_still_reports_what_it_spent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = tmp_path / "duration.md"
    task.write_text("write parse_duration")
    results = tmp_path / "results.jsonl"
    args = make_args(
        [
            "--task",
            str(task),
            "--module",
            "duration.py",
            "--judge",
            "haiku",
            "--workspace",
            str(tmp_path / "runs"),
        ],
        monkeypatch,
    )
    monkeypatch.setattr(cli, "sdk_runner", RaisingRunner({"implementer": ["built it"]}))

    records = await cli.run_arms(args, results)

    assert records[0].failure == "RuntimeError: transport died"
    assert records[0].total_cost_usd == 0.25
    written = json.loads(results.read_text().strip())
    assert written["total_cost_usd"] == 0.25
    assert written["orchestrator_model"] == "haiku"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tasks/duration.md", "duration"),
        ("benchmark/bigcodebench/BigCodeBench_310/spec.md", "BigCodeBench_310"),
        ("benchmark/bigcodebench/BigCodeBench_592/SPEC.md", "BigCodeBench_592"),
        ("benchmark/bigcodebench/BigCodeBench_592/task.md", "BigCodeBench_592"),
        ("benchmark/bigcodebench/BigCodeBench_592/README.md", "BigCodeBench_592"),
        ("benchmark/bigcodebench/BigCodeBench_592/prompt.md", "BigCodeBench_592"),
        ("tasks/specification.md", "specification"),
        ("tasks/spectrum.md", "spectrum"),
    ],
)
def test_a_generic_stem_takes_the_name_from_the_directory_holding_it(
    path: str,
    expected: str,
    tmp_path: Path,
) -> None:
    # Twenty benchmark tasks each keep their specification in a spec.md, so the stem names
    # the kind of file and the directory names the task. Taking the stem would write twenty
    # results lines all reading task_id="spec" into one workspace that is emptied per run.
    assert cli.resolve_task_id(None, tmp_path / path) == expected


@pytest.mark.parametrize("given", ["BigCodeBench_310", " BigCodeBench_310 "])
def test_an_explicit_task_id_overrides_the_path(given: str, tmp_path: Path) -> None:
    assert cli.resolve_task_id(given, tmp_path / "benchmark/BigCodeBench_310/spec.md") == "BigCodeBench_310"


def test_a_blank_task_id_falls_back_to_the_path(tmp_path: Path) -> None:
    assert cli.resolve_task_id("   ", tmp_path / "tasks/duration.md") == "duration"


async def test_the_task_id_names_the_record_the_results_line_and_the_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_dir = tmp_path / "benchmark" / "BigCodeBench_310"
    task_dir.mkdir(parents=True)
    (task_dir / "spec.md").write_text("write parse_duration")
    results = tmp_path / "results.jsonl"
    args = make_args(
        [
            "--task",
            str(task_dir / "spec.md"),
            "--module",
            "duration.py",
            "--judge",
            "haiku",
            "--workspace",
            str(tmp_path / "runs"),
        ],
        monkeypatch,
    )
    monkeypatch.setattr(
        cli,
        "sdk_runner",
        ScriptedRunner({"implementer": ["built it"], "reviewer": ["```json\n[]\n```"]}),
    )

    records = await cli.run_arms(args, results)

    assert records[0].task_id == "BigCodeBench_310"
    assert json.loads(results.read_text().strip())["task_id"] == "BigCodeBench_310"
    assert (tmp_path / "runs" / "BigCodeBench_310_pair").is_dir()


class BaselineRaisingRunner(ScriptedRunner):
    """Dies on the single baseline turn, as a transport error would."""

    async def __call__(self, role, system, prompt, model, workspace, tools):
        raise RuntimeError("transport died")


async def test_a_crashed_baseline_run_is_accounted_for_like_a_crashed_pair_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = tmp_path / "duration.md"
    task.write_text("write parse_duration")
    results = tmp_path / "results.jsonl"
    args = make_args(
        [
            "--task",
            str(task),
            "--arm",
            "baseline",
            "--module",
            "duration.py",
            "--workspace",
            str(tmp_path / "runs"),
        ],
        monkeypatch,
    )
    monkeypatch.setattr(cli, "sdk_runner", BaselineRaisingRunner({}))

    records = await cli.run_arms(args, results)

    assert records[0].failure == "RuntimeError: transport died"
    assert records[0].seconds > 0.0
    written = json.loads(results.read_text().strip())
    assert written["failure"] == "RuntimeError: transport died"
    assert written["arm"] == "baseline"
