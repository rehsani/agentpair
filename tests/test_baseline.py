"""Tests for the single-agent control arm."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentpair.baseline import run_baseline
from agentpair.models import AgentTurn, RoleUsage, RunFailed


class RecordingRunner:
    """Captures the single call the baseline is allowed to make."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
        self.calls.append({"role": role, "system": system, "prompt": prompt, "model": model})
        usage = RoleUsage(role=role, input_tokens=100, output_tokens=20, cost_usd=0.01, seconds=1.0)
        return AgentTurn(text="built it", usage=usage)


async def test_baseline_makes_exactly_one_call(tmp_path: Path) -> None:
    runner = RecordingRunner()
    record = await run_baseline(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        model="opus",
        runner=runner,
    )
    assert len(runner.calls) == 1
    assert runner.calls[0]["role"] == "implementer"
    assert "no reviewer" in runner.calls[0]["system"]
    assert record.arm == "baseline"
    assert record.reviewer_model is None
    assert record.findings == []
    assert record.total_tokens == 120


async def test_baseline_passes_the_spec_through_unchanged(tmp_path: Path) -> None:
    runner = RecordingRunner()
    await run_baseline(
        task_id="duration",
        spec="the exact spec text",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        model="opus",
        runner=runner,
    )
    assert runner.calls[0]["prompt"] == "the exact spec text"


async def test_a_baseline_turn_stopped_by_a_cap_is_recorded(tmp_path: Path) -> None:
    class CutShortRunner(RecordingRunner):
        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            turn = await super().__call__(role, system, prompt, model, workspace, tools)
            return AgentTurn(text=turn.text, usage=turn.usage, cut_short="the turn stopped early: max_turns")

    record = await run_baseline(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        model="opus",
        runner=CutShortRunner(),
    )
    assert record.cut_short == ["implementer: the turn stopped early: max_turns"]


async def test_a_baseline_turn_that_ran_to_completion_records_nothing(tmp_path: Path) -> None:
    record = await run_baseline(
        task_id="duration",
        spec="write parse_duration",
        module="duration.py",
        test="test_duration.py",
        workspace=tmp_path,
        model="opus",
        runner=RecordingRunner(),
    )
    assert record.cut_short == []


async def test_a_crashed_baseline_turn_carries_its_record_out_with_the_failure(tmp_path: Path) -> None:
    class RaisingRunner(RecordingRunner):
        async def __call__(self, role, system, prompt, model, workspace, tools) -> AgentTurn:
            await super().__call__(role, system, prompt, model, workspace, tools)
            raise RuntimeError("transport died")

    with pytest.raises(RunFailed) as raised:
        await run_baseline(
            task_id="duration",
            spec="write parse_duration",
            module="duration.py",
            test="test_duration.py",
            workspace=tmp_path,
            model="opus",
            runner=RaisingRunner(),
        )

    # The pair arm already does this. Without it the caller keeps the empty record it built
    # beforehand, and the two arms being compared account for a crash differently.
    record = raised.value.record
    assert record.arm == "baseline"
    assert record.implementer_model == "opus"
    assert record.failure == "RuntimeError: transport died"
    assert record.seconds > 0.0
    assert record.suite_status is None
