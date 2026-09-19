"""The single-agent control arm.

One agent implements the task, writes its own tests and declares itself done, with no
review. It uses the same runner, the same accounting, the same spec text and the same
style brief as the pair arm, and the same model as its implementer.

The arms do not differ in exactly one thing, and the docstring used to claim they did.
The caps in sdk_runner are per turn, so the pair's implementer gets two turns of thirty
internal turns and two dollars each while this arm gets one of each. The arms therefore
differ in review and in the compute ceiling the implementing model is given. The claim is
stated rather than fixed because equalising the ceiling means a budget carried across
turns, which is a run-level budget and a separate piece of work. On the measured runs so
far the ceiling has not bound: the pair implementer spent well under one turn's cap across
both its turns, so the asymmetry is in the ceiling, not in what was spent. A run where the
cap does fire now records it, per turn, on RunRecord.cut_short, so this confound is visible
in the record rather than inferred.
"""

from __future__ import annotations

import time
from pathlib import Path

from agentpair import roles
from agentpair.metrics import merge_usage
from agentpair.models import RunFailed, Runner, RunRecord
from agentpair.orchestrator import AUTHORING_TOOLS
from agentpair.verify import run_suite


async def run_baseline(
    task_id: str,
    spec: str,
    module: str,
    test: str,
    workspace: Path,
    model: str,
    runner: Runner,
    style: str = roles.DEFAULT_STYLE,
) -> RunRecord:
    """Run one task through the single-agent arm and return its record.

    Args:
        task_id: Identifier recorded with the run.
        spec: The task specification, identical to the one the pair arm receives.
        module: Implementation filename to produce.
        test: Test filename to produce.
        workspace: Directory the agent works in.
        model: Model for the lone agent.
        runner: The model runner.
        style: Style brief handed to the agent.

    Returns:
        The run record. It carries no findings and no resolutions, since nothing reviewed
        the work, and its reviewer_model stays None. A turn stopped by a cap is recorded
        on cut_short here as it is in the pair arm, so the one arm cannot look clean while
        the other reports its caps.

    Raises:
        RunFailed: If the turn raised. The exception carries the record, with the failure
            and the elapsed time already on it, exactly as run_pair does. Without it the
            caller keeps the empty record it built beforehand and writes the crashed run
            down as costing nothing, while a crashed pair run reports what it spent, so
            the two arms being compared would account for failure differently and the
            comparison would favour whichever arm crashed.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = RunRecord(task_id=task_id, arm="baseline", implementer_model=model)

    try:
        built = await runner(
            role="implementer",
            system=roles.BASELINE_SYSTEM.substitute(module=module, test=test, style=style),
            prompt=spec,
            model=model,
            workspace=workspace,
            tools=AUTHORING_TOOLS,
        )
    except Exception as error:
        record.failure = f"{type(error).__name__}: {error}"
        record.seconds = time.monotonic() - started
        raise RunFailed(record) from error

    record.usage = merge_usage([built.usage])
    if built.cut_short:
        record.cut_short.append(f"implementer: {built.cut_short}")

    suite = run_suite(workspace, test)
    record.suite_status = suite.status
    record.tests_passed = suite.passed
    record.tests_failed = suite.failed
    record.seconds = time.monotonic() - started
    return record
