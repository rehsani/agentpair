"""Scoring a produced solution against the benchmark's own held-out tests.

The agents never see these tests. They are written by the benchmark, not by the agent, so
they measure whether the code is correct rather than whether the agent agreed with itself.
That distinction is the whole point of scoring separately from `verify.run_suite`, which
runs the suite the agent wrote.

Scoring happens in a throwaway copy. The run workspace is the artifact of the experiment
and is read afterwards, so the held-out test is never written into it: an answer left in
the workspace would be visible to anything that looks at the workspace later, and would
make the run indistinguishable from one that had been given the answer.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentpair.verify import DEFAULT_SUITE_TIMEOUT_SECONDS, SuiteStatus, run_suite

# The filename the benchmark's tests import from, as `from solution import *`. A solution
# under any other name scores zero however correct it is, so the name is fixed here.
SOLUTION_MODULE = "solution.py"

# The benchmark's own tests, kept outside the repository so an agent cannot read them.
HIDDEN_TEST = "hidden_test.py"

# The canonical solution, scored only to prove the held-out test is sound.
REFERENCE_MODULE = "reference.py"


@dataclass(frozen=True)
class Score:
    """What the held-out tests said about one solution.

    Attributes:
        task_id: The task that was scored.
        arm: Which arm produced the solution, or "reference" when the canonical solution
            was scored to check the test itself.
        status: "passed", "failed", "missing" when no solution was produced, or "error"
            when the tests gave no usable verdict.
        passed: How many held-out tests passed.
        failed: How many failed or errored.
        output: The tail of pytest's output, kept so a failure is diagnosable.
    """

    task_id: str
    arm: str
    status: SuiteStatus
    passed: int
    failed: int
    output: str

    @property
    def solved(self) -> bool:
        """Return whether the solution passed every held-out test.

        A partially passing solution is not solved. The benchmark's tests are the
        specification made executable, so anything short of all of them means the task was
        not done.
        """
        return self.status == "passed" and self.failed == 0 and self.passed > 0


def score_solution(
    solution_path: Path,
    answers_dir: Path,
    task_id: str,
    arm: str,
    timeout: int = DEFAULT_SUITE_TIMEOUT_SECONDS,
) -> Score:
    """Run one task's held-out tests against one produced solution.

    Args:
        solution_path: The solution file to score. Copied, never modified.
        answers_dir: Directory holding this task's hidden_test.py.
        task_id: Identifier recorded with the score.
        arm: Which arm produced this solution, recorded with the score.
        timeout: Seconds before the held-out suite is killed.

    Returns:
        The score. A solution that was never written is reported as "missing" rather than
        raising, because an agent failing to produce one is itself an outcome.

    Raises:
        FileNotFoundError: If the task's held-out test is absent, which means the answers
            were not unpacked and every score would otherwise read as a failure.
    """
    hidden_test = answers_dir / HIDDEN_TEST
    if not hidden_test.exists():
        raise FileNotFoundError(f"no {HIDDEN_TEST} in {answers_dir}; are the answers unpacked?")

    if not solution_path.exists():
        return Score(
            task_id=task_id,
            arm=arm,
            status="missing",
            passed=0,
            failed=0,
            output=f"{solution_path.name} was never written",
        )

    with tempfile.TemporaryDirectory() as scratch:
        arena = Path(scratch)
        shutil.copy(solution_path, arena / SOLUTION_MODULE)
        shutil.copy(hidden_test, arena / HIDDEN_TEST)
        result = run_suite(arena, HIDDEN_TEST, timeout=timeout)

    return Score(
        task_id=task_id,
        arm=arm,
        status=result.status,
        passed=result.passed,
        failed=result.failed,
        output=result.output,
    )


def score_reference(
    answers_dir: Path,
    task_id: str,
    timeout: int = DEFAULT_SUITE_TIMEOUT_SECONDS,
) -> Score:
    """Score the canonical solution, to prove the held-out test is sound.

    A held-out test the reference cannot pass is a broken test, and every agent scored
    against it would be marked wrong for the benchmark's own mistake. Checking it costs a
    subprocess and removes that reading entirely.

    Args:
        answers_dir: Directory holding this task's hidden_test.py and reference.py.
        task_id: Identifier recorded with the score.
        timeout: Seconds before the held-out suite is killed.

    Returns:
        The reference's score, which should be "passed" for every task.
    """
    return score_solution(
        solution_path=answers_dir / REFERENCE_MODULE,
        answers_dir=answers_dir,
        task_id=task_id,
        arm="reference",
        timeout=timeout,
    )
