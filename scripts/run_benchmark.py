"""Run every benchmark task through both arms and record what each one produced.

One task at a time, both arms, sequentially. The runs are not parallelised on purpose:
wall clock is one of the measured quantities, and concurrent runs competing for CPU and
network inflate every duration, which would make the pair-versus-baseline timing
comparison meaningless.

The sweep is resumable. Each finished run appends a line to the results file, and a task
whose (task_id, arm) pair is already in that file is skipped, so a sweep that dies part way
through does not re-buy the runs that already succeeded.

A task that fails does not stop the sweep. agentpair records the failure on that task's
own record, and this script names every non-zero exit in the summary it prints at the end.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# The module filename every task must produce. The benchmark's held-out tests open with
# "from solution import *", so a different name scores zero however good the code is.
SOLUTION_MODULE = "solution.py"

# The file inside each task directory holding the text the agents are given.
SPEC_FILENAME = "spec.md"

ARMS = ("pair", "baseline")


@dataclass(frozen=True)
class TaskRun:
    """One task's outcome in this sweep.

    Attributes:
        task_id: Directory name of the task, which is also how agentpair records it.
        skipped: True when both arms were already in the results file.
        returncode: agentpair's exit status, or None when the task was skipped.
        seconds: Wall clock spent on this task, including both arms.
    """

    task_id: str
    skipped: bool
    returncode: int | None
    seconds: float


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run every benchmark task through both arms, one at a time.")
    parser.add_argument(
        "--tasks-dir",
        required=True,
        help="directory holding one subdirectory per task, each with a spec.md",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="directory the run workspaces, the results file and the per-task logs are written to",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="model used for every role in both arms unless overridden below, so by default the only "
        "difference between the arms is the review mechanism and not model capability",
    )
    parser.add_argument(
        "--implementer-model",
        default=None,
        help="override the implementer's model, which is also the baseline arm's model. Naming a "
        "different model here and for the reviewer deliberately breaks the same-model design, which "
        "is what isolates the reviewer's capability as the variable",
    )
    parser.add_argument(
        "--reviewer-model",
        default=None,
        help="override the reviewer's model. The judge is always the reviewer, so this is also the "
        "model that rules on disputes",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="run only this task directory name; repeatable. Default is every task found",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many tasks, used to validate the sweep on a couple before going wide",
    )
    parser.add_argument(
        "--results-name",
        default="results.jsonl",
        help="filename inside --results-dir that each finished run is appended to",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command for each task without running anything",
    )
    return parser.parse_args()


def find_tasks(
    tasks_dir: Path,
    wanted: list[str] | None,
) -> list[Path]:
    """Return the task directories to run, in a stable order.

    Args:
        tasks_dir: Directory holding one subdirectory per task.
        wanted: Names to restrict to, or None for every task found.

    Returns:
        The task directories, sorted by name so a resumed sweep visits them in the same
        order as the sweep it is resuming.

    Raises:
        SystemExit: If a name given with --task does not exist, which is a typo worth
            stopping for rather than silently running a shorter sweep.
    """
    found = sorted(path for path in tasks_dir.iterdir() if path.is_dir() and (path / SPEC_FILENAME).exists())
    if wanted is None:
        return found

    by_name = {path.name: path for path in found}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise SystemExit(f"no such task directory: {', '.join(missing)}")
    return [by_name[name] for name in wanted]


def completed_arms(results_path: Path) -> set[tuple[str, str]]:
    """Return the (task_id, arm) pairs already recorded, so they are not run again.

    A line that does not parse is ignored rather than fatal: the results file is appended
    to by a separate process, and a truncated final line from a killed sweep should cost
    one re-run, not the whole resume.

    Args:
        results_path: The results file, which need not exist yet.

    Returns:
        Every (task_id, arm) pair found.
    """
    if not results_path.exists():
        return set()

    done: set[tuple[str, str]] = set()
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id, arm = record.get("task_id"), record.get("arm")
        if isinstance(task_id, str) and isinstance(arm, str):
            done.add((task_id, arm))
    return done


def build_command(
    task_dir: Path,
    results_dir: Path,
    results_name: str,
    implementer_model: str,
    reviewer_model: str,
) -> list[str]:
    """Return the agentpair invocation for one task.

    When both models are the same, the two arms differ only in the review mechanism, which
    is the design that isolates review from capability. When they differ, the reviewer's
    capability becomes the variable and the baseline arm is the implementer's model alone,
    which is the comparison such a run is asking about.

    The judge is always the reviewer. With one model that is the only honest option, and
    with two it keeps the ruling consistent with every other round.

    Args:
        task_dir: The task's directory, holding its spec.md.
        results_dir: Where workspaces, results and logs go.
        results_name: Filename inside results_dir to append finished runs to.
        implementer_model: Model for the implementer and for the baseline arm.
        reviewer_model: Model for the reviewer, which also rules on disputes.

    Returns:
        The argument list to execute.
    """
    return [
        "uv",
        "run",
        "agentpair",
        "--task",
        str(task_dir / SPEC_FILENAME),
        "--module",
        SOLUTION_MODULE,
        "--arm",
        "both",
        "--implementer-model",
        implementer_model,
        "--reviewer-model",
        reviewer_model,
        "--judge",
        "reviewer",
        "--workspace",
        str(results_dir),
        "--results",
        str(results_dir / results_name),
    ]


def run_task(
    task_dir: Path,
    results_dir: Path,
    results_name: str,
    implementer_model: str,
    reviewer_model: str,
) -> TaskRun:
    """Run one task through both arms, tee-ing its console output to a log.

    Args:
        task_dir: The task's directory.
        results_dir: Where workspaces, results and logs go.
        results_name: Filename inside results_dir to append finished runs to.
        implementer_model: Model for the implementer and for the baseline arm.
        reviewer_model: Model for the reviewer, which also rules on disputes.

    Returns:
        What happened, including the exit status and the wall clock.
    """
    log_path = results_dir / f"{task_dir.name}.log"
    started = time.monotonic()
    with log_path.open("w") as log:
        completed = subprocess.run(
            build_command(task_dir, results_dir, results_name, implementer_model, reviewer_model),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started

    tail = log_path.read_text().strip().splitlines()
    for line in tail[-len(ARMS) :]:
        print(f"    {line}")
    return TaskRun(task_id=task_dir.name, skipped=False, returncode=completed.returncode, seconds=elapsed)


def summarise(runs: list[TaskRun]) -> str:
    """Render what the sweep did, naming anything that went wrong.

    Args:
        runs: One entry per task visited.

    Returns:
        A short report. Failures are named individually, because a sweep that reports only
        a count leaves the reader to find which task to re-run.
    """
    ran = [run for run in runs if not run.skipped]
    skipped = [run for run in runs if run.skipped]
    failed = [run for run in ran if run.returncode != 0]

    lines = [
        "",
        f"{len(ran)} task(s) run, {len(skipped)} skipped as already recorded, {len(failed)} failed.",
        f"wall clock: {sum(run.seconds for run in ran) / 60:.1f} min",
    ]
    for run in failed:
        lines.append(f"  FAILED {run.task_id} (exit {run.returncode})")
    return "\n".join(lines)


def main() -> None:
    """Run every requested task through both arms, skipping what is already recorded."""
    args = parse_args()
    tasks_dir = Path(args.tasks_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / args.results_name

    tasks = find_tasks(tasks_dir, args.task)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    implementer_model = args.implementer_model or args.model
    reviewer_model = args.reviewer_model or args.model
    print(f"implementer={implementer_model}  reviewer={reviewer_model}  judge=reviewer")

    done = completed_arms(results_path)
    runs: list[TaskRun] = []

    for position, task_dir in enumerate(tasks, start=1):
        already = {arm for arm in ARMS if (task_dir.name, arm) in done}
        if already == set(ARMS):
            print(f"[{position}/{len(tasks)}] {task_dir.name}: already recorded, skipping")
            runs.append(TaskRun(task_id=task_dir.name, skipped=True, returncode=None, seconds=0.0))
            continue
        if already:
            # Both arms are run together, so a half-recorded task re-runs both and the
            # earlier arm's line stays in the file. Say so rather than silently duplicating.
            print(f"[{position}/{len(tasks)}] {task_dir.name}: only {sorted(already)} recorded, re-running both arms")

        print(f"[{position}/{len(tasks)}] {task_dir.name}: running")
        if args.dry_run:
            command = build_command(task_dir, results_dir, args.results_name, implementer_model, reviewer_model)
            print("    " + " ".join(command))
            continue
        runs.append(run_task(task_dir, results_dir, args.results_name, implementer_model, reviewer_model))

    print(summarise(runs))
    if any(run.returncode not in (0, None) for run in runs):
        sys.exit(1)


if __name__ == "__main__":
    main()
