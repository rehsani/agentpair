"""Command line entry point.

Exposes the two arms over a task file, with models, workspace and output path as flags
rather than constants, so a sweep changes arguments and never the code.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

from agentpair import roles
from agentpair.arbitration import self_arbitration_reason
from agentpair.baseline import run_baseline
from agentpair.metrics import append_run
from agentpair.models import RunFailed, RunRecord
from agentpair.orchestrator import run_pair
from agentpair.sdk_runner import sdk_runner

# File stems that name the kind of file rather than the task in it. A benchmark keeps each
# task's specification in a spec.md inside a directory named for the task, so the stem is
# the same for every task and the directory is where the name is.
GENERIC_TASK_STEMS = frozenset({"spec", "task", "prompt", "readme", "index"})

# The value of --judge that reuses the reviewer's model rather than naming a third one.
JUDGE_REVIEWER = "reviewer"


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run one task through the pair arm, the baseline arm, or both.")
    parser.add_argument("--task", required=True, help="path to the task specification file")
    parser.add_argument("--arm", choices=["pair", "baseline", "both"], default="pair", help="which arm to run")
    parser.add_argument("--module", required=True, help="implementation filename the agent must produce")
    parser.add_argument("--test", default=None, help="test filename (default: test_<module>.py)")
    parser.add_argument(
        "--task-id",
        default=None,
        help="identifier recorded with the run and used to name its workspace (default: the task "
        "file's stem, or its parent directory's name when the stem is a generic one such as spec)",
    )
    parser.add_argument("--implementer-model", default="opus", help="model for the implementer and the baseline")
    parser.add_argument("--reviewer-model", default="opus", help="model for the reviewer")
    parser.add_argument(
        "--judge",
        default=None,
        help="who settles disputes, required for the pair arm: 'reviewer' reuses the reviewer's "
        "model, which adds no third model to the run and is the cheaper option, and any other "
        "value is the name of a model that rules as a third party, for example 'haiku'",
    )
    parser.add_argument("--workspace", default="runs", help="parent directory for each run's working directory")
    parser.add_argument("--results", default="runs/results.jsonl", help="file each finished run is appended to")
    parser.add_argument("--style", default=None, help="path to a file holding a style brief that overrides the default")
    return parser.parse_args()


def resolve_judge(args: argparse.Namespace) -> tuple[str, bool]:
    """Return the model that will rule on disputes, and whether it is a disputant's own.

    There are two honest answers to who judges, and the flag asks for one of them rather
    than choosing on the caller's behalf. "reviewer" reuses the reviewer's model: it costs
    no third model, and it is defensible because the reviewer is shown the implementer's
    rebuttal and any probe output before it rules, which it had not seen when it raised the
    finding. Naming a model instead puts a third party in the chair, which is the stronger
    claim and the more expensive one, at two turns per dispute rather than per run.

    There is deliberately no third answer. The implementer cannot judge: it has already
    written down its position, so asking it to rule is asking it the same question twice.

    The flag used to be optional and to default to the reviewer's model, which the
    self-arbitration guard then refused, so the command chose a value it was guaranteed to
    reject and every pair run without the flag died at startup. The choice is now made by
    the caller or not at all.

    Args:
        args: The parsed command line.

    Returns:
        The model that rules, and whether it is a disputant's own model, which run_pair
        needs in order to accept it.

    Raises:
        SystemExit: If --judge is missing, or names the implementer's model, or names the
            reviewer's model by a spelling other than "reviewer".
    """
    if not args.judge or not args.judge.strip():
        raise SystemExit(
            "--judge is required for the pair arm. Pass --judge reviewer to have the reviewer's "
            f"model ({args.reviewer_model}) settle disputes, which adds no third model to the run, "
            "or --judge <model>, for example --judge haiku, to have a third model rule."
        )

    choice = args.judge.strip()
    if choice.casefold() == JUDGE_REVIEWER:
        return args.reviewer_model, True

    reason = self_arbitration_reason(args.implementer_model, args.reviewer_model, choice)
    if reason is not None:
        raise SystemExit(
            f"refusing to run: {reason}. Pass --judge reviewer to choose that deliberately, "
            "or name a model that is neither side's."
        )
    return choice, False


def resolve_task_id(
    task_id: str | None,
    task_path: Path,
) -> str:
    """Return the identifier this run is recorded and workspaced under.

    The file's stem is the right name for tasks/duration.md and the wrong one for a
    benchmark, where every task's specification is a spec.md inside a directory named for
    the task. Twenty such runs would all be written down as "spec" and would share one
    workspace, which _clean_workspace empties at the start of each run, so the results
    file would carry twenty lines that cannot be told apart. A generic stem therefore
    falls back to the directory holding it, which is where the name actually is, and
    --task-id overrides both for a caller that knows better.

    Args:
        task_id: The identifier given on the command line, or None.
        task_path: Resolved path to the task specification file.

    Returns:
        The identifier to record.
    """
    if task_id and task_id.strip():
        return task_id.strip()
    stem = task_path.stem
    if stem.casefold() in GENERIC_TASK_STEMS and task_path.parent.name:
        return task_path.parent.name
    return stem


def _clean_workspace(workspace: Path) -> Path:
    """Remove and recreate a run's working directory.

    A reused workspace lets an agent start from a finished implementation left by an
    earlier run, which silently contaminates the comparison between arms.

    Args:
        workspace: The directory for this arm's run.

    Returns:
        The same path, now empty.
    """
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    return workspace


def summarise(record: RunRecord) -> str:
    """Render a finished run as one console line.

    Args:
        record: The finished run.

    Returns:
        A single line naming the arm, the verified suite outcome, findings, acceptances,
        disputes, resolutions, tokens and cost. "probed" is the disputes the orchestrator ran code
        on before ruling, "judged" the ones it ruled on from the arguments alone, and "applied"
        "sent_back" the rulings that went against the implementer and were sent back to it to
        carry out, which is not a claim that the files changed.
        Protocol errors, probe failures and turns that ended early are counted too: each was
        added so a degraded run would stop hiding inside the other fields, which it would
        go on doing if only the file said so and the console did not.
    """
    probed = sum(1 for resolution in record.resolutions if resolution.method == "judged_with_probe")
    judged = sum(1 for resolution in record.resolutions if resolution.method == "judged")
    return (
        f"{record.arm:<8} suite={record.suite_status} ({record.tests_passed}p/{record.tests_failed}f) "
        f"findings={len(record.findings)} accepted={len(record.accepted)} "
        f"disputed={len(record.disputes)} "
        f"resolved={len(record.resolutions)} "
        f"(probed={probed} judged={judged}) sent_back={len(record.sent_back_after_ruling)} "
        f"tokens={record.total_tokens} "
        f"uncached={record.uncached_tokens} "
        f"cost=${record.total_cost_usd:.4f} {record.seconds:.1f}s"
        + (f" unanswered={len(record.unanswered)}" if record.unanswered else "")
        + (f" unresolved={len(record.unresolved)}" if record.unresolved else "")
        + (f" protocol_errors={len(record.protocol_errors)}" if record.protocol_errors else "")
        + (f" probe_failures={len(record.probe_failures)}" if record.probe_failures else "")
        + (f" cut_short={len(record.cut_short)}" if record.cut_short else "")
        + (f" FAILED: {record.failure}" if record.failure else "")
    )


async def run_arms(
    args: argparse.Namespace,
    results_path: Path,
) -> list[RunRecord]:
    """Run whichever arms were requested, each in a clean workspace, appending as they finish.

    A record is written the moment its arm returns, so a crash in the second arm cannot
    discard the first arm's paid result. Any exception from an arm is caught and recorded
    on that arm's record rather than raised, for the same reason. run_pair handles a parse
    failure itself and keeps its usage, and both arms raise RunFailed carrying their
    record for everything else, so a crashed run still reports what it spent and the two
    arms being compared account for a crash the same way. The bare catch
    remains for a failure with no record behind it at all.

    Args:
        args: The parsed command line.
        results_path: File each finished record is appended to.

    Returns:
        One record per arm that ran.
    """
    orchestrator_model, judge_is_a_disputant = resolve_judge(args) if args.arm in {"pair", "both"} else (None, False)
    task_path = Path(args.task).expanduser().resolve()
    spec = task_path.read_text()
    task_id = resolve_task_id(args.task_id, task_path)
    test = args.test or f"test_{Path(args.module).stem}.py"
    style = Path(args.style).read_text() if args.style else roles.DEFAULT_STYLE
    workspace_root = Path(args.workspace).expanduser().resolve()

    records: list[RunRecord] = []
    if args.arm in {"pair", "both"}:
        workspace = _clean_workspace(workspace_root / f"{task_id}_pair")
        record = RunRecord(
            task_id=task_id,
            arm="pair",
            implementer_model=args.implementer_model,
            reviewer_model=args.reviewer_model,
            orchestrator_model=orchestrator_model,
        )
        try:
            record = await run_pair(
                task_id=task_id,
                spec=spec,
                module=args.module,
                test=test,
                workspace=workspace,
                implementer_model=args.implementer_model,
                reviewer_model=args.reviewer_model,
                orchestrator_model=orchestrator_model,
                runner=sdk_runner,
                allow_self_arbitration=judge_is_a_disputant,
                style=style,
            )
        except RunFailed as failed:
            # Carries the record of the turns already paid for, so the cost is not lost.
            record = failed.record
        except Exception as error:  # noqa: BLE001 - a crashed arm must not discard a finished one
            record.failure = f"{type(error).__name__}: {error}"
        append_run(record, results_path)
        print(summarise(record))
        records.append(record)

    if args.arm in {"baseline", "both"}:
        workspace = _clean_workspace(workspace_root / f"{task_id}_baseline")
        record = RunRecord(task_id=task_id, arm="baseline", implementer_model=args.implementer_model)
        try:
            record = await run_baseline(
                task_id=task_id,
                spec=spec,
                module=args.module,
                test=test,
                workspace=workspace,
                model=args.implementer_model,
                runner=sdk_runner,
                style=style,
            )
        except RunFailed as failed:
            # As in the pair arm: the record carries the turn's elapsed time and whatever
            # was paid for, so the two arms account for a crash the same way.
            record = failed.record
        except Exception as error:  # noqa: BLE001 - a crashed arm must not discard a finished one
            record.failure = f"{type(error).__name__}: {error}"
        append_run(record, results_path)
        print(summarise(record))
        records.append(record)

    return records


def main() -> None:
    """Run the requested arms. Each record is appended and printed as its arm finishes."""
    args = parse_args()
    asyncio.run(run_arms(args, Path(args.results).expanduser().resolve()))


if __name__ == "__main__":
    main()
