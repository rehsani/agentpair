"""Score every produced solution against the benchmark's held-out tests.

Each task's reference solution is scored first. A held-out test its own reference cannot
pass is broken, and every agent measured against it would be marked wrong for the
benchmark's mistake, so an unsound task is reported and its arm scores are not trusted.

Scores are written as JSON lines beside the run results, one line per (task, arm), so the
scoring can be re-run without re-running any agent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from agentpair.scoring import SOLUTION_MODULE, Score, score_reference, score_solution

ARMS = ("pair", "baseline")


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Score every produced solution against the held-out tests.")
    parser.add_argument("--results-dir", required=True, help="directory holding the <task>_<arm> run workspaces")
    parser.add_argument("--answers-dir", required=True, help="unpacked answers, one subdirectory per task")
    parser.add_argument("--scores-name", default="scores.jsonl", help="filename inside --results-dir to write")
    parser.add_argument("--timeout", type=int, default=120, help="seconds before a held-out suite is killed")
    return parser.parse_args()


def score_task(
    task_id: str,
    results_dir: Path,
    answers_dir: Path,
    timeout: int,
) -> list[Score]:
    """Score one task's reference and both arms.

    Args:
        task_id: The task directory name.
        results_dir: Directory holding the <task>_<arm> workspaces.
        answers_dir: Directory holding this task's hidden_test.py and reference.py.
        timeout: Seconds before a held-out suite is killed.

    Returns:
        The reference score followed by one score per arm.
    """
    scores = [score_reference(answers_dir, task_id, timeout=timeout)]
    for arm in ARMS:
        scores.append(
            score_solution(
                solution_path=results_dir / f"{task_id}_{arm}" / SOLUTION_MODULE,
                answers_dir=answers_dir,
                task_id=task_id,
                arm=arm,
                timeout=timeout,
            )
        )
    return scores


def main() -> None:
    """Score every task found in the answers directory and report the totals."""
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    answers_dir = Path(args.answers_dir).expanduser().resolve()
    scores_path = results_dir / args.scores_name

    tasks = sorted(path.name for path in answers_dir.iterdir() if path.is_dir())
    all_scores: list[Score] = []
    unsound: list[str] = []

    for task_id in tasks:
        scores = score_task(task_id, results_dir, answers_dir / task_id, args.timeout)
        reference = scores[0]
        if not reference.solved:
            unsound.append(task_id)
        all_scores.extend(scores)
        flags = "  ".join(f"{score.arm}={score.passed}/{score.passed + score.failed}" for score in scores)
        print(f"{task_id:20} {flags}{'   [REFERENCE UNSOUND]' if not reference.solved else ''}")

    scores_path.write_text("\n".join(json.dumps(asdict(score)) for score in all_scores) + "\n")

    print()
    sound = [task for task in tasks if task not in unsound]
    for arm in ARMS:
        solved = [s for s in all_scores if s.arm == arm and s.task_id in sound and s.solved]
        print(f"{arm:9} solved {len(solved)}/{len(sound)} sound tasks")
    if unsound:
        print(f"\n{len(unsound)} task(s) whose own reference fails, excluded: {', '.join(unsound)}")
    print(f"\nscores written to {scores_path}")


if __name__ == "__main__":
    main()
