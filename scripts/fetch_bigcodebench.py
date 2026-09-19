"""Fetch BigCodeBench-hard tasks and lay them out as agentpair benchmark cases.

Each case becomes a directory holding three files:

    spec.md          the instruction given to the agent, and the only file it ever sees
    hidden_test.py   the benchmark's own tests, used to score the agent's final code
    reference.py     the canonical solution, used only to prove the hidden test is sound

A task is kept only when its hidden test passes against its own reference solution in this
environment. A test that cannot pass on correct code would score every agent as wrong and
would look like a finding about agents rather than about the harness.

Usage:
    python scripts/fetch_bigcodebench.py --out benchmark/bigcodebench --limit 20
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from agentpair.verify import run_suite

ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=bigcode%2Fbigcodebench-hard&config=default&split={split}&offset={offset}&length={length}"
)
PAGE_SIZE = 100
TOTAL_TASKS = 148
FETCH_ATTEMPTS = 4
FETCH_BACKOFF_SECONDS = 3

# The module filename the agent is asked to produce, and which the hidden test imports.
SOLUTION_MODULE = "solution"

# Libraries whose tasks run offline with nothing installed. Third-party and network
# dependent tasks are excluded: a run that fails because pandas is absent or a host is
# unreachable measures the environment, not the agent.
STDLIB_SAFE = {
    "ast", "base64", "bisect", "collections", "contextlib", "copy", "csv", "datetime",
    "decimal", "difflib", "enum", "fractions", "functools", "glob", "gzip", "hashlib",
    "heapq", "hmac", "io", "itertools", "json", "logging", "math", "operator", "os",
    "pathlib", "pickle", "pprint", "random", "re", "secrets", "shutil", "sqlite3",
    "statistics", "string", "struct", "subprocess", "sys", "tarfile", "tempfile",
    "textwrap", "threading", "time", "typing", "unicodedata", "uuid", "warnings",
    "zipfile",
}


@dataclass(frozen=True)
class Task:
    """One benchmark case as this harness needs it.

    Attributes:
        task_id: The benchmark's own identifier, for example "BigCodeBench/15".
        name: Filesystem-safe form of the identifier, used as the directory name.
        spec: The instruction handed to the agent.
        hidden_test: The benchmark's tests, rewritten to import the agent's module.
        reference: The canonical solution, kept only to validate the hidden test.
        libs: Libraries the task touches.
    """

    task_id: str
    name: str
    spec: str
    hidden_test: str
    reference: str
    libs: list[str]


def fetch_rows(split: str) -> list[dict]:
    """Download every row of the hard split.

    Args:
        split: Dataset version to pull, for example "v0.1.4".

    Returns:
        The raw rows as the datasets server returns them.
    """
    rows: list[dict] = []
    for offset in range(0, TOTAL_TASKS, PAGE_SIZE):
        url = ROWS_URL.format(split=split, offset=offset, length=PAGE_SIZE)
        # The datasets server returns a transient 502 often enough to be worth retrying,
        # and a half-fetched task list would silently shrink the benchmark.
        for attempt in range(FETCH_ATTEMPTS):
            try:
                with urllib.request.urlopen(url, timeout=120) as handle:
                    rows.extend(item["row"] for item in json.load(handle)["rows"])
                break
            except urllib.error.URLError as error:
                if attempt == FETCH_ATTEMPTS - 1:
                    raise
                print(f"  retrying offset {offset} after {error}")
                time.sleep(FETCH_BACKOFF_SECONDS * (attempt + 1))
    return rows


def to_task(row: dict) -> Task:
    """Convert one dataset row into a benchmark case.

    The benchmark's tests assume the function sits in their own namespace, so they are
    prefixed with an import of the module the agent is asked to write.

    Args:
        row: One row from the dataset.

    Returns:
        The task, with its files' contents built but not yet written.
    """
    libs = ast.literal_eval(row["libs"]) if isinstance(row["libs"], str) else list(row["libs"])
    name = row["task_id"].replace("/", "_")
    return Task(
        task_id=row["task_id"],
        name=name,
        spec=row["instruct_prompt"],
        hidden_test=f"from {SOLUTION_MODULE} import *  # noqa: F403\n\n{row['test']}\n",
        reference=f"{row['code_prompt']}\n{row['canonical_solution']}\n",
        libs=libs,
    )


def is_self_contained(task: Task) -> bool:
    """Return whether the task runs offline with no third-party package installed.

    Args:
        task: The task to judge.

    Returns:
        True when every library it touches is in the standard library and none of them
        reach the network.
    """
    return bool(task.libs) and set(task.libs) <= STDLIB_SAFE


def hidden_test_is_sound(
    task: Task,
    timeout: int,
) -> tuple[bool, str]:
    """Run the hidden test against the task's own reference solution.

    This is the only filter that matters. A hidden test that fails on the canonical
    solution cannot score an agent, and keeping it would produce a benchmark where every
    run looks like a failure.

    The run goes through agentpair.verify.run_suite, which is what will score the agent's
    code later. Calling pytest here with flags of its own certified a task under one
    configuration and scored it under another, and two copies of the invocation drift the
    moment one of them is changed.

    Args:
        task: The task to validate.
        timeout: Seconds before the test run is killed.

    Returns:
        Whether it passed, and the tail of pytest's output when it did not.
    """
    with tempfile.TemporaryDirectory() as workspace:
        directory = Path(workspace)
        (directory / f"{SOLUTION_MODULE}.py").write_text(task.reference)
        (directory / "hidden_test.py").write_text(task.hidden_test)
        result = run_suite(directory, "hidden_test.py", timeout=timeout)
    return result.status == "passed", result.output[-400:]


def write_task(
    task: Task,
    out_dir: Path,
) -> Path:
    """Write one task's three files into its own directory.

    Args:
        task: The task to write.
        out_dir: Parent directory for all cases.

    Returns:
        The directory the task was written to.
    """
    directory = out_dir / task.name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "spec.md").write_text(task.spec)
    (directory / "hidden_test.py").write_text(task.hidden_test)
    (directory / "reference.py").write_text(task.reference)
    return directory


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Fetch and validate BigCodeBench-hard tasks for agentpair.")
    parser.add_argument("--out", default="benchmark/bigcodebench", help="directory the task cases are written to")
    parser.add_argument("--limit", type=int, default=20, help="how many validated tasks to keep")
    parser.add_argument("--split", default="v0.1.4", help="dataset version to pull")
    parser.add_argument("--timeout", type=int, default=60, help="seconds allowed per validation run")
    parser.add_argument("--keep-all-libs", action="store_true", help="skip the standard-library-only filter")
    return parser.parse_args()


def main() -> None:
    """Fetch, filter, validate and write the benchmark cases, then report what was kept."""
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)

    rows = fetch_rows(args.split)
    print(f"fetched {len(rows)} tasks from {args.split}")

    tasks = [to_task(row) for row in rows]
    if not args.keep_all_libs:
        tasks = [task for task in tasks if is_self_contained(task)]
        print(f"{len(tasks)} are standard-library only and offline")

    kept: list[Task] = []
    for task in tasks:
        if len(kept) >= args.limit:
            break
        sound, output = hidden_test_is_sound(task, args.timeout)
        if sound:
            kept.append(task)
            print(f"  keep    {task.task_id:22} {task.libs}")
        else:
            print(f"  DROP    {task.task_id:22} hidden test fails on its own reference: {output.splitlines()[-1:]}")

    manifest = []
    for task in kept:
        directory = write_task(task, out_dir)
        # Relative to the manifest, not absolute: an absolute path names the machine the
        # benchmark was fetched on and is wrong everywhere else.
        manifest.append(
            {
                "task_id": task.task_id,
                "name": task.name,
                "libs": task.libs,
                "path": directory.name,
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nwrote {len(kept)} validated tasks to {out_dir}")
    if len(kept) < args.limit:
        print(f"WARNING: asked for {args.limit} but only {len(kept)} passed validation")


if __name__ == "__main__":
    main()
