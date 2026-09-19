"""Running the agent's own suite so a run has a measured outcome.

Neither arm proves anything by reporting that it finished. Without this the record says
only that a model claimed to be done, and the dependent variable has to be measured by
hand afterwards. The harness runs the suite itself and records what happened.

This measures the suite the agent wrote, which is not the same as correctness. An agent
that writes weak tests passes here. Held-out tests are what measure correctness, and they
belong to the benchmark, not to this module.

The counts come from pytest's JUnit XML report, not from its terminal summary. The summary
is written for a person: it is one line of prose whose wording moves between versions, and
anything the tests themselves print can imitate it, so a suite that prints "999 passed"
was read as 999 passing tests. The XML carries tests, failures, errors and skipped as
attributes of an element, which nothing the suite prints can reach. The status still comes
from pytest's exit code, which is already a machine-readable value.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree

DEFAULT_SUITE_TIMEOUT_SECONDS = 120

# An absent pytest is an interpreter error on stderr, not a report, so there is nothing
# structured to read instead. Detecting it by message is the only option.
_NO_PYTEST = re.compile(r"No module named pytest", re.IGNORECASE)

# The two pytest exit codes that are an outcome about the tests. Every other code, 5 for
# "collected nothing" included, means pytest gave no verdict on them.
_PYTEST_PASSED = 0
_PYTEST_FAILED = 1

SuiteStatus = Literal["passed", "failed", "missing", "error"]


@dataclass(frozen=True)
class SuiteResult:
    """The outcome of running an agent-written test suite.

    Attributes:
        status: "passed" when pytest reported success, "failed" when tests failed,
            "missing" when the test file was never written, and "error" when pytest gave no
            usable verdict: a collection or import failure, a file containing no tests, or
            pytest itself being absent from the run interpreter.
        passed: Number of tests that passed.
        failed: Number that failed or errored.
        output: The tail of pytest's output, kept so a failure is diagnosable later.
    """

    status: SuiteStatus
    passed: int
    failed: int
    output: str


def _read_counts(report_path: Path) -> tuple[int, int] | None:
    """Read the pass and fail counts out of pytest's JUnit XML report.

    Args:
        report_path: Path pytest was told to write its XML report to.

    Returns:
        How many tests passed and how many failed or errored, or None when no usable
        report was written, which is itself an outcome the caller has to record.
    """
    try:
        root = ElementTree.parse(report_path).getroot()
    except (OSError, ElementTree.ParseError):
        return None
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        return None
    tests = failures = errors = skipped = 0
    for suite in suites:
        tests += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
    # A skipped test did not pass and did not fail, and pytest reports an xfail as skipped,
    # so counting either as a pass would credit the agent for a test that never ran.
    failed = failures + errors
    return max(tests - failed - skipped, 0), failed


def _status_from_returncode(returncode: int) -> SuiteStatus:
    """Map pytest's exit code to the outcome recorded for the suite.

    Args:
        returncode: The exit status pytest reported.

    Returns:
        "passed" on success, "failed" when tests failed, and "error" for everything else,
        including exit code 5, which means pytest collected no tests at all. A file with
        no tests in it passes vacuously, which must never read as success.
    """
    if returncode == _PYTEST_PASSED:
        return "passed"
    if returncode == _PYTEST_FAILED:
        return "failed"
    return "error"


def run_suite(
    workspace: Path,
    test: str,
    timeout: int = DEFAULT_SUITE_TIMEOUT_SECONDS,
) -> SuiteResult:
    """Run one agent-written test file and report what happened.

    The workspace is resolved here rather than trusted as given. pytest runs with the
    workspace as its working directory and is also told the rootdir, so a relative path
    resolves twice and points at ws/ws, which pytest rejects. That surfaces as status
    "error" for every task rather than as a loud failure, which is the exact confusion
    this module exists to prevent, so the path is made absolute at the one point every
    caller passes through.

    The XML report is written outside the workspace. The workspace is the artifact the run
    is measured on and is read afterwards, so dropping a report into it would leave the
    harness's own file among the agent's.

    Args:
        workspace: The run's working directory, relative or absolute.
        test: The test filename the agent was asked to produce.
        timeout: Seconds before the suite is killed.

    Returns:
        The suite result. A missing test file is reported rather than raised, because an
        agent failing to write it is itself an outcome worth recording.
    """
    workspace = workspace.resolve()
    test_path = workspace / test
    if not test_path.exists():
        return SuiteResult(status="missing", passed=0, failed=0, output=f"{test} was never written")

    with tempfile.TemporaryDirectory() as reports:
        report_path = Path(reports) / "report.xml"
        try:
            completed = subprocess.run(
                # -c with no config file stops pytest inheriting this project's own
                # pyproject settings when a run workspace happens to sit inside the repo.
                # It also makes the config file's directory the rootdir, so without
                # --rootdir every failure prints as ../../../../../../../dev::test_name.
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--tb=no",
                    "-p",
                    "no:cacheprovider",
                    "-c",
                    os.devnull,
                    "--rootdir",
                    str(workspace),
                    f"--junit-xml={report_path}",
                    test,
                ],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SuiteResult(status="error", passed=0, failed=0, output=f"suite timed out after {timeout}s")

        output = (completed.stdout + completed.stderr).strip()
        counts = _read_counts(report_path)

    # An absent pytest exits 1, the same code as a genuine test failure. Reporting that as
    # "failed" would make a broken environment look like a finding about the agent, so it
    # is detected by message and reported as an environment error instead.
    if _NO_PYTEST.search(output):
        return SuiteResult(status="error", passed=0, failed=0, output="pytest is not installed in the run interpreter")

    if counts is None:
        # pytest ran but left no report to read, so there is no count to record and the
        # run cannot be scored. Saying so beats reporting zeroes beside a status of passed.
        return SuiteResult(
            status="error",
            passed=0,
            failed=0,
            output=f"pytest wrote no usable JUnit report (exit {completed.returncode})\n{output}"[-2000:],
        )

    passed, failed = counts
    return SuiteResult(
        status=_status_from_returncode(completed.returncode), passed=passed, failed=failed, output=output[-2000:]
    )
