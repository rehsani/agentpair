"""Tests for running an agent-written suite.

These distinguish the four outcomes that must never be confused: the suite passed, the
suite genuinely failed, the file was never written, and the environment could not produce
a verdict. Collapsing the last into "failed" would make a broken environment look like a
finding about the agent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agentpair.verify import run_suite

PASSING = "def test_one():\n    assert 1 + 1 == 2\n"
FAILING = "def test_one():\n    assert 1 + 1 == 3\n"


def test_a_missing_test_file_is_its_own_outcome(tmp_path: Path) -> None:
    result = run_suite(tmp_path, "test_thing.py")
    assert result.status == "missing"
    assert "never written" in result.output


def test_a_missing_pytest_is_an_environment_error_not_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "test_thing.py").write_text(PASSING)

    def fake_run(*args, **kwargs):
        # An interpreter without pytest exits 1, exactly as a real test failure does.
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr=f"{sys.executable}: No module named pytest\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_suite(tmp_path, "test_thing.py")
    assert result.status == "error"
    assert "pytest is not installed" in result.output


def test_a_hanging_suite_times_out(tmp_path: Path) -> None:
    (tmp_path / "test_thing.py").write_text("import time\n\n\ndef test_slow():\n    time.sleep(5)\n")
    result = run_suite(tmp_path, "test_thing.py", timeout=1)
    assert result.status == "error"
    assert "timed out" in result.output


def test_a_relative_workspace_is_resolved_rather_than_read_as_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The path pytest is given must not resolve twice, once as cwd and once as rootdir.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "test_thing.py").write_text(FAILING)
    monkeypatch.chdir(tmp_path)
    result = run_suite(Path("ws"), "test_thing.py")
    assert result.status == "failed"
    assert result.passed == 0
    assert result.failed == 1


@pytest.mark.parametrize("given", ["ws", "./ws", "../{parent}/ws", "ws/../ws"])
def test_every_relative_spelling_of_the_workspace_reaches_the_same_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    given: str,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "test_thing.py").write_text(PASSING)
    monkeypatch.chdir(tmp_path)
    result = run_suite(Path(given.format(parent=tmp_path.name)), "test_thing.py")
    assert result.status == "passed"
    assert result.passed == 1


def test_a_relative_workspace_with_no_test_file_is_still_missing_not_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "ws").mkdir()
    monkeypatch.chdir(tmp_path)
    result = run_suite(Path("ws"), "test_thing.py")
    assert result.status == "missing"


# The ten degenerate suite shapes, with the counts pinned. Each expected pair was read off
# a real run and then checked by hand against what the shape contains, so a change in how
# the counts are obtained has to reproduce all ten.
EMPTY = ""
IMPORT_ERROR = "import a_module_that_does_not_exist\n\n\ndef test_a():\n    assert True\n"
SYNTAX_ERROR = "def test_a(:\n    assert True\n"
TWO_PASSING = "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n"
ONE_EACH = "def test_a():\n    assert True\n\n\ndef test_b():\n    assert False\n"
XFAIL = "import pytest\n\n\n@pytest.mark.xfail\ndef test_a():\n    assert False\n"
SKIPPED = "import pytest\n\n\n@pytest.mark.skip(reason='not yet')\ndef test_a():\n    assert True\n"
UNITTEST_STYLE = (
    "import unittest\n\n\nclass T(unittest.TestCase):\n"
    "    def test_a(self):\n        self.assertEqual(1, 1)\n\n"
    "    def test_b(self):\n        self.assertEqual(1, 2)\n"
)
PRINTS_A_FAKE_SUMMARY = 'def test_a():\n    print("999 passed, 7 failed in 0.01s")\n    assert True\n'
NAMES_A_FAKE_SUMMARY = (
    'import pytest\n\n\n@pytest.mark.parametrize("case", ["999 passed"])\ndef test_a(case):\n    assert False\n'
)


@pytest.mark.parametrize(
    ("shape", "body", "status", "passed", "failed"),
    [
        ("passing", TWO_PASSING, "passed", 2, 0),
        ("mixed", ONE_EACH, "failed", 1, 1),
        ("empty file", EMPTY, "error", 0, 0),
        ("import error", IMPORT_ERROR, "error", 0, 1),
        ("syntax error", SYNTAX_ERROR, "error", 0, 1),
        ("missing file", None, "missing", 0, 0),
        ("xfail", XFAIL, "passed", 0, 0),
        ("skip", SKIPPED, "passed", 0, 0),
        ("unittest style", UNITTEST_STYLE, "failed", 1, 1),
        ("prints a fake summary", PRINTS_A_FAKE_SUMMARY, "passed", 1, 0),
    ],
)
def test_the_counts_come_from_the_report_and_not_from_the_summary_line(
    tmp_path: Path,
    shape: str,
    body: str | None,
    status: str,
    passed: int,
    failed: int,
) -> None:
    if body is not None:
        (tmp_path / "test_thing.py").write_text(body)
    result = run_suite(tmp_path, "test_thing.py")
    assert (result.status, result.passed, result.failed) == (status, passed, failed), shape


def test_a_test_id_that_reads_like_a_summary_line_is_not_read_as_one(tmp_path: Path) -> None:
    # Measured against the previous implementation: scraping "(\\d+) passed" out of pytest's
    # terminal output read this suite as 999 passed and 1 failed, because pytest prints the
    # failing test's id, and the id is written by the suite. Nothing the suite can write
    # reaches an attribute of the XML report.
    (tmp_path / "test_thing.py").write_text(NAMES_A_FAKE_SUMMARY)
    result = run_suite(tmp_path, "test_thing.py")
    assert result.passed == 0
    assert result.failed == 1
    assert "999 passed" in result.output


def test_a_skipped_test_is_neither_passed_nor_failed(tmp_path: Path) -> None:
    # pytest reports an xfail as a skip, so counting skips as passes would credit the agent
    # for a test it wrote down as not expected to work.
    (tmp_path / "test_thing.py").write_text(SKIPPED + XFAIL.replace("test_a", "test_b"))
    result = run_suite(tmp_path, "test_thing.py")
    assert (result.passed, result.failed) == (0, 0)


def test_a_suite_pytest_leaves_no_report_for_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If the report is not there the run cannot be scored, and reporting zero passed beside
    # a status of "passed" would read as a suite with no tests that succeeded.
    (tmp_path / "test_thing.py").write_text(TWO_PASSING)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="2 passed in 0.01s\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_suite(tmp_path, "test_thing.py")
    assert result.status == "error"
    assert result.passed == 0
    assert "no usable JUnit report" in result.output


def test_the_report_is_not_left_in_the_workspace(tmp_path: Path) -> None:
    # The workspace is the artifact the run is measured on and is read afterwards, so the
    # harness's own report must not end up among the agent's files.
    (tmp_path / "test_thing.py").write_text(TWO_PASSING)
    run_suite(tmp_path, "test_thing.py")
    # __pycache__ is the interpreter's, and pytest's own cache is already off.
    left = sorted(path.name for path in tmp_path.iterdir() if path.name != "__pycache__")
    assert left == ["test_thing.py"]
