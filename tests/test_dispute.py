"""Tests for running a probe and recording how a dispute was settled."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentpair.dispute import (
    MAX_OBSERVATION_CHARS,
    ProbeResult,
    execute_probe,
    judged_resolution,
    probed_resolution,
)
from agentpair.models import Dispute, Finding


def make_dispute() -> Dispute:
    return Dispute(
        finding=Finding(id="f1", claim="parse_duration('1h30m') is wrong", evidence="duration.py:12 drops seconds"),
        implementer_position="it returns 5415, which the spec asks for",
    )


def test_execute_probe_captures_stdout_and_retains_the_file(tmp_path: Path) -> None:
    probe_path, result = execute_probe("print(2 + 3)\n", tmp_path, "f1", 1)
    assert probe_path == tmp_path / "probe_1_f1.py"
    assert probe_path.exists()
    assert result.returncode == 0
    assert result.observation == "5"


def test_execute_probe_reports_a_crash_as_an_observation(tmp_path: Path) -> None:
    _, result = execute_probe("raise ValueError('boom')\n", tmp_path, "f2", 1)
    assert result.returncode == 1
    assert "ValueError: boom" in result.observation


def test_execute_probe_times_out(tmp_path: Path) -> None:
    _, result = execute_probe("import time\ntime.sleep(5)\n", tmp_path, "f3", 1, timeout=1)
    assert result.returncode is None
    assert "timed out after 1s" in result.observation


def test_execute_probe_refuses_source_that_does_not_compile(tmp_path: Path) -> None:
    with pytest.raises(SyntaxError):
        execute_probe("def broken(\n", tmp_path, "f4", 1)
    assert not (tmp_path / "probe_1_f4.py").exists()


def test_execute_probe_allows_a_main_guard(tmp_path: Path) -> None:
    source = "def main():\n    print('ran')\n\nif __name__ == '__main__':\n    main()\n"
    _, result = execute_probe(source, tmp_path, "f5", 1)
    assert result.observation == "ran"


def test_execute_probe_cannot_change_the_measured_workspace(tmp_path: Path) -> None:
    (tmp_path / "duration.py").write_text("VALUE = 1\n")
    source = "open('duration.py', 'w').write('VALUE = 2\\n')\nprint('overwritten')\n"
    _, result = execute_probe(source, tmp_path, "f6", 1)
    assert result.observation == "overwritten"
    assert (tmp_path / "duration.py").read_text() == "VALUE = 1\n"


def test_execute_probe_sanitises_a_traversing_finding_id(tmp_path: Path) -> None:
    probe_path, _ = execute_probe("print(1)\n", tmp_path, "../../escaped", 1)
    assert probe_path.parent == tmp_path
    assert ".." not in probe_path.name


def test_observation_falls_back_when_nothing_was_printed() -> None:
    assert ProbeResult(returncode=0, stdout="", stderr="").observation == "(no output)"


def test_probed_resolution_carries_the_observation_and_the_ruling(tmp_path: Path) -> None:
    probe_path = tmp_path / "probe_f1.py"
    result = ProbeResult(returncode=0, stdout="5415\n", stderr="")
    resolution = probed_resolution(make_dispute(), "implementer", probe_path, result, "the output is 5415")
    assert resolution.method == "judged_with_probe"
    assert resolution.favours == "implementer"
    assert "observation: 5415" in resolution.rationale
    assert "ruling: the output is 5415" in resolution.rationale
    assert resolution.probe_path == str(probe_path)


def test_judged_resolution_has_no_probe() -> None:
    resolution = judged_resolution(make_dispute(), "reviewer", "naming preference only")
    assert resolution.method == "judged"
    assert resolution.probe_path is None


def test_probes_for_distinct_findings_do_not_share_a_file(tmp_path: Path) -> None:
    first, _ = execute_probe("print('first')\n", tmp_path, "f/1", 1)
    second, _ = execute_probe("print('second')\n", tmp_path, "f_1", 2)
    assert first != second
    assert first.read_text() == "print('first')\n"
    assert second.read_text() == "print('second')\n"


def test_a_short_observation_is_passed_through_untouched(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("x = 1\n")
    _, result = execute_probe("print('5415')", tmp_path, "f1", 1)
    assert result.observation == "5415"


def test_a_chatty_probe_is_bounded_before_it_reaches_the_ruling_turn(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("x = 1\n")
    _, result = execute_probe("print('a' * 50_000)", tmp_path, "f1", 1)
    assert len(result.stdout) > 50_000
    # The ruling turn is shown the ceiling plus the one marker line, not the whole dump.
    assert len(result.observation) < MAX_OBSERVATION_CHARS + 100
    assert "characters omitted" in result.observation


def test_truncation_keeps_the_answer_at_the_start_and_the_traceback_at_the_end(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("x = 1\n")
    source = "print('ANSWER IS 5415')\nprint('filler ' * 20_000)\nraise ValueError('the real crash')"
    _, result = execute_probe(source, tmp_path, "f1", 1)
    observation = result.observation
    assert "ANSWER IS 5415" in observation
    assert "the real crash" in observation
    assert "characters omitted" in observation
