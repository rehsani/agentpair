"""Running a probe for the orchestrator, and recording how a dispute was settled.

When the implementer rejects a finding, the orchestrator decides who is right. It may run
code first if running something would help, and the probe it writes is executed here and
retained beside the run so the decision stays auditable.

The orchestrator both writes the probe and rules on it, so there is no adversary to guard
against inside the probe. Nothing here restricts what a probe may contain.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentpair.models import Dispute, Resolution, Side

DEFAULT_TIMEOUT_SECONDS = 30

# Characters of probe output the orchestrator is shown before ruling. The same ceiling the
# suite's output already uses. A probe that loops or dumps a structure would otherwise put
# its whole output into the ruling turn's context, and the format asks for a short answer
# precisely so this never binds. Head and tail are both kept, because a probe's answer is
# usually what it printed first and its traceback is always last, and keeping only the tail
# would drop the answer.
MAX_OBSERVATION_CHARS = 2000


@dataclass(frozen=True)
class ProbeResult:
    """What running one probe produced.

    Attributes:
        returncode: Process exit status, or None if it was killed by the timeout.
        stdout: Everything the probe printed.
        stderr: Traceback or interpreter noise, kept because a crash is itself an
            observation about the code's behaviour.
    """

    returncode: int | None
    stdout: str
    stderr: str

    @property
    def observation(self) -> str:
        """Return the combined output the orchestrator reads before ruling, bounded.

        Returns:
            Everything the probe printed, truncated in the middle when it exceeds
            MAX_OBSERVATION_CHARS, with a line naming how much was dropped so the
            orchestrator rules knowing it was not shown all of it.
        """
        parts = [self.stdout.strip(), self.stderr.strip()]
        joined = "\n".join(part for part in parts if part) or "(no output)"
        return _truncate_middle(joined, MAX_OBSERVATION_CHARS)


def _truncate_middle(
    text: str,
    limit: int,
) -> str:
    """Shorten text to a ceiling, keeping its start and its end.

    Args:
        text: The text to bound.
        limit: Maximum characters to keep, excluding the marker line.

    Returns:
        The text unchanged when it is already short enough, otherwise its first and last
        halves with a line between them naming how many characters were dropped.
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    dropped = len(text) - 2 * half
    return f"{text[:half]}\n... {dropped} characters omitted ...\n{text[-half:]}"


def _safe_name(finding_id: str) -> str:
    """Reduce a model-supplied finding id to something safe to use as a filename.

    The id comes from the reviewer's JSON, so it can contain path separators or dots that
    would write the probe outside the workspace.

    Args:
        finding_id: The reviewer's identifier for the finding.

    Returns:
        The id with every character outside letters, digits, dash and underscore replaced,
        or "unnamed" if nothing usable remains.
    """
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in finding_id)
    return cleaned.strip("_") or "unnamed"


def execute_probe(
    source: str,
    workspace: Path,
    finding_id: str,
    index: int,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[Path, ProbeResult]:
    """Write a probe beside the run, execute it against a copy of the workspace, and report.

    The probe runs in a throwaway copy, because a probe that writes a file or overwrites the
    module would otherwise change the code the run is measured on afterwards. The probe
    source itself is kept in the real workspace so the decision stays auditable.

    Args:
        source: The probe's Python source.
        workspace: Directory whose contents the probe runs against.
        finding_id: Identifier of the finding under dispute, used for the filename.
        index: Position of this dispute in the run. Sanitising the id collapses distinct
            ids onto one name, so without it "f/1" and "f_1" write the same file and one
            resolution's retained probe is another dispute's source.
        timeout: Seconds before the probe is killed.

    Returns:
        The retained probe path and its result.

    Raises:
        SyntaxError: If the probe source does not compile.
    """
    compile(source, "<probe>", "exec")
    probe_path = workspace / f"probe_{index}_{_safe_name(finding_id)}.py"
    probe_path.write_text(source)

    with tempfile.TemporaryDirectory() as sandbox:
        sandbox_path = Path(sandbox) / "workspace"
        shutil.copytree(workspace, sandbox_path)
        try:
            completed = subprocess.run(
                [sys.executable, probe_path.name],
                cwd=sandbox_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return probe_path, ProbeResult(returncode=None, stdout="", stderr=f"probe timed out after {timeout}s")
    return probe_path, ProbeResult(completed.returncode, completed.stdout, completed.stderr)


def probed_resolution(
    dispute: Dispute,
    favours: Side,
    probe_path: Path,
    result: ProbeResult,
    rationale: str,
) -> Resolution:
    """Record a dispute the orchestrator settled after running code.

    Args:
        dispute: The disagreement being settled.
        favours: The side the orchestrator ruled for.
        probe_path: The retained probe.
        result: What the probe saw.
        rationale: The orchestrator's reasoning.

    Returns:
        A resolution carrying both the observation and the ruling.
    """
    return Resolution(
        dispute=dispute,
        favours=favours,
        method="judged_with_probe",
        rationale=f"observation: {result.observation}\nruling: {rationale}",
        probe_path=str(probe_path),
    )


def judged_resolution(
    dispute: Dispute,
    favours: Side,
    rationale: str,
) -> Resolution:
    """Record a dispute the orchestrator settled without running anything.

    Args:
        dispute: The disagreement being settled.
        favours: The side the orchestrator ruled for.
        rationale: Its reasoning.

    Returns:
        A resolution marked as judged, carrying no probe.
    """
    return Resolution(dispute=dispute, favours=favours, method="judged", rationale=rationale)
