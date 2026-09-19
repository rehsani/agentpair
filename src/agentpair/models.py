"""Shapes passed between the roles.

Every later module reads and writes these, so they are kept free of logic, SDK imports
and task-specific fields. A reviewer emits Findings. A Finding the implementer applies
becomes an Acceptance, and one it refuses becomes a Dispute. Every Dispute the
orchestrator settles ends in a Resolution, which records whether the orchestrator ran code
before deciding.

AgentTurn and the Runner protocol live here too. They are the shape of one model turn and
the callable that produces one, which every arm and the SDK backend all speak; they sat in
the run loop only because that is where they were first needed, which made the loop an
import target for modules that wanted nothing else from it.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

# The two experimental arms: the implementer and reviewer pair, or one agent alone.
Arm = Literal["pair", "baseline"]

# Which side a resolution favoured.
Side = Literal["implementer", "reviewer"]

# How the orchestrator reached its decision.
# "judged_with_probe": it ran code first and decided with the output in front of it.
# "judged": it decided without running anything, because nothing would have settled it.
Method = Literal["judged_with_probe", "judged"]


@dataclass(frozen=True)
class Finding:
    """One point raised by the reviewer.

    Attributes:
        id: Short stable identifier, used to tie a dispute and resolution back here.
        claim: What the reviewer asserts is wrong, in its own words.
        evidence: Why it believes that, citing the file and line, the input that breaks, or
            the case the tests never reach. This is what the implementer answers and what
            the orchestrator weighs, so a claim with nothing behind it is weak by
            construction.
    """

    id: str
    claim: str
    evidence: str = ""

    def __post_init__(self) -> None:
        """Reject an empty identifier or claim, which would make the record unusable."""
        if not self.id.strip():
            raise ValueError("Finding.id must not be empty")
        if not self.claim.strip():
            raise ValueError("Finding.claim must not be empty")


@dataclass(frozen=True)
class Acceptance:
    """A finding the implementer applied, with the reason it gave.

    Acceptance is the outcome of almost every finding, and until now it was the only
    outcome with no record: the results file carried the findings, the disputes and the
    unanswered ids, and acceptance had to be reconstructed as findings minus disputes
    minus unanswered. That subtraction cannot distinguish a finding that was applied from
    one the record lost track of, and it threw away the reason, which is the implementer's
    only statement about what it changed and why.

    Attributes:
        finding: The reviewer's original point.
        reason: What the implementer said when it accepted. Empty when it gave none, which
            the format allows, since a bare acceptance is still an acceptance.
    """

    finding: Finding
    reason: str = ""


@dataclass(frozen=True)
class Dispute:
    """A finding the implementer declined to apply, with its counter-argument.

    Attributes:
        finding: The reviewer's original point.
        implementer_position: Why the implementer believes the finding is wrong.
    """

    finding: Finding
    implementer_position: str

    def __post_init__(self) -> None:
        """Reject an empty counter-argument, since a dispute with no case is not one."""
        if not self.implementer_position.strip():
            raise ValueError("Dispute.implementer_position must not be empty")


@dataclass(frozen=True)
class Resolution:
    """How the orchestrator settled one dispute.

    Attributes:
        dispute: The disagreement being settled.
        favours: Which side the orchestrator ruled for.
        method: Whether it ran a probe before deciding.
        rationale: Its reasoning, taken from the "reason" field of the ruling it wrote, so
            the record carries the argument and not the turn's whole response. The probe's
            output is included in front of it when there was one.
        probe_path: Path to the retained probe, set only when a probe was run. Keeping it
            is what makes the decision auditable afterwards.
    """

    dispute: Dispute
    favours: Side
    method: Method
    rationale: str
    probe_path: str | None = None

    def __post_init__(self) -> None:
        """Enforce that a probed decision retained its probe and a bare one has none."""
        if self.method == "judged_with_probe" and not self.probe_path:
            raise ValueError("a judged_with_probe resolution must retain its probe")
        if self.method == "judged" and self.probe_path:
            raise ValueError("a judged resolution has no probe")


@dataclass(frozen=True)
class RoleUsage:
    """What one role consumed during a run.

    Attributes:
        role: The role this accounts for.
        input_tokens: Uncached input tokens, which is what the model was actually sent
            fresh. Kept apart from cache reads so a token ratio between arms can be read
            against a cost ratio, which a single merged number makes impossible.
        output_tokens: Tokens the model produced.
        cost_usd: Cost as reported by the SDK, not re-derived from token counts.
        seconds: Wall clock spent in this role's turns.
        cache_read_tokens: Input served from cache, cheap but still context moved.
        cache_creation_tokens: Input written to cache, billed above the uncached rate.
    """

    role: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    seconds: float
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def __post_init__(self) -> None:
        """Reject negative counts, which indicate a broken accounting path upstream."""
        if min(self.input_tokens, self.cache_read_tokens, self.cache_creation_tokens, self.output_tokens) < 0:
            raise ValueError("token counts must not be negative")
        if min(self.cost_usd, self.seconds) < 0:
            raise ValueError("cost and duration must not be negative")


@dataclass(frozen=True)
class AgentTurn:
    """One model turn: what it said and what it cost.

    Attributes:
        text: The assistant text produced.
        usage: The turn's accounting, already labelled with the role.
        cut_short: Why the turn ended early, when it did. A turn stopped by a turn or
            budget cap returns partial text, and without this the cap firing looks
            identical to the model ignoring the requested format.
    """

    text: str
    usage: RoleUsage
    cut_short: str | None = None


class Runner(Protocol):
    """Callable that performs one model turn.

    Injected so the loop can be tested without the SDK, and so a future arm can swap in
    a different backend without touching the protocol.
    """

    def __call__(
        self,
        role: str,
        system: str,
        prompt: str,
        model: str,
        workspace: Path,
        tools: list[str],
    ) -> Awaitable[AgentTurn]: ...


@dataclass
class RunRecord:
    """Everything one task run produced, for one arm.

    Attributes:
        task_id: Identifier of the task that was run.
        arm: Which arm produced this record.
        implementer_model: Model used for the implementer, or for the lone baseline agent.
        reviewer_model: Model used for the reviewer, None in the baseline arm.
        orchestrator_model: Model that ruled on the disputes, None in the baseline arm. A
            reader cannot weigh "the review found N real problems" without knowing whether
            the model that ruled was one of the two arguing, so it is recorded, not implied.
        findings: Every point the reviewer raised.
        accepted: The findings the implementer applied, each with the reason it gave.
        disputes: The findings the implementer rejected.
        resolutions: One entry per dispute the orchestrator settled.
        sent_back_after_ruling: Ids of the findings the orchestrator ruled for the reviewer
            on, which the implementer was then sent back to apply. It records that the turn
            was asked for, not that the files changed: nothing here checks the edits, so the
            name says sent back rather than applied. The suite is run after that turn.
        unanswered: Ids of findings the implementer never answered. These are neither
            accepted nor disputed, and were once indistinguishable from accepted.
        unresolved: One "<finding id>: <reason>" line per dispute the orchestrator could
            not settle, so a failed resolution never reads the same as a clean run.
        protocol_errors: Malformed responses that broke the protocol without stopping the
            run, for example a decision naming a finding that does not exist.
        probe_failures: One "<finding id>: <reason>" line per dispute whose probe could not
            be produced. The dispute is still ruled on, so this is what separates a ruling
            made without evidence by choice from one made without it by accident.
        cut_short: One "<role>: <reason>" line per turn that ended early, whether or not
            its partial text then parsed. A turn stopped by the per-turn budget or by the
            internal turn cap answers with whatever it had reached, and a short answer can
            parse perfectly well, so without this a run whose caps fired repeatedly reads
            as a clean run.
        failure: Why the run stopped early, if it did. None on a completed run.
        suite_status: What happened when the harness ran the agent's own tests: "passed",
            "failed", "missing" or "error". None if the suite was never run. This measures
            the suite the agent wrote, not correctness, which needs held-out tests.
        tests_passed: How many of the agent's tests passed.
        tests_failed: How many failed or errored.
        usage: Per-role accounting, one entry per role that ran.
        started_at: When the record was created, as an ISO 8601 string in UTC. Two runs of
            the same task and arm are otherwise identical rows, so a re-run cannot be told
            from the run it replaced, and the results file already carries lines that
            differ only in what they measured.
        seconds: Wall clock for the whole run.
    """

    task_id: str
    arm: Arm
    implementer_model: str
    reviewer_model: str | None = None
    orchestrator_model: str | None = None
    findings: list[Finding] = field(default_factory=list)
    accepted: list[Acceptance] = field(default_factory=list)
    disputes: list[Dispute] = field(default_factory=list)
    resolutions: list[Resolution] = field(default_factory=list)
    sent_back_after_ruling: list[str] = field(default_factory=list)
    unanswered: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    protocol_errors: list[str] = field(default_factory=list)
    probe_failures: list[str] = field(default_factory=list)
    cut_short: list[str] = field(default_factory=list)
    failure: str | None = None
    suite_status: str | None = None
    tests_passed: int = 0
    tests_failed: int = 0
    usage: list[RoleUsage] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Return every token moved, cached and uncached, summed across every role."""
        return sum(
            entry.input_tokens + entry.cache_read_tokens + entry.cache_creation_tokens + entry.output_tokens
            for entry in self.usage
        )

    @property
    def uncached_tokens(self) -> int:
        """Return uncached input plus output, the part billed at full rate."""
        return sum(entry.input_tokens + entry.output_tokens for entry in self.usage)

    @property
    def total_cost_usd(self) -> float:
        """Return the summed cost across every role."""
        return sum(entry.cost_usd for entry in self.usage)


class RunFailed(Exception):
    """Raised when a run died mid-flight, carrying the record of what was already paid for.

    An exception escaping a run leaves its caller with nothing but the empty record it
    built beforehand, so the run is written out reporting zero cost for turns that were
    genuinely billed. Carrying the record out with the exception is what lets the caller
    write down what the failed run actually spent.

    Attributes:
        record: The run so far, with its failure and its usage already set.
    """

    def __init__(self, record: RunRecord) -> None:
        """Store the partially completed record and take the failure text as the message.

        Args:
            record: The run so far, with `failure` and `usage` already populated.
        """
        super().__init__(record.failure or "the run failed")
        self.record = record
