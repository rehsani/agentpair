"""The run loop.

Drives one task through implement, review, decide and resolve, then stops.

The implementer writes the code and its tests. The reviewer, a different model, raises
findings with evidence. The implementer applies what it accepts and rejects the rest with
a reason. Every rejection goes to the orchestrator, which decides who is right. The
orchestrator may run code first when running something would settle it, and the record
says which of the two happened. Where it ruled for the reviewer, the implementer is sent
back once to apply what it lost, so a ruling changes the code rather than only the record.
That is the whole sequence; nothing here repeats it.

All three roles answer in a JSON fence, the orchestrator's two turns included, so nothing
here reads a side or a refusal out of model prose. An orchestrator turn whose answer does
not parse is retried, because that is its instrument failing and not evidence about the
disputed code.

The model calls go through an injected runner, so the loop can be exercised end to end
with scripted responses and no SDK session.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from agentpair import roles
from agentpair.arbitration import SelfArbitration, self_arbitration_reason
from agentpair.dispute import (
    ProbeResult,
    execute_probe,
    judged_resolution,
    probed_resolution,
)
from agentpair.metrics import merge_usage
from agentpair.models import (
    Acceptance,
    AgentTurn,
    Dispute,
    Finding,
    Resolution,
    RoleUsage,
    RunFailed,
    Runner,
    RunRecord,
)
from agentpair.parsing import (
    Decision,
    ParseError,
    Verdict,
    parse_decisions,
    parse_findings,
    parse_probe,
    parse_verdict,
)
from agentpair.verify import run_suite

AUTHORING_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]
READING_TOOLS = ["Read", "Glob", "Grep"]

# Attempts allowed for each of the orchestrator's two turns. Both answer in a JSON fence,
# and an answer that does not carry the object, or a probe that does not compile, is the
# orchestrator's own instrument failing rather than evidence about the disputed code. So
# either is retried with the reason fed back, on the same terms for both turns.
DEFAULT_ORCHESTRATOR_ATTEMPTS = 3

# Attempts allowed for the reviewer's turn, on the same reasoning and with the reason fed
# back the same way. It had one, and an answer that did not carry the array ended the run:
# the turn whose failure is most expensive was the one turn given no second chance, while
# the two cheap orchestrator turns retried three times. A review that arrives wrapped in an
# object, or with its array quoted back before the real one, is the reviewer's formatting
# and not a statement that the code is clean, so it is re-asked rather than recorded as a
# dead run.
DEFAULT_REVIEWER_ATTEMPTS = 3

# Attempts allowed for the implementer's decision turn. It had one, and a decision list
# that did not parse ended the run with every finding already paid for and none of them
# answered. This turn differs from the other retried ones in that it edits files, so its
# re-ask says the edits already made stand; see _decision_retry_prompt.
DEFAULT_DECISION_ATTEMPTS = 3


def _format_findings(findings: list[Finding]) -> str:
    """Render findings as the text the implementer is shown.

    Args:
        findings: The reviewer's findings.

    Returns:
        One block per finding, carrying its id so decisions can be matched back, and its
        evidence so the implementer answers the argument rather than the headline.
    """
    blocks = []
    for finding in findings:
        block = f"{finding.id}: {finding.claim}"
        if finding.evidence:
            block += f"\n    evidence: {finding.evidence}"
        blocks.append(block)
    return "\n".join(blocks)


@dataclass(frozen=True)
class TaskContext:
    """The task a dispute is about, as the orchestrator has to be shown it.

    The orchestrator used to be given the two sides' arguments and nothing else. It was
    told to import the module without being told what the module is called, so locating it
    cost tool calls or produced a probe that would not compile; and a dispute over whether
    the code meets the specification was decided by the one party that had never read the
    specification. The harness holds all three of these at the point it builds the prompt,
    so they are passed rather than rediscovered.

    Attributes:
        spec: The task specification, the same text the implementer and reviewer were given.
        module: Implementation filename, in the workspace the orchestrator's turns run in.
        test: Test filename, in that same workspace.
    """

    spec: str
    module: str
    test: str


def _format_rulings(resolutions: list[Resolution]) -> str:
    """Render the rulings that went against the implementer as the text it must apply.

    Args:
        resolutions: The resolutions that favoured the reviewer.

    Returns:
        One block per ruling, carrying the finding, the position the implementer took, and
        the orchestrator's rationale. The rationale is included because the implementer
        rejected this finding once on the merits, and an order to apply it that does not
        say what overrode its reasoning is the reviewer's claim repeated, not a ruling.
    """
    blocks = []
    for resolution in resolutions:
        finding = resolution.dispute.finding
        block = f"{finding.id}: {finding.claim}"
        if finding.evidence:
            block += f"\n    reviewer's evidence: {finding.evidence}"
        block += f"\n    you rejected it, saying: {resolution.dispute.implementer_position}"
        block += f"\n    the orchestrator ruled for the reviewer: {resolution.rationale}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _describe(
    dispute: Dispute,
    task: TaskContext,
) -> str:
    """Render a dispute as the text both orchestrator turns are shown.

    The specification comes first because it is what the argument is measured against, and
    the filenames come next because the probe turn is asked to import the module and the
    ruling turn is asked to read the lines the argument turns on. Both were previously left
    for the orchestrator to guess at.

    Args:
        dispute: The disagreement.
        task: The specification and the filenames the argument is about.

    Returns:
        The specification, the files it produced, the reviewer's claim and evidence, and
        the implementer's position.
    """
    evidence = dispute.finding.evidence or "(none given)"
    return (
        f"The task was:\n\n{task.spec}\n\n"
        f"It was implemented in {task.module}, with tests in {task.test}. "
        f"Both are in your working directory.\n\n"
        f"Reviewer claims: {dispute.finding.claim}\n"
        f"Reviewer's evidence: {evidence}\n"
        f"Implementer rejected it, saying: {dispute.implementer_position}"
    )


def _record_turn(
    turn: AgentTurn,
    record: RunRecord,
    usage: list[RoleUsage],
    role: str,
) -> AgentTurn:
    """Account for a finished turn: keep its usage, and note it if it ended early.

    Every turn goes through here so the cap firing is written down once, at the point it
    is known, rather than only where a parse also failed. A cut-short turn whose partial
    text happens to parse is the case that used to leave no trace at all.

    Args:
        turn: The finished turn.
        record: The run record, whose cut_short list gains a line when the turn ended early.
        usage: Accumulator the turn's accounting is appended to.
        role: Whose turn this was, named in the cut_short line.

    Returns:
        The same turn, so a caller can account for it and use it in one expression.
    """
    usage.append(turn.usage)
    if turn.cut_short:
        record.cut_short.append(f"{role}: {turn.cut_short}")
    return turn


def _retry_prompt(
    prompt: str,
    reason: str,
) -> str:
    """Re-ask an orchestrator turn, naming what was wrong with its last answer.

    Args:
        prompt: The original description of the dispute.
        reason: Why the last answer could not be used.

    Returns:
        The same dispute with the reason appended, so the turn is retried against the
        failure rather than against silence.
    """
    return f"{prompt}\n\nYour last answer could not be used: {reason}. Answer in the format you were given."


def _decision_retry_prompt(
    prompt: str,
    reason: str,
) -> str:
    """Re-ask the implementer for its decision list without re-ordering the edits.

    The reviewer's turn and the orchestrator's two turns only read, so re-asking them costs
    tokens and nothing else. This turn edits the module and the tests, and a plain re-ask
    reads as the whole instruction repeated, which invites the findings it already applied
    to be applied a second time. So the re-ask states that the edits stand and that what is
    missing is the list.

    Args:
        prompt: The original findings message.
        reason: Why the last answer could not be used.

    Returns:
        The same message with the reason and the standing edits appended.
    """
    return (
        f"{prompt}\n\nYour last answer could not be used: {reason}. Any edits you have already made "
        "are kept, so do not apply them a second time. Answer with the decision list in exactly the "
        "format you were given, and write no prose after it."
    )


def _decision_failure(
    turn: AgentTurn,
    error: ParseError,
) -> str:
    """Name why a decision turn could not be read.

    Args:
        turn: The decision turn.
        error: The parse failure.

    Returns:
        The reason, fed back to the next attempt and recorded as the run's failure if every
        attempt fails. The wording matches what a single unparseable answer already
        recorded, so the results file reads the same either way.
    """
    if turn.cut_short:
        return f"implementer answer unusable because {turn.cut_short}"
    return f"implementer answer did not parse: {error}"


async def _decide(
    findings: list[Finding],
    module: str,
    test: str,
    workspace: Path,
    record: RunRecord,
    runner: Runner,
    usage: list[RoleUsage],
    style: str,
) -> list[Decision]:
    """Ask the implementer to answer every finding, re-asking an unreadable answer.

    Args:
        findings: The reviewer's findings, shown with their evidence.
        module: Implementation filename the implementer edits.
        test: Test filename the implementer edits.
        workspace: Directory the agent works in.
        record: The run record, read for the implementer's model, appended to when a turn
            ends early, and whose failure is set when every attempt fails.
        runner: The model runner.
        usage: Accumulator every turn is appended to.
        style: Style brief handed to the implementer.

    Returns:
        One decision per entry the implementer wrote, or an empty list when no attempt
        produced a usable answer, in which case the record carries the failure.
    """
    prompt = roles.IMPLEMENTER_RESPONSE.substitute(findings=_format_findings(findings))
    last_reason = "implementer answer did not parse"
    for attempt in range(DEFAULT_DECISION_ATTEMPTS):
        answered = _record_turn(
            await runner(
                role="implementer",
                system=roles.IMPLEMENTER_ANSWER_SYSTEM.substitute(module=module, test=test, style=style),
                prompt=prompt if attempt == 0 else _decision_retry_prompt(prompt, last_reason),
                model=record.implementer_model,
                workspace=workspace,
                tools=AUTHORING_TOOLS,
            ),
            record,
            usage,
            "implementer",
        )
        try:
            return parse_decisions(answered.text)
        except ParseError as error:
            last_reason = _decision_failure(answered, error)
    record.failure = f"{last_reason} (after {DEFAULT_DECISION_ATTEMPTS} attempts)"
    record.usage = merge_usage(usage)
    return []


def _probe_failure(
    turn: AgentTurn,
    error: ParseError,
) -> str:
    """Name why a probe turn could not be read.

    A cap firing and a model ignoring the format produce the same unusable text, and the
    reasons exist to tell those apart, so the cap is named whenever it fired.

    Args:
        turn: The probe turn.
        error: The parse failure.

    Returns:
        The reason, fed back to the next attempt and recorded if every attempt fails.
    """
    if turn.cut_short:
        return f"the probe turn was unusable because {turn.cut_short}"
    return str(error)


def _ruling_failure(
    turn: AgentTurn,
    error: ParseError,
) -> str:
    """Name why a ruling turn could not be read.

    Args:
        turn: The ruling turn.
        error: The parse failure.

    Returns:
        The reason, fed back to the next attempt and recorded as the dispute's unresolved
        line if every attempt fails.
    """
    if turn.cut_short:
        return f"the orchestrator's ruling was unusable because {turn.cut_short}"
    return f"the orchestrator gave no usable verdict: {error}"


def _review_failure(
    turn: AgentTurn,
    error: ParseError,
) -> str:
    """Name why a review turn could not be read.

    The wording matches what a single unparseable review already recorded, so a reader of
    the results file sees the same reason whether it was reached in one attempt or three.

    Args:
        turn: The review turn.
        error: The parse failure.

    Returns:
        The reason, fed back to the next attempt and recorded as the run's failure if
        every attempt fails.
    """
    if turn.cut_short:
        return f"reviewer answer unusable because {turn.cut_short}"
    return f"reviewer answer did not parse: {error}"


async def _review(
    spec: str,
    module: str,
    test: str,
    workspace: Path,
    record: RunRecord,
    runner: Runner,
    usage: list[RoleUsage],
) -> list[Finding]:
    """Ask the reviewer for its findings, re-asking an answer that carries no array.

    An empty array is a clean review and is returned as one. A missing or malformed array
    is the reviewer's instrument failing, and is retried with the reason fed back, exactly
    as the orchestrator's two turns are.

    Args:
        spec: The task specification, shown so the review is against the task.
        module: Implementation filename to review.
        test: Test filename to review.
        workspace: Directory holding the files.
        record: The run record, read for the reviewer's model, appended to when a turn ends
            early, and whose failure is set when every attempt fails.
        runner: The model runner.
        usage: Accumulator every turn is appended to.

    Returns:
        The findings, or an empty list when no attempt produced a usable answer, in which
        case the record carries the failure and the usage paid for the attempts.
    """
    prompt = f"The task was:\n\n{spec}\n\nReview {module} and {test} in this directory."
    last_reason = "reviewer answer did not parse"
    for attempt in range(DEFAULT_REVIEWER_ATTEMPTS):
        reviewed = _record_turn(
            await runner(
                role="reviewer",
                system=roles.REVIEWER_SYSTEM,
                prompt=prompt if attempt == 0 else _retry_prompt(prompt, last_reason),
                model=record.reviewer_model,
                workspace=workspace,
                tools=READING_TOOLS,
            ),
            record,
            usage,
            "reviewer",
        )
        try:
            return parse_findings(reviewed.text)
        except ParseError as error:
            last_reason = _review_failure(reviewed, error)
    record.failure = f"{last_reason} (after {DEFAULT_REVIEWER_ATTEMPTS} attempts)"
    record.usage = merge_usage(usage)
    return []


async def _write_probe(
    dispute: Dispute,
    task: TaskContext,
    workspace: Path,
    record: RunRecord,
    runner: Runner,
    usage: list[RoleUsage],
    index: int,
) -> tuple[Path | None, ProbeResult | None, str | None]:
    """Ask the orchestrator whether code would settle this, and run what it writes.

    The turn answers with {"probe": "<script>"} or {"probe": null}, so writing a script and
    declining to write one are two values of one field rather than a fence to be found and a
    phrase to be searched for. There is nothing left to arbitrate between: a response cannot
    both carry a probe and refuse to write one, a quotation cannot be mistaken for the
    instrument, and the fence tag no longer decides anything.

    A probe that will not compile, and an answer that does not carry the object at all, are
    both retried with the reason fed back. Both are the orchestrator's instrument failing
    rather than evidence about the disputed code.

    Args:
        dispute: The disagreement.
        task: The specification and the filenames the argument is about.
        workspace: Directory the probe runs against.
        record: The run record, read for the orchestrator's model and appended to when a
            turn ends early.
        runner: The model runner.
        usage: Accumulator every turn is appended to.
        index: Position of this dispute in the run, used to keep probe filenames distinct.

    Returns:
        The probe path and result, or (None, None, reason) when no probe was run. The
        reason is None when the orchestrator answered {"probe": null}, which is a valid
        outcome, and a string when the probe could not be produced.
    """
    prompt = _describe(dispute, task)
    last_reason = "no probe was produced"
    for attempt in range(DEFAULT_ORCHESTRATOR_ATTEMPTS):
        written = _record_turn(
            await runner(
                role="orchestrator_probe",
                system=roles.ORCHESTRATOR_PROBE_SYSTEM,
                prompt=prompt if attempt == 0 else _retry_prompt(prompt, last_reason),
                model=record.orchestrator_model,
                workspace=workspace,
                tools=READING_TOOLS,
            ),
            record,
            usage,
            "orchestrator_probe",
        )
        try:
            source = parse_probe(written.text)
        except ParseError as error:
            last_reason = _probe_failure(written, error)
            continue
        if source is None:
            return None, None, None
        try:
            probe_path, result = execute_probe(source, workspace, dispute.finding.id, index)
        except SyntaxError as error:
            last_reason = f"the script did not compile: {error}"
            continue
        return probe_path, result, None
    return None, None, f"{last_reason} (after {DEFAULT_ORCHESTRATOR_ATTEMPTS} attempts)"


async def _ask_for_a_ruling(
    prompt: str,
    workspace: Path,
    record: RunRecord,
    runner: Runner,
    usage: list[RoleUsage],
) -> tuple[Verdict | None, str | None]:
    """Ask the orchestrator to rule, retrying an answer that carries no verdict.

    The ruling turn answers with {"favours": "implementer"} or {"favours": "reviewer"}, so
    the recorded side is a value the orchestrator wrote and not a side inferred from where
    a keyword appeared in its prose. An answer that carries no such object is retried on
    the same terms as a probe that will not compile: both are the instrument failing, and a
    dispute abandoned for a formatting slip has already cost four paid turns.

    Args:
        prompt: The dispute, with the probe's output or the reason there is none.
        workspace: The run's working directory, which the turn may read.
        record: The run record, read for the orchestrator's model and appended to when a
            turn ends early.
        runner: The model runner.
        usage: Accumulator every turn is appended to.

    Returns:
        The ruling and None, or None and the reason no verdict could be read after every
        attempt. The rationale on the record comes from the ruling's own "reason" field,
        so nothing downstream holds the turn's raw response as data: a response carrying
        narration, a quoted fence and the answer is stored as the answer.
    """
    last_reason = "the orchestrator gave no usable verdict"
    for attempt in range(DEFAULT_ORCHESTRATOR_ATTEMPTS):
        ruling = _record_turn(
            await runner(
                role="orchestrator_rule",
                system=roles.ORCHESTRATOR_RULE_SYSTEM,
                prompt=prompt if attempt == 0 else _retry_prompt(prompt, last_reason),
                model=record.orchestrator_model,
                workspace=workspace,
                tools=READING_TOOLS,
            ),
            record,
            usage,
            "orchestrator_rule",
        )
        try:
            return parse_verdict(ruling.text), None
        except ParseError as error:
            last_reason = _ruling_failure(ruling, error)
    return None, f"{last_reason} (after {DEFAULT_ORCHESTRATOR_ATTEMPTS} attempts)"


async def _resolve(
    dispute: Dispute,
    task: TaskContext,
    workspace: Path,
    record: RunRecord,
    runner: Runner,
    usage: list[RoleUsage],
    index: int,
) -> tuple[Resolution | None, str | None, str | None]:
    """Settle one dispute: run code if that would help, then rule.

    The orchestrator decides who is right, and running code is a way to help it decide, not
    a precondition for deciding. A probe that could not be produced therefore degrades to a
    ruling on the arguments, exactly as a declined probe does, rather than throwing away
    four paid turns and recording nothing. The ruling turn is told which of the two
    happened.

    The ruling turn is given the reading tools rather than nothing. Ruling from the two
    paragraphs of prose alone is the common path, not the fallback, since every declined
    probe lands there, and each turn opens a fresh session, so whatever the probe turn
    read is gone by the time this one runs. Reading tools were chosen over inlining the
    module and test, because the orchestrator can then read whichever file the argument
    turns on, including one neither side named, without the prompt guessing in advance.

    Args:
        dispute: The disagreement.
        task: The specification and the filenames the argument is about.
        workspace: The run's working directory.
        record: The run record, read for the orchestrator's model and appended to when a
            turn ends early.
        runner: The model runner.
        usage: Accumulator every turn is appended to.
        index: Position of this dispute in the run, used to keep probe filenames distinct.

    Returns:
        The resolution and None, or None and the reason it could not be settled, plus the
        reason the probe could not be produced when the ruling was made without one.
    """
    probe_path, result, probe_failure = await _write_probe(dispute, task, workspace, record, runner, usage, index)

    prompt = _describe(dispute, task)
    if result is not None:
        prompt += f"\n\nYou ran a probe. It printed:\n{result.observation}"
    elif probe_failure is not None:
        prompt += f"\n\nYou tried to write a probe and it could not be run: {probe_failure}. Rule on the arguments."
    else:
        prompt += "\n\nNothing you could run would settle this, so rule on the arguments."

    verdict, ruling_failure = await _ask_for_a_ruling(prompt, workspace, record, runner, usage)
    if verdict is None:
        return None, ruling_failure, probe_failure

    if result is not None and probe_path is not None:
        return (
            probed_resolution(dispute, verdict.favours, probe_path, result, verdict.reason),
            None,
            probe_failure,
        )
    return judged_resolution(dispute, verdict.favours, verdict.reason), None, probe_failure


async def run_pair(
    task_id: str,
    spec: str,
    module: str,
    test: str,
    workspace: Path,
    implementer_model: str,
    reviewer_model: str,
    orchestrator_model: str,
    runner: Runner,
    allow_self_arbitration: bool = False,
    style: str = roles.DEFAULT_STYLE,
) -> RunRecord:
    """Run one task through the pair arm and return its record.

    The orchestrator model has no default. It used to fall back to the reviewer's, which
    meant every caller that did not name one had the model that raised the findings ruling
    on whether they were real, and said nothing about it. The command line refused that,
    but the library is a caller too and the refusal has to be where the run is, not only
    where the flags are parsed. The check is nominal and cannot resolve an alias such as
    "opus" to a full model id; see agentpair.arbitration.

    A blank orchestrator model is refused for the same reason. The self-arbitration guard
    passes it, since "" is nobody's model, so without this check a programmatic caller
    that forgot the argument reaches the SDK with model="" on the first dispute, three
    turns into a paid run.

    Args:
        task_id: Identifier recorded with the run.
        spec: The task specification given to the implementer.
        module: Implementation filename to produce.
        test: Test filename to produce.
        workspace: Directory the agents work in.
        implementer_model: Model for the implementer.
        reviewer_model: Model for the reviewer.
        orchestrator_model: Model that settles disputes. Required, and refused when it is
            a disputant's own model unless allow_self_arbitration is set.
        runner: The model runner.
        allow_self_arbitration: Accept a disputant's own model as the orchestrator. The run
            still records which model ruled, so the reader can weigh it.
        style: Style brief handed to the implementer.

    Returns:
        The run record, including every finding, dispute and resolution.

    Raises:
        ValueError: If orchestrator_model is missing, blank or only whitespace. Raised
            before any turn is paid for, and before the self-arbitration guard, which has
            nothing to say about a name that is nobody's.
        SelfArbitration: If the orchestrator model is the implementer's or the reviewer's
            and allow_self_arbitration was not set. Raised before any turn is paid for.
        RunFailed: If a turn raised. The exception carries the record, with the failure and
            the usage paid for so far already on it, so the caller can write down what the
            failed run cost instead of reporting zero.
    """
    if not orchestrator_model or not orchestrator_model.strip():
        raise ValueError(
            "refusing to run: orchestrator_model must name the model that settles disputes. "
            "A blank one reaches the SDK as model='' on the first dispute."
        )

    if not allow_self_arbitration:
        reason = self_arbitration_reason(implementer_model, reviewer_model, orchestrator_model)
        if reason is not None:
            raise SelfArbitration(
                f"refusing to run: {reason}. Pass a third model as orchestrator_model, or "
                f"allow_self_arbitration=True to accept it."
            )

    workspace.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    usage: list[RoleUsage] = []
    record = RunRecord(
        task_id=task_id,
        arm="pair",
        implementer_model=implementer_model,
        reviewer_model=reviewer_model,
        orchestrator_model=orchestrator_model,
    )
    try:
        await _run_pair(record, spec, module, test, workspace, runner, usage, style)
    except Exception as error:
        record.failure = f"{type(error).__name__}: {error}"
        record.usage = merge_usage(usage)
        record.seconds = time.monotonic() - started
        raise RunFailed(record) from error

    suite = run_suite(workspace, test)
    record.suite_status = suite.status
    record.tests_passed = suite.passed
    record.tests_failed = suite.failed
    record.usage = merge_usage(usage)
    record.seconds = time.monotonic() - started
    return record


async def _run_pair(
    record: RunRecord,
    spec: str,
    module: str,
    test: str,
    workspace: Path,
    runner: Runner,
    usage: list[RoleUsage],
    style: str,
) -> None:
    """Drive the model turns of one pair run, filling the record as it goes.

    Separated from run_pair so the paid turns are accumulated on a record that already
    exists when a turn raises, which is what makes the failure path accountable.

    The implementer speaks twice and the two turns have different jobs, so they are given
    different briefs: the first builds from the specification, the second answers a review
    of what it built and owes a decision list.

    Args:
        record: The run record, mutated in place.
        spec: The task specification given to the implementer.
        module: Implementation filename to produce.
        test: Test filename to produce.
        workspace: Directory the agents work in.
        runner: The model runner.
        usage: Accumulator every turn is appended to.
        style: Style brief handed to the implementer.
    """
    # The build turn's prose is never read; only its accounting and the files it leaves are.
    _record_turn(
        await runner(
            role="implementer",
            system=roles.IMPLEMENTER_SYSTEM.substitute(module=module, test=test, style=style),
            prompt=spec,
            model=record.implementer_model,
            workspace=workspace,
            tools=AUTHORING_TOOLS,
        ),
        record,
        usage,
        "implementer",
    )

    findings = await _review(spec, module, test, workspace, record, runner, usage)
    record.findings = findings
    if not findings or record.failure is not None:
        return

    decisions = await _decide(findings, module, test, workspace, record, runner, usage, style)

    by_id = {finding.id: finding for finding in findings}
    if record.failure is None:
        # Only meaningful when the answer was readable. An unparseable answer means we
        # do not know what was addressed, which is the failure, not a list of omissions.
        answered_ids = {decision.finding_id for decision in decisions}
        record.unanswered = [finding.id for finding in findings if finding.id not in answered_ids]
    for decision in decisions:
        finding = by_id.get(decision.finding_id)
        if finding is None:
            record.protocol_errors.append(f"{decision.finding_id}: decision names no known finding")
            continue
        if decision.accept:
            record.accepted.append(Acceptance(finding=finding, reason=decision.reason))
            continue
        dispute = Dispute(finding=finding, implementer_position=decision.reason or "no reason given")
        record.disputes.append(dispute)
        resolution, reason, probe_failure = await _resolve(
            dispute,
            TaskContext(spec=spec, module=module, test=test),
            workspace,
            record,
            runner,
            usage,
            len(record.disputes),
        )
        if probe_failure is not None:
            record.probe_failures.append(f"{finding.id}: {probe_failure}")
        if resolution is not None:
            record.resolutions.append(resolution)
        else:
            record.unresolved.append(f"{finding.id}: {reason}")

    await _apply_rulings(record, module, test, workspace, runner, usage, style)


async def _apply_rulings(
    record: RunRecord,
    module: str,
    test: str,
    workspace: Path,
    runner: Runner,
    usage: list[RoleUsage],
    style: str,
) -> None:
    """Send the implementer back to apply every finding the orchestrator upheld.

    Without this the orchestrator's ruling is a line in the record and nothing else: a
    finding the implementer rejected stays rejected whether it won the argument or lost it,
    and the code the run is measured on is identical either way. The ruling is the point at
    which the disagreement is over, so it is carried out.

    One turn covers every upheld ruling. The implementer has already argued each of these
    once and lost, so there is nothing further to decide and nothing to parse: the turn
    answers in prose and is read only through the files it leaves and the suite that runs
    after it.

    Args:
        record: The run record, read for its resolutions and appended to with the ids sent
            back, and with a cut_short line if the turn ended early.
        module: Implementation filename the implementer edits.
        test: Test filename the implementer edits.
        workspace: Directory the agent works in.
        runner: The model runner.
        usage: Accumulator the turn's accounting is appended to.
        style: Style brief handed to the implementer.
    """
    upheld = [resolution for resolution in record.resolutions if resolution.favours == "reviewer"]
    if not upheld:
        return

    record.sent_back_after_ruling = [resolution.dispute.finding.id for resolution in upheld]
    _record_turn(
        await runner(
            role="implementer",
            system=roles.IMPLEMENTER_REMEDIATION_SYSTEM.substitute(module=module, test=test, style=style),
            prompt=roles.IMPLEMENTER_REMEDIATION.substitute(
                findings=_format_rulings(upheld),
                module=module,
                test=test,
            ),
            model=record.implementer_model,
            workspace=workspace,
            tools=AUTHORING_TOOLS,
        ),
        record,
        usage,
        "implementer",
    )
