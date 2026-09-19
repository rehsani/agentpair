"""Extracting structured output from model prose.

Every function here raises rather than returning a default. A findings block that fails
to parse must not become an empty list, because an empty list is indistinguishable from
a clean review and would silently record the reviewer as having found nothing.

A field the harness branches on is parsed strictly and never coerced. A field that is only
ever pasted into a later prompt is rendered instead, so a list of line references costs no
retry. Which bucket a field is in is checkable rather than a judgment call: nothing here
tests the value of evidence or reason.

All three roles answer in a JSON fence: the reviewer's findings, the implementer's
decisions, and both of the orchestrator's turns. The orchestrator used to answer in prose,
and the verdict and the refusal to write a probe were then recovered by searching that
prose for a keyword. Four review rounds each found the keyword search reading the wrong
line, and each fix narrowed the search rather than removing the need for one. There is no
prose search left here: a side the orchestrator did not choose is now a JSON value it did
not write, which cannot be produced by quoting the format back or by reasoning about the
losing side.

Fences are scanned from the end, so a model that restates the requested format before
producing its real answer does not have its example parsed as the answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agentpair.models import Finding, Side

# The language tag may carry a digit ("python3") and may be followed by trailing spaces or
# a carriage return, none of which an [a-zA-Z] tag followed immediately by \n would match.
_FENCE = re.compile(r"```[ \t]*(?P<lang>[A-Za-z0-9_+#.\-]*)[ \t]*\r?\n(?P<body>.*?)```", re.DOTALL)

# The two answers the orchestrator's ruling turn may give. Anything else is a parse
# failure, which is recorded as an unresolved dispute and is visible in the record.
VERDICT_SIDES = frozenset({"implementer", "reviewer"})


class ParseError(ValueError):
    """Raised when a model's output does not carry the structure the role was asked for."""


@dataclass(frozen=True)
class Decision:
    """The implementer's answer to one finding.

    A wire shape. It is consumed immediately: an accepted finding becomes an Acceptance on
    the record and a rejected one becomes a Dispute, so both halves of the answer are kept
    rather than the acceptances being inferred later from what is left.

    Attributes:
        finding_id: The finding being answered.
        accept: Whether the implementer applied it.
        reason: Its stated reason, which becomes its position if this becomes a dispute,
            and is recorded beside the finding when it is accepted.
    """

    finding_id: str
    accept: bool
    reason: str


@dataclass(frozen=True)
class Verdict:
    """The orchestrator's ruling on one dispute.

    Both fields come out of the ruling's JSON object, so the record's rationale is the
    reasoning the orchestrator wrote under "reason" and not its whole response with the
    narration, the tool talk and the fence still in it.

    Attributes:
        favours: The side it ruled for.
        reason: Why, in its own words. Empty when it gave none, which the format asks for
            but which is not worth another paid turn to insist on: the side is the answer,
            and a missing reason is visible as an empty one.
    """

    favours: Side
    reason: str = ""


def extract_block(
    text: str,
    lang: str,
) -> str:
    """Return the body of the last fenced block of the given language.

    The tag is compared case-insensitively, so ```JSON is the same fence as ```json. There
    is no fallback to an untagged fence. That fallback existed because the probe was a
    python fence and a model that forgets a tag would otherwise have cost three turns; the
    probe is now a string inside the same ```json object every other role answers in, which
    is the format the three roles have obeyed across every real run, and guessing at an
    untagged block only reintroduces the question of whether the block was an answer or a
    quotation.

    Args:
        text: The model's full response.
        lang: Fence language to match. Every role answers in "json".

    Returns:
        The block body, stripped.

    Raises:
        ParseError: If no fence of that language is present.
    """
    matches = [match for match in _FENCE.finditer(text) if match.group("lang").casefold() == lang.casefold()]
    if matches:
        return matches[-1].group("body").strip()
    raise ParseError(f"no ```{lang} block found in the response")


def _load_json(
    text: str,
    what: str,
) -> object:
    """Parse the last JSON fence and return whatever it held.

    Args:
        text: The model's full response.
        what: Noun used in error messages, for example "findings".

    Returns:
        The parsed value, of whatever JSON type the fence carried.

    Raises:
        ParseError: If no json fence is present, or its body is not valid JSON.
    """
    body = extract_block(text, "json")
    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise ParseError(f"{what} block is not valid JSON: {error}") from error


def _load_array(
    text: str,
    what: str,
) -> list[dict]:
    """Parse the last JSON fence and require it to hold an array of objects.

    Args:
        text: The model's full response.
        what: Noun used in error messages, for example "findings".

    Returns:
        The parsed array.

    Raises:
        ParseError: If the block is not valid JSON, or is not an array of objects.
    """
    parsed = _load_json(text, what)
    if not isinstance(parsed, list):
        raise ParseError(f"{what} block must be a JSON array, got {type(parsed).__name__}")
    if any(not isinstance(item, dict) for item in parsed):
        raise ParseError(f"every entry in the {what} array must be an object")
    return parsed


def _load_object(
    text: str,
    what: str,
) -> dict:
    """Parse the last JSON fence and require it to hold one object.

    Args:
        text: The model's full response.
        what: Noun used in error messages, for example "probe".

    Returns:
        The parsed object.

    Raises:
        ParseError: If the block is not valid JSON, or is not a JSON object.
    """
    parsed = _load_json(text, what)
    if not isinstance(parsed, dict):
        raise ParseError(f"{what} block must be a JSON object, got {type(parsed).__name__}")
    return parsed


def _require_text(
    item: dict,
    key: str,
    what: str,
) -> str:
    """Return a required field as a non-blank string, or raise.

    The dataclasses reject a blank id or claim by raising a bare ValueError, and ParseError
    subclasses ValueError rather than the reverse, so that exception escapes the loop's
    `except ParseError` and takes the whole run's accounting with it. Coercing with str()
    first is worse than not checking at all: a JSON null becomes the literal "None" and
    passes. So the check happens here, where the failure is a parse failure.

    Args:
        item: The entry being read.
        key: Field name to read.
        what: Noun used in error messages, for example "finding".

    Returns:
        The field's value, stripped of surrounding whitespace.

    Raises:
        ParseError: If the value is not a string, or is blank.
    """
    value = item.get(key)
    if not isinstance(value, str):
        raise ParseError(f"{what} {key} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ParseError(f"{what} {key} must not be empty")
    return value.strip()


def _as_text(value: object) -> str:
    """Render any JSON value as the text an optional field carries.

    Args:
        value: The parsed JSON value.

    Returns:
        A string. Lists are joined, and anything else is dumped as JSON, so the content
        survives whatever container the model chose to put it in.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(part for part in (_as_text(item) for item in value) if part)
    return json.dumps(value, ensure_ascii=False)


def _optional_text(
    item: dict,
    key: str,
) -> str:
    """Return an optional field as text, defaulting to empty when absent or null.

    A present null is absent. JSON has no other way to say "I have nothing to put here",
    so a reviewer emitting `"evidence": null` for a finding it cannot cite is not
    malformed.

    A present value of any other type is rendered rather than refused. The format asks for
    a string and says so, but a reviewer that cites two lines as ["mod.py:2", "mod.py:5"]
    has written a usable finding, and refusing it cost a retry for a field nothing branches
    on: evidence and reason are only ever pasted into a later prompt, so there is no second
    reading of what a rendered list means. Required fields stay strict, because there the
    value is the answer and coercing it would invent a decision the model did not make.

    Args:
        item: The entry being read.
        key: Field name to read.

    Returns:
        The field's value as text, or "" when the field is absent or null.
    """
    value = item.get(key)
    if value is None:
        return ""
    return _as_text(value)


def _normalise_id(finding_id: str) -> str:
    """Reduce a finding id to the one spelling the record and the lookups use.

    The reviewer invents the id and the implementer echoes it back, and the two are matched
    by string equality, so "F1" answered as "f1" matched nothing: the decision was recorded
    as naming no known finding and the finding as never answered, which lost a decision the
    implementer had actually made. Both sides are folded here instead, so there is one
    spelling and the duplicate check sees "F1" and "f1" as the collision they are.

    Args:
        finding_id: The id as the model wrote it, already stripped.

    Returns:
        The id case folded.
    """
    return finding_id.casefold()


def parse_findings(text: str) -> list[Finding]:
    """Parse the reviewer's findings.

    Args:
        text: The reviewer's full response.

    Returns:
        The findings, which is empty only when the reviewer emitted an empty array.

    Raises:
        ParseError: If the block is missing or malformed, or an entry lacks a field, or an
            id or claim is blank or not a string.
    """
    findings = []
    seen: set[str] = set()
    for item in _load_array(text, "findings"):
        missing = {"id", "claim"} - item.keys()
        if missing:
            raise ParseError(f"finding is missing {sorted(missing)}")
        finding_id = _normalise_id(_require_text(item, "id", "finding"))
        if finding_id in seen:
            raise ParseError(f"duplicate finding id {finding_id!r}; ids must be unique")
        seen.add(finding_id)
        findings.append(
            Finding(
                id=finding_id,
                claim=_require_text(item, "claim", "finding"),
                evidence=_optional_text(item, "evidence"),
            )
        )
    return findings


def parse_decisions(text: str) -> list[Decision]:
    """Parse the implementer's accept or reject answer to each finding.

    Args:
        text: The implementer's full response.

    Returns:
        One decision per entry, in the order given.

    Raises:
        ParseError: If the block is missing or malformed, an entry lacks a field, an id is
            blank or not a string, or two decisions answer the same finding.
    """
    decisions = []
    answered: set[str] = set()
    for item in _load_array(text, "decisions"):
        missing = {"id", "accept"} - item.keys()
        if missing:
            raise ParseError(f"decision is missing {sorted(missing)}")
        if not isinstance(item["accept"], bool):
            raise ParseError(f"decision {item['id']}: accept must be true or false")
        finding_id = _normalise_id(_require_text(item, "id", "decision"))
        if finding_id in answered:
            raise ParseError(f"duplicate decision for finding {finding_id!r}")
        answered.add(finding_id)
        decisions.append(
            Decision(
                finding_id=finding_id,
                accept=item["accept"],
                reason=_optional_text(item, "reason"),
            )
        )
    return decisions


def parse_probe(text: str) -> str | None:
    """Parse the orchestrator's answer to whether running code would settle the dispute.

    The turn answers in the same ```json fence the reviewer and the implementer use, so
    the probe is a JSON string and not a fenced script inside prose. That removes three
    questions that cost paid turns in earlier rounds: which fence tag the script carried,
    whether an untagged fence was the script or the disputed code quoted back, and whether
    a refusal written somewhere in the response outranked a script written elsewhere in
    it. A script that prints three backticks still truncates the fence it is carried in,
    unchanged from the previous contract and still not worth a fence rewrite.

    Args:
        text: The orchestrator's full response.

    Returns:
        The probe's Python source, or None when the orchestrator answered that nothing it
        could run would settle the dispute.

    Raises:
        ParseError: If the block is missing or malformed, the "probe" key is absent, or
            its value is neither a string nor null. A blank string is refused too: it is
            neither a script nor the null that declines to write one, so reading it as
            either would put words in the orchestrator's mouth.
    """
    parsed = _load_object(text, "probe")
    if "probe" not in parsed:
        raise ParseError('probe object is missing "probe"')
    probe = parsed["probe"]
    if probe is None:
        return None
    if not isinstance(probe, str):
        raise ParseError(f"probe must be a string or null, got {type(probe).__name__}")
    if not probe.strip():
        raise ParseError('probe must not be empty; answer {"probe": null} to decline')
    return probe


def parse_verdict(text: str) -> Verdict:
    """Parse the orchestrator's ruling.

    The ruling is a JSON value, so the only way to record a side is for the orchestrator
    to have written that side as its answer. Naming a side while reasoning, quoting the
    two allowed answers back, or ruling in prose above the fence cannot change it, and
    each of those inverted a recorded verdict in an earlier round.

    Args:
        text: The orchestrator's full response.

    Returns:
        The side it favoured and the reason it gave.

    Raises:
        ParseError: If the block is missing or malformed, "favours" is absent, or it is
            not exactly "implementer" or "reviewer". A refusal is recorded as an
            unresolved dispute, which a reader sees; a verdict read off the wrong line is
            not.
    """
    parsed = _load_object(text, "verdict")
    if "favours" not in parsed:
        raise ParseError('verdict object is missing "favours"')
    favours = parsed["favours"]
    if not isinstance(favours, str):
        raise ParseError(f"favours must be a string, got {type(favours).__name__}")
    side = favours.strip().casefold()
    if side not in VERDICT_SIDES:
        raise ParseError(f'favours must be "implementer" or "reviewer", got {favours!r}')
    reason = _optional_text(parsed, "reason")
    return Verdict(favours="implementer" if side == "implementer" else "reviewer", reason=reason)
