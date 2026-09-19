"""Refusing a run in which a disputant's own model rules on the dispute.

The orchestrator decides who is right, so a run whose orchestrator is the implementer's
or the reviewer's model has one of the two arguing sides ruling on its own argument. The
check lives here, apart from both the command line and the loop, because both are callers
and an earlier version that kept it in the command line let every programmatic caller
self-arbitrate in silence.

The comparison is nominal. Names are compared after stripping surrounding whitespace and
case folding, so "Opus" and " opus" no longer pass a guard aimed at "opus". It cannot go
further than that: the SDK accepts an alias and a full model id interchangeably, so
"opus" against "claude-opus-4-5-20251101" is one model under two names and nothing here
can tell. The guard catches a slip in the flags, not a deliberate disguise.
"""

from __future__ import annotations


class SelfArbitration(ValueError):
    """Raised when the orchestrator model is a disputant's and that was not allowed."""


def normalise_model(name: str) -> str:
    """Reduce a model name to the form the guard compares.

    Args:
        name: A model name or alias as the caller wrote it.

    Returns:
        The name without surrounding whitespace and case folded.
    """
    return name.strip().casefold()


def self_arbitration_reason(
    implementer_model: str,
    reviewer_model: str | None,
    orchestrator_model: str,
) -> str | None:
    """Return why this trio self-arbitrates, or None when the orchestrator is a third party.

    Args:
        implementer_model: Model that wrote the code under dispute.
        reviewer_model: Model that raised the findings, or None when there is no reviewer.
        orchestrator_model: Model that would rule on the disputes.

    Returns:
        A sentence naming the clash, or None when no disputant's model is the orchestrator.
        The comparison is nominal and cannot resolve an alias to a full model id.
    """
    orchestrator = normalise_model(orchestrator_model)
    for role, disputant in (("implementer", implementer_model), ("reviewer", reviewer_model)):
        if disputant is not None and normalise_model(disputant) == orchestrator:
            return (
                f"the orchestrator model ({orchestrator_model}) is the {role}'s ({disputant}), "
                f"so a disputant's own model would rule on the dispute"
            )
    return None
