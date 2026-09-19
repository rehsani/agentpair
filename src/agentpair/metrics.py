"""Per-role accounting for a run.

Reads what the SDK reports at the end of a turn and turns it into a RoleUsage, then
writes finished runs as JSON lines. Kept task-agnostic so any corpus can be dropped in
later without touching this module.

Cost is taken from the SDK rather than re-derived from token counts and a price table,
so a pricing change cannot silently invalidate old results. Cached and uncached input are
kept apart, because merging them makes a token ratio between the two arms unreadable
against their cost ratio.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from agentpair.models import RoleUsage, RunRecord


class SupportsUsage(Protocol):
    """The part of the SDK's ResultMessage this module reads.

    Declared structurally so the accounting can be tested with a plain stand-in and no
    SDK session.
    """

    duration_ms: int
    total_cost_usd: float | None
    usage: dict[str, Any] | None


def usage_from_result(
    role: str,
    result: SupportsUsage,
) -> RoleUsage:
    """Convert one turn's SDK result into a role's accounting entry.

    Args:
        role: Label for the role that produced this turn.
        result: The SDK result message ending the turn.

    Returns:
        The role's token counts, cost and wall clock. Missing usage fields count as zero,
        which is what the SDK reports for a turn that made no model call.
    """
    usage = result.usage or {}
    return RoleUsage(
        role=role,
        input_tokens=int(usage.get("input_tokens", 0)),
        cache_read_tokens=int(usage.get("cache_read_input_tokens", 0)),
        cache_creation_tokens=int(usage.get("cache_creation_input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        cost_usd=float(result.total_cost_usd or 0.0),
        seconds=result.duration_ms / 1000,
    )


def merge_usage(entries: list[RoleUsage]) -> list[RoleUsage]:
    """Combine repeated turns by the same role into one entry each.

    A role speaks more than once in a run, and a per-turn breakdown is noise when the
    comparison is between arms.

    Args:
        entries: Every turn's accounting, in the order the turns happened.

    Returns:
        One entry per role, in first-seen order.
    """
    merged: dict[str, RoleUsage] = {}
    for entry in entries:
        existing = merged.get(entry.role)
        if existing is None:
            merged[entry.role] = entry
            continue
        merged[entry.role] = RoleUsage(
            role=entry.role,
            input_tokens=existing.input_tokens + entry.input_tokens,
            cache_read_tokens=existing.cache_read_tokens + entry.cache_read_tokens,
            cache_creation_tokens=existing.cache_creation_tokens + entry.cache_creation_tokens,
            output_tokens=existing.output_tokens + entry.output_tokens,
            cost_usd=existing.cost_usd + entry.cost_usd,
            seconds=existing.seconds + entry.seconds,
        )
    return list(merged.values())


def append_run(
    record: RunRecord,
    results_path: Path,
) -> None:
    """Append one finished run to the results file as a JSON line.

    One line per run means a crashed sweep keeps every run that already finished, and
    two arms can be appended to the same file and separated later by their arm field.

    Args:
        record: The finished run.
        results_path: File to append to; created with its parents if absent.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record) | {
        "total_tokens": record.total_tokens,
        "uncached_tokens": record.uncached_tokens,
        "total_cost_usd": record.total_cost_usd,
    }
    with results_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
