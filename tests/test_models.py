"""Tests for the record shapes.

These pin the validators, since the validators are the only behaviour in models.py and
they exist to stop a malformed record reaching the results file where it would be hard
to notice.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentpair.models import Acceptance, Dispute, Finding, Resolution, RoleUsage, RunRecord


def make_dispute() -> Dispute:
    """Build a minimal dispute for reuse across the resolution tests.

    Returns:
        A dispute over one finding.
    """
    return Dispute(
        finding=Finding(id="f1", claim="parse_duration('1h30m') is wrong", evidence="duration.py:12"),
        implementer_position="the spec adds 15 seconds, so 5415 is correct",
    )


def test_finding_rejects_blank_id() -> None:
    with pytest.raises(ValueError, match="id must not be empty"):
        Finding(id="  ", claim="a real claim")


def test_finding_rejects_blank_claim() -> None:
    with pytest.raises(ValueError, match="claim must not be empty"):
        Finding(id="f1", claim="")


def test_finding_evidence_defaults_to_empty() -> None:
    assert Finding(id="f1", claim="a real claim").evidence == ""


def test_dispute_rejects_blank_position() -> None:
    with pytest.raises(ValueError, match="implementer_position must not be empty"):
        Dispute(finding=Finding(id="f1", claim="a real claim"), implementer_position="")


def test_probed_resolution_requires_a_probe() -> None:
    with pytest.raises(ValueError, match="must retain its probe"):
        Resolution(dispute=make_dispute(), favours="reviewer", method="judged_with_probe", rationale="it printed 5400")


def test_judged_resolution_rejects_a_probe() -> None:
    with pytest.raises(ValueError, match="has no probe"):
        Resolution(
            dispute=make_dispute(),
            favours="implementer",
            method="judged",
            rationale="naming preference only",
            probe_path="runs/probe_f1.py",
        )


def test_role_usage_rejects_negative_tokens() -> None:
    with pytest.raises(ValueError, match="token counts must not be negative"):
        RoleUsage(role="reviewer", input_tokens=-1, output_tokens=10, cost_usd=0.01, seconds=1.0)


def test_role_usage_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="cost and duration must not be negative"):
        RoleUsage(role="reviewer", input_tokens=10, output_tokens=10, cost_usd=-0.01, seconds=1.0)


def test_run_record_totals_sum_across_roles() -> None:
    record = RunRecord(
        task_id="duration",
        arm="pair",
        implementer_model="opus",
        reviewer_model="sonnet",
        usage=[
            RoleUsage(role="implementer", input_tokens=1000, output_tokens=200, cost_usd=0.05, seconds=12.0),
            RoleUsage(role="reviewer", input_tokens=400, output_tokens=100, cost_usd=0.02, seconds=6.0),
        ],
    )
    assert record.total_tokens == 1700
    assert record.total_cost_usd == pytest.approx(0.07)


def test_run_record_separates_total_from_uncached_tokens() -> None:
    record = RunRecord(
        task_id="duration",
        arm="pair",
        implementer_model="opus",
        usage=[
            RoleUsage(
                role="implementer",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.01,
                seconds=1.0,
                cache_read_tokens=900,
                cache_creation_tokens=200,
            )
        ],
    )
    assert record.total_tokens == 1250
    assert record.uncached_tokens == 150


def test_run_record_totals_are_zero_when_nothing_ran() -> None:
    record = RunRecord(task_id="duration", arm="baseline", implementer_model="opus")
    assert record.total_tokens == 0
    assert record.total_cost_usd == 0.0
    assert record.reviewer_model is None


def test_an_acceptance_keeps_the_finding_and_the_reason() -> None:
    accepted = Acceptance(finding=Finding(id="f1", claim="drops seconds"), reason="applied at duration.py:12")
    assert accepted.finding.id == "f1"
    assert accepted.reason == "applied at duration.py:12"


def test_an_acceptance_with_no_reason_is_allowed() -> None:
    # The decision format makes the reason optional, and a bare acceptance is still an
    # acceptance. Refusing it here would put the modal outcome back to being inferred.
    assert Acceptance(finding=Finding(id="f1", claim="drops seconds")).reason == ""


def test_a_fresh_record_has_no_acceptances() -> None:
    assert RunRecord(task_id="duration", arm="pair", implementer_model="opus").accepted == []


def test_a_record_timestamps_itself_in_utc() -> None:
    before = datetime.now(UTC)
    record = RunRecord(task_id="duration", arm="pair", implementer_model="opus")
    started = datetime.fromisoformat(record.started_at)
    assert started.utcoffset().total_seconds() == 0
    assert before <= started <= datetime.now(UTC)


def test_two_records_of_the_same_run_are_told_apart_by_their_timestamps() -> None:
    # Two runs of one task and arm produced identical rows, so a re-run could not be told
    # from the run it replaced.
    first = RunRecord(task_id="duration", arm="pair", implementer_model="opus")
    second = RunRecord(task_id="duration", arm="pair", implementer_model="opus")
    assert first.started_at <= second.started_at
