"""Tests for usage extraction, per-role merging and the results file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from agentpair.metrics import append_run, merge_usage, usage_from_result
from agentpair.models import Acceptance, Finding, RoleUsage, RunRecord


@dataclass
class FakeResult:
    duration_ms: int
    total_cost_usd: float | None
    usage: dict[str, Any] | None


def test_usage_keeps_cached_and_uncached_input_apart() -> None:
    result = FakeResult(
        duration_ms=2500,
        total_cost_usd=0.042,
        usage={
            "input_tokens": 100,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 50,
            "output_tokens": 300,
        },
    )
    entry = usage_from_result("implementer", result)
    assert entry.input_tokens == 100
    assert entry.cache_read_tokens == 900
    assert entry.cache_creation_tokens == 50
    assert entry.output_tokens == 300
    assert entry.cost_usd == pytest.approx(0.042)
    assert entry.seconds == pytest.approx(2.5)


def test_usage_treats_missing_fields_as_zero() -> None:
    entry = usage_from_result("reviewer", FakeResult(duration_ms=0, total_cost_usd=None, usage=None))
    assert entry.input_tokens == 0
    assert entry.output_tokens == 0
    assert entry.cost_usd == 0.0


def test_merge_sums_repeated_turns_by_role() -> None:
    entries = [
        RoleUsage(role="implementer", input_tokens=100, output_tokens=10, cost_usd=0.01, seconds=1.0),
        RoleUsage(role="reviewer", input_tokens=200, output_tokens=20, cost_usd=0.02, seconds=2.0),
        RoleUsage(role="implementer", input_tokens=300, output_tokens=30, cost_usd=0.03, seconds=3.0),
    ]
    merged = merge_usage(entries)
    assert [entry.role for entry in merged] == ["implementer", "reviewer"]
    assert merged[0].input_tokens == 400
    assert merged[0].output_tokens == 40
    assert merged[0].cost_usd == pytest.approx(0.04)
    assert merged[0].seconds == pytest.approx(4.0)


def test_merge_of_nothing_is_empty() -> None:
    assert merge_usage([]) == []


def test_append_run_writes_one_json_line_per_run(tmp_path: Path) -> None:
    results_path = tmp_path / "runs" / "results.jsonl"
    record = RunRecord(
        task_id="duration",
        arm="pair",
        implementer_model="opus",
        reviewer_model="sonnet",
        usage=[RoleUsage(role="implementer", input_tokens=100, output_tokens=10, cost_usd=0.01, seconds=1.0)],
    )
    append_run(record, results_path)
    append_run(record, results_path)

    lines = results_path.read_text().strip().split("\n")
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["task_id"] == "duration"
    assert payload["arm"] == "pair"
    assert payload["total_tokens"] == 110
    assert payload["total_cost_usd"] == pytest.approx(0.01)


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


def test_the_written_payload_carries_the_acceptances_and_the_timestamp(tmp_path: Path) -> None:
    # Both were absent from the file: acceptance had to be reconstructed by subtraction,
    # and two runs of one task and arm wrote identical rows.
    results_path = tmp_path / "results.jsonl"
    record = RunRecord(
        task_id="duration",
        arm="pair",
        implementer_model="opus",
        reviewer_model="sonnet",
        findings=[Finding(id="f1", claim="drops seconds", evidence="duration.py:12")],
        accepted=[Acceptance(finding=Finding(id="f1", claim="drops seconds"), reason="applied")],
    )
    append_run(record, results_path)

    payload = json.loads(results_path.read_text().strip())
    assert payload["accepted"] == [
        {"finding": {"id": "f1", "claim": "drops seconds", "evidence": ""}, "reason": "applied"}
    ]
    assert payload["started_at"] == record.started_at
    assert datetime.fromisoformat(payload["started_at"]).utcoffset().total_seconds() == 0
