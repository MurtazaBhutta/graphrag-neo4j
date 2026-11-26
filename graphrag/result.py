"""The shape every executor returns, plus the post-processing both executors share.

Cypher and the Python reference both return *raw* aggregates (sums and counts). Derived values
(averages, rates), ordering and tie-breaks are computed once, here, so a difference between the
two executors can only come from which rows they matched, which is what the tests check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .plan import Metric, QueryPlan


@dataclass
class Result:
    rows: list[dict]
    value: float | int | None = None  # the single answer for ungrouped aggregates
    evidence: set[str] = field(default_factory=set)  # every job id the answer depends on
    cypher: str = ""

    def comparable(self) -> tuple:
        return (self.value, self.rows, sorted(self.evidence))


def finish_aggregate(plan: QueryPlan, groups: dict, evidence: set[str]) -> Result:
    """``groups`` maps a group key (None when ungrouped) to raw (numerator, denominator)."""
    values = {key: derive(plan.metric, num, den) for key, (num, den) in groups.items()}
    if plan.group_by is None:
        return Result(rows=[], value=values.get(None, derive(plan.metric, 0, 0)), evidence=evidence)
    ordered = sorted(values.items(), key=lambda kv: (-kv[1], str(kv[0])))[: plan.limit]
    return Result(rows=[{"key": k, "value": v} for k, v in ordered], evidence=evidence)


def derive(metric: Metric, numerator: int, denominator: int) -> float | int:
    if metric in (Metric.COUNT, Metric.PART_QUANTITY):
        return int(numerator)
    if metric is Metric.TOTAL_COST:
        return round(numerator / 100, 2)  # pence -> pounds
    if denominator == 0:
        return 0.0
    if metric is Metric.AVG_COST:
        return round(numerator / denominator / 100, 2)
    return round(numerator / denominator, 4)  # SLA_RATE


def finish_list(rows: list[dict], limit: int) -> Result:
    rows = rows[:limit]
    return Result(rows=rows, evidence={r["job_id"] for r in rows})


def finish_repeats(rows: list[dict], limit: int) -> Result:
    rows = sorted(rows, key=lambda r: (-r["repeats"], r["asset"]))[:limit]
    evidence = {job for r in rows for job in r.pop("jobs")}
    return Result(rows=rows, evidence=evidence)
