"""The typed query plan: the only thing the LLM produces on the graph path.

The model never writes Cypher. It fills in this structure: an intent, filters, a metric and a
grouping, all drawn from closed sets. ``cypher.py`` turns that into a parameterised, read-only
query from fixed fragments. A question can only reach the database through these fields, so a
prompt-injected question cannot become ``DETACH DELETE``. See docs/adr/0001.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class Intent(StrEnum):
    JOB_DETAIL = "job_detail"  # one job by id
    LIST_JOBS = "list_jobs"  # jobs matching filters, newest first (or most similar, if semantic)
    AGGREGATE = "aggregate"  # a metric over matching jobs, optionally grouped
    REPEAT_VISITS = "repeat_visits"  # assets with at least N follow-up visits


class Metric(StrEnum):
    COUNT = "count"
    TOTAL_COST = "total_cost"
    AVG_COST = "avg_cost"
    SLA_RATE = "sla_rate"
    PART_QUANTITY = "part_quantity"


class GroupBy(StrEnum):
    CUSTOMER = "customer"
    SITE = "site"
    ENGINEER = "engineer"
    ASSET = "asset"
    ASSET_TYPE = "asset_type"
    FAULT_CATEGORY = "fault_category"
    PART = "part"
    MONTH = "month"


class AssetType(StrEnum):
    BOILER = "boiler"
    CHILLER = "chiller"
    AHU = "ahu"
    LIFT = "lift"
    FIRE_ALARM = "fire_alarm"
    ELECTRICAL_BOARD = "electrical_board"
    GENERATOR = "generator"


class Priority(StrEnum):
    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"


class Status(StrEnum):
    COMPLETED = "completed"
    OPEN = "open"
    CANCELLED = "cancelled"


class Filters(BaseModel):
    """Names are as the user wrote them; they are resolved to ids before querying."""

    job_id: str | None = Field(description="e.g. 'J-004211'")
    customer: str | None = Field(description="customer name")
    site: str | None = Field(description="site name")
    engineer: str | None = Field(description="engineer full name")
    part: str | None = Field(description="part name")
    asset_type: AssetType | None
    fault_category: str | None
    priority: Priority | None
    status: Status | None
    date_from: str | None = Field(description="inclusive, YYYY-MM-DD")
    date_to: str | None = Field(description="inclusive, YYYY-MM-DD")

    @field_validator("date_from", "date_to")
    @classmethod
    def _iso_date(cls, value: str | None) -> str | None:
        if value is not None:
            date.fromisoformat(value)
        return value


class QueryPlan(BaseModel):
    intent: Intent
    filters: Filters
    metric: Metric = Field(description="for aggregate; use count otherwise")
    group_by: GroupBy | None
    min_repeats: int = Field(description="for repeat_visits; 1 otherwise")
    semantic_query: str | None = Field(
        description="for list_jobs only: free-text description of what the notes should say"
    )
    limit: int = Field(description="rows to return, 1-100")

    @field_validator("limit")
    @classmethod
    def _limit(cls, value: int) -> int:
        if not 1 <= value <= 100:
            raise ValueError("limit must be between 1 and 100")
        return value

    @model_validator(mode="after")
    def _coherent(self) -> QueryPlan:
        part_rows = self.group_by is GroupBy.PART or self.metric is Metric.PART_QUANTITY
        if part_rows and self.metric not in (Metric.COUNT, Metric.PART_QUANTITY):
            raise ValueError(f"{self.metric} cannot be combined with part grouping")
        if self.intent is Intent.JOB_DETAIL and not self.filters.job_id:
            raise ValueError("job_detail needs filters.job_id")
        if self.semantic_query and self.intent is not Intent.LIST_JOBS:
            raise ValueError("semantic_query is only valid with list_jobs")
        return self


def make_plan(
    intent: Intent,
    *,
    metric: Metric = Metric.COUNT,
    group_by: GroupBy | None = None,
    min_repeats: int = 1,
    semantic_query: str | None = None,
    limit: int = 20,
    **filters,
) -> QueryPlan:
    """Build a plan in code without spelling out every null (the model must spell them out)."""
    every = dict.fromkeys(Filters.model_fields)
    return QueryPlan(
        intent=intent,
        filters=Filters(**{**every, **filters}),
        metric=metric,
        group_by=group_by,
        min_repeats=min_repeats,
        semantic_query=semantic_query,
        limit=limit,
    )


def plan_from_partial(data: dict) -> QueryPlan:
    """Accept a plan with omitted fields (the HTTP API), filling the same defaults as make_plan."""
    data = dict(data)
    intent = Intent(data.pop("intent"))
    filters = data.pop("filters", None) or {}
    return make_plan(intent, **data, **filters)
