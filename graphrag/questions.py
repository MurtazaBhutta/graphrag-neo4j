"""The benchmark questions, built from the generated world.

Entities are chosen by rule (the busiest customer, the median engineer, ...), so the same world
always gives the same questions. Each question carries:

* the natural-language text a user would type,
* the **gold plan**, the plan a correct planner would produce,
* the **gold evidence**, every job the answer depends on, and
* the **gold answer**, computed by the plain-Python reference executor (structured questions),
  or the planted jobs (semantic questions; similarity has no reference implementation).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .datagen import World
from .plan import (
    AssetType,
    GroupBy,
    Intent,
    Metric,
    Priority,
    QueryPlan,
    Status,
    make_plan,
)
from .reference import ReferenceExecutor
from .result import Result

Y2025 = {"date_from": "2025-01-01", "date_to": "2025-12-31"}
Y2024 = {"date_from": "2024-01-01", "date_to": "2024-12-31"}


@dataclass
class Question:
    id: str
    category: str
    text: str
    plan: QueryPlan
    gold_evidence: set[str]
    gold: Result | None  # None for semantic questions

    @property
    def semantic(self) -> bool:
        return self.gold is None


def build_questions(world: World) -> list[Question]:
    ref = ReferenceExecutor(world)
    jobs = list(world.jobs.values())
    by_customer = Counter(world.sites[j.site_id].customer_id for j in jobs)
    by_site = Counter(j.site_id for j in jobs)
    by_engineer = Counter(j.engineer_id for j in jobs)

    focus = world.customers[world.focus_customer].name
    # A mid-sized customer that has lift faults on record (one question is about them).
    lift_customers = {
        world.sites[j.site_id].customer_id
        for j in jobs
        if world.assets[j.asset_id].type == "lift" and j.reported_on >= "2025"
    }
    ranked_customers = sorted(
        (c for c in by_customer if c in lift_customers), key=lambda c: (-by_customer[c], c)
    )
    mid_customer_id = ranked_customers[len(ranked_customers) // 2]
    mid_customer = world.customers[mid_customer_id].name
    mid_sites = sorted(
        (s for s in world.sites.values() if s.customer_id == mid_customer_id),
        key=lambda s: (-by_site[s.id], s.id),
    )
    busy_site = mid_sites[0].name
    engineer = world.engineers[
        sorted(by_engineer, key=lambda e: (by_engineer[e], e))[len(by_engineer) // 2]
    ].name
    with_parts = [j for j in jobs if len(j.parts) >= 2]
    lookups = [
        jobs[len(jobs) // 7].id,
        jobs[len(jobs) // 3].id,
        with_parts[len(with_parts) // 2].id,
    ]

    agg, lst, rep = Intent.AGGREGATE, Intent.LIST_JOBS, Intent.REPEAT_VISITS
    specs = [
        (
            "lookup",
            f"What work was done on job {lookups[0]}, and who attended?",
            make_plan(Intent.JOB_DETAIL, job_id=lookups[0], limit=1),
        ),
        (
            "lookup",
            f"Show me the details of job {lookups[1]}.",
            make_plan(Intent.JOB_DETAIL, job_id=lookups[1], limit=1),
        ),
        (
            "lookup",
            f"Which parts were fitted on job {lookups[2]}?",
            make_plan(Intent.JOB_DETAIL, job_id=lookups[2], limit=1),
        ),
        (
            "relationship",
            f"Which engineers have attended jobs at {busy_site}?",
            make_plan(agg, group_by=GroupBy.ENGINEER, site=busy_site, limit=50),
        ),
        (
            "relationship",
            f"List the open jobs for {mid_customer}.",
            make_plan(lst, customer=mid_customer, status=Status.OPEN, limit=100),
        ),
        (
            "relationship",
            f"Which customers has {engineer} done work for?",
            make_plan(agg, group_by=GroupBy.CUSTOMER, engineer=engineer, limit=50),
        ),
        (
            "relationship",
            f"What parts were used at {busy_site} in 2025?",
            make_plan(
                agg,
                metric=Metric.PART_QUANTITY,
                group_by=GroupBy.PART,
                site=busy_site,
                limit=50,
                **Y2025,
            ),
        ),
        (
            "relationship",
            f"Which of {mid_customer}'s sites had lift faults in 2025?",
            make_plan(
                agg,
                group_by=GroupBy.SITE,
                customer=mid_customer,
                asset_type=AssetType.LIFT,
                limit=50,
                **Y2025,
            ),
        ),
        (
            "aggregation",
            f"How many jobs did {focus} raise in the first quarter of 2025?",
            make_plan(agg, customer=focus, date_from="2025-01-01", date_to="2025-03-31"),
        ),
        (
            "aggregation",
            "How many emergency call-outs were there across all customers in 2025?",
            make_plan(agg, priority=Priority.EMERGENCY, **Y2025),
        ),
        (
            "aggregation",
            "Which five sites had the most jobs in 2025?",
            make_plan(agg, group_by=GroupBy.SITE, limit=5, **Y2025),
        ),
        (
            "aggregation",
            "Who were the five busiest engineers in 2025?",
            make_plan(agg, group_by=GroupBy.ENGINEER, limit=5, **Y2025),
        ),
        (
            "aggregation",
            f"What did {focus} spend on reactive maintenance in 2025?",
            make_plan(agg, metric=Metric.TOTAL_COST, customer=focus, **Y2025),
        ),
        (
            "aggregation",
            "What is the average cost of a lift job?",
            make_plan(agg, metric=Metric.AVG_COST, asset_type=AssetType.LIFT),
        ),
        (
            "aggregation",
            f"What share of {mid_customer}'s jobs met their SLA in 2025?",
            make_plan(agg, metric=Metric.SLA_RATE, customer=mid_customer, **Y2025),
        ),
        (
            "aggregation",
            "Break down boiler faults by category for 2025.",
            make_plan(agg, group_by=GroupBy.FAULT_CATEGORY, asset_type=AssetType.BOILER, **Y2025),
        ),
        (
            "aggregation",
            f"How many jobs did {focus} log each month in 2025?",
            make_plan(agg, group_by=GroupBy.MONTH, customer=focus, limit=12, **Y2025),
        ),
        (
            "aggregation",
            "Which part was used most in 2025?",
            make_plan(agg, metric=Metric.PART_QUANTITY, group_by=GroupBy.PART, limit=1, **Y2025),
        ),
        (
            "repeat_failures",
            "Which assets have needed three or more return visits?",
            make_plan(rep, min_repeats=3, limit=20),
        ),
        (
            "repeat_failures",
            f"Which {focus} assets had a repeat visit in 2025?",
            make_plan(rep, min_repeats=1, customer=focus, limit=100, **Y2025),
        ),
        (
            "repeat_failures",
            "Which boilers have needed two or more follow-up visits?",
            make_plan(rep, min_repeats=2, asset_type=AssetType.BOILER, limit=20),
        ),
        (
            "semantic",
            "Find jobs where people noticed a burning smell from electrical equipment.",
            make_plan(lst, semantic_query="burning smell from electrical equipment", limit=10),
        ),
        (
            "semantic",
            "Find jobs where the lift doors kept re-opening and would not close.",
            make_plan(
                lst, semantic_query="lift doors keep re-opening and will not close", limit=10
            ),
        ),
        (
            "semantic_filtered",
            f"Which {focus} jobs in 2025 involved water coming through a ceiling?",
            make_plan(
                lst,
                semantic_query="water coming through the ceiling",
                customer=focus,
                limit=10,
                **Y2025,
            ),
        ),
        (
            "semantic_filtered",
            "Were any burning smells reported at electrical boards in 2024?",
            make_plan(
                lst,
                semantic_query="burning smell at the electrical board",
                asset_type=AssetType.ELECTRICAL_BOARD,
                limit=10,
                **Y2024,
            ),
        ),
    ]

    themes = {
        "burning smell": "burning_smell",
        "lift doors": "lift_doors",
        "water coming": "ceiling_leak",
    }
    questions = []
    for i, (category, text, plan) in enumerate(specs, start=1):
        if plan.semantic_query:
            theme = next(t for key, t in themes.items() if plan.semantic_query.startswith(key))
            candidates = {j.id for j in ref.matching(plan)}
            gold_evidence = set(world.themes[theme]) & candidates
            questions.append(Question(f"q{i:02d}", category, text, plan, gold_evidence, None))
        else:
            gold = ref.execute(plan)
            questions.append(Question(f"q{i:02d}", category, text, plan, gold.evidence, gold))
    return questions
