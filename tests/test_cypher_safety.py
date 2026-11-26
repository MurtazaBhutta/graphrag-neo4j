"""User-supplied text must reach Neo4j only as parameters, never as Cypher text."""

import pytest

from graphrag.cypher import CypherExecutor
from graphrag.plan import GroupBy, Intent, Metric, make_plan

NASTY = "x' }) DETACH DELETE n //"


def compile_with(plan):
    params = {
        "limit": plan.limit,
        "min_repeats": plan.min_repeats,
        "part_sku": None,
        "customer_id": "C001" if plan.filters.customer else None,
        "site_id": "S0001" if plan.filters.site else None,
        "engineer_id": "E001" if plan.filters.engineer else None,
    }
    cypher, _ = CypherExecutor(driver=None).compile(plan, params)
    return cypher, params


@pytest.mark.parametrize(
    "plan",
    [
        make_plan(Intent.AGGREGATE, fault_category=NASTY, customer=NASTY, group_by=GroupBy.SITE),
        make_plan(Intent.LIST_JOBS, semantic_query=NASTY, site=NASTY, date_from="2025-01-01"),
        make_plan(Intent.JOB_DETAIL, job_id=NASTY),
        make_plan(Intent.REPEAT_VISITS, engineer=NASTY, fault_category=NASTY),
        make_plan(
            Intent.AGGREGATE,
            metric=Metric.PART_QUANTITY,
            group_by=GroupBy.PART,
            fault_category=NASTY,
        ),
    ],
)
def test_values_never_appear_in_the_query_text(plan):
    cypher, _ = compile_with(plan)
    assert NASTY not in cypher
    assert "DETACH" not in cypher and "DELETE" not in cypher


def test_every_group_and_metric_compiles():
    for group in [None, *GroupBy]:
        for metric in Metric:
            try:
                plan = make_plan(Intent.AGGREGATE, group_by=group, metric=metric)
            except ValueError:
                continue  # an invalid combination, rejected by the plan itself
            cypher, _ = compile_with(plan)
            assert cypher.startswith("MATCH (j:Job)")
