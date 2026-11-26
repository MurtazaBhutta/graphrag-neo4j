"""Against a real Neo4j holding the benchmark world. Skipped without one; CI fails on skips."""

import random

import pytest

from graphrag.cypher import CypherExecutor
from graphrag.plan import AssetType, GroupBy, Intent, Metric, Priority, Status, make_plan
from graphrag.questions import build_questions
from graphrag.reference import LinkError, ReferenceExecutor
from graphrag.retrievers import VectorRetriever

pytestmark = pytest.mark.neo4j


@pytest.fixture(scope="module")
def executors(loaded, embedder):
    driver, world = loaded
    return CypherExecutor(driver, embedder), ReferenceExecutor(world), world


def test_every_benchmark_question_matches_the_reference(executors):
    cypher, reference, world = executors
    for q in build_questions(world):
        if q.semantic:
            continue
        assert cypher.execute(q.plan).comparable() == q.gold.comparable(), q.id


def random_plan(rng: random.Random, world):
    names = {
        "customer": [c.name for c in world.customers.values()],
        "site": [s.name for s in world.sites.values()],
        "engineer": [e.name for e in world.engineers.values()],
        "part": [p.name for p in world.parts.values()],
    }
    filters = {}
    for field, choices in names.items():
        if rng.random() < 0.2:
            filters[field] = rng.choice(choices)
    for field, enum in (("asset_type", AssetType), ("priority", Priority), ("status", Status)):
        if rng.random() < 0.25:
            filters[field] = rng.choice(list(enum))
    if rng.random() < 0.5:
        start = f"202{rng.choice('45')}-{rng.randint(1, 12):02d}-01"
        filters["date_from"] = start
        filters["date_to"] = f"{start[:4]}-12-31"
    intent = rng.choice([Intent.AGGREGATE] * 5 + [Intent.LIST_JOBS, Intent.REPEAT_VISITS])
    if intent is Intent.AGGREGATE:
        group = rng.choice([None, *GroupBy])
        metrics = [Metric.COUNT, Metric.PART_QUANTITY] if group is GroupBy.PART else list(Metric)
        metric = rng.choice(metrics)
        if metric is Metric.PART_QUANTITY and group not in (None, GroupBy.PART):
            metric = Metric.COUNT
        return make_plan(intent, metric=metric, group_by=group, limit=rng.randint(1, 30), **filters)
    if intent is Intent.REPEAT_VISITS:
        return make_plan(intent, min_repeats=rng.randint(1, 4), limit=50, **filters)
    return make_plan(intent, limit=rng.randint(1, 100), **filters)


def test_random_plans_match_the_reference(executors):
    """300 random plans across every intent, metric, grouping and filter combination."""
    cypher, reference, world = executors
    rng = random.Random(2026)
    for i in range(300):
        plan = random_plan(rng, world)
        assert cypher.execute(plan).comparable() == reference.execute(plan).comparable(), (
            i,
            plan.model_dump(exclude_none=True),
        )


def test_semantic_search_respects_filters_and_finds_most_planted_jobs(executors):
    """Filters are exact (a hard guarantee). Similarity is not: the planted notes are worded
    to avoid shared keywords, and a small CPU embedding model does not find all of them in the
    top 10. The floor catches a broken ranking (wrong embedding property, wrong sign) without
    pretending recall is perfect. Measured values are in the README."""
    cypher, reference, world = executors
    for q in build_questions(world):
        if not q.semantic:
            continue
        result = cypher.execute(q.plan)
        candidates = {j.id for j in reference.matching(q.plan)}
        assert result.evidence <= candidates, q.id  # never leaks outside the filters
        recall = len(q.gold_evidence & result.evidence) / len(q.gold_evidence)
        assert recall >= 0.5, (q.id, recall)


def test_entity_linking_tolerates_a_typo(executors):
    cypher, reference, world = executors
    name = world.customers[world.focus_customer].name
    typo = name[:-2] + name[-1] + name[-2]  # swap the last two letters
    plan = make_plan(Intent.AGGREGATE, customer=typo)
    exact = make_plan(Intent.AGGREGATE, customer=name)
    assert cypher.execute(plan).value == reference.execute(exact).value


def test_entity_linking_asks_when_unsure(executors):
    cypher, _, world = executors
    prefix = world.customers[world.focus_customer].name.split()[0]
    shared = [c.name for c in world.customers.values() if c.name.startswith(prefix + " ")]
    if len(shared) < 2:
        pytest.skip("no two customers share a first word in this world")
    with pytest.raises(LinkError) as info:
        cypher.execute(make_plan(Intent.AGGREGATE, customer=prefix))
    assert set(info.value.candidates) <= set(shared)


def test_vector_baseline_returns_k_records(loaded, embedder):
    driver, _ = loaded
    retrieved = VectorRetriever(driver, embedder, k=7).retrieve("boiler lockout at a hotel")
    assert len(retrieved.job_ids) == 7
    assert retrieved.context.count("\nNotes: ") == 7
