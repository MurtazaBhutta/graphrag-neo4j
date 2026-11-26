"""The two retrieval strategies under test, behind one interface.

Both run against the same Neo4j database and the same embeddings. The vector baseline sees every
field of every job (``render_record``) and only lacks structure. See docs/adr/0002.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from neo4j import Driver

from .cypher import CypherExecutor
from .plan import QueryPlan
from .result import Result


@dataclass
class Retrieved:
    context: str  # what an answering model would be given
    job_ids: set[str]  # the jobs whose information is in the context
    latency_ms: float
    result: Result | None = None  # graph only: the structured answer
    cypher: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def approx_tokens(self) -> int:
        return len(self.context) // 4  # rough rule of thumb; exact counts need a tokenizer call


class VectorRetriever:
    """Plain vector RAG: embed the question, take the top-k most similar job records."""

    def __init__(self, driver: Driver, embedder, k: int = 10, index: str = "job_embedding"):
        """``index``: job_embedding (whole records, the default) or notes_embedding."""
        self.driver, self.embedder, self.k, self.index = driver, embedder, k, index

    def retrieve(self, question: str) -> Retrieved:
        started = time.perf_counter()
        vector = self.embedder.embed_query(question).tolist()
        with self.driver.session() as session:
            rows = session.execute_read(
                lambda tx: tx.run(
                    """CALL db.index.vector.queryNodes($index, $k, $vector)
                   YIELD node AS j, score
                   MATCH (j)-[:AT_SITE]->(s:Site)-[:BELONGS_TO]->(c:Customer)
                   MATCH (j)-[:ON_ASSET]->(a:Asset), (j)-[:ATTENDED_BY]->(e:Engineer)
                   OPTIONAL MATCH (j)-[u:USED_PART]->(p:Part)
                   WITH j, s, c, a, e, score,
                        collect(CASE WHEN p IS NULL THEN null ELSE p.name + ' x' + u.qty END)
                        AS parts
                   RETURN j.id AS id, j.reported_on AS reported_on, c.name AS customer,
                          s.name AS site, a.type AS asset_type, e.name AS engineer,
                          j.fault_category AS fault, j.priority AS priority,
                          j.status AS status, j.sla_met AS sla_met, j.cost_pence AS cost,
                          j.notes AS notes, parts, score
                   ORDER BY score DESC""",
                    k=self.k,
                    index=self.index,
                    vector=vector,
                ).data()
            )
        latency = (time.perf_counter() - started) * 1000
        context = "\n\n".join(
            f"Job {r['id']} reported {r['reported_on']} | {r['customer']} | {r['site']} | "
            f"{r['asset_type']} | engineer {r['engineer']} | {r['fault']} | {r['priority']} | "
            f"{r['status']} | SLA met {'yes' if r['sla_met'] else 'no'} | "
            f"£{r['cost'] / 100:.2f} | parts: {', '.join(r['parts']) or 'none'}\n"
            f"Notes: {r['notes']}"
            for r in rows
        )
        return Retrieved(context, {r["id"] for r in rows}, latency)


class GraphRetriever:
    """Graph RAG: a typed plan, compiled to Cypher, returning exact structured results."""

    def __init__(self, executor: CypherExecutor):
        self.executor = executor

    def retrieve(self, plan: QueryPlan) -> Retrieved:
        started = time.perf_counter()
        result = self.executor.execute(plan)
        latency = (time.perf_counter() - started) * 1000
        return Retrieved(
            render_result(plan, result), result.evidence, latency, result, result.cypher
        )


def render_result(plan: QueryPlan, result: Result, sample: int = 20) -> str:
    """Structured result as compact text for the answering model."""
    lines = [
        f"Query: {plan.intent}"
        + (f", {plan.metric}" if plan.intent == "aggregate" else "")
        + (f" by {plan.group_by}" if plan.group_by else "")
    ]
    if result.value is not None:
        lines.append(f"Answer: {result.value}")
    for row in result.rows:
        lines.append(" | ".join(f"{k}: {v}" for k, v in row.items() if k != "score"))
    if not result.rows and result.value is None:
        lines.append("No matching records.")
    evidence = sorted(result.evidence)
    lines.append(
        f"Based on {len(evidence)} job(s)"
        + (f", e.g. {', '.join(evidence[:sample])}" if evidence else "")
    )
    return "\n".join(lines)
