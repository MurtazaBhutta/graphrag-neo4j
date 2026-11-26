"""Compile a ``QueryPlan`` into one parameterised, read-only Cypher query, and run it.

Every piece of Cypher text comes from the fixed fragments in this module, keyed by enum values.
Every user-supplied value (names, dates, free text) travels as a query parameter. Queries run in
read transactions. See docs/adr/0001.
"""

from __future__ import annotations

from neo4j import Driver

from .plan import GroupBy, Intent, Metric, QueryPlan
from .reference import LinkError
from .result import Result, finish_aggregate, finish_list, finish_repeats

MATCH_JOB = """MATCH (j:Job)-[:AT_SITE]->(s:Site)-[:BELONGS_TO]->(c:Customer)
MATCH (j)-[:ON_ASSET]->(a:Asset)
MATCH (j)-[:ATTENDED_BY]->(e:Engineer)"""

GROUP_KEY = {
    GroupBy.CUSTOMER: "c.name",
    GroupBy.SITE: "s.name",
    GroupBy.ENGINEER: "e.name",
    GroupBy.ASSET: "a.id",
    GroupBy.ASSET_TYPE: "a.type",
    GroupBy.FAULT_CATEGORY: "j.fault_category",
    GroupBy.MONTH: "substring(j.reported_on, 0, 7)",
    GroupBy.PART: "p.name",
}

# metric -> (numerator, denominator); derived values are computed in result.py
JOB_METRIC = {
    Metric.COUNT: ("count(j)", "0"),
    Metric.TOTAL_COST: ("sum(j.cost_pence)", "count(j)"),
    Metric.AVG_COST: ("sum(j.cost_pence)", "count(j)"),
    Metric.SLA_RATE: ("sum(CASE WHEN j.sla_met THEN 1 ELSE 0 END)", "count(j)"),
}
PART_METRIC = {
    Metric.COUNT: ("count(DISTINCT j)", "0"),
    Metric.PART_QUANTITY: ("sum(u.qty)", "0"),
}

LIST_RETURN = """RETURN j.id AS job_id, j.reported_on AS reported_on, c.name AS customer,
       s.name AS site, a.type AS asset_type, e.name AS engineer,
       j.fault_category AS fault_category, j.priority AS priority, j.status AS status"""


class CypherExecutor:
    def __init__(self, driver: Driver, embedder=None):
        self.driver = driver
        self.embedder = embedder

    def execute(self, plan: QueryPlan) -> Result:
        params = self.link(plan)
        cypher, finish = self.compile(plan, params)
        if plan.semantic_query:
            params["query_vector"] = self.embedder.embed_query(plan.semantic_query).tolist()
        with self.driver.session() as session:
            rows = session.execute_read(lambda tx: [r.data() for r in tx.run(cypher, params)])
        result = finish(rows)
        result.cypher = cypher
        return result

    # -- compile -------------------------------------------------------------------------------

    def compile(self, plan: QueryPlan, params: dict):
        where = self._where(plan, params)
        if plan.intent is Intent.JOB_DETAIL:
            return self._detail(where), self._finish_detail
        if plan.intent is Intent.LIST_JOBS:
            return self._list(plan, where), lambda rows: finish_list(rows, plan.limit)
        if plan.intent is Intent.REPEAT_VISITS:
            return self._repeats(plan, where), lambda rows: finish_repeats(rows, plan.limit)
        return self._aggregate(plan, where), lambda rows: self._finish_aggregate(plan, rows)

    def _where(self, plan: QueryPlan, params: dict) -> str:
        f = plan.filters
        clauses = []
        simple = {
            "job_id": "j.id = $job_id",
            "customer_id": "c.id = $customer_id",
            "site_id": "s.id = $site_id",
            "engineer_id": "e.id = $engineer_id",
            "asset_type": "a.type = $asset_type",
            "fault_category": "j.fault_category = $fault_category",
            "priority": "j.priority = $priority",
            "status": "j.status = $status",
            "date_from": "j.reported_on >= $date_from",
            "date_to": "j.reported_on <= $date_to",
        }
        for name in (
            "job_id",
            "asset_type",
            "fault_category",
            "priority",
            "status",
            "date_from",
            "date_to",
        ):
            value = getattr(f, name)
            if value is not None:
                params[name] = str(value)
        for name, clause in simple.items():
            if params.get(name) is not None:
                clauses.append(clause)
        if params.get("part_sku") and not self._part_rows(plan):
            clauses.append("EXISTS { (j)-[:USED_PART]->(:Part {sku: $part_sku}) }")
        return "WHERE " + "\n  AND ".join(clauses) if clauses else ""

    @staticmethod
    def _part_rows(plan: QueryPlan) -> bool:
        return plan.group_by is GroupBy.PART or plan.metric is Metric.PART_QUANTITY

    def _aggregate(self, plan: QueryPlan, where: str) -> str:
        key = GROUP_KEY[plan.group_by] if plan.group_by else "null"
        if self._part_rows(plan):
            num, den = PART_METRIC[plan.metric]
            part_match = "MATCH (j)-[u:USED_PART]->(p:Part)"
            part_where = "WHERE $part_sku IS NULL OR p.sku = $part_sku"
            return (
                f"{MATCH_JOB}\n{where}\n{part_match}\n{part_where}\n"
                f"RETURN {key} AS key, {num} AS num, {den} AS den, "
                f"collect(DISTINCT j.id) AS evidence"
            )
        num, den = JOB_METRIC[plan.metric]
        return (
            f"{MATCH_JOB}\n{where}\n"
            f"RETURN {key} AS key, {num} AS num, {den} AS den, collect(j.id) AS evidence"
        )

    @staticmethod
    def _list(plan: QueryPlan, where: str) -> str:
        if plan.semantic_query:
            # Graph first, then rank: filters pick the candidates exactly, and similarity is
            # computed only over them. See docs/adr/0003.
            return (
                f"{MATCH_JOB}\n{where}\n"
                "WITH j, s, c, a, e, vector.similarity.cosine(j.notes_embedding, $query_vector) "
                "AS score\nORDER BY score DESC, j.id\nLIMIT $limit\n"
                f"{LIST_RETURN}, score"
            )
        return (
            f"{MATCH_JOB}\n{where}\n{LIST_RETURN}\n"
            "ORDER BY reported_on DESC, job_id DESC\nLIMIT $limit"
        )

    @staticmethod
    def _repeats(plan: QueryPlan, where: str) -> str:
        return (
            f"{MATCH_JOB}\n{where}\n"
            "MATCH (j)-[:FOLLOW_UP_OF]->(:Job)\n"
            "WITH a, count(j) AS repeats, collect(j.id) AS jobs\n"
            "WHERE repeats >= $min_repeats\n"
            "MATCH (a)-[:INSTALLED_AT]->(site:Site)\n"
            "RETURN a.id AS asset, a.type AS asset_type, site.name AS site, repeats, jobs"
        )

    @staticmethod
    def _detail(where: str) -> str:
        return (
            f"{MATCH_JOB}\n{where}\n"
            "OPTIONAL MATCH (j)-[u:USED_PART]->(p:Part)\n"
            "OPTIONAL MATCH (j)-[:FOLLOW_UP_OF]->(prev:Job)\n"
            "WITH j, s, c, a, e, prev, "
            "collect(CASE WHEN p IS NULL THEN null\n"
            "             ELSE {part: p.name, qty: u.qty, sku: p.sku} END) AS parts\n"
            f"{LIST_RETURN}, s.city AS site_city, a.id AS asset, j.cost_pence AS cost_pence, "
            "j.sla_met AS sla_met, j.notes AS notes, parts, prev.id AS follow_up_of"
        )

    # -- finish --------------------------------------------------------------------------------

    @staticmethod
    def _finish_detail(rows: list[dict]) -> Result:
        if not rows:
            return Result(rows=[])
        row = rows[0]
        row["cost"] = round(row.pop("cost_pence") / 100, 2)
        parts = sorted(row.pop("parts"), key=lambda p: p["sku"])
        row["parts"] = [{"part": p["part"], "qty": p["qty"]} for p in parts]
        order = [
            "job_id",
            "reported_on",
            "customer",
            "site",
            "asset_type",
            "engineer",
            "fault_category",
            "priority",
            "status",
            "site_city",
            "asset",
            "cost",
            "sla_met",
            "notes",
            "parts",
            "follow_up_of",
        ]
        return Result(rows=[{k: row[k] for k in order}], evidence={row["job_id"]})

    @staticmethod
    def _finish_aggregate(plan: QueryPlan, rows: list[dict]) -> Result:
        groups = {r["key"]: (r["num"] or 0, r["den"] or 0) for r in rows if r["num"] or r["den"]}
        evidence = {job for r in rows for job in r["evidence"]}
        return finish_aggregate(plan, groups, evidence)

    # -- entity linking ------------------------------------------------------------------------

    def link(self, plan: QueryPlan) -> dict:
        """Resolve names to ids: exact (case-insensitive) first, then full-text, accepting a
        full-text hit only when it clearly beats the runner-up. Otherwise ask, don't guess."""
        params: dict = {
            "limit": plan.limit,
            "min_repeats": plan.min_repeats,
            "part_sku": None,
            "customer_id": None,
            "site_id": None,
            "engineer_id": None,
        }
        labels = {
            "customer": ("Customer", "id", "customer_id"),
            "site": ("Site", "id", "site_id"),
            "engineer": ("Engineer", "id", "engineer_id"),
            "part": ("Part", "sku", "part_sku"),
        }
        with self.driver.session() as session:
            for field, (label, id_prop, param) in labels.items():
                name = getattr(plan.filters, field)
                if name is None:
                    continue
                params[param] = session.execute_read(self._resolve, label, id_prop, field, name)
        return params

    @staticmethod
    def _resolve(tx, label: str, id_prop: str, field: str, name: str) -> str:
        exact = tx.run(
            f"MATCH (n:{label}) WHERE toLower(n.name) = toLower($name) RETURN n.{id_prop} AS id",
            name=name.strip(),
        ).values()
        if len(exact) == 1:
            return exact[0][0]
        hits = tx.run(
            "CALL db.index.fulltext.queryNodes('entity_names', $q) YIELD node, score "
            f"WHERE node:{label} RETURN node.{id_prop} AS id, node.name AS name, score LIMIT 3",
            q=_fulltext_query(name),
        ).data()
        if len(hits) == 1 or (len(hits) > 1 and hits[0]["score"] >= 1.5 * hits[1]["score"]):
            return hits[0]["id"]
        raise LinkError(field, name, [h["name"] for h in hits])


def _fulltext_query(name: str) -> str:
    """Escape Lucene syntax and allow one typo per word."""
    words = ["".join(ch for ch in w if ch.isalnum()) for w in name.split()]
    return " AND ".join(f"{w}~1" for w in words if w) or '""'
