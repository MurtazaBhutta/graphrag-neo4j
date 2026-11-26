"""A plain-Python executor for query plans, over the in-memory world.

It exists only to check the Cypher. It shares no query code with ``cypher.py``, so when the two
agree on every benchmark question and on hundreds of random plans, that is real evidence the
Cypher is right, not the same bug written twice.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

from .datagen import Job, World
from .plan import GroupBy, Intent, Metric, QueryPlan
from .result import Result, finish_aggregate, finish_list, finish_repeats


class LinkError(ValueError):
    """A name in the plan matches no entity, or matches several. Carries suggestions."""

    def __init__(self, field: str, value: str, candidates: list[str]):
        self.field, self.value, self.candidates = field, value, candidates
        hint = f"; did you mean {', '.join(candidates)}?" if candidates else ""
        super().__init__(f"no {field} called {value!r}{hint}")


class ReferenceExecutor:
    def __init__(self, world: World):
        self.w = world
        self._names = {
            "customer": {c.name.lower(): c.id for c in world.customers.values()},
            "site": {s.name.lower(): s.id for s in world.sites.values()},
            "engineer": {e.name.lower(): e.id for e in world.engineers.values()},
            "part": {p.name.lower(): p.sku for p in world.parts.values()},
        }

    # -- public --------------------------------------------------------------------------------

    def execute(self, plan: QueryPlan) -> Result:
        if plan.intent is Intent.JOB_DETAIL:
            job = self.w.jobs.get(plan.filters.job_id or "")
            return Result(
                rows=[self.detail_row(job)] if job else [], evidence={job.id} if job else set()
            )
        if plan.intent is Intent.LIST_JOBS:
            if plan.semantic_query:
                raise NotImplementedError("the reference executor has no notion of similarity")
            jobs = sorted(self.matching(plan), key=lambda j: (j.reported_on, j.id), reverse=True)
            return finish_list([self.list_row(j) for j in jobs], plan.limit)
        if plan.intent is Intent.REPEAT_VISITS:
            return self._repeats(plan)
        return self._aggregate(plan)

    def matching(self, plan: QueryPlan) -> Iterator[Job]:
        f = plan.filters
        ids = {k: self._link(k, getattr(f, k)) for k in ("customer", "site", "engineer", "part")}
        for job in self.w.jobs.values():
            site = self.w.sites[job.site_id]
            if f.job_id and job.id != f.job_id:
                continue
            if ids["customer"] and site.customer_id != ids["customer"]:
                continue
            if ids["site"] and job.site_id != ids["site"]:
                continue
            if ids["engineer"] and job.engineer_id != ids["engineer"]:
                continue
            if ids["part"] and ids["part"] not in {sku for sku, _ in job.parts}:
                continue
            if f.asset_type and self.w.assets[job.asset_id].type != f.asset_type:
                continue
            if f.fault_category and job.fault_category != f.fault_category:
                continue
            if f.priority and job.priority != f.priority:
                continue
            if f.status and job.status != f.status:
                continue
            if f.date_from and job.reported_on < f.date_from:
                continue
            if f.date_to and job.reported_on > f.date_to:
                continue
            yield job

    def detail_row(self, job: Job) -> dict:
        site = self.w.sites[job.site_id]
        return {
            **self.list_row(job),
            "site_city": site.city,
            "asset": job.asset_id,
            "cost": round(job.cost_pence / 100, 2),
            "sla_met": job.sla_met,
            "notes": job.notes,
            "parts": [{"part": self.w.parts[s].name, "qty": q} for s, q in job.parts],
            "follow_up_of": job.follow_up_of,
        }

    def list_row(self, job: Job) -> dict:
        site = self.w.sites[job.site_id]
        return {
            "job_id": job.id,
            "reported_on": job.reported_on,
            "customer": self.w.customers[site.customer_id].name,
            "site": site.name,
            "asset_type": self.w.assets[job.asset_id].type,
            "engineer": self.w.engineers[job.engineer_id].name,
            "fault_category": job.fault_category,
            "priority": job.priority,
            "status": job.status,
        }

    # -- intents -------------------------------------------------------------------------------

    def _aggregate(self, plan: QueryPlan) -> Result:
        groups: dict = defaultdict(lambda: [0, 0])
        evidence: set[str] = set()
        part_filter = self._link("part", plan.filters.part)
        part_rows = plan.group_by is GroupBy.PART or plan.metric is Metric.PART_QUANTITY

        if part_rows:
            jobs_per_key: dict = defaultdict(set)
            for job in self.matching(plan):
                for sku, qty in job.parts:
                    if part_filter and sku != part_filter:
                        continue
                    key = self._key(plan.group_by, job, sku)
                    evidence.add(job.id)
                    jobs_per_key[key].add(job.id)
                    groups[key][0] += qty
            if plan.metric is Metric.COUNT:
                groups = {k: [len(v), 0] for k, v in jobs_per_key.items()}
        else:
            for job in self.matching(plan):
                evidence.add(job.id)
                g = groups[self._key(plan.group_by, job)]
                if plan.metric is Metric.COUNT:
                    g[0] += 1
                elif plan.metric in (Metric.TOTAL_COST, Metric.AVG_COST):
                    g[0] += job.cost_pence
                    g[1] += 1
                elif plan.metric is Metric.SLA_RATE:
                    g[0] += int(job.sla_met)
                    g[1] += 1
        return finish_aggregate(plan, dict(groups), evidence)

    def _repeats(self, plan: QueryPlan) -> Result:
        per_asset: dict[str, list[str]] = defaultdict(list)
        for job in self.matching(plan):
            if job.follow_up_of:
                per_asset[job.asset_id].append(job.id)
        rows = [
            {
                "asset": aid,
                "asset_type": self.w.assets[aid].type,
                "site": self.w.sites[self.w.assets[aid].site_id].name,
                "repeats": len(jobs),
                "jobs": jobs,
            }
            for aid, jobs in per_asset.items()
            if len(jobs) >= plan.min_repeats
        ]
        return finish_repeats(rows, plan.limit)

    # -- helpers -------------------------------------------------------------------------------

    def _key(self, group_by: GroupBy | None, job: Job, sku: str | None = None):
        site = self.w.sites[job.site_id]
        match group_by:
            case None:
                return None
            case GroupBy.CUSTOMER:
                return self.w.customers[site.customer_id].name
            case GroupBy.SITE:
                return site.name
            case GroupBy.ENGINEER:
                return self.w.engineers[job.engineer_id].name
            case GroupBy.ASSET:
                return job.asset_id
            case GroupBy.ASSET_TYPE:
                return self.w.assets[job.asset_id].type
            case GroupBy.FAULT_CATEGORY:
                return job.fault_category
            case GroupBy.MONTH:
                return job.reported_on[:7]
            case GroupBy.PART:
                return self.w.parts[sku].name

    def _link(self, field: str, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return self._names[field][value.strip().lower()]
        except KeyError:
            raise LinkError(field, value, []) from None
