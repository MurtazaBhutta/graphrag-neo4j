"""Schema and bulk load. Batched UNWIND writes; idempotent (MERGE on ids, constraints first)."""

from __future__ import annotations

import time
from collections.abc import Callable

from neo4j import Driver

from .datagen import World, render_record
from .embeddings import Embedder

BATCH = 2_000

SCHEMA = [
    "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (n:Customer) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT site_id IF NOT EXISTS FOR (n:Site) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT asset_id IF NOT EXISTS FOR (n:Asset) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT engineer_id IF NOT EXISTS FOR (n:Engineer) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT part_sku IF NOT EXISTS FOR (n:Part) REQUIRE n.sku IS UNIQUE",
    "CREATE CONSTRAINT job_id IF NOT EXISTS FOR (n:Job) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX job_reported IF NOT EXISTS FOR (n:Job) ON (n.reported_on)",
    # Entity linking: users type names, not ids.
    "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS "
    "FOR (n:Customer|Site|Engineer|Part) ON EACH [n.name]",
]


# Two embeddings per job (docs/adr/0002):
#   job_embedding:   the whole rendered record, the standard "RAG over records" document
#   notes_embedding: the engineer's free-text notes only, the part with no structure to query
VECTOR_INDEXES = {"job_embedding": "embedding", "notes_embedding": "notes_embedding"}


def vector_index(name: str, prop: str, dimensions: int) -> str:
    return (
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (j:Job) ON j.{prop} "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dimensions)}, "
        "`vector.similarity_function`: 'cosine'}}"
    )


def load(driver: Driver, world: World, embedder: Embedder, *, log: Callable = print) -> None:
    started = time.perf_counter()
    with driver.session() as s:
        s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS").consume()
        # `IF NOT EXISTS` would keep an index built for a different embedding size, and every
        # vector query would then fail. Recreate it so its dimensions always match the embedder.
        for name, prop in VECTOR_INDEXES.items():
            s.run(f"DROP INDEX {name} IF EXISTS").consume()
            s.run(vector_index(name, prop, embedder.dimensions)).consume()
        for statement in SCHEMA:
            s.run(statement).consume()

        s.run(
            "UNWIND $rows AS r CREATE (:Customer {id: r.id, name: r.name, sector: r.sector})",
            rows=[vars(c) for c in world.customers.values()],
        ).consume()
        s.run(
            """UNWIND $rows AS r MATCH (c:Customer {id: r.customer_id})
                 CREATE (:Site {id: r.id, name: r.name, city: r.city, postcode: r.postcode})
                        -[:BELONGS_TO]->(c)""",
            rows=[vars(x) for x in world.sites.values()],
        ).consume()
        s.run(
            """UNWIND $rows AS r MATCH (s:Site {id: r.site_id})
                 CREATE (:Asset {id: r.id, type: r.type, make: r.make})-[:INSTALLED_AT]->(s)""",
            rows=[vars(a) for a in world.assets.values()],
        ).consume()
        s.run(
            "UNWIND $rows AS r CREATE (:Engineer {id: r.id, name: r.name, trade: r.trade, "
            "region: r.region})",
            rows=[vars(e) for e in world.engineers.values()],
        ).consume()
        s.run(
            "UNWIND $rows AS r CREATE (:Part {sku: r.sku, name: r.name, "
            "unit_cost_pence: r.unit_cost_pence})",
            rows=[vars(p) for p in world.parts.values()],
        ).consume()
        log(f"reference data loaded: {len(world.sites)} sites, {len(world.assets)} assets")

        jobs = list(world.jobs.values())
        log(f"embedding {len(jobs)} job records (cached after the first run)...")
        vectors = embedder.embed_documents([render_record(j, world) for j in jobs])
        unique_notes = sorted({j.notes for j in jobs})
        note_vectors = dict(zip(unique_notes, embedder.embed_documents(unique_notes), strict=True))

        for start in range(0, len(jobs), BATCH):
            rows = [
                {
                    "id": j.id,
                    "reported_on": j.reported_on,
                    "site_id": j.site_id,
                    "asset_id": j.asset_id,
                    "engineer_id": j.engineer_id,
                    "fault_category": j.fault_category,
                    "priority": j.priority,
                    "status": j.status,
                    "sla_met": j.sla_met,
                    "cost_pence": j.cost_pence,
                    "notes": j.notes,
                    "embedding": vectors[i].tolist(),
                    "notes_embedding": note_vectors[j.notes].tolist(),
                    "parts": [{"sku": sku, "qty": qty} for sku, qty in j.parts],
                }
                for i, j in enumerate(jobs[start : start + BATCH], start=start)
            ]
            s.run(
                """
                UNWIND $rows AS r
                MATCH (s:Site {id: r.site_id}), (a:Asset {id: r.asset_id}),
                      (e:Engineer {id: r.engineer_id})
                CREATE (j:Job {id: r.id, reported_on: r.reported_on,
                               fault_category: r.fault_category, priority: r.priority,
                               status: r.status, sla_met: r.sla_met, cost_pence: r.cost_pence,
                               notes: r.notes})
                CREATE (j)-[:AT_SITE]->(s), (j)-[:ON_ASSET]->(a), (j)-[:ATTENDED_BY]->(e)
                WITH j, r
                CALL db.create.setNodeVectorProperty(j, 'embedding', r.embedding)
                CALL db.create.setNodeVectorProperty(j, 'notes_embedding', r.notes_embedding)
                WITH j, r
                UNWIND r.parts AS used
                MATCH (p:Part {sku: used.sku})
                CREATE (j)-[:USED_PART {qty: used.qty}]->(p)
            """,
                rows=rows,
            ).consume()
        s.run(
            """UNWIND $rows AS r MATCH (j:Job {id: r.id}), (o:Job {id: r.of})
                 CREATE (j)-[:FOLLOW_UP_OF]->(o)""",
            rows=[{"id": j.id, "of": j.follow_up_of} for j in jobs if j.follow_up_of],
        ).consume()

        s.run(
            "MERGE (m:Meta {key: 'world'}) SET m.seed = $seed, m.jobs = $n, m.fingerprint = $f",
            seed=world.seed,
            n=len(jobs),
            f=world.fingerprint(),
        ).consume()
        s.run("CALL db.awaitIndexes(300)").consume()
    log(f"loaded {len(jobs)} jobs in {time.perf_counter() - started:.0f}s")


def loaded_fingerprint(driver: Driver) -> str | None:
    with driver.session() as s:
        record = s.run("MATCH (m:Meta {key: 'world'}) RETURN m.fingerprint AS f").single()
    return record["f"] if record else None
