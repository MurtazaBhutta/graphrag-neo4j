"""Command line: ``python -m graphrag load | bench | ask``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from neo4j import GraphDatabase

from . import bench
from .cypher import CypherExecutor
from .datagen import generate
from .loader import load, loaded_fingerprint
from .questions import build_questions
from .retrievers import GraphRetriever, VectorRetriever

INDEXES = {"records": "job_embedding", "notes": "notes_embedding"}


def driver_from_env():
    return GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "graphrag-dev"),
        ),
    )


def make_embedder():
    from .embeddings import FastEmbedder

    return FastEmbedder()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphrag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="generate the synthetic world and load it into Neo4j")
    p_bench = sub.add_parser("bench", help="vector RAG vs graph RAG on the benchmark questions")
    for p in (p_load, p_bench):
        p.add_argument("--jobs", type=int, default=50_000, help="base jobs (follow-ups are extra)")
        p.add_argument("--seed", type=int, default=7)
    p_bench.add_argument("--k", type=int, default=10, help="top-k for the vector baseline")
    p_bench.add_argument(
        "--vector-index",
        choices=["records", "notes"],
        default="records",
        help="what the baseline searches: whole job records (default) or notes only",
    )
    p_bench.add_argument(
        "--planner",
        choices=["gold", "claude", "azure_openai"],
        default="gold",
        help="gold: use the labelled plans; claude or azure_openai: plan with that live model",
    )
    p_bench.add_argument("--out", type=Path, help="write the full results as JSON")
    p_bench.add_argument("--gate", action="store_true", help="exit 1 if the gate fails")

    p_ask = sub.add_parser(
        "ask", help="answer a question with graph RAG and the model in GRAPHRAG_LLM_PROVIDER"
    )
    p_ask.add_argument("question")

    args = parser.parse_args(argv)
    driver = driver_from_env()
    try:
        if args.command == "load":
            load(driver, generate(args.jobs, args.seed), make_embedder())
            return 0
        if args.command == "bench":
            return _bench(driver, args)
        return _ask(driver, args.question)
    finally:
        driver.close()


def _bench(driver, args) -> int:
    world = generate(args.jobs, args.seed)
    if loaded_fingerprint(driver) != world.fingerprint():
        print(
            f"Neo4j does not hold the world for --jobs {args.jobs} --seed {args.seed}. "
            "Run `python -m graphrag load` with the same options first.",
            file=sys.stderr,
        )
        return 2
    embedder = make_embedder()
    questions = build_questions(world)
    planner = None
    if args.planner != "gold":
        from .llm import make_planner

        planner = make_planner("anthropic" if args.planner == "claude" else args.planner)
    scores = bench.run(
        questions,
        VectorRetriever(driver, embedder, k=args.k, index=INDEXES[args.vector_index]),
        GraphRetriever(CypherExecutor(driver, embedder)),
        planner=planner,
    )
    print(bench.format_table(scores, questions))
    if args.out:
        meta = {
            "jobs": len(world.jobs),
            "seed": args.seed,
            "k": args.k,
            "vector_index": args.vector_index,
            "planner": args.planner,
            "fingerprint": world.fingerprint(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(bench.to_json(scores, meta), encoding="utf-8")
    problems = bench.gate(scores)
    print("\nGATE: " + ("PASS" if not problems else "FAIL\n  " + "\n  ".join(problems)))
    return 1 if (args.gate and problems) else 0


def _ask(driver, question: str) -> int:
    from .llm import make_planner, make_writer
    from .service import GraphRAGService

    service = GraphRAGService(driver, make_embedder(), make_planner(), make_writer())
    reply = service.ask(question)
    print(reply.answer)
    print(f"\n[plan] {reply.plan.model_dump_json(exclude_none=True)}")
    print(f"[evidence] {len(reply.evidence)} jobs; latency {reply.latency_ms:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
