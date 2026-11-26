"""Question in, answer out: plan -> link -> Cypher -> answer, with a clarification path."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .cypher import CypherExecutor
from .llm import PlanError
from .plan import QueryPlan
from .reference import LinkError
from .retrievers import GraphRetriever, VectorRetriever


@dataclass
class Reply:
    answer: str
    mode: str
    plan: QueryPlan | None = None
    evidence: list[str] = field(default_factory=list)
    cypher: str = ""
    clarification: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class GraphRAGService:
    def __init__(self, driver, embedder, planner, writer, k: int = 10):
        self.executor = CypherExecutor(driver, embedder)
        self.graph = GraphRetriever(self.executor)
        self.vector = VectorRetriever(driver, embedder, k=k)
        self.planner, self.writer = planner, writer

    def ask(self, question: str, mode: str = "graph") -> Reply:
        started = time.perf_counter()
        if self.planner is None or self.writer is None:
            return Reply(
                "No language model is configured; set ANTHROPIC_API_KEY, or set "
                "GRAPHRAG_LLM_PROVIDER=azure_openai with AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT. "
                "POST /query still works with an explicit plan.",
                mode,
            )
        if mode == "vector":
            retrieved = self.vector.retrieve(question)
            answer = self.writer.answer(question, retrieved.context)
            return Reply(
                answer, mode, evidence=sorted(retrieved.job_ids), latency_ms=_since(started)
            )
        try:
            plan = self.planner.plan(question)
            retrieved = self.graph.retrieve(plan)
        except PlanError as exc:
            return Reply(
                f"I could not turn that into a query: {exc}", mode, latency_ms=_since(started)
            )
        except LinkError as exc:
            # Ask rather than guess: a wrong customer gives a confident, wrong number.
            return Reply(str(exc), mode, clarification=exc.candidates, latency_ms=_since(started))
        answer = self.writer.answer(question, retrieved.context)
        return Reply(
            answer,
            mode,
            plan,
            sorted(retrieved.job_ids),
            retrieved.cypher,
            latency_ms=_since(started),
        )


def _since(started: float) -> float:
    return (time.perf_counter() - started) * 1000
