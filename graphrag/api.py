"""HTTP API.

* ``POST /query``: run a typed plan directly, with no model involved. Useful for dashboards and
  integrations, and it shows that the graph path is deterministic once a plan exists.
* ``POST /ask``: natural-language question -> plan -> Cypher -> answer (graph mode), or top-k
  retrieval -> answer (vector mode, for comparison).
* ``GET /health``: database reachability and which world is loaded.
"""

from __future__ import annotations

from typing import Literal

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from .plan import QueryPlan, plan_from_partial
from .reference import LinkError


class AskRequest(BaseModel):
    question: str
    mode: Literal["graph", "vector"] = "graph"


class AskResponse(BaseModel):
    answer: str
    mode: str
    plan: QueryPlan | None
    evidence: list[str]
    cypher: str
    clarification: list[str]
    latency_ms: float


class QueryResponse(BaseModel):
    value: float | int | None
    rows: list[dict]
    evidence_count: int
    evidence_sample: list[str]
    cypher: str


def create_app(service_factory) -> FastAPI:
    """``service_factory`` returns a GraphRAGService; injected so tests can supply fakes."""
    app = FastAPI(
        title="graphrag-neo4j",
        version="0.1.0",
        description="Graph RAG over a field-service knowledge graph.",
    )
    state: dict = {}

    def service():
        if "service" not in state:
            state["service"] = service_factory()
        return state["service"]

    @app.get("/health")
    def health():
        from .loader import loaded_fingerprint

        try:
            fingerprint = loaded_fingerprint(service().executor.driver)
        except Exception as exc:  # surface the reason; a health check should not 500
            return {"status": "down", "error": type(exc).__name__}
        return {"status": "ok", "world": fingerprint}

    example = {
        "intent": "aggregate",
        "metric": "count",
        "group_by": "site",
        "limit": 5,
        "filters": {"date_from": "2025-01-01", "date_to": "2025-12-31"},
    }

    @app.post("/query", response_model=QueryResponse)
    def query(body: dict = Body(..., examples=[example])):  # noqa: B008
        """Run a query plan. Omitted fields take their defaults (see QueryPlan)."""
        try:
            plan = plan_from_partial(body)
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, f"invalid plan: {exc}") from exc
        try:
            result = service().executor.execute(plan)
        except LinkError as exc:
            raise HTTPException(422, {"detail": str(exc), "candidates": exc.candidates}) from exc
        evidence = sorted(result.evidence)
        return QueryResponse(
            value=result.value,
            rows=result.rows,
            evidence_count=len(evidence),
            evidence_sample=evidence[:20],
            cypher=result.cypher,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        reply = service().ask(request.question, request.mode)
        return AskResponse(**{**reply.__dict__, "evidence": reply.evidence[:50]})

    return app


def app_from_env() -> FastAPI:
    """Entry point for uvicorn: ``uvicorn graphrag.api:app_from_env --factory``."""

    def factory():
        import anthropic

        from .cli import driver_from_env, make_embedder
        from .llm import ModelNotConfigured, make_planner, make_writer
        from .service import GraphRAGService

        try:  # GRAPHRAG_LLM_PROVIDER picks Claude (default) or Azure OpenAI
            planner, writer = make_planner(), make_writer()
        except (anthropic.AnthropicError, ModelNotConfigured):
            planner = writer = None  # no credentials: /query and /health still work
        return GraphRAGService(driver_from_env(), make_embedder(), planner, writer)

    return create_app(factory)
