"""Benchmark: the same questions through plain vector RAG and graph RAG, scored on retrieval.

Scoring is deterministic; no model grades anything. For each question and each system:

* **evidence recall**: the share of the jobs the answer depends on that made it into the context.
* **sufficient**: whether the context is enough to answer exactly.
  - Graph, structured questions: the structured result equals the reference executor's result.
  - Graph, semantic questions, and vector, all questions: every gold job is in the context.
* **context tokens** (approximate) and **latency**.

What this measures is whether an answering model *could* be right. Whether it *is* right
depends on the model too; ``--answers`` runs Claude on both contexts for that.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass

from .questions import Question
from .retrievers import GraphRetriever, Retrieved, VectorRetriever

CATEGORIES = [
    "lookup",
    "relationship",
    "aggregation",
    "repeat_failures",
    "semantic",
    "semantic_filtered",
]


@dataclass
class Score:
    question: str
    category: str
    system: str
    recall: float
    sufficient: bool
    tokens: int
    latency_ms: float
    plan_matched: bool | None = None  # only when a live planner is used


def score(question: Question, system: str, retrieved: Retrieved) -> Score:
    gold = question.gold_evidence
    recall = len(gold & retrieved.job_ids) / len(gold) if gold else 1.0
    if system == "graph" and not question.semantic:
        sufficient = (
            retrieved.result is not None
            and retrieved.result.comparable() == question.gold.comparable()
        )
    else:
        sufficient = gold <= retrieved.job_ids
    return Score(
        question.id,
        question.category,
        system,
        round(recall, 4),
        sufficient,
        retrieved.approx_tokens,
        round(retrieved.latency_ms, 1),
    )


def run(
    questions: list[Question],
    vector: VectorRetriever,
    graph: GraphRetriever,
    planner=None,
    repeats: int = 3,
) -> list[Score]:
    """``planner`` None means gold plans (isolates retrieval); otherwise plans come from it."""
    scores = []
    for q in questions:
        plan, matched = q.plan, None
        if planner is not None:
            plan = planner.plan(q.text)
            matched = plan == q.plan
        # Warm once, then keep the fastest of a few runs, so latency is not dominated by JIT
        # and cache effects that a long-lived service would not see.
        runs_v = [vector.retrieve(q.text) for _ in range(repeats + 1)][1:]
        runs_g = [graph.retrieve(plan) for _ in range(repeats + 1)][1:]
        v = min(runs_v, key=lambda r: r.latency_ms)
        g = min(runs_g, key=lambda r: r.latency_ms)
        scores.append(score(q, "vector", v))
        s = score(q, "graph", g)
        s.plan_matched = matched
        scores.append(s)
    return scores


def summarise(scores: list[Score]) -> dict:
    summary: dict = {}
    for system in ("vector", "graph"):
        mine = [s for s in scores if s.system == system]
        summary[system] = {
            "overall": _agg(mine),
            **{c: _agg([s for s in mine if s.category == c]) for c in CATEGORIES},
        }
    return summary


def _agg(scores: list[Score]) -> dict:
    if not scores:
        return {}
    return {
        "questions": len(scores),
        "sufficient": sum(s.sufficient for s in scores),
        "sufficient_rate": round(sum(s.sufficient for s in scores) / len(scores), 3),
        "mean_recall": round(statistics.mean(s.recall for s in scores), 3),
        "median_tokens": int(statistics.median(s.tokens for s in scores)),
        "p50_latency_ms": round(statistics.median(s.latency_ms for s in scores), 1),
    }


def gate(scores: list[Score]) -> list[str]:
    """Failures that should stop a merge. Empty list means pass."""
    problems = []
    for s in scores:
        if (
            s.system == "graph"
            and s.category not in ("semantic", "semantic_filtered")
            and not s.sufficient
        ):
            problems.append(f"{s.question}: graph answer differs from the reference executor")
    by_cat = summarise(scores)
    for category in ("semantic", "semantic_filtered"):
        g, v = by_cat["graph"][category], by_cat["vector"][category]
        if g and g["mean_recall"] < v["mean_recall"]:
            problems.append(
                f"{category}: graph recall {g['mean_recall']} is below vector {v['mean_recall']}"
            )
    return problems


def format_table(scores: list[Score], questions: list[Question]) -> str:
    text = {q.id: q.text for q in questions}
    pairs: dict[str, dict[str, Score]] = {}
    for s in scores:
        pairs.setdefault(s.question, {})[s.system] = s
    lines = [f"{'id':<4} {'category':<18} {'vector':>14} {'graph':>14}  question", "-" * 110]
    for qid, pair in pairs.items():
        v, g = pair["vector"], pair["graph"]
        lines.append(f"{qid:<4} {v.category:<18} {_cell(v):>14} {_cell(g):>14}  {text[qid][:60]}")
    summary = summarise(scores)
    lines += [
        "-" * 110,
        f"{'sufficient (mean recall)':<25} {'vector':>16} {'graph':>16}",
    ]
    for cat in ["overall", *CATEGORIES]:
        v, g = summary["vector"][cat], summary["graph"][cat]
        if v:
            cells = [f"{x['sufficient']}/{x['questions']} ({x['mean_recall']:.2f})" for x in (v, g)]
            lines.append(f"{cat:<25} {cells[0]:>16} {cells[1]:>16}")
    o = summary["vector"]["overall"], summary["graph"]["overall"]
    lines.append(
        f"{'median context tokens':<23} {o[0]['median_tokens']:>14} {o[1]['median_tokens']:>14}"
    )
    lines.append(
        f"{'p50 latency (ms)':<23} {o[0]['p50_latency_ms']:>14} {o[1]['p50_latency_ms']:>14}"
    )
    return "\n".join(lines)


def _cell(s: Score) -> str:
    return f"{'OK' if s.sufficient else '--'} r={s.recall:.2f}"


def to_json(scores: list[Score], meta: dict) -> str:
    return json.dumps(
        {"meta": meta, "summary": summarise(scores), "scores": [asdict(s) for s in scores]},
        indent=2,
    )
