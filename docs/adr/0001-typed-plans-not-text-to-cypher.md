# 0001: The model produces a typed query plan, never Cypher

**Status:** accepted

## Context

The usual way to put an LLM in front of a graph database is text-to-Cypher: show the model the
schema and let it write the query. It demos well and fails in three ways that matter in
production:

1. **Safety.** Generated Cypher is code. A question such as "…and then delete the test jobs" is
   one prompt injection away from `DETACH DELETE`. Read-only credentials help, but Neo4j
   Community has no role-based access control.
2. **Correctness that cannot be checked.** A generated query that runs and returns a plausible
   number is indistinguishable from a correct one. Aggregations are where this bites: a missing
   `DISTINCT`, a join that double-counts parts, or a date compared as the wrong type.
3. **No regression signal.** When the model changes, every generated query changes with it, and
   nothing tells you which answers moved.

## Decision

The model fills in a `QueryPlan`, a Pydantic model returned through structured outputs:

- `intent`, `metric`, `group_by`, `asset_type`, `priority`, `status` are **closed enums**
- names, dates and free text are **values**, validated (ISO dates, limit 1–100) and passed to
  Neo4j only as **query parameters**
- incoherent plans are rejected before any query runs (for example average cost grouped by part)

`cypher.py` compiles the plan from fixed fragments keyed by those enums. Queries run in read
transactions. `test_cypher_safety.py` puts an injection string in every free-text field and
asserts it never appears in the query text.

Names go through **entity linking**: an exact match first, then a full-text match accepted only
if it clearly beats the runner-up (score at least 1.5×). Otherwise the user gets a
clarification with candidates instead of a guessed customer. A confident answer about the wrong
customer is the worst failure this system can have.

## Consequences

- Each intent's Cypher can be tested exhaustively. `tests/test_integration.py` checks every
  benchmark question and 300 random plans against an independent Python implementation
  (`reference.py`).
- Planner quality becomes measurable: a plan either equals the gold plan or it does not
  (`bench --planner claude`).
- The cost is coverage. A question outside the four intents gets no answer rather than an
  improvised query. Adding an intent means adding Cypher, a reference implementation and tests,
  which is deliberate friction.
