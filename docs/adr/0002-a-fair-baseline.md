# 0002: The vector baseline gets every advantage except structure

**Status:** accepted

## Context

"Graph RAG beats vector RAG" is easy to show against a weak baseline: chunk unrelated text,
embed it with a small model, and keep k at 3. That result says nothing about whether a graph is
worth running in production.

## Decision

The baseline is plain vector RAG done properly:

- **Same database, same embedding model.** Both pipelines query the same Neo4j instance with
  the same `BAAI/bge-small-en-v1.5` model, so differences come from the retrieval strategy, not
  the infrastructure. Each job carries two vectors: one of the whole record (`job_embedding`, the
  baseline's default) and one of the notes alone (`notes_embedding`, which the graph path ranks
  by). The baseline is run against **both** indexes and reported both ways (`--vector-index`),
  so it is never held to the weaker one.
- **Whole records, not fragments.** Each job is embedded as a rendered record holding every
  field: customer, site, asset, engineer, fault, priority, status, SLA, cost, parts, follow-up
  link and notes (`render_record`). The baseline has all the information; what it lacks is the
  ability to filter, join and count exactly.
- **k is a parameter**, and results are reported at more than one k.
- **Retrieval is scored, not generated answers.** A question is *sufficient* for the vector
  baseline when every record the answer depends on is in its context, the most generous
  possible reading. Whether an LLM then counts 340 rows correctly is a separate, further loss
  that the benchmark does not charge to the baseline.

## Consequences

- Where vector RAG loses here, it loses for structural reasons: top-k cannot contain the 2,000
  records behind a count, and similarity cannot apply "only this customer, only 2025".
- Where it is competitive (single-theme semantic search), the numbers show that too. A hybrid
  is the right production design, and the graph path uses vector similarity for exactly that.
- Pure dense retrieval is weak at exact identifiers such as `J-004211`. That is a real property
  of dense retrieval, but adding BM25 would fix lookups for the baseline without touching its
  aggregate failures. The README says so rather than claiming the lookup gap as a graph win.
