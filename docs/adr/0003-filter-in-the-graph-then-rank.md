# 0003: Filtered semantic search filters in the graph first, then ranks by similarity

**Status:** accepted

## Context

"Which of Meridian Hotels' jobs in 2025 involved water coming through a ceiling?" combines a
structured filter (one customer, one year) with a semantic condition (the notes describe a
leak). There are two ways to serve it:

1. **Vector first, then filter:** take the top-k nearest neighbours from the index, then drop
   those that fail the filter. This is cheap, but it silently loses recall. When the theme is
   common across customers, the k nearest are mostly *other* customers' leaks, and after
   filtering only a few results remain, or none.
2. **Graph first, then rank:** match the filtered candidates exactly with Cypher, then order
   them by `vector.similarity.cosine(job.embedding, $query)`.

## Decision

Graph first. A filtered candidate set for one customer and year is a few hundred to a few
thousand jobs; computing cosine similarity over that is milliseconds and exact, with no
approximate-index recall loss.

The benchmark includes a case built to expose option 1. The "ceiling leak" theme is planted 24
times across many customers, 3 of them at the focus customer in 2025, so a global top-10 cannot
hold all three.

## Consequences

- Filtered semantic questions get exact filtering: nothing outside the filter can appear, and
  similarity only has to rank the candidates (recall 0.73 against 0.27 for vector-first).
- Unfiltered semantic questions have no candidates to narrow, so the exact scan covers all
  53k jobs. Measured at 53k: about **0.8 s**, against about **16 ms** for the HNSW index. The
  index is not a free win, though. Over the same notes embeddings it found **0.17** of the planted
  jobs against **0.58** for the exact scan, because templated background notes produce thousands
  of identical vectors, and approximate graph search degrades badly on duplicates. Real notes
  are far less repetitive. The production choice is: exact when the filters leave fewer than
  roughly 20k candidates, index above that, with recall checked on a labelled set before
  switching.
