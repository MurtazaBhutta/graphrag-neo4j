#!/usr/bin/env bash
# API smoke test against the Compose stack (after a load). No model key needed: it uses /query.
set -euo pipefail
BASE="${BASE:-http://localhost:8000}"

health="$(curl -sS "$BASE/health")"
echo "health: $health"
[[ "$health" == *'"status":"ok"'* && "$health" != *'"world":null'* ]] || { echo "FAIL health" >&2; exit 1; }

body='{"intent": "aggregate", "group_by": "site", "limit": 3,
       "filters": {"date_from": "2025-01-01", "date_to": "2025-12-31"}}'
top="$(curl -sS -X POST "$BASE/query" -H 'Content-Type: application/json' -d "$body")"
echo "top sites 2025: $top" | cut -c1-300
rows="$(python3 -c 'import json,sys; print(len(json.load(sys.stdin)["rows"]))' <<<"$top")"
[[ "$rows" == "3" ]] || { echo "FAIL expected 3 rows, got $rows" >&2; exit 1; }

code="$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$BASE/query" \
  -H 'Content-Type: application/json' -d '{"intent": "aggregate", "filters": {"customer": "No Such Customer Ltd"}}')"
[[ "$code" == "422" ]] || { echo "FAIL unknown customer should be 422, got $code" >&2; exit 1; }
echo "ok   unknown customer -> 422"
echo "smoke test passed"
