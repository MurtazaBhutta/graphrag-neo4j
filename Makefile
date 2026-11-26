PY ?= python
JOBS ?= 50000
PLANNER ?= claude

.PHONY: install up load bench bench-live test test-integration api down

install:
	$(PY) -m pip install -e ".[dev]"

up:            ## Neo4j only, for running the CLI and tests from your machine
	docker compose up -d --wait neo4j

load:          ## generate the world and load it (first run embeds ~53k records, ~10 min on CPU)
	$(PY) -m graphrag load --jobs $(JOBS)

bench:         ## vector RAG vs graph RAG, gold plans, deterministic
	$(PY) -m graphrag bench --jobs $(JOBS) --out benchmark/results/bench-$(JOBS).json

bench-live:    ## same, but plans come from a live model (PLANNER=claude|azure_openai; costs money)
	$(PY) -m graphrag bench --jobs $(JOBS) --planner $(PLANNER) --out benchmark/results/bench-live-$(PLANNER)-$(JOBS).json

test:
	$(PY) -m pytest -m "not neo4j"

test-integration:
	GRAPHRAG_TEST_JOBS=$(JOBS) $(PY) -m pytest -m neo4j -rs

api:
	uvicorn graphrag.api:app_from_env --factory --reload

down:
	docker compose down -v
