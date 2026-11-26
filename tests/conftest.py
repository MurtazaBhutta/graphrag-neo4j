import os

import pytest

from graphrag.datagen import generate

SMALL = 2_000  # unit tests: fast, no database
INTEGRATION_JOBS = int(os.environ.get("GRAPHRAG_TEST_JOBS", "50000"))


@pytest.fixture(scope="session")
def small_world():
    return generate(SMALL, seed=7)


@pytest.fixture(scope="session")
def loaded():
    """(driver, world) for the world currently in Neo4j. Skips when Neo4j is unreachable or holds
    a different world; CI fails the build if anything was skipped."""
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable

    from graphrag.loader import loaded_fingerprint

    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "graphrag-dev"),
        ),
    )
    try:
        fingerprint = loaded_fingerprint(driver)
    except ServiceUnavailable:
        pytest.skip("Neo4j is not reachable")
    world = generate(INTEGRATION_JOBS, seed=7)
    if fingerprint != world.fingerprint():
        pytest.skip(f"Neo4j does not hold the {INTEGRATION_JOBS}-job world; run graphrag load")
    yield driver, world
    driver.close()


@pytest.fixture(scope="session")
def embedder():
    from graphrag.embeddings import FastEmbedder

    return FastEmbedder()
