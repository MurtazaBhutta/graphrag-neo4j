from graphrag.bench import gate, score, summarise
from graphrag.plan import Intent, make_plan
from graphrag.questions import Question, build_questions
from graphrag.result import Result
from graphrag.retrievers import Retrieved


def structured(gold_value=5, evidence=frozenset({"J-1", "J-2"})):
    gold = Result(rows=[], value=gold_value, evidence=set(evidence))
    return Question(
        "q1", "aggregation", "how many?", make_plan(Intent.AGGREGATE), set(evidence), gold
    )


def semantic():
    return Question(
        "q2",
        "semantic",
        "leaks?",
        make_plan(Intent.LIST_JOBS, semantic_query="x"),
        {"J-1", "J-2"},
        None,
    )


def retrieved(ids, result=None):
    return Retrieved(context="x" * 400, job_ids=set(ids), latency_ms=1.0, result=result)


def test_graph_structured_is_judged_on_the_exact_answer():
    q = structured()
    right = Result(rows=[], value=5, evidence={"J-1", "J-2"})
    wrong = Result(rows=[], value=4, evidence={"J-1", "J-2"})
    assert score(q, "graph", retrieved(q.gold_evidence, right)).sufficient
    assert not score(q, "graph", retrieved(q.gold_evidence, wrong)).sufficient


def test_vector_needs_every_gold_record_in_context():
    q = structured()
    half = score(q, "vector", retrieved({"J-1", "J-9"}))
    assert half.recall == 0.5 and not half.sufficient
    assert score(q, "vector", retrieved({"J-1", "J-2", "J-9"})).sufficient


def test_semantic_is_judged_on_recall_for_both():
    q = semantic()
    assert score(q, "graph", retrieved({"J-1", "J-2"})).sufficient
    assert not score(q, "graph", retrieved({"J-1"})).sufficient


def test_gate_fails_on_a_wrong_graph_answer():
    q = structured()
    good = [
        score(q, "vector", retrieved(set())),
        score(q, "graph", retrieved(q.gold_evidence, q.gold)),
    ]
    bad = [good[0], score(q, "graph", retrieved(q.gold_evidence, Result(rows=[], value=0)))]
    assert gate(good) == []
    assert gate(bad) and "differs from the reference" in gate(bad)[0]


def test_gate_fails_if_graph_semantic_recall_falls_below_vector():
    q = semantic()
    scores = [score(q, "vector", retrieved({"J-1", "J-2"})), score(q, "graph", retrieved({"J-1"}))]
    assert any("semantic" in p for p in gate(scores))


def test_summary_counts(small_world):
    q = structured()
    scores = [score(q, "vector", retrieved(set())), score(q, "graph", retrieved(set(), q.gold))]
    s = summarise(scores)
    assert s["graph"]["overall"]["sufficient"] == 1
    assert s["vector"]["overall"]["sufficient"] == 0


def test_benchmark_questions_are_well_formed(small_world):
    questions = build_questions(small_world)
    assert len(questions) == 25
    assert len({q.id for q in questions}) == 25
    for q in questions:
        assert q.gold_evidence, f"{q.id} has no evidence: the question would be vacuous"
        if q.semantic:
            assert len(q.gold_evidence) <= q.plan.limit
