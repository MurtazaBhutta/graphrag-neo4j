"""Model adapters against fake clients, and the HTTP API against fake services."""

from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from graphrag.api import create_app
from graphrag.llm import AnswerWriter, ClaudePlanner, PlanError
from graphrag.plan import Intent, QueryPlan, make_plan
from graphrag.reference import LinkError
from graphrag.result import Result

PLAN = make_plan(Intent.AGGREGATE, customer="Harbour Group")


class FakeMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    create = parse


def client_returning(response):
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


class TestPlanner:
    def test_request_shape(self):
        client, messages = client_returning(
            SimpleNamespace(stop_reason="end_turn", parsed_output=PLAN)
        )
        planner = ClaudePlanner(client=client, today=date(2026, 3, 1))

        assert planner.plan("How many jobs for Harbour Group?") == PLAN
        call = messages.calls[0]
        assert call["output_format"] is QueryPlan
        assert "Today is 2026-03-01" in call["system"]
        assert call["messages"] == [{"role": "user", "content": "How many jobs for Harbour Group?"}]

    @pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
    def test_unusable_responses_raise(self, stop_reason):
        client, _ = client_returning(SimpleNamespace(stop_reason=stop_reason, parsed_output=None))
        with pytest.raises(PlanError):
            ClaudePlanner(client=client).plan("?")


class TestAnswerWriter:
    def test_context_goes_in_the_user_turn(self):
        text = SimpleNamespace(type="text", text="42 jobs (J-1).")
        client, messages = client_returning(SimpleNamespace(stop_reason="end_turn", content=[text]))
        assert AnswerWriter(client=client).answer("how many?", "Answer: 42") == "42 jobs (J-1)."
        assert "<context>\nAnswer: 42\n</context>" in messages.calls[0]["messages"][0]["content"]

    def test_refusal_is_reported_not_raised(self):
        client, _ = client_returning(SimpleNamespace(stop_reason="refusal", content=[]))
        assert "declined" in AnswerWriter(client=client).answer("?", "")


class FakeExecutor:
    driver = None

    def __init__(self, result=None, error=None):
        self.result, self.error, self.plans = result, error, []

    def execute(self, plan):
        self.plans.append(plan)
        if self.error:
            raise self.error
        return self.result


def api(executor, reply=None):
    service = SimpleNamespace(executor=executor, ask=lambda q, mode: reply)
    return TestClient(create_app(lambda: service))


class TestApi:
    def test_query_accepts_a_partial_plan(self):
        executor = FakeExecutor(Result(rows=[], value=7, evidence={"J-2", "J-1"}, cypher="MATCH"))
        response = api(executor).post(
            "/query", json={"intent": "aggregate", "filters": {"status": "open"}}
        )
        assert response.status_code == 200
        assert response.json() == {
            "value": 7,
            "rows": [],
            "evidence_count": 2,
            "evidence_sample": ["J-1", "J-2"],
            "cypher": "MATCH",
        }
        assert executor.plans[0].filters.status == "open"

    def test_invalid_plan_is_422(self):
        response = api(FakeExecutor()).post(
            "/query", json={"intent": "aggregate", "filters": {"asset_type": "rocket"}}
        )
        assert response.status_code == 422

    def test_unknown_name_is_422_with_suggestions(self):
        executor = FakeExecutor(error=LinkError("customer", "Harbor", ["Harbour Group"]))
        response = api(executor).post(
            "/query", json={"intent": "aggregate", "filters": {"customer": "Harbor"}}
        )
        assert response.status_code == 422
        assert response.json()["detail"]["candidates"] == ["Harbour Group"]

    def test_ask_returns_the_reply(self):
        from graphrag.service import Reply

        reply = Reply("12 jobs.", "graph", PLAN, ["J-1"], "MATCH", [], 5.0)
        body = api(FakeExecutor(), reply).post("/ask", json={"question": "how many?"}).json()
        assert body["answer"] == "12 jobs." and body["plan"]["intent"] == "aggregate"

    def test_openapi_lists_the_endpoints(self):
        paths = api(FakeExecutor()).get("/openapi.json").json()["paths"]
        assert {"/query", "/ask", "/health"} <= set(paths)
