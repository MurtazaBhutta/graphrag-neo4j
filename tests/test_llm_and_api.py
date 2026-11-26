"""Model adapters against fake clients, and the HTTP API against fake services."""

from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from graphrag.api import create_app
from graphrag.llm import (
    AnswerWriter,
    AzureOpenAIAnswerWriter,
    AzureOpenAIPlanner,
    ClaudePlanner,
    ModelNotConfigured,
    PlanError,
    make_planner,
    make_writer,
    provider_from_env,
)
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


class FakeCompletions:
    def __init__(self, result):
        self.result, self.calls = result, []

    def _respond(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    parse = create = _respond


def azure_client_returning(result):
    completions = FakeCompletions(result)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def completion(*, parsed=None, content=None, refusal=None, finish_reason="stop"):
    message = SimpleNamespace(parsed=parsed, content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def raw_completion(finish_reason):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": ""},
                }
            ],
        }
    )


class TestAzureOpenAIPlanner:
    def test_request_shape(self):
        client, completions = azure_client_returning(completion(parsed=PLAN))
        planner = AzureOpenAIPlanner(client=client, deployment="gpt-4o", today=date(2026, 3, 1))

        assert planner.plan("How many jobs for Harbour Group?") == PLAN
        call = completions.calls[0]
        assert call["model"] == "gpt-4o"
        assert call["response_format"] is QueryPlan
        assert call["max_completion_tokens"] > 0
        system, user = call["messages"]
        assert system["role"] == "system" and "Today is 2026-03-01" in system["content"]
        assert user == {"role": "user", "content": "How many jobs for Harbour Group?"}

    @pytest.mark.parametrize(
        "result",
        [
            completion(refusal="I can't help with that."),
            completion(parsed=None),
        ],
        ids=["refusal", "no-parsed-output"],
    )
    def test_unusable_responses_raise(self, result):
        client, _ = azure_client_returning(result)
        with pytest.raises(PlanError):
            AzureOpenAIPlanner(client=client, deployment="d").plan("?")

    @pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
    def test_sdk_finish_reason_errors_become_plan_errors(self, finish_reason):
        import openai

        error = (
            openai.LengthFinishReasonError(completion=raw_completion(finish_reason))
            if finish_reason == "length"
            else openai.ContentFilterFinishReasonError()
        )
        client, _ = azure_client_returning(error)
        with pytest.raises(PlanError):
            AzureOpenAIPlanner(client=client, deployment="d").plan("?")

    def test_plan_rule_violation_becomes_plan_error(self):
        try:
            QueryPlan.model_validate({"intent": "job_detail"})
        except ValidationError as exc:
            error = exc
        client, _ = azure_client_returning(error)
        with pytest.raises(PlanError, match="invalid plan"):
            AzureOpenAIPlanner(client=client, deployment="d").plan("?")


class TestAzureOpenAIAnswerWriter:
    def test_context_goes_in_the_user_turn(self):
        client, completions = azure_client_returning(completion(content="42 jobs (J-1)."))
        writer = AzureOpenAIAnswerWriter(client=client, deployment="gpt-4o")
        assert writer.answer("how many?", "Answer: 42") == "42 jobs (J-1)."
        system, user = completions.calls[0]["messages"]
        assert system["role"] == "system"
        assert "<context>\nAnswer: 42\n</context>" in user["content"]
        assert completions.calls[0]["model"] == "gpt-4o"

    @pytest.mark.parametrize(
        "result",
        [completion(refusal="no"), completion(content="", finish_reason="content_filter")],
        ids=["refusal", "content-filter"],
    )
    def test_declines_are_reported_not_raised(self, result):
        client, _ = azure_client_returning(result)
        assert "declined" in AzureOpenAIAnswerWriter(client=client, deployment="d").answer("?", "")


class TestProviderSelection:
    AZURE_VARS = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT")

    def test_default_is_anthropic(self, monkeypatch):
        monkeypatch.delenv("GRAPHRAG_LLM_PROVIDER", raising=False)
        assert provider_from_env() == "anthropic"

    def test_unknown_provider_is_rejected(self, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_LLM_PROVIDER", "gemini")
        with pytest.raises(ValueError, match="GRAPHRAG_LLM_PROVIDER"):
            provider_from_env()

    def test_azure_selected_from_env(self, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_LLM_PROVIDER", "azure_openai")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        planner, writer = make_planner(), make_writer()
        assert isinstance(planner, AzureOpenAIPlanner)
        assert isinstance(writer, AzureOpenAIAnswerWriter)
        assert str(planner._client.base_url) == "https://example.openai.azure.com/openai/v1/"

    @pytest.mark.parametrize("missing", AZURE_VARS)
    def test_missing_azure_setting_is_a_clear_error(self, monkeypatch, missing):
        for name, value in zip(
            self.AZURE_VARS, ("https://x.openai.azure.com", "k", "d"), strict=True
        ):
            monkeypatch.setenv(name, value)
        monkeypatch.delenv(missing)
        with pytest.raises(ModelNotConfigured, match=missing):
            make_planner("azure_openai")

    def test_api_without_credentials_still_serves_query(self, monkeypatch):
        """``app_from_env`` must build even when the chosen provider has no settings."""
        from graphrag import api as api_module

        for name in self.AZURE_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("GRAPHRAG_LLM_PROVIDER", "azure_openai")
        created = {}

        class FakeService:
            def __init__(self, driver, embedder, planner, writer):
                created.update(planner=planner, writer=writer)

        monkeypatch.setattr("graphrag.cli.driver_from_env", lambda: None)
        monkeypatch.setattr("graphrag.cli.make_embedder", lambda: None)
        monkeypatch.setattr("graphrag.service.GraphRAGService", FakeService)
        monkeypatch.setattr(api_module, "create_app", lambda factory: factory)
        api_module.app_from_env()()  # the service factory the app would call
        assert created == {"planner": None, "writer": None}


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
