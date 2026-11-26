"""The two model calls: question -> typed plan, and (question, retrieved context) -> answer.

Each call has two implementations, Claude (the default) and Azure OpenAI, chosen with
``GRAPHRAG_LLM_PROVIDER``. They share the prompts and the ``QueryPlan`` schema, so the rest of the
service, and the benchmark's plan-match score, do not depend on which provider is used.
"""

from __future__ import annotations

import os
from datetime import date

import anthropic
import openai
from pydantic import ValidationError

from .datagen import ASSET_TYPES
from .plan import QueryPlan

MODEL = "claude-opus-5-5"

PLANNER_SYSTEM = """You turn questions about a facilities-maintenance business into a query plan.
The data is a graph: Customer -< Site -< Asset; each Job is at a Site, on an Asset, attended by an
Engineer, and may use Parts and be a follow-up (return visit) of an earlier Job.

Choose the intent:
- job_detail: the question names one job id (J-######).
- list_jobs: the user wants the jobs themselves. If they describe what happened in words (a
  symptom, a smell, a leak), put that description in semantic_query; filters still apply.
- aggregate: counts, totals, averages, rates, rankings or breakdowns. Rankings and breakdowns use
  group_by; "top five" means limit 5. Spend is total_cost; SLA performance is sla_rate; parts
  quantities use part_quantity with group_by part.
- repeat_visits: assets that needed return visits; min_repeats is the minimum number of follow-up
  visits (1 for "any repeat visit").

Filters: copy names exactly as the user wrote them; do not invent ids. Leave a filter null unless
the question states it. Dates are inclusive YYYY-MM-DD; "in 2025" is 2025-01-01..2025-12-31, a
quarter is its three months. Today is {today}.
Fault categories by asset type: {faults}.
Use metric count and min_repeats 1 where they do not apply, and limit 20 unless the question
implies another number (at most 100)."""

ANSWER_SYSTEM = """Answer the question using only the context, which was retrieved from the
company's maintenance records. Be direct: lead with the answer, then the supporting detail.
Cite job ids where they support a claim. If the context does not contain enough to answer
exactly, say what is missing instead of estimating."""


PROVIDERS = ("anthropic", "azure_openai")
MAX_OUTPUT_TOKENS = 16000
DECLINED = "The model declined to answer this question."


class PlanError(ValueError):
    pass


class ModelNotConfigured(RuntimeError):
    """The selected provider is missing its settings (endpoint, key or deployment)."""


def _planner_system(today: date | None) -> str:
    faults = "; ".join(f"{t}: {', '.join(f)}" for t, (_, _, f) in ASSET_TYPES.items())
    return PLANNER_SYSTEM.format(today=(today or date.today()).isoformat(), faults=faults)


def _user_turn(question: str, context: str) -> str:
    return f"<context>\n{context}\n</context>\n\nQuestion: {question}"


class ClaudePlanner:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = MODEL,
        today: date | None = None,
    ):
        self._client = client or anthropic.Anthropic()
        self._model = model
        self._system = _planner_system(today)

    def plan(self, question: str) -> QueryPlan:
        try:
            response = self._client.beta.messages.parse(
                model=self._model,
                max_tokens=16000,
                system=self._system,
                messages=[{"role": "user", "content": question}],
                output_format=QueryPlan,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except ValidationError as exc:  # schema-valid JSON that breaks a plan rule
            raise PlanError(f"the planner produced an invalid plan: {exc}") from exc
        if response.stop_reason in ("refusal", "max_tokens") or response.parsed_output is None:
            raise PlanError(f"no usable plan (stop_reason={response.stop_reason})")
        return response.parsed_output


class AnswerWriter:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = MODEL):
        self._client = client or anthropic.Anthropic()
        self._model = model

    def answer(self, question: str, context: str) -> str:
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=16000,
            system=ANSWER_SYSTEM,
            messages=[{"role": "user", "content": _user_turn(question, context)}],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return DECLINED
        return "".join(block.text for block in response.content if block.type == "text")


# --- Azure OpenAI ---------------------------------------------------------------------------


def azure_openai_client() -> openai.OpenAI:
    """A client for the Azure OpenAI v1 API, from ``AZURE_OPENAI_ENDPOINT`` and ``_API_KEY``.

    The v1 API takes the plain OpenAI client with the resource's ``/openai/v1/`` base URL and
    needs no ``api-version``.
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    missing = [
        name
        for name, value in (("AZURE_OPENAI_ENDPOINT", endpoint), ("AZURE_OPENAI_API_KEY", key))
        if not value
    ]
    if missing:
        raise ModelNotConfigured(f"Azure OpenAI needs {', '.join(missing)}")
    return openai.OpenAI(base_url=f"{endpoint.rstrip('/')}/openai/v1/", api_key=key)


def azure_openai_deployment() -> str:
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    if not deployment:
        raise ModelNotConfigured("Azure OpenAI needs AZURE_OPENAI_DEPLOYMENT")
    return deployment


class AzureOpenAIPlanner:
    """The same plan, from an Azure OpenAI deployment that supports structured outputs."""

    def __init__(
        self,
        client: openai.OpenAI | None = None,
        deployment: str | None = None,
        today: date | None = None,
    ):
        self._deployment = deployment or azure_openai_deployment()
        self._client = client or azure_openai_client()
        self._system = _planner_system(today)

    def plan(self, question: str) -> QueryPlan:
        try:
            completion = self._client.chat.completions.parse(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": question},
                ],
                response_format=QueryPlan,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
            )
        except ValidationError as exc:  # schema-valid JSON that breaks a plan rule
            raise PlanError(f"the planner produced an invalid plan: {exc}") from exc
        except openai.LengthFinishReasonError as exc:
            raise PlanError("no usable plan (output was truncated)") from exc
        except openai.ContentFilterFinishReasonError as exc:
            raise PlanError("no usable plan (blocked by the content filter)") from exc
        message = completion.choices[0].message
        if message.refusal:
            raise PlanError("no usable plan (the model refused)")
        if message.parsed is None:
            raise PlanError("no usable plan (no structured output returned)")
        return message.parsed


class AzureOpenAIAnswerWriter:
    def __init__(self, client: openai.OpenAI | None = None, deployment: str | None = None):
        self._deployment = deployment or azure_openai_deployment()
        self._client = client or azure_openai_client()

    def answer(self, question: str, context: str) -> str:
        completion = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": _user_turn(question, context)},
            ],
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
        choice = completion.choices[0]
        if choice.finish_reason == "content_filter" or choice.message.refusal:
            return DECLINED
        return choice.message.content or ""


# --- selection ------------------------------------------------------------------------------


def provider_from_env() -> str:
    provider = os.environ.get("GRAPHRAG_LLM_PROVIDER", "anthropic").strip() or "anthropic"
    if provider not in PROVIDERS:
        raise ValueError(f"GRAPHRAG_LLM_PROVIDER must be one of {PROVIDERS}, not {provider!r}")
    return provider


def make_planner(provider: str | None = None):
    provider = provider or provider_from_env()
    if provider == "azure_openai":
        return AzureOpenAIPlanner()
    if provider == "anthropic":
        return ClaudePlanner()
    raise ValueError(f"unknown provider {provider!r}")


def make_writer(provider: str | None = None):
    provider = provider or provider_from_env()
    if provider == "azure_openai":
        return AzureOpenAIAnswerWriter()
    if provider == "anthropic":
        return AnswerWriter()
    raise ValueError(f"unknown provider {provider!r}")
