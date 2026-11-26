"""The two model calls: question -> typed plan, and (question, retrieved context) -> answer."""

from __future__ import annotations

from datetime import date

import anthropic
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


class PlanError(ValueError):
    pass


class ClaudePlanner:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = MODEL,
        today: date | None = None,
    ):
        self._client = client or anthropic.Anthropic()
        self._model = model
        faults = "; ".join(f"{t}: {', '.join(f)}" for t, (_, _, f) in ASSET_TYPES.items())
        self._system = PLANNER_SYSTEM.format(
            today=(today or date.today()).isoformat(), faults=faults
        )

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
            messages=[
                {
                    "role": "user",
                    "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}",
                }
            ],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return "The model declined to answer this question."
        return "".join(block.text for block in response.content if block.type == "text")
