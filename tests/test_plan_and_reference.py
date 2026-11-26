import pytest
from pydantic import ValidationError

from graphrag.plan import GroupBy, Intent, Metric, make_plan, plan_from_partial
from graphrag.reference import LinkError, ReferenceExecutor
from graphrag.result import derive


class TestPlanRules:
    def test_part_grouping_only_counts_or_sums_quantity(self):
        with pytest.raises(ValidationError, match="cannot be combined with part grouping"):
            make_plan(Intent.AGGREGATE, metric=Metric.AVG_COST, group_by=GroupBy.PART)

    def test_job_detail_needs_a_job_id(self):
        with pytest.raises(ValidationError, match="job_id"):
            make_plan(Intent.JOB_DETAIL)

    def test_semantic_query_only_on_list_jobs(self):
        with pytest.raises(ValidationError, match="semantic_query"):
            make_plan(Intent.AGGREGATE, semantic_query="leaks")

    @pytest.mark.parametrize("limit", [0, 101])
    def test_limit_bounds(self, limit):
        with pytest.raises(ValidationError):
            make_plan(Intent.LIST_JOBS, limit=limit)

    def test_dates_must_be_iso(self):
        with pytest.raises(ValidationError):
            make_plan(Intent.AGGREGATE, date_from="01/02/2025")

    def test_unknown_enum_values_are_rejected(self):
        with pytest.raises(ValidationError):
            make_plan(Intent.AGGREGATE, asset_type="spaceship")

    def test_partial_plan_fills_defaults(self):
        plan = plan_from_partial(
            {"intent": "aggregate", "group_by": "site", "filters": {"date_from": "2025-01-01"}}
        )
        assert plan == make_plan(Intent.AGGREGATE, group_by=GroupBy.SITE, date_from="2025-01-01")


class TestDerive:
    def test_money_is_pence_until_the_end(self):
        assert derive(Metric.TOTAL_COST, 123456, 0) == 1234.56
        assert derive(Metric.AVG_COST, 1000, 3) == 3.33

    def test_rates_and_empty_groups(self):
        assert derive(Metric.SLA_RATE, 2, 3) == 0.6667
        assert derive(Metric.SLA_RATE, 0, 0) == 0.0


class TestReference:
    """Spot-check the reference against hand-written comprehensions."""

    def test_count_for_a_customer_in_a_quarter(self, small_world):
        w = small_world
        name = w.customers[w.focus_customer].name
        expected = sum(
            1
            for j in w.jobs.values()
            if w.customer_of(j).id == w.focus_customer
            and "2025-01-01" <= j.reported_on <= "2025-03-31"
        )
        plan = make_plan(
            Intent.AGGREGATE, customer=name, date_from="2025-01-01", date_to="2025-03-31"
        )
        assert ReferenceExecutor(w).execute(plan).value == expected

    def test_top_sites_are_sorted_with_stable_ties(self, small_world):
        result = ReferenceExecutor(small_world).execute(
            make_plan(Intent.AGGREGATE, group_by=GroupBy.SITE, limit=10)
        )
        keys = [(-r["value"], r["key"]) for r in result.rows]
        assert keys == sorted(keys)

    def test_names_are_case_insensitive_but_must_exist(self, small_world):
        ref = ReferenceExecutor(small_world)
        name = next(iter(small_world.customers.values())).name
        assert ref.execute(make_plan(Intent.AGGREGATE, customer=name.upper())).value > 0
        with pytest.raises(LinkError):
            ref.execute(make_plan(Intent.AGGREGATE, customer="Nobody Ltd"))

    def test_repeat_visits_count_follow_ups(self, small_world):
        w = small_world
        result = ReferenceExecutor(w).execute(
            make_plan(Intent.REPEAT_VISITS, min_repeats=1, limit=100)
        )
        follow_ups = [j for j in w.jobs.values() if j.follow_up_of]
        top = result.rows[0]
        assert top["repeats"] == max(
            sum(1 for j in follow_ups if j.asset_id == a) for a in {j.asset_id for j in follow_ups}
        )
