import os
import subprocess
import sys

from graphrag.datagen import THEMES, generate, render_record


def test_same_seed_same_world_across_processes():
    """Python randomises string hashing per process, so set iteration order differs between
    runs. A world that is only reproducible inside one process cannot be checked against a
    database loaded by another, so this runs the generator in three fresh interpreters."""
    code = "from graphrag.datagen import generate; print(generate(1500, 7).fingerprint())"
    prints = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(prints) == 1


def test_different_seed_different_world():
    assert generate(500, 7).fingerprint() != generate(500, 8).fingerprint()


def test_shape(small_world):
    w = small_world
    assert len(w.customers) == 40
    assert len(w.jobs) > 2_000  # follow-ups come on top of the base jobs
    assert {len(v) for k, v in w.themes.items() if k != "ceiling_leak"} == {6}
    assert len(w.themes["ceiling_leak"]) == 24


def test_follow_ups_are_later_visits_to_the_same_asset(small_world):
    for job in small_world.jobs.values():
        if job.follow_up_of:
            original = small_world.jobs[job.follow_up_of]
            assert original.asset_id == job.asset_id
            assert original.reported_on <= job.reported_on


def test_planted_phrasings_appear_only_in_planted_jobs(small_world):
    planted = {j for ids in small_world.themes.values() for j in ids}
    phrasings = {text for _, texts in THEMES.values() for text in texts}
    for job in small_world.jobs.values():
        assert (job.notes in phrasings) == (job.id in planted), job.id


def test_semantic_filter_question_has_gold_at_the_focus_customer(small_world):
    w = small_world
    at_focus_2025 = [
        j
        for j in w.themes["ceiling_leak"]
        if w.customer_of(w.jobs[j]).id == w.focus_customer
        and w.jobs[j].reported_on.startswith("2025")
    ]
    assert len(at_focus_2025) >= 3


def test_rendered_record_carries_every_field(small_world):
    job = next(j for j in small_world.jobs.values() if j.parts and j.follow_up_of)
    text = render_record(job, small_world)
    site = small_world.sites[job.site_id]
    for needle in (
        job.id,
        job.reported_on,
        site.name,
        small_world.customer_of(job).name,
        small_world.engineers[job.engineer_id].name,
        job.fault_category,
        job.priority,
        job.status,
        job.notes,
        job.follow_up_of,
        small_world.parts[job.parts[0][0]].name,
    ):
        assert needle in text
