"""FR-MNT-5 tests — the scheduled runner: run-once, isolation, idempotence, cron.

The runner is tested against tiny fake jobs and against the real default job set
wired over the conftest fakes, all offline (no APScheduler clock advance needed
to assert scheduling registration).
"""

from __future__ import annotations

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from mnemozine.config import Settings
from mnemozine.interfaces import MaintenanceJob, MaintenanceReport
from mnemozine.maintenance.runner import (
    MaintenanceRunner,
    build_default_jobs,
)
from tests.conftest import FakeEmbeddingProvider, FakeLLMProvider, InMemoryStorage


class _RecordingJob:
    """A minimal MaintenanceJob that records how many times it ran."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.runs = 0

    @property
    def name(self) -> str:
        return self._name

    async def run(self) -> MaintenanceReport:
        self.runs += 1
        return MaintenanceReport(job_name=self._name, consolidated=1)


class _ExplodingJob:
    @property
    def name(self) -> str:
        return "boom"

    async def run(self) -> MaintenanceReport:
        raise RuntimeError("kaboom")


def test_recording_job_satisfies_protocol() -> None:
    assert isinstance(_RecordingJob("x"), MaintenanceJob)


@pytest.mark.asyncio
async def test_run_once_runs_all_jobs_in_order() -> None:
    jobs = [_RecordingJob("a"), _RecordingJob("b"), _RecordingJob("c")]
    runner = MaintenanceRunner(jobs, settings=Settings())
    reports = await runner.run_once()
    assert [r.job_name for r in reports] == ["a", "b", "c"]
    assert all(j.runs == 1 for j in jobs)


@pytest.mark.asyncio
async def test_run_once_is_idempotent_under_repeat() -> None:
    jobs = [_RecordingJob("a"), _RecordingJob("b")]
    runner = MaintenanceRunner(jobs, settings=Settings())
    await runner.run_once()
    await runner.run_once()
    assert all(j.runs == 2 for j in jobs)  # each ran once per pass, no overlap


@pytest.mark.asyncio
async def test_one_failing_job_does_not_abort_the_pass() -> None:
    after = _RecordingJob("after")
    runner = MaintenanceRunner([_ExplodingJob(), after], settings=Settings())
    reports = await runner.run_once()
    # The exploding job yields an ERROR report, the next job still runs.
    assert reports[0].job_name == "boom"
    assert any("ERROR" in n for n in reports[0].notes)
    assert after.runs == 1


def test_schedule_registers_cron_job() -> None:
    settings = Settings()
    settings.maintenance.cron = "0 4 * * *"
    runner = MaintenanceRunner([_RecordingJob("a")], settings=settings)
    scheduler = AsyncIOScheduler()
    runner.schedule(scheduler)
    job = scheduler.get_job("mnemozine-maintenance")
    assert job is not None
    assert job.max_instances == 1
    assert job.coalesce is True


@pytest.mark.asyncio
async def test_default_job_set_runs_end_to_end_offline() -> None:
    storage = InMemoryStorage()
    llm = FakeLLMProvider(text_responder=lambda p, s: "consolidated")
    embeddings = FakeEmbeddingProvider()
    jobs = build_default_jobs(storage, llm, embeddings, settings=Settings())
    names = [j.name for j in jobs]
    # Category merge (the category analogue of entity resolution) joins the
    # default scheduled set between resolution and decay.
    assert names == [
        "consolidation",
        "entity_resolution",
        "category_merge",
        "decay",
        "audit",
    ]
    runner = MaintenanceRunner(jobs, settings=Settings())
    # No data: every job runs cleanly and returns a report.
    reports = await runner.run_once()
    assert len(reports) == 5
    assert {r.job_name for r in reports} == set(names)
