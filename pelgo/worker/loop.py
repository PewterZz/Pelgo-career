from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from pelgo.agent.runner import AgentRunner
from pelgo.agent.schemas import CandidateProfile, WorkEntry
from pelgo.db import repository
from pelgo.db.models import MatchJob
from pelgo.db.session import AsyncSessionLocal
from pelgo.logging import log

_POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SEC", "1.0"))
_NUM_WORKERS = int(os.environ.get("WORKER_CONCURRENCY", "2"))


async def process_job(worker_id: int, job: MatchJob, runner: AgentRunner) -> None:
    log("job_claimed", job_id=str(job.job_id), worker_id=worker_id, attempt=job.attempt_count)
    try:
        async with AsyncSessionLocal() as session:
            candidate_row = await repository.get_candidate(session, str(job.candidate_id))

        if candidate_row is None:
            raise ValueError(f"Candidate {job.candidate_id} not found")

        candidate = CandidateProfile(
            candidate_id=str(candidate_row.candidate_id),
            name=candidate_row.name,
            email=candidate_row.email,
            skills=candidate_row.skills or [],
            years_experience=candidate_row.years_experience,
            seniority_level=candidate_row.seniority_level,
            domain=candidate_row.domain,
            education=candidate_row.education or [],
            work_history=[WorkEntry(**w) for w in (candidate_row.work_history or [])],
            raw_text=candidate_row.raw_text or "",
        )

        result = await runner.run(candidate, job.jd_input, job_id=str(job.job_id))

        async with AsyncSessionLocal() as session:
            await repository.mark_completed(
                session, str(job.job_id), result.model_dump(mode="json")
            )

        log(
            "job_status_transition",
            job_id=str(job.job_id),
            status="completed",
            score=result.overall_score,
            worker_id=worker_id,
        )

    except Exception as exc:
        log(
            "job_status_transition",
            job_id=str(job.job_id),
            status="error",
            error=str(exc),
            attempt=job.attempt_count,
            worker_id=worker_id,
            level="error",
        )
        async with AsyncSessionLocal() as session:
            await repository.mark_failed(
                session,
                str(job.job_id),
                str(exc),
                attempt_count=job.attempt_count,
            )


async def worker_loop(worker_id: int, runner: AgentRunner) -> None:
    while True:
        try:
            async with AsyncSessionLocal() as session:
                job = await repository.claim_next_job(session)

            if job is None:
                await asyncio.sleep(_POLL_INTERVAL)
                continue

            await process_job(worker_id, job, runner)

        except Exception as exc:
            log("worker_error", worker_id=worker_id, error=str(exc), level="error")
            await asyncio.sleep(_POLL_INTERVAL)


async def main() -> None:
    load_dotenv()

    async with AsyncSessionLocal() as session:
        reset_count = await repository.reset_stuck_jobs(session)
    if reset_count:
        log("worker_startup_reset", count=reset_count)

    runner = AgentRunner()
    workers = [worker_loop(i, runner) for i in range(_NUM_WORKERS)]
    log("worker_startup", concurrency=_NUM_WORKERS)
    await asyncio.gather(*workers)
