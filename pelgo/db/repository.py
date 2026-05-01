from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pelgo.agent.schemas import CandidateProfile
from pelgo.db.models import Candidate, MatchJob


async def create_candidate(session: AsyncSession, profile: CandidateProfile) -> Candidate:
    candidate = Candidate(
        candidate_id=uuid.UUID(profile.candidate_id),
        name=profile.name,
        email=profile.email,
        seniority_level=profile.seniority_level,
        domain=profile.domain,
        years_experience=profile.years_experience,
        skills=profile.skills,
        education=profile.education,
        work_history=[w.model_dump() for w in profile.work_history],
        raw_text=profile.raw_text,
        created_at=datetime.now(timezone.utc),
    )
    session.add(candidate)
    await session.commit()
    await session.refresh(candidate)
    return candidate


async def get_candidate(session: AsyncSession, candidate_id: str) -> Candidate | None:
    result = await session.execute(
        select(Candidate).where(Candidate.candidate_id == uuid.UUID(candidate_id))
    )
    return result.scalar_one_or_none()


async def create_match_jobs(
    session: AsyncSession, candidate_id: str, jd_inputs: list[str]
) -> list[MatchJob]:
    now = datetime.now(timezone.utc)
    jobs = [
        MatchJob(
            candidate_id=uuid.UUID(candidate_id),
            jd_input=jd,
            status="pending",
            attempt_count=0,
            created_at=now,
            updated_at=now,
        )
        for jd in jd_inputs
    ]
    session.add_all(jobs)
    await session.commit()
    for job in jobs:
        await session.refresh(job)
    return jobs


async def get_match_job(session: AsyncSession, job_id: str) -> MatchJob | None:
    result = await session.execute(
        select(MatchJob).where(MatchJob.job_id == uuid.UUID(job_id))
    )
    return result.scalar_one_or_none()


async def list_match_jobs(
    session: AsyncSession,
    status: str | None,
    limit: int,
    offset: int,
) -> list[MatchJob]:
    stmt = select(MatchJob).order_by(MatchJob.updated_at.desc()).limit(limit).offset(offset)
    if status:
        stmt = stmt.where(MatchJob.status == status)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def claim_next_job(session: AsyncSession) -> MatchJob | None:
    """Claim one pending job using SELECT FOR UPDATE SKIP LOCKED (race-condition safe)."""
    stmt = (
        select(MatchJob)
        .where(MatchJob.status == "pending")
        .where(MatchJob.attempt_count < 3)
        .order_by(MatchJob.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    result = await session.execute(stmt)
    job = result.scalar_one_or_none()
    if job is None:
        return None

    now = datetime.now(timezone.utc)
    job.status = "processing"
    job.attempt_count += 1
    job.processing_started_at = now
    job.updated_at = now
    await session.commit()
    await session.refresh(job)
    return job


async def mark_completed(session: AsyncSession, job_id: str, result: dict) -> None:
    now = datetime.now(timezone.utc)
    await session.execute(
        update(MatchJob)
        .where(MatchJob.job_id == uuid.UUID(job_id))
        .values(status="completed", result=result, updated_at=now)
    )
    await session.commit()


async def mark_failed(
    session: AsyncSession,
    job_id: str,
    error: str,
    attempt_count: int,
    partial_trace: dict | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    if attempt_count >= 3:
        trace = partial_trace or {"tool_calls": [], "total_llm_calls": 0, "fallbacks_triggered": 0}
        values = {
            "status": "failed",
            "error_detail": error,
            "result": {"agent_trace": trace},
            "updated_at": now,
        }
    else:
        values = {
            "status": "pending",
            "error_detail": error,
            "updated_at": now,
        }
    await session.execute(
        update(MatchJob).where(MatchJob.job_id == uuid.UUID(job_id)).values(**values)
    )
    await session.commit()


async def requeue_job(session: AsyncSession, job_id: str) -> MatchJob | None:
    now = datetime.now(timezone.utc)
    await session.execute(
        update(MatchJob)
        .where(MatchJob.job_id == uuid.UUID(job_id))
        .values(status="pending", attempt_count=0, error_detail=None, updated_at=now)
    )
    await session.commit()
    return await get_match_job(session, job_id)


async def reset_stuck_jobs(session: AsyncSession, timeout_minutes: int = 10) -> int:
    """Reset jobs stuck in 'processing' (worker likely crashed mid-run)."""
    from sqlalchemy import text

    result = await session.execute(
        text(
            """
            UPDATE match_jobs
            SET status = 'pending', updated_at = now()
            WHERE status = 'processing'
              AND processing_started_at < now() - make_interval(mins => :mins)
            """
        ),
        {"mins": timeout_minutes},
    )
    await session.commit()
    return result.rowcount
