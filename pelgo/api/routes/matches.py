from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from pelgo.api.deps import get_db
from pelgo.api.schemas import (
    JobRef,
    MatchJobListResponse,
    MatchJobResponse,
    SubmitMatchesRequest,
    SubmitMatchesResponse,
)
from pelgo.db import repository
from pelgo.db.models import MatchJob
from pelgo.logging import log

router = APIRouter(tags=["matches"])


def _job_to_response(job: MatchJob) -> MatchJobResponse:
    return MatchJobResponse(
        job_id=str(job.job_id),
        candidate_id=str(job.candidate_id),
        status=job.status,
        attempt_count=job.attempt_count,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        result=job.result,
        error_detail=job.error_detail,
    )


@router.post("/matches", response_model=SubmitMatchesResponse)
async def submit_matches(
    body: SubmitMatchesRequest,
    db: AsyncSession = Depends(get_db),
) -> SubmitMatchesResponse:
    candidate = await repository.get_candidate(db, body.candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Candidate {body.candidate_id} not found")

    jobs = await repository.create_match_jobs(db, body.candidate_id, body.jds)

    for job in jobs:
        log(
            "job_enqueued",
            job_id=str(job.job_id),
            candidate_id=body.candidate_id,
            jd_preview=job.jd_input[:80],
        )

    return SubmitMatchesResponse(
        jobs=[JobRef(job_id=str(j.job_id)) for j in jobs],
        message=f"{len(jobs)} job(s) enqueued",
    )


@router.get("/matches/{job_id}", response_model=MatchJobResponse)
async def get_match(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> MatchJobResponse:
    job = await repository.get_match_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_to_response(job)


@router.get("/matches", response_model=MatchJobListResponse)
async def list_matches(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(..., ge=1, le=100),
    offset: int = Query(..., ge=0),
    db: AsyncSession = Depends(get_db),
) -> MatchJobListResponse:
    valid_statuses = {"pending", "processing", "completed", "failed"}
    if status and status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"status must be one of {sorted(valid_statuses)}",
        )

    jobs = await repository.list_match_jobs(db, status, limit, offset)
    return MatchJobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        limit=limit,
        offset=offset,
    )
