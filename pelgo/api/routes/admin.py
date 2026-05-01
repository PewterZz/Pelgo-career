from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from pelgo.api.deps import get_db
from pelgo.api.schemas import RequeueResponse
from pelgo.db import repository
from pelgo.logging import log

router = APIRouter(tags=["admin"])


@router.post("/admin/matches/{job_id}/requeue", response_model=RequeueResponse)
async def requeue_match(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> RequeueResponse:
    job = await repository.get_match_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status == "processing":
        raise HTTPException(status_code=409, detail="Job is currently processing; wait or reset stuck jobs first")

    updated = await repository.requeue_job(db, job_id)
    log("job_requeued", job_id=job_id, previous_status=job.status)

    return RequeueResponse(
        job_id=job_id,
        status=updated.status,
        attempt_count=updated.attempt_count,
        message="Job re-queued successfully",
    )
