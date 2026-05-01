from __future__ import annotations

import asyncio
import tempfile

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from pelgo.api.deps import get_db
from pelgo.api.schemas import CandidateIngestRequest, CandidateIngestResponse
from pelgo.db import repository
from pelgo.pdf_extractor import parse_resume_pdf, parse_resume_text
from pelgo.logging import log

router = APIRouter(tags=["candidates"])


@router.post("/candidate", response_model=CandidateIngestResponse)
async def ingest_candidate(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CandidateIngestResponse:
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        resume = form.get("resume")
        if resume is None:
            raise HTTPException(status_code=400, detail="Expected 'resume' file in multipart form")
        if not getattr(resume, "filename", "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF resumes are supported")
        content = await resume.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        profile = await asyncio.to_thread(parse_resume_pdf, tmp_path)

    elif "application/json" in content_type:
        data = await request.json()
        body = CandidateIngestRequest(**data)
        profile = await asyncio.to_thread(parse_resume_text, body.resume_text)

    else:
        raise HTTPException(
            status_code=415,
            detail="Use application/json with {resume_text} or multipart/form-data with a PDF file",
        )

    candidate = await repository.create_candidate(db, profile)
    log("candidate_ingested", candidate_id=str(candidate.candidate_id), name=candidate.name)

    return CandidateIngestResponse(
        candidate_id=str(candidate.candidate_id),
        name=candidate.name,
        email=candidate.email,
        seniority_level=candidate.seniority_level,
        domain=candidate.domain,
        years_experience=candidate.years_experience,
        skills=candidate.skills or [],
    )
