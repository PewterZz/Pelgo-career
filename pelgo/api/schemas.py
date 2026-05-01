from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CandidateIngestRequest(BaseModel):
    resume_text: str = Field(min_length=50)


class CandidateIngestResponse(BaseModel):
    candidate_id: str
    name: str
    email: str | None
    seniority_level: str
    domain: str
    years_experience: float
    skills: list[str]
    message: str = "Candidate profile stored successfully"


class SubmitMatchesRequest(BaseModel):
    candidate_id: str
    jds: list[str] = Field(
        min_length=1,
        max_length=10,
        description="Up to 10 JD strings or URLs",
    )


class JobRef(BaseModel):
    job_id: str
    status: Literal["pending"] = "pending"


class SubmitMatchesResponse(BaseModel):
    jobs: list[JobRef]
    message: str


class MatchJobResponse(BaseModel):
    job_id: str
    candidate_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    attempt_count: int
    created_at: str
    updated_at: str
    result: dict | None = None
    error_detail: str | None = None


class MatchJobListResponse(BaseModel):
    jobs: list[MatchJobResponse]
    limit: int
    offset: int


class RequeueResponse(BaseModel):
    job_id: str
    status: str
    attempt_count: int
    message: str
