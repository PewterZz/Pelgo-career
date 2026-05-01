"""Insert a sample candidate and two match jobs for demo and manual testing.

Idempotent: skips if the sample candidate already exists.
"""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import select

from pelgo.agent.schemas import CandidateProfile
from pelgo.db.models import Candidate
from pelgo.db.repository import create_candidate, create_match_jobs
from pelgo.db.session import AsyncSessionLocal

SAMPLE_EMAIL = "jane@example.com"

SAMPLE_PROFILE = CandidateProfile(
    name="Jane Smith",
    email=SAMPLE_EMAIL,
    skills=["python", "fastapi", "postgresql", "docker", "kubernetes"],
    years_experience=7.0,
    seniority_level="senior",
    domain="software engineering",
    education=["B.S. Computer Science, State University"],
    work_history=[],
    raw_text="Jane Smith\njane@example.com\nSenior Backend Engineer — 7 years\nPython FastAPI PostgreSQL Docker Kubernetes",
)

SAMPLE_JDS = [
    (
        "Senior Backend Engineer\n"
        "Required: Python, FastAPI, PostgreSQL, Docker\n"
        "Nice to have: Kubernetes, Terraform, Redis\n"
        "Responsibilities: Design and implement REST APIs, optimize database queries, "
        "lead backend architecture decisions."
    ),
    (
        "Staff Engineer\n"
        "Required: Python, Go, Kubernetes, AWS, Terraform\n"
        "Responsibilities: Cross-team platform architecture, drive engineering standards, "
        "mentor engineers across squads."
    ),
]


async def main() -> None:
    async with AsyncSessionLocal() as session:
        existing = await session.execute(
            select(Candidate).where(Candidate.email == SAMPLE_EMAIL)
        )
        if existing.scalar_one_or_none() is not None:
            print("Seed data already exists, skipping.")
            return

    async with AsyncSessionLocal() as session:
        candidate = await create_candidate(session, SAMPLE_PROFILE)
        print(f"Seeded candidate: {candidate.candidate_id}  ({candidate.name})")

    async with AsyncSessionLocal() as session:
        jobs = await create_match_jobs(session, str(candidate.candidate_id), SAMPLE_JDS)
        print(f"Seeded {len(jobs)} match jobs:")
        for job in jobs:
            print(f"  {job.job_id}  status={job.status}")


asyncio.run(main())
