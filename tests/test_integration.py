"""
Integration test: full stack — ingest candidate → submit JD → worker runs → result returned.

Uses testcontainers for ephemeral Postgres; mocks AgentRunner.run to avoid real LLM calls.
Requires Docker to be running (tests are skipped when Docker is unavailable).
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, patch


def _docker_available() -> bool:
    try:
        import docker
        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="Docker daemon not available"
)

from testcontainers.postgres import PostgresContainer


# ---------------------------------------------------------------------------
# Patch _gemini_seniority so candidate ingest never calls real Gemini
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_gemini_seniority():
    with patch("pelgo.pdf_extractor._gemini_seniority", return_value="senior"):
        yield


# ---------------------------------------------------------------------------
# Session-scoped Postgres container + Alembic migration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pg_container():
    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture(scope="session")
def psycopg2_url(pg_container):
    return pg_container.get_connection_url()


@pytest.fixture(scope="session")
def asyncpg_url(pg_container):
    url = pg_container.get_connection_url()
    return (
        url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
           .replace("postgresql://", "postgresql+asyncpg://")
    )


@pytest.fixture(scope="session", autouse=True)
def run_migrations(psycopg2_url):
    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", psycopg2_url)
    command.upgrade(cfg, "head")


# ---------------------------------------------------------------------------
# Fixtures: point the app + worker at the test DB
# ---------------------------------------------------------------------------


@pytest.fixture
def test_app(asyncpg_url):
    """Return the FastAPI app wired to the test DB."""
    import pelgo.db.session as session_mod
    session_mod.reset_for_testing(asyncpg_url)

    from pelgo.api.main import app
    return app


@pytest.fixture
def stub_result():
    from pelgo.agent.schemas import (
        AgentTrace,
        DimensionScores,
        MatchResult,
        ToolCallRecord,
    )

    return MatchResult(
        job_id="00000000-0000-0000-0000-000000000001",
        overall_score=82,
        confidence="high",
        dimension_scores=DimensionScores(skills=90, experience=80, seniority_fit=75),
        matched_skills=["python", "fastapi"],
        gap_skills=["terraform"],
        reasoning="Strong match on core skills.",
        learning_plan=[],
        agent_trace=AgentTrace(
            tool_calls=[
                ToolCallRecord(
                    tool="extract_jd_requirements", status="success", latency_ms=120
                ),
                ToolCallRecord(
                    tool="score_candidate_against_requirements",
                    status="success",
                    latency_ms=5,
                ),
            ],
            total_llm_calls=3,
            fallbacks_triggered=0,
        ),
    )


# ---------------------------------------------------------------------------
# The integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_submit_run_retrieve(test_app, stub_result):
    """
    1. POST /api/v1/candidate  → candidate stored, candidate_id returned
    2. POST /api/v1/matches    → job enqueued, status=pending
    3. Worker claims + processes the job (mocked runner)
    4. GET /api/v1/matches/{id} → status=completed, valid agent_trace
    """
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://testserver"
    ) as client:
        # Step 1: ingest candidate
        resp = await client.post(
            "/api/v1/candidate",
            json={
                "resume_text": (
                    "Jane Smith\njane@example.com\n"
                    "Senior Backend Engineer — 7 years experience\n"
                    "Python FastAPI PostgreSQL Docker Kubernetes Redis"
                )
            },
        )
        assert resp.status_code == 200, resp.text
        candidate_id = resp.json()["candidate_id"]
        assert candidate_id

        # Step 2: submit a JD
        resp = await client.post(
            "/api/v1/matches",
            json={
                "candidate_id": candidate_id,
                "jds": [
                    "Senior Backend Engineer. Required: Python, FastAPI, PostgreSQL. "
                    "Nice to have: Kubernetes, Terraform."
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        jobs_data = resp.json()["jobs"]
        assert len(jobs_data) == 1
        job_id = jobs_data[0]["job_id"]
        assert jobs_data[0]["status"] == "pending"

        # Step 3: worker claims and processes the job
        from pelgo.db import repository
        from pelgo.db.session import AsyncSessionLocal
        from pelgo.worker.loop import process_job

        async with AsyncSessionLocal() as session:
            job = await repository.claim_next_job(session)

        assert job is not None
        assert str(job.job_id) == job_id

        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=stub_result)
        await process_job(worker_id=0, job=job, runner=mock_runner)

        # Step 4: verify result via API
        resp = await client.get(f"/api/v1/matches/{job_id}")
        assert resp.status_code == 200, resp.text

        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"] is not None

        result = data["result"]
        assert result["overall_score"] == 82
        assert result["confidence"] == "high"

        trace = result["agent_trace"]
        assert trace["total_llm_calls"] == 3
        assert trace["fallbacks_triggered"] == 0
        assert len(trace["tool_calls"]) >= 1
        assert trace["tool_calls"][0]["tool"] == "extract_jd_requirements"
        assert trace["tool_calls"][0]["status"] == "success"


@pytest.mark.asyncio
async def test_list_and_filter_by_status(test_app):
    """GET /api/v1/matches supports status filter and pagination."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://testserver"
    ) as client:
        resp = await client.get("/api/v1/matches", params={"limit": 10, "offset": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert "limit" in data
        assert "offset" in data

        resp = await client.get(
            "/api/v1/matches", params={"status": "completed", "limit": 10, "offset": 0}
        )
        assert resp.status_code == 200
        for job in resp.json()["jobs"]:
            assert job["status"] == "completed"


@pytest.mark.asyncio
async def test_candidate_not_found(test_app):
    """POST /api/v1/matches returns 404 for unknown candidate."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://testserver"
    ) as client:
        resp = await client.post(
            "/api/v1/matches",
            json={
                "candidate_id": "00000000-0000-0000-0000-000000000999",
                "jds": ["Some JD text"],
            },
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_requeue_completed_job(test_app, stub_result):
    """Admin requeue resets a completed job back to pending."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://testserver"
    ) as client:
        # Ingest + submit
        resp = await client.post(
            "/api/v1/candidate",
            json={"resume_text": "Bob Lee\nbob@example.com\nMid Backend Engineer 2 years Python FastAPI"},
        )
        candidate_id = resp.json()["candidate_id"]

        resp = await client.post(
            "/api/v1/matches",
            json={"candidate_id": candidate_id, "jds": ["Python Backend role"]},
        )
        job_id = resp.json()["jobs"][0]["job_id"]

        # Process the job
        from pelgo.db import repository
        from pelgo.db.session import AsyncSessionLocal
        from pelgo.worker.loop import process_job

        async with AsyncSessionLocal() as session:
            job = await repository.claim_next_job(session)

        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=stub_result)
        await process_job(worker_id=0, job=job, runner=mock_runner)

        # Requeue
        resp = await client.post(f"/api/v1/admin/matches/{job_id}/requeue")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"
        assert resp.json()["attempt_count"] == 0
