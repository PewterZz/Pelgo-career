"""Smoke test: runs the full agent pipeline with stubbed tools.

Run with: /opt/anaconda3/bin/python -m pytest tests/ -v
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from pelgo.agent.schemas import (
    AgentState,
    CandidateProfile,
    DimensionScores,
    JDRequirements,
    MatchResult,
)
from pelgo.pdf_extractor import parse_resume_text

SAMPLE_RESUME = """
Jane Smith
jane@example.com

Senior Software Engineer with 7 years of experience.
Skills: Python, FastAPI, PostgreSQL, Docker, Kubernetes, Redis, AWS

Work History:
- Senior Backend Engineer at Acme Corp, 2019-2024
  Python, FastAPI, PostgreSQL, Redis

Education: B.S. Computer Science
"""

SAMPLE_JD = """
We are looking for a Senior Backend Engineer to join our team.
Required skills: Python, FastAPI, PostgreSQL, Redis, Docker
Nice to have: Kubernetes, Terraform
Responsibilities:
- Build and maintain scalable microservices
- Design database schemas
- Code review and mentorship
Seniority: Senior
Domain: backend engineering
"""


def test_parse_resume_text():
    with patch("pelgo.pdf_extractor._gemini_seniority", return_value="senior"):
        profile = parse_resume_text(SAMPLE_RESUME)
    assert profile.name == "Jane Smith"
    assert profile.email == "jane@example.com"
    assert "python" in profile.skills
    assert profile.years_experience >= 5
    assert profile.seniority_level == "senior"


def test_candidate_profile_schema():
    with patch("pelgo.pdf_extractor._gemini_seniority", return_value="senior"):
        profile = parse_resume_text(SAMPLE_RESUME)
    dumped = profile.model_dump()
    restored = CandidateProfile.model_validate(dumped)
    assert restored.name == profile.name


def test_match_result_schema_validation():
    result = MatchResult(
        job_id="test-123",
        overall_score=75,
        confidence="medium",
        dimension_scores=DimensionScores(skills=80, experience=70, seniority_fit=75),
        matched_skills=["python", "fastapi"],
        gap_skills=["terraform"],
        reasoning="Strong backend match with minor infra gap.",
        learning_plan=[],
        agent_trace={"tool_calls": [], "total_llm_calls": 3, "fallbacks_triggered": 0},
    )
    assert result.overall_score == 75
    assert result.confidence == "medium"
    out = result.model_dump_json()
    assert "agent_trace" in out


@pytest.mark.asyncio
async def test_full_agent_pipeline_stubbed():
    """Integration smoke test: agent runs end-to-end with mocked LLM tools."""
    from pelgo.agent.runner import AgentRunner
    from pelgo.agent.schemas import AgentTrace, ToolCallRecord

    with patch("pelgo.pdf_extractor._gemini_seniority", return_value="senior"):
        profile = parse_resume_text(SAMPLE_RESUME)

    stub_jd = JDRequirements(
        required_skills=["python", "fastapi", "postgresql", "redis", "docker"],
        nice_to_have_skills=["kubernetes", "terraform"],
        seniority_level="senior",
        domain="backend engineering",
        responsibilities=["Build microservices", "Design schemas"],
    )

    stub_score = {
        "overall_score": 82,
        "confidence": "high",
        "dimension_scores": {"skills": 90, "experience": 80, "seniority_fit": 75},
        "matched_skills": ["python", "fastapi", "postgresql", "redis", "docker"],
        "gap_skills": ["terraform"],
    }

    stub_priority = {
        "prioritized_skills": [
            {
                "skill": "terraform",
                "priority_rank": 1,
                "estimated_match_gain_pct": 5.0,
                "rationale": "Infrastructure-as-code skill required for senior DevOps alignment",
            }
        ]
    }

    stub_resources = {
        "skill": "terraform",
        "resources": [
            {
                "title": "HashiCorp Terraform Associate",
                "url": "https://www.coursera.org/learn/terraform",
                "estimated_hours": 15,
                "type": "course",
                "relevance_score": 0.9,
            }
        ],
    }

    stub_final_json = json.dumps({
        "job_id": "test-job-001",
        "overall_score": 82,
        "confidence": "high",
        "dimension_scores": {"skills": 90, "experience": 80, "seniority_fit": 75},
        "matched_skills": ["python", "fastapi", "postgresql", "redis", "docker"],
        "gap_skills": ["terraform"],
        "reasoning": "Jane matches 5 of 5 required skills. Seven years experience fits senior seniority. Terraform is the only gap.",
        "learning_plan": [
            {
                "skill": "terraform",
                "priority_rank": 1,
                "estimated_match_gain_pct": 5.0,
                "resources": [
                    {
                        "title": "HashiCorp Terraform Associate",
                        "url": "https://www.coursera.org/learn/terraform",
                        "estimated_hours": 15,
                        "type": "course",
                    }
                ],
                "rationale": "Infrastructure skill to round out senior backend profile",
            }
        ],
    })

    from unittest.mock import MagicMock

    async def fake_run_async(**kwargs):
        from google.adk.events import Event, EventActions
        from google.genai import types

        call_event = Event(
            author="career_intelligence_agent",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id="fc1",
                            name="extract_jd_requirements",
                            args={"job_url_or_text": SAMPLE_JD},
                        )
                    )
                ],
            ),
        )
        yield call_event

        response_event = Event(
            author="career_intelligence_agent",
            content=types.Content(
                role="tool",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="fc1",
                            name="extract_jd_requirements",
                            response=stub_jd.model_dump(),
                        )
                    )
                ],
            ),
        )
        yield response_event

        final_event = Event(
            author="career_intelligence_agent",
            content=types.Content(
                role="model",
                parts=[types.Part(text=stub_final_json)],
            ),
        )
        yield final_event

    runner = AgentRunner()
    runner._runner.run_async = fake_run_async

    result = await runner.run(profile, SAMPLE_JD, job_id="test-job-001")

    assert isinstance(result, MatchResult)
    assert result.overall_score >= 0
    assert result.confidence in ("low", "medium", "high")
    assert result.agent_trace is not None
    assert isinstance(result.agent_trace.tool_calls, list)
    assert result.agent_trace.tool_calls[0].tool == "extract_jd_requirements"
    assert result.agent_trace.tool_calls[0].status == "success"
    print(f"\nSmoke test passed: score={result.overall_score}, confidence={result.confidence}")
    print(f"Trace: {result.agent_trace.model_dump_json(indent=2)}")
