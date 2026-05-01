"""Smoke test: runs the full agent pipeline with stubbed tools.

Run with: /opt/anaconda3/bin/python -m pytest tests/ -v
"""
from __future__ import annotations

import asyncio
import json
import os as _os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pelgo.agent.schemas import (
    AgentState,
    AgentTrace,
    CandidateProfile,
    DimensionScores,
    JDRequirements,
    LearningPlanItem,
    MatchResult,
    SkillResource,
    ToolCallRecord,
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


# ── AI/ML Engineer JD constants ───────────────────────────────────────────────

AIML_JD = """\
AI/ML Engineer — Role Purpose

Requirements:
- 2-3 years professional software development with AI integration focus
- Strong grasp of LLMs, RAG, Autonomous Agents, Prompt Engineering
- MCP (Model Context Protocol) familiarity highly valued

Stack:
- LangChain, LangGraph, FastAPI
- AWS Bedrock, SageMaker or Vertex AI, Azure AI Foundry
- LangSmith or similar LLM monitoring tools
- Vector DBs: OpenSearch, pgvector
- NoSQL/Relational: DynamoDB

Responsibilities:
- Design and deploy scalable AI-driven solutions (RAG, Agentic workflows)
- Build multi-turn conversational agents and autonomous automation
- Monitor and fine-tune LLMs and ML models in production
- Develop integrations between AI systems and diverse data sources
"""

AIML_JD_REQS = JDRequirements(
    required_skills=[
        "LangChain", "LangGraph", "FastAPI", "LLMs", "RAG",
        "Autonomous Agents", "Prompt Engineering", "Vertex AI",
        "LangSmith", "pgvector", "DynamoDB",
    ],
    nice_to_have_skills=["MCP", "AWS Bedrock", "SageMaker", "OpenSearch"],
    seniority_level="mid",
    domain="AI engineering",
    responsibilities=[
        "Design and deploy RAG and Agentic workflows",
        "Build conversational agents",
        "Monitor and fine-tune LLMs",
        "Develop data integrations",
    ],
)

AIML_STRONG_PROFILE = CandidateProfile(
    candidate_id="test-aiml-strong-001",
    name="Alex Kim",
    email="alex@example.com",
    skills=[
        "python", "langchain", "langgraph", "fastapi", "rag", "llms",
        "prompt engineering", "vertex ai", "langsmith", "pgvector",
        "dynamodb", "autonomous agents", "mcp",
    ],
    years_experience=3.0,
    seniority_level="mid",
    domain="AI engineering",
    education=["B.S. Computer Science"],
)

AIML_WEAK_PROFILE = CandidateProfile(
    candidate_id="test-aiml-weak-001",
    name="Bob Johnson",
    email="bob@example.com",
    skills=["java", "spring boot", "mysql", "rest apis"],
    years_experience=2.0,
    seniority_level="junior",
    domain="backend engineering",
    education=["B.S. Computer Science"],
)

_REAL_RESUME_PATH = _os.path.join(
    _os.path.dirname(__file__),
    "../resumes/Peter_Nelson_Subrata_s_Resume (12).pdf",
)


# ── LLM configuration ─────────────────────────────────────────────────────────

def test_temperature_is_zero_on_all_agents():
    from pelgo.agent.agent import build_agent
    from pelgo.agent.tools import _SKILL_RESEARCH_SUB_AGENT

    orchestrator = build_agent()
    assert orchestrator.generate_content_config.temperature == 0.0, (
        "Orchestrator must use temperature=0.0 for deterministic output"
    )
    assert _SKILL_RESEARCH_SUB_AGENT.generate_content_config.temperature == 0.0, (
        "Research sub-agent must use temperature=0.0 for deterministic output"
    )


# ── Scoring unit tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_score_aiml_strong_match():
    from pelgo.agent.tools import score_candidate_against_requirements

    ctx = MagicMock()
    ctx.state = {}
    result = await score_candidate_against_requirements(
        candidate_profile_json=AIML_STRONG_PROFILE.model_dump_json(),
        requirements_json=AIML_JD_REQS.model_dump_json(),
        tool_context=ctx,
    )

    assert result["overall_score"] == 100
    assert result["confidence"] == "high"
    assert set(result["matched_skills"]) == set(AIML_JD_REQS.required_skills)
    assert result["gap_skills"] == []
    assert ctx.state["scoring_result"] == result


@pytest.mark.asyncio
async def test_score_aiml_weak_match():
    from pelgo.agent.tools import score_candidate_against_requirements

    ctx = MagicMock()
    ctx.state = {}
    result = await score_candidate_against_requirements(
        candidate_profile_json=AIML_WEAK_PROFILE.model_dump_json(),
        requirements_json=AIML_JD_REQS.model_dump_json(),
        tool_context=ctx,
    )

    assert result["overall_score"] < 40
    assert result["matched_skills"] == []
    assert set(result["gap_skills"]) == set(AIML_JD_REQS.required_skills)


@pytest.mark.asyncio
async def test_skill_matching_fuzzy():
    """Plural forms and compound variants in candidate skills still match JD."""
    from pelgo.agent.tools import score_candidate_against_requirements

    edge_profile = CandidateProfile(
        candidate_id="test-edge-001",
        name="Edge Case",
        email="edge@example.com",
        skills=["llms", "vertex ai/gcp", "autonomous agents"],
        years_experience=3.0,
        seniority_level="mid",
        domain="AI engineering",
    )
    edge_reqs = JDRequirements(
        required_skills=["LLMs", "Vertex AI", "Autonomous Agents"],
        nice_to_have_skills=[],
        seniority_level="mid",
        domain="AI engineering",
        responsibilities=[],
    )

    ctx = MagicMock()
    ctx.state = {}
    result = await score_candidate_against_requirements(
        candidate_profile_json=edge_profile.model_dump_json(),
        requirements_json=edge_reqs.model_dump_json(),
        tool_context=ctx,
    )

    assert set(result["matched_skills"]) == {"LLMs", "Vertex AI", "Autonomous Agents"}
    assert result["gap_skills"] == []


@pytest.mark.asyncio
async def test_score_requirements_envelope_unwrapping():
    """Model sometimes passes requirements wrapped as {"extract_jd_requirements_response": {...}}.
    The tool must unwrap and still produce correct scores."""
    from pelgo.agent.tools import score_candidate_against_requirements
    import json as _json

    wrapped_reqs = _json.dumps({
        "extract_jd_requirements_response": AIML_JD_REQS.model_dump()
    })

    ctx = MagicMock()
    ctx.state = {}
    result = await score_candidate_against_requirements(
        candidate_profile_json=AIML_STRONG_PROFILE.model_dump_json(),
        requirements_json=wrapped_reqs,
        tool_context=ctx,
    )

    assert result["overall_score"] == 100
    assert result["confidence"] == "high"
    assert set(result["matched_skills"]) == set(AIML_JD_REQS.required_skills)


@pytest.mark.asyncio
async def test_scoring_stable_under_skill_reordering():
    """Shuffling candidate.skills list must not change score or match sets."""
    from pelgo.agent.tools import score_candidate_against_requirements

    skills = AIML_STRONG_PROFILE.skills.copy()
    profile_a = AIML_STRONG_PROFILE.model_copy(update={"skills": skills})
    profile_b = AIML_STRONG_PROFILE.model_copy(update={"skills": list(reversed(skills))})

    results = []
    for profile in (profile_a, profile_b):
        ctx = MagicMock()
        ctx.state = {}
        results.append(
            await score_candidate_against_requirements(
                candidate_profile_json=profile.model_dump_json(),
                requirements_json=AIML_JD_REQS.model_dump_json(),
                tool_context=ctx,
            )
        )

    assert results[0]["overall_score"] == results[1]["overall_score"]
    assert results[0]["confidence"] == results[1]["confidence"]
    assert set(results[0]["matched_skills"]) == set(results[1]["matched_skills"])
    assert set(results[0]["gap_skills"]) == set(results[1]["gap_skills"])


# ── Full pipeline with AI/ML JD ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_agent_pipeline_aiml_jd():
    """Stubbed end-to-end run with the AI/ML Engineer JD and a strong candidate."""
    from pelgo.agent.runner import AgentRunner

    stub_final_json = json.dumps({
        "job_id": "aiml-test-001",
        "overall_score": 79,
        "confidence": "high",
        "dimension_scores": {"skills": 91, "experience": 67, "seniority_fit": 85},
        "matched_skills": [
            "LangChain", "LangGraph", "FastAPI", "LLMs", "RAG",
            "Autonomous Agents", "Prompt Engineering", "Vertex AI",
            "LangSmith", "pgvector",
        ],
        "gap_skills": ["DynamoDB"],
        "reasoning": (
            "Strong AI/ML stack alignment with 10 of 11 required skills. "
            "Only DynamoDB is missing from the candidate profile."
        ),
        "learning_plan": [
            {
                "skill": "DynamoDB",
                "priority_rank": 1,
                "estimated_match_gain_pct": 9.0,
                "resources": [
                    {
                        "title": "AWS DynamoDB Developer Guide",
                        "url": "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/",
                        "estimated_hours": 12,
                        "type": "doc",
                    }
                ],
                "rationale": "Required NoSQL DB listed explicitly in the job stack.",
            }
        ],
    })

    async def fake_run_async(**kwargs):
        from google.adk.events import Event
        from google.genai import types

        yield Event(
            author="career_intelligence_agent",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id="fc1",
                            name="extract_jd_requirements",
                            args={"job_url_or_text": AIML_JD},
                        )
                    )
                ],
            ),
        )
        yield Event(
            author="career_intelligence_agent",
            content=types.Content(
                role="tool",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="fc1",
                            name="extract_jd_requirements",
                            response=AIML_JD_REQS.model_dump(),
                        )
                    )
                ],
            ),
        )
        yield Event(
            author="career_intelligence_agent",
            content=types.Content(
                role="model",
                parts=[types.Part(text=stub_final_json)],
            ),
        )

    runner = AgentRunner()
    runner._runner.run_async = fake_run_async
    result = await runner.run(AIML_STRONG_PROFILE, AIML_JD, job_id="aiml-test-001")

    assert isinstance(result, MatchResult)
    assert result.overall_score == 79
    assert result.confidence == "high"
    assert "DynamoDB" in result.gap_skills
    assert "LangChain" in result.matched_skills
    assert len(result.learning_plan) == 1
    assert result.learning_plan[0].skill == "DynamoDB"
    assert result.agent_trace.tool_calls[0].tool == "extract_jd_requirements"
    assert result.agent_trace.tool_calls[0].status == "success"
    print(f"\nAI/ML pipeline test passed: score={result.overall_score}, gap={result.gap_skills}")


# ── Real Gemini integration tests (opt-in) ────────────────────────────────────
# Run with: GOOGLE_CLOUD_PROJECT=... GOOGLE_GENAI_USE_VERTEXAI=1 pytest tests/ -v -k real

_needs_gemini = pytest.mark.skipif(
    not _os.getenv("GOOGLE_CLOUD_PROJECT") or not _os.getenv("GOOGLE_GENAI_USE_VERTEXAI"),
    reason="Requires GOOGLE_CLOUD_PROJECT and GOOGLE_GENAI_USE_VERTEXAI",
)
_needs_pdf = pytest.mark.skipif(
    not _os.path.exists(_REAL_RESUME_PATH)
    or not _os.getenv("GOOGLE_CLOUD_PROJECT")
    or not _os.getenv("GOOGLE_GENAI_USE_VERTEXAI"),
    reason="Requires real PDF, GOOGLE_CLOUD_PROJECT, and GOOGLE_GENAI_USE_VERTEXAI",
)


@_needs_pdf
def test_real_pdf_parsing():
    """Live integration: parse the actual resume PDF and verify basic fields."""
    from pelgo.pdf_extractor import parse_resume_pdf

    profile = parse_resume_pdf(_REAL_RESUME_PATH)

    assert profile.name and profile.name != "Unknown Candidate"
    assert profile.seniority_level in ("junior", "mid", "senior", "lead", "staff", "principal")
    assert profile.years_experience > 0
    assert len(profile.skills) > 0
    print(
        f"\nReal PDF: name={profile.name!r}, seniority={profile.seniority_level}, "
        f"years={profile.years_experience}, skills={profile.skills}"
    )


@_needs_gemini
@pytest.mark.asyncio
async def test_real_extract_jd_requirements():
    """Live: extract structured requirements from the AI/ML JD via Gemini."""
    from pelgo.agent.tools import extract_jd_requirements

    ctx = MagicMock()
    ctx.state = {}
    result = await extract_jd_requirements(AIML_JD, ctx)

    assert "required_skills" in result, "required_skills missing from extraction"
    assert len(result["required_skills"]) >= 5, "expected at least 5 required skills"
    required_lower = [s.lower() for s in result["required_skills"]]
    assert any("langchain" in s for s in required_lower), "LangChain not extracted"
    assert any("rag" in s or "retrieval" in s for s in required_lower), "RAG not extracted"
    assert result["seniority_level"] in ("junior", "mid", "senior", "lead", "staff", "principal")
    assert result.get("domain"), "domain should be non-empty"
    print(f"\nExtracted JD: {result}")


@_needs_gemini
@pytest.mark.asyncio
async def test_real_prioritise_skill_gaps():
    """Live: Gemini ranks gap skills by expected match gain."""
    from pelgo.agent.tools import prioritise_skill_gaps

    gap_skills = ["LangChain", "RAG", "pgvector", "DynamoDB", "LangSmith"]
    ctx = MagicMock()
    ctx.state = {
        "jd_requirements": AIML_JD_REQS.model_dump(),
        "scoring_result": {"overall_score": 46, "confidence": "medium"},
    }
    result = await prioritise_skill_gaps(
        gap_skills_json=json.dumps(gap_skills),
        job_market_context="AI engineering",
        tool_context=ctx,
    )

    assert "prioritized_skills" in result
    assert len(result["prioritized_skills"]) == len(gap_skills)
    ranks = sorted(p["priority_rank"] for p in result["prioritized_skills"])
    assert ranks == list(range(1, len(gap_skills) + 1)), "ranks must be 1..N with no duplicates"
    for p in result["prioritized_skills"]:
        assert p["rationale"], "each skill needs a rationale"
        assert p["estimated_match_gain_pct"] > 0
    print(f"\nPrioritized gaps: {[(p['skill'], p['priority_rank']) for p in result['prioritized_skills']]}")


@_needs_gemini
@pytest.mark.asyncio
async def test_real_full_pipeline_aiml():
    """Live end-to-end: full agent run against the AI/ML Engineer JD."""
    from pelgo.agent.runner import AgentRunner

    runner = AgentRunner()
    result = await runner.run(AIML_STRONG_PROFILE, AIML_JD)

    assert isinstance(result, MatchResult)
    assert result.overall_score > 0, "score should be non-zero for a strong AI/ML candidate"
    assert result.confidence in ("low", "medium", "high")
    assert len(result.matched_skills) > 0, "strong candidate should match at least some skills"
    assert result.agent_trace.total_llm_calls >= 3
    tool_names = [t.tool for t in result.agent_trace.tool_calls]
    assert "extract_jd_requirements" in tool_names
    assert "score_candidate_against_requirements" in tool_names
    assert "prioritise_skill_gaps" in tool_names
    assert result.agent_trace.fallbacks_triggered == 0, "no tool errors expected"
    print(
        f"\nFull pipeline: score={result.overall_score}, confidence={result.confidence}, "
        f"matched={result.matched_skills}, top_gap={result.gap_skills[:3]}"
    )
    print(f"Trace: {result.agent_trace.model_dump_json(indent=2)}")


@_needs_gemini
@pytest.mark.asyncio
async def test_real_full_pipeline_weak_candidate():
    """Live: weak Java dev profile should score low and produce a substantial learning plan."""
    from pelgo.agent.runner import AgentRunner

    runner = AgentRunner()
    result = await runner.run(AIML_WEAK_PROFILE, AIML_JD)

    assert result.overall_score < 50, f"Expected score < 50 for weak candidate, got {result.overall_score}"
    assert len(result.gap_skills) >= 5, f"Expected >= 5 gap skills, got {result.gap_skills}"
    assert len(result.learning_plan) >= 3, f"Expected >= 3 learning plan items, got {len(result.learning_plan)}"
    for item in result.learning_plan:
        assert len(item.resources) >= 1, f"Learning plan item {item.skill!r} has no resources"
    print(
        f"\nWeak candidate: score={result.overall_score}, gaps={result.gap_skills}, "
        f"plan_items={len(result.learning_plan)}"
    )


@_needs_gemini
@pytest.mark.asyncio
async def test_real_tool_call_sequence():
    """Live: verify tools appear in the mandatory order in agent_trace."""
    from pelgo.agent.runner import AgentRunner

    runner = AgentRunner()
    result = await runner.run(AIML_STRONG_PROFILE, AIML_JD)

    tool_names = [t.tool for t in result.agent_trace.tool_calls]
    extract_idx = next((i for i, n in enumerate(tool_names) if n == "extract_jd_requirements"), None)
    score_idx = next((i for i, n in enumerate(tool_names) if n == "score_candidate_against_requirements"), None)
    prioritise_idx = next((i for i, n in enumerate(tool_names) if n == "prioritise_skill_gaps"), None)
    research_indices = [i for i, n in enumerate(tool_names) if n == "research_skill_resources"]

    assert extract_idx is not None, f"extract_jd_requirements not in trace: {tool_names}"
    assert score_idx is not None, f"score_candidate_against_requirements not in trace: {tool_names}"
    assert prioritise_idx is not None, f"prioritise_skill_gaps not in trace: {tool_names}"
    assert extract_idx < score_idx, "extract_jd_requirements must precede score_candidate_against_requirements"
    assert score_idx < prioritise_idx, "score_candidate_against_requirements must precede prioritise_skill_gaps"
    if research_indices:
        assert prioritise_idx < min(research_indices), "prioritise_skill_gaps must precede research_skill_resources"
    print(f"\nTool call order: {tool_names}")


@_needs_gemini
@pytest.mark.asyncio
async def test_real_resources_from_external_api():
    """Live: at least one resource URL should come from coursera.org or github.com."""
    from pelgo.agent.runner import AgentRunner

    runner = AgentRunner()
    result = await runner.run(AIML_STRONG_PROFILE, AIML_JD)

    all_urls = [res.url for item in result.learning_plan for res in item.resources]
    assert any(
        "coursera.org" in url or "github.com" in url or "huggingface.co" in url
        for url in all_urls
    ), f"Expected at least one external API URL (coursera.org, github.com, or huggingface.co), got: {all_urls}"
    external = [u for u in all_urls if any(d in u for d in ("coursera.org", "github.com", "huggingface.co"))]
    print(f"\nExternal API URLs: {external}")


@_needs_gemini
@_needs_pdf
@pytest.mark.asyncio
async def test_real_full_pipeline_from_pdf():
    """Live: parse real resume PDF and run through the full agent pipeline."""
    from pelgo.agent.runner import AgentRunner
    from pelgo.pdf_extractor import parse_resume_pdf

    profile = parse_resume_pdf(_REAL_RESUME_PATH)
    runner = AgentRunner()
    result = await runner.run(profile, AIML_JD)

    assert result.overall_score > 0, "Score should be non-zero for a real candidate"
    assert result.agent_trace.total_llm_calls >= 3
    assert len(result.learning_plan) >= 1, "Expected at least one learning plan item"
    print(
        f"\nPDF pipeline: score={result.overall_score}, confidence={result.confidence}, "
        f"matched={result.matched_skills}, plan_items={len(result.learning_plan)}"
    )


@_needs_gemini
@pytest.mark.asyncio
async def test_real_pipeline_determinism():
    """Live: two runs with the same input should produce structurally consistent results."""
    from pelgo.agent.runner import AgentRunner

    results = []
    for _ in range(2):
        runner = AgentRunner()
        results.append(await runner.run(AIML_STRONG_PROFILE, AIML_JD, job_id="det-real-001"))

    r1, r2 = results
    # Neither run should have fallen back to the empty result
    assert r1.overall_score > 0 and r2.overall_score > 0, (
        f"One run fell back to empty result: {r1.overall_score} vs {r2.overall_score}"
    )
    assert abs(r1.overall_score - r2.overall_score) <= 20, (
        f"Scores diverged too much: {r1.overall_score} vs {r2.overall_score}"
    )
    # Confidence labels may differ by one tier (high/medium) due to LLM synthesis — that's acceptable.
    _tiers = {"low": 0, "medium": 1, "high": 2}
    assert abs(_tiers[r1.confidence] - _tiers[r2.confidence]) <= 1, (
        f"Confidence diverged by more than one tier: {r1.confidence} vs {r2.confidence}"
    )
    common_matched = set(r1.matched_skills) & set(r2.matched_skills)
    union_matched = set(r1.matched_skills) | set(r2.matched_skills)
    if union_matched:
        overlap = len(common_matched) / len(union_matched)
        assert overlap >= 0.5, (
            f"Matched skills overlap too low: {overlap:.0%} — {r1.matched_skills} vs {r2.matched_skills}"
        )
    print(
        f"\nRun 1: score={r1.overall_score}, confidence={r1.confidence}, matched={r1.matched_skills}\n"
        f"Run 2: score={r2.overall_score}, confidence={r2.confidence}, matched={r2.matched_skills}\n"
        f"Overlap: {len(common_matched)}/{len(union_matched)}"
    )


# ── Failure / resilience tests ────────────────────────────────────────────────
# Helpers

import httpx as _httpx


class _TimeoutClient:
    """Mock httpx.AsyncClient that always raises TimeoutException."""
    def __init__(self, **kwargs): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def get(self, *args, **kwargs):
        raise _httpx.TimeoutException("simulated timeout")


class _EmptyResponseClient:
    """Mock httpx.AsyncClient that returns 200 with empty data structures."""
    def __init__(self, **kwargs): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def get(self, url, **kwargs):
        class _Resp:
            status_code = 200
            def json(self_inner):
                if "coursera" in url:
                    return {"elements": []}
                if "duckduckgo" in url:
                    return {"RelatedTopics": []}
                return {}
            def raise_for_status(self_inner): pass
        return _Resp()


class _CourseraFailsDDGSuccessClient:
    """Coursera returns 500; DDG returns one valid topic."""
    def __init__(self, **kwargs): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def get(self, url, **kwargs):
        class _Resp:
            def raise_for_status(self_inner): pass
            def json(self_inner): return self_inner._data
        r = _Resp()
        if "coursera" in url:
            r.status_code = 500
            r._data = {}
        else:
            r.status_code = 200
            r._data = {"RelatedTopics": [{"FirstURL": "https://ddg.example.com/langchain", "Text": "LangChain tutorial"}]}
        return r


# ── Tool: network timeouts ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_research_skill_resources_cache_hit():
    """Second call with same skill returns cached result without any HTTP requests."""
    from pelgo.agent import tools as _tools
    from pelgo.agent.tools import research_skill_resources

    _tools._RESOURCE_CACHE.clear()

    http_call_count = 0

    class _CountingClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url, **kwargs):
            nonlocal http_call_count
            http_call_count += 1
            class _Resp:
                status_code = 200
                def json(self_inner):
                    return {"elements": [{"slug": "test-langchain", "name": "LangChain Fundamentals"}]}
                def raise_for_status(self_inner): pass
            return _Resp()

    with patch("pelgo.agent.tools.httpx.AsyncClient", _CountingClient):
        ctx1 = MagicMock()
        ctx1.state = {}
        result1 = await research_skill_resources("TestSkillCache", "mid", ctx1)

        ctx2 = MagicMock()
        ctx2.state = {}
        result2 = await research_skill_resources("TestSkillCache", "mid", ctx2)

    assert result1["resources"] == result2["resources"]
    assert http_call_count == 1, f"Expected 1 HTTP call (cache hit on second), got {http_call_count}"


@pytest.mark.asyncio
async def test_research_all_apis_timeout_returns_fallback():
    """All HTTP calls time out → tool returns the hardcoded Google search fallback."""
    from pelgo.agent.tools import research_skill_resources

    ctx = MagicMock()
    ctx.state = {}
    with patch("pelgo.agent.tools.httpx.AsyncClient", _TimeoutClient):
        result = await research_skill_resources("LangChain", "mid", ctx)

    assert len(result["resources"]) >= 1
    fallback = result["resources"][-1]
    assert "google.com/search" in fallback["url"]
    assert fallback["type"] == "doc"


@pytest.mark.asyncio
async def test_research_coursera_empty_falls_back_to_ddg():
    """Coursera returns 500 → DDG fallback produces resources."""
    from pelgo.agent.tools import research_skill_resources

    ctx = MagicMock()
    ctx.state = {}
    with patch("pelgo.agent.tools.httpx.AsyncClient", _CourseraFailsDDGSuccessClient):
        result = await research_skill_resources("LangChain", "mid", ctx)

    urls = [r["url"] for r in result["resources"]]
    assert any("ddg.example.com" in u for u in urls), f"Expected DDG url, got: {urls}"


@pytest.mark.asyncio
async def test_research_all_apis_empty_returns_google_fallback():
    """All APIs return empty data → hardcoded Google search URL is the fallback."""
    from pelgo.agent.tools import research_skill_resources

    ctx = MagicMock()
    ctx.state = {}
    with patch("pelgo.agent.tools.httpx.AsyncClient", _EmptyResponseClient):
        result = await research_skill_resources("pgvector", "senior", ctx)

    assert len(result["resources"]) == 1
    assert "google.com/search" in result["resources"][0]["url"]
    assert "pgvector" in result["resources"][0]["url"]


# ── Tool: URL fetch failure ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extract_jd_url_timeout_records_error_and_falls_back():
    """URL fetch timeout is caught, recorded in tool_errors, and extraction continues on raw text."""
    from pelgo.agent.tools import extract_jd_requirements

    stub_reqs = JDRequirements(
        required_skills=["Python", "FastAPI"],
        nice_to_have_skills=[],
        seniority_level="mid",
        domain="backend engineering",
        responsibilities=["Build APIs"],
    )
    ctx = MagicMock()
    ctx.state = {}
    with patch("pelgo.agent.tools.httpx.AsyncClient", _TimeoutClient), \
         patch("pelgo.agent.tools._gemini_extract", return_value=stub_reqs.model_dump_json()):
        result = await extract_jd_requirements("https://example.com/job", ctx)

    assert "required_skills" in result
    assert "tool_errors" in ctx.state
    assert any("URL fetch failed" in e for e in ctx.state["tool_errors"])


@pytest.mark.asyncio
async def test_extract_jd_gemini_parse_fails_returns_empty_requirements():
    """If Gemini returns garbage JSON, extract_jd falls back to empty requirements."""
    from pelgo.agent.tools import extract_jd_requirements

    ctx = MagicMock()
    ctx.state = {}
    with patch("pelgo.agent.tools._gemini_extract", return_value="not valid json at all"):
        result = await extract_jd_requirements("Job: needs Python", ctx)

    assert result["required_skills"] == []
    assert result["seniority_level"] == "mid"
    assert result["domain"] == "unknown"
    assert any("parse failed" in e for e in ctx.state.get("tool_errors", []))


# ── Tool: invalid inputs ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_score_malformed_candidate_json_returns_error():
    from pelgo.agent.tools import score_candidate_against_requirements

    ctx = MagicMock()
    ctx.state = {}
    result = await score_candidate_against_requirements(
        candidate_profile_json="not json {{{",
        requirements_json=AIML_JD_REQS.model_dump_json(),
        tool_context=ctx,
    )
    assert "error" in result
    assert "JSON parse failed" in result["error"]


@pytest.mark.asyncio
async def test_score_malformed_requirements_json_returns_error():
    from pelgo.agent.tools import score_candidate_against_requirements

    ctx = MagicMock()
    ctx.state = {}
    result = await score_candidate_against_requirements(
        candidate_profile_json=AIML_STRONG_PROFILE.model_dump_json(),
        requirements_json="{'bad': 'quotes'}",
        tool_context=ctx,
    )
    assert "error" in result


@pytest.mark.asyncio
async def test_prioritise_empty_gaps_returns_no_llm_call():
    """Empty gap list returns immediately without calling Gemini."""
    from pelgo.agent.tools import prioritise_skill_gaps

    ctx = MagicMock()
    ctx.state = {}
    with patch("pelgo.agent.tools._gemini_extract", side_effect=AssertionError("should not be called")):
        result = await prioritise_skill_gaps("[]", "AI engineering", ctx)

    assert result == {"prioritized_skills": []}


@pytest.mark.asyncio
async def test_prioritise_llm_failure_uses_deterministic_fallback():
    """If Gemini call fails, prioritise_skill_gaps falls back to index-based ranking."""
    from pelgo.agent.tools import prioritise_skill_gaps

    gap_skills = ["LangChain", "RAG", "DynamoDB"]
    ctx = MagicMock()
    ctx.state = {}
    with patch("pelgo.agent.tools._gemini_extract", side_effect=Exception("network error")):
        result = await prioritise_skill_gaps(json.dumps(gap_skills), "AI engineering", ctx)

    items = result["prioritized_skills"]
    assert len(items) == 3
    skills_out = [p["skill"] for p in sorted(items, key=lambda x: x["priority_rank"])]
    assert skills_out == gap_skills
    for p in items:
        assert p["estimated_match_gain_pct"] > 0
        assert p["rationale"]


# ── Runner: malformed / invalid final output ───────────────────────────────────

@pytest.mark.asyncio
async def test_runner_final_text_without_json_returns_empty_result():
    """Agent emits plain text (no JSON braces) as final output → _empty_result returned."""
    from pelgo.agent.runner import AgentRunner

    async def fake_run(**kwargs):
        from google.adk.events import Event
        from google.genai import types
        yield Event(
            author="career_intelligence_agent",
            content=types.Content(role="model", parts=[types.Part(text="I have reviewed the job and the candidate looks good.")]),
        )

    runner = AgentRunner()
    runner._runner.run_async = fake_run
    result = await runner.run(AIML_STRONG_PROFILE, AIML_JD, job_id="empty-001")

    assert result.overall_score == 0
    assert result.confidence == "low"
    assert result.reasoning == "Agent run produced no valid output."
    assert result.agent_trace.total_llm_calls == 0


@pytest.mark.asyncio
async def test_runner_final_json_with_schema_violation_returns_empty_result():
    """Agent returns parseable JSON that fails Pydantic validation → _empty_result returned."""
    from pelgo.agent.runner import AgentRunner

    bad_json = json.dumps({"overall_score": "not-a-number", "confidence": "high"})

    async def fake_run(**kwargs):
        from google.adk.events import Event
        from google.genai import types
        yield Event(
            author="career_intelligence_agent",
            content=types.Content(role="model", parts=[types.Part(text=bad_json)]),
        )

    runner = AgentRunner()
    runner._runner.run_async = fake_run
    result = await runner.run(AIML_STRONG_PROFILE, AIML_JD, job_id="schema-err-001")

    assert result.overall_score == 0
    assert result.confidence == "low"


@pytest.mark.asyncio
async def test_runner_tool_error_response_increments_fallbacks():
    """A tool returning {"error": "..."} is recorded in the trace as error + fallback."""
    from pelgo.agent.runner import AgentRunner

    stub_final = json.dumps({
        "job_id": "tool-err-001",
        "overall_score": 30,
        "confidence": "low",
        "dimension_scores": {"skills": 30, "experience": 30, "seniority_fit": 30},
        "matched_skills": [],
        "gap_skills": ["LangChain"],
        "reasoning": "Tool error encountered; partial result.",
        "learning_plan": [],
    })

    async def fake_run(**kwargs):
        from google.adk.events import Event
        from google.genai import types
        yield Event(
            author="career_intelligence_agent",
            content=types.Content(role="model", parts=[
                types.Part(function_call=types.FunctionCall(id="fc1", name="extract_jd_requirements", args={"job_url_or_text": AIML_JD}))
            ]),
        )
        yield Event(
            author="career_intelligence_agent",
            content=types.Content(role="tool", parts=[
                types.Part(function_response=types.FunctionResponse(
                    id="fc1",
                    name="extract_jd_requirements",
                    response={"error": "Gemini rate limit hit"},
                ))
            ]),
        )
        yield Event(
            author="career_intelligence_agent",
            content=types.Content(role="model", parts=[types.Part(text=stub_final)]),
        )

    runner = AgentRunner()
    runner._runner.run_async = fake_run
    result = await runner.run(AIML_STRONG_PROFILE, AIML_JD, job_id="tool-err-001")

    assert result.agent_trace.fallbacks_triggered == 1
    error_calls = [t for t in result.agent_trace.tool_calls if t.status == "error"]
    assert len(error_calls) == 1
    assert error_calls[0].tool == "extract_jd_requirements"


@pytest.mark.asyncio
async def test_runner_low_confidence_output_is_accepted():
    """Agent producing low confidence score is returned as-is (not rejected or retried by runner)."""
    from pelgo.agent.runner import AgentRunner

    low_conf_json = json.dumps({
        "job_id": "low-conf-001",
        "overall_score": 18,
        "confidence": "low",
        "dimension_scores": {"skills": 10, "experience": 30, "seniority_fit": 25},
        "matched_skills": ["Python"],
        "gap_skills": ["LangChain", "RAG", "pgvector"],
        "reasoning": "Candidate has minimal overlap with the required AI/ML stack.",
        "learning_plan": [],
    })

    async def fake_run(**kwargs):
        from google.adk.events import Event
        from google.genai import types
        yield Event(
            author="career_intelligence_agent",
            content=types.Content(role="model", parts=[types.Part(text=low_conf_json)]),
        )

    runner = AgentRunner()
    runner._runner.run_async = fake_run
    result = await runner.run(AIML_WEAK_PROFILE, AIML_JD, job_id="low-conf-001")

    assert result.overall_score == 18
    assert result.confidence == "low"
    assert "Python" in result.matched_skills


@pytest.mark.asyncio
async def test_runner_multiple_tool_errors_all_recorded():
    """Multiple tool errors accumulate in the trace; fallbacks_triggered counts each one."""
    from pelgo.agent.runner import AgentRunner

    stub_final = json.dumps({
        "job_id": "multi-err-001",
        "overall_score": 5,
        "confidence": "low",
        "dimension_scores": {"skills": 5, "experience": 10, "seniority_fit": 5},
        "matched_skills": [],
        "gap_skills": ["LangChain", "RAG"],
        "reasoning": "Multiple tool failures.",
        "learning_plan": [],
    })

    async def fake_run(**kwargs):
        from google.adk.events import Event
        from google.genai import types
        for fc_id, tool_name in [("fc1", "extract_jd_requirements"), ("fc2", "score_candidate_against_requirements")]:
            yield Event(
                author="career_intelligence_agent",
                content=types.Content(role="model", parts=[
                    types.Part(function_call=types.FunctionCall(id=fc_id, name=tool_name, args={}))
                ]),
            )
            yield Event(
                author="career_intelligence_agent",
                content=types.Content(role="tool", parts=[
                    types.Part(function_response=types.FunctionResponse(
                        id=fc_id, name=tool_name,
                        response={"error": f"{tool_name} failed"},
                    ))
                ]),
            )
        yield Event(
            author="career_intelligence_agent",
            content=types.Content(role="model", parts=[types.Part(text=stub_final)]),
        )

    runner = AgentRunner()
    runner._runner.run_async = fake_run
    result = await runner.run(AIML_WEAK_PROFILE, AIML_JD, job_id="multi-err-001")

    assert result.agent_trace.fallbacks_triggered == 2
    assert len([t for t in result.agent_trace.tool_calls if t.status == "error"]) == 2


# ── Output audit ──────────────────────────────────────────────────────────────

def _audit_match_result(result: MatchResult) -> None:
    """Comprehensive structural invariants every MatchResult must satisfy."""
    assert isinstance(result, MatchResult)
    assert 0 <= result.overall_score <= 100, f"score out of range: {result.overall_score}"
    assert result.confidence in ("low", "medium", "high"), f"bad confidence: {result.confidence}"
    assert result.reasoning.strip(), "reasoning must not be blank"
    assert result.job_id, "job_id must not be blank"

    ds = result.dimension_scores
    assert 0 <= ds.skills <= 100
    assert 0 <= ds.experience <= 100
    assert 0 <= ds.seniority_fit <= 100

    all_required = set(AIML_JD_REQS.required_skills)
    declared = set(result.matched_skills) | set(result.gap_skills)
    # matched and gap must be disjoint
    assert not (set(result.matched_skills) & set(result.gap_skills)), (
        f"Skills appear in both matched and gap: {set(result.matched_skills) & set(result.gap_skills)}"
    )

    for item in result.learning_plan:
        assert item.skill, "learning plan skill must not be blank"
        assert item.priority_rank >= 1
        assert 0.0 <= item.estimated_match_gain_pct <= 100.0, (
            f"gain_pct out of range: {item.estimated_match_gain_pct}"
        )
        assert item.rationale.strip(), "rationale must not be blank"
        for res in item.resources:
            assert res.title, "resource title must not be blank"
            assert res.url.startswith("http"), f"resource url invalid: {res.url}"
            assert res.estimated_hours > 0, f"estimated_hours must be positive: {res.estimated_hours}"
            assert res.type in ("course", "project", "cert", "doc", "search"), f"bad resource type: {res.type}"

    trace = result.agent_trace
    assert trace.total_llm_calls >= 0
    assert trace.fallbacks_triggered >= 0
    for tc in trace.tool_calls:
        assert tc.tool, "tool name must not be blank"
        assert tc.status in ("success", "error", "timeout", "skipped")
        assert tc.latency_ms >= 0


def test_match_result_full_structural_audit():
    """A fully populated MatchResult passes every structural invariant."""
    result = MatchResult(
        job_id="audit-001",
        overall_score=72,
        confidence="medium",
        dimension_scores=DimensionScores(skills=75, experience=80, seniority_fit=60),
        matched_skills=["LangChain", "RAG", "LLMs"],
        gap_skills=["LangGraph", "pgvector", "DynamoDB"],
        reasoning="Strong LLM and RAG skills but missing graph orchestration and vector DB experience.",
        learning_plan=[
            LearningPlanItem(
                skill="LangGraph",
                priority_rank=1,
                estimated_match_gain_pct=12.5,
                resources=[
                    SkillResource(title="LangGraph Quickstart", url="https://python.langchain.com/docs/langgraph", estimated_hours=8, type="doc"),
                ],
                rationale="Core orchestration framework listed explicitly in the stack.",
            ),
            LearningPlanItem(
                skill="pgvector",
                priority_rank=2,
                estimated_match_gain_pct=8.0,
                resources=[
                    SkillResource(title="pgvector GitHub", url="https://github.com/pgvector/pgvector", estimated_hours=4, type="project"),
                ],
                rationale="Required vector DB for retrieval pipelines.",
            ),
        ],
        agent_trace=AgentTrace(
            tool_calls=[
                ToolCallRecord(tool="extract_jd_requirements", status="success", latency_ms=5200),
                ToolCallRecord(tool="score_candidate_against_requirements", status="success", latency_ms=2),
                ToolCallRecord(tool="prioritise_skill_gaps", status="success", latency_ms=14000),
                ToolCallRecord(tool="research_skill_resources", status="success", latency_ms=6800),
                ToolCallRecord(tool="research_skill_resources", status="success", latency_ms=5400),
            ],
            total_llm_calls=6,
            fallbacks_triggered=0,
        ),
    )
    _audit_match_result(result)


def test_match_result_audit_catches_disjoint_violation():
    """Audit detects when the same skill appears in both matched and gap."""
    result = MatchResult(
        job_id="audit-bad-001",
        overall_score=50,
        confidence="medium",
        dimension_scores=DimensionScores(skills=50, experience=50, seniority_fit=50),
        matched_skills=["LangChain", "RAG"],
        gap_skills=["RAG", "pgvector"],  # RAG in both — violation
        reasoning="Test.",
        learning_plan=[],
        agent_trace=AgentTrace(),
    )
    with pytest.raises(AssertionError, match="both matched and gap"):
        _audit_match_result(result)


def test_match_result_audit_catches_score_out_of_range():
    """Audit detects overall_score outside [0, 100] — Pydantic enforces this at construction."""
    with pytest.raises(Exception):
        MatchResult(
            job_id="audit-bad-002",
            overall_score=150,  # violates ge=0, le=100
            confidence="high",
            dimension_scores=DimensionScores(skills=100, experience=100, seniority_fit=100),
            matched_skills=[],
            gap_skills=[],
            reasoning="Over 100.",
            learning_plan=[],
            agent_trace=AgentTrace(),
        )
