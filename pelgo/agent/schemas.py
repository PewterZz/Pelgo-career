from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class WorkEntry(BaseModel):
    title: str
    company: str
    years: float
    skills_used: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str | None = None
    skills: list[str]
    years_experience: float
    seniority_level: Literal["junior", "mid", "senior", "lead", "staff", "principal"]
    domain: str
    education: list[str] = Field(default_factory=list)
    work_history: list[WorkEntry] = Field(default_factory=list)
    raw_text: str = ""


class JDRequirements(BaseModel):
    required_skills: list[str]
    nice_to_have_skills: list[str]
    seniority_level: str
    domain: str
    responsibilities: list[str]


class DimensionScores(BaseModel):
    skills: int = Field(ge=0, le=100)
    experience: int = Field(ge=0, le=100)
    seniority_fit: int = Field(ge=0, le=100)


class SkillResource(BaseModel):
    title: str
    url: str
    estimated_hours: int
    type: Literal["course", "project", "cert", "doc"]


class PrioritizedSkill(BaseModel):
    skill: str
    priority_rank: int
    estimated_match_gain_pct: float
    rationale: str


class LearningPlanItem(BaseModel):
    skill: str
    priority_rank: int
    estimated_match_gain_pct: float
    resources: list[SkillResource] = Field(default_factory=list)
    rationale: str


class ToolCallRecord(BaseModel):
    tool: str
    status: Literal["success", "error", "timeout", "skipped"]
    latency_ms: int


class AgentTrace(BaseModel):
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    total_llm_calls: int = 0
    fallbacks_triggered: int = 0


class MatchResult(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    overall_score: int = Field(ge=0, le=100)
    confidence: Literal["low", "medium", "high"]
    dimension_scores: DimensionScores
    matched_skills: list[str]
    gap_skills: list[str]
    reasoning: str
    learning_plan: list[LearningPlanItem]
    agent_trace: AgentTrace


class AgentState(BaseModel):
    """Typed state persisted across tool calls in a single agent run."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    candidate_profile: CandidateProfile | None = None
    jd_input: str = ""
    jd_requirements: JDRequirements | None = None
    scoring_result: dict | None = None
    prioritized_gaps: list[PrioritizedSkill] = Field(default_factory=list)
    researched_resources: dict[str, list[SkillResource]] = Field(default_factory=dict)
    low_confidence_retries: int = 0
    tool_errors: list[str] = Field(default_factory=list)
