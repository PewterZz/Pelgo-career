from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any
from urllib.parse import quote_plus

import httpx
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types
from pydantic import ValidationError

from .schemas import JDRequirements, PrioritizedSkill, SkillResource

_TOOL_TIMEOUT_SEC = float(os.getenv("TOOL_TIMEOUT_SEC", "30"))
_RESEARCH_TIMEOUT_SEC = float(os.getenv("RESEARCH_TIMEOUT_SEC", "15"))

_RESOURCE_CACHE: dict[tuple[str, str], dict] = {}


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _gemini_extract(prompt: str) -> str:
    """Call Gemini to get a JSON string back."""
    from google import genai

    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "0") == "1"
    client = genai.Client(
        vertexai=use_vertex,
        project=os.getenv("GOOGLE_CLOUD_PROJECT") if use_vertex else None,
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global") if use_vertex else None,
    )
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    return response.text


async def extract_jd_requirements(
    job_url_or_text: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Extract structured requirements from a job description (URL or raw text).

    Returns JDRequirements schema: required_skills, nice_to_have_skills,
    seniority_level, domain, responsibilities.
    """
    raw_text = job_url_or_text
    if job_url_or_text.startswith("http"):
        try:
            async with httpx.AsyncClient(timeout=_TOOL_TIMEOUT_SEC) as client:
                resp = await client.get(
                    job_url_or_text,
                    headers={"User-Agent": "PelgoBot/1.0"},
                    follow_redirects=True,
                )
                resp.raise_for_status()
                raw_text = resp.text[:8000]
        except Exception as exc:
            tool_context.state["tool_errors"] = tool_context.state.get("tool_errors", []) + [
                f"extract_jd_requirements URL fetch failed: {exc}"
            ]
            raw_text = job_url_or_text

    prompt = f"""Extract a structured job description from the text below.
Return ONLY a JSON object with these exact keys:
- required_skills: list of short skill/technology names only (e.g. "Python", "LangChain", "RAG", "FastAPI", "PostgreSQL"). Do NOT include experience requirements like "3+ years of X" — extract just the skill name "X". Do NOT include soft skills or vague requirements.
- nice_to_have_skills: same format, for preferred but optional skills
- seniority_level: one of junior/mid/senior/lead/staff/principal
- domain: primary technical domain (e.g. "backend engineering", "data science", "AI engineering")
- responsibilities: list of 3-7 key responsibilities as short phrases

Job description:
{raw_text[:6000]}
"""
    try:
        raw_json = await asyncio.get_event_loop().run_in_executor(None, _gemini_extract, prompt)
        result = JDRequirements.model_validate_json(raw_json)
    except (ValidationError, Exception) as exc:
        tool_context.state["tool_errors"] = tool_context.state.get("tool_errors", []) + [
            f"extract_jd_requirements parse failed: {exc}"
        ]
        result = JDRequirements(
            required_skills=[],
            nice_to_have_skills=[],
            seniority_level="mid",
            domain="unknown",
            responsibilities=[],
        )

    serialized = result.model_dump()
    tool_context.state["jd_requirements"] = serialized
    return serialized


async def score_candidate_against_requirements(
    candidate_profile_json: str,
    requirements_json: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Score a candidate against job requirements.

    Confidence heuristic:
      - high: jd_completeness >= 0.7 AND match_ratio >= 0.5
      - medium: jd_completeness >= 0.4 OR match_ratio >= 0.3
      - low: otherwise

    jd_completeness = len(required_skills) / 10 (capped at 1.0)
    match_ratio = len(matched_skills) / max(len(required_skills), 1)
    """
    import os as _os, sys as _sys
    if _os.getenv("PELGO_DEBUG") == "1":
        print(f"[score] profile[:200]={str(candidate_profile_json)[:200]!r}", file=_sys.stderr, flush=True)
        print(f"[score] reqs[:200]={str(requirements_json)[:200]!r}", file=_sys.stderr, flush=True)
    try:
        profile = json.loads(candidate_profile_json)
        requirements = json.loads(requirements_json)
        # Unwrap model-side envelope: {"extract_jd_requirements_response": {...}}
        if isinstance(requirements, dict) and "extract_jd_requirements_response" in requirements:
            requirements = requirements["extract_jd_requirements_response"]
        if isinstance(requirements, dict) and "required_skills" not in requirements:
            for v in requirements.values():
                if isinstance(v, dict) and "required_skills" in v:
                    requirements = v
                    break
    except json.JSONDecodeError as exc:
        print(f"[score] JSON PARSE ERROR: {exc}", file=_sys.stderr, flush=True)
        return {"error": f"JSON parse failed: {exc}"}

    candidate_skills_lower = [s.lower() for s in profile.get("skills", [])]
    required = requirements.get("required_skills", [])
    nice_to_have = requirements.get("nice_to_have_skills", [])

    def _skill_matches(jd_skill: str) -> bool:
        jd = jd_skill.lower().rstrip("s")  # strip plural
        for cs in candidate_skills_lower:
            cs_stem = cs.rstrip("s")
            # match if either is a substring of the other (handles "llm"↔"llms", "gcp"↔"vertex ai/gcp")
            if jd in cs_stem or cs_stem in jd:
                return True
        return False

    matched = [s for s in required if _skill_matches(s)]
    gap = [s for s in required if not _skill_matches(s)]

    jd_completeness = min(len(required) / 10.0, 1.0)
    match_ratio = len(matched) / max(len(required), 1)

    candidate_years = float(profile.get("years_experience", 0))
    seniority_map = {"junior": 1, "mid": 3, "senior": 6, "lead": 8, "staff": 10, "principal": 12}
    jd_seniority = requirements.get("seniority_level", "mid").lower()
    required_years = seniority_map.get(jd_seniority, 3)

    skills_score = int(match_ratio * 100)
    exp_score = min(int((candidate_years / max(required_years, 1)) * 100), 100)

    candidate_seniority = profile.get("seniority_level", "mid").lower()
    seniority_diff = abs(
        seniority_map.get(candidate_seniority, 3) - seniority_map.get(jd_seniority, 3)
    )
    seniority_score = max(0, 100 - seniority_diff * 15)

    overall = int(skills_score * 0.5 + exp_score * 0.3 + seniority_score * 0.2)

    if jd_completeness >= 0.7 and match_ratio >= 0.5:
        confidence = "high"
    elif jd_completeness >= 0.4 or match_ratio >= 0.3:
        confidence = "medium"
    else:
        confidence = "low"

    domain_distance = (
        0
        if profile.get("domain", "").lower() == requirements.get("domain", "").lower()
        else 1
    )
    if domain_distance > 0 and confidence == "high":
        confidence = "medium"

    result = {
        "overall_score": overall,
        "confidence": confidence,
        "dimension_scores": {
            "skills": skills_score,
            "experience": exp_score,
            "seniority_fit": seniority_score,
        },
        "matched_skills": matched,
        "gap_skills": gap,
    }
    if _os.getenv("PELGO_DEBUG") == "1":
        print(f"[score] result={json.dumps(result)}", file=_sys.stderr, flush=True)
    tool_context.state["scoring_result"] = result
    return result


async def research_skill_resources(
    skill_name: str,
    seniority_context: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Search the web for learning resources for a specific skill.

    Makes a real external call to the Coursera public API, falling back to
    DuckDuckGo Instant Answers if Coursera is unavailable.
    """
    cache_key = (skill_name.lower(), seniority_context)
    if cache_key in _RESOURCE_CACHE:
        cached = _RESOURCE_CACHE[cache_key]
        existing = tool_context.state.get("researched_resources", {})
        existing[skill_name] = cached["resources"]
        tool_context.state["researched_resources"] = existing
        return cached

    resources: list[dict] = []

    try:
        async with httpx.AsyncClient(timeout=_RESEARCH_TIMEOUT_SEC) as client:
            resp = await client.get(
                "https://api.coursera.org/api/courses.v1",
                params={"q": "search", "query": skill_name, "limit": 5},
            )
            if resp.status_code == 200:
                data = resp.json()
                for course in data.get("elements", []):
                    slug = course.get("slug", "")
                    name = course.get("name", skill_name)
                    resources.append({
                        "title": name,
                        "url": f"https://www.coursera.org/learn/{slug}",
                        "estimated_hours": 20,
                        "type": "course",
                        "relevance_score": 0.8,
                    })
    except Exception:
        pass

    if not resources:
        try:
            async with httpx.AsyncClient(timeout=_RESEARCH_TIMEOUT_SEC) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": f"learn {skill_name} {seniority_context} tutorial course",
                        "format": "json",
                        "no_html": "1",
                        "no_redirect": "1",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for topic in data.get("RelatedTopics", [])[:5]:
                        if isinstance(topic, dict) and topic.get("FirstURL"):
                            resources.append({
                                "title": topic.get("Text", skill_name)[:120],
                                "url": topic["FirstURL"],
                                "estimated_hours": 10,
                                "type": "doc",
                                "relevance_score": 0.6,
                            })
        except Exception:
            pass

    if not resources:
        resources.append({
            "title": f"Official documentation for {skill_name}",
            "url": f"https://www.google.com/search?q={quote_plus(skill_name)}+documentation",
            "estimated_hours": 5,
            "type": "doc",
            "relevance_score": 0.4,
        })

    result = {"skill": skill_name, "resources": resources}
    _RESOURCE_CACHE[cache_key] = result
    existing = tool_context.state.get("researched_resources", {})
    existing[skill_name] = resources
    tool_context.state["researched_resources"] = existing
    return result


async def prioritise_skill_gaps(
    gap_skills_json: str,
    job_market_context: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Rank gap skills by expected match gain using LLM reasoning.

    Ranking is based on: skill frequency in JD, demand signal from job_market_context,
    and estimated match score improvement per skill.
    """
    try:
        gap_skills: list[str] = json.loads(gap_skills_json)
    except json.JSONDecodeError:
        gap_skills = [s.strip() for s in gap_skills_json.strip("[]").split(",") if s.strip()]

    if not gap_skills:
        return {"prioritized_skills": []}

    jd_requirements = tool_context.state.get("jd_requirements", {})
    scoring = tool_context.state.get("scoring_result", {})
    required_skills = jd_requirements.get("required_skills", [])
    total_required = max(len(required_skills), 1)
    gain_per_skill = round(100 / total_required, 1)

    prompt = f"""Rank these skill gaps by priority for a candidate targeting this role.
Consider: which skills appear most in required_skills, which are foundational vs advanced,
and which have highest market demand.

Gap skills: {gap_skills}
Job required skills: {required_skills}
Job market context: {job_market_context}

Return ONLY a JSON array where each element has:
- skill: string
- priority_rank: integer (1 = highest priority)
- estimated_match_gain_pct: float (estimated score gain from acquiring this skill)
- rationale: one-sentence reason for this ranking

Rank all {len(gap_skills)} skills.
"""
    try:
        raw_json = await asyncio.get_event_loop().run_in_executor(None, _gemini_extract, prompt)
        items = json.loads(raw_json)
        prioritized = [PrioritizedSkill(**item) for item in items]
    except Exception:
        prioritized = [
            PrioritizedSkill(
                skill=skill,
                priority_rank=i + 1,
                estimated_match_gain_pct=gain_per_skill,
                rationale=f"Required skill missing from candidate profile (rank {i+1})",
            )
            for i, skill in enumerate(gap_skills)
        ]

    serialized = [p.model_dump() for p in prioritized]
    tool_context.state["prioritized_gaps"] = serialized
    return {"prioritized_skills": serialized}


# ── Stretch: ADK AgentTool — research sub-agent ───────────────────────────────
# research_skill_resources is re-implemented here as an ADK LlmAgent sub-agent
# wired into the main orchestrator via AgentTool. The parent agent calls
# research_skill_resources(request="...") and ADK routes execution to the
# sub-agent, which uses its own FunctionTools to fetch and curate results.


async def _coursera_search(skill_name: str) -> dict[str, Any]:
    """Fetch up to 5 Coursera courses matching skill_name via the public API."""
    results: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=_RESEARCH_TIMEOUT_SEC) as client:
            resp = await client.get(
                "https://api.coursera.org/api/courses.v1",
                params={"q": "search", "query": skill_name, "limit": 5},
            )
            if resp.status_code == 200:
                for course in resp.json().get("elements", []):
                    slug = course.get("slug", "")
                    results.append({
                        "title": course.get("name", skill_name),
                        "url": f"https://www.coursera.org/learn/{slug}",
                        "estimated_hours": 20,
                        "type": "course",
                        "relevance_score": 0.85,
                    })
    except Exception:
        pass
    return {"results": results}


async def _github_search(skill_name: str) -> dict[str, Any]:
    """Search GitHub for awesome lists and tutorial repos for a skill.

    Uses the unauthenticated public API (60 req/hr). Set GITHUB_TOKEN in .env
    to raise the limit to 5 000 req/hr.
    """
    results: list[dict] = []
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient(timeout=_RESEARCH_TIMEOUT_SEC) as client:
            resp = await client.get(
                "https://api.github.com/search/repositories",
                params={
                    "q": f"awesome {skill_name} learning tutorial",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 5,
                },
                headers=headers,
            )
            if resp.status_code == 200:
                for repo in resp.json().get("items", []):
                    stars = repo.get("stargazers_count", 0)
                    results.append({
                        "title": repo.get("full_name", skill_name),
                        "url": repo.get("html_url", ""),
                        "estimated_hours": 15,
                        "type": "project",
                        "relevance_score": round(min(0.5 + stars / 100_000, 0.95), 2),
                    })
    except Exception:
        pass
    return {"results": results}


_HF_KNOWN_COURSES: dict[str, tuple[str, str, int]] = {
    "nlp": ("HuggingFace NLP Course", "https://huggingface.co/learn/nlp-course", 30),
    "transformers": ("HuggingFace NLP Course", "https://huggingface.co/learn/nlp-course", 30),
    "diffusion": ("HuggingFace Diffusion Models Course", "https://huggingface.co/learn/diffusion-course", 20),
    "reinforcement learning": ("HuggingFace Deep RL Course", "https://huggingface.co/learn/deep-rl-course", 40),
    "deep rl": ("HuggingFace Deep RL Course", "https://huggingface.co/learn/deep-rl-course", 40),
    "audio": ("HuggingFace Audio Course", "https://huggingface.co/learn/audio-course", 15),
    "computer vision": ("HuggingFace CV Course", "https://huggingface.co/learn/computer-vision-course", 20),
}


async def _huggingface_search(skill_name: str) -> dict[str, Any]:
    """Search HuggingFace for datasets and check known free course URLs."""
    results: list[dict] = []

    skill_lower = skill_name.lower()
    for keyword, (title, url, hours) in _HF_KNOWN_COURSES.items():
        if keyword in skill_lower:
            results.append({
                "title": title,
                "url": url,
                "estimated_hours": hours,
                "type": "course",
                "relevance_score": 0.95,
            })
            break

    try:
        async with httpx.AsyncClient(timeout=_RESEARCH_TIMEOUT_SEC) as client:
            resp = await client.get(
                "https://huggingface.co/api/datasets",
                params={"search": skill_name, "limit": 4},
            )
            if resp.status_code == 200:
                for ds in resp.json():
                    ds_id = ds.get("id", "")
                    if ds_id:
                        results.append({
                            "title": f"Dataset: {ds_id}",
                            "url": f"https://huggingface.co/datasets/{ds_id}",
                            "estimated_hours": 8,
                            "type": "doc",
                            "relevance_score": 0.7,
                        })
    except Exception:
        pass

    return {"results": results}


async def _ddg_search(skill_name: str, seniority_context: str = "mid") -> dict[str, Any]:
    """Search DuckDuckGo Instant Answers for learning resources (last-resort fallback)."""
    results: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=_RESEARCH_TIMEOUT_SEC) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": f"learn {skill_name} {seniority_context} tutorial course",
                    "format": "json",
                    "no_html": "1",
                    "no_redirect": "1",
                },
            )
            if resp.status_code == 200:
                for topic in resp.json().get("RelatedTopics", [])[:5]:
                    if isinstance(topic, dict) and topic.get("FirstURL"):
                        results.append({
                            "title": topic.get("Text", skill_name)[:120],
                            "url": topic["FirstURL"],
                            "estimated_hours": 10,
                            "type": "doc",
                            "relevance_score": 0.6,
                        })
    except Exception:
        pass
    return {"results": results}


_SKILL_RESEARCH_SUB_AGENT = LlmAgent(
    name="research_skill_resources",
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    generate_content_config=genai_types.GenerateContentConfig(temperature=0.0),
    instruction="""\
You are a Skill Resource Researcher. Given a request with a skill name and seniority level:

1. Call coursera_search — structured online courses.
2. Call github_search — curated awesome lists and tutorial repos ranked by GitHub stars.
3. Call huggingface_search — free HuggingFace courses (best for ML/AI skills) and datasets.
4. If all three return empty results, call ddg_search as a last-resort fallback.
5. Merge, deduplicate by URL, and rank all results by relevance_score descending.
   Adjust relevance_score (0.0–1.0) upward for resources that closely match the
   seniority level and downward for generic or off-topic results.
6. Return ONLY a valid JSON object — no prose before or after:
{
  "skill": "<skill_name>",
  "resources": [
    {
      "title": "...",
      "url": "...",
      "estimated_hours": <int>,
      "type": "course|project|cert|doc",
      "relevance_score": <float 0.0-1.0>
    }
  ]
}
If no resources are found from any source, include one Google search fallback resource.
""",
    tools=[
        FunctionTool(_coursera_search),
        FunctionTool(_github_search),
        FunctionTool(_huggingface_search),
        FunctionTool(_ddg_search),
    ],
)

research_skill_resources_adk = AgentTool(agent=_SKILL_RESEARCH_SUB_AGENT)
