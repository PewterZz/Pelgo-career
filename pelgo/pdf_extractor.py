from __future__ import annotations

import datetime
import os
import re
from typing import Literal

import pypdf

from .agent.schemas import CandidateProfile, WorkEntry

_SENIORITY_LEVELS = ("junior", "mid", "senior", "lead", "staff", "principal")

_SKILL_KEYWORDS = [
    "python", "javascript", "typescript", "java", "go", "rust", "c\\+\\+", "c#",
    "react", "vue", "angular", "nextjs", "node", "fastapi", "django", "flask",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "aws", "gcp", "azure", "kubernetes", "docker", "terraform",
    "machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn",
    "sql", "spark", "kafka", "airflow", "dbt",
    "graphql", "rest", "grpc", "microservices", "ci/cd",
    "llm", "langchain", "openai", "gemini", "rag",
]


def extract_text_from_pdf(path: str) -> str:
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _gemini_seniority(text: str) -> str:
    """Ask Gemini to judge seniority from resume substance."""
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(
        vertexai=bool(os.getenv("GOOGLE_GENAI_USE_VERTEXAI")),
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    )
    prompt = f"""Classify the professional seniority level of this candidate.

Step 1 — Identify post-graduation professional experience only:
  Find roles that are (a) after the candidate finished their most recent degree and (b) not labelled intern, capstone, student, or research assistant. These are the ONLY roles that count toward seniority.

Step 2 — Discount student-era work:
  Any role held while the candidate was still enrolled in a degree program — even if it involved leading a team, publishing papers, or winning competitions — does NOT raise seniority above mid on its own. Impressive student work shows potential, not professional seniority.

Step 3 — Apply the scale to post-graduation work only:
- junior: less than 1 year post-graduation, or purely supervised work
- mid: 1-3 years post-graduation, works independently, owns deliverables end-to-end
- senior: 4+ years post-graduation OR strong 2-3 year track record driving architecture and mentoring
- lead: sustained accountability for a professional team's technical delivery (not a student team)
- staff: cross-org technical leadership in a professional setting
- principal: company-wide strategy

Resume:
{text[:5000]}

Reply with ONLY the one-word level (junior/mid/senior/lead/staff/principal)."""

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=0),
    )
    result = response.text.strip().lower().split()[0]
    if result not in _SENIORITY_LEVELS:
        raise ValueError(f"Gemini returned unexpected seniority: {response.text!r}")
    return result


def _extract_skills(text: str) -> list[str]:
    lower = text.lower()
    return [kw for kw in _SKILL_KEYWORDS if re.search(rf"\b{kw}\b", lower)]


def _extract_years(text: str) -> float:
    current_year = datetime.date.today().year
    lower = text.lower()

    explicit = re.findall(r"(\d+)\+?\s*years?\s+of\s+experience", lower)
    if explicit:
        return float(max(int(m) for m in explicit))

    # Isolate the experience section to avoid education/award date noise
    exp_match = re.search(r"\bexperience\b", lower)
    next_section = re.search(
        r"\b(projects?|publications?|skills?|honors?|awards?|certifications?|open source)\b",
        lower[exp_match.end():] if exp_match else lower,
    )
    if exp_match:
        exp_text = lower[exp_match.end(): exp_match.end() + (next_section.start() if next_section else len(lower))]
    else:
        exp_text = lower

    # Match "Mon. YYYY" patterns and bare years within the experience block
    year_strs = re.findall(
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?\s+(20\d\d)", exp_text
    )
    year_strs += re.findall(r"\b(20\d\d)\b", exp_text)
    year_ints = [int(y) for y in year_strs if 2000 <= int(y) <= current_year]

    if year_ints:
        return float(current_year - min(year_ints))
    return 2.0


def _extract_name(text: str) -> str:
    first_line = text.strip().split("\n")[0].strip()
    if first_line and len(first_line.split()) <= 5:
        return first_line
    return "Unknown Candidate"


def _extract_email(text: str) -> str | None:
    match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None


def _extract_domain(text: str, skills: list[str]) -> str:
    lower = text.lower()
    if any(s in lower for s in ["machine learning", "data science", "pytorch", "tensorflow"]):
        return "data science"
    if any(s in lower for s in ["frontend", "react", "vue", "angular", "css", "html"]):
        return "frontend engineering"
    if any(s in lower for s in ["devops", "kubernetes", "terraform", "infra"]):
        return "platform engineering"
    if any(s in lower for s in ["llm", "langchain", "rag", "openai", "gemini"]):
        return "AI engineering"
    return "software engineering"


def parse_resume_pdf(path: str) -> CandidateProfile:
    """Extract a CandidateProfile from a PDF resume path."""
    text = extract_text_from_pdf(path)
    skills = _extract_skills(text)
    years = _extract_years(text)
    return CandidateProfile(
        name=_extract_name(text),
        email=_extract_email(text),
        skills=skills,
        years_experience=years,
        seniority_level=_gemini_seniority(text),
        domain=_extract_domain(text, skills),
        raw_text=text[:4000],
    )


def parse_resume_text(text: str) -> CandidateProfile:
    """Extract a CandidateProfile from raw resume text."""
    skills = _extract_skills(text)
    years = _extract_years(text)
    return CandidateProfile(
        name=_extract_name(text),
        email=_extract_email(text),
        skills=skills,
        years_experience=years,
        seniority_level=_gemini_seniority(text),
        domain=_extract_domain(text, skills),
        raw_text=text[:4000],
    )
