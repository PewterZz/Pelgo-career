# Pelgo — AI-First Career Intelligence Agent

total time taken = 9h 8min

full prompts are located in the /prompts folder

---

## Assumptions 

- 1. tool-call sequence: Always extract -> score -> prioritise -> research (top 3 only). Each step needs the previous step's output, so there's no parallelism to exploit.
- 2. When to stop: Once we have a score (confidence >= medium, or one retry exhausted), prioritised gaps, and researched the top 3 skills it emits the final JSON.
- 3. What counts as a failed tool call: timeouts, schema violations, and empty results are all failures. Timeouts skip and move on, schema errors retry once, empty results downgrade confidence. Nothing crashes the run.
- 4. JD caching: each candidate run extracts the job description independently. Skill resource lookups are cached per worker process so repeated skills aren't re-fetched.
- 5. Candidate profile fields: skills, years_experience, seniority_level, domain, work_history, education. Scoring uses skills for match ratio, years for experience, seniority for fit, and domain for a confidence penalty on cross-domain transitions.
 
  Notable tradeoffs:
  - could've run gemini calls in parallel to reduce latency for the research part but decided to keep them sequential for simplicity.
  - perfect matches, dont return anything substantial to learn from quite yet, only weak or mediocre resumes get a proper study plan.
  - gemini from google can still be a little non deterministic on its own even after prompt enforcements or temperature tweaks, so some judgements from resume parsing or output can be slightly different.

---

## Quick Start — Full Stack (docker compose)

**Prerequisites:** Docker, Docker Compose, a Google AI Studio API key (or GCP credentials for Vertex AI).

```bash
# 1. Configure credentials
cp .env.example .env
# Edit .env — set GOOGLE_API_KEY (required)

# 2. Start everything
docker compose up --build

# 3. Open the frontend
open http://localhost:8001
```

This starts PostgreSQL, runs migrations, seeds sample data, launches the API and two background workers. The seed script creates a sample candidate and two match jobs that the workers will pick up automatically.

### Optional — using Vertex AI instead of AI Studio

If you'd rather authenticate via GCP (e.g. you already use `gcloud` and don't want to issue an AI Studio key), the agent and `_gemini_extract` helpers honour the standard `GOOGLE_GENAI_USE_VERTEXAI` switch.

```bash
# 1. Authenticate locally — creates ~/.config/gcloud/application_default_credentials.json
gcloud auth application-default login

# 2. Set Vertex flags in .env
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global   # or us-central1, etc.
# GOOGLE_API_KEY can be left blank in this mode

# 3. Mount the ADC file into each container so the SDK can find it.
#    Add this to api / worker-1 / worker-2 in docker-compose.yml:
#    environment:
#      GOOGLE_APPLICATION_CREDENTIALS: /tmp/gcloud/application_default_credentials.json
#    volumes:
#      - ~/.config/gcloud/application_default_credentials.json:/tmp/gcloud/application_default_credentials.json:ro

docker compose up --build
```

The default `docker-compose.yml` ships configured for AI Studio (no mount, just `GOOGLE_API_KEY`) so a fresh clone runs with one command. Switch to Vertex only if you have a project with the Generative Language API or Vertex AI API enabled and adequate quota for `gemini-2.5-flash`.

---

## Quick Start — CLI only (Part A)

**Prerequisites:** Python 3.12, conda (or any venv), a Google AI Studio API key.

```bash
# 1. Clone and create env
conda create -n pelgo python=3.12 -y
conda activate pelgo
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env — see Environment Variables below

# 3. Run
python -m pelgo --resume path/to/resume.pdf --jd "https://jobs.example.com/swe" 

# or with raw JD text
python -m pelgo --resume path/to/resume.pdf --jd "We are hiring a senior Python engineer..."
```

---

## Environment Variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `GOOGLE_API_KEY` | Yes (AI Studio) | — | Google AI Studio key. Omit if using Vertex AI. |
| `GOOGLE_CLOUD_PROJECT` | Yes (Vertex AI) | — | GCP project ID. Omit if using AI Studio. |
| `GOOGLE_CLOUD_LOCATION` | No | `global` | Vertex AI region. |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Any Gemini model name. |
| `TOOL_TIMEOUT_SEC` | No | `30` | Timeout for JD fetch and scoring tools. |
| `RESEARCH_TIMEOUT_SEC` | No | `15` | Timeout per resource-search API call. |
| `GITHUB_TOKEN` | No | — | GitHub PAT. Raises rate limit from 60 to 5 000 req/hr. |
| `DATABASE_URL` | Yes (Part B) | — | PostgreSQL connection string. Set automatically in docker-compose. |
| `WORKER_CONCURRENCY` | No | `2` | Number of async worker tasks per worker process. |
| `POLL_INTERVAL_SEC` | No | `1.0` | How often workers poll for pending jobs (seconds). |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests run without network access or live credentials — the ADK runner is stubbed.

---

## Architecture

```
Resume PDF ──► pdf_extractor.py ──► CandidateProfile
                                          │
                    Job Description ──────┤
                                          ▼
                            career_intelligence_agent  (LlmAgent)
                            ┌─────────────────────────────────────┐
                            │  extract_jd_requirements            │  FunctionTool
                            │  score_candidate_against_reqs       │  FunctionTool
                            │  prioritise_skill_gaps              │  FunctionTool
                            │  research_skill_resources ──────────┼─► AgentTool
                            └─────────────────────────────────────┘      │
                                                                          ▼
                                                          _SKILL_RESEARCH_SUB_AGENT (LlmAgent)
                                                          ┌──────────────────────────┐
                                                          │  _coursera_search        │
                                                          │  _github_search          │
                                                          │  _huggingface_search     │
                                                          │  _ddg_search (fallback)  │
                                                          └──────────────────────────┘
                                          │
                                          ▼
                                     MatchResult JSON
                            (score · learning plan · agent_trace)
```

PDF extraction is pre-agent — it runs before the LLM is invoked and its output is passed as structured input, not fetched by a tool. This keeps the agent's tool budget focused on reasoning, not file I/O.

---

## A1 — Framework Choice

**Google ADK** was chosen for three reasons that directly address the assignment requirements: it provides `InMemorySessionService` for typed, orchestrator-managed state that persists across tool calls within a run (required by A1); its event stream emits `FunctionCall` / `FunctionResponse` events that let the runner populate `agent_trace` from real execution rather than LLM fabrication (required by A3); and `AgentTool` / `LlmAgent` composition gives a first-class multi-agent graph primitive for the stretch goal without third-party wiring.

**Alternatives considered:**

| Framework | Why not chosen |
|---|---|
| LangGraph | Excellent for complex DAGs; overhead is unjustified for a linear 4-tool pipeline. State management is manual. |
| CrewAI | Role-based abstraction is a poor fit — this is one orchestrator, not a crew. |
| AutoGen | Conversational multi-agent model doesn't map cleanly to a single-run, structured-output use case. |
| Custom ReAct loop | Maximum control, but re-implements session state, event streaming, and tool registration that ADK provides. |

### Google ADK Stretch — What ADK Provides

Since ADK is the primary framework, every tool benefits from ADK primitives. Three capabilities were critical:

1. **`FunctionTool` + `AgentTool` registration** — tools are registered once and the orchestrator handles argument marshalling and invocation. `research_skill_resources` is an `AgentTool` wrapping a dedicated `LlmAgent` sub-agent, giving real multi-agent composition with zero custom routing code.
2. **`InMemorySessionService` for typed state** — session state (`tool_errors`, `low_confidence_retries`, scoring results) persists across tool calls within a run, managed by ADK rather than a global variable or manual dict passing.
3. **Event streaming** — the `Runner.run_async()` event stream emits `FunctionCall` and `FunctionResponse` events in real time, allowing `agent_trace` to be populated by the orchestrator from actual execution events rather than asking the LLM to self-report. This is what makes the trace verifiable.

---

## A2 — Tool Suite

### Tool 1 — `extract_jd_requirements(job_url_or_text)`

Extracts structured requirements from a job description supplied as either a URL or raw text.

- If a URL is provided, fetches the page via `httpx` (follows redirects, 30 s timeout).
- Calls Gemini with `response_mime_type="application/json"` to extract a structured object.
- Schema-validates the result with Pydantic before returning. On validation failure, retries once with plain text, then returns a partial result with `confidence: low`.

**Returns:** `required_skills[]`, `nice_to_have_skills[]`, `seniority_level`, `domain`, `responsibilities[]`

---

### Tool 2 — `score_candidate_against_requirements(candidate_profile_json, requirements_json)`

Scores the candidate against the extracted JD using deterministic heuristics — no LLM call — for reproducibility and speed.

**Scoring formula:**

| Dimension | Weight | How computed |
|---|---|---|
| Skills | 50% | `len(matched_skills) / len(required_skills) × 100` |
| Experience | 30% | `candidate_years / seniority_expected_years × 100`, capped at 100 |
| Seniority fit | 20% | `max(0, 100 − abs(seniority_gap) × 15)` |

**Confidence heuristic:**

```
jd_completeness  = min(len(required_skills) / 10, 1.0)
match_ratio      = len(matched_skills) / max(len(required_skills), 1)

high:   jd_completeness >= 0.7  AND  match_ratio >= 0.5
medium: jd_completeness >= 0.4  OR   match_ratio >= 0.3
low:    otherwise

Domain distance penalty: if candidate.domain ≠ jd.domain, confidence is capped at medium.
```

Confidence is derived from three detectable signals — JD completeness (how many required skills were listed), match ratio (fraction of those skills the candidate holds), and domain distance (cross-domain transitions carry inherent uncertainty). It is never asserted arbitrarily.

**Returns:** `overall_score`, `dimension_scores`, `matched_skills[]`, `gap_skills[]`, `confidence`

---

### Tool 3 — `research_skill_resources(skill_name, seniority_context)`

Implemented as an ADK `AgentTool` wrapping a dedicated `LlmAgent` sub-agent. The parent orchestrator calls it like any function; ADK routes execution to the sub-agent, which runs its own tool loop across four sources before returning curated JSON.

**Sources (in priority order):**

| Source | API | Auth | Best for |
|---|---|---|---|
| Coursera | `api.coursera.org/api/courses.v1` | None | Structured multi-week courses |
| GitHub | `api.github.com/search/repositories` | Optional `GITHUB_TOKEN` | Awesome lists, tutorial repos (ranked by stars) |
| HuggingFace | `huggingface.co/api/datasets` + known course URLs | None | ML/AI skills; free HF courses |
| DuckDuckGo | `api.duckduckgo.com` | None | Last-resort general web fallback |

The sub-agent merges results from all sources, deduplicates by URL, adjusts `relevance_score` for seniority fit, and returns a ranked list.

**Returns:** `resources[]` — each with `title`, `url`, `estimated_hours`, `type`, `relevance_score`

---

### Tool 4 — `prioritise_skill_gaps(gap_skills_json, job_market_context)`

Ranks gap skills by expected match-score gain using Gemini reasoning over three signals: frequency in the JD's required skills list, market demand from `job_market_context`, and foundational vs. advanced positioning. Falls back to estimated gain-per-skill (`100 / total_required`) if the LLM call fails.

Ranking is reasoned, not alphabetical. The agent uses this output to decide which skills to research first — it calls `research_skill_resources` for the top 3 only, not all gaps.

**Returns:** `prioritized_skills[]` — each with `skill`, `priority_rank`, `estimated_match_gain_pct`, `rationale`

---

## A3 — Agent System Prompt

```
You are a Career Intelligence Agent for Pelgo, a career-transition platform.
Given a candidate profile (JSON) and a job description (text or URL), you autonomously
orchestrate a multi-step reasoning process to produce a match score and personalized
learning plan.

## Available Tools (use ONLY these exact names)
- extract_jd_requirements
- score_candidate_against_requirements
- prioritise_skill_gaps
- research_skill_resources

Do NOT call any other tool name. These are the only tools available to you.

## Mandatory Tool Sequence
1. Call extract_jd_requirements with the job description to obtain structured requirements.
2. Call score_candidate_against_requirements with the candidate profile JSON and the
   requirements JSON.
   - If confidence is "low", enrich the context and call score_candidate_against_requirements
     once more. Do not finalize a low-confidence score without a retry.
3. Call prioritise_skill_gaps with the gap_skills list and the job domain as context.
4. For the top 3 priority skills (not all), call research_skill_resources one at a time.
5. Output the final MatchResult JSON.

## Failure Handling
- If extract_jd_requirements returns an error or empty required_skills: retry once with the
  raw JD text as plain text input. If it fails again, proceed with partial data and set
  confidence to "low".
- If score_candidate_against_requirements returns confidence "low": retry once, passing
  additional context in candidate_profile_json (include extra skills inferred from work
  history).
- If research_skill_resources times out or returns no resources: skip that skill and note it.
  Never block the run waiting for a single research call.
- Never abort the run. Produce the best possible output from available information.

## Termination Condition
Produce the final output when you have:
  - Extracted JD requirements (or exhausted retries)
  - A score with confidence >= "medium" (or exhausted retries)
  - Prioritized the skill gaps
  - Researched the top 3 gaps (or all gaps if fewer than 3)

## Output Format
End your response with a single JSON object matching this schema exactly:
{
  "job_id": "<uuid>",
  "overall_score": <0-100>,
  "confidence": "low|medium|high",
  "dimension_scores": {"skills": <0-100>, "experience": <0-100>, "seniority_fit": <0-100>},
  "matched_skills": [...],
  "gap_skills": [...],
  "reasoning": "<2-3 sentence plain English explanation>",
  "learning_plan": [
    {
      "skill": "<skill>",
      "priority_rank": <int>,
      "estimated_match_gain_pct": <float>,
      "resources": [{"title": "...", "url": "...", "estimated_hours": <int>, "type": "course|project|cert|doc"}],
      "rationale": "<why this skill first>"
    }
  ]
}
Do NOT include agent_trace in your output — the orchestrator injects it.
```

---

## A4 — Failure Mode Handling

### Tool timeout — `research_skill_resources`
The system prompt instructs the agent to skip the skill and note it rather than block. `_RESEARCH_TIMEOUT_SEC` (default 15 s) applies per source independently inside the sub-agent's tool loop — a slow Coursera call does not prevent GitHub from being tried. At the runner level, `agent_trace` records status `"timeout"` or `"error"` per tool call so the evaluator can see what was skipped and why.

### Invalid tool output — `extract_jd_requirements`
The tool schema-validates its Gemini response with Pydantic before returning. On `ValidationError`, it catches the exception, logs it to `session.state["tool_errors"]`, and retries once with the raw JD text as plain-text input (bypassing URL fetch). If the second attempt also fails, it returns an empty `JDRequirements` and the scoring step proceeds with `confidence: low`. The run never crashes.

### Low confidence score
If `score_candidate_against_requirements` returns `confidence: "low"`, the system prompt requires the agent to retry once — passing an enriched `candidate_profile_json` that includes skills inferred from work history entries, not just the top-level skills list. Only after the retry may the agent finalise a low-confidence result. The retry count is tracked in `session.state["low_confidence_retries"]` to prevent infinite loops.

---

## Trade-offs

| Decision | Trade-off |
|---|---|
| Deterministic scoring (no LLM for Tool 2) | Reproducible and fast; misses nuanced skill synonyms (e.g. "Postgres" ≠ "PostgreSQL" without normalisation). Chosen to make scores auditable. |
| AgentTool sub-agent for research | Real multi-agent ADK orchestration; adds one extra LLM call per research invocation vs. a plain function. Acceptable given research is I/O-bound, not latency-critical. |
| Top-3 skills researched only | Keeps token usage and latency bounded; the agent may miss resources for lower-priority gaps. Prioritisation tool runs first precisely to make this cut defensible. |
| PDF extraction pre-agent | Simpler agent prompt; loses the ability to ask clarifying questions about ambiguous resume content. Acceptable for a structured PDF input. |
| No LinkedIn / LinkedIn Learning | Both require OAuth approval and a subscription. Browser automation was considered and rejected: it would require personal credentials in a public repo and violates ToS. |

---

## Structured Logging

Every agent run emits JSON lines to stderr. Each line is a self-contained record:

```
{"ts": "2026-05-01T15:00:00Z", "level": "info", "event": "job_start",      "job_id": "...", "candidate": "Alex Kim"}
{"ts": "...",                   "level": "info", "event": "tool_call_start", "job_id": "...", "tool": "extract_jd_requirements"}
{"ts": "...",                   "level": "info", "event": "tool_call_end",   "job_id": "...", "tool": "extract_jd_requirements", "status": "success", "latency_ms": 340}
{"ts": "...",                   "level": "info", "event": "llm_usage",       "job_id": "...", "llm_call_n": 1, "prompt_tokens": 812, "candidates_tokens": 204, "total_tokens": 1016}
{"ts": "...",                   "level": "info", "event": "job_complete",    "job_id": "...", "status": "completed", "score": 82, "confidence": "high", "total_llm_calls": 4, "fallbacks": 0, "total_ms": 18420}
```

Fields covered: `job_id`, `tool` name, call `status` (success/error), `latency_ms` per tool, `prompt_tokens` / `candidates_tokens` / `total_tokens` per LLM call, final `score`, `confidence`, and run `total_ms`. The logger lives in `pelgo/logging.py` and writes to stderr so it doesn't pollute stdout JSON output.

---

## Part B — Async Infrastructure

### B1 — API Surface

FastAPI application (`pelgo/api/main.py`) with four core endpoints and one admin endpoint:

| Endpoint | Method | Behaviour |
|---|---|---|
| `/api/v1/candidate` | POST | Accepts `multipart/form-data` (PDF resume) or `application/json` (resume text). Parses, extracts structured profile, stores in PostgreSQL. Returns `candidate_id`, `name`, `seniority_level`, `domain`, `years_experience`, `skills`. |
| `/api/v1/matches` | POST | Accepts `candidate_id` and up to 10 JDs (text or URL). Creates one `MatchJob` per JD with `status: pending`. Returns immediately with job IDs. |
| `/api/v1/matches/{id}` | GET | Returns status and full structured agent output (including `agent_trace`) for one job. Status: `pending | processing | completed | failed`. |
| `/api/v1/matches` | GET | Paginated list of match jobs. Filterable by `status`. Requires `limit` and `offset`. |
| `/api/v1/admin/matches/{job_id}/requeue` | POST | Resets a failed job to `pending`, zeros `attempt_count`, clears `error_detail`. |

---

### B2 — Background Workers

Workers run out-of-process via `python -m pelgo.worker` (separate containers in docker-compose).

- **Concurrency:** `docker-compose.yml` starts two worker containers (`worker-1`, `worker-2`), each running `WORKER_CONCURRENCY=2` async tasks — 4 concurrent workers total.
- **Race-condition-safe claiming:** `claim_next_job()` uses `SELECT FOR UPDATE SKIP LOCKED` to atomically claim one pending job. No duplicate processing.
- **Failure isolation:** Each job runs in a `try/except` — a failed agent run logs the error and marks the job accordingly without crashing the worker or blocking other jobs.
- **Dead-letter after 3 attempts:** `mark_failed()` checks `attempt_count >= 3`. If so, sets `status: failed` with `error_detail` and partial `agent_trace`. Otherwise resets to `pending` for retry.
- **Stuck job recovery:** On startup, `reset_stuck_jobs()` resets any job stuck in `processing` for over 10 minutes (handles crashed workers).
- **Polling interval:** Configurable via `POLL_INTERVAL_SEC` (default 1 s).

---

### B3 — Data Model

PostgreSQL 16 with Alembic migrations (`alembic/versions/0001_initial_schema.py`).

**`candidates` table:**

| Column | Type | Notes |
|---|---|---|
| `candidate_id` | UUID PK | Auto-generated |
| `name` | String | Required |
| `email` | String | Optional, indexed |
| `seniority_level` | String | e.g. "senior", "mid" |
| `domain` | String | e.g. "backend", "data science" |
| `years_experience` | Float | |
| `skills` | JSONB | List of skill strings |
| `education` | JSONB | Structured education entries |
| `work_history` | JSONB | Structured work entries |
| `raw_text` | Text | Original resume text |
| `created_at` | DateTime | Indexed |

**`match_jobs` table:**

| Column | Type | Notes |
|---|---|---|
| `job_id` | UUID PK | Auto-generated |
| `candidate_id` | UUID FK | Indexed |
| `jd_input` | Text | Raw JD text or URL |
| `status` | String | Check constraint: `pending | processing | completed | failed`. Indexed with `created_at` and `updated_at`. |
| `attempt_count` | Integer | Tracks retries (dead-letter at 3) |
| `processing_started_at` | DateTime | Set on claim, used for stuck-job detection |
| `error_detail` | Text | Error message on failure |
| `result` | JSONB | Full `MatchResult` including `agent_trace` |
| `created_at` | DateTime | |
| `updated_at` | DateTime | |

**Queryable by design:** all jobs for a candidate (`idx_match_jobs_candidate_id`), all jobs by status (`idx_match_jobs_status_created`), and `agent_trace` for a specific job (`result->'agent_trace'`).

---

### Docker Compose — Full Stack

```bash
docker compose up --build
```

Starts the entire system with one command:

| Service | Role |
|---|---|
| `postgres` | PostgreSQL 16 with health check |
| `migrate` | Runs `alembic upgrade head`, exits on success |
| `seed` | Runs `scripts/seed.py` (sample candidate + 2 JDs), exits on success |
| `api` | FastAPI on port 8001 (maps to 8000 inside container) |
| `worker-1` | Background worker (2 concurrent tasks) |
| `worker-2` | Background worker (2 concurrent tasks) |

The seed script creates a sample candidate (Jane Smith, 7 years, senior backend engineer) and two match jobs with different JDs.

---

### Frontend

Single-page app served at `/` via FastAPI's `StaticFiles` (`static/index.html`).

Three tabs:
1. **Upload Resume** — PDF upload or paste raw text. Calls `POST /api/v1/candidate`.
2. **Match Jobs** — Submit JDs for a candidate. Displays jobs table with status badges.
3. **Results** — Match cards with overall score, dimension score bars, learning plan, and expandable agent trace. Polls for updates while jobs are pending/processing.
