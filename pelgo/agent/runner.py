from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agent import build_agent
from .schemas import (
    AgentState,
    AgentTrace,
    CandidateProfile,
    LearningPlanItem,
    MatchResult,
    SkillResource,
    ToolCallRecord,
)

_USER_ID = "pelgo-system"


class AgentRunner:
    """Wraps ADK Runner with trace capture and structured output validation."""

    def __init__(self) -> None:
        self._session_service = InMemorySessionService()
        self._agent = build_agent()
        self._runner = Runner(
            agent=self._agent,
            session_service=self._session_service,
            app_name="pelgo",
        )

    async def run(
        self,
        candidate_profile: CandidateProfile,
        jd_input: str,
        job_id: str | None = None,
    ) -> MatchResult:
        job_id = job_id or str(uuid.uuid4())

        session = await self._session_service.create_session(
            app_name="pelgo",
            user_id=_USER_ID,
        )
        session_id = session.id

        initial_state = AgentState(
            job_id=job_id,
            candidate_profile=candidate_profile,
            jd_input=jd_input,
        )
        await self._session_service.append_event(
            session=session,
            event=_state_event(initial_state),
        )

        message = _build_message(candidate_profile, jd_input)
        trace = AgentTrace()
        llm_call_count = 0

        pending_call_times: dict[str, float] = {}

        async for event in self._runner.run_async(
            user_id=_USER_ID,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=message)],
            ),
        ):
            for fc in event.get_function_calls():
                pending_call_times[fc.id] = time.monotonic()

            for fr in event.get_function_responses():
                start = pending_call_times.pop(fr.id, time.monotonic())
                latency_ms = int((time.monotonic() - start) * 1000)
                response_data = fr.response or {}
                status = "error" if "error" in response_data else "success"
                trace.tool_calls.append(
                    ToolCallRecord(
                        tool=fr.name,
                        status=status,
                        latency_ms=latency_ms,
                    )
                )
                if status == "error":
                    trace.fallbacks_triggered += 1

            if event.usage_metadata:
                llm_call_count += 1

            if event.is_final_response() and event.content:
                raw_text = _extract_text(event)
                trace.total_llm_calls = llm_call_count
                result = _parse_and_validate(raw_text, job_id, trace)

                session_obj = await self._session_service.get_session(
                    app_name="pelgo",
                    user_id=_USER_ID,
                    session_id=session_id,
                )
                agent_state_dict = session_obj.state if session_obj else {}
                _enrich_result_from_state(result, agent_state_dict)
                return result

        trace.total_llm_calls = llm_call_count
        return _empty_result(job_id, trace)


def _state_event(state: AgentState):
    """Create an ADK event that seeds session.state from AgentState."""
    from google.adk.events import Event, EventActions

    return Event(
        author="user",
        actions=EventActions(state_delta=state.model_dump(mode="json")),
        content=types.Content(role="user", parts=[]),
    )


def _build_message(profile: CandidateProfile, jd_input: str) -> str:
    return (
        f"Candidate profile:\n{profile.model_dump_json(indent=2)}\n\n"
        f"Job description:\n{jd_input}"
    )


def _extract_text(event) -> str:
    if not event.content or not event.content.parts:
        return ""
    return " ".join(p.text for p in event.content.parts if hasattr(p, "text") and p.text)


def _parse_and_validate(raw_text: str, job_id: str, trace: AgentTrace) -> MatchResult:
    json_start = raw_text.find("{")
    json_end = raw_text.rfind("}")
    if json_start == -1 or json_end == -1:
        return _empty_result(job_id, trace)
    try:
        data = json.loads(raw_text[json_start: json_end + 1])
        data["job_id"] = job_id
        data["agent_trace"] = trace.model_dump()
        return MatchResult.model_validate(data)
    except Exception:
        return _empty_result(job_id, trace)


def _enrich_result_from_state(result: MatchResult, state: dict) -> None:
    """Backfill learning plan resources from state if the LLM omitted them."""
    researched: dict = state.get("researched_resources", {})
    if not researched:
        return
    for item in result.learning_plan:
        if not item.resources and item.skill in researched:
            item.resources = [SkillResource(**r) for r in researched[item.skill]]


def _empty_result(job_id: str, trace: AgentTrace) -> MatchResult:
    from .schemas import DimensionScores

    return MatchResult(
        job_id=job_id,
        overall_score=0,
        confidence="low",
        dimension_scores=DimensionScores(skills=0, experience=0, seniority_fit=0),
        matched_skills=[],
        gap_skills=[],
        reasoning="Agent run produced no valid output.",
        learning_plan=[],
        agent_trace=trace,
    )
