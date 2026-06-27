"""FastAPI service for the research agent.

* ``GET  /health``   - readiness probe for the container healthcheck.
* ``POST /research`` - run the agent and return the answer, citations and trail.

The agent is built once at startup and reused; the trail rides along in the
response so callers can see which sources produced an answer.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .assembly import build_agent

# key=value logging to stdout, level from LOG_LEVEL (default INFO).
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("research_agent")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["agent"] = build_agent()
    yield
    _state.clear()


app = FastAPI(title="Multi-Source Research Agent", version="1.0.0", lifespan=lifespan)


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class ResearchResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    trail: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "agent_ready": "agent" in _state}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest) -> ResearchResponse:
    log.info("event=request_received question_chars=%d", len(req.question))
    result = _state["agent"].run(req.question)
    d = result.to_dict()
    log.info("event=response citations=%d trail_steps=%d",
             len(d["citations"]), len(d["trail"]))
    return ResearchResponse(**d)
