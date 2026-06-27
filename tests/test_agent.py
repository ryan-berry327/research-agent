"""Tests for the research agent. All run fully offline against the StubLLM."""
from __future__ import annotations

import time

from src.agent.core import ResearchAgent, MAX_SUBQUESTIONS, BUDGET_EXCEEDED_ANSWER
from src.agent.llm import StubLLM
from src.assembly import build_agent
from src.eval.harness import run_eval, gate, GOLDEN
from src.sources.fixtures import CORPUS
from src.sources.sources import CorpusSource, TableSource, WebSource


class _SlowLLM:
    """Sleeps and returns empty JSON on every call - a stand-in for a slow or
    looping provider, so the runaway guard can be tested offline."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = 0

    def complete(self, system: str, prompt: str) -> str:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return "{}"


def test_agent_runs_all_phases():
    agent = build_agent(StubLLM())
    result = agent.run("What is the latest trend in government AI?")
    phases = [s.phase for s in result.trail]
    assert "plan" in phases
    assert "route" in phases
    assert "gather" in phases
    assert "analyse" in phases
    assert "synthesise" in phases
    assert result.answer


def test_planner_is_bounded():
    agent = build_agent(StubLLM())
    long_q = " and ".join([f"question number {i} about policy" for i in range(8)])
    result = agent.run(long_q)
    plan_step = next(s for s in result.trail if s.phase == "plan")
    assert len(plan_step.detail["sub_questions"]) <= MAX_SUBQUESTIONS


def test_router_selects_table_for_quantitative():
    agent = build_agent(StubLLM())
    result = agent.run("How many departments are using LLMs?")
    used = {s.detail["source"] for s in result.trail if s.phase == "gather"}
    assert "table" in used


def test_router_selects_web_for_current_events():
    agent = build_agent(StubLLM())
    result = agent.run("What is the latest trend in government AI?")
    used = {s.detail["source"] for s in result.trail if s.phase == "gather"}
    assert "web" in used


def test_every_citation_is_grounded():
    agent = build_agent(StubLLM())
    result = agent.run("Why does evaluation rigour matter and what is the current trend?")
    gathered = {
        eid for s in result.trail if s.phase == "gather"
        for eid in s.detail["evidence_ids"]
    }
    assert all(c in gathered for c in result.citations)


def test_corpus_semantic_retrieval_orders_by_similarity():
    src = CorpusSource({"d": "semantic layer matches a question to data\nunrelated cooking recipe"})
    hits = src.retrieve("semantic layer for matching questions", k=2)
    assert hits
    assert "semantic layer" in hits[0].text


def test_corpus_abstains_on_out_of_estate_query():
    # words absent from the corpus -> no evidence, which is what makes the agent abstain
    src = CorpusSource(CORPUS)
    hits = src.retrieve("average rainfall in the Amazon rainforest in March")
    assert hits == []


def test_table_keyword_retrieval():
    src = TableSource([{"metric": "departments using LLMs", "value": "17"}])
    hits = src.retrieve("departments using LLMs", k=1)
    assert hits and "17" in hits[0].text


def test_web_source_offline_fixture():
    src = WebSource({"u": "agentic systems plan their own steps"})
    hits = src.retrieve("agentic systems planning", k=1)
    assert hits


def test_eval_gate_passes_on_golden():
    report = run_eval(build_agent(StubLLM()), GOLDEN)
    assert gate(report), report.summary()


def test_eval_abstains_on_unanswerable():
    report = run_eval(build_agent(StubLLM()), GOLDEN)
    assert report.abstention_rate == 1.0


def test_run_aborts_on_time_budget():
    # slow backend + tiny timeout: the run aborts cleanly instead of dragging on
    agent = build_agent(_SlowLLM(delay=0.05))
    result = agent.run("anything at all about government policy", timeout_s=0.01)
    assert result.answer == BUDGET_EXCEEDED_ANSWER
    assert result.citations == []
    assert any(s.phase == "aborted" for s in result.trail)


def test_run_aborts_on_llm_call_ceiling():
    slow = _SlowLLM(delay=0.0)
    agent = build_agent(slow)
    result = agent.run("anything at all about government policy",
                       max_llm_calls=1, timeout_s=30)
    assert result.answer == BUDGET_EXCEEDED_ANSWER
    aborted = next(s for s in result.trail if s.phase == "aborted")
    assert aborted.detail["llm_calls"] == 1
    assert slow.calls == 1  # the guard stopped further calls
