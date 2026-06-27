"""Research agent: plan, route, gather, analyse, synthesise.

Splits a question into sub-questions, picks sources for each, gathers evidence,
summarises it and returns one answer plus the full trail. Bounded by hard caps
on sub-questions and sources, and by a per-run time/call budget.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from .llm import LLMClient
from ..sources.sources import Evidence, Source

# The service configures the root logger. In eval/tests it's unconfigured, so
# these records are simply dropped and the eval output stays clean.
log = logging.getLogger("research_agent")

MAX_SUBQUESTIONS = 3
MAX_SOURCES_PER_SUB = 3

# Per-run budget so a slow or looping backend can't run forever. Worst case is
# 8 calls (plan + subs*(route+analyse) + synth); the ceiling leaves headroom.
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_LLM_CALLS = 16
BUDGET_EXCEEDED_ANSWER = "Research could not complete within its time/step budget."

PLANNER_SYS = "You are a research planner. Output only valid JSON."
ROUTER_SYS = "You are a source router. Output only valid JSON."
ANALYST_SYS = "You are a careful analyst. Be concise and grounded."
SYNTH_SYS = "You write clear, auditable answers for senior non-technical readers."


@dataclass
class Step:
    """One recorded action in the audit trail."""
    phase: str
    detail: dict[str, Any]


@dataclass
class AgentResult:
    question: str
    answer: str
    citations: list[str]
    trail: list[Step] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class _BudgetExceeded(Exception):
    """Raised internally when a run hits its time or LLM-call budget."""


@dataclass
class _Budget:
    """Checked before each LLM call.

    The time check happens between calls, so it bounds a slow or looping
    backend but not a single call that hangs forever - that would need a worker
    thread, which we skip to stay dependency-free.
    """
    deadline: float          # time.monotonic() value past which the run aborts
    max_calls: int
    calls: int = 0

    def spend(self) -> None:
        if self.calls >= self.max_calls:
            raise _BudgetExceeded(f"exceeded max LLM calls ({self.max_calls})")
        if time.monotonic() > self.deadline:
            raise _BudgetExceeded("exceeded time budget")
        self.calls += 1


class ResearchAgent:
    def __init__(self, llm: LLMClient, sources: dict[str, Source]):
        self.llm = llm
        self.sources = sources  # {"corpus": ..., "table": ..., "web": ...}

    # All model calls go through here so the budget is enforced in one place.
    def _complete(self, budget: _Budget, system: str, prompt: str) -> str:
        budget.spend()
        return self.llm.complete(system, prompt)

    def _plan(self, question: str, trail: list[Step], budget: _Budget) -> list[str]:
        prompt = (
            "Break the user question into at most three focused sub-questions.\n"
            f"QUESTION: {question}\n"
            'Respond as JSON: {"sub_questions": [...]}'
        )
        raw = self._complete(budget, PLANNER_SYS, prompt)
        subs = _safe_json(raw).get("sub_questions", [question])[:MAX_SUBQUESTIONS]
        trail.append(Step("plan", {"sub_questions": subs}))
        log.info("event=phase phase=plan sub_questions=%d", len(subs))
        return subs

    def _route(self, sub: str, trail: list[Step], budget: _Budget) -> list[str]:
        prompt = (
            "Decide which sources to consult for this sub-question.\n"
            f"SUB_QUESTION: {sub}\n"
            f"AVAILABLE: {list(self.sources)}\n"
            'Respond as JSON: {"sources": [...]}'
        )
        raw = self._complete(budget, ROUTER_SYS, prompt)
        chosen = _safe_json(raw).get("sources", list(self.sources))
        chosen = [s for s in chosen if s in self.sources][:MAX_SOURCES_PER_SUB]
        chosen = chosen or list(self.sources)[:1]
        trail.append(Step("route", {"sub_question": sub, "sources": chosen}))
        log.info("event=phase phase=route sources=%s", ",".join(chosen))
        return chosen

    def _gather(self, sub: str, source_names: list[str], trail: list[Step]) -> list[Evidence]:
        evidence: list[Evidence] = []
        for name in source_names:
            hits = self.sources[name].retrieve(sub, k=2)
            evidence.extend(hits)
            trail.append(Step("gather", {
                "sub_question": sub, "source": name,
                "evidence_ids": [e.source_id for e in hits],
            }))
            log.info("event=phase phase=gather source=%s evidence=%d", name, len(hits))
        return evidence

    def _analyse(self, sub: str, evidence: list[Evidence], trail: list[Step],
                 budget: _Budget) -> tuple[str, list[str]]:
        joined = "\n".join(f"[{e.source_id}] {e.text}" for e in evidence)
        prompt = (
            "Write one or two sentences answering the sub-question using only the "
            "evidence below. Do not invent facts.\n"
            f"SUB_QUESTION: {sub}\n"
            f"EVIDENCE:\n{joined}\n"
        )
        finding = self._complete(budget, ANALYST_SYS, prompt).strip()
        cited = [e.source_id for e in evidence]
        trail.append(Step("analyse", {"sub_question": sub, "finding": finding,
                                      "citations": cited}))
        log.info("event=phase phase=analyse citations=%d", len(cited))
        return finding, cited

    def _synthesise(self, question: str, findings: list[str], trail: list[Step],
                    budget: _Budget) -> str:
        joined = "\n".join(f"- {f}" for f in findings if f)
        prompt = (
            "Compose a final answer to the question from these findings.\n"
            f"QUESTION: {question}\n"
            f"FINDINGS:\n{joined}\n"
        )
        answer = self._complete(budget, SYNTH_SYS, prompt).strip()
        trail.append(Step("synthesise", {"answer": answer}))
        log.info("event=phase phase=synthesise answer_chars=%d", len(answer))
        return answer

    def run(self, question: str, *, timeout_s: float = DEFAULT_TIMEOUT_S,
            max_llm_calls: int = DEFAULT_MAX_LLM_CALLS) -> AgentResult:
        """Run the loop under a time/call budget.

        If the budget runs out, return the partial trail with an ``aborted``
        step and no citations instead of raising.
        """
        trail: list[Step] = []
        budget = _Budget(deadline=time.monotonic() + timeout_s,
                         max_calls=max_llm_calls)

        findings: list[str] = []
        citations: list[str] = []
        try:
            subs = self._plan(question, trail, budget)
            for sub in subs:
                source_names = self._route(sub, trail, budget)
                evidence = self._gather(sub, source_names, trail)
                finding, cited = self._analyse(sub, evidence, trail, budget)
                findings.append(finding)
                citations.extend(cited)

            answer = self._synthesise(question, findings, trail, budget)
        except _BudgetExceeded as exc:
            trail.append(Step("aborted", {"reason": str(exc),
                                          "llm_calls": budget.calls}))
            log.warning("event=aborted reason=%r llm_calls=%d", str(exc), budget.calls)
            return AgentResult(question, BUDGET_EXCEEDED_ANSWER, [], trail)

        citations = list(dict.fromkeys(citations))  # dedupe, keep order
        return AgentResult(question, answer, citations, trail)


def _safe_json(raw: str) -> dict:
    """Parse model JSON defensively; tolerate ```json fences and stray text."""
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}
