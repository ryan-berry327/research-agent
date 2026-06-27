"""Evaluation harness. Scores the agent on three axes:

1. routing    - did it consult the sources it should have, per sub-question?
2. grounding  - is every citation a real, gathered evidence id?
3. abstention - does it decline when nothing in the estate can answer?

``run_eval`` returns the metrics plus a per-case breakdown; ``gate`` turns
them into the pass/fail CI uses to block regressions.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..agent.core import ResearchAgent


@dataclass
class Case:
    question: str
    expect_sources: set[str]      # sources that SHOULD appear in the trail
    answerable: bool              # False => agent should abstain


GOLDEN: list[Case] = [
    Case("What is the latest trend in government AI?", {"web", "corpus"}, True),
    Case("How many departments are using LLMs?", {"table", "corpus"}, True),
    Case("Why does evaluation rigour matter and what is the current trend?",
         {"corpus", "web"}, True),
    Case("What is the average rainfall in the Amazon rainforest in March?",
         set(), False),
    # table-only
    Case("How many AI pilots are reaching production?", {"table"}, True),
    # corpus-only
    Case("Why must production AI services be monitored?", {"corpus"}, True),
    # multi-source (web + table)
    Case("What is the latest figure for AI pilots reaching production?",
         {"web", "table"}, True),
    # multi-source (corpus + web)
    Case("What is the current trend in evaluation rigour?", {"web", "corpus"}, True),
    # out-of-estate -> abstain
    Case("What is the boiling point of helium on Jupiter?", set(), False),
    Case("Who won the 1998 World Cup final?", set(), False),
]

ABSTAIN_MARKERS = ("no relevant evidence", "could not find", "cannot answer",
                   "no evidence")


@dataclass
class EvalReport:
    routing_accuracy: float
    grounding_rate: float
    abstention_rate: float
    n: int
    cases: list[dict]

    def summary(self) -> str:
        return (f"routing={self.routing_accuracy:.2f} "
                f"grounding={self.grounding_rate:.2f} "
                f"abstention={self.abstention_rate:.2f} (n={self.n})")


def run_eval(agent: ResearchAgent, cases: list[Case] | None = None) -> EvalReport:
    cases = cases or GOLDEN
    routing_hits = grounding_hits = 0
    abstain_total = abstain_hits = 0
    rows = []

    for c in cases:
        result = agent.run(c.question)
        used = {s.detail["source"] for s in result.trail if s.phase == "gather"}

        # 1. routing: every expected source was actually consulted
        routed_ok = c.expect_sources.issubset(used) if c.expect_sources else True
        routing_hits += int(routed_ok)

        # 2. grounding: every citation corresponds to a real gathered evidence id
        gathered_ids = {
            eid for s in result.trail if s.phase == "gather"
            for eid in s.detail["evidence_ids"]
        }
        grounded_ok = all(cit in gathered_ids for cit in result.citations)
        grounding_hits += int(grounded_ok)

        # 3. abstention: unanswerable cases should produce an abstention signal
        if not c.answerable:
            abstain_total += 1
            text = result.answer.lower()
            if any(m in text for m in ABSTAIN_MARKERS) or not result.citations:
                abstain_hits += 1

        rows.append({
            "question": c.question,
            "routed_ok": routed_ok,
            "grounded_ok": grounded_ok,
            "sources_used": sorted(used),
            "citations": result.citations,
        })

    n = len(cases)
    return EvalReport(
        routing_accuracy=routing_hits / n,
        grounding_rate=grounding_hits / n,
        abstention_rate=(abstain_hits / abstain_total) if abstain_total else 1.0,
        n=n,
        cases=rows,
    )


def gate(report: EvalReport, *, min_routing=0.75, min_grounding=1.0,
         min_abstention=1.0) -> bool:
    """CI quality gate. Grounding and abstention must be perfect; routing >= 0.75."""
    return (report.routing_accuracy >= min_routing
            and report.grounding_rate >= min_grounding
            and report.abstention_rate >= min_abstention)
