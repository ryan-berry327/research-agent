"""LLM client: one ``LLMClient`` protocol, three backends.

* ``StubLLM``      - deterministic and offline, used by tests and CI.
* ``OllamaLLM``    - local models for development.
* ``AnthropicLLM`` - production.

The agent depends only on the protocol, so swapping backends never touches it.
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface the agent needs from any language model."""

    def complete(self, system: str, prompt: str) -> str:  # pragma: no cover - protocol
        ...


class StubLLM:
    """Deterministic offline model. It doesn't understand text - it matches the
    structured prompts the agent sends so the control flow can be tested without
    a network. Reasoning quality is measured against a real provider instead."""

    def complete(self, system: str, prompt: str) -> str:
        if "Break the user question" in prompt:
            # Planner: emit 1-3 sub-questions as JSON.
            question = _extract(prompt, "QUESTION:")
            subs = _naive_subquestions(question)
            return json.dumps({"sub_questions": subs})

        if "Decide which sources" in prompt:
            # Router: choose sources for a single sub-question.
            sub = _extract(prompt, "SUB_QUESTION:").lower()
            sources = []
            if any(w in sub for w in ("trend", "latest", "today", "current", "news")):
                sources.append("web")
            if any(w in sub for w in ("how many", "rate", "number", "figure", "statistic", "spend")):
                sources.append("table")
            sources.append("corpus")  # always consult the internal corpus
            return json.dumps({"sources": list(dict.fromkeys(sources))})

        if "Write one or two sentences" in prompt:
            # Analyst: summarise retrieved evidence for one sub-question.
            evidence = _extract(prompt, "EVIDENCE:")
            first = evidence.strip().split("\n")[0][:240] if evidence.strip() else ""
            return first or "No relevant evidence was found for this sub-question."

        if "Compose a final answer" in prompt:
            # Synthesiser: stitch the per-sub-question findings together.
            findings = _extract(prompt, "FINDINGS:")
            return "Based on the gathered evidence:\n" + findings.strip()

        return ""


class OllamaLLM:  # pragma: no cover - requires a running Ollama daemon
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
        self.model, self.host = model, host

    def complete(self, system: str, prompt: str) -> str:
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["response"]


class AnthropicLLM:  # pragma: no cover - requires network + API key
    def __init__(self, model: str = "claude-sonnet-4-6"):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")


def get_llm() -> LLMClient:
    """Select a backend from the LLM_PROVIDER env var (default: offline stub)."""
    provider = os.environ.get("LLM_PROVIDER", "stub").lower()
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "ollama":
        return OllamaLLM()
    return StubLLM()


# --- helpers ---------------------------------------------------------------
def _extract(prompt: str, marker: str) -> str:
    """Return the text following a marker up to the next ALL-CAPS marker line."""
    idx = prompt.find(marker)
    if idx == -1:
        return ""
    tail = prompt[idx + len(marker):]
    stop = re.search(r"\n[A-Z_]{3,}:", tail)
    return (tail[: stop.start()] if stop else tail).strip()


def _naive_subquestions(question: str) -> list[str]:
    """Offline planner: split on ' and ' / '?'; cap at three sub-questions.

    Guards against prompt scaffolding (e.g. a trailing 'Respond as JSON' line)
    bleeding into the question by taking only the first line and dropping any
    fragment that looks like an instruction rather than a question.
    """
    question = question.splitlines()[0].strip() if question.strip() else question
    parts = re.split(r"\?|\band\b", question, flags=re.IGNORECASE)
    subs = [
        p.strip(" .,") for p in parts
        if len(p.strip()) > 8 and "respond as json" not in p.lower()
    ]
    return (subs or [question.strip()])[:3]
