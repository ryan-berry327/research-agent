"""Wire up an agent from the default sources."""
from __future__ import annotations

from pathlib import Path

from .agent.core import ResearchAgent
from .agent.llm import get_llm, LLMClient
from .sources.fixtures import CORPUS, TABLE, WEB
from .sources.loader import load_brain
from .sources.sources import CorpusSource, TableSource, WebSource

DEFAULT_BRAIN = Path(__file__).resolve().parent.parent / "brain"


def build_agent(llm: LLMClient | None = None, brain_dir: str | Path | None = None) -> ResearchAgent:
    """Build the agent. The corpus loads from the brain folder if it exists,
    otherwise it falls back to the in-code fixture so tests run with no setup."""
    brain_path = Path(brain_dir) if brain_dir is not None else DEFAULT_BRAIN
    documents = load_brain(brain_path) or CORPUS

    sources = {
        "corpus": CorpusSource(documents),
        "table": TableSource(TABLE),
        "web": WebSource(WEB),
    }
    return ResearchAgent(llm or get_llm(), sources)
