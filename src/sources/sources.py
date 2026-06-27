"""Data sources behind one ``Source`` protocol.

* ``CorpusSource``  - internal documents, ranked by word overlap (TF-cosine).
                      Swap a real embedding model in at the ``_embed`` call site.
* ``TableSource``   - structured rows, matched by keyword.
* ``WebSource``     - the open web. Offline fixture by default; pass a fetcher
                      for the real thing.

Each returns ``Evidence`` with a stable ``source_id`` so every answer traces
back to its source.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

# Stopwords carry no topical signal. Dropping them from the overlap check stops
# a query matching on words like "the/in/is" alone, which is what lets the
# agent abstain on out-of-estate questions.
_STOPWORDS = {
    "the", "a", "an", "in", "on", "of", "to", "is", "are", "and", "or", "our",
    "what", "how", "why", "for", "with", "this", "that", "it", "its", "be",
    "as", "at", "by", "we", "you", "i", "do", "does",
}


@dataclass(frozen=True)
class Evidence:
    source_id: str          # e.g. "corpus:policy_brief#2"
    source_type: str        # "corpus" | "table" | "web"
    text: str
    score: float            # retrieval confidence, 0..1


class Source(Protocol):
    name: str

    def retrieve(self, query: str, k: int = 3) -> list[Evidence]:  # pragma: no cover
        ...


def _embed(text: str) -> dict[str, float]:
    """Bag-of-words term-frequency vector. Swap in a real embedding model here
    and the cosine below works unchanged."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    vec: dict[str, float] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0.0) + 1.0
    return vec


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[t] * b[t] for t in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class CorpusSource:
    """Word-overlap (TF-cosine) retrieval over internal documents. Not embedding
    based - ``_embed`` is bag-of-words and ``_cosine`` ranks by overlap."""

    name = "corpus"

    # Below this score a chunk is just shared common words, not a real match, so
    # we drop it and let the agent abstain rather than cite noise.
    MIN_RELEVANCE = 0.15

    def __init__(self, documents: dict[str, str], min_relevance: float | None = None):
        self.min_relevance = self.MIN_RELEVANCE if min_relevance is None else min_relevance
        # split each doc into paragraphs and pre-embed them
        self._chunks: list[tuple[str, str, dict[str, float]]] = []
        for doc_id, text in documents.items():
            for i, para in enumerate(p for p in text.split("\n") if p.strip()):
                self._chunks.append((f"{doc_id}#{i}", para.strip(), _embed(para)))

    def retrieve(self, query: str, k: int = 3) -> list[Evidence]:
        q = _embed(query)
        # A chunk must share at least one non-stopword query term, otherwise an
        # out-of-estate question matches nothing and the agent abstains.
        content_terms = {t for t in q if t not in _STOPWORDS}
        scored: list[Evidence] = []
        for cid, text, emb in self._chunks:
            if content_terms and not (content_terms & set(emb)):
                continue
            score = round(_cosine(q, emb), 4)
            scored.append(Evidence(f"corpus:{cid}", "corpus", text, score))
        scored.sort(key=lambda e: e.score, reverse=True)
        return [e for e in scored[:k] if e.score >= self.min_relevance]


class TableSource:
    """Keyword lookup over structured rows."""

    name = "table"

    def __init__(self, rows: list[dict[str, str]]):
        self._rows = rows

    def retrieve(self, query: str, k: int = 3) -> list[Evidence]:
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        out: list[Evidence] = []
        for i, row in enumerate(self._rows):
            blob = " ".join(str(v) for v in row.values()).lower()
            hits = sum(1 for t in terms if t in blob)
            if hits:
                rendered = "; ".join(f"{k2}={v}" for k2, v in row.items())
                out.append(Evidence(f"table:row{i}", "table", rendered,
                                    round(hits / max(len(terms), 1), 4)))
        out.sort(key=lambda e: e.score, reverse=True)
        return out[:k]


class WebSource:
    """Open-web source. Offline fixture by default; real fetcher pluggable."""

    name = "web"

    MIN_RELEVANCE = 0.15

    def __init__(self, pages: dict[str, str] | None = None, fetcher=None,
                 min_relevance: float | None = None):
        self._pages = pages or {}
        self._fetcher = fetcher  # callable(query) -> list[(url, snippet)]
        self.min_relevance = self.MIN_RELEVANCE if min_relevance is None else min_relevance

    def retrieve(self, query: str, k: int = 3) -> list[Evidence]:
        if self._fetcher is not None:  # pragma: no cover - real network path
            return [
                Evidence(f"web:{url}", "web", snippet, 0.5)
                for url, snippet in self._fetcher(query)[:k]
            ]
        q = _embed(query)
        scored = [
            Evidence(f"web:{url}", "web", text, round(_cosine(q, _embed(text)), 4))
            for url, text in self._pages.items()
        ]
        scored.sort(key=lambda e: e.score, reverse=True)
        return [e for e in scored[:k] if e.score >= self.min_relevance]
