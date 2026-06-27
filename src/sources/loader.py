"""Load a folder of mixed-format documents into {doc_id: text}.

Drop files into the brain folder and the agent retrieves over them at runtime.
One extractor per format, registered in EXTRACTORS:
    .txt  .md   - plain text / markdown
    .pdf        - pypdf
    .html .htm  - tags stripped (BeautifulSoup)
    .csv        - each row as "col=value; ..."
    .json       - flattened to "key: value" lines

Adding a format is a one-line registration.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

# --- per-format text extractors -------------------------------------------
def _extract_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _extract_csv(path: Path) -> str:
    out: list[str] = []
    with path.open(encoding="utf-8", errors="ignore", newline="") as fh:
        for row in csv.DictReader(fh):
            out.append("; ".join(f"{k}={v}" for k, v in row.items()))
    return "\n".join(out)


def _extract_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    lines: list[str] = []

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}{i}.")
        else:
            lines.append(f"{prefix.rstrip('.')}: {obj}")

    walk(data)
    return "\n".join(lines)


EXTRACTORS: dict[str, Callable[[Path], str]] = {
    ".txt": _extract_text,
    ".md": _extract_text,
    ".pdf": _extract_pdf,
    ".html": _extract_html,
    ".htm": _extract_html,
    ".csv": _extract_csv,
    ".json": _extract_json,
}

SUPPORTED = set(EXTRACTORS)


def load_brain(folder: str | Path) -> dict[str, str]:
    """Load every supported file under ``folder`` into {doc_id: text}.

    doc_id is the path relative to ``folder``, so a citation like
    ``corpus:guidance/ai_delivery.md#2`` points at a real file. Unsupported
    files are skipped; a read error is kept as the doc body so it's visible.
    """
    root = Path(folder)
    docs: dict[str, str] = {}
    if not root.exists():
        return docs

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTRACTORS:
            continue
        doc_id = path.relative_to(root).as_posix()  # forward slashes on every OS
        try:
            text = EXTRACTORS[path.suffix.lower()](path)
        except Exception as exc:  # surface, don't swallow
            text = f"[could not read {doc_id}: {exc}]"
        if text.strip():
            docs[doc_id] = text
    return docs
