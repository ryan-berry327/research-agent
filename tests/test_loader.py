"""Tests for the local 'brain' document loader."""
from __future__ import annotations

from pathlib import Path

from src.sources.loader import load_brain, SUPPORTED


def test_loads_real_brain_folder():
    brain = Path(__file__).resolve().parent.parent / "brain"
    docs = load_brain(brain)
    # at least the seeded docs should load
    assert docs
    assert any(d.endswith(".md") for d in docs)
    assert any(d.endswith(".csv") for d in docs)


def test_missing_folder_returns_empty():
    assert load_brain("/no/such/path") == {}


def test_supports_common_formats():
    for ext in (".txt", ".md", ".pdf", ".html", ".csv", ".json"):
        assert ext in SUPPORTED


def test_loads_each_format(tmp_path):
    (tmp_path / "a.txt").write_text("plain text doc")
    (tmp_path / "b.md").write_text("# markdown doc")
    (tmp_path / "c.csv").write_text("col1,col2\nv1,v2\n")
    (tmp_path / "d.json").write_text('{"key": "json value"}')
    (tmp_path / "e.html").write_text("<html><body><p>html doc</p></body></html>")
    docs = load_brain(tmp_path)
    assert len(docs) == 5
    assert "json value" in docs["d.json"]
    assert "html doc" in docs["e.html"]
    assert "col1=v1" in docs["c.csv"]


def test_html_strips_scripts(tmp_path):
    (tmp_path / "x.html").write_text(
        "<html><body><p>keep this</p><script>drop this</script></body></html>"
    )
    docs = load_brain(tmp_path)
    assert "keep this" in docs["x.html"]
    assert "drop this" not in docs["x.html"]


def test_unreadable_file_is_surfaced_not_swallowed(tmp_path):
    # a .json file with invalid content should produce a visible error doc
    (tmp_path / "bad.json").write_text("{not valid json")
    docs = load_brain(tmp_path)
    assert "could not read" in docs["bad.json"]
