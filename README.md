# Multi-Source Research Agent

A research agent that breaks a question into sub-questions, decides which data
sources to consult for each, gathers evidence across internal documents,
structured tables and the open web, summarises it, and returns one answer with
the full trail of where every claim came from.

It runs fully offline by default (a deterministic stub model and local sources),
so you can install and evaluate it with no API keys or network access. A real
provider (Anthropic or local Ollama) drops in behind the same interface.

## How it works

The loop is **plan → route → gather → analyse → synthesise**:

- **plan** splits the question into at most three sub-questions
- **route** picks the sources for each sub-question
- **gather** retrieves evidence from those sources
- **analyse** writes a short finding per sub-question from the evidence
- **synthesise** combines the findings into one answer

Each run is bounded: hard caps on sub-questions and sources, plus a per-run time
and LLM-call budget so a slow or looping backend can't run away (it aborts
cleanly and returns the partial trail). Every piece of evidence has a stable id,
and the full trail comes back in the response so you can see exactly what
produced an answer. The eval harness runs in CI and fails the build on any
routing, grounding or abstention regression.

## Quickstart

```bash
pip install -r requirements-dev.txt

# run the tests (offline)
python -m pytest -q

# run the eval harness + gate (offline)
python -m src.run_eval

# serve the API
uvicorn src.service:app --reload
# then:  curl -s localhost:8000/research -H 'content-type: application/json' \
#              -d '{"question":"How many departments are using LLMs?"}' | jq
```

Pick a provider with an env var (default is the offline stub):

```bash
LLM_PROVIDER=ollama    uvicorn src.service:app   # local, free
LLM_PROVIDER=anthropic uvicorn src.service:app   # needs ANTHROPIC_API_KEY
```

Logging is key=value to stdout; set `LOG_LEVEL` (default `INFO`).

## The local "brain"

The corpus is a folder of real documents, not a fixture. Drop files into
`brain/` and the agent retrieves over them at runtime. Supported formats:
`.txt`, `.md`, `.pdf`, `.html`, `.csv`, `.json`, each with its own extractor
registered in one place, so adding a format is a one-line change. Citations
point at the real relative path (e.g. `corpus:guidance/ai_delivery.md#2`). The
folder ships with sample documents; replace them with your own. If the folder is
missing, the agent falls back to an in-code fixture so tests still run.

```
brain/
  guidance/   ai_delivery.md  semantic_layer.txt  responsible_ai.pdf  monitoring.html
  data/       ai_adoption_stats.csv  provider_catalogue.json
```

## Layout

```
src/
  agent/      the loop (plan/route/gather/analyse/synthesise) + LLM clients
  sources/    the data sources (corpus / table / web) + the brain loader
  eval/       eval harness + golden cases + gate
  service.py  FastAPI app (/health, /research)
  run_eval.py gate entry point
tests/        offline test suite
Dockerfile    multi-stage, non-root, healthcheck
.github/      CI: tests + eval gate on every push
```

## Evaluation

`python -m src.run_eval` scores the agent over a golden set of 10 cases:

- **routing**: did it consult the source(s) it should have?
- **grounding**: is every citation a real, gathered evidence id (no fabrication)?
- **abstention**: does it decline when nothing in the estate can answer?

The gate requires perfect grounding and abstention and routing ≥ 0.75; CI fails
the build on any regression.

## Notes on the retrieval

Corpus and web retrieval rank by **word overlap (TF-cosine)**, not embeddings.
`_embed` builds bag-of-words vectors and `_cosine` scores the overlap. For a real
deployment, swap a sentence-transformer or provider embedding in at the `_embed`
call site (the cosine maths is unchanged), and give `WebSource` a real fetcher in
place of the offline fixture. The container runs non-root with a healthcheck, and
CI runs the tests and the eval gate on every push.
