# Multi-stage build keeps the runtime image small and free of build tooling.
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime
# Run as a non-root user: least privilege for a service that may reach senior users.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/
# brain/ sits next to src/ (assembly.py looks for ../brain) so the agent uses
# the real corpus, not the fixture fallback.
COPY brain/ ./brain/
USER appuser
EXPOSE 8000

# Container healthcheck backs the orchestrator's readiness signal.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
