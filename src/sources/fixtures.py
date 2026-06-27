"""Fallback sample data for tests and demos when no brain folder is present:
policy prose (corpus), figures (table), and web-style snippets (web)."""

CORPUS = {
    "ai_adoption_brief": (
        "Departments are increasingly adopting large language models for drafting and triage.\n"
        "Evaluation rigour remains the main barrier to deploying AI in high-stakes settings.\n"
        "A semantic layer lets a system match a question to the most relevant internal data.\n"
    ),
    "delivery_guidance": (
        "Production AI services must be monitored continuously once deployed.\n"
        "Agentic systems should be bounded to remain reliable and auditable.\n"
        "Clear communication with non-technical stakeholders builds trust in AI tools.\n"
    ),
}

TABLE = [
    {"metric": "departments using LLMs", "value": "17", "year": "2025"},
    {"metric": "average evaluation pass rate", "value": "82%", "year": "2025"},
    {"metric": "AI pilots reaching production", "value": "34%", "year": "2025"},
]

WEB = {
    "https://example.gov/ai-trends": (
        "The latest trend in government AI is agentic systems that plan their own research steps."
    ),
    "https://example.gov/today-news": (
        "A current review highlights monitoring and auditability as priorities for AI in the public sector."
    ),
}
