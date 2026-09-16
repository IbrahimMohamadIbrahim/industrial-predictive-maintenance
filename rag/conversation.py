"""Conversational routing for the Maintenance Assistant.

Classifies an incoming message so that greetings, thanks/goodbyes, and
"what can you do" style questions are answered naturally, WITHOUT calling
the retriever -- per the project requirement that the assistant "must not
force every question through retrieval."

Project-specific questions (anything mentioning a dataset column, failure
mode, model, SQL concept, etc.) are left for the normal retrieval path in
RAGAssistant.answer(), which already handles the "found" / "not found"
cases correctly. This module only intercepts the cases that retrieval was
never meant to answer.
"""

from __future__ import annotations

import re

GREETING_PATTERNS = (
    r"^\s*hello\s*[!.]*\s*$",
    r"^\s*hi\s*[!.]*\s*$",
    r"^\s*hey\s*[!.]*\s*$",
    r"^\s*(good\s+(morning|afternoon|evening))\s*[!.]*\s*$",
    r"^\s*how\s+are\s+you\??\s*$",
)

CLOSING_PATTERNS = (
    r"^\s*thanks?\s*(you)?\s*[!.]*\s*$",
    r"^\s*thank\s+you\s*[!.]*\s*$",
    r"^\s*(bye|goodbye|see\s+you)\s*[!.]*\s*$",
)

CAPABILITY_PATTERNS = (
    r"what\s+can\s+you\s+do",
    r"how\s+can\s+you\s+help",
    r"who\s+are\s+you",
    r"what\s+are\s+you",
    r"what\s+is\s+this\s+(assistant|bot|chat)",
)

# Small heuristic vocabulary used only to decide routing, never to answer.
# Extend this list as sources are added under rag/sources/.
DOMAIN_TERMS = (
    "machine failure", "failure", "predictive maintenance", "maintenance",
    "sensor", "air temperature", "process temperature", "rotational speed",
    "torque", "tool wear", "twf", "hdf", "pwf", "osf", "rnf",
    "dataset", "ai4i", "model", "mlp", "autoencoder", "anomaly",
    "threshold", "recall", "precision", "sql", "schema", "shap",
    "feature", "explainability", "risk", "prediction", "classifier",
    "leakage", "drift",
)

GREETING_RESPONSE = (
    "Hello! How can I help you with the industrial predictive-maintenance project?"
)

CLOSING_RESPONSE = (
    "You're welcome! Let me know if you have more questions about the project."
)

CAPABILITY_RESPONSE = (
    "I'm the maintenance assistant for this predictive-maintenance project. I can explain "
    "the dataset and its columns, the failure modes (TWF, HDF, PWF, OSF, RNF), how the "
    "classical ML and deep-learning models work, the SQL schema, explainability methods, "
    "and the project's known limitations -- all grounded in this project's own documentation."
)

UNRELATED_RESPONSE = (
    "That's outside the scope of this maintenance assistant. I can help with machine-failure "
    "prediction, sensor data, anomaly detection, model results, and maintenance concepts from "
    "this project."
)

CLARIFY_RESPONSE = (
    "Could you clarify what you'd like to know? For example, you can ask about a sensor "
    "column, a failure mode (TWF/HDF/PWF/OSF/RNF), model performance, or the SQL schema."
)


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _looks_domain_related(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in DOMAIN_TERMS)


def classify(message: str) -> str:
    """Return one of: 'greeting', 'closing', 'capability', 'domain', 'ambiguous', 'unrelated'."""
    text = message.strip()

    if not text:
        return "ambiguous"
    if _matches_any(GREETING_PATTERNS, text):
        return "greeting"
    if _matches_any(CLOSING_PATTERNS, text):
        return "closing"
    if _matches_any(CAPABILITY_PATTERNS, text):
        return "capability"
    if _looks_domain_related(text):
        return "domain"

    # Short, non-domain, non-greeting messages are ambiguous rather than
    # confidently "unrelated" (e.g. a short follow-up like "and torque?").
    # Longer non-domain messages are more likely genuinely off-topic.
    if len(text.split()) <= 3:
        return "ambiguous"
    return "unrelated"