"""Regression tests for conversational routing in the maintenance assistant.

Covers the bug reported against the Streamlit "Maintenance assistant" tab:
plain greetings like "hello" were being sent through TF-IDF retrieval,
scoring below the similarity threshold, and incorrectly returning the
"could not find an answer in the knowledge base" fallback.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag.assistant import RAGAssistant
from rag.conversation import classify
from rag.retriever import SourceRetriever

TESTS_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def retriever() -> SourceRetriever:
    # Deliberately NOT using pytest's tmp_path_factory here. It defaults to
    # the OS temp directory (tempfile.gettempdir()), and on some Windows
    # machines that directory ends up with permissions pytest can't use
    # ("PermissionError: [WinError 5] Access is denied"), which is an
    # environment/OS issue unrelated to this test's logic. Using a folder
    # inside the repo's own tests/ directory avoids the OS temp dir
    # entirely, so it works regardless of that machine-specific problem.
    sources_dir = TESTS_DIR / "_scratch_rag_sources"
    if sources_dir.exists():
        shutil.rmtree(sources_dir)
    sources_dir.mkdir(parents=True)
    (sources_dir / "overview.md").write_text(
        "# AI4I Overview\n\n"
        "## Failure modes\n\n"
        "HDF is heat-dissipation failure. TWF is tool-wear failure.\n",
        encoding="utf-8",
    )
    try:
        yield SourceRetriever(sources_dir)
    finally:
        shutil.rmtree(sources_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def assistant(retriever: SourceRetriever) -> RAGAssistant:
    # groq_api_key=None -> offline/retrieval-only mode, no network calls in tests.
    return RAGAssistant(retriever=retriever, groq_api_key=None)


@pytest.mark.parametrize(
    "message,expected_category",
    [
        ("hello", "greeting"),
        ("Hi", "greeting"),
        ("hey!", "greeting"),
        ("good morning", "greeting"),
        ("thanks!", "closing"),
        ("goodbye", "closing"),
        ("What can you do?", "capability"),
        ("Who are you?", "capability"),
        ("What does HDF mean?", "domain"),
        ("What is the latest Bitcoin price?", "unrelated"),
        ("", "ambiguous"),
    ],
)
def test_classify(message: str, expected_category: str) -> None:
    assert classify(message) == expected_category


def test_greeting_does_not_use_retrieval_fallback(assistant: RAGAssistant) -> None:
    result = assistant.answer("hello")
    assert result.mode == "greeting"
    assert result.no_answer is False
    assert result.sources == ()
    assert "knowledge base" not in result.text.lower()


def test_capability_question_answered_without_retrieval(assistant: RAGAssistant) -> None:
    result = assistant.answer("What can you do?")
    assert result.mode == "capability"
    assert result.no_answer is False


def test_unrelated_question_is_redirected_politely(assistant: RAGAssistant) -> None:
    result = assistant.answer("What is the latest Bitcoin price?")
    assert result.mode == "unrelated"
    assert result.no_answer is False


def test_supported_project_question_still_uses_retrieval(assistant: RAGAssistant) -> None:
    result = assistant.answer("What does HDF mean?")
    assert result.mode in {"retrieval-only", "generated"}
    assert result.sources  # retrieval actually found the HDF chunk


def test_unsupported_project_question_still_falls_back(assistant: RAGAssistant) -> None:
    # Contains a domain term (so it is NOT treated as "unrelated"), but the
    # tiny test corpus has no SHAP content, so retrieval legitimately finds
    # nothing. The original "not found in knowledge base" message must
    # still fire here.
    result = assistant.answer("What SHAP method is used for explainability?")
    assert result.mode == "no-answer"
    assert result.no_answer is True