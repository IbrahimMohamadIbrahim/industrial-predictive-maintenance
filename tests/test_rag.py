from __future__ import annotations

from types import SimpleNamespace

from rag.assistant import RAGAssistant
from rag.retriever import SourceRetriever


def test_retriever_finds_published_failure_mode_definition() -> None:
    retriever = SourceRetriever()

    results = retriever.retrieve("What does HDF mean in the AI4I dataset?")

    assert results
    assert any(result.chunk.source_file == "03_ai4i_failure_modes.md" for result in results)
    assert any("heat dissipation" in result.chunk.text.lower() for result in results)


def test_retriever_returns_no_result_for_unrelated_question() -> None:
    retriever = SourceRetriever()

    results = retriever.retrieve("What is the latest cryptocurrency price today?")

    assert results == []


def test_assistant_has_safe_offline_fallback_with_citations() -> None:
    assistant = RAGAssistant(retriever=SourceRetriever(), groq_api_key=None)

    answer = assistant.answer("What does power failure mean in AI4I?")

    assert answer.mode == "retrieval-only"
    assert not answer.no_answer
    assert answer.sources
    assert "retrieval-only" in answer.text.lower()
    assert any(result.chunk.source_file == "03_ai4i_failure_modes.md" for result in answer.sources)


def test_assistant_declines_to_answer_without_retrieved_evidence() -> None:
    assistant = RAGAssistant(retriever=SourceRetriever(), groq_api_key=None)

    answer = assistant.answer("Which factory should I buy next week?")

    assert answer.no_answer
    assert answer.sources == ()


def test_assistant_uses_groq_chat_completion_with_retrieved_context() -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded generated answer."))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    assistant = RAGAssistant(
        retriever=SourceRetriever(),
        groq_api_key="test-key",
        model="test-model",
        client=fake_client,
    )

    answer = assistant.answer("What does HDF mean in AI4I?")

    assert answer.mode == "generated"
    assert answer.text == "Grounded generated answer."
    assert captured["model"] == "test-model"
    assert "Retrieved project context" in captured["messages"][1]["content"]  # type: ignore[index]
