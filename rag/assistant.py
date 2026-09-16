"""Grounded answer generation over locally retrieved RAG source chunks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping, Sequence

from .conversation import (
    CAPABILITY_RESPONSE,
    CLARIFY_RESPONSE,
    CLOSING_RESPONSE,
    GREETING_RESPONSE,
    UNRELATED_RESPONSE,
    classify,
)
from .retriever import SearchResult, SourceRetriever

SYSTEM_INSTRUCTIONS = """You are the Industrial Predictive Maintenance project assistant.

Response rules:
- Answer only from the retrieved project context supplied in the user message.
- If the context does not support an answer, say so plainly. Do not use outside knowledge.
- Treat AI4I 2020 as a synthetic benchmark. Do not claim real equipment reliability,
  a confirmed failure cause, or a real-world safety conclusion.
- A model probability is an estimate, not a diagnosis. Failure-mode labels describe
  the AI4I simulator and are not proof of a real machine failure mode.
- Do not prescribe repairs, shutdowns, or safety-critical operating actions. Direct
  users to approved site procedures and qualified personnel for real equipment.
- Keep the answer concise. Do not invent citations or source names.
"""

NO_ANSWER = (
    "I could not find an answer in the current project knowledge base. "
    "Please add an approved source document or rephrase the question."
)


@dataclass(frozen=True)
class RAGAnswer:
    """A grounded response with transparent retrieval metadata."""

    text: str
    sources: tuple[SearchResult, ...]
    mode: str
    no_answer: bool = False
    warning: str | None = None


class RAGAssistant:
    """Retrieve local evidence and optionally ask Groq to answer."""

    def __init__(
        self,
        retriever: SourceRetriever | None = None,
        *,
        groq_api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.retriever = retriever or SourceRetriever()
        self.groq_api_key = (
            groq_api_key if groq_api_key is not None else os.getenv("GROQ_API_KEY")
        )
        self.model = model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        self.client = client

    @staticmethod
    def _format_prediction_context(prediction_context: Mapping[str, Any] | None) -> str:
        if not prediction_context:
            return "No current model prediction was supplied."
        entries = [f"- {key}: {value}" for key, value in prediction_context.items()]
        return "Current project prediction (context only, not a diagnosis):\n" + "\n".join(entries)

    @staticmethod
    def _format_retrieved_context(results: Sequence[SearchResult]) -> str:
        blocks = []
        for number, result in enumerate(results, start=1):
            chunk = result.chunk
            blocks.append(
                f"[Source {number}: {chunk.citation}]\n{chunk.text}"
            )
        return "\n\n".join(blocks)

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("The groq package is not installed.") from exc
        return Groq(api_key=self.groq_api_key)

    @staticmethod
    def _offline_answer(results: Sequence[SearchResult]) -> str:
        """Give a useful evidence-only result when no model provider is configured."""
        evidence = []
        for result in results[:2]:
            excerpt = " ".join(result.chunk.text.split())
            evidence.append(f"**{result.chunk.section}**: {excerpt}")
        return (
            "The answer-generation API is not configured, so this is a retrieval-only response.\n\n"
            + "\n\n".join(evidence)
        )

    def answer(
        self,
        question: str,
        *,
        prediction_context: Mapping[str, Any] | None = None,
        top_k: int = 4,
    ) -> RAGAnswer:
        """Return a source-grounded response, with a safe offline fallback."""
        cleaned_question = question.strip()
        if not cleaned_question:
            return RAGAnswer(text="Please enter a question.", sources=(), mode="no-answer", no_answer=True)
        if len(cleaned_question) > 1_500:
            return RAGAnswer(
                text="Please keep the question under 1,500 characters.",
                sources=(),
                mode="no-answer",
                no_answer=True,
            )

        # Route greetings, thanks/goodbyes, capability questions, and clearly
        # off-topic/ambiguous messages BEFORE calling the retriever. These are
        # not project questions, so a TF-IDF search for them either scores
        # near zero (falsely triggering NO_ANSWER, e.g. for "hello") or is
        # simply the wrong tool for the message. Project-specific questions
        # ("domain") fall through unchanged to the existing retrieval flow
        # below, so the "not found in knowledge base" fallback still applies
        # correctly to real unsupported project questions.
        category = classify(cleaned_question)
        if category == "greeting":
            return RAGAnswer(text=GREETING_RESPONSE, sources=(), mode="greeting")
        if category == "closing":
            return RAGAnswer(text=CLOSING_RESPONSE, sources=(), mode="closing")
        if category == "capability":
            return RAGAnswer(text=CAPABILITY_RESPONSE, sources=(), mode="capability")
        if category == "unrelated":
            return RAGAnswer(text=UNRELATED_RESPONSE, sources=(), mode="unrelated")
        if category == "ambiguous":
            return RAGAnswer(text=CLARIFY_RESPONSE, sources=(), mode="clarify")

        results = tuple(self.retriever.retrieve(cleaned_question, top_k=top_k))
        if not results:
            return RAGAnswer(text=NO_ANSWER, sources=(), mode="no-answer", no_answer=True)

        if not self.groq_api_key and self.client is None:
            return RAGAnswer(
                text=self._offline_answer(results),
                sources=results,
                mode="retrieval-only",
                warning="Set GROQ_API_KEY to enable generated, source-grounded answers.",
            )

        user_input = "\n\n".join(
            (
                f"User question:\n{cleaned_question}",
                self._format_prediction_context(prediction_context),
                "Retrieved project context:\n" + self._format_retrieved_context(results),
            )
        )
        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_input},
                ],
            )
            answer_text = (response.choices[0].message.content or "").strip()
            if not answer_text:
                raise RuntimeError("The answer provider returned no text.")
        except Exception:
            return RAGAnswer(
                text=self._offline_answer(results),
                sources=results,
                mode="retrieval-only",
                warning="Generated answer unavailable; showing retrieved project evidence instead.",
            )

        return RAGAnswer(text=answer_text, sources=results, mode="generated")