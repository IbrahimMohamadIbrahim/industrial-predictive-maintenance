"""Local, deterministic retrieval over the project's curated RAG sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_SOURCES_DIR = Path(__file__).resolve().parent / "sources"
_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
_FAILURE_MODE_ACRONYMS = frozenset({"TWF", "HDF", "PWF", "OSF", "RNF"})


@dataclass(frozen=True)
class SourceChunk:
    """A citeable section of one curated Markdown source file."""

    source_file: str
    document_title: str
    section: str
    text: str

    @property
    def citation(self) -> str:
        """Return a concise citation label for the Streamlit interface."""
        return f"{self.document_title} - {self.section} ({self.source_file})"


@dataclass(frozen=True)
class SearchResult:
    """A retrieved source chunk and its local cosine-similarity score."""

    chunk: SourceChunk
    score: float


def _document_title(markdown: str, fallback: str) -> str:
    """Use the first H1 as a human-readable document title."""
    match = re.search(r"(?m)^#\s+(.+?)\s*$", markdown)
    return match.group(1).strip() if match else fallback


def _split_large_text(text: str, max_chunk_chars: int) -> list[str]:
    """Split long sections on paragraph boundaries without discarding content."""
    if len(text) <= max_chunk_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > max_chunk_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate

        while len(current) > max_chunk_chars:
            split_at = current.rfind(" ", 0, max_chunk_chars)
            split_at = split_at if split_at > 0 else max_chunk_chars
            chunks.append(current[:split_at].strip())
            current = current[split_at:].strip()

    if current:
        chunks.append(current)
    return chunks


def _markdown_chunks(path: Path, max_chunk_chars: int) -> list[SourceChunk]:
    """Convert a Markdown file into heading-aware source chunks."""
    markdown = path.read_text(encoding="utf-8").strip()
    if not markdown:
        return []

    title = _document_title(markdown, path.stem.replace("_", " ").title())
    headings = list(_HEADING.finditer(markdown))
    sections: list[tuple[str, str]] = []

    if not headings:
        sections.append((title, markdown))
    else:
        for index, heading in enumerate(headings):
            next_start = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
            section_text = markdown[heading.end():next_start].strip()
            section_name = heading.group(2).strip()
            if section_text:
                sections.append((section_name, section_text))

    chunks: list[SourceChunk] = []
    for section, section_text in sections:
        for chunk_text in _split_large_text(section_text, max_chunk_chars):
            chunks.append(
                SourceChunk(
                    source_file=path.name,
                    document_title=title,
                    section=section,
                    text=chunk_text,
                )
            )
    return chunks


class SourceRetriever:
    """Build an in-memory TF-IDF vector index for the curated RAG corpus.

    The index is intentionally local and rebuilt from source files at process
    start. This avoids sending documents to an external service during
    retrieval and keeps source updates immediately visible after an app reload.
    """

    def __init__(self, sources_dir: Path | str = DEFAULT_SOURCES_DIR, max_chunk_chars: int = 1_200):
        self.sources_dir = Path(sources_dir)
        self.max_chunk_chars = max_chunk_chars
        self.chunks = self._load_chunks()
        if not self.chunks:
            raise ValueError(f"No Markdown source documents found in {self.sources_dir}")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        corpus = [self._index_text(chunk) for chunk in self.chunks]
        self._matrix = self.vectorizer.fit_transform(corpus)

    def _load_chunks(self) -> list[SourceChunk]:
        if not self.sources_dir.is_dir():
            raise FileNotFoundError(f"RAG source directory does not exist: {self.sources_dir}")
        chunks: list[SourceChunk] = []
        for path in sorted(self.sources_dir.glob("*.md")):
            chunks.extend(_markdown_chunks(path, self.max_chunk_chars))
        return chunks

    @staticmethod
    def _index_text(chunk: SourceChunk) -> str:
        return "\n".join((chunk.document_title, chunk.section, chunk.text))

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 4,
        min_score: float = 0.08,
    ) -> list[SearchResult]:
        """Return the strongest relevant chunks, or an empty list for no match."""
        normalized_query = query.strip()
        if not normalized_query or top_k < 1:
            return []

        query_vector = self.vectorizer.transform([normalized_query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()

        # AI4I's five failure-mode abbreviations are named entities. Give an
        # explicit asked-for mode a deterministic heading boost so its canonical
        # definition wins over a broad AI4I overview. Generic words are never
        # boosted, preventing false matches for unrelated questions.
        requested_modes = {
            token.upper()
            for token in re.findall(r"\b[A-Za-z]{3}\b", normalized_query)
            if token.upper() in _FAILURE_MODE_ACRONYMS
        }
        if requested_modes:
            for index, chunk in enumerate(self.chunks):
                heading_terms = set(
                    re.findall(r"\b[A-Z]{3}\b", f"{chunk.document_title} {chunk.section}".upper())
                )
                scores[index] += 0.3 * len(requested_modes.intersection(heading_terms))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

        results: list[SearchResult] = []
        for index, score in ranked:
            if score < min_score:
                break
            results.append(SearchResult(chunk=self.chunks[index], score=float(score)))
            if len(results) == top_k:
                break
        return results
