"""
Text Splitter for TTS Batch Processing

Splits long text into chunks of complete sentences for TTS generation.

Rules:
- Sentence boundaries: `.`, `?`, `!` (optionally followed by closing quote/bracket).
- A chunk holds one or more whole sentences. Never split inside a sentence.
- `max_chars` is a soft packing target. A single sentence longer than
  `max_chars` is emitted whole (with a warning) to preserve audio integrity.
- Markdown artifacts (`#`, `*`, `_`, `---`, `>`, list markers, links, code
  fences, table pipes) are stripped before splitting for both `.txt` and `.md`.
"""

import logging
import re
from typing import List, Tuple


_FENCED_CODE = re.compile(r'```.*?```', re.DOTALL)
_INLINE_CODE = re.compile(r'`([^`]+)`')
_HEADER = re.compile(r'^[ \t]*#+\s+', re.MULTILINE)
_BOLD_STAR = re.compile(r'\*\*(.+?)\*\*', re.DOTALL)
_BOLD_UNDER = re.compile(r'__(.+?)__', re.DOTALL)
_EM_STAR = re.compile(r'\*(.+?)\*', re.DOTALL)
_EM_UNDER = re.compile(r'(?<!\w)_(.+?)_(?!\w)', re.DOTALL)
_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_IMAGE = re.compile(r'!\[[^\]]*\]\([^)]+\)')
_RULE_LINE = re.compile(r'^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$', re.MULTILINE)
_BLOCKQUOTE = re.compile(r'^[ \t]*>+\s?', re.MULTILINE)
_LIST_MARKER = re.compile(r'^[ \t]*(?:[-*+]|\d+\.)\s+', re.MULTILINE)
_TABLE_PIPE = re.compile(r'\|')

_SENTENCE_PATTERN = re.compile(r'([.!?]+["\')\]]*\s+|[.!?]+["\')\]]*$)')


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting so artifacts don't reach the TTS model."""
    text = _FENCED_CODE.sub('', text)
    text = _IMAGE.sub('', text)
    text = _LINK.sub(r'\1', text)
    text = _RULE_LINE.sub('', text)
    text = _HEADER.sub('', text)
    text = _BLOCKQUOTE.sub('', text)
    text = _LIST_MARKER.sub('', text)
    text = _BOLD_STAR.sub(r'\1', text)
    text = _BOLD_UNDER.sub(r'\1', text)
    text = _EM_STAR.sub(r'\1', text)
    text = _EM_UNDER.sub(r'\1', text)
    text = _INLINE_CODE.sub(r'\1', text)
    text = _TABLE_PIPE.sub(' ', text)
    return text


def _split_into_sentences(text: str) -> List[str]:
    """Split a single line/paragraph (no newlines) into whole sentences."""
    parts = _SENTENCE_PATTERN.split(text)

    sentences: List[str] = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and _SENTENCE_PATTERN.match(parts[i + 1]):
            sentences.append((parts[i] + parts[i + 1]).strip())
            i += 2
        elif parts[i].strip():
            sentences.append(parts[i].strip())
            i += 1
        else:
            i += 1
    return sentences


def _pack_sentences(sentences: List[str], max_chars: int) -> List[str]:
    """Pack whole sentences into chunks no larger than max_chars (soft)."""
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) <= max_chars:
            current += " " + sentence
            continue

        if current:
            chunks.append(current)

        if len(sentence) > max_chars:
            logging.warning(
                "Sentence exceeds max_chars (%d > %d); emitting whole to "
                "preserve sentence integrity.", len(sentence), max_chars
            )
        current = sentence

    if current:
        chunks.append(current)

    return chunks


def split_text(text: str, max_chars: int = 300) -> List[str]:
    """
    Split text into chunks of whole sentences (newlines collapsed).

    Args:
        text: Input text (may contain markdown; artifacts are stripped).
        max_chars: Soft packing target. Single sentences exceeding this are
            emitted whole.

    Returns:
        List of chunks. Every chunk is one or more complete sentences ending
        in `.`, `?`, or `!` (allowing trailing quote/bracket).
    """
    if not text or not text.strip():
        return []

    text = _strip_markdown(text)
    text = re.sub(r'\s+', ' ', text.strip())
    if not text:
        return []

    sentences = _split_into_sentences(text)
    return _pack_sentences(sentences, max_chars)


def split_markdown(text: str, max_chars: int = 300) -> List[str]:
    """Alias for split_text. Both strip markdown before splitting."""
    return split_text(text, max_chars)


def split_text_with_breaks(text: str, max_chars: int = 300) -> List[Tuple[str, bool]]:
    """
    Split text into chunks, tagging each chunk with whether it begins a new
    line / paragraph in the source.

    A newline in the source marks a paragraph boundary. The first chunk of each
    line is tagged True (-> longer leading silence). Later chunks of the same
    line, produced only because a sentence run exceeded `max_chars`, are tagged
    False (-> shorter leading silence).

    Args:
        text: Input text (may contain markdown; artifacts are stripped).
        max_chars: Soft packing target per chunk.

    Returns:
        List of (chunk_text, is_paragraph_start) tuples.
    """
    if not text or not text.strip():
        return []

    text = _strip_markdown(text)

    # Any run of newlines is a paragraph / line boundary.
    paragraphs = re.split(r'\n+', text)

    result: List[Tuple[str, bool]] = []
    for para in paragraphs:
        para = re.sub(r'\s+', ' ', para.strip())
        if not para:
            continue

        sentences = _split_into_sentences(para)
        chunks = _pack_sentences(sentences, max_chars)
        for idx, chunk in enumerate(chunks):
            result.append((chunk, idx == 0))

    return result


if __name__ == "__main__":
    test_text = """
    # Heading

    This is a short sentence. This is another sentence, with commas, that should stay together as one chunk.
    This sentence is extremely long and definitely exceeds the maximum character limit that we have set for individual chunks in this text-to-speech system, so it must be emitted whole to keep audio intact!
    Is this working correctly? Yes, it should be.

    ---

    > A blockquote line.
    - bullet one
    - bullet two

    Let's test with a final **bold** sentence and a [link](https://example.com).
    """

    print("Testing splitter with max_chars=100")
    print("=" * 60)
    chunks = split_text(test_text, max_chars=100)
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"  {chunk}")
    print("\n" + "=" * 60)
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.1f} chars")
