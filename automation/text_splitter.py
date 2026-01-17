"""
Text Splitter for TTS Batch Processing

Intelligently splits long text into chunks suitable for TTS generation.
Ensures chunks end at punctuation marks and respect word boundaries.
"""

import re
from typing import List


def split_text(text: str, max_chars: int = 300) -> List[str]:
    """
    Split text into chunks ending with punctuation at word boundaries.

    Args:
        text: Input text to split
        max_chars: Maximum characters per chunk (target, may slightly exceed for word boundaries)

    Returns:
        List of text chunks ready for TTS processing

    Rules:
        1. Target max_chars per chunk
        2. Chunks must end with punctuation: . , ; ! ? : etc.
        3. If sentence > max_chars, split at word boundaries
        4. Preserve natural sentence flow
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace and newlines
    text = re.sub(r'\s+', ' ', text.strip())

    # Split on sentence-ending punctuation while preserving the punctuation
    # Pattern matches: . ! ? ; , : followed by space or end of string
    sentence_pattern = r'([.!?;,:]\s+|[.!?;,:]$)'
    parts = re.split(sentence_pattern, text)

    # Recombine parts with their punctuation
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and re.match(sentence_pattern, parts[i + 1]):
            # Combine sentence with its punctuation
            sentences.append(parts[i] + parts[i + 1].strip())
            i += 2
        elif parts[i].strip():
            sentences.append(parts[i].strip())
            i += 1
        else:
            i += 1

    # Build chunks
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence fits in current chunk
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If sentence itself exceeds max_chars, split it at word boundaries
            if len(sentence) > max_chars:
                word_chunks = _split_long_sentence(sentence, max_chars)
                # Add all but the last word chunk
                chunks.extend(word_chunks[:-1])
                # Start new chunk with the last piece
                current_chunk = word_chunks[-1]
            else:
                current_chunk = sentence

    # Add final chunk
    if current_chunk and current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    """
    Split a long sentence at word boundaries.

    Args:
        sentence: Long sentence to split
        max_chars: Maximum characters per chunk

    Returns:
        List of sentence fragments
    """
    words = sentence.split()
    chunks = []
    current = ""

    for word in words:
        # If adding this word would exceed limit
        if len(current) + len(word) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
                current = word
            else:
                # Single word exceeds limit, add it anyway
                chunks.append(word)
        else:
            if current:
                current += " " + word
            else:
                current = word

    # Add remaining words
    if current:
        chunks.append(current.strip())

    return chunks if chunks else [sentence]


def split_markdown(text: str, max_chars: int = 300) -> List[str]:
    """
    Split markdown text while preserving some structure.

    This is a simple implementation that strips markdown formatting.
    For more advanced handling, use a markdown parser.

    Args:
        text: Markdown text to split
        max_chars: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)

    # Remove markdown links [text](url) -> text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)

    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.+?)`', r'\1', text)

    # Now split as regular text
    return split_text(text, max_chars)


if __name__ == "__main__":
    # Test the splitter
    test_text = """
    This is a short sentence. This is another sentence that is a bit longer
    and contains multiple clauses, which makes it more interesting to process.
    This sentence is extremely long and definitely exceeds the maximum character
    limit that we have set for individual chunks in this text-to-speech system,
    so it should be intelligently split at word boundaries to ensure that we
    maintain readability and proper pronunciation! Is this working correctly?
    Yes, it should be. Let's test with a final sentence.
    """

    print("Testing text splitter with max_chars=100")
    print("=" * 60)

    chunks = split_text(test_text, max_chars=100)

    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(f"  {chunk}")

    print("\n" + "=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.1f} chars")
