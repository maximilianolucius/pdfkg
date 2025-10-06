"""
Text chunking utilities.
"""

import re
import uuid

from pdfkg.config import Chunk, default_tokenizer


def text_to_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.

    Args:
        text: Input text.

    Returns:
        List of sentences.
    """
    # Simple sentence boundary detection
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def section_chunks(
    section_id: str,
    section_text: str,
    page_hint: int,
    max_tokens: int = 500,
    tokenizer=None,
) -> list[Chunk]:
    """
    Chunk section text by accumulating sentences within token budget.

    Args:
        section_id: Section identifier.
        section_text: Text of the section.
        page_hint: Starting page of the section.
        max_tokens: Maximum tokens per chunk.
        tokenizer: Callable to tokenize text (default: whitespace split).

    Returns:
        List of Chunk objects.
    """
    if tokenizer is None:
        tokenizer = default_tokenizer

    sentences = text_to_sentences(section_text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        tokens = tokenizer(sentence)
        token_count = len(tokens)

        if current_tokens + token_count > max_tokens and current_chunk:
            # Flush current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    section_id=section_id,
                    page=page_hint,
                    text=chunk_text,
                )
            )
            current_chunk = []
            current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += token_count

    # Flush remaining
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                section_id=section_id,
                page=page_hint,
                text=chunk_text,
            )
        )

    return chunks


def build_chunks(
    pages: list[dict], sections: dict[str, dict], max_tokens: int = 500
) -> list[Chunk]:
    """
    Build chunks from pages and sections.

    Args:
        pages: List of page dicts from extract_pages.
        sections: Section tree from build_section_tree.
        max_tokens: Maximum tokens per chunk.

    Returns:
        List of Chunk objects.
    """
    # Sort sections by page
    sections_by_page = sorted(
        [(s["page"], s["id"], s) for s in sections.values()], key=lambda x: x[0]
    )

    chunks = []

    if not sections_by_page:
        # No sections, treat entire doc as one section
        all_text = " ".join([p["text"] for p in pages])
        chunks.extend(section_chunks("root", all_text, 1, max_tokens))
        return chunks

    # Build page text map
    page_texts = {p["page"]: p["text"] for p in pages}

    for i, (start_page, sec_id, sec) in enumerate(sections_by_page):
        # Determine end page (exclusive)
        if i + 1 < len(sections_by_page):
            end_page = sections_by_page[i + 1][0]
        else:
            end_page = max(page_texts.keys()) + 1

        # Concatenate text from start_page to end_page-1
        section_text = ""
        for p in range(start_page, end_page):
            if p in page_texts:
                section_text += page_texts[p] + "\n"

        section_text = section_text.strip()
        if section_text:
            chunks.extend(section_chunks(sec_id, section_text, start_page, max_tokens))

    return chunks
