"""
Cross-reference extraction and resolution.
"""

import re

from pdfkg.config import Chunk, Mention

# Regex patterns for cross-reference detection
SECTION_RX = re.compile(
    r"\b(?:see|refer to|cf\.|see also)\s+(?:section|chap(?:ter)?)\s+([\d\.]+)\b", re.I
)
PAGE_RX = re.compile(r"\b(?:see|refer to)\s+page\s+(\d{1,3})\b", re.I)
FIGURE_RX = re.compile(
    r"\b(?:see|refer to|cf\.)\s*(?:fig(?:\.|ure)?)\s*([0-9]+[A-Z]?)\b", re.I
)
TABLE_RX = re.compile(
    r"\b(?:see|refer to)\s*(?:tab(?:\.|le)?)\s*([0-9]+[A-Z]?)\b", re.I
)


def extract_mentions(chunk: Chunk) -> list[Mention]:
    """
    Extract cross-reference mentions from a chunk.

    Args:
        chunk: Chunk object.

    Returns:
        List of Mention objects (unresolved).
    """
    mentions = []
    text = chunk.text

    # Section references
    for match in SECTION_RX.finditer(text):
        mentions.append(
            Mention(
                source_chunk_id=chunk.id,
                kind="section",
                raw_text=match.group(0),
                target_hint=match.group(1),
            )
        )

    # Page references
    for match in PAGE_RX.finditer(text):
        mentions.append(
            Mention(
                source_chunk_id=chunk.id,
                kind="page",
                raw_text=match.group(0),
                target_hint=match.group(1),
            )
        )

    # Figure references
    for match in FIGURE_RX.finditer(text):
        mentions.append(
            Mention(
                source_chunk_id=chunk.id,
                kind="figure",
                raw_text=match.group(0),
                target_hint=match.group(1).upper(),
            )
        )

    # Table references
    for match in TABLE_RX.finditer(text):
        mentions.append(
            Mention(
                source_chunk_id=chunk.id,
                kind="table",
                raw_text=match.group(0),
                target_hint=match.group(1).upper(),
            )
        )

    return mentions


def resolve_mentions(
    mentions: list[Mention],
    sections: dict[str, dict],
    figures: dict[str, str],
    tables: dict[str, str],
    n_pages: int,
) -> list[Mention]:
    """
    Resolve target_id for mentions using sections, figures, tables, pages.

    Args:
        mentions: List of Mention objects.
        sections: Section tree dict.
        figures: Figure index dict (number -> id).
        tables: Table index dict (number -> id).
        n_pages: Total number of pages.

    Returns:
        List of Mention objects with target_id populated where possible.
    """
    resolved = []

    for mention in mentions:
        target_id = None

        if mention.kind == "section":
            # Match section by prefix
            hint = mention.target_hint
            for sec_id in sections:
                if sec_id.startswith(hint) or sec_id == hint:
                    target_id = sec_id
                    break

        elif mention.kind == "page":
            page_num = int(mention.target_hint)
            if 1 <= page_num <= n_pages:
                target_id = f"page:{page_num}"

        elif mention.kind == "figure":
            target_id = figures.get(mention.target_hint)

        elif mention.kind == "table":
            target_id = tables.get(mention.target_hint)

        mention.target_id = target_id
        resolved.append(mention)

    return resolved
