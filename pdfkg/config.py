"""
Configuration dataclasses and constants for pdfkg.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Paths:
    """Input/output paths for the pipeline."""

    pdf: Path
    out: Path


@dataclass
class Chunk:
    """A text chunk with metadata."""

    id: str
    section_id: str
    page: int
    text: str


@dataclass
class Mention:
    """A detected cross-reference mention."""

    source_chunk_id: str
    kind: Literal["section", "page", "figure", "table"]
    raw_text: str
    target_hint: str
    target_id: str | None = None


# Constants
DOC_ID_DEFAULT = "document"


def default_tokenizer(text: str) -> list[str]:
    """Simple whitespace tokenizer."""
    return text.split()
