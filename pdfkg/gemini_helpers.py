"""
Optional Gemini (Google Generative AI) helpers for visual cross-reference extraction.
"""

import os
from pathlib import Path
import time
from dotenv import load_dotenv

# Load .env file
load_dotenv()

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from pdfkg.config import Mention


def gemini_extract_crossrefs(
    pdf_path: str | Path, page_start: int, page_end: int
) -> dict:
    """
    Use Gemini to extract cross-references from PDF pages via vision.

    Args:
        pdf_path: Path to PDF file.
        page_start: Starting page (1-indexed, inclusive).
        page_end: Ending page (1-indexed, inclusive).

    Returns:
        Dict with schema:
        {
            "cross_references": [
                {
                    "page": int,
                    "quote": str,
                    "kind": "section"|"page"|"figure"|"table",
                    "target_text": str|null,
                    "target_number": str|null,
                    "target_page": int|null
                }
            ]
        }
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google-generativeai not installed")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    # Upload PDF (compatible with v0.8.x and v0.9+)
    pdf_path_str = str(pdf_path)
    pdf_file = genai.upload_file(pdf_path_str)

    # Wait for processing
    while pdf_file.state.name == "PROCESSING":
        time.sleep(2)
        pdf_file = genai.get_file(pdf_file.name)

    if pdf_file.state.name == "FAILED":
        raise RuntimeError(f"File processing failed: {pdf_file.name}")

    # Define schema for structured output
    schema = {
        "type": "object",
        "properties": {
            "cross_references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer"},
                        "quote": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": ["section", "page", "figure", "table"],
                        },
                        "target_text": {"type": "string", "nullable": True},
                        "target_number": {"type": "string", "nullable": True},
                        "target_page": {"type": "integer", "nullable": True},
                    },
                    "required": [
                        "page",
                        "quote",
                        "kind",
                    ],
                },
            }
        },
        "required": ["cross_references"],
    }

    # Get model name from environment or use default
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    prompt = f"""
You are analyzing pages {page_start} to {page_end} of a technical PDF manual.

Extract all cross-references (mentions of sections, pages, figures, tables).

For each cross-reference found:
- Identify the page number where it appears
- Quote the exact text (e.g., "see section 11.4.2", "refer to Fig. 3")
- Classify the kind: section, page, figure, or table
- Extract target information if clear:
  - target_text: the referenced section title or caption text (if visible)
  - target_number: the section/figure/table number
  - target_page: the page where the target is located (if mentioned or visible)
- Leave fields null if uncertain; do NOT guess.

Return strict JSON matching the schema.
"""

    model = genai.GenerativeModel(
        model_name,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )

    response = model.generate_content([prompt, pdf_file])
    import json

    return json.loads(response.text)


def merge_gemini_crossrefs(
    mentions: list[Mention],
    gemini_json: dict,
    sections: dict[str, dict],
    figures: dict[str, str],
    tables: dict[str, str],
) -> list[Mention]:
    """
    Merge Gemini-extracted cross-references to fill unresolved mentions.

    Args:
        mentions: List of Mention objects (some unresolved).
        gemini_json: Output from gemini_extract_crossrefs.
        sections: Section tree dict.
        figures: Figure index dict.
        tables: Table index dict.

    Returns:
        Updated list of Mention objects.
    """
    # Build lookup from Gemini data
    gemini_refs = gemini_json.get("cross_references", [])

    # For each unresolved mention, try to find a matching Gemini ref
    for mention in mentions:
        if mention.target_id:
            continue  # Already resolved

        # Match by kind and hint
        for gref in gemini_refs:
            if gref["kind"] != mention.kind:
                continue

            # Try to match target_number to hint
            if gref["target_number"] and gref["target_number"].upper() == mention.target_hint.upper():
                # Resolve based on kind
                if mention.kind == "section":
                    # Try to find section by number
                    for sec_id in sections:
                        if sec_id.startswith(gref["target_number"]):
                            mention.target_id = sec_id
                            break
                elif mention.kind == "figure":
                    mention.target_id = figures.get(gref["target_number"].upper())
                elif mention.kind == "table":
                    mention.target_id = tables.get(gref["target_number"].upper())
                elif mention.kind == "page" and gref["target_page"]:
                    mention.target_id = f"page:{gref['target_page']}"
                break

    return mentions
