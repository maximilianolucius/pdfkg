"""
PDF parsing utilities using PyMuPDF.
"""

from pathlib import Path

import fitz


def load_pdf(path: str | Path) -> fitz.Document:
    """Load a PDF document."""
    return fitz.open(str(path))


def extract_pages(doc: fitz.Document) -> list[dict]:
    """
    Extract pages with text and blocks ordered top→bottom, left→right.

    Returns:
        List of dicts with keys: page (int), text (str), blocks (list of dicts).
    """
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        # Extract blocks with coordinates for ordering
        blocks_data = page.get_text("dict")["blocks"]
        blocks = []
        for block in blocks_data:
            if "lines" in block:  # text block
                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"]
                    block_text += "\n"
                blocks.append(
                    {
                        "bbox": block["bbox"],
                        "text": block_text.strip(),
                    }
                )
        # Sort blocks top→bottom, left→right
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        pages.append(
            {
                "page": page_num + 1,  # 1-indexed
                "text": text,
                "blocks": blocks,
            }
        )
    return pages


def extract_toc(doc: fitz.Document) -> list[dict]:
    """
    Extract table of contents.

    Returns:
        List of dicts with keys: level (int), title (str), page (int).
    """
    toc_raw = doc.get_toc(simple=False)
    toc = []
    for item in toc_raw:
        level, title, page = item[0], item[1], item[2]
        toc.append({"level": level, "title": title, "page": page})
    return toc
